"""Сравнение биграмм и триграмм на общей выборке новостей. Python 3.10+."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
BOS, EOS = "<s>", "</s>"
MAX_WORDS = 30
SEEDS = tuple(range(30))


def word(token):
    return token[0].isalnum()


def lexical(tokens):
    return tuple(t.casefold() for t in tokens if word(t))


def windows(items, width):
    return [items[i:i + width] for i in range(len(items) - width + 1)]


def display(tokens):
    """Простая детокенизация. Знаки и кавычки остаются частью генерации."""
    pieces = []
    no_space = True
    quote_open = False
    for token in tokens:
        opening = token in ("«", "„", "(", "[")
        closing = token in (".", ",", "!", "?", ":", ";", "...", "…", "»", "”", ")", "]", "%")
        if token == '"':
            opening, closing = not quote_open, quote_open
            quote_open = not quote_open
        pieces.append(("" if no_space or closing else " ") + token)
        no_space = opening
    return "".join(pieces)


class NGramModel:
    def __init__(self, corpus, order):
        self.order = order
        self.transitions = defaultdict(Counter)
        for item in corpus:
            context = (BOS,) * (order - 1)
            for token in [*item["tokens"], EOS]:
                self.transitions[context][token] += 1
                context = (*context[1:], token)

    def generate(self, seed):
        rng = random.Random(seed)
        context = (BOS,) * (self.order - 1)
        tokens, trace = [], []
        count = 0
        while True:
            options = self.transitions[context]
            total = sum(options.values())
            draw = rng.randrange(total)
            for token, frequency in options.items():
                draw -= frequency
                if draw < 0:
                    break
            trace.append(dict(context=context, next=token, occurrences=total,
                              alternatives=len(options), selected_frequency=options[token]))
            if token == EOS:
                return tokens, "EOS", trace
            if word(token) and count == MAX_WORDS:
                return tokens, "limit", trace
            tokens.append(token)
            count += word(token)
            context = (*context[1:], token)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.jsonl")
    parser.add_argument("--ratings", type=Path, default=ROOT / "ratings.csv")
    parser.add_argument("--blind", action="store_true", help="выгрузить анонимный набор для оценки")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    corpus = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line]
    if len(corpus) < 300:
        parser.error("Корпус должен содержать не менее 300 предложений.")
    sequences, phrases = {}, {}
    for item in corpus:
        words = lexical(item["tokens"])
        sequences.setdefault(words, item["id"])
        for phrase in windows(words, 4):
            phrases.setdefault(phrase, item["id"])

    samples, paths, model_stats = [], {}, {}
    for order in (2, 3):
        model = NGramModel(corpus, order)
        contexts = [c for ctx, c in model.transitions.items() if all(word(t) for t in ctx)]
        rare_count = sum(sum(c.values()) <= 2 for c in contexts)
        model_stats[order] = dict(lexical_contexts=len(contexts), rare_contexts=rare_count,
                                  single_successor=sum(len(c) == 1 for c in contexts))
        for seed in SEEDS:
            tokens, stop, trace = model.generate(seed)
            words = lexical(tokens)
            matches = [(p, phrases[p]) for p in windows(words, 4) if p in phrases]
            pairs = windows(words, 2)
            sample = dict(order=order, seed=seed, text=display(tokens), words=len(words),
                          stop=stop, copied4=int(bool(matches)), exact=int(words in sequences),
                          repeated2=int(len(pairs) != len(set(pairs))),
                          example4=" ".join(matches[0][0]) if matches else "",
                          source4=matches[0][1] if matches else "",
                          source_exact=sequences.get(words, ""))
            samples.append(sample)
            paths[order, seed] = trace

    shuffled = list(samples)
    random.Random(47613).shuffle(shuffled)
    blinded = []
    for i, row in enumerate(shuffled, 1):
        row["id"] = f"T{i:02}"
        signature = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        blinded.append(dict(id=row["id"], text=row["text"], sha256=signature,
                            coherence="", syntax="", comment=""))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.blind:
        write_csv(args.out / "blind.csv", blinded)
        print("Создан blind.csv. Оцените связность (0–2) и завершённость конструкции (0/1).")
        return

    ratings = read_csv(args.ratings)
    annotations = {a["id"]: a for a in ratings}
    if set(annotations) != {b["id"] for b in blinded} or len(ratings) != len(blinded):
        parser.error("Идентификаторы оценок не совпали с экспериментом.")
    for row, blind in zip(shuffled, blinded):
        a = annotations[row["id"]]
        if (a["sha256"] != blind["sha256"] or a["text"] != row["text"]
                or a["coherence"] not in {"0", "1", "2"} or a["syntax"] not in {"0", "1"}):
            parser.error(f"Проверьте текст и оценку {row['id']}.")
        row.update(coherence=int(a["coherence"]), syntax=int(a["syntax"]),
                   correct_end=int(row["stop"] == "EOS" and a["syntax"] == "1"), comment=a["comment"])

    summaries = []
    for order in (2, 3):
        group = [r for r in samples if r["order"] == order]
        sums = {k: sum(r[k] for r in group) for k in ("copied4", "exact", "repeated2", "correct_end")}
        summaries.append(dict(order=order, n=len(group), **sums, eos=sum(r["stop"] == "EOS" for r in group),
                              mean_length=mean(r["words"] for r in group),
                              coherence_mean=mean(r["coherence"] for r in group),
                              coherence_hist=[sum(r["coherence"] == n for r in group) for n in range(3)],
                              **model_stats[order]))
    write_csv(args.out / "samples.csv", samples)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (args.out / "transitions.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            entry = dict(id=row["id"], steps=paths[row["order"], row["seed"]])
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    for s in summaries:
        print(f"{s['order']}-граммы: фраза корпуса {s['copied4']}/{s['n']}, "
              f"корректный конец {s['correct_end']}/{s['n']}, связность {s['coherence_mean']:.2f}/2.")


if __name__ == "__main__":
    main()
