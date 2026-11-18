"""Собрать корпус из выпуска v1.1 Lenta.Ru-News-Dataset (необязательный этап)."""
import argparse
import bz2
import csv
import datetime
import html
import json
import re
from pathlib import Path

from nltk.tokenize import RegexpTokenizer, sent_tokenize


ROOT = Path(__file__).resolve().parent
TOKENIZER = RegexpTokenizer(r"\d+(?:[.,:/–-]\d+)+|\w+(?:[-‑’']\w+)*|\.{3}|[^\w\s]")
TERMINAL = re.compile(r'[.!?…][»”"\')\]]*$')


def extract(archive, target):
    seen = set()
    articles = set()
    records = []
    csv.field_size_limit(10_000_000)
    with bz2.open(archive, "rt", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["topic"].strip() != "Спорт":
                continue
            url = item["url"].strip()
            if url in articles:
                continue
            match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
            if not match:
                continue
            date = datetime.date(*map(int, match.groups())).isoformat()
            articles.add(url)
            body = " ".join(html.unescape(item["text"]).split())
            sentences = sent_tokenize(body, language="russian")
            taken = 0
            for position, sentence in enumerate(sentences, 1):
                if not TERMINAL.search(sentence) or re.search(r"<[^>]+>|https?://", sentence):
                    continue
                tokens = TOKENIZER.tokenize(sentence)
                # Отсекаем спорные границы (включая сокращения с отдельной точкой).
                core = list(tokens)
                while core and core[-1] in ('»', '”', '"', "'", ')', ']', '.', '!', '?', '…', '...'):
                    core.pop()
                if any(t in {'.', '!', '?', '…', '...'} for t in core):
                    continue
                words = [t.casefold() for t in tokens if t[0].isalnum()]
                key = tuple(words)
                if len(words) < 4 or not re.search("[А-Яа-яЁё]", sentence) or key in seen:
                    continue
                seen.add(key)
                records.append(dict(id=f"S{len(records)+1:03}", url=url, date=date,
                                    title=item["title"], position=position, text=sentence,
                                    tokens=tokens))
                taken += 1
                if len(records) == target:
                    return records
                if taken == 5:
                    break
    raise ValueError(f"В архиве найдено только {len(records)} подходящих предложений.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "corpus.jsonl")
    args = parser.parse_args()
    corpus = extract(args.archive, 500)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in corpus),
                        encoding="utf-8", newline="\n")
    print(f"Предложений: {len(corpus)}. Статей: {len({r['url'] for r in corpus})}.")
