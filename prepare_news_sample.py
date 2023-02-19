import json
from typing import List, Dict
from random import shuffle, seed
import pathlib
import smart_open
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("prepare_news_sample")
    parser.add_argument("infile", type=pathlib.Path, help="JSONL files with news in ubertext format")
    parser.add_argument("outfile", type=argparse.FileType("w"), help="JSONL files with news in ubertext format")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--samples", type=int, default=100)

    args = parser.parse_args()
    seed(args.seed)

    articles: List[Dict] = []
    with smart_open.open(args.infile, "rt") as fp_in:
        for article in fp_in:
            articles.append(json.loads(article))

    print(f"Read {len(articles)} articles")

    articles = [article for article in articles if article.get("title", "").strip() and article.get("tags", [])]

    print(f"{len(articles)} articles left after filtering")

    shuffle(articles)

    for article in articles[:args.samples]:
        args.outfile.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Wrote {min(len(articles), args.samples)} to outfile")
