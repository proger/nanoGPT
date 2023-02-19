import json
import csv
from tqdm import tqdm
import argparse
from rouge import Rouge
from typing import List


def get_jaccard_sim(hypothesis: List[str], reference: List[str]) -> float:
    a = set(map(str.lower, reference))
    b = set(map(str.lower, hypothesis))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_tags_accuracy(hypothesis: List[str], reference: List[str]) -> float:
    a = set(map(str.lower, reference))
    b = set(map(str.lower, hypothesis))
    c = a.intersection(b)

    return float(len(c) / len(a))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_generated")
    parser.add_argument("infile", type=argparse.FileType("r"), help="JSONL files with original and generated tags and titles")
    parser.add_argument("outfile", type=argparse.FileType("w"), help="CSV file with the results")
    args = parser.parse_args()

    rouge = Rouge()

    w = csv.DictWriter(args.outfile, fieldnames=["_id", "tags_jaccard", "tags_accuracy", "rouge-1_f", "rouge-1_p", "rouge-1_r", "rouge-2_f", "rouge-2_p", "rouge-2_r", "rouge-l_f", "rouge-l_p", "rouge-l_r"])
    
    w.writeheader()    
    for doc in tqdm(map(json.loads, args.infile)):
        rouge_scores = rouge.get_scores(doc["generated_title"], doc["title"])

        w.writerow({
            "_id": doc["_id"],
            "tags_jaccard": get_jaccard_sim(hypothesis=doc["generated_tags"], reference=doc["tags"]),
            "tags_accuracy": get_tags_accuracy(hypothesis=doc["generated_tags"], reference=doc["tags"]),
            "rouge-1_f": rouge_scores[0]["rouge-1"]["f"],
            "rouge-1_p": rouge_scores[0]["rouge-1"]["p"],
            "rouge-1_r": rouge_scores[0]["rouge-1"]["r"],
            "rouge-2_f": rouge_scores[0]["rouge-2"]["f"],
            "rouge-2_p": rouge_scores[0]["rouge-2"]["p"],
            "rouge-2_r": rouge_scores[0]["rouge-2"]["r"],
            "rouge-l_f": rouge_scores[0]["rouge-l"]["f"],
            "rouge-l_p": rouge_scores[0]["rouge-l"]["p"],
            "rouge-l_r": rouge_scores[0]["rouge-l"]["r"],
        })
