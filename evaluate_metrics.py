import json
import csv
from tqdm import tqdm
import argparse
from rouge import Rouge
from typing import List
import stanza

nlp = stanza.Pipeline('uk')


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
        generated_title = ''
        for part in doc["generated_title"].split("\n"):
            if part.startswith("рік: ") or part.startswith("мітки: "):
                break
            if generated_title:
                generated_title += " "
            generated_title += part.strip().lower()

        ref_title = doc["title"].lower()
        lemmatized_generated_title = ' '.join(word.lemma.lower() for word in nlp(generated_title).iter_words() if word.upos != 'PUNCT')
        lemmatized_ref_title = ' '.join(word.lemma.lower() for word in nlp(ref_title).iter_words() if word.upos != 'PUNCT')

        rouge_scores = rouge.get_scores(lemmatized_generated_title, lemmatized_ref_title)
        if rouge_scores[0]["rouge-1"]["f"] == 0:
            print('zero:', generated_title, "---", ref_title)
            print('zero:', lemmatized_generated_title, "---", lemmatized_ref_title)

        generated_tags = []
        for tag in doc["generated_tags"]:
            parts = tag.split("\n", maxsplit=1)
            generated_tags.append(parts[0].strip().lower())
            if len(parts) > 1:
                break

        w.writerow({
            "_id": doc["_id"],
            "tags_jaccard": get_jaccard_sim(hypothesis=generated_tags, reference=doc["tags"]),
            "tags_accuracy": get_tags_accuracy(hypothesis=generated_tags, reference=doc["tags"]),
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
