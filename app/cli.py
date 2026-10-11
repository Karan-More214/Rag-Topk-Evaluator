"""python -m app.cli --docs data/docs --qa data/qa.json"""
import argparse
import json
from pathlib import Path

from . import evaluate
from .loaders import extract_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--chunk-sizes", default="16,32,64,128")
    ap.add_argument("--ks", default="1,2,3,5,8,12")
    ap.add_argument("--target", type=float, default=0.9)
    a = ap.parse_args()
    docs = [{"name": p.name, "text": extract_text(p.name, p.read_bytes())}
            for p in sorted(Path(a.docs).iterdir()) if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    cfg = evaluate.Config([int(x) for x in a.chunk_sizes.split(",")], 0.125,
                          [int(x) for x in a.ks.split(",")], target_recall=a.target)
    res = evaluate.run(docs, json.loads(Path(a.qa).read_text()), cfg)
    print(f"{'retriever':9} {'chunk':>5} {'k':>3} {'recall':>7} {'answerable':>10} {'mrr':>6} {'tokens':>7}")
    for r in res["rows"]:
        print(f"{r['retriever']:9} {r['chunk_size']:>5} {r['k']:>3} {r['recall']:>7.2f} {r['recall_answerable']:>10.2f} {r['mrr']:>6.2f} {r['context_tokens']:>7}")
    print("\nRecommendation:", res["recommendation"])
    if any(res["unanswerable"].values()):
        print("Questions whose answer is split across chunks (unanswerable per chunk size):", res["unanswerable"])


if __name__ == "__main__":
    main()
