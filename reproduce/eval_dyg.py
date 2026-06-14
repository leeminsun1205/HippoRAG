"""Offline QA evaluation for the DyG-RAG temporal datasets (timeqa / tempreason / complextr),
using the SAME metrics as DyG-RAG and IA-RAG so numbers are comparable to their published
HippoRAG baseline.

Metrics (matching graphrag/evaluate.py in the DyG-RAG repo):
  - Accuracy : answer-string containment. 1 if every normalized gold answer appears as a
               substring of the normalized prediction, else 0. (This is their headline metric.)
  - Recall   : token overlap = |gold ∩ pred tokens| / |gold tokens|, maxed over gold answers.
  - EM / F1  : SQuAD-style, reported alongside for internal rigor (NOT what IA-RAG reports).

normalize_answer is identical to DyG-RAG's: lowercase -> strip punctuation -> drop articles
(a/an/the) -> collapse whitespace.

Reads the question file + prediction file(s) saved by main.py (predictions_tw_{on,off}.json).
No heavy deps; runs offline.

Examples:
    python reproduce/eval_dyg.py --dataset timeqa
    python reproduce/eval_dyg.py --dataset tempreason \
        --predictions outputs/tempreason/<...>/predictions_tw_off.json
"""

import os
import re
import json
import string
import argparse
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PUNC = set(string.punctuation)


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def accuracy(gold_list, pred):
    """DyG-RAG Accuracy: all gold answers must be substrings of the prediction."""
    npred = normalize_answer(pred)
    golds = [normalize_answer(g) for g in gold_list]
    golds = [g for g in golds if g]
    if not golds:
        return 0.0
    return 1.0 if all(g in npred for g in golds) else 0.0


def recall(gold_list, pred):
    """DyG-RAG Recall: token overlap |gold ∩ pred| / |gold|, best over gold answers."""
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in gold_list:
        gt = normalize_answer(g).split()
        if not gt:
            continue
        num_same = sum((Counter(pt) & Counter(gt)).values())
        best = max(best, num_same / len(gt))
    return best


def exact_match(gold_list, pred):
    return max((1.0 if normalize_answer(g) == normalize_answer(pred) else 0.0) for g in gold_list)


def f1_score(gold_list, pred):
    def f1(gold, pred):
        gt, pt = normalize_answer(gold).split(), normalize_answer(pred).split()
        common = Counter(pt) & Counter(gt)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pt)
        rec = num_same / len(gt)
        return 2 * precision * rec / (precision + rec)
    return max(f1(g, pred) for g in gold_list)


def evaluate(questions, predictions):
    pred_by_q = {p["question"]: (p.get("answer") or "") for p in predictions}
    missing = sum(1 for q in questions if q["question"] not in pred_by_q)
    if missing:
        print(f"  WARNING: {missing} questions had no matching prediction (skipped).")

    agg = {}
    for q in questions:
        pred = pred_by_q.get(q["question"])
        if pred is None:
            continue
        cat = q.get("category", "all")
        a = agg.setdefault(cat, {"n": 0, "acc": 0.0, "rec": 0.0, "em": 0.0, "f1": 0.0})
        gold = q["answer"]
        a["n"] += 1
        a["acc"] += accuracy(gold, pred)
        a["rec"] += recall(gold, pred)
        a["em"] += exact_match(gold, pred)
        a["f1"] += f1_score(gold, pred)
    return agg


def report(label, agg):
    cats = sorted(agg.keys())
    tot = {k: sum(agg[c][k] for c in cats) for k in ("n", "acc", "rec", "em", "f1")}
    print(f"\n=== {label} ===")
    print(f"  {'category':<20} {'n':>5}  {'Acc↑':>6} {'Recall↑':>7}  {'EM':>6} {'F1':>6}")

    def row(name, a):
        n = a["n"] or 1
        return (f"  {name:<20} {a['n']:>5}  {a['acc']/n:6.3f} {a['rec']/n:7.3f}  "
                f"{a['em']/n:6.3f} {a['f1']/n:6.3f}")

    for c in cats:
        print(row(c, agg[c]))
    print("  " + "-" * 56)
    print(row("OVERALL", tot))


def main():
    ap = argparse.ArgumentParser(description="DyG-RAG / IA-RAG style QA eval for temporal datasets")
    ap.add_argument("--dataset", required=True, choices=["timeqa", "tempreason", "complextr"])
    ap.add_argument("--queries", default=None, help="Override path to {dataset}.json")
    ap.add_argument("--predictions", nargs="*", default=None,
                    help="Prediction file(s). Defaults to outputs/{dataset}/**/predictions_tw_{off,on}.json")
    ap.add_argument("--save_dir", default=None, help="Root to search for prediction files")
    args = ap.parse_args()

    qpath = args.queries or os.path.join(THIS_DIR, "dataset", f"{args.dataset}.json")
    questions = json.load(open(qpath))

    pred_files = args.predictions
    if not pred_files:
        root = args.save_dir or os.path.join("outputs", args.dataset)
        found = []
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if re.fullmatch(r"predictions_tw_(on|off)(_cot_on)?\.json", fn):
                    found.append(os.path.join(dirpath, fn))
        pred_files = sorted(found)
    if not pred_files:
        raise SystemExit(f"No prediction files found under outputs/{args.dataset}. "
                         f"Run main.py --dataset {args.dataset} first, or pass --predictions.")

    print(f"Dataset: {args.dataset} | Questions: {len(questions)} | Prediction files: {len(pred_files)}")
    print("Metrics: Acc=answer containment (IA-RAG headline), Recall=token overlap; EM/F1 for reference.")
    for pf in pred_files:
        report(os.path.relpath(pf), evaluate(questions, json.load(open(pf))))


if __name__ == "__main__":
    main()
