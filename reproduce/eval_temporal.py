"""Offline temporal evaluation for the `temporal_mvp` benchmark (Goal 6 / phase b).

Reads the question file and one or more prediction files saved by `main.py`
(`predictions_tw_{on,off}.json`) and reports, per category:
  - EM / F1 (SQuAD-style, matching the repo's qa_eval)
  - gold-hit rate (normalized containment — forgiving of verbose QA output)
  - stale rate (answered with a wrong-version value; only where well-defined:
    latest_wins / conflict / time_specific)

No heavy dependencies — runs anywhere, lets you iterate on metrics without
re-running inference. Compares prediction files side by side for on/off A/B.

Examples:
    # auto-detect outputs/temporal_mvp/predictions_tw_{off,on}.json
    python reproduce/eval_temporal.py

    # explicit files
    python reproduce/eval_temporal.py --predictions outputs/temporal_mvp/predictions_tw_off.json \
                                                     outputs/temporal_mvp/predictions_tw_on.json
"""

import os
import re
import json
import string
import argparse
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_PATH = os.path.join(THIS_DIR, "dataset", "temporal_mvp.json")
CATEGORY_ORDER = ["latest_wins", "time_specific", "multihop_updated", "conflict", "control"]
STALE_CATEGORIES = {"latest_wins", "conflict", "time_specific"}


def normalize_answer(s):
    """SQuAD-style normalization — same transforms as src/.../eval_utils.normalize_answer."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


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
        recall = num_same / len(gt)
        return 2 * precision * recall / (precision + recall)
    return max(f1(g, pred) for g in gold_list)


def hit(targets, pred):
    """Normalized containment: True if any target appears as a substring of pred."""
    npred = normalize_answer(pred)
    return any(normalize_answer(t) and normalize_answer(t) in npred for t in targets)


def evaluate(questions, predictions):
    pred_by_q = {p["question"]: p.get("answer") or "" for p in predictions}
    missing = [q["question"] for q in questions if q["question"] not in pred_by_q]
    if missing:
        print(f"  WARNING: {len(missing)} questions had no matching prediction (skipped).")

    # per-category accumulators
    agg = {c: {"n": 0, "em": 0.0, "f1": 0.0, "gold_hit": 0,
               "stale_n": 0, "stale_hit": 0} for c in CATEGORY_ORDER}

    for q in questions:
        pred = pred_by_q.get(q["question"])
        if pred is None:
            continue
        c = q["category"]
        a = agg[c]
        a["n"] += 1
        a["em"] += exact_match(q["answer"], pred)
        a["f1"] += f1_score(q["answer"], pred)
        a["gold_hit"] += 1 if hit(q["answer"], pred) else 0
        stale = q.get("stale_answers") or []
        if c in STALE_CATEGORIES and stale:
            a["stale_n"] += 1
            # stale only counts if it picked a stale value AND not the gold one
            if hit(stale, pred) and not hit(q["answer"], pred):
                a["stale_hit"] += 1

    return agg


def summarize(agg):
    rows = []
    tot = {"n": 0, "em": 0.0, "f1": 0.0, "gold_hit": 0, "stale_n": 0, "stale_hit": 0}
    for c in CATEGORY_ORDER:
        a = agg[c]
        if a["n"] == 0:
            continue
        for k in tot:
            tot[k] += a[k]
        rows.append((c, a))
    return rows, tot


def fmt_row(name, a):
    n = a["n"]
    em = a["em"] / n if n else 0.0
    f1 = a["f1"] / n if n else 0.0
    gh = a["gold_hit"] / n if n else 0.0
    stale = (a["stale_hit"] / a["stale_n"]) if a["stale_n"] else None
    stale_str = f"{stale:6.3f}" if stale is not None else "   n/a"
    return f"  {name:<18} {n:>3}  {em:6.3f}  {f1:6.3f}  {gh:6.3f}  {stale_str}"


def report(label, agg):
    print(f"\n=== {label} ===")
    print(f"  {'category':<18} {'n':>3}  {'EM':>6}  {'F1':>6}  {'gold↑':>6}  {'stale↓':>6}")
    rows, tot = summarize(agg)
    for c, a in rows:
        print(fmt_row(c, a))
    print("  " + "-" * 52)
    print(fmt_row("OVERALL", tot))


def main():
    parser = argparse.ArgumentParser(description="Offline temporal evaluation for temporal_mvp")
    parser.add_argument("--queries", default=QUERY_PATH, help="Path to temporal_mvp.json")
    parser.add_argument("--predictions", nargs="*", default=None,
                        help="Prediction JSON file(s). Defaults to outputs/temporal_mvp/predictions_tw_{off,on}.json")
    parser.add_argument("--save_dir", default="outputs/temporal_mvp",
                        help="Where to look for default prediction files")
    args = parser.parse_args()

    questions = json.load(open(args.queries))

    pred_files = args.predictions
    if not pred_files:
        pred_files = [os.path.join(args.save_dir, f"predictions_tw_{f}.json") for f in ("off", "on")]
        pred_files = [p for p in pred_files if os.path.isfile(p)]
    if not pred_files:
        raise SystemExit("No prediction files found. Run main.py on temporal_mvp first, or pass --predictions.")

    print(f"Questions: {len(questions)} | Prediction files: {len(pred_files)}")
    print("Legend: gold↑ = normalized gold-hit rate (higher better); "
          "stale↓ = answered with a wrong-version value (lower better).")
    for pf in pred_files:
        label = os.path.basename(pf)
        agg = evaluate(questions, json.load(open(pf)))
        report(label, agg)


if __name__ == "__main__":
    main()
