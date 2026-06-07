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


AGG_KEYS = ["n", "em", "f1", "gold_hit",
            "ver_n", "gold_only", "stale_only", "both", "neither"]


def evaluate(questions, predictions):
    pred_by_q = {p["question"]: p.get("answer") or "" for p in predictions}
    missing = [q["question"] for q in questions if q["question"] not in pred_by_q]
    if missing:
        print(f"  WARNING: {len(missing)} questions had no matching prediction (skipped).")

    agg = {c: {k: 0 if k != "em" and k != "f1" else 0.0 for k in AGG_KEYS} for c in CATEGORY_ORDER}

    for q in questions:
        pred = pred_by_q.get(q["question"])
        if pred is None:
            continue
        a = agg[q["category"]]
        a["n"] += 1
        a["em"] += exact_match(q["answer"], pred)
        a["f1"] += f1_score(q["answer"], pred)
        contains_gold = hit(q["answer"], pred)
        a["gold_hit"] += 1 if contains_gold else 0

        # version-pick: for questions with a known other-version value, classify
        # which version the answer actually committed to. This is the metric
        # that EM/F1/gold-hit can't show (gold-hit double-counts "mentions both").
        stale = q.get("stale_answers") or []
        if stale:
            contains_stale = hit(stale, pred)
            a["ver_n"] += 1
            if contains_gold and not contains_stale:
                a["gold_only"] += 1     # committed to the temporally-correct version
            elif contains_stale and not contains_gold:
                a["stale_only"] += 1    # committed to the wrong version
            elif contains_gold and contains_stale:
                a["both"] += 1          # hedged / mentioned both versions
            else:
                a["neither"] += 1
    return agg


def summarize(agg):
    rows = [(c, agg[c]) for c in CATEGORY_ORDER if agg[c]["n"] > 0]
    tot = {k: sum(agg[c][k] for c, _ in rows) for k in AGG_KEYS}
    return rows, tot


def _safe(x, n):
    return x / n if n else 0.0


def qa_row(name, a):
    n = a["n"]
    return f"  {name:<18} {n:>3}  {_safe(a['em'],n):6.3f}  {_safe(a['f1'],n):6.3f}  {_safe(a['gold_hit'],n):6.3f}"


def ver_row(name, a):
    m = a["ver_n"]
    return (f"  {name:<18} {m:>3}  {_safe(a['gold_only'],m):7.3f} {_safe(a['stale_only'],m):7.3f} "
            f"{_safe(a['both'],m):6.3f} {_safe(a['neither'],m):6.3f}")


def report(label, agg):
    rows, tot = summarize(agg)
    print(f"\n=== {label} ===")
    print(f"  {'category':<18} {'n':>3}  {'EM':>6}  {'F1':>6}  {'gold↑':>6}")
    for c, a in rows:
        print(qa_row(c, a))
    print("  " + "-" * 44)
    print(qa_row("OVERALL", tot))

    ver_rows = [(c, a) for c, a in rows if a["ver_n"] > 0]
    if ver_rows:
        print(f"\n  version-pick (versioned questions only)")
        print(f"  {'category':<18} {'n':>3}  {'v-acc↑':>7} {'stale↓':>7} {'both':>6} {'none':>6}")
        for c, a in ver_rows:
            print(ver_row(c, a))
        vtot = {k: sum(a[k] for _, a in ver_rows) for k in AGG_KEYS}
        print("  " + "-" * 48)
        print(ver_row("OVERALL", vtot))


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
    print("Legend: gold↑=gold value appears in answer (lenient). version-pick (versioned Qs):")
    print("  v-acc↑=committed to correct version only | stale↓=committed to wrong version only")
    print("  both=mentioned both versions (hedged) | none=mentioned neither")
    for pf in pred_files:
        label = os.path.basename(pf)
        agg = evaluate(questions, json.load(open(pf)))
        report(label, agg)


if __name__ == "__main__":
    main()
