"""Convert DyG-RAG processed temporal QA datasets into the repo's corpus + question format.

Source: the DyG-RAG processed datasets (TimeQA / TempReason / ComplexTR), downloaded from
their Google Drive into:
    reproduce/dataset/dyg_raw/{TimeQA,TempReason,ComplexTR}/{Corpus,Question}.json
These are the SAME processed corpora used by DyG-RAG and IA-RAG, so results computed with
reproduce/eval_dyg.py (token-overlap Accuracy / Recall) are directly comparable to the
HippoRAG baseline rows published in those papers.

Outputs (consumed by main.py --dataset {timeqa,tempreason,complextr}):
    reproduce/dataset/{ds}_corpus.json
    reproduce/dataset/{ds}.json

Run from the repo root:
    python reproduce/build_dyg.py --dataset timeqa
    python reproduce/build_dyg.py --dataset tempreason
    python reproduce/build_dyg.py --dataset complextr

Notes / design decisions:
- DyG-RAG corpus entries are {doc_id, title, context}; questions are {id, question, answer}
  with answer as a STRING (TempReason also has {date, original_id}). We rename fields to the
  repo schema: corpus -> {title, text, idx, timestamp, provenance}; question -> {id, question,
  answer (wrapped to a list), question_time, category}.
- There are NO gold supporting-passage labels in these datasets, so we emit NO `paragraphs`.
  main.py's get_gold_docs() then raises and falls back to gold_docs=None -> retrieval recall is
  NOT evaluated (consistent with DyG-RAG/IA-RAG, which report QA token-overlap only).
- Corpus carries no per-passage timestamp -> timestamp=0 (recency / temporal_weighting stays
  inert here; the time signal lives in the text and question, as in IA-RAG). question_time is
  recorded for the future time-scoped retrieval step: from the question `date` field when present
  (TempReason), else parsed from the question text.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "dataset", "dyg_raw")
OUT_DIR = os.path.join(HERE, "dataset")

# dataset key -> (raw folder name, category label for eval bucketing)
DATASETS = {
    "timeqa": ("TimeQA", "time_specific"),
    "tempreason": ("TempReason", "tempreason"),      # refined to tempreason_L1/L2/L3 below
    "complextr": ("ComplexTR", "multihop_temporal"),
}

_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")


def parse_year(text):
    """Last 4-digit year mentioned (the temporal anchor usually trails). None if absent."""
    years = _YEAR_RE.findall(text or "")
    return int(years[-1]) if years else None


def main():
    ap = argparse.ArgumentParser(description="Build a DyG-RAG temporal dataset for HippoRAG.")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--max_questions", type=int, default=0,
                    help="Cap on questions (0 = all). The corpus is always emitted in full.")
    args = ap.parse_args()

    folder, base_category = DATASETS[args.dataset]
    corpus_in = os.path.join(RAW_DIR, folder, "Corpus.json")
    quest_in = os.path.join(RAW_DIR, folder, "Question.json")
    for p in (corpus_in, quest_in):
        if not os.path.isfile(p):
            raise SystemExit(f"Missing {p}. Download the DyG-RAG processed datasets into "
                             f"{RAW_DIR}/ first (see module docstring).")

    raw_corpus = json.load(open(corpus_in, encoding="utf-8"))
    raw_quest = json.load(open(quest_in, encoding="utf-8"))

    # --- corpus: rename fields, attach default (no real) temporal metadata --------------
    corpus = []
    for c in raw_corpus:
        idx = c.get("doc_id", len(corpus))
        title = c.get("title", "") or f"{args.dataset}_{idx}"
        corpus.append({
            "title": title,
            "text": c.get("context", ""),
            "idx": idx,
            "timestamp": 0,                 # no real per-passage time (see module docstring)
            "provenance": title,
        })

    # --- questions: wrap answer to a list, attach question_time + category --------------
    questions = []
    n_time = 0
    for q in raw_quest:
        ans = q.get("answer", "")
        ans_list = ans if isinstance(ans, list) else [ans]
        ans_list = [a for a in ans_list if isinstance(a, str) and a.strip()]
        if not ans_list:
            continue

        # question_time: prefer the explicit `date` field (TempReason), else parse the question.
        qtime = parse_year(q["date"]) if q.get("date") else parse_year(q.get("question", ""))
        if qtime is not None:
            n_time += 1

        # category: refine TempReason by reasoning level (L1/L2/L3) from original_id.
        category = base_category
        if args.dataset == "tempreason":
            # original_id looks like "L2_Q457939_P108_0" -> level is the first token.
            level = (q.get("original_id", "") or "").split("_")[0]
            category = f"tempreason_{level}" if level in ("L1", "L2", "L3") else base_category

        questions.append({
            "id": q.get("id", len(questions)),
            "question": q.get("question", ""),
            "answer": ans_list,
            "question_time": qtime,
            "category": category,
        })
        if args.max_questions and len(questions) >= args.max_questions:
            break

    corpus_path = os.path.join(OUT_DIR, f"{args.dataset}_corpus.json")
    quest_path = os.path.join(OUT_DIR, f"{args.dataset}.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(quest_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    cats = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"[{args.dataset}] wrote {len(corpus)} passages -> {corpus_path}")
    print(f"[{args.dataset}] wrote {len(questions)} questions -> {quest_path}")
    print(f"  question_time parsed: {n_time}/{len(questions)}")
    print(f"  categories: {cats}")
    print("  (no gold paragraphs -> retrieval recall disabled; use eval_dyg.py for QA metrics)")


if __name__ == "__main__":
    main()
