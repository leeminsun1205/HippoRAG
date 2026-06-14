# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HippoRAG 2 is a graph-based RAG framework. This repository is a research fork of upstream HippoRAG 2. It extends the original framework by attaching **temporal (`timestamp`) and `provenance` metadata to knowledge-graph edges**, threaded through indexing from the corpus JSON down to the igraph edge attributes.

When touching `index()`, `add_fact_edges`, `add_passage_edges`, or `add_new_edges` in `src/hipporag/HippoRAG.py`, preserve this metadata plumbing:

- `self.text_to_meta`
- `self.node_to_node_temporal`
- the `valid_weights` dict carrying `timestamp` / `provenance`

Do not silently remove or ignore temporal and provenance metadata.

---

## Current Implementation Status

This distinguishes what **already exists in the code** from what is still a **research goal**. Do not assume planned features exist — verify in code before relying on them.

**Implemented (present in the codebase today):**

- `index(docs, doc_timestamps, doc_provenances)` accepts per-document `timestamp` and `provenance`.
- `main.py` reads `timestamp` / `provenance` from corpus JSON (defaults: `0` and the doc title).
- During graph construction, edges receive `timestamp` and `provenance` attributes via `self.text_to_meta` → `self.node_to_node_temporal` → the `valid_weights` dict in `add_new_edges`.
- For an entity–entity (fact) edge produced by multiple chunks, the **latest** timestamp wins (`_update_edge_meta`), not whichever chunk was processed last. Comparison falls back to overwrite for non-comparable timestamp types.
- `self.text_to_meta` / `self.node_to_node_temporal` are initialized in `__init__`, so retrieval-only sessions (load prebuilt graph, no `index()` call) don't crash in `add_fact_edges`.
- The metadata is **stored** on the igraph object and persisted in the graph pickle.
- **Recency-weighted PPR (Goal 5), config-gated.** `_get_ppr_edge_weights()` scales PPR edge weights by edge `timestamp` when `temporal_weighting=True` (default **False** = original static behavior). Recency interpolates linearly: newest edge ×1.0, oldest ×`temporal_weight_floor` (default 0.5); edges with no real timestamp (≤0: synonymy/passage/legacy graphs) stay ×1.0. Toggle via `main.py --temporal_weighting true`. `provenance`/reliability are not yet used in scoring (Goal 5 partial).
- **`temporal_mvp` benchmark (Goal 6, MVP).** A controlled synthetic enterprise KB with versioned facts, generated deterministically by `reproduce/build_temporal_mvp.py` (seed 42) into `reproduce/dataset/temporal_mvp{_corpus,}.json` (33 passages, 36 questions across `latest_wins`/`time_specific`/`multihop_updated`/`conflict`/`control`). Plugs into `main.py` unchanged: `python main.py --dataset temporal_mvp ...`. See the Datasets section for the schema and the system-vs-oracle field split.
- **Temporal evaluation (Goal 6/7).** `main.py` now saves per-question predictions to `outputs/<dataset>/predictions_tw_{on,off}.json` (named by `temporal_weighting`). `reproduce/eval_temporal.py` reads those offline (no GPU) and reports per-category EM/F1, gold-hit (lenient containment), plus a **version-pick** breakdown for versioned questions — `v-acc` (committed to the correct version only) / `stale` (wrong version only) / `both` (hedged) / `none`. The version-pick split is the discriminating metric: gold-hit alone reads 1.0 even when the answer mentions both versions. Workflow: run `main.py` twice (`--temporal_weighting false`/`true`, no cache clear between) → `python reproduce/eval_temporal.py`. Metric grounding: EM/F1 are SQuAD-standard, gold-hit is the common answer-string-match; the version-pick buckets are our operationalization of the knowledge-update / knowledge-conflict eval tradition (not a single named metric).

- **Chunking (DyG-RAG / IA-RAG style), config-gated.** `main.py::_chunk_text` + `build_docs` split each corpus doc into sliding-window chunks (tiktoken `cl100k_base`, default 1200 tokens / 64 overlap — the **same boundaries DyG-RAG and IA-RAG use**), gated by `--chunk_size` (default **0 = no chunking = original behavior**, so static benchmarks are untouched). Each chunk becomes its own indexed doc `f"{title}\n{piece}"` and **inherits its parent doc's `timestamp`/`provenance`**, so the metadata plumbing stays intact (`text_to_meta` is keyed by the chunk text that is actually indexed). Chunked runs get a separate working dir (`{save_dir}_chunk{size}`) so their graph/cache never collides with the non-chunked index. Requires `tiktoken` for faithful boundaries; falls back to whitespace-word windows otherwise.
- **Real-data temporal tier (Goal 6).** `reproduce/build_dyg.py` converts the **DyG-RAG processed corpora** (TimeQA / TempReason / ComplexTR, dropped into `reproduce/dataset/dyg_raw/{TimeQA,TempReason,ComplexTR}/{Corpus,Question}.json`) into the repo schema → `reproduce/dataset/{timeqa,tempreason,complextr}{_corpus,}.json`, consumed by `main.py --dataset {timeqa,...}` unchanged. These are the **same processed datasets used by DyG-RAG and IA-RAG**, so HippoRAG numbers are directly comparable to their published HippoRAG baseline rows. `reproduce/eval_dyg.py` reports the **same metrics they use**: **Accuracy** (answer-string containment — their headline), **Recall** (token overlap), plus EM/F1 for internal rigor. No gold supporting passages exist in these datasets → `get_gold_docs` falls back to `None` and retrieval recall is not evaluated (consistent with DyG-RAG/IA-RAG, which report QA token-overlap only). Per-passage `timestamp=0` (recency/`temporal_weighting` stays inert here — the time signal lives in text + question, as in IA-RAG); `question_time` is parsed (from the `date` field or the question text) and stored for the future time-scoped retrieval step.

  Known limitations of the current metadata (not bugs, but watch out): synonymy edges carry the default `(0, 'Unknown')`; temporal lives only on edges (not on fact/entity/passage nodes). (Chunking no longer breaks `text_to_meta` — it is keyed by the chunk text that is actually indexed; see the Chunking bullet above.)

**Not implemented yet (research TODO — no code exists for these):**

- **Conflict detection** (rule-based or NLI / DeBERTa-MNLI). The `conflict` concept described below is a design plan, not code.
- **`superseded` marking.** No triple/edge is currently marked superseded; nothing sets or reads such a status. `delete()` hard-deletes nodes — there is no soft-supersede path yet.
- **Provenance-/reliability-weighted scoring.** Only recency (timestamp) feeds PPR so far; `provenance` is still carried-but-unused (rest of Goal 5).
- **Time-scoped retrieval.** Nothing consumes `valid_from`/`valid_to`/`question_time` yet, so `time_specific` questions are answered with the newest version (wrong by design until this is built).

In short: recency now optionally affects retrieval (Goal 5, behind `temporal_weighting`) and is measurable per-category (Goal 6/7); conflict/supersede, provenance-weighting, and time-scoped retrieval are still unbuilt. Tasks touching those mean building new logic, not editing existing logic.

**Maintenance rule:** When a code change implements, removes, or materially changes any feature listed above, update this section **in the same task** — move completed TODOs from "Not implemented yet" to "Implemented", and record any new limitations or follow-up TODOs. Keep this section in sync with the code; a stale status here is worse than none.

---

## Research Context

This repository is used for the VDS / Viettel Digital Services research project.

**Vietnamese title:** Truy xuất tri thức đa bước nhận biết được mâu thuẫn và phiên bản thời gian

**English working title:** Multi-hop Knowledge Retrieval with Conflict Awareness and Temporal Versioning

The project studies how to extend graph-based RAG systems so that they can handle continuously changing enterprise knowledge. In real internal knowledge bases, regulations, project specifications, personnel information, and business rules may change over time. A normal RAG or graph-based RAG system may retrieve outdated information or mix old and new information without telling the LLM that multiple versions exist.

The main research direction is to add two capabilities to graph-based retrieval:

1. **Temporal awareness**: each triple, node, or edge should preserve temporal information such as timestamp, version, or validity period.
2. **Conflict detection**: when new triples are inserted, the system should detect whether they contradict older triples and mark older information as superseded instead of deleting it.

During retrieval, the system should prefer newer and more reliable information, but it should still expose older versions to the LLM when they are relevant for reasoning or comparison.

---

## Research Goals

The project has the following goals:

1. Reproduce an open-source graph-based RAG baseline, mainly HippoRAG, on standard multi-hop QA benchmarks.
2. Extend the graph schema to attach `timestamp` and `provenance` metadata to triples and graph edges.
3. Implement a lightweight conflict detection mechanism, possibly using an NLI model such as DeBERTa-MNLI, to detect contradictions between old and new triples.
4. Mark outdated or contradicted triples as `superseded` instead of deleting them.
5. Modify graph propagation / Personalized PageRank scoring so that edge weights can depend on recency, provenance, and reliability.
6. Build or adapt a temporal QA benchmark by injecting updated facts into an existing corpus or using temporal QA datasets.
7. Evaluate the system on both static benchmarks and temporal benchmarks.

---

## Expected Outputs

The final project should produce:

1. Reproducible source code.
2. Augmented temporal dataset or scripts for building it.
3. Experimental results on original static benchmarks.
4. Experimental results on temporal / conflict-aware benchmarks.
5. A technical report analyzing the trade-off between temporal accuracy and insertion/update overhead.
6. Error analysis explaining when conflict detection fails.
7. At least 5 qualitative case studies showing typical success and failure cases.

---

## Running

The package is imported locally as:

```python
from src.hipporag import HippoRAG
```

Scripts should be run from the repository root. Do not assume the package is installed globally as `hipporag`.

There is no build step. This repository is a Python library plus driver scripts.

Main experiment driver:

```sh
python main.py --dataset sample --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

Local vLLM backend:

```sh
# Start vLLM first:
# vllm serve <model>

python main.py --dataset sample --llm_base_url http://localhost:8000/v1 --llm_name meta-llama/Llama-3.3-70B-Instruct --embedding_name nvidia/NV-Embed-v2
```

vLLM offline batch OpenIE:

```sh
python main.py --dataset sample --llm_name meta-llama/Llama-3.3-70B-Instruct --openie_mode offline --skip_graph
```

Notes:

- `main.py` reads `reproduce/dataset/{dataset}_corpus.json` and `reproduce/dataset/{dataset}.json`.
- It indexes the corpus, runs RAG QA, and prints retrieval + QA metrics.
- `main_dpr.py` runs the dense-passage-retrieval-only baseline through `rag_qa_dpr`.
- `main_azure.py` targets Azure OpenAI.
- `demo*.py` are minimal `index → retrieve → rag_qa` examples per backend.
- Use `--dataset sample` for quick debugging. It is tiny and cheap.

---

## Tests

There is no pytest harness. Tests are standalone scripts asserting indexing, graph loading, deletion, and incremental-update behavior on core modules. Each requires a live LLM or model backend.

```sh
python tests_openai.py      # needs OPENAI_API_KEY
python tests_local.py       # needs a local vLLM server
python tests_azure.py       # Azure OpenAI
python test_transformers.py # HuggingFace Transformers backend
```

When modifying core indexing, retrieval, graph construction, or metadata logic, run the smallest relevant test first before running large experiments.

---

## Re-running / Cache Invalidation

Indexing is heavily cached. To force a clean rerun of an experiment, delete both the OpenIE cache and the working directory:

```sh
rm <save_dir>/openie_results_ner_<llm_name>.json
rm -rf outputs/<dataset>/<llm_label>_<embedding_label>
```

Example:

```sh
rm -rf outputs/sample/gpt-4o-mini_nvidia_NV-Embed-v2
```

Alternative flags:

```sh
--force_index_from_scratch true
--force_openie_from_scratch true
```

The working directory is named:

```text
{save_dir}/{llm_name}_{embedding_model_name}
```

with `/` replaced by `_`, so each LLM + embedding combination gets its own graph and embedding stores.

When debugging strange retrieval results, always consider stale cache as a possible cause.

---

## Architecture

Everything funnels through one config object and one orchestrator class.

### `src/hipporag/utils/config_utils.py` — `BaseConfig`

`BaseConfig` is a single dataclass holding all major tunables:

- LLM settings
- embedding settings
- graph construction settings
- retrieval settings
- QA settings
- evaluation settings

It is passed everywhere as `global_config`.

Add new knobs here instead of scattering new function parameters across the codebase.

### `src/hipporag/HippoRAG.py` — `HippoRAG`

`HippoRAG` is the only high-level class.

Main flow:

```text
index(docs, doc_timestamps, doc_provenances)
→ OpenIE extraction
→ embed chunks / entities / facts
→ build igraph
→ add fact edges
→ add passage-entity edges
→ add synonymy edges via KNN
→ pickle graph
```

Retrieval flow:

```text
retrieve()
→ embed query
→ score facts
→ rerank_facts through DSPy filter
→ graph_search_with_fact_entities
→ assign phrase / passage weights
→ Personalized PageRank
→ top-k documents
```

If no facts survive reranking, retrieval falls back to dense passage retrieval.

QA flow:

```text
rag_qa()
→ retrieve if needed
→ qa()
→ optional EM / F1 / recall evaluation
```

`retrieve_dpr` and `rag_qa_dpr` are the no-graph DPR baseline.

---

## Node Types and Embedding Stores

The graph contains three main node types:

1. **Passage / chunk**
2. **Entity / phrase**
3. **Fact / triple**

Each type is backed by a separate `EmbeddingStore` in `embedding_store.py`, persisted under the working directory.

Node / item IDs are based on `compute_mdhash_id` content hashes, prefixed by the store namespace:

- `chunk-` (passages)
- `entity-` (phrases)
- `fact-` (triples)

When modifying graph logic, preserve the distinction between passage, entity, and fact nodes.

---

## Pluggable Backends

Backend dispatch is based on name strings.

### LLM backend

`llm/__init__.py::_get_llm_class`

Dispatch logic:

- `bedrock*` → `BedrockLLM`
- `Transformers/*` → `TransformersLLM`
- otherwise → `CacheOpenAI`

`CacheOpenAI` covers OpenAI and OpenAI-compatible endpoints, including local vLLM through `llm_base_url`.

### Embedding backend

`embedding_model/__init__.py::_get_embedding_model_class`

Dispatch is based on substrings of `embedding_model_name`, including:

- `GritLM`
- `NV-Embed-v2`
- `contriever`
- `text-embedding`
- `cohere`
- `Transformers/`
- `VLLM/`

The `Transformers/` backend (`Transformers.py`, used here for `Transformers/BAAI/bge-base-en-v1.5`) wraps `SentenceTransformer`. It **L2-normalizes embeddings** (`normalize_embeddings=norm`, default from `embedding_return_as_normalized`) and prepends the query `instruction` to query texts only — both are required because retrieval scores with dot product as a cosine proxy. For maximum quality you could swap the generic HippoRAG instruction for bge's native query prefix, but normalization is the part that matters.

The `VLLM/` backend (`VLLM.py`, an HTTP client to a vLLM `/v1/embeddings` server) does the **same** L2-normalization client-side and the same query-only instruction handling.

(Both backends previously dropped `norm`/`instruction` silently — and `VLLM.py` additionally referenced an undefined `self.base_url` so it never ran at all. Both fixed.)

### OpenIE backend

`openie_mode` selects one of:

- `OpenIE`
- `VLLMOfflineOpenIE`
- `TransformersOfflineOpenIE`

Relevant files are under `information_extraction/`.

To add a backend, implement the base class in:

- `llm/base.py`
- `embedding_model/base.py`

Then register it in the corresponding `_get_*_class` getter.

---

## Prompts

`prompts/prompt_template_manager.py` loads templates from `prompts/templates/`.

QA prompts are dataset-keyed:

```text
rag_qa_{dataset}
```

If no dataset-specific template exists, it falls back to:

```text
rag_qa_musique
```

Fact reranking uses a compiled DSPy program at:

```text
prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json
```

through:

```text
rerank.py::DSPyFilter
```

`parse_filter` accepts both the wrapped `{"fact": [...]}` form and a bare list of triples, since smaller / quantized LLMs (e.g. `Qwen2.5-7B-Instruct-AWQ`) often emit the latter. When a query's reranking yields no facts, `retrieve()` falls back to plain dense passage retrieval and logs a per-run summary: `DPR fallback (no facts after reranking): N/M queries`. **Watch this number** — a high ratio means the run is effectively closer to DPR than full HippoRAG, usually because the LLM isn't producing parseable facts.

When changing prompts, explain whether the change affects only answer generation, fact reranking, retrieval, or evaluation.

---

## Datasets

Retrieval corpus files:

```text
reproduce/dataset/*_corpus.json
```

Query / answer files:

```text
reproduce/dataset/*.json
```

Corpus entries usually contain:

```json
{
  "title": "...",
  "text": "...",
  "idx": "..."
}
```

This research fork may also include:

```json
{
  "timestamp": "...",
  "provenance": "..."
}
```

`main.py::get_gold_docs` and `main.py::get_gold_answers` handle different gold-label schemas across:

- HotpotQA
- 2WikiMultiHopQA
- MuSiQue

Full datasets live on the HuggingFace `osunlp/HippoRAG_2` dataset.

### `temporal_mvp` (derived, synthetic)

Generated by `reproduce/build_temporal_mvp.py` (deterministic, seed 42) — **regenerate, don't hand-edit**. Corpus entries extend the standard schema additively:

- Consumed today: `title`, `text`, `idx`, `timestamp`, `provenance` (only `title`/`text` are indexed; only `timestamp`/`provenance` reach graph edges).
- Future system signal (time-scoped retrieval): `valid_from`, `valid_to` (`null` = still valid).
- **Eval-only oracle — the system must NOT read these at inference** (they are the gold for conflict/supersede): `claim_id` (groups versions of one fact), `version`, `status` (`active`/`superseded`).

Question entries add `category`, `question_time`, `gold_timestamp`, `gold_claim_id`, `stale_answers` (other-version values, for version-pick scoring) on top of the standard `question`/`answer`/`paragraphs` (so existing eval works). Because only recency-PPR exists so far, `time_specific` is expected to score low (answered with the newest version) — that's the motivation for time-scoped retrieval, not a bug.

**Credibility caveat (important):** `temporal_mvp` is fully synthetic. Its numbers are a **controlled diagnostic / mechanism probe** (clean gold, known conflicts) — they are NOT externally valid performance claims. The thesis needs a second evaluation tier on real/established data (adapt TimeQA, inject updates into MuSiQue/2Wiki, or use an existing temporal/conflict benchmark) before any number is reported as a result. Treat synthetic findings as "how the mechanism behaves under control", not "how good the system is".

### `timeqa` / `tempreason` / `complextr` (real-data, DyG-RAG / IA-RAG comparable)

The second evaluation tier the synthetic caveat above calls for. These are the **DyG-RAG processed versions** of three established temporal QA benchmarks, reused verbatim so HippoRAG results line up with the published HippoRAG baseline rows in **DyG-RAG** (arXiv 2507.13396) and **IA-RAG** (arXiv 2606.06044).

- **Source (not committed; download manually):** the DyG-RAG processed datasets (their Google Drive) go into `reproduce/dataset/dyg_raw/{TimeQA,TempReason,ComplexTR}/{Corpus,Question.json}`.
- **Build:** `python reproduce/build_dyg.py --dataset {timeqa|tempreason|complextr}` → emits `{ds}_corpus.json` (`{title,text,idx,timestamp=0,provenance}`) and `{ds}.json` (`{id,question,answer(list),question_time,category}`). `tempreason` is sub-bucketed by reasoning level into `tempreason_L1/L2/L3` (from `original_id`); `timeqa`→`time_specific`, `complextr`→`multihop_temporal`.
- **Run:** `python main.py --dataset timeqa --chunk_size 1200 ...` (chunk to match DyG-RAG/IA-RAG indexing).
- **Eval:** `python reproduce/eval_dyg.py --dataset timeqa` (offline; reads `predictions_tw_{on,off}.json`). Headline = **Accuracy** (containment); also Recall/EM/F1.
- **No gold paragraphs** → retrieval recall disabled (same as DyG-RAG/IA-RAG). `timestamp=0` everywhere → `temporal_weighting` is inert on these; they test whether the **base + chunking** pipeline reproduces the published baseline, and are the substrate for the still-unbuilt time-scoped retrieval (`question_time` is already parsed and stored).

**Why this matters / don't forget:** the deliberate strategy is to first reproduce the **HippoRAG baseline under the DyG-RAG/IA-RAG protocol** (their corpora, their chunking, their Accuracy/Recall metrics) before adding our temporal mechanisms — so any later gain is measured against a faithful, comparable baseline, not a home-grown one.

---

## Benchmarks

Static multi-hop QA benchmarks:

- MuSiQue
- 2WikiMultiHopQA
- HotpotQA

Potential temporal QA benchmarks or references:

- TempLAMA
- TimeQA
- TempReason
- Augmented versions of existing QA corpora with injected outdated and updated facts

The system must not regress significantly on static benchmarks while improving temporal retrieval behavior.

---

## Temporal and Conflict-Aware Design Notes

This fork extends HippoRAG with temporal and provenance metadata. When working with indexing or graph construction, preserve the metadata flow from corpus entries to graph edges.

Important concepts:

- `timestamp`: temporal information associated with a document, triple, node, or edge.
- `provenance`: source information, such as document ID, title, corpus name, or version.
- `superseded`: status indicating that an older triple has been replaced or contradicted by a newer triple.
- `conflict`: a relation between two triples that cannot both be true under the same temporal context.
- `recency weight`: a score adjustment that gives newer information higher retrieval priority.

Potential conflict detection flow:

1. Extract a new triple from an updated document.
2. Search for existing triples with similar subject and relation or semantically similar content.
3. Use rule-based checks or an NLI model to compare the new triple with candidate old triples.
4. If contradiction is detected, keep both triples but mark the older one as superseded.
5. Store metadata showing which triple superseded which older triple.
6. During retrieval, prefer active and newer triples, while still allowing access to superseded triples when needed.

When implementing conflict detection, prefer a simple baseline first:

```text
same subject + same or similar relation + different object
```

Then improve with NLI or semantic similarity if needed.

---

## Experiment Rules

When modifying this repository, follow these rules:

- Do not modify original benchmark datasets directly.
- Keep original train / dev / test splits unchanged.
- If a dataset is augmented, save it as a separate derived dataset.
- Do not change evaluation metrics without explicit approval.
- Do not silently remove temporal or provenance metadata.
- Do not delete old triples when conflicts are found; mark them as `superseded`.
- Preserve reproducibility: every experiment should have a clear command, config, dataset path, model name, and output directory.
- Prefer minimal, well-scoped changes over large refactors.
- If a change affects retrieval, indexing, graph construction, or evaluation, explain the expected impact before editing code.

---

## Evaluation Principles

Static evaluation should answer:

- Does the modified system preserve performance on normal multi-hop QA?
- Does temporal metadata introduce unacceptable overhead?
- Does conflict detection hurt indexing speed or retrieval accuracy?

Temporal evaluation should answer:

- Does the system prefer newer information when old and new facts conflict?
- Can the system identify that multiple versions of a fact exist?
- Can the system avoid mixing incompatible facts from different time periods?
- Does temporal weighting improve retrieval quality?
- In which cases does NLI-based conflict detection fail?

Report at least the following when available:

- EM
- F1
- Retrieval Recall
- Indexing time
- Retrieval time
- Conflict detection overhead
- Number of detected conflicts
- Number of superseded triples

For qualitative analysis, include at least 5 case studies showing typical success and failure cases.

---

## Runtime Environment

Development may happen locally, but larger experiments are expected to run on Kaggle or another GPU environment.

Local usage is mainly for:

- reading code
- small debugging
- running `sample`
- checking syntax
- testing small indexing / retrieval flows

Kaggle or GPU server usage is mainly for:

- larger benchmark reproduction
- vLLM / Transformers backend experiments
- full indexing
- full retrieval and QA evaluation

Known environment assumptions:

- Python 3.10
- conda env: `hipporag`
- `torch==2.5.1`
- `transformers==4.45.2`
- `vllm==0.6.6.post1`

Relevant environment variables:

- `OPENAI_API_KEY`
- `HF_HOME`
- `CUDA_VISIBLE_DEVICES`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

When debugging vLLM offline mode, consider setting:

```text
tensor_parallel_size=1
```

in:

```text
llm/vllm_offline.py
```

When generating commands, prefer commands that can be run from the repository root.

---

## Development Preferences

When helping with this project, Claude should follow these preferences:

- Explain the root cause before proposing code changes.
- Explain the plan before editing files.
- Mention the affected files and functions.
- Prefer simple and reproducible implementations.
- Avoid unrelated refactoring.
- Do not install new packages unless necessary.
- If adding a dependency, explain why it is needed and where it is used.
- Keep code readable for undergraduate thesis work.
- Prioritize correctness and reproducibility before optimization.
- When debugging, identify whether the problem comes from environment, data format, cache, model backend, or code logic.
- When unsure, inspect the relevant code before making broad claims.
- For large changes, propose a small first step that can be tested quickly.

---

## References

Important references for this project:

1. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models
2. HippoRAG GitHub repository by OSU-NLP-Group
3. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs
4. MuSiQue: Multihop Questions via Single-hop Question Composition
5. TempLAMA / TimeQA / TempReason
6. Microsoft GraphRAG
7. **DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning** (arXiv 2507.13396; code: github.com/RingBDStack/DyG-RAG). Event-centric: Dynamic Event Units + event graph + time-aware traversal + Time Chain-of-Thought. **Source of our `timeqa`/`tempreason`/`complextr` processed corpora, chunking (1200/64), and Accuracy/Recall eval protocol.**
8. **IA-RAG: Interval-Algebra–Driven Temporal Reasoning for Dynamic Knowledge Retrieval** (arXiv 2606.06044). Interval Event Units + Thematic Forest + Allen's Interval Algebra + Sub-graph Time Tightening. Uses the same TimeQA/TempReason/ComplexTR benchmarks; its HippoRAG baseline rows are what we reproduce.
9. **It's High Time: A Survey of Temporal Information Retrieval and Question Answering** (arXiv 2505.20243). Survey framing temporal intent detection, time normalization, event ordering, recency/conflict — the conceptual map for this fork's goals.

Do not overfit the implementation to one paper. Use these references to guide design choices, experiments, and comparison.
