import os
from typing import List
import json
import argparse
import logging
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def _chunk_text(text, chunk_size, overlap):
    """Sliding-window chunking matching DyG-RAG (tiktoken cl100k_base, 1200/64).

    Uses tiktoken when available so chunk boundaries match DyG-RAG/IA-RAG exactly;
    falls back to whitespace-word windows otherwise (approximate — words are coarser
    than tokens, so install tiktoken for a faithful comparison).
    """
    step = max(1, chunk_size - overlap)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= chunk_size:
            return [text]
        return [enc.decode(toks[s:s + chunk_size]).strip()
                for s in range(0, len(toks), step)]
    except ImportError:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        return [" ".join(words[s:s + chunk_size]) for s in range(0, len(words), step)]


def build_docs(corpus, chunk_size=0, chunk_overlap=64):
    """Build the (docs, timestamps, provenances) lists fed to index().

    When chunk_size <= 0 this is the original one-doc-per-entry behavior (static
    benchmarks unaffected). When chunk_size > 0 each document's text is split into
    sliding-window chunks; every chunk inherits its parent doc's timestamp/provenance
    so the temporal metadata plumbing stays intact (text_to_meta keyed by chunk text).
    """
    docs, timestamps, provenances = [], [], []
    for doc in corpus:
        title = doc['title']
        ts = doc.get('timestamp', 0)
        prov = doc.get('provenance', title)
        if chunk_size and chunk_size > 0:
            pieces = _chunk_text(doc['text'], chunk_size, chunk_overlap)
        else:
            pieces = [doc['text']]
        for piece in pieces:
            docs.append(f"{title}\n{piece}")
            timestamps.append(ts)
            provenances.append(prov)
    return docs, timestamps, provenances


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument('--temporal_weighting', type=str, default='false',
                        help='If True, scale PPR edge weights by recency (newer edges weigh more). Default False = static behavior.')
    parser.add_argument('--chunk_size', type=int, default=0,
                        help='If > 0, split each corpus doc into sliding-window chunks of this many tokens before indexing (DyG-RAG/IA-RAG use 1200). 0 = no chunking = original behavior, so static benchmarks are unaffected.')
    parser.add_argument('--chunk_overlap', type=int, default=64,
                        help='Token overlap between chunks when --chunk_size > 0 (DyG-RAG default 64).')
    parser.add_argument('--fact_time_anchor', type=str, default='false',
                        help='D1 (opt-in). If True, the LLM assigns a timestamp to each extracted triple (post-OpenIE pass) and that per-fact time is put on the fact edge instead of the doc-level timestamp. Uses a separate working dir (_ft suffix). Default false = original behavior.')
    parser.add_argument('--time_cot', type=str, default='false',
                        help='D2 (opt-in). If True, retrieved facts are ordered chronologically into a timeline prepended to the QA prompt with a temporal-reasoning instruction (DyG-RAG Time-CoT). QA-time only; reuses the same working dir. Pair with --fact_time_anchor for real timestamps. Default false = original behavior.')
    parser.add_argument('--time_scoped', type=str, default='false',
                        help='D3 (opt-in). If True, fact scores are weighted by temporal proximity to the year asked in the question (scopes to the asked time, not newest). Retrieval-time only; reuses the same working dir. Pair with --fact_time_anchor for real timestamps. Default false = original behavior.')
    parser.add_argument('--time_scope_tau', type=float, default=3.0,
                        help='D3 Gaussian decay scale in years for temporal proximity. Smaller = sharper scoping. Default 3.0.')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name
    # Chunked and non-chunked indexes must not share a working dir, or the cached
    # graph from one would be silently reused by the other.
    if args.chunk_size and args.chunk_size > 0:
        save_dir = save_dir + f'_chunk{args.chunk_size}'
    # D1: keep the per-fact-time graph in its own working dir so the baseline
    # graph/cache is never overwritten and stays reproducible.
    if string_to_bool(args.fact_time_anchor):
        save_dir = save_dir + '_ft'

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs, doc_timestamps, doc_provenances = build_docs(corpus, args.chunk_size, args.chunk_overlap)

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode=args.openie_mode,
        temporal_weighting=string_to_bool(args.temporal_weighting),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fact_time_anchor=string_to_bool(args.fact_time_anchor),
        time_cot=string_to_bool(args.time_cot),
        time_scoped=string_to_bool(args.time_scoped),
        time_scope_tau=args.time_scope_tau
    )

    logging.basicConfig(level=logging.INFO)
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs, doc_timestamps=doc_timestamps, doc_provenances=doc_provenances)
    
    res = hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)

    # Save per-question predictions so offline analysis (e.g. reproduce/eval_temporal.py)
    # can compute per-category / stale-rate metrics without re-running inference.
    # Named by the temporal_weighting flag so on/off A/B runs don't overwrite.
    queries_solutions = res[0]
    tw_flag = 'on' if config.temporal_weighting else 'off'
    # D2: mark Time-CoT runs so they don't overwrite the non-CoT predictions in
    # the same working dir. Omitted when off so baseline filenames are unchanged.
    cot_suffix = '_cot_on' if config.time_cot else ''
    ts_suffix = '_ts_on' if config.time_scoped else ''
    pred_path = os.path.join(save_dir, f"predictions_tw_{tw_flag}{cot_suffix}{ts_suffix}.json")
    with open(pred_path, "w") as f:
        json.dump([qs.to_dict() for qs in queries_solutions], f, indent=2)

    print("\n" + "="*10 + " EVALUATION METRICS " + "="*10)
    print("Retrieval Metrics:", res[3] if len(res) == 5 else "N/A")
    print("QA Metrics:", res[4] if len(res) == 5 else "N/A")
    print(f"Saved predictions to {pred_path}")
    print("="*40)

if __name__ == "__main__":
    main()
