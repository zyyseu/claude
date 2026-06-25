"""
Convert common reranking datasets to the JSONL format expected by train_reranker.py.

Supported input formats:
  - MS MARCO (TREC-DL): queries.tsv + qrels.tsv + passages
  - BEIR / MTEB: corpus.jsonl + queries.jsonl + qrels/
  - HuggingFace datasets

Output format (JSONL):
  {"query": "what is ...", "documents": ["d1", "d2", ...], "labels": [1.0, 0.0, ...]}
"""

import os
import json
import argparse
import logging
import csv
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_jsonl(groups: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(groups)} query groups to {output_path}")


def msmarco_to_groups(
    queries_path: str,
    qrels_path: str,
    passages_path: str,
    max_docs: int = 20,
    max_queries: int = 0,
) -> List[Dict]:
    """
    Convert MS MARCO TREC-DL format.

    queries.tsv:  qid \t query
    qrels.tsv:    qid \t 0 \t pid \t relevance
    passages.tsv: pid \t text
    """
    logger.info("Loading MS MARCO data...")

    queries: Dict[str, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    logger.info(f"  {len(queries)} queries loaded")

    passages: Dict[str, str] = {}
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                passages[parts[0]] = parts[1]
    logger.info(f"  {len(passages)} passages loaded")

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels[qid][pid] = rel
    logger.info(f"  {len(qrels)} queries with relevant passages")

    groups = []
    for qid, pid2rel in qrels.items():
        if qid not in queries:
            continue

        # Positive docs
        pos_pids = sorted(pid2rel.keys(), key=lambda p: pid2rel[p], reverse=True)
        pos_docs = [(passages[p], pid2rel[p]) for p in pos_pids if p in passages]

        if len(pos_docs) < 2:
            continue

        docs, labels = zip(*pos_docs[:max_docs])

        groups.append({
            "query": queries[qid],
            "query_id": qid,
            "documents": list(docs),
            "labels": list(labels),
        })

        if max_queries and len(groups) >= max_queries:
            break

    return groups


def beir_to_groups(
    corpus_path: str,
    queries_path: str,
    qrels_dir: str,
    max_docs: int = 20,
    max_queries: int = 0,
) -> List[Dict]:
    """
    Convert BEIR / MTEB format.

    corpus.jsonl:   {"_id": "pid", "text": "...", "title": "..."}
    queries.jsonl:  {"_id": "qid", "text": "..."}
    qrels/dev.tsv:  qid \t 0 \t pid \t score
    """
    logger.info("Loading BEIR data...")

    passages: Dict[str, str] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            text = item.get("title", "") + " " + item["text"]
            passages[item["_id"]] = text.strip()
    logger.info(f"  {len(passages)} passages loaded")

    queries: Dict[str, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            queries[item["_id"]] = item["text"]
    logger.info(f"  {len(queries)} queries loaded")

    # Load all qrels files in the directory
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for fname in os.listdir(qrels_dir):
        fpath = os.path.join(qrels_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, _, pid, score = parts[0], parts[1], parts[2], int(parts[3])
                    qrels[qid][pid] = max(qrels[qid].get(pid, 0), score)
    logger.info(f"  {len(qrels)} queries with relevant passages")

    groups = []
    for qid, pid2rel in qrels.items():
        if qid not in queries:
            continue

        pos_pids = sorted(pid2rel.keys(), key=lambda p: pid2rel[p], reverse=True)
        pos_docs = [(passages[p], pid2rel[p]) for p in pos_pids if p in passages]

        if len(pos_docs) < 2:
            continue

        docs, labels = zip(*pos_docs[:max_docs])

        groups.append({
            "query": queries[qid],
            "query_id": qid,
            "documents": list(docs),
            "labels": list(labels),
        })

        if max_queries and len(groups) >= max_queries:
            break

    return groups


def inject_negatives(
    groups: List[Dict],
    corpus_passages: Dict[str, str],
    num_negatives: int = 4,
) -> List[Dict]:
    """
    Add random negative documents to each query group to enable more
    effective listwise training.
    """
    import random
    all_pids = list(corpus_passages.keys())

    for g in groups:
        n_pos = len(g["documents"])
        n_neg = min(num_negatives, len(all_pids) - n_pos)
        neg_docs = random.sample(all_pids, n_neg * 2)  # oversample
        neg_docs = [p for p in neg_docs if p not in g.get("positive_pids", [])]
        neg_docs = neg_docs[:n_neg]

        for pid in neg_docs:
            g["documents"].append(corpus_passages[pid])
            g["labels"].append(0.0)

    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert reranking datasets to JSONL for train_reranker.py"
    )
    p.add_argument("--format", required=True, choices=["msmarco", "beir"])
    p.add_argument("--output", required=True, help="Output JSONL file")

    # MS MARCO
    p.add_argument("--queries", help="Path to queries.tsv")
    p.add_argument("--qrels", help="Path to qrels.tsv")
    p.add_argument("--passages", help="Path to passages.tsv")

    # BEIR
    p.add_argument("--corpus", help="Path to corpus.jsonl")
    p.add_argument("--qrels_dir", help="Directory containing qrels TSV files")

    # Options
    p.add_argument("--max_docs", type=int, default=20)
    p.add_argument("--max_queries", type=int, default=0,
                   help="Limit number of queries (0 = all)")

    args = p.parse_args()

    if args.format == "msmarco":
        groups = msmarco_to_groups(
            queries_path=args.queries,
            qrels_path=args.qrels,
            passages_path=args.passages,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
    elif args.format == "beir":
        groups = beir_to_groups(
            corpus_path=args.corpus,
            queries_path=args.queries,
            qrels_dir=args.qrels_dir,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )

    save_jsonl(groups, args.output)


if __name__ == "__main__":
    main()
