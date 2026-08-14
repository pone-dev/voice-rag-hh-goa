import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


DATASET = Path("data/processed/marathi_documents.jsonl")
TOP_K = 5


def load_documents():
    with DATASET.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return text.lower().split()


def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Documents: {len(documents):,}")

    # Group candidate passages by query.
    grouped = defaultdict(list)

    for index, doc in enumerate(documents):
        grouped[doc["query_id"]].append(index)

    query_ids = list(grouped.keys())

    print(f"Queries: {len(query_ids):,}")

    print("\nBuilding BM25 index...")

    corpus = [
        tokenize(doc["text"])
        for doc in documents
    ]

    bm25 = BM25Okapi(corpus)

    recall_hits = []
    reciprocal_ranks = []

    print("\nRunning retrieval...")

    for query_id in tqdm(query_ids):
        candidate_indices = grouped[query_id]

        query = documents[candidate_indices[0]]["query"]
        query_tokens = tokenize(query)

        scores = bm25.get_scores(query_tokens)

        # IMPORTANT:
        # For this benchmark, rank only the 10 candidate passages
        # associated with this query, matching our dense baseline.
        ranked = sorted(
            candidate_indices,
            key=lambda idx: scores[idx],
            reverse=True,
        )

        relevant = {
            idx
            for idx in candidate_indices
            if documents[idx]["is_selected"] == 1
        }

        top_k = ranked[:TOP_K]

        hit = any(idx in relevant for idx in top_k)
        recall_hits.append(int(hit))

        rank = None

        for position, idx in enumerate(ranked, start=1):
            if idx in relevant:
                rank = position
                break

        reciprocal_ranks.append(
            0.0 if rank is None else 1.0 / rank
        )

    recall_at_k = float(np.mean(recall_hits))
    mrr = float(np.mean(reciprocal_ranks))

    print("\n==============================")
    print("BM25 RETRIEVAL RESULTS")
    print("==============================")
    print(f"Queries:     {len(query_ids):,}")
    print(f"Recall@{TOP_K}: {recall_at_k:.4f}")
    print(f"MRR:         {mrr:.4f}")


if __name__ == "__main__":
    main()
