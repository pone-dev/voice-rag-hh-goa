import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATASET = Path("data/processed/marathi_documents.jsonl")

# Small multilingual model suitable for our first experiment.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5


def load_documents():
    documents = []

    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    return documents


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Documents: {len(documents):,}")

    # Keep document indices so retrieval does not repeatedly scan the list.
    grouped = defaultdict(list)

    for index, doc in enumerate(documents):
        grouped[doc["query_id"]].append(index)

    query_ids = list(grouped.keys())

    # Every query appears with its candidate passages.
    queries = [
        documents[grouped[qid][0]]["query"]
        for qid in query_ids
    ]

    print(f"Queries: {len(queries):,}")

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("\nEncoding passages...")

    passage_texts = [
        doc["text"]
        for doc in documents
    ]

    passage_embeddings = model.encode(
        passage_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nEncoding queries...")

    query_embeddings = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Cosine similarity because embeddings are normalized.
    similarity = query_embeddings @ passage_embeddings.T

    recall_hits = []
    reciprocal_ranks = []

    for query_index, query_id in enumerate(query_ids):
        scores = similarity[query_index]

        # Only rank passages belonging to this query.
        candidate_indices = grouped[query_id]

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

        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)

    recall_at_k = float(np.mean(recall_hits))
    mrr = float(np.mean(reciprocal_ranks))

    print("\n==============================")
    print("BASELINE RETRIEVAL RESULTS")
    print("==============================")
    print(f"Queries:     {len(query_ids):,}")
    print(f"Recall@{TOP_K}: {recall_at_k:.4f}")
    print(f"MRR:         {mrr:.4f}")


if __name__ == "__main__":
    main()
