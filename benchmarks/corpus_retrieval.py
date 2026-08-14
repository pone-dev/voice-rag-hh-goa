import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DATASET = Path("data/processed/marathi_documents.jsonl")
EMBEDDINGS = Path("data/processed/embeddings/passages.npy")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_VALUES = [1, 3, 5, 10]
RRF_K = 60


def load_documents():
    with DATASET.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return text.lower().split()


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (k + rank)

    return [
        index
        for index, _ in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def evaluate(ranked, relevant, top_k):
    top = ranked[:top_k]

    hit = any(index in relevant for index in top)

    rr = 0.0

    for rank, index in enumerate(ranked, start=1):
        if index in relevant:
            rr = 1.0 / rank
            break

    return int(hit), rr


def print_results(name, results):
    print(f"\n{name}")
    print("-" * 40)

    for k in TOP_K_VALUES:
        print(
            f"Recall@{k}: "
            f"{np.mean(results[k]['hits']):.4f}"
        )

    print(
        f"MRR: "
        f"{np.mean(results['mrr']):.4f}"
    )


def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Documents: {len(documents):,}")

    # ---------------------------------------------------------
    # Build query groups and ground-truth relevance.
    # ---------------------------------------------------------

    queries = {}

    for index, doc in enumerate(documents):
        query_id = doc["query_id"]

        if query_id not in queries:
            queries[query_id] = {
                "query": doc["query"],
                "indices": [],
                "relevant": set(),
            }

        queries[query_id]["indices"].append(index)

        if doc["is_selected"] == 1:
            queries[query_id]["relevant"].add(index)

    print(f"Queries: {len(queries):,}")

    # ---------------------------------------------------------
    # Load dense embeddings.
    # ---------------------------------------------------------

    print("\nLoading cached embeddings...")

    passage_embeddings = np.load(
        EMBEDDINGS,
        mmap_mode="r",
    )

    print(f"Embedding shape: {passage_embeddings.shape}")

    model = SentenceTransformer(MODEL_NAME)

    query_ids = list(queries.keys())

    query_texts = [
        queries[qid]["query"]
        for qid in query_ids
    ]

    print("\nEncoding queries...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print("\nBuilding BM25 index...")

    corpus = [
        tokenize(doc["text"])
        for doc in documents
    ]

    bm25 = BM25Okapi(corpus)

    # ---------------------------------------------------------
    # Result containers
    # ---------------------------------------------------------

    dense_results = {
        k: {"hits": []}
        for k in TOP_K_VALUES
    }
    dense_results["mrr"] = []

    bm25_results = {
        k: {"hits": []}
        for k in TOP_K_VALUES
    }
    bm25_results["mrr"] = []

    rrf_results = {
        k: {"hits": []}
        for k in TOP_K_VALUES
    }
    rrf_results["mrr"] = []

    dense_latency = []
    bm25_latency = []
    rrf_latency = []

    print("\nRunning corpus-wide retrieval...")

    for query_position, query_id in enumerate(query_ids):

        query = queries[query_id]
        relevant = query["relevant"]

        # =====================================================
        # Dense
        # =====================================================

        start = time.perf_counter()

        scores = passage_embeddings @ query_embeddings[query_position]

        dense_ranking = np.argsort(scores)[::-1][:100].tolist()

        dense_latency.append(
            (time.perf_counter() - start) * 1000
        )

        for k in TOP_K_VALUES:
            hit, _ = evaluate(
                dense_ranking,
                relevant,
                k,
            )
            dense_results[k]["hits"].append(hit)

        _, rr = evaluate(
            dense_ranking,
            relevant,
            100,
        )
        dense_results["mrr"].append(rr)

        # =====================================================
        # BM25
        # =====================================================

        start = time.perf_counter()

        bm25_scores = bm25.get_scores(
            tokenize(query["query"])
        )

        bm25_ranking = np.argsort(bm25_scores)[::-1][:100].tolist()

        bm25_latency.append(
            (time.perf_counter() - start) * 1000
        )

        for k in TOP_K_VALUES:
            hit, _ = evaluate(
                bm25_ranking,
                relevant,
                k,
            )
            bm25_results[k]["hits"].append(hit)

        _, rr = evaluate(
            bm25_ranking,
            relevant,
            100,
        )
        bm25_results["mrr"].append(rr)

        # =====================================================
        # RRF
        # =====================================================

        start = time.perf_counter()

        rrf_ranking = reciprocal_rank_fusion(
            [
                dense_ranking,
                bm25_ranking,
            ],
            k=RRF_K,
        )[:100]

        rrf_latency.append(
            (time.perf_counter() - start) * 1000
        )

        for k in TOP_K_VALUES:
            hit, _ = evaluate(
                rrf_ranking,
                relevant,
                k,
            )
            rrf_results[k]["hits"].append(hit)

        _, rr = evaluate(
            rrf_ranking,
            relevant,
            100,
        )
        rrf_results["mrr"].append(rr)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 50)
    print("CORPUS-WIDE RETRIEVAL RESULTS")
    print("=" * 50)

    print_results("DENSE", dense_results)
    print_results("BM25", bm25_results)
    print_results("DENSE + BM25 (RRF)", rrf_results)

    print("\nLatency")
    print("-" * 40)

    print(
        f"Dense mean: "
        f"{np.mean(dense_latency):.3f} ms"
    )

    print(
        f"BM25 mean:  "
        f"{np.mean(bm25_latency):.3f} ms"
    )

    print(
        f"RRF mean:   "
        f"{np.mean(rrf_latency):.3f} ms"
    )


if __name__ == "__main__":
    main()
