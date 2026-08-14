import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE = Path("data/processed/chunks")
EMBEDDING_BASE = Path("data/processed/chunk_embeddings")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STRATEGIES = [
    "original",
    "adaptive",
    "sentence_window",
]

TOP_K_VALUES = [1, 3, 5, 10]
RRF_K = 60


def load_chunks(strategy):
    path = BASE / f"{strategy}.jsonl"

    with path.open("r", encoding="utf-8") as f:
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


def evaluate(ranking, relevant_parents, chunks, top_k):
    top = ranking[:top_k]

    hit = any(
        chunks[index]["parent_id"] in relevant_parents
        for index in top
    )

    rr = 0.0

    for rank, index in enumerate(ranking, start=1):
        parent_id = chunks[index]["parent_id"]

        if parent_id in relevant_parents:
            rr = 1.0 / rank
            break

    return int(hit), rr


def benchmark_strategy(strategy, model):
    print("\n" + "=" * 60)
    print(f"STRATEGY: {strategy}")
    print("=" * 60)

    chunks = load_chunks(strategy)

    print(f"Chunks: {len(chunks):,}")

    # ---------------------------------------------------------
    # Group chunks by query.
    # ---------------------------------------------------------

    queries = {}

    for index, chunk in enumerate(chunks):
        query_id = chunk["query_id"]

        if query_id not in queries:
            queries[query_id] = {
                "query": chunk["query"],
                "relevant_parents": set(),
            }

        if chunk["is_selected"] == 1:
            queries[query_id]["relevant_parents"].add(
                chunk["parent_id"]
            )

    query_ids = list(queries.keys())

    print(f"Queries: {len(query_ids):,}")

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    EMBEDDING_BASE.mkdir(parents=True, exist_ok=True)

    embedding_file = (
        EMBEDDING_BASE / f"{strategy}.npy"
    )

    if embedding_file.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(
            embedding_file,
            mmap_mode="r",
        )

    else:
        print("Encoding chunks...")

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        np.save(
            embedding_file,
            embeddings,
        )

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------

    query_texts = [
        queries[qid]["query"]
        for qid in query_ids
    ]

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

    print("Building BM25 index...")

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    results = {
        k: []
        for k in TOP_K_VALUES
    }

    reciprocal_ranks = []

    retrieval_latencies = []

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    for q_index, query_id in enumerate(query_ids):

        query = queries[query_id]

        start = time.perf_counter()

        # Dense
        dense_scores = (
            embeddings @ query_embeddings[q_index]
        )

        dense_ranking = np.argsort(
            dense_scores
        )[::-1][:100].tolist()

        # BM25
        bm25_scores = bm25.get_scores(
            tokenize(query["query"])
        )

        bm25_ranking = np.argsort(
            bm25_scores
        )[::-1][:100].tolist()

        # RRF
        ranking = reciprocal_rank_fusion(
            [
                dense_ranking,
                bm25_ranking,
            ],
            k=RRF_K,
        )[:100]

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        retrieval_latencies.append(elapsed)

        relevant = query["relevant_parents"]

        for k in TOP_K_VALUES:
            hit, _ = evaluate(
                ranking,
                relevant,
                chunks,
                k,
            )

            results[k].append(hit)

        _, rr = evaluate(
            ranking,
            relevant,
            chunks,
            100,
        )

        reciprocal_ranks.append(rr)

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\nRESULTS")

    for k in TOP_K_VALUES:
        print(
            f"Recall@{k}: "
            f"{np.mean(results[k]):.4f}"
        )

    print(
        f"MRR: "
        f"{np.mean(reciprocal_ranks):.4f}"
    )

    print(
        f"Mean retrieval: "
        f"{np.mean(retrieval_latencies):.3f} ms"
    )


def main():
    model = SentenceTransformer(MODEL_NAME)

    for strategy in STRATEGIES:
        benchmark_strategy(
            strategy,
            model,
        )


if __name__ == "__main__":
    main()
