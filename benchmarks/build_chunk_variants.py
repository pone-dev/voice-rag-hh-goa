import json
from pathlib import Path

from ingestion.chunking.adaptive import (
    adaptive_chunks,
    keep_intact,
    sentence_windows,
)


INPUT = Path("data/processed/marathi_documents.jsonl")
OUTPUT_DIR = Path("data/processed/chunks")


def serialize(chunk, document):
    return {
        "query_id": document["query_id"],
        "passage_id": document["passage_id"],
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "strategy": chunk.strategy,
        "text": chunk.text,
        "english_text": document["english_text"],
        "query": document["query"],
        "query_type": document["query_type"],
        "is_selected": document["is_selected"],
        "start_sentence": chunk.start_sentence,
        "end_sentence": chunk.end_sentence,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writers = {
        "original": (
            OUTPUT_DIR / "original.jsonl"
        ).open("w", encoding="utf-8"),
        "sentence_window": (
            OUTPUT_DIR / "sentence_window.jsonl"
        ).open("w", encoding="utf-8"),
        "adaptive": (
            OUTPUT_DIR / "adaptive.jsonl"
        ).open("w", encoding="utf-8"),
    }

    counts = {
        "original": 0,
        "sentence_window": 0,
        "adaptive": 0,
    }

    try:
        with INPUT.open("r", encoding="utf-8") as source:
            for line in source:
                document = json.loads(line)

                parent_id = (
                    f"{document['query_id']}:"
                    f"{document['passage_id']}"
                )

                variants = {
                    "original": keep_intact(
                        document["text"],
                        parent_id,
                    ),
                    "sentence_window": sentence_windows(
                        document["text"],
                        parent_id,
                    ),
                    "adaptive": adaptive_chunks(
                        document["text"],
                        parent_id,
                    ),
                }

                for strategy, chunks in variants.items():
                    for chunk in chunks:
                        writers[strategy].write(
                            json.dumps(
                                serialize(chunk, document),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        counts[strategy] += 1

    finally:
        for writer in writers.values():
            writer.close()

    print("\n==============================")
    print("CHUNK VARIANTS CREATED")
    print("==============================")

    for strategy, count in counts.items():
        print(f"{strategy:18s}: {count:,}")


if __name__ == "__main__":
    main()
