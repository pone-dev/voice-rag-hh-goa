import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATASET = Path("data/processed/marathi_documents.jsonl")
OUTPUT_DIR = Path("data/processed/embeddings")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBEDDINGS_FILE = OUTPUT_DIR / "passages.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    print(f"Documents: {len(documents):,}")

    if EMBEDDINGS_FILE.exists():
        print(f"\nEmbeddings already exist: {EMBEDDINGS_FILE}")
        print("Delete the file only if you intentionally want to rebuild.")
        return

    print(f"\nLoading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [doc["text"] for doc in documents]

    print("\nEncoding passages...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nSaving embeddings...")

    np.save(EMBEDDINGS_FILE, embeddings)

    metadata = [
        {
            "query_id": doc["query_id"],
            "passage_id": doc["passage_id"],
            "is_selected": doc["is_selected"],
        }
        for doc in documents
    ]

    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("\n==============================")
    print("DENSE INDEX CREATED")
    print("==============================")
    print(f"Embeddings: {EMBEDDINGS_FILE}")
    print(f"Shape:      {embeddings.shape}")
    print(f"Metadata:   {METADATA_FILE}")


if __name__ == "__main__":
    main()
