import json
from pathlib import Path

import pyarrow.parquet as pq


INPUT = Path("data/raw/marval.parquet")
OUTPUT = Path("data/processed/marathi_sample.jsonl")

# Small enough for development, large enough for retrieval experiments.
SAMPLE_SIZE = 1000

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Opening: {INPUT}")

parquet = pq.ParquetFile(INPUT)

print(f"Rows available: {parquet.metadata.num_rows:,}")
print(f"Columns: {parquet.schema.names}")

# Read incrementally instead of materializing the entire dataset.
columns = [
    "source_lang",
    "target_lang",
    "Answer",
    "query_id",
    "query_type",
    "passages",
    "Eng_Query",
    "Eng_Answer",
    "query",
]

written = 0

with OUTPUT.open("w", encoding="utf-8") as out:
    for batch in parquet.iter_batches(
        batch_size=100,
        columns=columns,
    ):
        rows = batch.to_pylist()

        for row in rows:
            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

            if written >= SAMPLE_SIZE:
                break

        if written >= SAMPLE_SIZE:
            break

print(f"\nWrote {written:,} records")
print(f"Output: {OUTPUT}")
