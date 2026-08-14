import pyarrow.parquet as pq
import fsspec

LANGUAGES = [
    "asm", "ben", "guj", "hin", "kan",
    "mal", "mar", "nep", "ori", "pan",
    "san", "tam", "tel", "urd"
]

BASE = (
    "https://huggingface.co/datasets/"
    "ai4bharat/MSMARCO-XI/resolve/main/"
    "validation"
)

fs = fsspec.filesystem("https")

print(f"{'Language':<10} {'Rows':>12} {'Size (MB)':>14} {'Row groups':>12}")
print("-" * 52)

for lang in LANGUAGES:
    filename = f"{lang}val.parquet"
    url = f"{BASE}/{filename}"

    try:
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            rows = pf.metadata.num_rows
            size = pf.metadata.serialized_size / 1024 / 1024

            # Sum compressed column chunks for a better physical-size estimate
            compressed = 0
            rg = pf.metadata.row_group(0)

            for i in range(rg.num_columns):
                compressed += rg.column(i).total_compressed_size

            print(
                f"{lang:<10} "
                f"{rows:>12,} "
                f"{compressed / 1024 / 1024:>14.2f} "
                f"{pf.metadata.num_row_groups:>12}"
            )

    except Exception as e:
        print(f"{lang:<10} ERROR: {e}")
