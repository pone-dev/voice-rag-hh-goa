import pyarrow.parquet as pq
import fsspec

URL = (
    "https://huggingface.co/datasets/"
    "ai4bharat/MSMARCO-XI/resolve/main/"
    "train/martrain.parquet"
)

TARGETS = {
    "query",
    "Eng_Query",
    "Eng_Answer",
    "Answer",
    "query_id",
    "query_type",
}

print("Inspecting column chunk sizes...\n")

fs = fsspec.filesystem("https")

with fs.open(URL, "rb") as f:
    pf = pq.ParquetFile(f)
    rg = pf.metadata.row_group(0)

    total = 0

    for i in range(rg.num_columns):
        column = rg.column(i)
        name = column.path_in_schema

        size_mb = column.total_compressed_size / 1024 / 1024
        total += column.total_compressed_size

        marker = " <-- TARGET" if name in TARGETS else ""

        print(
            f"{i:2d}  "
            f"{name:30s} "
            f"{size_mb:10.2f} MB"
            f"{marker}"
        )

    print("\nTotal compressed size:")
    print(f"{total / 1024 / 1024:.2f} MB")
