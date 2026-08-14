import pyarrow.parquet as pq
import fsspec

URL = (
    "https://huggingface.co/datasets/"
    "ai4bharat/MSMARCO-XI/resolve/main/"
    "train/martrain.parquet"
)

print("Opening Parquet metadata...")
print(URL)

fs = fsspec.filesystem("https")
with fs.open(URL, "rb") as f:
    parquet = pq.ParquetFile(f)

    print("\n=== DATASET METADATA ===")
    print("Rows:", parquet.metadata.num_rows)
    print("Columns:", parquet.metadata.num_columns)
    print("Row groups:", parquet.metadata.num_row_groups)

    print("\n=== COLUMNS ===")
    for i in range(parquet.metadata.num_columns):
        column = parquet.metadata.schema.column(i)
        print(f"{i}: {column.name} | {column.physical_type}")

    print("\n=== ROW GROUPS ===")

    for i in range(min(parquet.metadata.num_row_groups, 10)):
        group = parquet.metadata.row_group(i)

        print(
            f"Row group {i}: "
            f"{group.num_rows:,} rows, "
            f"{group.total_byte_size / 1024 / 1024:.2f} MB"
        )
