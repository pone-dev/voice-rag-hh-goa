from huggingface_hub import hf_hub_download
import json

REPO = "ai4bharat/MSMARCO-XI"

# Start with Marathi because it is useful for our Indian-language voice demo.
LANGUAGE = "mr"
FILE = f"train/{LANGUAGE}train.jsonl"

print(f"Dataset: {REPO}")
print(f"Language: {LANGUAGE}")
print(f"File: {FILE}")
print("\nDownloading only the dataset file through the HF cache...")

path = hf_hub_download(
    repo_id=REPO,
    filename=FILE,
    repo_type="dataset",
)

print(f"\nLocal cached file: {path}")

print("\nReading first example...\n")

with open(path, "r", encoding="utf-8") as f:
    line = f.readline()

example = json.loads(line)

print("TOP-LEVEL FIELDS:")
for key, value in example.items():
    print(f"  {key}: {type(value).__name__}")

print("\nCONTENT:\n")

for key, value in example.items():
    if key == "passages":
        print("passages:")
        print(f"  fields: {list(value.keys())}")

        for pkey, pvalue in value.items():
            if isinstance(pvalue, list):
                print(f"  {pkey}: {len(pvalue)} items")

                if pvalue:
                    sample = str(pvalue[0])
                    print(f"    sample: {sample[:300]}")

    elif isinstance(value, dict):
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue}")

    else:
        text = str(value)
        print(f"{key}: {text[:500]}")

print("\nInspection complete.")
