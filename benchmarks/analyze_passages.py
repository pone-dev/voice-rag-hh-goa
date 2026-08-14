import json
import numpy as np

PATH = "data/processed/marathi_documents.jsonl"

lengths = []

with open(PATH, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = row["text"]

        lengths.append({
            "chars": len(text),
            "words": len(text.split()),
        })

chars = np.array([x["chars"] for x in lengths])
words = np.array([x["words"] for x in lengths])

print("================================")
print("PASSAGE LENGTH ANALYSIS")
print("================================")

for name, values in [
    ("Characters", chars),
    ("Words", words),
]:
    print(f"\n{name}")
    print(f"  Min:    {np.min(values):.0f}")
    print(f"  P25:    {np.percentile(values, 25):.0f}")
    print(f"  Median: {np.median(values):.0f}")
    print(f"  P75:    {np.percentile(values, 75):.0f}")
    print(f"  P90:    {np.percentile(values, 90):.0f}")
    print(f"  P95:    {np.percentile(values, 95):.0f}")
    print(f"  P99:    {np.percentile(values, 99):.0f}")
    print(f"  Max:    {np.max(values):.0f}")
