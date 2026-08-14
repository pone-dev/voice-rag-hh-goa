import json
from pathlib import Path


INPUT = Path("data/processed/marathi_sample.jsonl")
OUTPUT = Path("data/processed/marathi_documents.jsonl")


def normalize_record(record):
    passages = record["passages"]

    english = passages["English_passages"]
    translated = passages["Translated_passages"]
    selected = passages["is_selected"]

    documents = []

    for passage_id, (en, mr, label) in enumerate(
        zip(english, translated, selected)
    ):
        documents.append(
            {
                "query_id": record["query_id"],
                "passage_id": passage_id,
                "source_lang": record["source_lang"],
                "target_lang": record["target_lang"],
                "query_type": record["query_type"],
                "query": record["query"],
                "english_query": record["Eng_Query"],
                "answer": record["Answer"],
                "english_answer": record["Eng_Answer"],
                "text": mr,
                "english_text": en,
                "is_selected": int(label),
            }
        )

    return documents


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    document_count = 0

    with INPUT.open("r", encoding="utf-8") as source, \
         OUTPUT.open("w", encoding="utf-8") as destination:

        for line in source:
            record = json.loads(line)

            documents = normalize_record(record)

            for document in documents:
                destination.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                document_count += 1

    print(f"Created {document_count:,} retrieval documents.")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
