import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    parent_id: str
    chunk_id: str
    strategy: str
    start_sentence: int
    end_sentence: int


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?।॥])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def keep_intact(
    text: str,
    parent_id: str,
) -> list[Chunk]:
    return [
        Chunk(
            text=text,
            parent_id=parent_id,
            chunk_id=f"{parent_id}:0",
            strategy="original",
            start_sentence=0,
            end_sentence=0,
        )
    ]


def sentence_windows(
    text: str,
    parent_id: str,
    window_size: int = 2,
) -> list[Chunk]:
    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []

    for i in range(len(sentences)):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        chunks.append(
            Chunk(
                text=" ".join(sentences[start:end]),
                parent_id=parent_id,
                chunk_id=f"{parent_id}:{i}",
                strategy="sentence_window",
                start_sentence=start,
                end_sentence=end - 1,
            )
        )

    return chunks


def adaptive_chunks(
    text: str,
    parent_id: str,
    max_words: int = 100,
    target_words: int = 70,
    hard_max_words: int = 140,
) -> list[Chunk]:

    word_count = len(text.split())

    # Most passages in our dataset are short.
    # Keep them intact to avoid unnecessary chunk explosion.
    if word_count <= max_words:
        return keep_intact(text, parent_id)

    sentences = split_sentences(text)

    if not sentences:
        return keep_intact(text, parent_id)

    # If sentence detection cannot split a long passage,
    # fall back to hard word-based chunks.
    if len(sentences) == 1:
        words = text.split()

        chunks = []

        for i in range(0, len(words), hard_max_words):
            chunk_words = words[i:i + hard_max_words]

            chunks.append(
                Chunk(
                    text=" ".join(chunk_words),
                    parent_id=parent_id,
                    chunk_id=f"{parent_id}:{len(chunks)}",
                    strategy="adaptive",
                    start_sentence=0,
                    end_sentence=0,
                )
            )

        return chunks

    chunks = []

    current = []
    current_words = 0
    start_sentence = 0

    for index, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())

        # A single sentence may itself exceed the hard limit.
        if sentence_words > hard_max_words:
            if current:
                chunks.append(
                    Chunk(
                        text=" ".join(current),
                        parent_id=parent_id,
                        chunk_id=f"{parent_id}:{len(chunks)}",
                        strategy="adaptive",
                        start_sentence=start_sentence,
                        end_sentence=index - 1,
                    )
                )

                current = []
                current_words = 0

            words = sentence.split()

            for i in range(0, len(words), hard_max_words):
                chunk_words = words[i:i + hard_max_words]

                chunks.append(
                    Chunk(
                        text=" ".join(chunk_words),
                        parent_id=parent_id,
                        chunk_id=f"{parent_id}:{len(chunks)}",
                        strategy="adaptive",
                        start_sentence=index,
                        end_sentence=index,
                    )
                )

            start_sentence = index + 1
            continue

        # Start a new chunk when adding this sentence would
        # move us beyond the target size.
        if (
            current
            and current_words + sentence_words > target_words
        ):
            chunks.append(
                Chunk(
                    text=" ".join(current),
                    parent_id=parent_id,
                    chunk_id=f"{parent_id}:{len(chunks)}",
                    strategy="adaptive",
                    start_sentence=start_sentence,
                    end_sentence=index - 1,
                )
            )

            current = []
            current_words = 0
            start_sentence = index

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(
            Chunk(
                text=" ".join(current),
                parent_id=parent_id,
                chunk_id=f"{parent_id}:{len(chunks)}",
                strategy="adaptive",
                start_sentence=start_sentence,
                end_sentence=len(sentences) - 1,
            )
        )

    return chunks
