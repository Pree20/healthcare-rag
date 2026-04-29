from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Takes the parsed sections from parse_fda.py and splits each one
    into overlapping chunks. Returns a flat list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Splitter tries these separators in order.
        # It tries to break on a paragraph first, then a sentence,
        # then a word, and only splits mid-word as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,  # counts characters, not tokens
    )

    chunks = []

    for section in sections:
        # split_text() returns a list of strings
        texts = splitter.split_text(section["text"])

        for i, text in enumerate(texts):
            chunks.append({
                # The actual text that will be embedded and stored
                "text": text,
                # Metadata travels alongside the chunk forever
                # This is what powers your citations at query time
                "metadata": {
                    "drug_name": section["drug_name"],
                    "section":   section["section"],
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
                # Unique ID for ChromaDB — must be a string
                "id": f"{section['drug_name']}_{section['section']}_{i}".replace(" ", "_")
            })

    return chunks