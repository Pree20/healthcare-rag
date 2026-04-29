import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from config.settings import DATA_PROCESSED_DIR


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _load_chunks() -> list[dict]:
    cache_path = Path(DATA_PROCESSED_DIR) / "chunks_with_embeddings.json"
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index() -> tuple[BM25Okapi, list[dict]]:
    """Load chunks from disk and build a BM25Okapi index over their text."""
    chunks = _load_chunks()
    tokenized = [_tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized)
    return index, chunks


def search(query: str, index: BM25Okapi, chunks: list[dict], k: int = 5) -> list[dict]:
    """
    Return the top-k chunks ranked by BM25 score.
    Output format mirrors vectorstore/store.py query() so results can be merged.
    """
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        {
            "text":     chunks[i]["text"],
            "metadata": chunks[i]["metadata"],
            "score":    float(scores[i]),
        }
        for i in top_indices
    ]
