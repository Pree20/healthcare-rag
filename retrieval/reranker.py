from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _load_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
        print(f"Cross-encoder loaded: {RERANK_MODEL}")
    return _model


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Score each (query, chunk text) pair with a cross-encoder and return
    the top_n chunks sorted by descending cross-encoder score.

    Cross-encoders see the query and document together, giving much more
    accurate relevance scores than bi-encoder cosine similarity alone.
    """
    model = _load_model()

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    return [
        {
            "text":     chunk["text"],
            "metadata": chunk["metadata"],
            "score":    float(score),
        }
        for score, chunk in ranked[:top_n]
    ]
