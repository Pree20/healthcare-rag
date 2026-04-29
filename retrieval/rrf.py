def _chunk_key(chunk: dict) -> tuple:
    m = chunk["metadata"]
    return (m["drug_name"], m["section"], m["chunk_index"])


def fuse(bm25_results: list[dict], vector_results: list[dict], k: int = 60) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Each chunk's RRF score = sum of 1/(k + rank) across whichever lists it appears in.
    Chunks that appear in only one list still get a score from that list alone.
    Ranks are 1-based. k=60 is the standard constant from the original RRF paper.

    Returns a single list sorted by descending RRF score.
    Output format: [{text, metadata, score}] where score is the RRF score.
    """
    scores: dict[tuple, float] = {}
    chunks: dict[tuple, dict] = {}

    for rank, chunk in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks[key] = chunk

    for rank, chunk in enumerate(vector_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks[key] = chunk

    ranked = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)

    return [
        {
            "text":     chunks[key]["text"],
            "metadata": chunks[key]["metadata"],
            "score":    scores[key],
        }
        for key in ranked
    ]
