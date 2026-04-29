from retrieval.bm25_retriever import build_index, search as bm25_search
from retrieval.vector_retriever import search as vector_search
from retrieval.rrf import fuse

# Build the BM25 index once at import time — reading the JSON and constructing
# the index is cheap enough to do upfront rather than on each query.
_bm25_index, _bm25_chunks = build_index()


def search(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    drug_name: str | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 + dense vector search fused via Reciprocal Rank Fusion.

    fetch_k controls how many candidates each retriever returns before fusion.
    drug_name: when provided, restricts both retrievers to that drug only —
    BM25 results are post-filtered and ChromaDB uses a metadata where-filter.
    """
    bm25_results = bm25_search(query, _bm25_index, _bm25_chunks, k=fetch_k)
    if drug_name:
        bm25_results = [
            r for r in bm25_results
            if r["metadata"]["drug_name"] == drug_name
        ]

    vector_results = vector_search(query, k=fetch_k, drug_name=drug_name)

    fused = fuse(bm25_results, vector_results)
    return fused[:k]
