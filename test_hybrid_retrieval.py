import sys
sys.stdout.reconfigure(encoding="utf-8")

from retrieval.hybrid_retriever import search as hybrid_search
from retrieval.reranker import rerank

QUERY = "What are the contraindications for Letrozole?"


if __name__ == "__main__":
    print(f"\nQuery: {QUERY}\n")

    # Fetch 20 hybrid candidates (BM25 + vector → RRF), then rerank to top 5
    candidates = hybrid_search(QUERY, k=20)
    results = rerank(QUERY, candidates, top_n=5)

    print(f"{'='*70}")
    print(f"  Top 5 Results — Hybrid Retrieval + Cross-Encoder Reranking")
    print(f"{'='*70}")

    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        print(f"\nResult {i}")
        print(f"  Drug   : {m['drug_name']}")
        print(f"  Section: {m['section']}")
        print(f"  Score  : {r['score']:.4f}")
        print(f"  Text   : {r['text'][:200]}")

    print()
