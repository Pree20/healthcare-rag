from ingestion.parse_fda   import parse_all
from ingestion.chunk        import chunk_sections
from ingestion.embed        import embed_chunks
from vectorstore.store      import add_chunks
import json
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR

def run_ingestion(limit: int = None, data_dir: Path = None):
    print("Step 1/4 — Parsing FDA XML files...")
    sections = parse_all(limit=limit, data_dir=data_dir)
    print(f"  Got {len(sections)} sections")

    print("Step 2/4 — Chunking sections...")
    chunks = chunk_sections(sections)
    print(f"  Got {len(chunks)} chunks")

    print(f"\nSample chunk:")
    print(f"  Drug: {chunks[0]['metadata']['drug_name']}")
    print(f"  Section: {chunks[0]['metadata']['section']}")
    print(f"  Chunk index: {chunks[0]['metadata']['chunk_index']} of {chunks[0]['metadata']['total_chunks']}")
    print(f"  Text length: {len(chunks[0]['text'])} chars")
    print(f"  Text preview: {chunks[0]['text'][:300]}")
    print(f"\nTotal chunks: {len(chunks)}")

    print("Step 3/4 — Generating embeddings (this takes a few minutes)...")
    chunks = embed_chunks(chunks)

    # Save chunks with embeddings to disk so you can re-run
    # store.py without re-running the slow embedding step
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_PROCESSED_DIR / "chunks_with_embeddings.json"
    with open(cache_path, "w") as f:
        json.dump(chunks, f)
    print(f"  Saved cached embeddings to {cache_path}")

    print("Step 4/4 — Storing in ChromaDB...")
    add_chunks(chunks)
    print("Done. ChromaDB collection is ready.")

if __name__ == "__main__":
    import os
    if os.getenv("CI"):
        from config.settings import CI_DATA_DIR
        print("CI environment detected — using ci_fixtures data")
        run_ingestion(data_dir=CI_DATA_DIR)
    else:
        run_ingestion(limit=5)