# RAG Project Specification

## Overview

A **Retrieval-Augmented Generation (RAG) system for FDA drug labels**. Ingests FDA Structured Product Labeling (SPL) XML files, chunks and embeds text using a medical-specific transformer model, stores embeddings in ChromaDB, and enables semantic similarity search over FDA-approved drug information.

---

## Project Structure

```
e:/RAG project/
├── spec.md                          # This file
├── requirements.txt                 # Python dependencies
├── unzipper.py                      # Extracts XMLs from zip archives
├── config/
│   ├── settings.py                  # Central configuration (paths, model, chunking)
│   └── prompts.yaml                 # Versioned system prompts (citation enforcement)
├── ingestion/
│   ├── parse_fda.py                 # Parses SPL XML → section dicts
│   ├── chunk.py                     # Splits sections into overlapping chunks
│   ├── embed.py                     # Generates MedCPT embeddings
│   └── ingest.py                    # Orchestrates the full 4-step pipeline
├── generation/
│   └── answer.py                    # LLM answer generation (Claude claude-sonnet-4-6, citation extraction)
├── retrieval/
│   ├── hybrid_retriever.py          # Single entry point: BM25 + vector → RRF fusion
│   ├── bm25_retriever.py            # BM25 keyword search over cached chunks
│   ├── vector_retriever.py          # Dense vector search via MedCPT-Query-Encoder + ChromaDB
│   ├── rrf.py                       # Reciprocal Rank Fusion — merges BM25 + vector results
│   └── reranker.py                  # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
├── vectorstore/
│   ├── store.py                     # ChromaDB read/write operations
│   └── chroma_store/                # Persistent ChromaDB SQLite store
└── data/
    ├── raw/
    │   ├── *.zip                    # 20+ FDA drug label zip archives
    │   └── xml/                     # Extracted XML files (19+ files)
    └── processed/
        ├── sections.json            # Parsed section-level text cache
        └── chunks_with_embeddings.json  # Chunk+embedding cache (3.4 MB)
```

---

## Configuration (`config/settings.py`)

| Setting | Value |
|---------|-------|
| `BASE_DIR` | `E:\RAG project` |
| `DATA_RAW_DIR` | `data/raw/xml` |
| `DATA_PROCESSED_DIR` | `data/processed` |
| `CHROMA_DIR` | `vectorstore/chroma_store` |
| `CHUNK_SIZE` | 600 characters |
| `CHUNK_OVERLAP` | 100 characters |
| `EMBED_MODEL` | `ncbi/MedCPT-Article-Encoder` |
| `COLLECTION_NAME` | `fda_drug_labels` |

No `.env` file — all paths hardcoded for Windows.

## Prompt Configuration (`config/prompts.yaml`)

| Field | Value |
|-------|-------|
| `version` | `"1.0"` |
| `system` | System prompt template with `{context}` placeholder |

The system prompt enforces three behaviours:
- **Grounding** — LLM may only use retrieved chunks, no outside knowledge
- **Citations** — every claim must be cited as `[Drug Name — Section]`
- **Declination** — if chunks lack sufficient evidence, respond with a fixed refusal string rather than speculating

---

## Pipeline Architecture

```
FDA Zip Files (raw)
    ↓ [unzipper.py]
XML Files (data/raw/xml/)
    ↓ [ingestion/parse_fda.py]
Sections JSON (drug_name, section, text)
    ↓ [ingestion/chunk.py]
Chunks with metadata (600-char pieces, 100-char overlap)
    ↓ [ingestion/embed.py]  — MedCPT-Article-Encoder (768-dim, CLS pooling)
Chunks + embeddings (cached to disk)
    ↓ [vectorstore/store.py]
ChromaDB Vector Store (cosine similarity, persistent SQLite)
    ↓ [vectorstore/store.py → query()]
Top N similar chunks + metadata (drug_name, section, chunk_index)
```

### Step 1 — Parsing (`ingestion/parse_fda.py`)
- Reads HL7v3 SPL XML (namespace: `urn:hl7-org:v3`)
- Extracts drug name and 11 sections:
  - INDICATIONS & USAGE, CONTRAINDICATIONS, WARNINGS AND PRECAUTIONS
  - DOSAGE & ADMINISTRATION, ADVERSE REACTIONS, DRUG INTERACTIONS
  - BOXED WARNING, USE IN SPECIFIC POPULATIONS, OVERDOSAGE
  - CLINICAL PHARMACOLOGY, MECHANISM OF ACTION
- Skips sections with <50 characters (boilerplate/empty)
- Gracefully handles malformed XML

### Step 2 — Chunking (`ingestion/chunk.py`)
- Uses `RecursiveCharacterTextSplitter` (LangChain)
- Split order: paragraphs → sentences → words → characters
- Preserves metadata: `drug_name`, `section`, `chunk_index`, `total_chunks`

### Step 3 — Embedding (`ingestion/embed.py`)
- Model: `ncbi/MedCPT-Article-Encoder`
- Vector dimension: 768
- Tokenization: max 512 tokens, padding + truncation
- Pooling: CLS token (first hidden state)
- Batch size: 16 (memory-efficient)
- GPU-accelerated (falls back to CPU)

### Step 4 — Storage (`vectorstore/store.py`)
- ChromaDB persistent client (SQLite backend)
- Distance metric: cosine
- Batch insert: 500 chunks/batch
- Metadata fields: `drug_name`, `section`, `chunk_index`, `total_chunks`

---

## Data Schemas

### `data/processed/sections.json`
```json
[
  {
    "drug_name": "Letrozole",
    "section": "INDICATIONS & USAGE SECTION",
    "text": "1 INDICATIONS AND USAGE Letrozole Tablets..."
  }
]
```

### `data/processed/chunks_with_embeddings.json`
```json
[
  {
    "text": "...",
    "metadata": {
      "drug_name": "Letrozole",
      "section": "INDICATIONS & USAGE SECTION",
      "chunk_index": 0,
      "total_chunks": 4
    },
    "id": "Letrozole_INDICATIONS_&_USAGE_SECTION_0",
    "embedding": [-0.267, 0.076, ...]  // 768-dim float array
  }
]
```

### ChromaDB Collection (`fda_drug_labels`)
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk identifier |
| `documents` | string | Text content |
| `embeddings` | float[768] | MedCPT vector |
| `metadatas` | dict | drug_name, section, chunk_index, total_chunks |

---

## Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.2.16 | LLM orchestration / text splitting |
| `langchain-community` | 0.2.16 | Extended integrations |
| `chromadb` | 0.5.5 | Vector database |
| `sentence-transformers` | 3.0.1 | Embedding model loading |
| `transformers` | 4.44.2 | Hugging Face model infrastructure |
| `torch` | 2.4.0 | GPU/CPU tensor computation |
| `requests` | 2.32.3 | HTTP client |
| `lxml` | 5.3.0 | XML parsing |
| `tqdm` | 4.66.5 | Progress bars |
| `pyyaml` | 6.0.2 | YAML config support |
| `rank-bm25` | unpinned | BM25Okapi index for keyword retrieval |
| `anthropic` | unpinned | Anthropic Python SDK for Claude API |

---

## Generation Layer

### Answer Generation (`generation/answer.py`)
- Model: `claude-sonnet-4-6`
- `generate(query, chunks)` → `{answer, citations, prompt_version}` or `{declined, reason, prompt_version}`
- Loads system prompt template from `config/prompts.yaml` at startup (cached at module level)
- Splits system prompt at `Retrieved context:` so static instructions are sent with `cache_control: ephemeral` — eligible for prompt caching across repeated queries
- Context block (per-query) is appended as a second system content block without caching
- Citations are extracted from the response via regex matching `[Drug Name — Section]`
- If response contains the exact declination phrase, returns a structured `declined` dict instead of an answer
- `ANTHROPIC_API_KEY` must be set in the environment

---

## Retrieval Layer

### BM25 (`retrieval/bm25_retriever.py`)
- Loads chunks from `data/processed/chunks_with_embeddings.json`
- Tokenizes with lowercase whitespace splitting (symmetric for index and query)
- Builds a `BM25Okapi` index (`rank-bm25` library)
- `build_index()` → `(BM25Okapi, list[dict])` — call once, reuse the returned pair
- `search(query, index, chunks, k=5)` → `[{text, metadata, score}]`

### Vector (`retrieval/vector_retriever.py`)
- Model: `ncbi/MedCPT-Query-Encoder` (max 64 tokens, CLS pooling)
- Model is cached at module level — loads once per process
- `search(query, k=5)` → encodes query, calls `vectorstore/store.py:query()`, returns `[{text, metadata, score}]`
- Output shape is identical to BM25 retriever for easy merging

### Hybrid Orchestrator (`retrieval/hybrid_retriever.py`)
- **Single entry point for all retrieval** — callers should not use BM25 or vector retrievers directly
- `search(query, k=5, fetch_k=20)` → `[{text, metadata, score}]`
- Fetches `fetch_k` candidates from each retriever, fuses via RRF, trims to `k`
- BM25 index is built once at module import time and reused across calls

### Cross-Encoder Reranker (`retrieval/reranker.py`)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (loaded via `sentence-transformers`)
- Model is cached at module level — loads once per process
- `rerank(query, chunks, top_n=5)` → scores each `(query, text)` pair jointly, returns top_n sorted by descending cross-encoder score
- Intended to run over the top 20 hybrid results; scores are raw logits (not probabilities)

### Reciprocal Rank Fusion (`retrieval/rrf.py`)
- `fuse(bm25_results, vector_results, k=60)` → merged `[{text, metadata, score}]`
- RRF formula: each chunk's score = Σ 1/(k + rank) summed across both lists
- Deduplicates by `(drug_name, section, chunk_index)` from metadata
- Chunks appearing in only one list still receive a partial RRF score
- k=60 is the standard constant from the original RRF paper (Cormack et al. 2009)

---

## Entry Points

| File | Command / Call | Purpose |
|------|----------------|---------|
| `unzipper.py` | `python unzipper.py` | Extract XMLs from zips |
| `ingestion/ingest.py` | `run_ingestion(limit=5)` | Run full pipeline (limit = # files) |
| `vectorstore/store.py` | `query(query_embedding, n_results=5)` | Low-level vector search |
| `test_hybrid_retrieval.py` | `python test_hybrid_retrieval.py` | **End-to-end pipeline validation** |
| `generation/answer.py` | `generate(query, chunks)` | LLM answer generation with citations |
| `retrieval/hybrid_retriever.py` | `search(query, k=5)` | **Primary retrieval entry point** |
| `retrieval/bm25_retriever.py` | `index, chunks = build_index(); search(query, index, chunks, k=5)` | BM25 keyword search (internal) |
| `retrieval/vector_retriever.py` | `search(query, k=5)` | Dense semantic search (internal) |
| `retrieval/rrf.py` | `fuse(bm25_results, vector_results, k=60)` | Merge BM25 + vector results via RRF (internal) |
| `retrieval/reranker.py` | `rerank(query, chunks, top_n=5)` | Cross-encoder reranking over hybrid candidates |
| `eval/run_eval.py` | `python eval/run_eval.py` | **RAGAS faithfulness evaluation** — exits 0 if avg ≥ threshold |

---

## Current Dataset

- **20+ FDA drug label zip files** in `data/raw/`
- **19+ extracted XMLs** in `data/raw/xml/`
- **Processed data** currently covers Letrozole (breast cancer treatment) with ~30 sections
- ChromaDB store: `vectorstore/chroma_store/chroma.sqlite3` (~3.6 MB)

---

## Technical Specs Summary

| Aspect | Value |
|--------|-------|
| Embedding model | `ncbi/MedCPT-Article-Encoder` (medical domain) |
| Vector dimension | 768 |
| Distance metric | Cosine similarity |
| Chunk size | 600 characters |
| Chunk overlap | 100 characters |
| Max token length | 512 (truncated) |
| Embed batch size | 16 |
| Store batch size | 500 |
| Vector DB | ChromaDB (SQLite) |
| Framework | LangChain 0.2.16 |
| GPU support | Yes (CUDA, fallback CPU) |
| Data source | FDA SPL XML labels |
| Sections indexed | 11 FDA label sections |

---

## Evaluation Layer

### RAGAS Faithfulness Eval (`eval/run_eval.py`)
- Golden dataset: `eval/golden_dataset.json` — 55 questions across Letrozole, Acyclovir, Hydrocortisone, Liraglutide, Pristiq
- Each question runs the full pipeline: hybrid retrieval → reranking → generation
- Faithfulness metric (RAGAS 0.4+): verifies every claim in the LLM answer is inferrable from the retrieved contexts
- LLM for RAGAS scoring: `llama-3.3-70b-versatile` via Groq (`langchain-groq` wrapper)
- Declined answers are excluded from faithfulness scoring (recorded as `"faithfulness": null`)
- Threshold: `0.5` — exit code 0 if avg faithfulness ≥ threshold, exit code 1 otherwise
- Output: `eval/results.json` — per-question scores + aggregate summary

### `eval/results.json` Schema
```json
{
  "average_faithfulness": 0.85,
  "threshold": 0.5,
  "passed": true,
  "total_questions": 55,
  "answered_questions": 13,
  "declined_questions": 42,
  "results": [
    {
      "id": "letrozole_001",
      "question": "...",
      "expected_answer": "...",
      "drug_name": "Letrozole",
      "expected_section": "CONTRAINDICATIONS SECTION",
      "contexts": ["..."],
      "answer": "...",
      "citations": [{"drug_name": "...", "section": "..."}],
      "declined": false,
      "faithfulness": 0.92
    }
  ]
}
```

---

## Notes & Known Gaps

- Full pipeline complete: ingestion → retrieval → generation → RAGAS evaluation
- Paths are hardcoded Windows paths — not portable across OSes without modifying `settings.py`
- Default `ingest.py` limits to 5 files for testing; increase `limit` for full dataset
- Dataset currently only covers Letrozole — 42 of 55 golden questions will be declined until remaining drugs are ingested
- **Query vs Article encoder**: Articles are embedded with `ncbi/MedCPT-Article-Encoder`; queries must use `ncbi/MedCPT-Query-Encoder` (max 64 tokens). Using mismatched encoders degrades retrieval quality.
- **Windows console encoding**: FDA labels contain Unicode characters (e.g., `●` U+25CF) that crash the cp1252 terminal. Run scripts with `PYTHONIOENCODING=utf-8` or set it in the environment.
