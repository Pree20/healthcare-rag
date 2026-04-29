from pathlib import Path

# All paths relative to the project root
BASE_DIR           = Path(__file__).parent.parent
DATA_RAW_DIR       = BASE_DIR / "data" / "raw" / "xml"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR         = BASE_DIR / "vectorstore" / "chroma_store"
CI_DATA_DIR        = BASE_DIR / "data" / "ci_fixtures"

# Chunking settings
CHUNK_SIZE    = 600   # tokens per chunk
CHUNK_OVERLAP = 100   # tokens of overlap between chunks

# Embedding model — document side (NOT the query encoder)
EMBED_MODEL = "ncbi/MedCPT-Article-Encoder"

# ChromaDB collection name
COLLECTION_NAME = "fda_drug_labels"