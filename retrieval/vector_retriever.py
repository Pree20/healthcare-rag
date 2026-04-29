import torch
from transformers import AutoTokenizer, AutoModel
from vectorstore.store import query as chroma_query

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    _model = AutoModel.from_pretrained(QUERY_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    _model.eval()

    print(f"Query encoder loaded on {device}")
    return _tokenizer, _model


def _encode_query(query: str) -> list[float]:
    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    # Query encoder is capped at 64 tokens (MedCPT-Query-Encoder constraint)
    encoded = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    # CLS token pooling — same strategy as the article encoder
    embedding = output.last_hidden_state[:, 0, :]
    return embedding.cpu().numpy().tolist()[0]


def search(query: str, k: int = 5, drug_name: str | None = None) -> list[dict]:
    """
    Encode query with MedCPT-Query-Encoder and return top-k chunks from ChromaDB.
    Output format: [{text, metadata, score}] — same shape as bm25_retriever.search().

    drug_name: when provided, restricts results to chunks from that drug only.
    """
    query_embedding = _encode_query(query)
    where = {"drug_name": drug_name} if drug_name else None
    return chroma_query(query_embedding, n_results=k, where=where)
