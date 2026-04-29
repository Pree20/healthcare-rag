import chromadb
from config.settings import CHROMA_DIR, COLLECTION_NAME


def get_collection():
    """
    Creates or opens an existing ChromaDB collection.
    PersistentClient saves to disk so your embeddings survive restarts.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # get_or_create_collection is safe to call multiple times —
    # it returns the existing collection if it already exists
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine distance is better than L2 for semantic similarity
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks(chunks: list[dict]):
    """
    Loads all chunks into ChromaDB.
    ChromaDB wants three parallel lists: ids, embeddings, documents (text),
    and an optional metadatas list.
    """
    collection = get_collection()

    ids         = [c["id"]        for c in chunks]
    embeddings  = [c["embedding"] for c in chunks]
    documents   = [c["text"]      for c in chunks]
    metadatas   = [c["metadata"]  for c in chunks]

    # add() in batches of 500 — ChromaDB has an internal limit
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids        = ids[i : i + batch_size],
            embeddings = embeddings[i : i + batch_size],
            documents  = documents[i : i + batch_size],
            metadatas  = metadatas[i : i + batch_size],
        )
        print(f"Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")


def query(
    query_embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Given a query embedding vector, return the top n_results chunks.

    where: optional ChromaDB metadata filter, e.g. {"drug_name": "Letrozole"}.
    When provided, only chunks whose metadata matches are returned.
    """
    collection = get_collection()

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # ChromaDB returns nested lists (one per query) — unwrap the first
    chunks_out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks_out.append({
            "text":     doc,
            "metadata": meta,
            "score":    1 - dist,  # cosine distance → cosine similarity
        })

    return chunks_out