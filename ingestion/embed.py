import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config.settings import EMBED_MODEL


def load_model():
    """
    Load MedCPT article encoder once and return (tokenizer, model).
    Moves model to GPU if available, otherwise stays on CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # disables dropout — important for deterministic embeddings

    print(f"Model loaded on {device}")
    return tokenizer, model


def embed_chunks(chunks: list[dict], batch_size: int = 16) -> list[dict]:
    """
    Adds an 'embedding' key to each chunk dict.
    Processes in batches to avoid loading 500 chunks into memory at once.
    """
    tokenizer, model = load_model()
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Tokenize: truncate to 512 tokens (MedCPT's max), pad shorter texts
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",  # returns PyTorch tensors, not lists
        )

        # Move tensors to same device as the model
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # torch.no_grad() tells PyTorch not to track gradients
        # We're doing inference, not training — this saves memory
        with torch.no_grad():
            output = model(**encoded)

        # MedCPT uses mean pooling over token embeddings
        # output.last_hidden_state shape: (batch_size, seq_len, 768)
        # We average across seq_len dimension to get one vector per text
        embeddings = output.last_hidden_state[:, 0, :]

        # Move back to CPU and convert to plain Python lists
        # ChromaDB expects lists, not tensors
        embeddings = embeddings.cpu().numpy().tolist()

        # Attach embedding back to each chunk in the batch
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

    return chunks