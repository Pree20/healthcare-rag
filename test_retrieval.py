import torch
from transformers import AutoTokenizer, AutoModel
from vectorstore.store import query

# Load the QUERY encoder (not the article encoder)
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
model.eval()

# A test question
test_query = "What are the contraindications for Letrozole?"

# Encode the query using CLS token pooling
with torch.no_grad():
    encoded = tokenizer(
        test_query,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    query_embedding = model(**encoded).last_hidden_state[:, 0, :].squeeze().tolist()

# Retrieve top 3 chunks
results = query(query_embedding, n_results=3)

print(f"Query: {test_query}\n")
for i, chunk in enumerate(results):
    print(f"Result {i+1}")
    print(f"  Drug: {chunk['metadata']['drug_name']}")
    print(f"  Section: {chunk['metadata']['section']}")
    print(f"  Score: {chunk['score']:.4f}")
    print(f"  Text: {chunk['text'][:200]}")
    print()
