import os
import re
import yaml
from dotenv import load_dotenv
from groq import Groq
from config.settings import BASE_DIR

load_dotenv(BASE_DIR / ".env")

PROMPTS_PATH = BASE_DIR / "config" / "prompts.yaml"
MODEL = "llama-3.3-70b-versatile"
DECLINE_PHRASE = "I cannot answer this question from the available drug label information."

_client = None
_prompt_config = None
_known_drugs: list[tuple[str, str]] | None = None  # [(lower, original)]


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set. Add it to .env or your environment.")
        _client = Groq(api_key=api_key)
    return _client


def _load_prompts() -> dict:
    global _prompt_config
    if _prompt_config is None:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            _prompt_config = yaml.safe_load(f)
    return _prompt_config


def _format_context(chunks: list[dict]) -> str:
    """Render chunks into numbered, labelled blocks for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk["metadata"]
        parts.append(
            f"[{i}] Drug: {m['drug_name']} | Section: {m['section']}\n"
            f"{chunk['text'].strip()}"
        )
    return "\n\n".join(parts)


def _extract_citations(text: str) -> list[dict]:
    """Pull [Drug Name — Section] markers out of the answer text."""
    pattern = r"\[([^\]\-—]+?)\s*[—\-]\s*([^\]]+?)\]"
    seen = set()
    citations = []
    for match in re.finditer(pattern, text):
        drug = match.group(1).strip()
        section = match.group(2).strip()
        key = (drug, section)
        if key not in seen:
            seen.add(key)
            citations.append({"drug_name": drug, "section": section})
    return citations


def _build_system_prompt(template: str, formatted_context: str) -> str:
    """Fill the {context} placeholder in the system template."""
    return template.replace("{context}", formatted_context)


def _get_known_drugs() -> list[tuple[str, str]]:
    """Load unique drug names from the ChromaDB collection (cached after first call)."""
    global _known_drugs
    if _known_drugs is None:
        from vectorstore.store import get_collection
        metadatas = get_collection().get(include=["metadatas"])["metadatas"]
        seen: dict[str, str] = {}
        for m in metadatas:
            name = m["drug_name"]
            seen[name.lower()] = name
        _known_drugs = list(seen.items())
    return _known_drugs


def _extract_drug_name(query: str) -> str | None:
    """Return the drug name found in query, or None if none match the index."""
    q_lower = query.lower()
    for lower_name, original_name in _get_known_drugs():
        if lower_name in q_lower:
            return original_name
    return None


def generate(query: str, chunks: list[dict] | None = None) -> dict:
    """
    Generate an answer from retrieved chunks using Groq.

    If chunks is None, retrieval is performed internally: the query is scanned
    for a known drug name, and only chunks for that drug are fetched (structural
    grounding guarantee). Pass chunks explicitly to use pre-computed results.

    Returns on success:
      {"answer": str, "citations": list[dict], "prompt_version": str}

    Returns on insufficient evidence:
      {"declined": True, "reason": str, "prompt_version": str}
    """
    if chunks is None:
        from retrieval.hybrid_retriever import search as hybrid_search
        from retrieval.reranker import rerank
        drug_name = _extract_drug_name(query)
        candidates = hybrid_search(query, k=20, drug_name=drug_name)
        chunks = rerank(query, candidates, top_n=5)

    config = _load_prompts()
    prompt_version = config["version"]
    system_prompt = _build_system_prompt(
        config["system"], _format_context(chunks)
    )

    print(f"[answer] Passing {len(chunks)} chunks to LLM:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["text"].replace("\n", " ")[:100]
        print(f"  [{i}] {chunk['metadata']['drug_name']} | {chunk['metadata']['section']} | {preview}")

    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    answer_text = response.choices[0].message.content or ""

    if DECLINE_PHRASE in answer_text:
        return {
            "declined": True,
            "reason": DECLINE_PHRASE,
            "prompt_version": prompt_version,
        }

    return {
        "answer": answer_text,
        "citations": _extract_citations(answer_text),
        "prompt_version": prompt_version,
    }
