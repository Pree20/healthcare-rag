"""eval/run_eval.py — RAGAS faithfulness evaluation over the golden dataset.

Loads eval/golden_dataset.json, runs each question through the full pipeline
(hybrid retrieval → reranking → generation), scores faithfulness via RAGAS,
writes per-question results and averages to eval/results.json.

Exit codes:
  0 — average faithfulness >= FAITHFULNESS_THRESHOLD
  1 — average faithfulness <  FAITHFULNESS_THRESHOLD
"""

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from config.settings import BASE_DIR

load_dotenv(BASE_DIR / ".env")

from retrieval.hybrid_retriever import search as hybrid_search
from retrieval.reranker import rerank
from generation.answer import generate

GOLDEN_DATASET = Path(__file__).parent / "golden_dataset.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
CHECKPOINT_FILE = Path(__file__).parent / "checkpoint.json"
FAITHFULNESS_THRESHOLD = 0.8
# Seconds to wait between generation calls to stay under Groq rate limits
REQUEST_DELAY = 1.5


def _build_ragas_llm():
    # Use a fast small model for RAGAS evaluation — keeps token spend separate
    # from the generation model and benefits from higher per-model rate limits.
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ["GROQ_API_KEY"],
        )
    )


def _run_pipeline(question: str) -> tuple[dict, list[dict]]:
    candidates = hybrid_search(question, k=20)
    chunks = rerank(question, candidates, top_n=5)
    result = generate(question, chunks)
    return result, chunks


def _parse_retry_seconds(error_message: str) -> int:
    """Extract the suggested wait time from a Groq 429 error message."""
    m = re.search(r"try again in (\d+)m([\d.]+)s", str(error_message))
    if m:
        return int(m.group(1)) * 60 + int(float(m.group(2))) + 5
    m = re.search(r"try again in ([\d.]+)s", str(error_message))
    if m:
        return int(float(m.group(1))) + 5
    return 90  # fallback


def _run_pipeline_with_retry(question: str, max_retries: int = 5) -> tuple[dict, list[dict]]:
    """Run pipeline with Groq-suggested wait on 429 rate-limit errors."""
    from groq import RateLimitError

    for attempt in range(max_retries):
        try:
            return _run_pipeline(question)
        except RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            wait = _parse_retry_seconds(str(exc))
            print(f"  [rate limit] waiting {wait}s before retry {attempt + 2}/{max_retries}…")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_checkpoint() -> dict[str, dict]:
    """Return {qid: record} for questions already processed in a prior run."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def _save_checkpoint(records: list[dict]) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def main() -> None:
    with open(GOLDEN_DATASET, encoding="utf-8") as f:
        golden = json.load(f)

    checkpoint = _load_checkpoint()
    records: list[dict] = []
    eval_questions: list[str] = []
    eval_answers: list[str] = []
    eval_contexts: list[list[str]] = []

    total = len(golden)
    skipped = sum(1 for item in golden if item["id"] in checkpoint)
    print(f"Evaluating {total} questions ({skipped} restored from checkpoint)…\n")

    for idx, item in enumerate(golden, 1):
        qid = item["id"]
        question = item["question"]

        if qid in checkpoint:
            record = checkpoint[qid]
            records.append(record)
            if not record.get("declined") and record.get("answer"):
                eval_questions.append(record["question"])
                eval_answers.append(record["answer"])
                eval_contexts.append(record["contexts"])
            print(f"[{idx}/{total}] {qid}: (restored)")
            continue

        print(f"[{idx}/{total}] {qid}: {question[:70]}…")

        try:
            result, chunks = _run_pipeline_with_retry(question)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            print("  Saving checkpoint and exiting — re-run to resume.")
            _save_checkpoint(records)
            sys.exit(2)

        contexts = [c["text"] for c in chunks]
        record = {
            "id": qid,
            "question": question,
            "expected_answer": item["expected_answer"],
            "drug_name": item["drug_name"],
            "expected_section": item["expected_section"],
            "contexts": contexts,
            "faithfulness": None,
        }

        if "declined" in result:
            record["declined"] = True
            record["answer"] = None
            print(f"  → declined")
        else:
            record["declined"] = False
            record["answer"] = result["answer"]
            record["citations"] = result.get("citations", [])
            eval_questions.append(question)
            eval_answers.append(result["answer"])
            eval_contexts.append(contexts)
            print(f"  → answered ({len(contexts)} contexts)")

        records.append(record)
        _save_checkpoint(records)
        time.sleep(REQUEST_DELAY)

    # RAGAS faithfulness scoring — one question at a time to stay within rate limits
    if eval_questions:
        print(f"\nScoring faithfulness on {len(eval_questions)} answered questions…")

        from datasets import Dataset as HFDataset
        from ragas import evaluate, RunConfig
        from ragas.metrics import Faithfulness

        ragas_llm = _build_ragas_llm()
        metric = Faithfulness(llm=ragas_llm)
        run_cfg = RunConfig(timeout=120, max_retries=5, max_wait=120)

        faith_scores: list[float | None] = []
        for i, (q, a, ctx) in enumerate(zip(eval_questions, eval_answers, eval_contexts), 1):
            print(f"  [{i}/{len(eval_questions)}] scoring…", end=" ", flush=True)
            try:
                ds = HFDataset.from_dict({"question": [q], "answer": [a], "contexts": [ctx]})
                result = evaluate(ds, metrics=[metric], raise_exceptions=False, run_config=run_cfg)
                score = result["faithfulness"][0]
                faith_scores.append(float(score) if score == score else None)  # NaN → None
                print(f"{faith_scores[-1]}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                faith_scores.append(None)
            time.sleep(REQUEST_DELAY)

        score_idx = 0
        for record in records:
            if not record.get("declined") and record.get("answer"):
                record["faithfulness"] = faith_scores[score_idx]
                score_idx += 1

    # Clean up checkpoint on successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Aggregate
    scored = [r["faithfulness"] for r in records if r["faithfulness"] is not None]
    avg_faithfulness = sum(scored) / len(scored) if scored else 0.0
    passed = avg_faithfulness >= FAITHFULNESS_THRESHOLD

    output = {
        "average_faithfulness": round(avg_faithfulness, 4),
        "threshold": FAITHFULNESS_THRESHOLD,
        "passed": passed,
        "total_questions": len(records),
        "answered_questions": len(scored),
        "declined_questions": sum(1 for r in records if r.get("declined")),
        "results": records,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Average faithfulness : {avg_faithfulness:.4f}")
    print(f"Threshold            : {FAITHFULNESS_THRESHOLD}")
    print(f"Verdict              : {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"Total questions      : {len(records)}")
    print(f"Answered             : {len(scored)}")
    print(f"Declined             : {output['declined_questions']}")
    print(f"Results written to   : {RESULTS_FILE}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
