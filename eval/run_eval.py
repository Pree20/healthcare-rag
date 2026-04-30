"""eval/run_eval.py — RAG evaluation entrypoint.

Two modes, selected by the CI_EVAL_MODE environment variable:

  CI_EVAL_MODE=retrieval_only  (default in CI)
      BM25-only retrieval precision check — no model downloads, no API calls.
      For each question in the dataset, checks whether the correct drug+section
      appears in the top-5 BM25 results.  Passes if precision >= 0.80.
      Runs in < 2 minutes.

  CI_EVAL_MODE unset  (full faithfulness eval, run locally before releases)
      Full pipeline: hybrid retrieval → reranking → Groq generation → RAGAS
      faithfulness scoring.  Writes per-question results to eval/results.json.
      Passes if average faithfulness >= 0.80.

Dataset path is read from the EVAL_DATASET environment variable; defaults to
eval/golden_dataset.json when unset.

Exit codes:
  0 — quality gate passed
  1 — quality gate failed
  2 — pipeline error (full mode only)
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from config.settings import BASE_DIR

load_dotenv(BASE_DIR / ".env")

_dataset_env = os.environ.get("EVAL_DATASET")
GOLDEN_DATASET = Path(_dataset_env) if _dataset_env else Path(__file__).parent / "golden_dataset.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
CHECKPOINT_FILE = Path(__file__).parent / "checkpoint.json"

FAITHFULNESS_THRESHOLD = 0.8
PRECISION_THRESHOLD = 0.8
PRECISION_TOP_K = 5


# ---------------------------------------------------------------------------
# CI mode: BM25 retrieval precision — no models, no API calls
# ---------------------------------------------------------------------------

def retrieval_only_main() -> None:
    from retrieval.bm25_retriever import build_index, search as bm25_search

    with open(GOLDEN_DATASET, encoding="utf-8") as f:
        golden = json.load(f)

    total = len(golden)
    print(f"Retrieval precision eval — {total} questions, top-{PRECISION_TOP_K} BM25\n")

    bm25_index, bm25_chunks = build_index()

    hits = 0
    for idx, item in enumerate(golden, 1):
        qid = item["id"]
        expected_drug = item["drug_name"]
        expected_section = item["expected_section"]

        results = bm25_search(item["question"], bm25_index, bm25_chunks, k=PRECISION_TOP_K)

        hit = any(
            r["metadata"]["drug_name"] == expected_drug
            and r["metadata"]["section"] == expected_section
            for r in results
        )
        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"[{idx:>2}/{total}] {status} {qid}")
        if not hit:
            found = [
                f"{r['metadata']['drug_name']} | {r['metadata']['section']}"
                for r in results
            ]
            print(f"       expected : {expected_drug} | {expected_section}")
            print(f"       top-{PRECISION_TOP_K} BM25: {found}")

    precision = hits / total if total else 0.0
    passed = precision >= PRECISION_THRESHOLD

    print(f"\n{'=' * 60}")
    print(f"Retrieval precision  : {precision:.4f}  ({hits}/{total})")
    print(f"Threshold            : {PRECISION_THRESHOLD}")
    print(f"Verdict              : {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


# ---------------------------------------------------------------------------
# Full mode: hybrid retrieval → reranking → generation → RAGAS faithfulness
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ["GROQ_API_KEY"],
        )
    )


def _parse_retry_seconds(error_message: str) -> int:
    import re
    m = re.search(r"try again in (\d+)m([\d.]+)s", str(error_message))
    if m:
        return int(m.group(1)) * 60 + int(float(m.group(2))) + 5
    m = re.search(r"try again in ([\d.]+)s", str(error_message))
    if m:
        return int(float(m.group(1))) + 5
    return 90


def _run_pipeline_with_retry(question: str, hybrid_search, rerank, generate, max_retries: int = 5):
    import time
    from groq import RateLimitError

    for attempt in range(max_retries):
        try:
            candidates = hybrid_search(question, k=20)
            chunks = rerank(question, candidates, top_n=5)
            result = generate(question, chunks)
            return result, chunks
        except RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            wait = _parse_retry_seconds(str(exc))
            print(f"  [rate limit] waiting {wait}s before retry {attempt + 2}/{max_retries}…")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def _save_checkpoint(records: list) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def faithfulness_main() -> None:
    import time
    from retrieval.hybrid_retriever import search as hybrid_search
    from retrieval.reranker import rerank
    from generation.answer import generate

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

    REQUEST_DELAY = 1.5

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
            result, chunks = _run_pipeline_with_retry(question, hybrid_search, rerank, generate)
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

    if eval_questions:
        print(f"\nScoring faithfulness on {len(eval_questions)} answered questions…")

        from datasets import Dataset as HFDataset
        from ragas import evaluate, RunConfig
        from ragas.metrics import Faithfulness

        ragas_llm = _build_ragas_llm()
        metric = Faithfulness(llm=ragas_llm)
        run_cfg = RunConfig(timeout=120, max_retries=5, max_wait=120)

        faith_scores: list = []
        for i, (q, a, ctx) in enumerate(zip(eval_questions, eval_answers, eval_contexts), 1):
            print(f"  [{i}/{len(eval_questions)}] scoring…", end=" ", flush=True)
            try:
                ds = HFDataset.from_dict({"question": [q], "answer": [a], "contexts": [ctx]})
                result = evaluate(ds, metrics=[metric], raise_exceptions=False, run_config=run_cfg)
                score = result["faithfulness"][0]
                faith_scores.append(float(score) if score == score else None)
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

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

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
    print(f"Verdict              : {'PASSED' if passed else 'FAILED'}")
    print(f"Total questions      : {len(records)}")
    print(f"Answered             : {len(scored)}")
    print(f"Declined             : {output['declined_questions']}")
    print(f"Results written to   : {RESULTS_FILE}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    if os.environ.get("CI_EVAL_MODE") == "retrieval_only":
        retrieval_only_main()
    else:
        faithfulness_main()
