"""eval/score_faithfulness.py — Score faithfulness on already-generated answers.

Reads eval/results.json (which contains answers + contexts from a prior pipeline run),
scores each answered question with RAGAS Faithfulness one at a time, then writes
updated scores back to results.json.

No generation API calls are made — only the RAGAS scoring LLM (llama-3.1-8b-instant).

Exit codes:
  0 — average faithfulness >= FAITHFULNESS_THRESHOLD
  1 — average faithfulness <  FAITHFULNESS_THRESHOLD
"""

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from config.settings import BASE_DIR

load_dotenv(BASE_DIR / ".env")

RESULTS_FILE = Path(__file__).parent / "results.json"
FAITHFULNESS_THRESHOLD = 0.5
INTER_REQUEST_DELAY = 2.0  # seconds between RAGAS calls


def _build_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ["GROQ_API_KEY"],
        )
    )


def _score_one(metric, q: str, a: str, ctx: list[str]) -> float | None:
    from datasets import Dataset as HFDataset
    from ragas import RunConfig, evaluate

    ds = HFDataset.from_dict({"question": [q], "answer": [a], "contexts": [ctx]})
    result = evaluate(
        ds,
        metrics=[metric],
        raise_exceptions=False,
        run_config=RunConfig(timeout=120, max_retries=5, max_wait=120),
    )
    score = result["faithfulness"][0]
    return None if (score is None or math.isnan(score)) else float(score)


def main() -> None:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    records = data["results"]
    to_score = [r for r in records if not r.get("declined") and r.get("answer")]
    print(f"Scoring {len(to_score)} answered questions (skipping {len(records) - len(to_score)} declined)…\n")

    from ragas.metrics import Faithfulness
    ragas_llm = _build_ragas_llm()
    metric = Faithfulness(llm=ragas_llm)

    for i, record in enumerate(to_score, 1):
        qid = record["id"]
        print(f"[{i}/{len(to_score)}] {qid}…", end=" ", flush=True)
        score = _score_one(metric, record["question"], record["answer"], record["contexts"])
        record["faithfulness"] = score
        print(score)
        # Persist after every question so a crash doesn't lose progress
        _write_results(data)
        time.sleep(INTER_REQUEST_DELAY)

    _write_results(data)
    _print_summary(data)


def _write_results(data: dict) -> None:
    records = data["results"]
    scored = [r["faithfulness"] for r in records if r.get("faithfulness") is not None]
    avg = sum(scored) / len(scored) if scored else float("nan")
    data["average_faithfulness"] = round(avg, 4) if not math.isnan(avg) else float("nan")
    data["passed"] = (not math.isnan(avg)) and avg >= FAITHFULNESS_THRESHOLD

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_summary(data: dict) -> None:
    records = data["results"]
    scored = [r["faithfulness"] for r in records if r.get("faithfulness") is not None]
    avg = data["average_faithfulness"]
    passed = data["passed"]

    print(f"\n{'=' * 60}")
    print(f"Average faithfulness : {avg}")
    print(f"Threshold            : {FAITHFULNESS_THRESHOLD}")
    print(f"Verdict              : {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"Total questions      : {len(records)}")
    print(f"Answered             : {data['answered_questions']}")
    print(f"Declined             : {data['declined_questions']}")
    print(f"Faithfulness scored  : {len(scored)}")
    print(f"Results written to   : {RESULTS_FILE}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
