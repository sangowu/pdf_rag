"""
Benchmark LLM latency (TTFT + generate time) by sampling questions from gold_answers.csv
and calling the /query API endpoint.

Saves results to results/{prefix}_latency.csv, compatible with plot_metrics.py --timing.

Usage:
    # Start FastAPI first:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

    python scripts/benchmark_latency.py --prefix debug_base --samples 50
    python scripts/benchmark_latency.py --prefix full_semantic --samples 200 --top-k 5
"""

import argparse
import csv
import logging
import random
import time
from pathlib import Path

import requests

from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

config = load_config()
DEFAULT_GOLD_CSV = config.get("paths", {}).get("gold_answers_csv", "data/answers/gold_answers.csv")
DEFAULT_RESULTS_DIR = "results"
DEFAULT_API_URL = "http://localhost:8000"


def load_questions(gold_csv: str, n: int, seed: int = 42) -> list[str]:
    path = Path(gold_csv)
    if not path.exists():
        raise FileNotFoundError(f"Gold answers CSV not found: {path}")
    questions = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            if q:
                questions.append(q)
    if not questions:
        raise ValueError("No questions found in gold answers CSV.")
    random.seed(seed)
    return random.sample(questions, min(n, len(questions)))


def benchmark(
    questions: list[str],
    api_url: str,
    top_k: int,
) -> list[dict]:
    results = []
    for i, q in enumerate(questions, 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{api_url}/query",
                json={"question": q, "top_k": top_k},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            lat = data.get("latency", {})
            results.append({
                "question_index": i,
                "ttft_ms": float(lat.get("ttft_ms", 0)),
                "generate_ms": float(lat.get("generate_ms", 0)),
                "retrieve_ms": float(lat.get("retrieve_ms", 0)),
                "total_ms": float(lat.get("total_ms", 0)),
            })
            if i % 10 == 0:
                logger.info("Progress: %d/%d", i, len(questions))
        except Exception as e:
            logger.warning("Query %d failed: %s", i, e)
    return results


def save_results(results: list[dict], prefix: str, results_dir: str) -> None:
    if not results:
        logger.warning("No results to save.")
        return
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-query details
    details_path = out_dir / f"{prefix}_latency_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_index", "ttft_ms", "generate_ms", "retrieve_ms", "total_ms"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote latency details to %s", details_path)

    # Summary (mean values, compatible with plot_eval_timing)
    nq = len(results)
    summary_path = out_dir / f"{prefix}_latency.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_queries", "ttft_mean_ms", "generate_mean_ms", "retrieve_mean_ms", "total_mean_ms"])
        w.writerow([
            nq,
            round(sum(r["ttft_ms"] for r in results) / nq, 1),
            round(sum(r["generate_ms"] for r in results) / nq, 1),
            round(sum(r["retrieve_ms"] for r in results) / nq, 1),
            round(sum(r["total_ms"] for r in results) / nq, 1),
        ])
    logger.info("Wrote latency summary to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM TTFT and generation latency via /query API.")
    parser.add_argument("--prefix", required=True, help="Output file prefix (e.g. debug_base, full_semantic)")
    parser.add_argument("--samples", type=int, default=50, help="Number of questions to sample (default: 50)")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default: 5)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"FastAPI base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--gold-csv", default=DEFAULT_GOLD_CSV, help="Gold answers CSV path")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    logger.info("Loading %d questions from %s", args.samples, args.gold_csv)
    questions = load_questions(args.gold_csv, args.samples, seed=args.seed)
    logger.info("Benchmarking %d queries against %s ...", len(questions), args.api_url)
    results = benchmark(questions, api_url=args.api_url, top_k=args.top_k)
    save_results(results, prefix=args.prefix, results_dir=args.results_dir)
    if results:
        nq = len(results)
        logger.info(
            "Done. n=%d | TTFT mean=%.0f ms | generate mean=%.0f ms | total mean=%.0f ms",
            nq,
            sum(r["ttft_ms"] for r in results) / nq,
            sum(r["generate_ms"] for r in results) / nq,
            sum(r["total_ms"] for r in results) / nq,
        )


if __name__ == "__main__":
    main()
