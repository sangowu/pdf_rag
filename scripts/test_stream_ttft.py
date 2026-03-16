"""
Compare client-perceived TTFT: streaming (/query/stream) vs non-streaming (/query).

For each sampled question:
  - Non-streaming: time from request sent → full response received (no visible output until done)
  - Streaming:     time from request sent → first token byte received by client

Saves results to results/stream_ttft_comparison.csv and prints a summary.

Usage:
    python scripts/test_stream_ttft.py --samples 30
    python scripts/test_stream_ttft.py --samples 50 --top-k 5 --api-url http://localhost:8000
"""

import argparse
import csv
import json
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
DEFAULT_API_URL  = "http://localhost:8000"
DEFAULT_OUT      = "results/stream_ttft_comparison.csv"
RESULTS_DIR      = "results"


def load_questions(gold_csv: str, n: int, seed: int = 42) -> list[str]:
    rows = []
    with open(gold_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            if q:
                rows.append(q)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def measure_non_stream(api_url: str, question: str, top_k: int) -> dict:
    """Client perceives nothing until the full response arrives."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    t_done = time.perf_counter()
    lat = resp.json().get("latency", {})
    return {
        "client_ttft_ms":     round((t_done - t0) * 1000, 1),  # = total, no earlier signal
        "client_total_ms":    round((t_done - t0) * 1000, 1),
        "server_ttft_ms":     float(lat.get("ttft_ms", 0)),
        "server_generate_ms": float(lat.get("generate_ms", 0)),
        "server_retrieve_ms": float(lat.get("retrieve_ms", 0)),
    }


def measure_stream(api_url: str, question: str, top_k: int) -> dict:
    """Client sees first token as soon as LLM produces it."""
    t0 = time.perf_counter()
    t_first = None
    server_ttft_ms = 0.0
    server_retrieve_ms = 0.0
    server_total_ms = 0.0

    with requests.post(
        f"{api_url}/query/stream",
        json={"question": question, "top_k": top_k},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            try:
                data = json.loads(line[6:])
            except Exception:
                continue
            if data["type"] == "token" and t_first is None:
                t_first = time.perf_counter()
            elif data["type"] == "done":
                server_ttft_ms     = float(data.get("ttft_ms", 0))
                server_retrieve_ms = float(data.get("retrieve_ms", 0))
                server_total_ms    = float(data.get("total_ms", 0))

    t_done = time.perf_counter()
    return {
        "client_ttft_ms":     round((t_first - t0) * 1000, 1) if t_first else 0.0,
        "client_total_ms":    round((t_done  - t0) * 1000, 1),
        "server_ttft_ms":     server_ttft_ms,
        "server_retrieve_ms": server_retrieve_ms,
        "server_total_ms":    server_total_ms,
    }


def run(questions: list[str], api_url: str, top_k: int) -> list[dict]:
    results = []
    for i, q in enumerate(questions, 1):
        try:
            ns = measure_non_stream(api_url, q, top_k)
            st = measure_stream(api_url, q, top_k)
            results.append({
                "idx": i,
                "question": q[:80],
                "ns_client_ttft_ms":     ns["client_ttft_ms"],
                "ns_client_total_ms":    ns["client_total_ms"],
                "ns_server_ttft_ms":     ns["server_ttft_ms"],
                "stream_client_ttft_ms": st["client_ttft_ms"],
                "stream_client_total_ms":st["client_total_ms"],
                "stream_server_ttft_ms": st["server_ttft_ms"],
                "ttft_saving_ms":        round(ns["client_ttft_ms"] - st["client_ttft_ms"], 1),
            })
            if i % 10 == 0:
                logger.info("Progress: %d/%d", i, len(questions))
        except Exception as e:
            logger.warning("Query %d failed: %s", i, e)
    return results


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if not n:
        print("No results.")
        return

    def mean(key): return sum(r[key] for r in results) / n

    print(f"\n{'='*58}")
    print(f"  Client-perceived TTFT comparison  (n={n})")
    print(f"{'='*58}")
    print(f"  {'Metric':<36} {'Non-stream':>9} {'Stream':>9}")
    print(f"  {'-'*54}")
    print(f"  {'Client TTFT (ms)':<36} {mean('ns_client_ttft_ms'):>9.0f} {mean('stream_client_ttft_ms'):>9.0f}")
    print(f"  {'Server TTFT (ms)':<36} {mean('ns_server_ttft_ms'):>9.0f} {mean('stream_server_ttft_ms'):>9.0f}")
    print(f"  {'Client total (ms)':<36} {mean('ns_client_total_ms'):>9.0f} {mean('stream_client_total_ms'):>9.0f}")
    print(f"  {'-'*54}")
    saving = mean("ttft_saving_ms")
    print(f"  Avg perceived TTFT saving (stream): {saving:.0f} ms")
    print(f"{'='*58}\n")


def save_summary_csv(results: list[dict], prefix: str, results_dir: str) -> Path:
    """Save per-query details + one-row summary CSV for plot_metrics.py."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)

    def mean(key): return round(sum(r[key] for r in results) / n, 1)

    # Per-query details
    detail_path = out_dir / f"{prefix}_stream_comparison_details.csv"
    fields = ["idx", "question", "ns_client_ttft_ms", "ns_client_total_ms",
              "ns_server_ttft_ms", "stream_client_ttft_ms", "stream_client_total_ms",
              "stream_server_ttft_ms", "ttft_saving_ms"]
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(results)

    # Summary (one row, read by plot_eval_timing)
    summary_path = out_dir / f"{prefix}_stream_comparison.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_queries", "ns_client_ttft_mean_ms", "ns_client_total_mean_ms",
                    "stream_client_ttft_mean_ms", "stream_client_total_mean_ms",
                    "server_ttft_mean_ms", "ttft_saving_mean_ms"])
        w.writerow([n,
                    mean("ns_client_ttft_ms"), mean("ns_client_total_ms"),
                    mean("stream_client_ttft_ms"), mean("stream_client_total_ms"),
                    mean("stream_server_ttft_ms"), mean("ttft_saving_ms")])
    logger.info("Saved summary to %s", summary_path)
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",   required=True, help="Config prefix, e.g. full_semantic")
    parser.add_argument("--samples",  type=int, default=30)
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--api-url",  default=DEFAULT_API_URL)
    parser.add_argument("--gold-csv", default=DEFAULT_GOLD_CSV)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading %d questions...", args.samples)
    questions = load_questions(args.gold_csv, args.samples, seed=args.seed)

    logger.info("Running %d queries (non-stream then stream each)...", len(questions))
    results = run(questions, api_url=args.api_url, top_k=args.top_k)

    print_summary(results)
    save_summary_csv(results, prefix=args.prefix, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
