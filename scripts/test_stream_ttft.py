"""
Compare client-perceived latency: streaming (/query/stream) vs non-streaming (/query).

Measurements:
  - Non-streaming: time from request sent → all characters visible (full round-trip)
  - Streaming:     time from request sent → first token visible (TTFT)
                   inter-token latency (ITL) for each subsequent token

Saves:
  results/{prefix}_stream_comparison.csv      — one-row summary (read by plot_metrics.py)
  results/{prefix}_stream_comparison_details.csv — per-query rows
  results/{prefix}_itl_details.csv            — per-token ITL rows (for distribution plot)

Usage:
    python scripts/test_stream_ttft.py --prefix full_semantic --samples 30
    python scripts/test_stream_ttft.py --prefix full_semantic --samples 50 --top-k 5
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
RESULTS_DIR      = "results"


def _pct(data: list, p: float) -> float:
    """Compute p-th percentile of data (linear interpolation)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


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
    """Non-streaming: client sees nothing until the full response arrives."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    t_done = time.perf_counter()
    lat = resp.json().get("latency", {})
    total_ms = round((t_done - t0) * 1000, 1)
    return {
        "client_total_ms":    total_ms,   # time until ALL chars visible
        "server_ttft_ms":     float(lat.get("ttft_ms", 0)),
        "server_generate_ms": float(lat.get("generate_ms", 0)),
        "server_retrieve_ms": float(lat.get("retrieve_ms", 0)),
    }


def measure_stream(api_url: str, question: str, top_k: int) -> dict:
    """Streaming: record timestamp for every token to compute TTFT and ITL."""
    t0 = time.perf_counter()
    token_times_ms: list[float] = []   # time from t0 when each token arrived (ms)
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
            if data["type"] == "token":
                token_times_ms.append((time.perf_counter() - t0) * 1000)
            elif data["type"] == "done":
                server_ttft_ms     = float(data.get("ttft_ms", 0))
                server_retrieve_ms = float(data.get("retrieve_ms", 0))
                server_total_ms    = float(data.get("total_ms", 0))

    t_done = time.perf_counter()
    client_ttft_ms  = round(token_times_ms[0], 1) if token_times_ms else 0.0
    client_total_ms = round((t_done - t0) * 1000, 1)

    # inter-token latencies (gap between consecutive tokens)
    itl_values = [
        round(token_times_ms[i] - token_times_ms[i - 1], 2)
        for i in range(1, len(token_times_ms))
    ]

    return {
        "client_ttft_ms":     client_ttft_ms,
        "client_total_ms":    client_total_ms,
        "server_ttft_ms":     server_ttft_ms,
        "server_retrieve_ms": server_retrieve_ms,
        "server_total_ms":    server_total_ms,
        "token_count":        len(token_times_ms),
        "itl_values":         itl_values,
        "itl_mean_ms":        round(sum(itl_values) / len(itl_values), 2) if itl_values else 0.0,
        "itl_p50_ms":         round(_pct(itl_values, 50), 2),
        "itl_p95_ms":         round(_pct(itl_values, 95), 2),
    }


def run(questions: list[str], api_url: str, top_k: int) -> tuple[list[dict], list[dict]]:
    """Returns (per_query_results, itl_detail_rows)."""
    results = []
    itl_rows = []
    for i, q in enumerate(questions, 1):
        try:
            ns = measure_non_stream(api_url, q, top_k)
            st = measure_stream(api_url, q, top_k)
            results.append({
                "idx":                   i,
                "question":              q[:80],
                "ns_client_total_ms":    ns["client_total_ms"],
                "ns_server_ttft_ms":     ns["server_ttft_ms"],
                "stream_client_ttft_ms": st["client_ttft_ms"],
                "stream_client_total_ms":st["client_total_ms"],
                "stream_server_ttft_ms": st["server_ttft_ms"],
                "ttft_saving_ms":        round(ns["client_total_ms"] - st["client_ttft_ms"], 1),
                "stream_token_count":    st["token_count"],
                "stream_itl_mean_ms":    st["itl_mean_ms"],
                "stream_itl_p50_ms":     st["itl_p50_ms"],
                "stream_itl_p95_ms":     st["itl_p95_ms"],
            })
            for j, itl in enumerate(st["itl_values"], 1):
                itl_rows.append({"query_idx": i, "token_idx": j, "itl_ms": itl})
            if i % 10 == 0:
                logger.info("Progress: %d/%d", i, len(questions))
        except Exception as e:
            logger.warning("Query %d failed: %s", i, e)
    return results, itl_rows


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if not n:
        print("No results.")
        return

    def mean(key): return sum(r[key] for r in results) / n
    def pct(key, p): return _pct([r[key] for r in results], p)

    all_itl = []
    for r in results:
        # itl_values not stored per-query in results; use itl_mean as proxy
        pass

    print(f"\n{'='*62}")
    print(f"  Client-perceived latency comparison  (n={n})")
    print(f"{'='*62}")
    print(f"  {'Metric':<38} {'Non-stream':>10} {'Stream':>10}")
    print(f"  {'-'*58}")
    print(f"  {'All chars visible (ms)':<38} {mean('ns_client_total_ms'):>10.0f} {mean('stream_client_total_ms'):>10.0f}")
    print(f"  {'First token visible / TTFT (ms)':<38} {'—':>10} {mean('stream_client_ttft_ms'):>10.0f}")
    print(f"  {'Server TTFT (ms)':<38} {mean('ns_server_ttft_ms'):>10.0f} {mean('stream_server_ttft_ms'):>10.0f}")
    print(f"  {'-'*58}")
    print(f"  Avg TTFT saving (stream vs non-stream): {mean('ttft_saving_ms'):.0f} ms")
    print(f"\n  Inter-Token Latency (streaming only)")
    print(f"  {'-'*58}")
    print(f"  {'Token count (mean)':<38} {mean('stream_token_count'):>10.0f}")
    print(f"  {'ITL mean (ms)':<38} {mean('stream_itl_mean_ms'):>10.1f}")
    print(f"  {'ITL p50  (ms)':<38} {mean('stream_itl_p50_ms'):>10.1f}")
    print(f"  {'ITL p95  (ms)':<38} {mean('stream_itl_p95_ms'):>10.1f}")
    print(f"{'='*62}\n")


def save_results(results: list[dict], itl_rows: list[dict], prefix: str, results_dir: str) -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    if not n:
        logger.warning("No results to save.")
        return

    def mean(key): return round(sum(r[key] for r in results) / n, 1)

    # Per-query details
    detail_fields = [
        "idx", "question",
        "ns_client_total_ms", "ns_server_ttft_ms",
        "stream_client_ttft_ms", "stream_client_total_ms", "stream_server_ttft_ms",
        "ttft_saving_ms",
        "stream_token_count", "stream_itl_mean_ms", "stream_itl_p50_ms", "stream_itl_p95_ms",
    ]
    detail_path = out_dir / f"{prefix}_stream_comparison_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # One-row summary (read by plot_metrics.py)
    summary_path = out_dir / f"{prefix}_stream_comparison.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "total_queries",
            "ns_client_total_mean_ms",
            "stream_client_ttft_mean_ms", "stream_client_total_mean_ms",
            "ns_server_ttft_mean_ms", "stream_server_ttft_mean_ms",
            "ttft_saving_mean_ms",
            "stream_token_count_mean",
            "stream_itl_mean_ms", "stream_itl_p50_ms", "stream_itl_p95_ms",
        ])
        w.writerow([
            n,
            mean("ns_client_total_ms"),
            mean("stream_client_ttft_ms"), mean("stream_client_total_ms"),
            mean("ns_server_ttft_ms"), mean("stream_server_ttft_ms"),
            mean("ttft_saving_ms"),
            mean("stream_token_count"),
            mean("stream_itl_mean_ms"), mean("stream_itl_p50_ms"), mean("stream_itl_p95_ms"),
        ])
    logger.info("Saved summary  → %s", summary_path)

    # Per-token ITL details (for distribution plot)
    if itl_rows:
        itl_path = out_dir / f"{prefix}_itl_details.csv"
        with itl_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_idx", "token_idx", "itl_ms"])
            w.writeheader()
            w.writerows(itl_rows)
        logger.info("Saved ITL details → %s", itl_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",      required=True, help="Config prefix, e.g. full_semantic")
    parser.add_argument("--samples",     type=int, default=30)
    parser.add_argument("--top-k",       type=int, default=5)
    parser.add_argument("--api-url",     default=DEFAULT_API_URL)
    parser.add_argument("--gold-csv",    default=DEFAULT_GOLD_CSV)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading %d questions...", args.samples)
    questions = load_questions(args.gold_csv, args.samples, seed=args.seed)

    logger.info("Running %d queries (non-stream then stream each)...", len(questions))
    results, itl_rows = run(questions, api_url=args.api_url, top_k=args.top_k)

    print_summary(results)
    save_results(results, itl_rows, prefix=args.prefix, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
