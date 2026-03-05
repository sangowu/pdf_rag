"""
Append retrieval metrics of the current run to a history CSV for version tracking.

Usage example:
    PYTHONPATH=. python -m scripts.update_metrics_history --tag embed_v2

This will:
    - Read the latest metrics CSV (paths.metrics_csv from config, default: results/metrics.csv)
    - For each K row in that file, append one row to results/metrics_history.csv with:
        timestamp_iso, tag, k, total_queries, num_hit, hit_rate, avg_hit_rate, mrr, metrics_csv
"""

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils import load_config


logger = logging.getLogger(__name__)


def _load_paths() -> tuple[Path, Path]:
    """Load metrics.csv and history path from config."""
    config = load_config()
    paths_cfg = config.get("paths", {})
    metrics_csv = Path(paths_cfg.get("metrics_csv", "results/metrics.csv"))
    history_csv = Path(paths_cfg.get("metrics_history_csv", "results/metrics_history.csv"))
    return metrics_csv, history_csv


def append_metrics_history(tag: str, note: str | None = None) -> None:
    """Append rows from metrics.csv into metrics_history.csv with run metadata."""
    metrics_path, history_path = _load_paths()

    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    history_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()

    with metrics_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    if not rows:
        logger.warning("No rows found in metrics file: %s", metrics_path)
        return

    write_header = not history_path.exists()
    fieldnames = [
        "timestamp_iso",
        "tag",
        "note",
        "k",
        "total_queries",
        "num_hit",
        "hit_rate",
        "avg_hit_rate",
        "mrr",
        "metrics_csv",
    ]

    with history_path.open("a", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "timestamp_iso": timestamp_iso,
                    "tag": tag,
                    "note": note or "",
                    "k": row.get("k", ""),
                    "total_queries": row.get("total_queries", ""),
                    "num_hit": row.get("num_hit", ""),
                    "hit_rate": row.get("hit_rate", ""),
                    "avg_hit_rate": row.get("avg_hit_rate", ""),
                    "mrr": row.get("mrr", ""),
                    "metrics_csv": str(metrics_path),
                }
            )

    logger.info("Appended %d rows to metrics history: %s", len(rows), history_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Append current retrieval metrics (metrics.csv) into metrics_history.csv for version tracking."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Short identifier for this run (e.g. embed_v2, chunk_640).",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional free-form note about this run (e.g. changed embedding model).",
    )
    args = parser.parse_args()

    append_metrics_history(tag=args.tag, note=args.note)


if __name__ == "__main__":
    main()

