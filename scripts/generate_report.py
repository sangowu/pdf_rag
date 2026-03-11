"""
Auto-generate an experiment comparison report in Markdown.

Discovers all *_metrics.csv, *_eval_timing.csv, and *_answer_metrics.csv
under results/, computes delta vs the 'base' experiment, and writes
results/experiment_report.md with tables and embedded plot images.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --results-dir results --baseline base --output results/experiment_report.md
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
BASELINE = "base"
OUTPUT = "results/experiment_report.md"

PLOT_METRICS = "plot/metrics_plot.png"
PLOT_TIMING = "plot/eval_timing_plot.png"
PLOT_CHUNK = "plot/chunk_size_plot.png"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _label_from_stem(stem: str, suffix: str) -> str:
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def discover(results_dir: Path, glob: str, suffix: str) -> dict[str, Path]:
    """Return label -> path dict for all matching files, sorted by label."""
    files = sorted(results_dir.glob(glob), key=lambda p: p.stem)
    return {_label_from_stem(p.stem, suffix): p for p in files}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["k"] = df["k"].astype(int)
    return df.set_index("k")


def load_timing(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "total_queries": int(row.get("total_queries", 0)),
        "embed_mean_ms": round(float(row.get("embed_mean_s", 0)) * 1000, 2),
        "chroma_mean_ms": round(float(row.get("chroma_mean_s", 0)) * 1000, 2),
        "rerank_mean_ms": round(float(row.get("rerank_mean_s", 0)) * 1000, 2),
        "total_mean_ms": round(float(row.get("total_mean_s", 0)) * 1000, 2),
    }


def load_answer_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "em": round(float(row.get("exact_match", 0)), 4),
        "rouge_l": round(float(row.get("rouge_l", 0)), 4),
        "f1": round(float(row.get("f1", 0)), 4),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _delta(v: float, base: float) -> str:
    if base == 0:
        return "—"
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}pp"


def _ms(v: float) -> str:
    return f"{v:.1f} ms"


def _delta_ms(v: float, base: float) -> str:
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f} ms"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_retrieval(
    metrics_by_label: dict[str, pd.DataFrame],
    baseline: str,
    k_list: list[int],
) -> list[str]:
    lines = ["## Retrieval Metrics (Hit@K & MRR)\n"]

    base_df = metrics_by_label.get(baseline)

    # Header
    hit_cols = [f"Hit@{k}" for k in k_list]
    delta_cols = [f"ΔHit@{k}" for k in k_list]
    header = ["Experiment"] + hit_cols + ["MRR"] + delta_cols + ["ΔMRR"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for label, df in sorted(metrics_by_label.items(), key=lambda x: (x[0] != baseline, x[0])):
        row = [f"**{label}**" if label == baseline else label]
        for k in k_list:
            hit = float(df.loc[k, "hit_rate"]) if k in df.index else 0.0
            row.append(_pct(hit))
        mrr_val = float(df["mrr"].iloc[-1]) if "mrr" in df.columns else 0.0
        row.append(_pct(mrr_val))
        # Deltas
        for k in k_list:
            hit = float(df.loc[k, "hit_rate"]) if k in df.index else 0.0
            base_hit = float(base_df.loc[k, "hit_rate"]) if (base_df is not None and k in base_df.index) else 0.0
            row.append("—" if label == baseline else _delta(hit, base_hit))
        base_mrr = float(base_df["mrr"].iloc[-1]) if base_df is not None and "mrr" in base_df.columns else 0.0
        row.append("—" if label == baseline else _delta(mrr_val, base_mrr))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def section_timing(
    timing_by_label: dict[str, dict],
    baseline: str,
) -> list[str]:
    lines = ["## Retrieval Latency (mean per query)\n"]

    base = timing_by_label.get(baseline, {})
    header = ["Experiment", "Embed", "ChromaDB", "Rerank", "Total", "ΔTotal"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for label, t in sorted(timing_by_label.items(), key=lambda x: (x[0] != baseline, x[0])):
        row = [f"**{label}**" if label == baseline else label]
        row.append(_ms(t.get("embed_mean_ms", 0)))
        row.append(_ms(t.get("chroma_mean_ms", 0)))
        row.append(_ms(t.get("rerank_mean_ms", 0)))
        row.append(_ms(t.get("total_mean_ms", 0)))
        delta = "—" if label == baseline else _delta_ms(
            t.get("total_mean_ms", 0), base.get("total_mean_ms", 0)
        )
        row.append(delta)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def section_answer(
    answer_by_label: dict[str, dict],
    baseline: str,
) -> list[str]:
    lines = ["## Answer Quality (EM / ROUGE-L / F1)\n"]

    base = answer_by_label.get(baseline, {})
    header = ["Experiment", "Exact Match", "ROUGE-L", "F1", "ΔEM", "ΔROUGE-L"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for label, m in sorted(answer_by_label.items(), key=lambda x: (x[0] != baseline, x[0])):
        row = [f"**{label}**" if label == baseline else label]
        row.append(_pct(m.get("em", 0)))
        row.append(_pct(m.get("rouge_l", 0)))
        row.append(_pct(m.get("f1", 0)))
        row.append("—" if label == baseline else _delta(m.get("em", 0), base.get("em", 0)))
        row.append("—" if label == baseline else _delta(m.get("rouge_l", 0), base.get("rouge_l", 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def section_plots(results_dir: Path) -> list[str]:
    lines = ["## Plots\n"]
    for title, rel_path in [
        ("Retrieval Metrics (Hit@K & MRR)", PLOT_METRICS),
        ("Retrieval Latency", PLOT_TIMING),
        ("Chunk Size Distribution", PLOT_CHUNK),
    ]:
        abs_path = results_dir / rel_path
        if abs_path.exists():
            lines.append(f"### {title}\n")
            lines.append(f"![{title}]({rel_path})\n")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(results_dir: str, baseline: str, output: str) -> None:
    rd = Path(results_dir)

    metrics_map = discover(rd, "*_metrics.csv", "_metrics")
    timing_map = discover(rd, "*_eval_timing.csv", "_eval_timing")
    answer_map = discover(rd, "*_answer_metrics.csv", "_answer_metrics")

    # Exclude per-query detail files
    timing_map = {k: v for k, v in timing_map.items() if "detail" not in k}

    if not metrics_map:
        logger.warning("No *_metrics.csv found under %s", results_dir)

    # Load data
    metrics_by_label: dict[str, pd.DataFrame] = {}
    for label, path in metrics_map.items():
        try:
            metrics_by_label[label] = load_metrics(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    timing_by_label: dict[str, dict] = {}
    for label, path in timing_map.items():
        try:
            timing_by_label[label] = load_timing(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    answer_by_label: dict[str, dict] = {}
    for label, path in answer_map.items():
        try:
            answer_by_label[label] = load_answer_metrics(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    # Determine k values from available metrics
    k_list: list[int] = []
    for df in metrics_by_label.values():
        k_list = sorted(df.index.tolist())
        break
    if not k_list:
        k_list = [1, 3, 5]

    # Build report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# RAG Experiment Report\n",
        f"Generated: {now} | Baseline: `{baseline}` | Results: `{results_dir}/`\n",
        "---\n",
    ]

    if metrics_by_label:
        lines += section_retrieval(metrics_by_label, baseline, k_list)

    if timing_by_label:
        lines += section_timing(timing_by_label, baseline)

    if answer_by_label:
        lines += section_answer(answer_by_label, baseline)

    lines += section_plots(rd)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", out_path)
    print(f"Report written to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate Markdown experiment comparison report.")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--baseline", default=BASELINE, help="Experiment name used as baseline (default: base)")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    generate_report(args.results_dir, args.baseline, args.output)


if __name__ == "__main__":
    main()
