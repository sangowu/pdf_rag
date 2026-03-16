"""
Plot retrieval metrics (Hit@K and MRR), chunk size distribution, and eval timing.

By default runs all three (metrics + chunk size + timing). Use --no-chunk-size or --no-timing to skip.
- Metrics: auto-discover *_metrics.csv, plot Hit@K + MRR.
- Chunk size: auto-discover *_chunk_size_*.csv, merge and plot.
- Timing: auto-discover *_eval_timing.csv, plot embed/chroma/rerank/total.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Optional matplotlib; fail gracefully with a clear message if not installed
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


RESULTS_DIR_DEFAULT = "results"
METRICS_GLOB = "*_metrics.csv"
CHUNK_SIZE_STATS_GLOB = "*_chunk_size_stats.csv"
CHUNK_SIZE_DIST_GLOB = "*_chunk_size_distribution.csv"
CHUNK_SIZE_PLOT_DEFAULT = "results/plot/chunk_size_plot.png"
EVAL_TIMING_GLOB = "*_eval_timing.csv"
LATENCY_GLOB = "*_latency.csv"
EVAL_TIMING_PLOT_DEFAULT = "results/plot/eval_timing_plot.png"


def discover_metrics_csv(results_dir: str) -> tuple[list[str], list[str]]:
    """
    Find all *_metrics.csv under results_dir; return (paths, labels) with label = file stem prefix.
    E.g. base_metrics.csv -> label 'base'; recursive_metrics.csv -> 'recursive'.
    Sorted by label for stable order.
    """
    root = Path(results_dir)
    if not root.is_dir():
        return [], []
    files = sorted(root.glob(METRICS_GLOB), key=lambda p: p.stem)
    paths = [str(p) for p in files]
    labels = [p.stem.replace("_metrics", "") if p.stem.endswith("_metrics") else p.stem for p in files]
    return paths, labels


def discover_chunk_size_csv(results_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Find all *_chunk_size_distribution.csv under results_dir; return (dist_paths, stats_paths, labels).
    Label = prefix from filename (e.g. recursive_chunk_size_distribution.csv -> recursive).
    Stats path is inferred: {prefix}_chunk_size_stats.csv. Sorted by label.
    """
    root = Path(results_dir)
    if not root.is_dir():
        return [], [], []
    dist_files = sorted(root.glob(CHUNK_SIZE_DIST_GLOB), key=lambda p: p.stem)
    dist_paths = []
    stats_paths = []
    labels = []
    for p in dist_files:
        stem = p.stem
        prefix = stem.replace("_chunk_size_distribution", "") if stem.endswith("_chunk_size_distribution") else stem
        stats_file = root / f"{prefix}_chunk_size_stats.csv"
        if stats_file.exists():
            dist_paths.append(str(p))
            stats_paths.append(str(stats_file))
            labels.append(prefix)
        else:
            dist_paths.append(str(p))
            stats_paths.append("")
            labels.append(prefix)
    return dist_paths, stats_paths, labels


def discover_eval_timing_csv(results_dir: str) -> tuple[list[str], list[str]]:
    """Find all *_eval_timing.csv under results_dir; return (paths, labels). Label = prefix (e.g. base, rerank)."""
    root = Path(results_dir)
    if not root.is_dir():
        return [], []
    files = sorted(root.glob(EVAL_TIMING_GLOB), key=lambda p: p.stem)
    paths = [str(p) for p in files]
    labels = [
        p.stem.replace("_eval_timing", "") if p.stem.endswith("_eval_timing") else p.stem
        for p in files
    ]
    return paths, labels


def discover_latency_csv(results_dir: str) -> dict[str, str]:
    """Find all *_latency.csv under results_dir; return {label: path}. Label = prefix."""
    root = Path(results_dir)
    if not root.is_dir():
        return {}
    result = {}
    for p in root.glob(LATENCY_GLOB):
        label = p.stem.replace("_latency", "") if p.stem.endswith("_latency") else p.stem
        result[label] = str(p)
    return result


def discover_stream_comparison_csv(results_dir: str) -> dict[str, str]:
    """Find all *_stream_comparison.csv; return {label: path}."""
    root = Path(results_dir)
    if not root.is_dir():
        return {}
    result = {}
    for p in root.glob("*_stream_comparison.csv"):
        label = p.stem.replace("_stream_comparison", "")
        result[label] = str(p)
    return result


def discover_itl_details_csv(results_dir: str) -> dict[str, str]:
    """Find all *_itl_details.csv (per-token ITL rows); return {label: path}."""
    root = Path(results_dir)
    if not root.is_dir():
        return {}
    result = {}
    for p in root.glob("*_itl_details.csv"):
        label = p.stem.replace("_itl_details", "")
        result[label] = str(p)
    return result


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load metrics CSV; expected columns: k, hit_rate, avg_hit_rate, mrr."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    df = pd.read_csv(path)
    for col in ("k", "hit_rate", "mrr"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path.name}; got {list(df.columns)}")
    return df


def plot_metrics(
    csv_paths: list[str],
    labels: Optional[list[str]] = None,
    output_path: str = "results/plot/metrics_plot.png",
    title: Optional[str] = None,
) -> None:
    """
    Plot Hit@K and MRR from one or more metrics.csv files.

    Args:
        csv_paths: Paths to metrics CSV files (one per version).
        labels: Optional display names for each file (default: file stem).
        output_path: Where to save the PNG.
        title: Optional figure title.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if not csv_paths:
        raise ValueError("At least one metrics CSV path is required")

    dfs = [load_metrics(p) for p in csv_paths]
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("labels length must match csv_paths")

    # Strip common "debug_" prefix from labels for brevity
    display_labels = [l.removeprefix("debug_") for l in labels]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x", "1", "2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Hit@K curve — sorted by label, alternating line styles for readability
    sorted_triples = sorted(zip(display_labels, dfs, range(len(dfs))), key=lambda x: x[0])
    for i, (label, df, _) in enumerate(sorted_triples):
        ax1.plot(
            df["k"], df["hit_rate"],
            marker=markers[i % len(markers)],
            linestyle=line_styles[i % len(line_styles)],
            label=label,
        )
    ax1.set_xlabel("K")
    ax1.set_ylabel("Hit rate")
    ax1.set_title("Hit@K")
    all_hit_rates = [v for df in dfs for v in df["hit_rate"]]
    y_min = max(0.0, min(all_hit_rates) - 0.05)
    ax1.set_ylim(y_min, 1.02)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, frameon=True, fontsize=6)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sorted(set(k for d in dfs for k in d["k"])))

    # MRR — horizontal bar chart, sorted ascending so best is at top
    mrr_pairs = sorted(zip(display_labels, dfs), key=lambda x: float(x[1]["mrr"].iloc[-1]))
    mrr_labels = [p[0] for p in mrr_pairs]
    mrr_vals   = [float(p[1]["mrr"].iloc[-1]) for p in mrr_pairs]
    y = range(len(mrr_labels))
    ax2.barh(list(y), mrr_vals, color="steelblue", alpha=0.8)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(mrr_labels, fontsize=7)
    ax2.set_xlabel("MRR")
    ax2.set_title("Mean Reciprocal Rank")
    mrr_min = max(0.0, min(mrr_vals) - 0.05)
    ax2.set_xlim(mrr_min, 1.0)
    ax2.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(mrr_vals):
        ax2.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved metrics plot to %s", out)


def plot_chunk_size(
    results_dir: str = RESULTS_DIR_DEFAULT,
    output_path: str = CHUNK_SIZE_PLOT_DEFAULT,
    title: Optional[str] = None,
) -> None:
    """
    Auto-discover *_chunk_size_stats.csv and *_chunk_size_distribution.csv under results_dir,
    merge by label (prefix), then plot summary stats (mean, median) and binned distribution.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")

    dist_paths, stats_paths, labels = discover_chunk_size_csv(results_dir)
    if not dist_paths or not labels:
        raise ValueError(
            f"No files matching '{CHUNK_SIZE_DIST_GLOB}' found under {results_dir}. "
            "Run validate_chunk_distribution with --prefix (e.g. base, recursive) first."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: summary stats (mean, median) per version
    means = []
    medians = []
    for sp in stats_paths:
        if not sp or not Path(sp).exists():
            means.append(0)
            medians.append(0)
            continue
        df = pd.read_csv(sp)
        stat_to_val = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        means.append(float(stat_to_val.get("mean", 0)))
        medians.append(float(stat_to_val.get("median", 0)))
    x = range(len(labels))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], means, width, label="mean", color="steelblue", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], medians, width, label="median", color="coral", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Chunk size (chars)")
    ax1.set_title("Chunk size: mean & median")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: binned distribution (grouped bar or lines per bin)
    dist_dfs = [pd.read_csv(p) for p in dist_paths]
    bin_labels = dist_dfs[0]["bin_label"].tolist() if len(dist_dfs[0]) else []
    x_bins = range(len(bin_labels))
    width_bar = 0.8 / max(len(labels), 1)
    for i, (label, df) in enumerate(zip(labels, dist_dfs)):
        counts = df["count"].tolist() if "count" in df.columns else []
        offset = (i - len(labels) / 2 + 0.5) * width_bar
        ax2.bar([xi + offset for xi in x_bins], counts, width_bar, label=label, alpha=0.8)
    ax2.set_xticks(x_bins)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Bin (chars)")
    ax2.set_title("Chunk size distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved chunk size plot to %s", out)


def plot_eval_timing(
    results_dir: str = RESULTS_DIR_DEFAULT,
    output_path: str = EVAL_TIMING_PLOT_DEFAULT,
    title: Optional[str] = None,
) -> None:
    """
    Auto-discover *_eval_timing.csv under results_dir and plot retrieval timing.
    If matching *_latency.csv files exist (from benchmark_latency.py), adds a third
    panel showing TTFT and LLM generate time per configuration.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")

    paths, labels = discover_eval_timing_csv(results_dir)
    if not paths or not labels:
        raise ValueError(
            f"No files matching '{EVAL_TIMING_GLOB}' found under {results_dir}. "
            "Run evaluation with output_prefix (e.g. run_evaluator --prefix base) first."
        )

    retrieval_rows = []
    for p in paths:
        df = pd.read_csv(p)
        if len(df) == 0:
            continue
        row = df.iloc[0]
        retrieval_rows.append({
            "embed_mean_s": float(row.get("embed_mean_s", 0)),
            "chroma_mean_s": float(row.get("chroma_mean_s", 0)),
            "rerank_mean_s": float(row.get("rerank_mean_s", 0)),
            "total_mean_s": float(row.get("total_mean_s", 0)),
        })
    if not retrieval_rows:
        raise ValueError("No timing data rows in discovered CSV files.")

    # Load matching latency CSVs (TTFT + generate, from benchmark_latency.py)
    latency_map = discover_latency_csv(results_dir)
    latency_rows = []
    for label in labels:
        if label in latency_map:
            df = pd.read_csv(latency_map[label])
            if len(df) > 0:
                row = df.iloc[0]
                latency_rows.append({
                    "ttft_mean_ms": float(row.get("ttft_mean_ms", 0)),
                    "generate_mean_ms": float(row.get("generate_mean_ms", 0)),
                })
            else:
                latency_rows.append(None)
        else:
            latency_rows.append(None)
    has_latency = any(r is not None for r in latency_rows)

    # Load stream comparison CSVs (from test_stream_ttft.py)
    stream_map = discover_stream_comparison_csv(results_dir)
    stream_rows = []
    for label in labels:
        if label in stream_map:
            df = pd.read_csv(stream_map[label])
            if len(df) > 0:
                row = df.iloc[0]
                stream_rows.append({
                    "ns_client_ttft_mean_ms":     float(row.get("ns_client_ttft_mean_ms", 0)),
                    "stream_client_ttft_mean_ms": float(row.get("stream_client_ttft_mean_ms", 0)),
                    "ns_client_total_mean_ms":    float(row.get("ns_client_total_mean_ms", 0)),
                    "stream_client_total_mean_ms":float(row.get("stream_client_total_mean_ms", 0)),
                    "ttft_saving_mean_ms":        float(row.get("ttft_saving_mean_ms", 0)),
                })
            else:
                stream_rows.append(None)
        else:
            stream_rows.append(None)
    has_stream = any(r is not None for r in stream_rows)

    # Load ITL details CSVs (per-token inter-token latency, from test_stream_ttft.py)
    itl_map = discover_itl_details_csv(results_dir)
    itl_data: dict[str, list[float]] = {}   # {label: [itl_ms, ...]}
    for label in labels:
        if label in itl_map:
            with open(itl_map[label], encoding="utf-8") as f:
                itl_data[label] = [float(r["itl_ms"]) for r in csv.DictReader(f)]
    has_itl = bool(itl_data)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Layout: 2×2 base; expand to 2×3 when ITL distribution panel is available
    ncols = 3 if has_itl else 2
    figsize = (21, 10) if ncols == 3 else (14, 10)
    fig, axes = plt.subplots(2, ncols, figsize=figsize)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0] if has_latency else None
    ax4 = axes[1, 1] if has_stream else None
    ax5 = axes[1, 2] if (ncols == 3 and has_itl) else None

    # hide unused panels
    if not has_latency:
        axes[1, 0].set_visible(False)
    if not has_stream:
        axes[1, 1].set_visible(False)
    if ncols == 3 and not has_itl:
        axes[1, 2].set_visible(False)
    if ncols == 3:
        axes[0, 2].set_visible(False)  # top-right cell always empty in 2×3

    x = range(len(labels))
    width = 0.35
    display_labels = [l.removeprefix("debug_") for l in labels]

    # Panel 1: embed + chroma
    ax1.bar([i - width / 2 for i in x], [r["embed_mean_s"] for r in retrieval_rows], width, label="embed", color="steelblue", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], [r["chroma_mean_s"] for r in retrieval_rows], width, label="chroma", color="seagreen", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, rotation=15, ha="right", fontsize=7)
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Embed + vector search (mean/query)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: rerank + retrieval total (log scale)
    rerank_vals = [r["rerank_mean_s"] for r in retrieval_rows]
    total_vals = [r["total_mean_s"] for r in retrieval_rows]
    ax2.bar([i - width / 2 for i in x], [max(v, 0.001) for v in rerank_vals], width, label="rerank", color="coral", alpha=0.8)
    ax2.bar([i + width / 2 for i in x], [max(v, 0.001) for v in total_vals], width, label="retrieval total", color="purple", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, rotation=15, ha="right", fontsize=7)
    ax2.set_ylabel("Time (s)")
    ax2.set_yscale("log")
    ax2.set_title("Rerank + retrieval total (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3 (optional): TTFT + generate time from benchmark_latency.py
    if ax3 is not None:
        valid_idx = [i for i, r in enumerate(latency_rows) if r is not None]
        x3 = [i for i in x if i in valid_idx]
        xlabels3 = [display_labels[i] for i in x3]
        ttft_vals = [latency_rows[i]["ttft_mean_ms"] for i in x3]
        gen_vals  = [latency_rows[i]["generate_mean_ms"] for i in x3]

        ax3.bar([i - width / 2 for i in range(len(x3))], ttft_vals, width, label="TTFT (ms)", color="darkorange", alpha=0.8)
        ax3.bar([i + width / 2 for i in range(len(x3))], gen_vals,  width, label="generate (ms)", color="teal", alpha=0.8)
        ax3.set_xticks(range(len(x3)))
        ax3.set_xticklabels(xlabels3, rotation=15, ha="right", fontsize=7)
        ax3.set_ylabel("Time (ms)")
        ax3.set_title("LLM latency: TTFT + generate (mean/query)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        for j, (tv, gv) in enumerate(zip(ttft_vals, gen_vals)):
            ax3.text(j - width / 2, tv + 1, f"{tv:.0f}", ha="center", fontsize=6)
            ax3.text(j + width / 2, gv + 1, f"{gv:.0f}", ha="center", fontsize=6)

        missing = [display_labels[i] for i in x if i not in valid_idx]
        if missing:
            logger.warning("No latency CSV found for: %s. Run benchmark_latency.py --prefix <name>.", missing)

    # Panel 4 (optional): client-perceived TTFT — stream first-token vs non-stream total
    if ax4 is not None:
        valid_idx4 = [i for i, r in enumerate(stream_rows) if r is not None]
        x4 = list(range(len(valid_idx4)))
        xlabels4  = [display_labels[i] for i in valid_idx4]
        ns_total  = [stream_rows[i]["ns_client_total_mean_ms"]     for i in valid_idx4]
        st_ttft   = [stream_rows[i]["stream_client_ttft_mean_ms"]  for i in valid_idx4]
        st_total  = [stream_rows[i]["stream_client_total_mean_ms"] for i in valid_idx4]
        savings   = [stream_rows[i]["ttft_saving_mean_ms"]         for i in valid_idx4]

        w4 = 0.25
        ax4.bar([p - w4 for p in x4], ns_total, w4*2, label="non-stream: all chars visible", color="steelblue", alpha=0.80)
        ax4.bar([p + w4 for p in x4], st_ttft,  w4*2, label="stream: first token (TTFT)",   color="darkorange", alpha=0.80)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(xlabels4, rotation=15, ha="right", fontsize=7)
        ax4.set_ylabel("Time (ms)")
        ax4.set_title("Client-perceived latency: non-stream total vs stream TTFT")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
        for j, (nt, st, sv) in enumerate(zip(ns_total, st_ttft, savings)):
            ax4.text(j - w4, nt + 5, f"{nt:.0f}", ha="center", fontsize=6)
            ax4.text(j + w4, st + 5, f"{st:.0f}", ha="center", fontsize=6)
            ax4.annotate(f"−{sv:.0f}ms", xy=(j + w4, st), xytext=(0, 14),
                         textcoords="offset points", ha="center", fontsize=6, color="green",
                         arrowprops=dict(arrowstyle="-", color="green", lw=0.8))

        missing4 = [display_labels[i] for i in range(len(labels)) if i not in valid_idx4]
        if missing4:
            logger.warning("No stream comparison CSV for: %s. Run test_stream_ttft.py --prefix <name>.", missing4)

    # Panel 5 (optional): inter-token latency distribution (box plot per config)
    if ax5 is not None:
        itl_labels = [l for l in labels if l in itl_data and itl_data[l]]
        itl_series  = [itl_data[l] for l in itl_labels]
        display_itl = [l.removeprefix("debug_") for l in itl_labels]

        bp = ax5.boxplot(
            itl_series,
            labels=display_itl,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="red", linewidth=1.5),
        )
        colors = ["steelblue", "seagreen", "darkorange", "purple", "coral"]
        for patch, color in zip(bp["boxes"], colors * 10):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # annotate mean + p95 per config
        for j, (lbl, series) in enumerate(zip(itl_labels, itl_series), 1):
            mean_v = sum(series) / len(series)
            p95_v  = sorted(series)[int(len(series) * 0.95)]
            ax5.text(j, mean_v, f"μ={mean_v:.0f}", ha="center", va="bottom", fontsize=6, color="navy")
            ax5.text(j, p95_v,  f"p95={p95_v:.0f}", ha="center", va="bottom", fontsize=5, color="gray")

        ax5.set_ylabel("Inter-token latency (ms)")
        ax5.set_title("Streaming inter-token latency distribution (outliers hidden)")
        ax5.grid(True, alpha=0.3, axis="y")
        ax5.tick_params(axis="x", labelsize=7)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eval timing plot to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Plot metrics, chunk size, and eval timing (default: all). Auto-discover CSVs in results dir."
    )
    parser.add_argument(
        "metrics_csv",
        nargs="*",
        help="Optional: paths to metrics CSV files. If omitted, auto-discover *_metrics.csv under --results-dir (unless --chunk-size-only).",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR_DEFAULT,
        help="Directory to scan for *_metrics.csv and *_chunk_size_*.csv (default: results)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display names for each metrics CSV; default: from filename prefix",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/plot/metrics_plot.png",
        help="Output PNG path for metrics plot (default: results/plot/metrics_plot.png)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title for metrics plot",
    )
    parser.add_argument(
        "--chunk-size",
        action="store_true",
        default=True,
        help="Plot chunk size (default: True). Use --no-chunk-size to disable.",
    )
    parser.add_argument(
        "--no-chunk-size",
        action="store_false",
        dest="chunk_size",
        help="Disable chunk size plot.",
    )
    parser.add_argument(
        "--chunk-size-only",
        action="store_true",
        help="Only plot chunk size; do not plot metrics or timing",
    )
    parser.add_argument(
        "--chunk-size-output",
        default=CHUNK_SIZE_PLOT_DEFAULT,
        help="Output PNG path for chunk size plot (default: results/plot/chunk_size_plot.png)",
    )
    parser.add_argument(
        "--chunk-size-title",
        default=None,
        help="Optional figure title for chunk size plot",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="Plot eval timing (default: True). Use --no-timing to disable.",
    )
    parser.add_argument(
        "--no-timing",
        action="store_false",
        dest="timing",
        help="Disable eval timing plot.",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Only plot eval timing; do not plot metrics or chunk size",
    )
    parser.add_argument(
        "--timing-output",
        default=EVAL_TIMING_PLOT_DEFAULT,
        help="Output PNG path for eval timing plot (default: results/plot/eval_timing_plot.png)",
    )
    parser.add_argument(
        "--timing-title",
        default=None,
        help="Optional figure title for eval timing plot",
    )
    args = parser.parse_args()

    if args.timing_only:
        paths, labels = discover_eval_timing_csv(args.results_dir)
        if not paths:
            raise SystemExit(
                f"No files matching '{EVAL_TIMING_GLOB}' found under {args.results_dir}. "
                "Run evaluation with --prefix (e.g. run_evaluator --prefix base) first."
            )
        logger.info("Auto-discovered %d eval timing file(s): %s", len(paths), labels)
        plot_eval_timing(
            results_dir=args.results_dir,
            output_path=args.timing_output,
            title=args.timing_title,
        )
        return

    if args.chunk_size_only:
        dist_paths, _, labels = discover_chunk_size_csv(args.results_dir)
        if not dist_paths:
            raise SystemExit(
                f"No files matching '{CHUNK_SIZE_DIST_GLOB}' found under {args.results_dir}. "
                "Run validate_chunk_distribution with --prefix (e.g. base, recursive) first."
            )
        logger.info("Auto-discovered %d chunk size file(s): %s", len(dist_paths), labels)
        plot_chunk_size(
            results_dir=args.results_dir,
            output_path=args.chunk_size_output,
            title=args.chunk_size_title,
        )
        return

    csv_paths = args.metrics_csv
    labels = args.labels
    if not csv_paths:
        csv_paths, labels = discover_metrics_csv(args.results_dir)
        if not csv_paths:
            raise SystemExit(
                f"No files matching '{METRICS_GLOB}' found under {args.results_dir}. "
                "Either pass metrics CSV paths or add *_metrics.csv (e.g. base_metrics.csv) to that directory."
            )
        logger.info("Auto-discovered %d metrics file(s): %s", len(csv_paths), labels)

    plot_metrics(
        csv_paths=csv_paths,
        labels=labels,
        output_path=args.output,
        title=args.title,
    )

    if args.chunk_size:
        dist_paths, _, _ = discover_chunk_size_csv(args.results_dir)
        if not dist_paths:
            logger.warning("No *_chunk_size_distribution.csv under %s; skipping chunk size plot.", args.results_dir)
        else:
            plot_chunk_size(
                results_dir=args.results_dir,
                output_path=args.chunk_size_output,
                title=args.chunk_size_title,
            )

    if args.timing:
        timing_paths, _ = discover_eval_timing_csv(args.results_dir)
        if not timing_paths:
            logger.warning("No *_eval_timing.csv under %s; skipping eval timing plot.", args.results_dir)
        else:
            plot_eval_timing(
                results_dir=args.results_dir,
                output_path=args.timing_output,
                title=args.timing_title,
            )


if __name__ == "__main__":
    main()
