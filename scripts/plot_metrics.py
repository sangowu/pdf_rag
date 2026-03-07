"""
Plot retrieval metrics (Hit@K and MRR), chunk size distribution, and eval timing.

By default runs all three (metrics + chunk size + timing). Use --no-chunk-size or --no-timing to skip.
- Metrics: auto-discover *_metrics.csv, plot Hit@K + MRR.
- Chunk size: auto-discover *_chunk_size_*.csv, merge and plot.
- Timing: auto-discover *_eval_timing.csv, plot embed/chroma/rerank/total.
"""

import argparse
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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Hit@K curve
    for df, label in zip(dfs, labels):
        ax1.plot(df["k"], df["hit_rate"], marker="o", label=label)
    ax1.set_xlabel("K")
    ax1.set_ylabel("Hit rate")
    ax1.set_title("Hit@K")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True, fontsize=6)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sorted(set(k for d in dfs for k in d["k"])))

    # MRR bar (one bar per version)
    mrrs = []
    for df in dfs:
        # MRR is constant across k in our format; take last row or first
        mrr = float(df["mrr"].iloc[-1]) if len(df) else 0.0
        mrrs.append(mrr)
    x = range(len(labels))
    ax2.bar(x, mrrs, color="steelblue", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("MRR")
    ax2.set_title("Mean Reciprocal Rank")
    ax2.set_ylim(0, 1.0)

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
    Auto-discover *_eval_timing.csv under results_dir, then plot mean time per stage
    (embed, chroma, rerank, total) as grouped bar chart for comparison.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")

    paths, labels = discover_eval_timing_csv(results_dir)
    if not paths or not labels:
        raise ValueError(
            f"No files matching '{EVAL_TIMING_GLOB}' found under {results_dir}. "
            "Run evaluation with output_prefix (e.g. run_evaluator --prefix base) first."
        )

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if len(df) == 0:
            continue
        row = df.iloc[0]
        rows.append({
            "embed_mean_s": float(row.get("embed_mean_s", 0)),
            "chroma_mean_s": float(row.get("chroma_mean_s", 0)),
            "rerank_mean_s": float(row.get("rerank_mean_s", 0)),
            "total_mean_s": float(row.get("total_mean_s", 0)),
        })
    if not rows:
        raise ValueError("No timing data rows in discovered CSV files.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(labels))
    width = 0.35

    # Left: embed + chroma (small scale, 0 ~ 0.05s typically)
    ax1.bar([i - width / 2 for i in x], [r["embed_mean_s"] for r in rows], width, label="embed (mean s)", color="steelblue", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], [r["chroma_mean_s"] for r in rows], width, label="chroma (mean s)", color="seagreen", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Embed + vector search (mean per query)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: rerank + total (log scale so both ~0.03s and ~20s are visible)
    rerank_vals = [r["rerank_mean_s"] for r in rows]
    total_vals = [r["total_mean_s"] for r in rows]
    ax2.bar([i - width / 2 for i in x], [max(v, 0.001) for v in rerank_vals], width, label="rerank (mean s)", color="coral", alpha=0.8)
    ax2.bar([i + width / 2 for i in x], [max(v, 0.001) for v in total_vals], width, label="total (mean s)", color="purple", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Time (s)")
    ax2.set_yscale("log")
    ax2.set_title("Rerank + total (mean per query, log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

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
