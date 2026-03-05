"""
Validate chunk size distribution from all_chunks.json.

Reads char_count (or len(text)) per chunk, computes summary statistics and
a binned distribution, then writes:
  - results/chunk_size_stats.csv (or {prefix}_chunk_size_stats.csv if --prefix set)
  - results/chunk_size_distribution.csv (or {prefix}_chunk_size_distribution.csv)
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from src.utils import load_config

logger = logging.getLogger(__name__)

DEFAULT_STATS_BASENAME = "chunk_size_stats.csv"
DEFAULT_DIST_BASENAME = "chunk_size_distribution.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_BIN_EDGES = [0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 999999]


def _output_paths(prefix: str | None, out_dir: str = DEFAULT_OUTPUT_DIR) -> tuple[str, str]:
    """Return (stats_path, dist_path); if prefix set, filenames get prefix (e.g. base_chunk_size_stats.csv)."""
    if prefix:
        stats_path = str(Path(out_dir) / f"{prefix}_{DEFAULT_STATS_BASENAME}")
        dist_path = str(Path(out_dir) / f"{prefix}_{DEFAULT_DIST_BASENAME}")
    else:
        stats_path = str(Path(out_dir) / DEFAULT_STATS_BASENAME)
        dist_path = str(Path(out_dir) / DEFAULT_DIST_BASENAME)
    return stats_path, dist_path


def _get_sizes(chunks: list[dict]) -> list[int]:
    """Extract size (char_count or len(text)) for each chunk."""
    out = []
    for c in chunks:
        size = c.get("char_count")
        if size is not None:
            out.append(int(size))
        else:
            text = c.get("text", "")
            out.append(len(text))
    return out


def _compute_stats(sizes: list[int]) -> dict[str, float]:
    """Compute summary statistics; no numpy required."""
    if not sizes:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0.0}
    n = len(sizes)
    sorted_s = sorted(sizes)
    min_s = sorted_s[0]
    max_s = sorted_s[-1]
    mean_s = sum(sizes) / n
    median_s = sorted_s[n // 2] if n % 2 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
    variance = sum((x - mean_s) ** 2 for x in sizes) / n
    std_s = variance ** 0.5

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(n * p / 100)))
        return float(sorted_s[idx])

    return {
        "count": n,
        "min": min_s,
        "max": max_s,
        "mean": round(mean_s, 2),
        "median": median_s,
        "std": round(std_s, 2),
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
    }


def _compute_binned(sizes: list[int], bin_edges: list[int]) -> list[tuple[str, int, int, int]]:
    """Return list of (bin_label, bin_low, bin_high, count)."""
    counts = [0] * (len(bin_edges) - 1)
    for s in sizes:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= s < bin_edges[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    out = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        label = f"{low}-{high}" if high < 999999 else f"{low}+"
        out.append((label, low, high, counts[i]))
    return out


def run(
    all_chunk_path: str | None = None,
    stats_path: str | None = None,
    dist_path: str | None = None,
    bin_edges: list[int] | None = None,
    prefix: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Load all_chunks.json, compute size distribution, write stats and histogram CSVs.

    Args:
        all_chunk_path: Path to all_chunks.json; default from config paths.all_chunk_path.
        stats_path: Output path for summary statistics CSV; if None, derived from prefix and output_dir.
        dist_path: Output path for binned distribution CSV; if None, derived from prefix and output_dir.
        bin_edges: Edges for histogram bins; default covers 0, 64, 128, ..., 512, ..., 8192, 999999.
        prefix: Optional label (e.g. base, recursive); applied to output filenames when stats_path/dist_path not given.
        output_dir: Directory for output files when paths are derived from prefix (default: results).
    """
    resolved_stats, resolved_dist = _output_paths(prefix, output_dir)
    if stats_path is None:
        stats_path = resolved_stats
    if dist_path is None:
        dist_path = resolved_dist
    config = load_config()
    path = all_chunk_path or config.get("paths", {}).get("all_chunk_path", "results/chunk_results/all_chunks.json")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"all_chunks.json not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list of chunks in {p}")

    sizes = _get_sizes(chunks)
    if not sizes:
        logger.warning("No chunks with size found in %s", p)
        stats = _compute_stats([])
    else:
        stats = _compute_stats(sizes)

    edges = bin_edges or DEFAULT_BIN_EDGES
    binned = _compute_binned(sizes, edges)

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dist_path).parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stat", "value"])
        for name, value in stats.items():
            w.writerow([name, value])

    with open(dist_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_label", "bin_low", "bin_high", "count"])
        for row in binned:
            w.writerow(row)

    logger.info("Chunk size validation: %d chunks, mean=%.1f, median=%s -> %s, %s", len(sizes), stats["mean"], stats["median"], stats_path, dist_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate chunk size distribution from all_chunks.json.")
    parser.add_argument("--all-chunk-path", default=None, help="Path to all_chunks.json (default: config paths.all_chunk_path)")
    parser.add_argument("--prefix", default=None, help="Optional label for output filenames (e.g. base, recursive) -> {prefix}_chunk_size_stats.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output files when using --prefix (default: results)")
    parser.add_argument("--stats", default=None, help="Output CSV for summary stats (overrides --prefix filename)")
    parser.add_argument("--dist", default=None, help="Output CSV for binned distribution (overrides --prefix filename)")
    args = parser.parse_args()
    run(
        all_chunk_path=args.all_chunk_path,
        stats_path=args.stats,
        dist_path=args.dist,
        prefix=args.prefix,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
