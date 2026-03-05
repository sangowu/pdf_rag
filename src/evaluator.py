import csv
import logging
import time
from pathlib import Path
from typing import Iterable

from src.utils import load_config
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEvaluator:

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._config = load_config()
        self._paths = self._config.get("paths", {})
        self._evaluation_cfg = self._config.get("evaluation", {})
        self._reranker_cfg = self._config.get("reranker", {})
        self.vector_store = vector_store or VectorStore()

        self.gold_answers_csv = self._paths.get("gold_answers_csv", "data/answers/gold_answers.csv")
        self.metrics_csv = self._paths.get("metrics_csv", "results/metrics.csv")
        default_top_k = int(self._evaluation_cfg.get("top_k", 5))
        self.default_k_list: list[int] = sorted({1, 3, default_top_k})

        self._reranker_enabled = bool(self._reranker_cfg.get("enabled", False))
        self._top_r = int(self._reranker_cfg.get("top_r", 20))

    def _search_chunk_ids(self, query: str, k: int) -> list[str]:
        """Search and optionally rerank; returns top-k chunk ids. Used when timing is not needed."""
        ids, _ = self._search_once(query, k)
        return ids

    def _search_once(
        self, query: str, max_k: int
    ) -> tuple[list[str], dict[str, float]]:
        """
        Run search (and optional rerank) once; return (top max_k chunk ids, timing dict).
        Timing dict: embed_time_s, chroma_time_s, rerank_time_s, total_time_s.
        """
        timings = {"embed_time_s": 0.0, "chroma_time_s": 0.0, "rerank_time_s": 0.0, "total_time_s": 0.0}
        if not query:
            return [], timings
        t_total_start = time.perf_counter()
        result = self.vector_store.search_by_text(query, k=self._top_r if self._reranker_enabled else max_k)
        timings["embed_time_s"] = float(result.get("embed_time_s", 0))
        timings["chroma_time_s"] = float(result.get("chroma_time_s", 0))
        ids_per_query = result.get("ids", [[]])
        docs_per_query = result.get("documents", [[]])
        ids = ids_per_query[0] if ids_per_query else []
        docs = docs_per_query[0] if docs_per_query else []
        if not ids or not docs:
            timings["total_time_s"] = time.perf_counter() - t_total_start
            return [], timings
        if self._reranker_enabled:
            from src.reranker import rerank
            doc_list = list(zip(ids, docs))
            t_rerank_start = time.perf_counter()
            ranked = rerank(query, doc_list, top_k=max_k)
            timings["rerank_time_s"] = time.perf_counter() - t_rerank_start
            ids = [chunk_id for chunk_id, _text, _score in ranked]
        else:
            ids = ids[:max_k]
        timings["total_time_s"] = time.perf_counter() - t_total_start
        return ids, timings

    def calculate_hit_at_k(self, retrieved_ids: list[str], gold_chunk_ids: set[str], k: int) -> dict:
        retrieved_ids = retrieved_ids[:k] if len(retrieved_ids) > k else retrieved_ids
        if not gold_chunk_ids:
            return {
                "k": k,
                "retrieved_chunk_ids": retrieved_ids,
                "gold_chunk_ids": [],
                "hit_count": 0,
                "hit_rate": 0.0,
                "hit": False,
            }
        hit_ids = gold_chunk_ids.intersection(set(retrieved_ids))
        hit_count = len(hit_ids)
        hit_rate = hit_count / len(gold_chunk_ids)
        return {
            "k": k,
            "retrieved_chunk_ids": retrieved_ids,
            "gold_chunk_ids": list(gold_chunk_ids),
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "hit": hit_count > 0,
        }

    def evaluate_batch(
        self,
        gold_answers_csv: str | None = None,
        k_list: Iterable[int] | None = None,
        output_prefix: str | None = None,
    ) -> None:

        gold_path = Path(gold_answers_csv or self.gold_answers_csv)
        if not gold_path.exists():
            logger.error("Gold answers file not found: %s", gold_path)
            raise FileNotFoundError(f"Gold answers file not found: {gold_path}")

        ks = sorted({int(k) for k in (k_list or self.default_k_list) if int(k) > 0})
        if not ks:
            logger.error("Empty k_list provided for evaluation.")
            raise ValueError("k_list must contain at least one positive integer.")

        stats: dict[int, dict[str, float | int]] = {
            k: {
                "total": 0,
                "num_hit": 0,
                "sum_hit_rate": 0.0,
                "sum_reciprocal_rank": 0.0,
            }
            for k in ks
        }
        details: list[dict] = []
        timing_details: list[dict] = []
        max_k = max(ks)

        with gold_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("question") or "").strip()
                gold_ids_str = (row.get("gold_chunk_ids") or "").strip()
                if not question or not gold_ids_str:
                    continue
                gold_ids = {cid for cid in gold_ids_str.split(",") if cid}
                if not gold_ids:
                    continue

                retrieved_ids_full, timings = self._search_once(question, max_k)
                timing_details.append({
                    "question_index": len(timing_details) + 1,
                    "embed_s": round(timings["embed_time_s"], 4),
                    "chroma_s": round(timings["chroma_time_s"], 4),
                    "rerank_s": round(timings["rerank_time_s"], 4),
                    "total_s": round(timings["total_time_s"], 4),
                })

                for k in ks:
                    result = self.calculate_hit_at_k(retrieved_ids_full, gold_ids, k)
                    retrieved_ids = result["retrieved_chunk_ids"]
                    s = stats[k]
                    s["total"] += 1
                    s["sum_hit_rate"] += result["hit_rate"]

                    first_rank = 0
                    for idx, cid in enumerate(retrieved_ids, start=1):
                        if cid in gold_ids:
                            first_rank = idx
                            break
                    if first_rank > 0:
                        s["sum_reciprocal_rank"] += 1.0 / first_rank

                    if result["hit"]:
                        s["num_hit"] += 1

                    details.append(
                        {
                            "question": question,
                            "k": k,
                            "gold_chunk_ids": ",".join(sorted(gold_ids)),
                            "retrieved_chunk_ids": ",".join(retrieved_ids),
                            "first_hit_rank": first_rank,
                            "hit": result["hit"],
                            "hit_rate": result["hit_rate"],
                        }
                    )

        base_metrics_path = Path(self.metrics_csv)
        metrics_dir = base_metrics_path.parent
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if output_prefix:
            metrics_path = metrics_dir / f"{output_prefix}_{base_metrics_path.name}"
        else:
            metrics_path = base_metrics_path

        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "total_queries", "num_hit", "hit_rate", "avg_hit_rate", "mrr"])
            for k in ks:
                s = stats[k]
                total = int(s["total"])
                if total == 0:
                    hit_rate = 0.0
                    avg_hit_rate = 0.0
                    mrr = 0.0
                else:
                    hit_rate = float(s["num_hit"]) / total
                    avg_hit_rate = float(s["sum_hit_rate"]) / total
                    mrr = float(s["sum_reciprocal_rank"]) / total
                writer.writerow([k, total, int(s["num_hit"]), hit_rate, avg_hit_rate, mrr])

        logger.info("Wrote retrieval metrics to %s", metrics_path)

        if output_prefix:
            details_path = metrics_path.parent / f"{output_prefix}_retrieval_details.csv"
        else:
            details_path = metrics_path.parent / "retrieval_details.csv"
        with details_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question",
                    "k",
                    "gold_chunk_ids",
                    "retrieved_chunk_ids",
                    "first_hit_rank",
                    "hit",
                    "hit_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(details)
        logger.info("Wrote retrieval details to %s", details_path)

        # Timing summary and per-query details
        if timing_details:
            nq = len(timing_details)
            sum_embed = sum(d["embed_s"] for d in timing_details)
            sum_chroma = sum(d["chroma_s"] for d in timing_details)
            sum_rerank = sum(d["rerank_s"] for d in timing_details)
            sum_total = sum(d["total_s"] for d in timing_details)
            timing_summary_path = metrics_dir / (f"{output_prefix}_eval_timing.csv" if output_prefix else "eval_timing.csv")
            with timing_summary_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "total_queries", "embed_total_s", "embed_mean_s", "chroma_total_s", "chroma_mean_s",
                    "rerank_total_s", "rerank_mean_s", "total_s", "total_mean_s",
                ])
                w.writerow([
                    nq,
                    round(sum_embed, 4), round(sum_embed / nq, 4),
                    round(sum_chroma, 4), round(sum_chroma / nq, 4),
                    round(sum_rerank, 4), round(sum_rerank / nq, 4),
                    round(sum_total, 4), round(sum_total / nq, 4),
                ])
            logger.info("Wrote eval timing summary to %s", timing_summary_path)
            timing_details_path = metrics_dir / (f"{output_prefix}_eval_timing_details.csv" if output_prefix else "eval_timing_details.csv")
            with timing_details_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["question_index", "embed_s", "chroma_s", "rerank_s", "total_s"])
                writer.writeheader()
                writer.writerows(timing_details)
            logger.info("Wrote eval timing details to %s", timing_details_path)


__all__ = ["RAGEvaluator"]

