# scripts/run_full_pipeline.py
import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils import load_config
from src.chunk_manager import ChunkManager
from src.vector_store import VectorStore
from src.evaluator import RAGEvaluator


def run_chunking(config: dict, file_limit: int | None = None) -> None:
    """Run chunking over OCR structured JSONs; writes per-chunk schemas and all_chunks.json."""
    paths = config.get("paths", {})
    ocr_structured_dir = paths.get("ocr_structured_dir", "results/ocr_structured")
    chunk_results_dir = paths.get("chunk_results_dir", "results/chunk_results")
    all_chunk_path = paths.get("all_chunk_path", "results/chunk_results/all_chunks.json")

    cm = ChunkManager()
    effective_limit = None if file_limit == 0 else file_limit
    file_list = cm.list_full_paths(ocr_structured_dir, "*.json", limit=effective_limit)
    if not file_list:
        limit_msg = "no limit" if effective_limit is None else str(effective_limit)
        logging.warning("No OCR structured JSONs under %s (limit=%s); skipping chunking.", ocr_structured_dir, limit_msg)
        return

    all_chunk = []
    for file_path in tqdm(file_list, desc="Chunking", unit="file"):
        all_chunk.extend(cm.generate_chunks(file_path))

    out_path = Path(all_chunk_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm._write_all_chunk(all_chunk, out_path)
    logging.info("Chunking done: %d files -> %d chunks -> %s", len(file_list), len(all_chunk), out_path)

def run_embedding(config: dict) -> None:
    vs = VectorStore()
    all_chunk_path = config["paths"]["all_chunk_path"]
    # 读 all_chunks.json
    all_chunk_data = vs._read_all_chunk(all_chunk_path)
    new_table = vs.embed_chunks(all_chunk_data)
    vs.add_chunks_to_chroma(new_table)

def run_generate_gold(config: dict) -> None:
    # Call QA generation with default args so it does not parse this script's argv (--skip-ocr, --eval-prefix, etc.)
    from scripts import generate_qa_from_chunks
    argv_saved = sys.argv
    try:
        sys.argv = ["scripts/generate_qa_from_chunks.py"]
        generate_qa_from_chunks.main()
    finally:
        sys.argv = argv_saved
    # Release LLM from GPU so later stages (e.g. reranker) have enough VRAM
    generate_qa_from_chunks.unload_local_model()

def run_evaluate(config: dict, eval_prefix: str | None = None) -> None:
    evaluator = RAGEvaluator()
    evaluator.evaluate_batch(output_prefix=eval_prefix)
    # Release reranker from GPU after evaluation
    try:
        from src.reranker import unload_reranker
        unload_reranker()
    except Exception:  # noqa: BLE001
        pass

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_config()
    chunk_cfg = config.get("chunking", {})
    default_file_limit = int(chunk_cfg.get("file_limit", 100))

    parser = argparse.ArgumentParser(description="Run full OCR→Chunk→Embedding→Retrieval→Eval pipeline.")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--eval-prefix",
        default=None,
        help="Prefix for evaluation outputs (e.g. base -> base_metrics.csv, base_retrieval_details.csv).",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=None,
        help="Max OCR JSON files to chunk; 0 = no limit (full-scale). Default: from config chunking.file_limit.",
    )
    args = parser.parse_args()

    file_limit = args.chunk_limit if args.chunk_limit is not None else default_file_limit
    if not args.skip_chunk:
        run_chunking(config, file_limit=file_limit)
    if not args.skip_embed:
        run_embedding(config)
    if not args.skip_gold:
        run_generate_gold(config)
    if not args.skip_eval:
        run_evaluate(config, eval_prefix=args.eval_prefix)

if __name__ == "__main__":
    main()