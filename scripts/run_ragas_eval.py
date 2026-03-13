"""
scripts/run_ragas_eval.py

End-to-end RAG evaluation using RAGAS.
Flow: qa_pairs.jsonl -> retrieve top-K chunks -> generate answer -> RAGAS metrics

Usage:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --mode api --output results/ragas_api_results.csv
"""
import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.utils import load_config
from src.vector_store import VectorStore
from src.answer_generator import generate_answer
from src.ragas_evaluator import run_ragas

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    ragas_cfg = config.get("ragas", {})
    paths = config.get("paths", {})

    parser = argparse.ArgumentParser(description="End-to-end RAG eval with RAGAS.")
    parser.add_argument(
        "--mode",
        default=ragas_cfg.get("mode", "local"),
        choices=["local", "api"],
        help="LLM backend for RAGAS judge (default: from config ragas.mode)",
    )
    parser.add_argument(
        "--output",
        default="results/ragas_results.csv",
        help="Output CSV path for RAGAS results",
    )
    parser.add_argument(
        "--qa-path",
        default=paths.get("qa_pairs_path", "data/answers/qa_pairs.jsonl"),
        help="Path to qa_pairs.jsonl",
    )
    parser.add_argument(
        "--eval-prefix",
        default=None,
        help="Optional prefix for output filename (e.g. semantic -> results/semantic_ragas_results.csv)",
    )
    args = parser.parse_args()

    output_path = args.output
    if args.eval_prefix:
        output_path = f"results/{args.eval_prefix}_ragas_results.csv"

    # 1. Load QA pairs (already reflects debug/full pipeline sample)
    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        raise FileNotFoundError(f"QA pairs file not found: {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), qa_path)

    # 2. Pre-load local LLM once before the generation loop
    from scripts.generate_qa_from_chunks import _ensure_local_model_loaded
    _ensure_local_model_loaded()

    # 3. Retrieval + answer generation
    vs = VectorStore()
    top_k = config.get("evaluation", {}).get("top_k", 5)
    dataset = []

    for item in tqdm(qa_pairs, desc="Generating answers", unit="qa"):
        question = item.get("question", "").strip()
        if not question:
            continue

        retrieval = vs.search_by_text(question, k=top_k)
        contexts = retrieval["documents"][0]

        answer = generate_answer(question, contexts)

        dataset.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
        })

    logger.info("Answer generation done: %d items", len(dataset))

    # 4. RAGAS evaluation (model stays loaded — RAGAS judge reuses it)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = run_ragas(dataset, mode=args.mode, output_path=output_path)

    # 6. Summary
    # 6. Unload LLM after everything is done
    from scripts.generate_qa_from_chunks import unload_local_model
    unload_local_model()

    print("\n=== RAGAS Results ===")
    for col in ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]:
        if col in df.columns:
            print(f"  {col:<45}: {df[col].mean():.4f}")


if __name__ == "__main__":
    main()
