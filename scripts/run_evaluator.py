import argparse
import logging

from src.evaluator import RAGEvaluator


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Run retrieval evaluation and write metrics CSVs.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output files, e.g. 'base' -> base_metrics.csv, base_retrieval_details.csv.",
    )
    parser.add_argument(
        "--gold-csv",
        default=None,
        help="Optional path to gold_answers.csv (otherwise uses config paths.gold_answers_csv).",
    )
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    evaluator.evaluate_batch(gold_answers_csv=args.gold_csv, output_prefix=args.prefix)


if __name__ == "__main__":
    main()
