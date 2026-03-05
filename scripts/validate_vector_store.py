
import logging
import random

import chromadb

from src.utils import load_config
from src.vector_store import VectorStore
from src.validators import (
    SMOKE_QUERIES,
    SAMPLE_SIZE,
    check_sample_consistency,
    get_collection_count_and_sample,
    get_expected_chunks,
    run_search_smoke,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    paths = config.get("paths", {})
    all_chunk_path = paths.get("all_chunk_path", "results/chunk_results/all_chunks.json")
    chroma_cfg = config.get("chromadb", {})
    persist_dir = chroma_cfg.get("persist_directory", "vectors/chroma_db")
    collection_name = chroma_cfg.get("collection_name", "pdf_chunks")

    expected_list = get_expected_chunks(all_chunk_path)
    expected_count = len(expected_list)
    expected_by_id = {c["chunk_id"]: c for c in expected_list}

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    sample_ids = random.sample(
        [c["chunk_id"] for c in expected_list],
        min(SAMPLE_SIZE, len(expected_list)),
    ) if expected_list else []
    actual_count, got = get_collection_count_and_sample(collection, sample_ids)
    sample_ok, sample_issues = check_sample_consistency(got, expected_by_id)

    vs = VectorStore()
    search_ok, search_issues = run_search_smoke(vs, SMOKE_QUERIES)

    report = [
        "=== ChromaDB sync validation report ===",
        f"Expected chunks (from {all_chunk_path}): {expected_count}",
        f"Actual collection count: {actual_count}",
        f"Count match: {'PASS' if expected_count == actual_count else 'FAIL'}",
        "",
        f"Sample ({SAMPLE_SIZE} ids): {'PASS' if sample_ok else 'FAIL'}",
    ]
    if sample_issues:
        report.append("Sample issues: " + "; ".join(sample_issues))
    report.extend([
        "",
        f"Search smoke ({len(SMOKE_QUERIES)} queries): {'PASS' if search_ok else 'FAIL'}",
    ])
    if search_issues:
        report.append("Search issues: " + "; ".join(search_issues))
    report.append("")
    empty_or_missing = "Yes" if sample_issues or (expected_count != actual_count) else "No"
    report.append(f"Empty text / missing metadata: {empty_or_missing}")

    text = "\n".join(report)
    logger.info("\n%s", text)
    print(text)


if __name__ == "__main__":
    main()
