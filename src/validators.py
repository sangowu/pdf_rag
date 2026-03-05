"""ChromaDB sync validation helpers: count, sample consistency, search smoke."""
import json
import logging
import random

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

EXPECTED_METADATA_KEYS = ("file_name", "page_index", "chunk_index", "char_count")
SAMPLE_SIZE = 5
SMOKE_QUERIES = ["English", "university", "teacher"]

def get_expected_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def get_collection_count_and_sample(collection, sample_ids: list[str]) -> tuple[int, dict]:
    actual_count = collection.count()
    got = collection.get(ids=sample_ids, include=["documents", "metadatas"])
    return actual_count, got

def check_sample_consistency(got: dict, expected_by_id: dict[str, dict]) -> tuple[bool, list[str]]:
    issues = []
    ids = got.get("ids") or []
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []
    for i, id_ in enumerate(ids):
        doc = documents[i] if i < len(documents) else None
        meta = metadatas[i] if i < len(metadatas) else None
        if doc is None or not str(doc).strip():
            issues.append(f"id={id_}: empty or missing document")
        if not meta:
            issues.append(f"id={id_}: missing metadata")
        else:
            for k in EXPECTED_METADATA_KEYS:
                if k not in meta:
                    issues.append(f"id={id_}: metadata missing key '{k}'")
        exp = expected_by_id.get(id_)
        if exp and meta:
            if exp.get("file_name") != meta.get("file_name"):
                issues.append(f"id={id_}: file_name mismatch")
            if exp.get("page_index") != meta.get("page_index"):
                issues.append(f"id={id_}: page_index mismatch")
    return len(issues) == 0, issues


def run_search_smoke(vs: VectorStore, queries: list[str]) -> tuple[bool, list[str]]:
    issues = []
    for q in queries:
        try:
            result = vs.search_by_text(q)
        except Exception as e:
            issues.append(f"query '{q}': error {e}")
            continue
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if not docs or not docs[0]:
            issues.append(f"query '{q}': no documents returned")
        else:
            first_doc = docs[0][0] if isinstance(docs[0], list) else docs[0]
            if not (first_doc and str(first_doc).strip()):
                issues.append(f"query '{q}': first document empty")
        if metas and metas[0]:
            first_meta = metas[0][0] if isinstance(metas[0], list) else metas[0]
            if "page_index" not in first_meta and "file_name" not in first_meta:
                issues.append(f"query '{q}': first result metadata missing page_index/file_name")
    return len(issues) == 0, issues