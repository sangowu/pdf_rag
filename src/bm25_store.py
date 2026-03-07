import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.utils import load_config

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def _tokenize_for_bm25(text: str) -> list[str]:
    if not (text and text.strip()):
        return []
    try:
        import jieba
        return list(jieba.cut(text.strip()))
    except ImportError:
        return list(text.strip())


class BM25Store:

    def __init__(self, all_chunk_path: Optional[str] = None) -> None:
        config = load_config()
        paths = config.get("paths", {})
        self._all_chunk_path = all_chunk_path or paths.get("all_chunk_path", "results/chunk_results/all_chunks.json")
        self._chunks: list[dict] = []
        self._corpus_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._top_k = config.get("evaluation", {}).get("top_k", 5)

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return
        if not HAS_BM25:
            raise RuntimeError("rank_bm25 is required for BM25Store. Install with: pip install rank_bm25")
        path = Path(self._all_chunk_path)
        if not path.exists():
            raise FileNotFoundError(f"all_chunks.json not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("all_chunks.json must be a list of chunk dicts")
        self._chunks = data
        self._corpus_ids = [c.get("chunk_id", "") for c in self._chunks]
        texts = [c.get("text", "") or "" for c in self._chunks]
        self._corpus_tokens = [_tokenize_for_bm25(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built: %d chunks from %s", len(self._chunks), path)

    def search_by_text(self, query_text: str, k: Optional[int] = None) -> dict:

        self._ensure_index()
        n_results = k if k is not None else self._top_k
        n_results = min(n_results, len(self._corpus_ids))

        t0 = time.perf_counter()
        query_tokens = _tokenize_for_bm25(query_text)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        bm25_time_s = time.perf_counter() - t0

        ids = [self._corpus_ids[i] for i in top_indices]
        documents = [self._chunks[i].get("text", "") or "" for i in top_indices]

        return {
            "ids": [ids],
            "documents": [documents],
            "embed_time_s": 0.0,
            "chroma_time_s": bm25_time_s,
        }