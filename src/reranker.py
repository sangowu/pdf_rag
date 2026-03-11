"""
Reranker module using BGE-reranker-v2-m3 (cross-encoder scoring).

Reranks (query, document) pairs and returns documents ordered by relevance score.
Used after first-stage retrieval to improve Hit@K and RAG context quality.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import load_config

logger = logging.getLogger(__name__)

CONFIG = load_config()
RERANK_CFG = CONFIG.get("reranker", {})

_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_max_length: int = 512


def _ensure_model_loaded() -> None:
    global _model, _tokenizer, _max_length
    if _model is not None:
        return

    model_path = RERANK_CFG.get("local_model_path") or RERANK_CFG.get("model_name", "BAAI/bge-reranker-v2-m3")
    device = RERANK_CFG.get("device", "cuda")
    torch_dtype_cfg = RERANK_CFG.get("torch_dtype", "auto")
    _max_length = int(RERANK_CFG.get("max_length", 512))

    if torch_dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_cfg == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    logger.info("Loading reranker model: %s", model_path)
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    ).eval()


@torch.no_grad()
def _score_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Tokenize and score (query, document) pairs in batches."""
    all_scores: list[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=_max_length,
            return_tensors="pt",
        )
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        scores = torch.sigmoid(_model(**inputs).logits.squeeze(-1))
        all_scores.extend(scores.cpu().tolist())
    return all_scores


def rerank(
    query: str,
    doc_list: list[tuple[str, str]],
    top_k: Optional[int] = None,
    instruction: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> list[tuple[str, str, float]]:
    """
    Rerank (chunk_id, text) pairs by relevance to the query.

    Args:
        query: User query.
        doc_list: List of (chunk_id, document_text) from first-stage retrieval.
        top_k: Return only top_k results; None means return all in order.
        instruction: Unused, kept for interface compatibility.
        batch_size: Override config batch_size for scoring.

    Returns:
        List of (chunk_id, text, score) sorted by score descending.
    """
    if not query or not doc_list:
        return []

    _ensure_model_loaded()

    batch_sz = batch_size or int(RERANK_CFG.get("batch_size", 8))
    pairs = [(query, text) for _, text in doc_list]
    scores = _score_pairs(pairs, batch_sz)

    out = [(chunk_id, text, score) for (chunk_id, text), score in zip(doc_list, scores)]
    out.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        out = out[:top_k]
    return out


def is_enabled() -> bool:
    """Return True if reranker is enabled in config."""
    return bool(RERANK_CFG.get("enabled", False))


def unload_reranker() -> None:
    """Release the reranker model and tokenizer from GPU memory."""
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Unloaded reranker and freed GPU cache.")
