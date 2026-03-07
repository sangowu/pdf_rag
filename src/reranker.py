"""
Reranker module using Qwen3-Reranker (causal LM yes/no scoring).

Reranks (query, document) pairs and returns documents ordered by relevance score.
Used after first-stage retrieval to improve Hit@K and RAG context quality.
"""

import logging
from typing import Optional

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_config

logger = logging.getLogger(__name__)

CONFIG = load_config()
RERANK_CFG = CONFIG.get("reranker", {})

DEFAULT_INSTRUCTION = "Judge whether this document passage contains information needed to answer the question. The passage or the answer may be in Chinese or English."

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_prefix_tokens: list[int] = []
_suffix_tokens: list[int] = []
_token_true_id: int = 0
_token_false_id: int = 0
_max_length: int = 8192


def _format_instruction(instruction: str, query: str, doc: str) -> str:
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def _ensure_model_loaded() -> None:
    global _model, _tokenizer, _prefix_tokens, _suffix_tokens, _token_true_id, _token_false_id, _max_length
    if _model is not None:
        return

    model_path = RERANK_CFG.get("local_model_path") or RERANK_CFG.get("model_name", "Qwen/Qwen3-Reranker-4B")
    device = RERANK_CFG.get("device", "cuda")
    torch_dtype_cfg = RERANK_CFG.get("torch_dtype", "auto")
    _max_length = int(RERANK_CFG.get("max_length", 8192))

    if torch_dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_cfg == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    logger.info("Loading reranker model: %s", model_path)
    _tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    token_false_id = _tokenizer.convert_tokens_to_ids("no")
    token_true_id = _tokenizer.convert_tokens_to_ids("yes")
    if token_true_id is None or token_false_id is None:
        raise ValueError("Reranker tokenizer must have 'yes' and 'no' tokens")
    _token_true_id = token_true_id
    _token_false_id = token_false_id

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _prefix_tokens = _tokenizer.encode(prefix, add_special_tokens=False)
    _suffix_tokens = _tokenizer.encode(suffix, add_special_tokens=False)


def _process_inputs(pairs: list[str]) -> dict:
    """Tokenize and pad to longest in batch (dynamic padding). Uses attention_mask so model ignores padding."""
    global _tokenizer, _prefix_tokens, _suffix_tokens, _max_length
    content_max = _max_length - len(_prefix_tokens) - len(_suffix_tokens)
    inputs = _tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=content_max,
    )
    for i in range(len(inputs["input_ids"])):
        inputs["input_ids"][i] = _prefix_tokens + inputs["input_ids"][i] + _suffix_tokens
    # Pad to longest in batch instead of global max_length to reduce compute.
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    inputs = _tokenizer.pad(
        inputs,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    device = next(_model.parameters()).device
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


@torch.no_grad()
def _compute_logits(inputs: dict) -> list[float]:
    global _model, _token_true_id, _token_false_id
    batch_scores = _model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, _token_true_id]
    false_vector = batch_scores[:, _token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()


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
        instruction: Override default instruction for the reranker prompt.
        batch_size: Override config batch_size for scoring.

    Returns:
        List of (chunk_id, text, score) sorted by score descending.
    """
    if not query or not doc_list:
        return []

    _ensure_model_loaded()

    instr = instruction or RERANK_CFG.get("instruction", DEFAULT_INSTRUCTION)
    batch_sz = batch_size or int(RERANK_CFG.get("batch_size", 8))

    pairs = [_format_instruction(instr, query, text) for _, text in doc_list]
    all_scores: list[float] = []

    for i in range(0, len(pairs), batch_sz):
        batch_pairs = pairs[i : i + batch_sz]
        inputs = _process_inputs(batch_pairs)
        batch_scores = _compute_logits(inputs)
        all_scores.extend(batch_scores)

    out = [(chunk_id, text, score) for (chunk_id, text), score in zip(doc_list, all_scores)]
    out.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        out = out[:top_k]
    return out


def is_enabled() -> bool:
    """Return True if reranker is enabled in config."""
    return bool(RERANK_CFG.get("enabled", False))


def unload_reranker() -> None:
    """Release the reranker model and tokenizer from GPU memory. Call after evaluation to free VRAM."""
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
