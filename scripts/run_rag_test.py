"""
Run end-to-end RAG inference on gold questions and write pred_answers.csv.

Pipeline for this script:
- Read gold_answers.csv (question, gold_answer, gold_chunk_ids, ...).
- For each question:
  - Use VectorStore to retrieve top-k relevant chunks from ChromaDB.
  - Concatenate retrieved chunk texts as context.
  - Use the configured LLM (local/api) to generate an answer in Chinese.
- Write results/pred_answers.csv with columns:
  - question, gold_answer, pred_answer, gold_chunk_ids, retrieved_chunk_ids
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from modelscope import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from src.utils import load_config
from src.vector_store import VectorStore


load_dotenv()

logger = logging.getLogger(__name__)

CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
LLM_CONFIG = CONFIG.get("llm", {})
API_LLM_CONFIG = LLM_CONFIG.get("api", {})
LOCAL_LLM_CONFIG = LLM_CONFIG.get("local", {})
DEFAULT_BACKEND = LLM_CONFIG.get("mode", "local")
EVAL_CFG = CONFIG.get("evaluation", {})
RERANK_CFG = CONFIG.get("reranker", {})

GOLD_ANSWERS_CSV = PATHS.get("gold_answers_csv", "data/answers/gold_answers.csv")
PRED_ANSWERS_CSV = PATHS.get("pred_answers_csv", "results/pred_answers.csv")
EVAL_TOP_K = int(EVAL_CFG.get("top_k", 5))
RERANK_ENABLED = bool(RERANK_CFG.get("enabled", False))
RERANK_TOP_R = int(RERANK_CFG.get("top_r", 20))

API_MODEL_ID = API_LLM_CONFIG.get("model", "Qwen/Qwen3-8B")
LOCAL_MODEL_NAME = LOCAL_LLM_CONFIG.get("model_name", "models/Qwen3-8B")

_LOCAL_TOKENIZER: Optional[AutoTokenizer] = None
_LOCAL_MODEL: Optional[AutoModelForCausalLM] = None


def _ensure_local_llm_loaded() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Lazily load local chat LLM for answer generation."""
    global _LOCAL_TOKENIZER, _LOCAL_MODEL
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL

    torch_dtype_cfg = LOCAL_LLM_CONFIG.get("torch_dtype", "auto")
    if torch_dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_cfg == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    logger.info("Loading local LLM model for RAG: %s", LOCAL_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _LOCAL_TOKENIZER = tokenizer
    _LOCAL_MODEL = model
    return tokenizer, model


def _generate_with_backend(prompt: str, backend: str, api_client: Optional[OpenAI]) -> str:
    """Generate completion using either API or local backend."""
    prompt = prompt or ""
    if not prompt.strip():
        return ""

    if backend == "api":
        if api_client is None:
            raise ValueError("API backend selected but api_client is None")
        response = api_client.chat.completions.create(
            model=API_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            extra_body={"enable_thinking": False},
        )
        return (response.choices[0].message.content or "").strip()

    tokenizer, model = _ensure_local_llm_loaded()
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    max_new_tokens = int(LOCAL_LLM_CONFIG.get("max_tokens", 256))
    temperature = float(LOCAL_LLM_CONFIG.get("temperature", 0.7))
    do_sample = bool(LOCAL_LLM_CONFIG.get("do_sample", True))
    top_p = float(LOCAL_LLM_CONFIG.get("top_p", 0.8))
    top_k = int(LOCAL_LLM_CONFIG.get("top_k", 20))
    min_p = float(LOCAL_LLM_CONFIG.get("min_p", 0.0))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if full_text.startswith(chat_text):
        return full_text[len(chat_text) :].strip()
    return full_text.strip()


def build_context_from_result(result: dict) -> tuple[str, list[str]]:
    """Build plain-text context and chunk_id list from a Chroma query result."""
    documents = result.get("documents", [[]])
    ids_per_query = result.get("ids", [[]])
    docs = documents[0] if documents else []
    chunk_ids = ids_per_query[0] if ids_per_query else []
    context = "\n\n".join(docs)
    return context, chunk_ids


def retrieve_for_question(question: str, vs: VectorStore) -> tuple[str, list[str]]:
    """Retrieve context and chunk ids for one question; uses reranker when enabled."""
    if RERANK_ENABLED:
        from src.reranker import rerank
        result = vs.search_by_text(question, k=RERANK_TOP_R)
        ids_list = result.get("ids", [[]])
        docs_list = result.get("documents", [[]])
        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        if not ids or not docs:
            return "", []
        doc_list = list(zip(ids, docs))
        ranked = rerank(question, doc_list, top_k=EVAL_TOP_K)
        texts = [text for _cid, text, _score in ranked]
        chunk_ids = [cid for cid, _text, _score in ranked]
        return "\n\n".join(texts), chunk_ids
    result = vs.search_by_text(question, k=EVAL_TOP_K)
    return build_context_from_result(result)


def build_answer_prompt(question: str, context: str) -> str:
    """Build a Chinese RAG answer-generation prompt."""
    return (
        "你是一个严谨的中文问答助手。\n\n"
        "下面给出用户的问题和检索到的文档片段，请根据文档内容作答。\n\n"
        "要求：\n"
        "- 必须用中文回答。\n"
        "- 回答要尽量简洁准确，只依赖给定文档中的信息。\n"
        "- 如果文档中没有足够信息，请回答“根据给定文档无法确定。”。\n\n"
        f"问题：\n{question}\n\n"
        "文档片段：\n"
        f"{context}\n"
    )


def run_rag_inference(
    gold_csv: str,
    output_csv: str,
    backend: str = "local",
    max_samples: Optional[int] = None,
) -> None:
    """Run retrieval + answer generation over all gold questions and write pred_answers.csv."""
    vs = VectorStore()
    api_client: Optional[OpenAI] = None
    if backend == "api":
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set QWEN_API_KEY or OPENAI_API_KEY in .env for API backend, or use --backend local")
        api_client = OpenAI(
            base_url=API_LLM_CONFIG.get("api_base", "https://api-inference.modelscope.cn/v1"),
            api_key=api_key,
        )

    gold_path = Path(gold_csv)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold answers file not found: {gold_path}")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with gold_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = ["question", "gold_answer", "pred_answer", "gold_chunk_ids", "retrieved_chunk_ids"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            question = (row.get("question") or "").strip()
            gold_answer = (row.get("gold_answer") or "").strip()
            gold_chunk_ids = (row.get("gold_chunk_ids") or "").strip()
            if not question:
                continue

            total += 1
            if max_samples is not None and total > max_samples:
                break

            # Retrieval (optionally: top_r -> rerank -> top_k)
            try:
                context, retrieved_chunk_ids = retrieve_for_question(question, vs)
            except Exception as exc:  # noqa: BLE001
                logger.error("Search failed for question: %s, error: %s", question, exc)
                continue
            if not context.strip():
                pred_answer = "根据给定文档无法确定。"
            else:
                prompt = build_answer_prompt(question, context)
                try:
                    pred_answer = _generate_with_backend(prompt, backend=backend, api_client=api_client)
                except Exception as exc:  # noqa: BLE001
                    logger.error("LLM generation failed for question: %s, error: %s", question, exc)
                    pred_answer = ""

            writer.writerow(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "gold_chunk_ids": gold_chunk_ids,
                    "retrieved_chunk_ids": ",".join(retrieved_chunk_ids),
                }
            )

    logger.info("Wrote %d predicted answers to %s", total, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run RAG inference over gold questions and write pred_answers.csv.")
    parser.add_argument("--gold-csv", default=GOLD_ANSWERS_CSV, help="Path to gold_answers.csv")
    parser.add_argument("--output", default=PRED_ANSWERS_CSV, help="Output CSV path for predicted answers")
    parser.add_argument(
        "--backend",
        choices=("api", "local"),
        default=DEFAULT_BACKEND,
        help="LLM backend: api (OpenAI-compatible HTTP API) or local (ModelScope transformers)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of questions to process (for quick testing)",
    )
    args = parser.parse_args()

    run_rag_inference(
        gold_csv=args.gold_csv,
        output_csv=args.output,
        backend=args.backend,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

