import os
import logging
from src.utils import load_config
from dotenv import load_dotenv

load_dotenv()
config = load_config()
logger = logging.getLogger(__name__)

ANSWER_PROMPT_ZH = """\
根据以下编号参考内容，简洁准确地回答问题。只使用参考内容中的信息，不要添加额外知识。
回答时在使用某条参考内容的句子末尾加上对应编号，格式为 [1]、[2] 等。

{context}

问题：{question}

回答："""

ANSWER_PROMPT_EN = """\
Based on the following numbered context passages, answer the question concisely and accurately. \
Only use information from the context provided. \
After each sentence that uses a specific passage, cite it inline as [1], [2], etc.

{context}

Question: {question}

Answer:"""


def _detect_lang(text: str) -> str:
    if not text:
        return "en"
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh / len(text) > 0.15 else "en"


def _build_prompt(question: str, contexts: list[str]) -> str:
    # Number each context so the LLM can cite them inline
    numbered = "\n\n".join(
        f"[{i}] {c.strip()}" for i, c in enumerate(contexts, 1) if c.strip()
    )
    lang = _detect_lang(question)
    return (ANSWER_PROMPT_ZH if lang == "zh" else ANSWER_PROMPT_EN).format(
        context=numbered[:3000], question=question
    )


def _generate_api(prompt: str, history: list[dict] | None = None) -> str:
    from openai import OpenAI
    llm_cfg = config.get("llm", {})
    api_cfg = llm_cfg.get("api", {})
    api_key = os.getenv("MODELSCOPE_API_KEY") or api_cfg.get("api_key", "")
    base_url = api_cfg.get("base_url", "https://api-inference.modelscope.cn/v1")
    model = api_cfg.get("model", "Qwen/Qwen3-8B")
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content.strip()


def generate_answer(
    question: str,
    contexts: list[str],
    mode: str | None = None,
    history: list[dict] | None = None,
) -> str:
    """Generate an answer from retrieved contexts.

    Args:
        mode: "api" | "local" | None (None reads from config llm.mode)
        history: list of {"role": "user"/"assistant", "content": "..."} for multi-turn
    """
    if mode is None:
        mode = config.get("llm", {}).get("mode", "local")

    prompt = _build_prompt(question, contexts)

    if mode == "api":
        return _generate_api(prompt, history=history)

    from scripts.generate_qa_from_chunks import _generate_with_backend
    return _generate_with_backend(prompt, backend="local", api_client=None)
