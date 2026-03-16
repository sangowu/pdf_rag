import os
import time
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


def _make_api_client():
    from openai import OpenAI
    llm_cfg = config.get("llm", {})
    api_cfg = llm_cfg.get("api", {})
    api_key = os.getenv("MODELSCOPE_API_KEY") or api_cfg.get("api_key", "")
    base_url = api_cfg.get("base_url", "https://api-inference.modelscope.cn/v1")
    model = api_cfg.get("model", "Qwen/Qwen3-8B")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def _generate_api(prompt: str, history: list[dict] | None = None) -> str:
    answer, _ = _generate_api_timed(prompt, history=history)
    return answer


def _generate_api_timed(prompt: str, history: list[dict] | None = None) -> tuple[str, float]:
    """Returns (full_answer, ttft_ms) using internal streaming to capture TTFT."""
    client, model = _make_api_client()
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        extra_body={"enable_thinking": False},
        stream=True,
    )
    t_start = time.perf_counter()
    t_first = None
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(delta)
    ttft_ms = round((t_first - t_start) * 1000, 1) if t_first else 0.0
    return "".join(chunks).strip(), ttft_ms


def _generate_api_stream(prompt: str, history: list[dict] | None = None):
    """Yields text chunks from the LLM via streaming API."""
    client, model = _make_api_client()
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        extra_body={"enable_thinking": False},
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_answer_stream(
    question: str,
    contexts: list[str],
    mode: str | None = None,
    history: list[dict] | None = None,
):
    """Generator: yields answer text chunks token by token.

    Local mode has no token-level streaming — yields the full answer in one chunk.
    """
    if mode is None:
        mode = config.get("llm", {}).get("mode", "local")
    prompt = _build_prompt(question, contexts)
    if mode == "api":
        yield from _generate_api_stream(prompt, history=history)
    else:
        from scripts.generate_qa_from_chunks import _generate_with_backend
        yield _generate_with_backend(prompt, backend="local", api_client=None)


def generate_answer_timed(
    question: str,
    contexts: list[str],
    mode: str | None = None,
    history: list[dict] | None = None,
) -> tuple[str, float]:
    """Returns (answer, ttft_ms).

    API mode: TTFT measured via internal streaming.
    Local mode: TTFT approximated as total generation time (model.generate is blocking).
    """
    if mode is None:
        mode = config.get("llm", {}).get("mode", "local")
    prompt = _build_prompt(question, contexts)
    if mode == "api":
        return _generate_api_timed(prompt, history=history)
    from scripts.generate_qa_from_chunks import _generate_with_backend
    t0 = time.perf_counter()
    answer = _generate_with_backend(prompt, backend="local", api_client=None)
    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)  # no token-level streaming in local
    return answer, ttft_ms


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
    answer, _ = generate_answer_timed(question, contexts, mode=mode, history=history)
    return answer
