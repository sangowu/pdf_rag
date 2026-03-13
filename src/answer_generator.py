from src.utils import load_config

config = load_config()

ANSWER_PROMPT_ZH = """\
根据以下参考内容，简洁准确地回答问题。只使用参考内容中的信息，不要添加额外知识。

参考内容：
{context}

问题：{question}

回答："""

ANSWER_PROMPT_EN = """\
Based on the following context, answer the question concisely and accurately. \
Only use information from the context provided.

Context:
{context}

Question: {question}

Answer:"""


def _detect_lang(text: str) -> str:
    if not text:
        return "en"
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh / len(text) > 0.15 else "en"


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer from retrieved contexts using the local LLM."""
    context_text = "\n\n".join(c.strip() for c in contexts if c.strip())
    lang = _detect_lang(question)
    prompt = (ANSWER_PROMPT_ZH if lang == "zh" else ANSWER_PROMPT_EN).format(
        context=context_text[:3000], question=question
    )
    from scripts.generate_qa_from_chunks import _generate_with_backend
    return _generate_with_backend(prompt, backend="local", api_client=None)
