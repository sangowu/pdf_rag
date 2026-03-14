"""
Agentic RAG: LLM decides whether to call search_documents tool or answer directly.

Flow:
    1. Send query + tools definition to LLM
    2. If LLM calls search_documents → run RAG retrieval → send contexts back → final answer
    3. If LLM answers directly → return immediately (no retrieval)
"""

import json
import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from src.utils import load_config

load_dotenv()
config = load_config()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document knowledge base to answer questions about specific facts, "
                "document content, exam problems, or domain knowledge. "
                "Use this when the question requires looking up information from documents. "
                "Do NOT use this for general knowledge, math calculations, or conversation tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to retrieve relevant document passages",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    used_rag: bool                        # Whether retrieval was triggered
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _get_client() -> tuple[OpenAI, str]:
    llm_cfg = config.get("llm", {})
    api_cfg = llm_cfg.get("api", {})
    api_key = os.getenv("MODELSCOPE_API_KEY") or api_cfg.get("api_key", "")
    base_url = api_cfg.get("base_url", "https://api-inference.modelscope.cn/v1")
    model = api_cfg.get("model", "Qwen/Qwen3-8B")
    return OpenAI(base_url=base_url, api_key=api_key), model


def run_agent(
    question: str,
    vs,                          # VectorStore instance
    top_k: int = 5,
    history: list[dict] | None = None,
) -> AgentResult:
    """
    Run one agentic RAG turn.

    Args:
        question:  User's question
        vs:        VectorStore instance (pre-loaded)
        top_k:     Number of chunks to retrieve if RAG is triggered
        history:   Previous conversation turns [{role, content}]

    Returns:
        AgentResult with answer, used_rag flag, and optional contexts/sources
    """
    client, model = _get_client()
    messages = list(history or []) + [{"role": "user", "content": question}]

    # ── Step 1: Ask LLM whether to use search_documents ──────────────────────
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
        temperature=0.7,
        extra_body={"enable_thinking": False},
    )
    msg = response.choices[0].message

    # ── Step 2a: LLM chose NOT to call a tool → direct answer ────────────────
    if not msg.tool_calls:
        answer = (msg.content or "").strip()
        logger.info("Agent: direct answer (no retrieval)")
        return AgentResult(answer=answer, used_rag=False)

    # ── Step 2b: LLM called search_documents → run RAG ───────────────────────
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    search_query = args.get("query", question)
    logger.info("Agent: search_documents(%r)", search_query)

    # Retrieval
    result = vs.search_by_text(search_query, k=top_k)
    contexts: list[str] = result["documents"][0]
    raw_metas: list[dict] = (result.get("metadatas") or [[]])[0]
    sources = [
        {
            "file_name": m.get("file_name", ""),
            "page_index": int(m.get("page_index", 0)),
            "chunk_index": int(m.get("chunk_index", 0)),
        }
        for m in raw_metas
    ]

    # Build numbered context for citation
    numbered = "\n\n".join(
        f"[{i}] {c.strip()}" for i, c in enumerate(contexts, 1) if c.strip()
    )

    # ── Step 3: Send tool result back → final answer ──────────────────────────
    messages_with_tool = messages + [
        msg,   # assistant message with tool_calls
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": numbered[:3000],
        },
    ]
    final_response = client.chat.completions.create(
        model=model,
        messages=messages_with_tool,
        max_tokens=512,
        temperature=0.7,
        extra_body={"enable_thinking": False},
    )
    answer = (final_response.choices[0].message.content or "").strip()
    logger.info("Agent: RAG answer generated")

    return AgentResult(answer=answer, used_rag=True, contexts=contexts, sources=sources)
