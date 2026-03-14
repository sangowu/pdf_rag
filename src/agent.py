"""
Agentic RAG with multi-tool support and agentic loop.

Tools available to the LLM:
  - rewrite_query:    rewrite an ambiguous/vague query into a clearer one
  - search_documents: retrieve relevant chunks from the knowledge base

Flow (LLM decides):
  query → [rewrite_query?] → [search_documents?] → final answer
                ↑ agentic loop: keep executing until no more tool calls ↑

Examples:
  "这道题答案" → rewrite_query → search_documents → answer with citations
  "中国碳中和目标" → search_documents → answer with citations
  "1+1等于多少"    → (no tools) → direct answer
  "翻译成英文"     → (no tools) → direct answer
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

MAX_TOOL_ROUNDS = 5   # safety limit for agentic loop

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rewrite_query",
            "description": (
                "Rewrite a vague, ambiguous, or incomplete query into a clearer, "
                "more specific search query. Use this when the user's question contains "
                "unclear references (e.g. '这道题', '它', '上面说的'), is too short, "
                "or would benefit from expansion before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rewritten": {
                        "type": "string",
                        "description": "The rewritten, clearer version of the query",
                    }
                },
                "required": ["rewritten"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document knowledge base to retrieve relevant passages. "
                "Use this when the question requires specific facts, document content, "
                "exam problems, or domain knowledge from the knowledge base. "
                "Do NOT use for general knowledge, math, or pure conversation tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (use rewritten query if available)",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    used_rag: bool
    rewritten_query: str | None = None   # set if rewrite_query was called
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict, vs, top_k: int) -> tuple[str, dict]:
    """
    Execute a tool call. Returns (tool_result_str, extra_data).
    extra_data carries contexts/sources/rewritten_query for AgentResult.
    """
    if name == "rewrite_query":
        rewritten = args.get("rewritten", "")
        logger.info("Agent: rewrite_query → %r", rewritten)
        return f"Query rewritten to: {rewritten}", {"rewritten_query": rewritten}

    if name == "search_documents":
        query = args.get("query", "")
        logger.info("Agent: search_documents(%r)", query)
        result = vs.search_by_text(query, k=top_k)
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
        numbered = "\n\n".join(
            f"[{i}] {c.strip()}" for i, c in enumerate(contexts, 1) if c.strip()
        )
        return numbered[:3000], {"contexts": contexts, "sources": sources}

    return f"Unknown tool: {name}", {}


# ---------------------------------------------------------------------------
# Agentic loop
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
    vs,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> AgentResult:
    """
    Run agentic RAG with multi-tool loop.

    The LLM can call tools in sequence:
      rewrite_query → search_documents → final answer
    or skip any/all tools and answer directly.
    """
    client, model = _get_client()
    messages = list(history or []) + [{"role": "user", "content": question}]

    # Accumulated state across tool rounds
    extra: dict = {"contexts": [], "sources": [], "rewritten_query": None, "used_rag": False}

    for round_i in range(MAX_TOOL_ROUNDS):
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

        # No tool calls → LLM gave final answer
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            logger.info("Agent: finished in %d round(s), used_rag=%s", round_i + 1, extra["used_rag"])
            return AgentResult(
                answer=answer,
                used_rag=extra["used_rag"],
                rewritten_query=extra["rewritten_query"],
                contexts=extra["contexts"],
                sources=extra["sources"],
            )

        # Append assistant message with tool_calls to history
        messages.append(msg)

        # Execute each tool call and append results
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_result, data = _execute_tool(name, args, vs, top_k)

            # Merge returned data into accumulated state
            if "rewritten_query" in data:
                extra["rewritten_query"] = data["rewritten_query"]
            if "contexts" in data:
                extra["contexts"] = data["contexts"]
                extra["sources"] = data["sources"]
                extra["used_rag"] = True

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    # Safety fallback: exceeded MAX_TOOL_ROUNDS
    logger.warning("Agent: exceeded MAX_TOOL_ROUNDS=%d", MAX_TOOL_ROUNDS)
    return AgentResult(
        answer="抱歉，处理请求时超出了最大步骤限制，请重试。",
        used_rag=extra["used_rag"],
        rewritten_query=extra["rewritten_query"],
        contexts=extra["contexts"],
        sources=extra["sources"],
    )
