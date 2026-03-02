"""
Generate one QA pair per chunk from all_chunks.json using LLM.
Output: data/answers/qa_pairs.jsonl (qid, question, answer, chunk_id, file_name, page_index).
"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.utils import load_config

load_dotenv()

CONFIG = load_config()
ALL_CHUNK_PATH = CONFIG.get("paths", {}).get("all_chunk_path", "results/chunk_results/all_chunks.json")
QA_OUTPUT_PATH = "data/answers/qa_pairs.jsonl"
MODEL_ID = "Qwen/Qwen3-8B"

QA_PROMPT = """Based on the following passage, generate exactly one question-answer pair.
Requirements:
- The answer MUST be a continuous substring of the passage (copy from the passage).
- Output valid JSON only, no other text: {"question": "...", "answer": "..."}

Passage:
---
{text}
---"""


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def call_llm(client: OpenAI, text: str) -> str:
    if not text or not text.strip():
        return ""
    prompt = QA_PROMPT.format(text=text.strip()[:4000])
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        extra_body={"enable_thinking": False},
    )
    return (response.choices[0].message.content or "").strip()


def parse_qa(raw: str) -> tuple[str, str]:
    """Parse JSON or fallback to simple extraction. Returns (question, answer)."""
    raw = raw.strip()
    # Strip markdown code block if present
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Try parse from first { to last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return (obj.get("question", ""), obj.get("answer", ""))
        except json.JSONDecodeError:
            pass
    # Fallback: simple key-value extraction
    q = re.search(r'"question"\s*:\s*"([^"]*)"', raw)
    a = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
    return (q.group(1) if q else "", a.group(1) if a else "")


def main():
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set QWEN_API_KEY or OPENAI_API_KEY in .env")
    client = OpenAI(
        base_url="https://api-inference.modelscope.cn/v1",
        api_key=api_key,
    )
    chunks = load_chunks(ALL_CHUNK_PATH)
    if not chunks:
        print("No chunks found.")
        return
    os.makedirs(os.path.dirname(QA_OUTPUT_PATH), exist_ok=True)
    qid = 0
    with open(QA_OUTPUT_PATH, "w", encoding="utf-8") as out:
        for chunk in tqdm(chunks, desc="Generating QA"):
            text = chunk.get("text", "")
            if not text or not text.strip():
                continue
            raw = call_llm(client, text)
            question, answer = parse_qa(raw)
            if not question or not answer:
                continue
            qid += 1
            record = {
                "qid": f"qa_{qid}",
                "question": question,
                "answer": answer,
                "chunk_id": chunk.get("chunk_id", ""),
                "file_name": chunk.get("file_name", ""),
                "page_index": chunk.get("page_index", 0),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {qid} QA pairs to {QA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
