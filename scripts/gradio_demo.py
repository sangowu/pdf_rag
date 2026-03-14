"""
Gradio chat demo for PDF Parser RAG.

Multi-turn chat interface calling FastAPI /query endpoint.
Each assistant message includes inline [N] citation badges with hover tooltips.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    python scripts/gradio_demo.py
    python scripts/gradio_demo.py --api-url http://localhost:8000 --share
"""

import argparse
import re
import requests
import gradio as gr

DEFAULT_API_URL = "http://localhost:8000"

BADGE_STYLE = (
    "display:inline-block;margin-left:2px;padding:0 5px;"
    "background:#e8f0fe;color:#1a73e8;border-radius:4px;"
    "font-size:0.78em;font-weight:600;cursor:help;"
    "border-bottom:1px dashed #1a73e8;"
)


def _badge(index: int, tooltip: str) -> str:
    return f'<span title="{tooltip}" style="{BADGE_STYLE}">[{index}]</span>'


def _render_answer(answer: str, sources: list[dict]) -> str:
    """Replace [N] in answer text with HTML tooltip badges."""
    badge_map = {}
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "未知文件")
        page = src.get("page_index", "?")
        chunk = src.get("chunk_index", "?")
        badge_map[i] = _badge(i, f"{file_name} | 第 {page} 页 | chunk #{chunk}")

    def _replace(m):
        n = int(m.group(1))
        return badge_map.get(n, m.group(0))

    return re.sub(r"\[(\d+)\]", _replace, answer)


def _sources_md(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["**来源**\n"]
    for i, src in enumerate(sources, 1):
        lines.append(f"**[{i}]** `{src.get('file_name','')}` 第 {src.get('page_index','?')} 页 chunk #{src.get('chunk_index','?')}")
    return "\n\n".join(lines)


def chat(api_url: str, question: str, history: list, top_k: int):
    """Called on each user message. Yields (history, sources_md, latency_md) incrementally."""
    if not question.strip():
        yield history, "", ""
        return

    # Convert Gradio history [{role, content}] to API format
    api_history = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        resp = requests.post(
            f"{api_url}/query",
            json={"question": question.strip(), "top_k": top_k, "history": api_history},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        history.append({"role": "assistant", "content": "❌ 无法连接到 API 服务，请确认 FastAPI 已启动。"})
        yield history, "", ""
        return
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 请求失败：{e}"})
        yield history, "", ""
        return

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    lat = data.get("latency", {})
    meta = data.get("metadata", {})

    answer_html = _render_answer(answer, sources)
    history.append({"role": "assistant", "content": answer_html})

    latency_md = (
        f"| 阶段 | 耗时 |\n|---|---|\n"
        f"| 检索 | {lat.get('retrieve_ms', 0):.0f} ms |\n"
        f"| 答案生成 | {lat.get('generate_ms', 0):.0f} ms |\n"
        f"| **总计** | **{lat.get('total_ms', 0):.0f} ms** |\n\n"
        f"策略: `{meta.get('chunk_strategy','')}` | LLM: `{meta.get('llm_mode','')}`"
    )

    yield history, _sources_md(sources), latency_md


def build_demo(api_url: str) -> gr.Blocks:
    # Gradio 6.0: theme moved to launch(); Chatbot uses tuple format [(user, assistant)]
    with gr.Blocks(title="PDF Parser RAG") as demo:
        gr.Markdown("# PDF Parser RAG")
        gr.Markdown("基于 OmniDocBench 的多轮检索增强问答")

        with gr.Row():
            # Left: chat
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="对话", height=560)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入问题，按 Enter 发送...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

            # Right: metadata panel
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="检索 Top-K",
                )
                latency_output = gr.Markdown(label="延迟")
                sources_output = gr.Markdown(label="来源")

        # history_state: list of {"role": ..., "content": ...} for API
        # chatbot_history: list of (user_str, assistant_str) tuples for display
        history_state = gr.State([])

        def on_submit(question, history, top_k):
            if not question.strip():
                # Convert history to display tuples
                display = _to_tuples(history)
                return display, "", "", history, ""

            # Append user turn to API history
            history = history + [{"role": "user", "content": question}]

            for h, src, lat in chat(api_url, question, history, top_k):
                pass

            display = _to_tuples(h)
            return display, src, lat, h, ""

        send_btn.click(
            fn=on_submit,
            inputs=[msg_input, history_state, top_k_slider],
            outputs=[chatbot, sources_output, latency_output, history_state, msg_input],
        )
        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, history_state, top_k_slider],
            outputs=[chatbot, sources_output, latency_output, history_state, msg_input],
        )
        clear_btn.click(
            fn=lambda: ([], [], "", ""),
            outputs=[chatbot, history_state, sources_output, latency_output],
        )

    return demo


def _to_tuples(history: list[dict]) -> list[tuple]:
    """Convert [{role, content}] to [(user, assistant)] tuples for gr.Chatbot."""
    tuples = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            user_msg = history[i]["content"]
            asst_msg = history[i + 1]["content"] if i + 1 < len(history) else None
            tuples.append((user_msg, asst_msg))
            i += 2
        else:
            i += 1
    return tuples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_demo(args.api_url).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
