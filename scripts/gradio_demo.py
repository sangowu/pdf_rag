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


def chat(api_url: str, question: str, history: list, top_k: int, agent_mode: bool):
    """Called on each user message. Returns (history, sources_md, latency_md)."""
    if not question.strip():
        return history, "", ""

    api_history = [{"role": m["role"], "content": m["content"]} for m in history]
    endpoint = "/agent/query" if agent_mode else "/query"

    try:
        resp = requests.post(
            f"{api_url}{endpoint}",
            json={"question": question.strip(), "top_k": top_k, "history": api_history},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        history.append({"role": "assistant", "content": "❌ 无法连接到 API 服务，请确认 FastAPI 已启动。"})
        return history, "", ""
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 请求失败：{e}"})
        return history, "", ""

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    answer_html = _render_answer(answer, sources)

    # Agent mode: show whether RAG was triggered
    if agent_mode:
        used_rag = data.get("used_rag", False)
        tag = "🔍 已检索文档" if used_rag else "💬 直接回答（未检索）"
        answer_html = f"<div style='color:gray;font-size:0.8em;margin-bottom:4px'>{tag}</div>" + answer_html
        latency_md = f"**总计**: {data.get('latency_ms', 0):.0f} ms | Agent 模式"
    else:
        lat = data.get("latency", {})
        meta = data.get("metadata", {})
        latency_md = (
            f"| 阶段 | 耗时 |\n|---|---|\n"
            f"| 检索 | {lat.get('retrieve_ms', 0):.0f} ms |\n"
            f"| 答案生成 | {lat.get('generate_ms', 0):.0f} ms |\n"
            f"| **总计** | **{lat.get('total_ms', 0):.0f} ms** |\n\n"
            f"策略: `{meta.get('chunk_strategy','')}` | LLM: `{meta.get('llm_mode','')}`"
        )

    history.append({"role": "assistant", "content": answer_html})
    return history, _sources_md(sources), latency_md


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
                agent_toggle = gr.Checkbox(
                    label="Agent 模式（自动判断是否检索）",
                    value=False,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="检索 Top-K",
                )
                latency_output = gr.Markdown(label="延迟")
                sources_output = gr.Markdown(label="来源")

        # Gradio 6.0: Chatbot natively uses [{role, content}] messages format
        history_state = gr.State([])

        def on_submit(question, history, top_k, agent_mode):
            if not question.strip():
                return history, "", "", history, ""

            history = history + [{"role": "user", "content": question}]
            h, src, lat = chat(api_url, question, history, top_k, agent_mode)
            return h, src, lat, h, ""

        send_btn.click(
            fn=on_submit,
            inputs=[msg_input, history_state, top_k_slider, agent_toggle],
            outputs=[chatbot, sources_output, latency_output, history_state, msg_input],
        )
        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, history_state, top_k_slider, agent_toggle],
            outputs=[chatbot, sources_output, latency_output, history_state, msg_input],
        )
        clear_btn.click(
            fn=lambda: ([], [], "", ""),
            outputs=[chatbot, history_state, sources_output, latency_output],
        )

    return demo




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
