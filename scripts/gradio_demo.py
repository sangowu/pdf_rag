"""
Gradio demo for PDF Parser RAG.

Calls FastAPI /query endpoint and displays answer + retrieved contexts + latency.

Usage:
    # Make sure FastAPI is running first:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

    # Then start Gradio:
    python scripts/gradio_demo.py
    python scripts/gradio_demo.py --api-url http://localhost:8000
"""

import argparse
import requests
import gradio as gr

DEFAULT_API_URL = "http://localhost:8000"


def query(api_url: str, question: str, top_k: int):
    if not question.strip():
        return "请输入问题。", "", ""

    try:
        resp = requests.post(
            f"{api_url}/query",
            json={"question": question.strip(), "top_k": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 API 服务，请确认 FastAPI 已启动。", "", ""
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请稍后重试。", "", ""
    except Exception as e:
        return f"❌ 请求失败：{e}", "", ""

    answer = data.get("answer", "")

    # Format retrieved contexts
    contexts = data.get("contexts", [])
    context_md = ""
    for i, ctx in enumerate(contexts, 1):
        context_md += f"**[{i}]** {ctx.strip()}\n\n---\n\n"

    # Format latency
    lat = data.get("latency", {})
    meta = data.get("metadata", {})
    latency_md = (
        f"| 阶段 | 耗时 |\n"
        f"|---|---|\n"
        f"| 检索 | {lat.get('retrieve_ms', 0):.0f} ms |\n"
        f"| 答案生成 | {lat.get('generate_ms', 0):.0f} ms |\n"
        f"| **总计** | **{lat.get('total_ms', 0):.0f} ms** |\n\n"
        f"Chunk 策略: `{meta.get('chunk_strategy', '')}` &nbsp;|&nbsp; "
        f"Collection: `{meta.get('collection', '')}` &nbsp;|&nbsp; "
        f"LLM: `{meta.get('llm_mode', '')}`"
    )

    return answer, context_md, latency_md


def build_demo(api_url: str) -> gr.Blocks:
    with gr.Blocks(title="PDF Parser RAG Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PDF Parser RAG Demo")
        gr.Markdown("基于 OmniDocBench 的检索增强问答系统")

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="问题",
                    placeholder="输入你的问题...",
                    lines=2,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="检索 Top-K",
                )
                submit_btn = gr.Button("提交", variant="primary")

            with gr.Column(scale=1):
                latency_output = gr.Markdown(label="延迟信息")

        answer_output = gr.Textbox(label="答案", lines=4, interactive=False)

        with gr.Accordion("检索到的上下文", open=False):
            context_output = gr.Markdown()

        submit_btn.click(
            fn=lambda q, k: query(api_url, q, k),
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output, latency_output],
        )
        question_input.submit(
            fn=lambda q, k: query(api_url, q, k),
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output, latency_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for PDF Parser RAG.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="FastAPI base URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    demo = build_demo(args.api_url)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
