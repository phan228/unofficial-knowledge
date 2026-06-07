from __future__ import annotations

import gradio as gr

from query import ask


def handle_query(question: str) -> tuple[str, str]:
    result = ask(question)
    source_lines = result.get("sources", [])
    sources = "\n".join(f"• {source}" for source in source_lines) if source_lines else "• No sources retrieved."
    return str(result.get("answer", "")), sources


with gr.Blocks(title="GWU Unofficial Guide") as demo:
    gr.Markdown("# GWU Unofficial Guide\nAsk a question about GWU using only the retrieved documents.")
    inp = gr.Textbox(label="Your question", placeholder="Which CS professors are worth taking?")
    btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=6)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()