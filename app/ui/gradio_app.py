import gradio as gr


def respond(text: str) -> str:
    return f"MindGuard received: {text}"


if __name__ == "__main__":
    demo = gr.Interface(fn=respond, inputs="text", outputs="text", title="MindGuard")
    demo.launch()
