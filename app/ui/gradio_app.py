from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr

# Allow running via `python app/ui/gradio_app.py` from project root.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.constants import CRISIS_RESOURCE_TEXT, SAFETY_DISCLAIMER_TEXT
from app.services.crisis_service import CrisisService
from app.services.emotion_service import EmotionService
from app.services.prompt_builder import build_response

emotion_service = EmotionService()
crisis_service = CrisisService()


def _format_emotions(emotions: List[dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for entry in emotions:
        label = str(entry.get("label", "unknown"))
        confidence = float(entry.get("confidence", 0.0))
        rows.append([label, f"{confidence:.3f}"])
    return rows


def analyze_text(text: str) -> Tuple[str, List[List[str]], str, str, str]:
    if not text or text.strip() == "":
        return (
            "Please enter your message so MindGuard can analyze it.",
            [],
            "N/A",
            "No",
            CRISIS_RESOURCE_TEXT,
        )

    emotion_result = emotion_service.predict(text)
    crisis_result = crisis_service.predict(text)

    top_emotion = str(emotion_result.get("top_emotion", "neutral"))
    crisis_detected = bool(crisis_result.get("crisis_detected", False))
    strategy = settings.default_response_strategy

    response_text = build_response(
        text=text,
        top_emotion=top_emotion,
        crisis=crisis_detected,
        strategy=strategy,
    )

    crisis_label = "Yes" if crisis_detected else "No"
    crisis_guidance = CRISIS_RESOURCE_TEXT if crisis_detected else ""

    return (
        response_text,
        _format_emotions(list(emotion_result.get("emotions", []))),
        top_emotion,
        crisis_label,
        crisis_guidance,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MindGuard") as demo:
        gr.Markdown("# MindGuard")
        gr.Markdown("Emotion-aware, crisis-aware response support for academic use.")
        gr.Markdown(f"**Safety Notice:** {SAFETY_DISCLAIMER_TEXT}")

        with gr.Column():
            user_text = gr.Textbox(
                label="Your Message",
                placeholder="Share what you are feeling...",
                lines=5,
            )
            submit = gr.Button("Analyze and Generate Response", variant="primary")

        with gr.Row(equal_height=True):
            response_out = gr.Textbox(
                label="Supportive Response",
                lines=6,
                interactive=False,
            )
            emotions_out = gr.Dataframe(
                headers=["Emotion", "Confidence"],
                datatype=["str", "str"],
                label="Detected Emotions",
                interactive=False,
                wrap=True,
            )

        with gr.Row(equal_height=True):
            top_emotion_out = gr.Textbox(label="Top Emotion", interactive=False)
            crisis_flag_out = gr.Textbox(label="Crisis Detected", interactive=False)

        crisis_resource_out = gr.Textbox(
            label="Crisis Guidance",
            interactive=False,
            lines=2,
        )

        submit.click(
            fn=analyze_text,
            inputs=[user_text],
            outputs=[
                response_out,
                emotions_out,
                top_emotion_out,
                crisis_flag_out,
                crisis_resource_out,
            ],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
