from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import gradio as gr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.constants import CRISIS_RESOURCE_TEXT
from app.services.crisis_service import CrisisService
from app.services.emotion_service import EmotionService
from app.services.prompt_builder import build_response

emotion_service = EmotionService()
crisis_service  = CrisisService()

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

body, .gradio-container {
    background: #f9f7ff !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── HERO ── */
.mg-hero {
    background: linear-gradient(135deg, #4f2dc8 0%, #7c3aed 55%, #9d6cf5 100%);
    border-radius: 0 0 28px 28px;
    padding: 0;
    margin-bottom: 1.2rem;
    overflow: hidden;
    position: relative;
}
.mg-hero-circles::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:200px; height:200px; background:rgba(255,255,255,0.06); border-radius:50%;
}
.mg-hero-circles::after {
    content:''; position:absolute; bottom:-80px; left:-40px;
    width:260px; height:260px; background:rgba(255,255,255,0.04); border-radius:50%;
}

/* ── AFFIRMATION BANNER ── */
.mg-affirmation {
    background: rgba(255,255,255,0.1);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding: 0.55rem 1rem;
    text-align: center;
    color: rgba(237,233,254,0.9);
    font-size: 0.8rem;
    font-style: italic;
    font-family: 'DM Serif Display', serif;
    letter-spacing: 0.02em;
    min-height: 2.2rem;
}

/* ── HERO CONTENT ── */
.mg-hero-content {
    padding: 1.6rem 2rem 1.8rem;
    text-align: center;
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}

/* ── BREATHING CIRCLE ── */
.mg-breath {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
}
.mg-breath-circle {
    width: 64px; height: 64px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.3);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase;
    animation: breathe 5s ease-in-out infinite;
    cursor: default;
}
@keyframes breathe {
    0%,100% { transform: scale(1); background: rgba(255,255,255,0.12); box-shadow: 0 0 0 0 rgba(255,255,255,0.15); }
    50% { transform: scale(1.22); background: rgba(255,255,255,0.22); box-shadow: 0 0 0 16px rgba(255,255,255,0.04); }
}
.mg-breath-label {
    color: rgba(221,214,254,0.7); font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── HERO TEXT ── */
.mg-hero-text { flex: 1; min-width: 240px; }
.mg-hero-text h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important; font-weight: 400 !important;
    color: white !important; margin: 0 0 0.35rem !important;
    letter-spacing: -0.01em !important;
}
.mg-hero-text h1 span { color: #c4b5fd; }
.mg-hero-text .greeting {
    color: rgba(237,233,254,0.85);
    font-size: 0.95rem; margin: 0 0 0.9rem;
    font-family: 'DM Serif Display', serif; font-style: italic;
}
.mg-tags { display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:center; }
.mg-tag {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px; color: white;
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.07em; padding: 0.28rem 0.85rem;
    text-transform: uppercase;
}

/* ── NATURE IMAGE STRIP ── */
.mg-nature {
    height: 80px;
    background: linear-gradient(180deg,
        rgba(79,45,200,0.0) 0%,
        rgba(79,45,200,0.0) 100%),
        url('https://images.unsplash.com/photo-1501854140801-50d01698950b?w=1200&q=60') center/cover no-repeat;
    position: relative;
}
.mg-nature::after {
    content:'';
    position:absolute; bottom:0; left:0; right:0; height:40px;
    background: linear-gradient(180deg, transparent, #f9f7ff);
}

/* ── SAFETY ── */
.mg-safety {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    border-radius: 8px; padding: 0.6rem 1rem;
    margin: 0 0 1rem; color: #78350f;
    font-size: 0.78rem; line-height: 1.55;
}

/* ── CARD ── */
.mg-card {
    background: white !important;
    border-radius: 18px !important;
    border: 1px solid #ede9fe !important;
    box-shadow: 0 2px 20px rgba(79,45,200,0.07) !important;
    padding: 1.3rem 1.4rem !important;
    margin-bottom: 1rem !important;
}
.mg-card label, .mg-card .label-wrap span {
    font-size: 0.7rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: #7c3aed !important;
}
.mg-card textarea {
    border: 1.5px solid #ede9fe !important; border-radius: 12px !important;
    background: #fafaff !important; color: #1e1b4b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important; line-height: 1.75 !important;
    padding: 0.9rem 1rem !important; transition: all 0.2s !important;
}
.mg-card textarea:focus {
    border-color: #7c3aed !important; background: white !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,0.08) !important;
}
.mg-card textarea::placeholder {
    color: #a78bfa !important; font-style: italic !important; font-size: 0.92rem !important;
}

/* ── BUTTON ── */
button.primary {
    background: linear-gradient(135deg, #4f2dc8, #7c3aed) !important;
    border: none !important; border-radius: 14px !important;
    color: white !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important; font-weight: 600 !important;
    padding: 0.85rem !important; width: 100% !important;
    box-shadow: 0 4px 18px rgba(79,45,200,0.32) !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 24px rgba(79,45,200,0.42) !important;
}

/* ── HINT ── */
.mg-hint { color:#a78bfa; font-size:0.72rem; text-align:center; margin:0.35rem 0 0.65rem; }

/* ── RESPONSE ── */
.mg-response textarea {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1rem !important; line-height: 1.9 !important;
    color: #2d2460 !important;
    background: linear-gradient(135deg, #faf5ff, #f0f4ff) !important;
    border: 1px solid #e0e7ff !important; border-radius: 12px !important;
    padding: 1.1rem !important;
}

/* ── DIVIDER ── */
.mg-divider {
    display:flex; align-items:center; gap:0.7rem;
    margin: 0.4rem 0 0.8rem; color:#a78bfa;
    font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.13em;
}
.mg-divider::before, .mg-divider::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, transparent, #ede9fe, transparent);
}

/* ── TABLE ── */
.mg-card table { width:100% !important; border-collapse:collapse !important; }
.mg-card th {
    background:#f5f3ff !important; color:#6d28d9 !important;
    font-size:0.68rem !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.08em !important;
    padding:0.5rem 0.7rem !important; border-bottom:2px solid #ede9fe !important; border-top:none !important;
}
.mg-card td {
    color:#3730a3 !important; font-size:0.88rem !important;
    padding:0.48rem 0.7rem !important; border-bottom:1px solid #f5f3ff !important;
}
.mg-card tr:last-child td { border-bottom:none !important; }
.mg-card tr:hover td { background:#faf8ff !important; }

/* ── SMALL OUTPUTS ── */
.mg-pill textarea, .mg-pill input {
    background:transparent !important; border:none !important;
    font-weight:600 !important; font-size:0.95rem !important;
    color:#2d2460 !important; padding:0.1rem 0 !important;
}
.mg-ok textarea { background:#f0fdf4 !important; border:1px solid #bbf7d0 !important; border-radius:10px !important; color:#166534 !important; font-weight:600 !important; }
.mg-alert textarea { background:#fff7ed !important; border:1px solid #fed7aa !important; border-radius:10px !important; color:#9a3412 !important; font-weight:600 !important; }
.mg-resources textarea { background:#fff7ed !important; border:1px solid #fed7aa !important; border-radius:10px !important; color:#9a3412 !important; font-size:0.88rem !important; line-height:1.7 !important; font-weight:500 !important; }

/* ── FOOTER ── */
.mg-footer {
    text-align:center; color:#a78bfa; font-size:0.73rem;
    line-height:1.9; border-top:1px solid #ede9fe;
    padding:0.9rem 0 0.4rem; margin-top:0.4rem;
}
.mg-footer strong { color:#7c3aed; }

/* ── JS AFFIRMATION ── */
#mg-affirmation-text { transition: opacity 0.8s ease; }
"""

AFFIRMATIONS = [
    "You are not alone in how you feel.",
    "Your feelings are valid — all of them.",
    "It takes courage to reach out. You've already taken the first step.",
    "Whatever you're carrying right now, you don't have to carry it alone.",
    "This moment will pass. You are stronger than you know.",
    "Asking for support is a sign of strength, not weakness.",
    "You matter. Your wellbeing matters.",
    "There is no rush. Take all the time you need.",
]

HERO_JS = """
<script>
(function() {
    const msgs = """ + str(AFFIRMATIONS) + """;
    let i = 0;
    function rotate() {
        const el = document.getElementById('mg-aff');
        if (!el) return;
        el.style.opacity = 0;
        setTimeout(() => { el.textContent = msgs[i++ % msgs.length]; el.style.opacity = 1; }, 800);
    }
    rotate();
    setInterval(rotate, 5000);

    // Time-based greeting
    const greet = document.getElementById('mg-greet');
    if (greet) {
        const h = new Date().getHours();
        const g = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
        greet.textContent = g + '. How are you feeling today?';
    }
})();
</script>
"""

HERO_HTML = """
<div class="mg-hero">
    <div class="mg-affirmation">
        <span id="mg-aff" style="transition:opacity 0.8s ease;">You are not alone in how you feel.</span>
    </div>
    <div class="mg-hero-content mg-hero-circles">
        <div class="mg-breath">
            <div class="mg-breath-circle">breathe</div>
            <span class="mg-breath-label">Take a breath</span>
        </div>
        <div class="mg-hero-text">
            <h1>Mind<span>Guard</span></h1>
            <p class="greeting" id="mg-greet">A safe space to share how you feel.</p>
            <div class="mg-tags">
                <span class="mg-tag">Emotion-Aware</span>
                <span class="mg-tag">Crisis-Sensitive</span>
                <span class="mg-tag">Always Here</span>
            </div>
        </div>
    </div>
    <div class="mg-nature"></div>
</div>
""" + HERO_JS

def _format_emotions(emotions: List[dict]) -> List[List[str]]:
    icons = {
        "sadness":"😔","joy":"😊","anger":"😠","fear":"😨","love":"❤️",
        "grief":"💔","nervousness":"😰","disappointment":"😞","remorse":"😓",
        "optimism":"🌟","neutral":"😐","surprise":"😲","confusion":"😕",
        "caring":"🤗","desire":"✨","gratitude":"🙏","excitement":"🎉",
        "pride":"😤","relief":"😌","amusement":"😄","annoyance":"😒",
    }
    rows = []
    for entry in emotions[:5]:
        label = str(entry.get("label","unknown"))
        conf  = float(entry.get("confidence", 0.0))
        filled = int(conf * 10)
        bar   = "█" * filled + "░" * (10 - filled)
        icon  = icons.get(label, "💭")
        rows.append([f"{icon}  {label.capitalize()}", f"{int(conf*100)}%", bar])
    return rows

def analyze_text(text: str) -> Tuple[str, List[List[str]], str, str, str]:
    if not text or not text.strip():
        return (
            "Whenever you're ready, I'm here. There's no rush — take your time. 💙",
            [], "—", "✅  No signals detected", "",
        )
    emotion_result  = emotion_service.predict(text)
    crisis_result   = crisis_service.predict(text)
    top_emotion     = str(emotion_result.get("top_emotion", "neutral"))
    crisis_detected = bool(crisis_result.get("crisis_detected", False))
    response_text   = build_response(
        text=text, top_emotion=top_emotion,
        crisis=crisis_detected, strategy=settings.default_response_strategy,
    )
    crisis_label    = "⚠️  Crisis signals detected — please reach out" if crisis_detected else "✅  No crisis signals detected"
    crisis_guidance = CRISIS_RESOURCE_TEXT if crisis_detected else ""
    return (
        response_text,
        _format_emotions(list(emotion_result.get("emotions", []))),
        top_emotion.capitalize(),
        crisis_label,
        crisis_guidance,
    )

_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
)

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MindGuard") as demo:

        gr.HTML(HERO_HTML)

        gr.HTML('<div class="mg-safety">⚠️ <strong>Safety Notice:</strong> MindGuard is a research prototype only — not a clinical tool. If someone is in immediate danger, please contact local emergency services immediately.</div>')

        with gr.Group(elem_classes="mg-card"):
            user_text = gr.Textbox(
                label="What's on your mind?",
                placeholder="This is a safe space. Share what you're feeling — big or small. I'm listening without judgment...",
                lines=4, max_lines=8,
            )
            gr.HTML('<p class="mg-hint">Press Enter to send &nbsp;·&nbsp; Shift+Enter for a new line</p>')
            submit = gr.Button("💙  Analyze & Generate a Supportive Response", variant="primary")

        gr.HTML('<div class="mg-divider">A message for you</div>')

        with gr.Group(elem_classes="mg-card mg-response"):
            response_out = gr.Textbox(label="MindGuard says", lines=6, interactive=False)

        gr.HTML('<div class="mg-divider">Emotional Analysis</div>')

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="mg-card"):
                    emotions_out = gr.Dataframe(
                        headers=["Emotion","Score","Intensity"],
                        datatype=["str","str","str"],
                        label="Detected Emotions",
                        interactive=False, wrap=True,
                        row_count=(5,"fixed"),
                    )
            with gr.Column(scale=2):
                with gr.Group(elem_classes="mg-card mg-pill"):
                    top_emotion_out = gr.Textbox(label="🎯 Primary Emotion", interactive=False, lines=1)
                with gr.Group(elem_classes="mg-card mg-pill mg-ok"):
                    crisis_flag_out = gr.Textbox(label="🛡️ Crisis Assessment", interactive=False, lines=1)

        with gr.Group(elem_classes="mg-card mg-resources"):
            crisis_resource_out = gr.Textbox(
                label="🆘 Support Resources", interactive=False, lines=2,
                placeholder="Support resources appear here if crisis signals are detected.",
            )

        gr.HTML("""
        <div class="mg-footer">
            <strong>Sri Lanka:</strong> Lifeline 1926 &nbsp;·&nbsp; CCCline 1333
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>International:</strong> Crisis Text Line — text HOME to 741741<br>
            MindGuard · Advanced AI Research Project · Not a substitute for professional mental health care
        </div>
        """)

        outputs = [response_out, emotions_out, top_emotion_out, crisis_flag_out, crisis_resource_out]
        submit.click(fn=analyze_text, inputs=[user_text], outputs=outputs)
        user_text.submit(fn=analyze_text, inputs=[user_text], outputs=outputs)

    return demo

if __name__ == "__main__":
    build_demo().launch(
        theme=_THEME, css=CUSTOM_CSS,
        server_name="0.0.0.0", server_port=7860, show_error=True,
    )