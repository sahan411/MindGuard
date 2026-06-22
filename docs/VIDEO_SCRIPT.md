# MindGuard — Video Presentation Script (5 Minutes)

> **Format**: 5 minutes max | 4 team members | Cover: problem, approach, results, demo
> **Tip**: Record each section separately and edit together. Use screen recordings for the demo.

---

## 🎬 Member 1 — Problem & Motivation (1:00)

### Slide 1: Title (10s)
> "Hi, we're Team [Name], and this is **MindGuard** — an AI-powered mental health support prototype."

### Slide 2: The Problem (25s)
> "Mental health is a growing global crisis. According to the WHO, one in four people will be affected by mental health conditions at some point. Yet, access to immediate, empathetic support is severely limited — especially in regions like Sri Lanka where professional resources are scarce."
>
> "The challenge: can AI help bridge this gap — not as a replacement for professionals, but as a **first responder** that recognizes emotions, detects crisis risk, and generates safe, empathetic responses?"

### Slide 3: What MindGuard Does (25s)
> "MindGuard is a three-stage AI pipeline:
> 1. **Emotion Classification** — identifies what you're feeling from text
> 2. **Crisis Detection** — flags high-risk situations for immediate intervention
> 3. **Empathetic Response** — generates supportive, context-aware replies
>
> Importantly, MindGuard is a research prototype — it does **not** provide clinical diagnosis."

---

## 🎬 Member 2 — AI Techniques & Architecture (1:30)

### Slide 4: AI Techniques Used (30s)
> "We implemented **five advanced AI techniques** from the course:
> 1. **NLP** — text preprocessing with normalization, TF-IDF vectorization
> 2. **Transformer-based model** — DistilBERT fine-tuned on GoEmotions dataset for 28-label multi-label emotion classification
> 3. **Generative AI** — a Variational Autoencoder for anomaly-based crisis detection
> 4. **Transfer learning** — leveraging BERT's pre-trained knowledge
> 5. **Prompt engineering** — zero-shot, few-shot, and chain-of-thought strategies for response generation via Llama-3"

### Slide 5: System Architecture (30s)
> "Here's how data flows through MindGuard:
> - User submits text through our **Gradio interface**
> - The **FastAPI backend** processes it through three services:
>   - The emotion service runs BERT inference with sigmoid outputs for multi-label classification
>   - The crisis service runs both VAE reconstruction error scoring AND keyword matching — a dual detection approach for maximum safety
>   - The prompt builder constructs context-aware prompts and calls Groq's Llama-3 API
> - Results flow back to the UI with safety disclaimers and crisis resources"

### Slide 6: Technical Stack (30s)
> "Our tech stack: Python 3.11, PyTorch and HuggingFace Transformers for model training, FastAPI for the backend, Gradio for the UI, scikit-learn for TF-IDF and evaluation metrics, and MLflow for experiment tracking.
>
> All code follows professional standards — PEP 8, type hints, docstrings, Black formatting, and Ruff linting. We have **52 automated tests** covering all services and API endpoints."

---

## 🎬 Member 3 — Results & Evaluation (1:30)

### Slide 7: Emotion Classification Results (30s)
> "For emotion classification, we fine-tuned DistilBERT on Google's GoEmotions dataset — 58,000 Reddit comments across 28 emotion labels.
>
> Results: **Micro-F1 of 0.575** and **Macro-F1 of 0.416** on the test set. The micro-F1 reflects strong performance on frequent emotions like admiration and gratitude, while the lower macro-F1 is expected — rare emotions like grief and pride have very few training examples.
>
> These numbers are competitive with published GoEmotions baselines."

### Slide 8: Crisis Detection — VAE vs Keyword (40s)
> "For crisis detection, we trained a VAE only on non-crisis text, using TF-IDF features. The idea: crisis text should have higher reconstruction error because the model hasn't learned those patterns.
>
> However, we found that the **standalone VAE achieved only 3.2% recall**. This is because crisis language often shares vocabulary with general negative text — the TF-IDF representation doesn't capture the semantic distinction well enough.
>
> The **keyword baseline achieved 35% recall** — better, but still limited.
>
> Our **key insight**: rather than picking one method, we implemented a **dual detection system** — flagging crisis if EITHER the VAE OR keywords trigger. This is a deliberate recall-oriented safety policy: we'd rather have false positives than miss a real crisis."

### Slide 9: Response Generation (20s)
> "For response generation, we tested three prompt engineering strategies:
> - **Zero-shot**: direct empathetic response
> - **Few-shot**: response guided by example patterns
> - **Chain-of-thought**: internal emotional reasoning before responding
>
> All strategies include safety constraints — crisis-positive responses always include professional help resources."

---

## 🎬 Member 4 — Live Demo & Conclusion (1:00)

### Slide 10: Live Demo (30s)
> *[Screen recording of Gradio UI]*
>
> "Let me show MindGuard in action. I'll type: *'I've been feeling really low lately and I don't see the point anymore.'*
>
> The system detects **sadness** as the top emotion, flags a **potential crisis**, and generates a supportive response with crisis resources. Notice the safety disclaimer at the top."
>
> *[Show a non-crisis example too]*
>
> "And for a non-crisis input like *'I had a great day at work today'* — it correctly identifies **joy** with no crisis flag."

### Slide 11: Challenges & Lessons (15s)
> "Key challenges: the VAE's low recall taught us that anomaly detection isn't a silver bullet — domain-specific features matter. The dual detection approach was our solution, and it's a genuine lesson in responsible AI design."

### Slide 12: Conclusion (15s)
> "MindGuard demonstrates that combining multiple AI techniques — transformers, generative models, and prompt engineering — can create a meaningful prototype for mental health support. It's not a replacement for professionals, but a step toward accessible, empathetic AI.
>
> Thank you."

---

## 📋 Production Notes

| Item | Recommendation |
|---|---|
| **Recording** | Each member records their section separately; edit together |
| **Slides** | Use clean, minimal slides with key visuals (architecture diagram, metrics tables, UI screenshots) |
| **Demo** | Pre-record the Gradio demo to avoid live API failures |
| **Time check** | Member 1: 60s, Member 2: 90s, Member 3: 90s, Member 4: 60s = **5:00 total** |
| **Transitions** | Brief "Thank you [Name], now [Name] will cover..." at handoffs |
