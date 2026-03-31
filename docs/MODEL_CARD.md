# MindGuard Model Card

Version: 0.1 (Draft)
Date: 2026-03-29
Status: In Progress

## 1. Model Overview
MindGuard is a multi-component AI prototype for emotional understanding, crisis-risk detection, and safe response support in text conversations.

Primary components:
- Emotion Model: BERT-based multi-label classifier (GoEmotions labels)
- Crisis Model: VAE-based anomaly detector (with keyword baseline comparator)
- Response Model: Prompted LLM response generation via Groq API

## 2. Intended Use
Intended use:
- Academic demonstration of NLP pipelines for emotional context and risk-aware response support
- Research evaluation of classification, anomaly detection, and prompting strategies

Intended users:
- Course team, instructors, and evaluators
- Research-focused developers

Out-of-scope use:
- Clinical diagnosis or treatment
- Emergency intervention replacement
- Autonomous high-stakes decision making

## 3. Safety and Human Oversight
- This system is non-clinical and must present a safety disclaimer in UI and API surfaces.
- Crisis-positive outputs must include clear professional-help guidance.
- Human oversight is required for interpretation and action.
- Raw sensitive text should not be stored by default.

## 4. Training Data
Emotion model:
- Dataset: GoEmotions (English, Reddit-sourced)
- Task: Multi-label emotion classification

Crisis model:
- Dataset: Course-approved crisis/non-crisis text set
- Training policy: VAE trained on non-crisis samples only

Known data constraints:
- Language and cultural bias risks
- Domain mismatch between training and real-world inputs
- Label ambiguity for multi-emotion expressions

## 5. Preprocessing
- Text normalization (case, whitespace, punctuation handling)
- Tokenization aligned with BERT tokenizer
- Optional lemmatization/cleanup for auxiliary features

All preprocessing steps must be versioned and reproducible.

## 6. Evaluation
Emotion metrics:
- Micro-F1
- Macro-F1
- Per-label precision/recall/F1

Crisis metrics:
- Precision, recall, F1
- Threshold rationale and validation distribution evidence
- Baseline comparison against keyword detector

Response metrics:
- BLEU/ROUGE (supporting metrics)
- Human rubric for empathy, relevance, and safety

## 7. Threshold and Decision Policy
- Thresholds must be tuned on validation only (never on test data).
- Threshold changes must be logged with reason and experiment ID.
- Crisis operating point should prioritize recall to reduce missed high-risk cases.

## 8. Ethical Considerations
- Potential harms: false negatives, false positives, user over-reliance
- Fairness concerns: demographic and cultural language variance
- Privacy concerns: accidental retention of sensitive text

Mitigations:
- Explicit limitations in report and UI
- Safety-first messaging for crisis cases
- Conservative release posture (academic prototype only)

## 9. Operational Constraints
- External dependency: Groq API availability and rate limits
- Required fallback: safe local response template when API fails
- Reproducibility requirement: MLflow tracking for all key experiments

## 10. Artifacts and Versioning
Record final artifacts here before submission:
- Emotion model file: `models/bert_emotion/smoke_best_model_tiny`
- Crisis model file: `models/vae_crisis_smoke/vae_state_dict.pt`
- Tokenizer/config versions:
	- Emotion tokenizer: DistilBERT tokenizer saved under `models/bert_emotion/smoke_best_model_tiny`
	- VAE vectorizer vocabulary: `models/vae_crisis_smoke/tfidf_vocabulary.json`
- Dataset snapshot IDs:
	- GoEmotions (simplified): `google-research-datasets/go_emotions`
	- AMOD counseling conversations: `Amod/mental_health_counseling_conversations`
- MLflow run IDs:
	- Emotion run: `664db6558d2648ccb5481fab8848b57b`
	- Crisis run: logged under `mindguard-vae-crisis` (see `data/processed/vae_threshold_summary_smoke.json`)

## 11. Results Summary (To Fill)
- Final emotion model metrics (current smoke run):
	- Validation micro-F1: 0.0947
	- Validation macro-F1: 0.0319
	- Test micro-F1: 0.1201
	- Test macro-F1: 0.0375
- Final crisis model metrics (current smoke run):
	- Precision: 0.0000
	- Recall: 0.0000
	- F1: 0.0000
	- Threshold percentile/value: 95.0 / 0.001951873884536326
- Baseline comparison summary (current smoke run):
	- VAE baseline: precision=0.0, recall=0.0, f1=0.0
	- Keyword baseline: precision=0.0, recall=0.0, f1=0.0
	- Explanation: this run used a non-crisis-only source (`data/raw/amod_mh_counseling/train.jsonl`) and therefore does not yet provide representative crisis discrimination performance.
- Response quality summary:
	- Prompt/orchestration behavior validated via fallback and crisis guidance tests.
	- Human rubric scoring still pending final integrated run.

## 13. Current Evidence Scope
- Current metrics above are from smoke artifacts to validate pipeline reproducibility and integration paths.
- Before final submission, rerun crisis evaluation on a dataset containing positive crisis labels and update this card with final non-smoke metrics.

## 12. Approval and Sign-off
Team members:
- Member 1:
- Member 2:

Final verification checklist:
- [ ] Safety disclaimer active in UI/API
- [ ] Crisis guidance active on positive detections
- [ ] Metrics reproducible from logged runs
- [ ] Limitations and ethics clearly documented
