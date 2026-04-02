# MindGuard Integration Handoff (B5 -> A6)

Version: 1.0
Date: 2026-03-31
From: Member B (data/train/eval)
To: Member A (API/integration)

## 1. Handoff Objective
Provide a deterministic integration contract for A6 endpoint orchestration using current trained/smoke artifacts and safety behavior requirements.

## 2. Required Artifact Inputs

### Emotion service inputs
- Model directory: `models/bert_emotion/smoke_best_model_tiny`
- Label map: `models/bert_emotion/smoke_best_model_tiny/label_map.json`
- Run summary reference: `data/processed/bert_run_summary.json`

### Crisis service inputs
- Model state: `models/vae_crisis_smoke/vae_state_dict.pt`
- Vectorizer vocabulary: `models/vae_crisis_smoke/tfidf_vocabulary.json`
- Threshold summary: `data/processed/vae_threshold_summary_smoke.json`
- Reconstruction evidence: `models/vae_crisis_smoke/reconstruction_errors.csv`

### Evaluation references
- `data/processed/evaluation/emotion_metrics.json`
- `data/processed/evaluation/crisis_baseline_metrics.json`

## 3. Runtime Configuration Contract
- Keep threshold externalized via settings/constants, never hardcoded in endpoint logic.
- Preserve emotion top-k behavior from existing settings fallback chain.
- Keep Groq API optional with safe local fallback response path.

## 4. Integration Behavior Contract

### Endpoint behavior
- Emotion endpoint:
  - Returns stable sorted emotions list and `top_emotion`.
  - Uses fallback payload when model artifact load fails.
- Crisis endpoint:
  - Returns `crisis_detected`, `reconstruction_error`, `threshold`, `method`, `keyword_match`, `crisis_guidance_required`.
  - Applies recall-oriented policy: VAE signal OR keyword signal triggers crisis guidance.
- Response endpoint:
  - Always includes safety-constrained tone.
  - Appends crisis guidance suffix when crisis is positive.

### Error behavior
- Missing artifact must degrade gracefully to fallback behavior where designed.
- API should return consistent schema even in fallback mode.

## 5. Known Limitations to Carry Forward
- Current metrics are smoke-stage and should not be presented as final effectiveness.
- Crisis smoke run used non-crisis-only source; final crisis evaluation must be rerun on mixed labels.

## 6. A6 Validation Checklist
- Add integration tests in `tests/test_api.py` for all three endpoints.
- Verify response schema stability under both model-loaded and fallback cases.
- Verify crisis-positive path always sets guidance-required semantics.
- Confirm non-clinical disclaimer and safety language remain visible in API/UI outputs.

## 7. Release Gate Recommendation
Before merging to `dev`:
- `python -m ruff check .`
- `python -m black . --check`
- `python -m pytest -q`
- Confirm docs are updated if endpoint behavior or artifact paths change.
