# MindGuard Project Execution Tasklist

Version: 1.0
Date: 2026-03-29
Goal: Start implementation with strict standards, architecture discipline, and low-risk parallel collaboration.

## 1. Working Model
- Branch model:
  - main: stable only
  - dev: integration
  - feature/member-a-core: Member A implementation
  - feature/member-b-data-eval: Member B implementation
- Commit style: Conventional Commits
- PR policy: small PRs, single scope, reviewed before merge

## 2. Global Quality Gates (Apply To Every Task)
- `black .`
- `ruff check .`
- `pytest -q`
- No secret keys in code
- Docs updated when behavior changes
- Safety language preserved in API/UI

## 3. Official Start Checklist (Before Coding)
1. Pull latest from remote:
   - `git checkout main`
   - `git pull origin main`
2. Switch to feature branch:
   - Member A: `git checkout feature/member-a-core`
   - Member B: `git checkout feature/member-b-data-eval`
3. Rebase on latest dev:
   - `git fetch origin`
   - `git rebase origin/dev`
4. Verify environment:
   - `pip install -r requirements.txt`
   - `python -m spacy download en_core_web_sm`
5. Run baseline checks:
   - `python -m pytest -q`
6. Confirm docs read by both members:
   - `docs/PROJECT_MASTER_GUIDE.md`
   - `docs/CODING_STANDARDS.md`
   - `docs/TEAM_DEVELOPMENT_SPLIT.md`
   - `docs/MODEL_CARD.md`

## 4. Member A (You) Detailed Micro-Task Plan
Owner: Core integration, APIs, services, orchestration, and safe runtime behavior.
Branch: `feature/member-a-core`

### Phase A1: Contracts and Config Hardening
1. Validate `app/api/schemas.py` request/response models against spec fields.
2. Add missing constraints (min length, optional defaults, enum constraints where needed).
3. Confirm `app/core/config.py` environment loading and default safety values.
4. Add/verify central constants in `app/core/constants.py` for:
   - Crisis threshold default
   - Top-k emotion output
   - Safety disclaimer text
   - Crisis hotline suffix text
5. Add docstrings and type hints for all modified symbols.
6. Run `pytest -q` and fix only schema/config-related failures.
7. Commit:
   - `feat(core): harden config and schema contracts`

### Phase A2: Preprocessing Service Reliability
1. Review `app/services/nlp_preprocessor.py` normalization pipeline step-by-step.
2. Make preprocessing deterministic (same input, same output).
3. Add guards for:
   - Empty text
   - Very short text
   - Excessive whitespace
   - Non-ASCII symbols/noise
4. Ensure preprocessor returns stable shape consumed by both emotion and crisis services.
5. Add unit tests for edge cases in `tests/test_preprocessor.py`.
6. Run `pytest -q tests/test_preprocessor.py`.
7. Commit:
   - `feat(service): harden nlp preprocessor edge handling`

### Phase A3: Emotion Service Integration Path
1. Review `app/models/bert_classifier.py` interface contract.
2. Implement robust load path (model missing, corrupted artifact, wrong label map).
3. Integrate inference path into `app/services/emotion_service.py`.
4. Return top emotions + confidence in a stable sorted format.
5. Ensure output stays backward compatible with API schema.
6. Add tests in `tests/test_emotion_service.py` for:
   - Happy path
   - Missing model fallback behavior
   - Empty text behavior
7. Run:
   - `pytest -q tests/test_emotion_service.py`
8. Commit:
   - `feat(emotion): integrate bert inference service path`

### Phase A4: Crisis Service and Safety Behavior
1. Review `app/models/vae_detector.py` loading/inference interface.
2. Integrate VAE inference into `app/services/crisis_service.py`.
3. Keep threshold externalized (config/constants), not hardcoded in logic.
4. Add keyword baseline comparison path (for debug/evaluation mode).
5. Enforce recall-oriented default threshold policy.
6. Ensure crisis-positive output always includes hotline guidance token/flag.
7. Add tests in `tests/test_crisis_service.py` for:
   - Threshold boundary conditions
   - Crisis-positive guidance behavior
   - Fallback when model not available
8. Run:
   - `pytest -q tests/test_crisis_service.py`
9. Commit:
   - `feat(crisis): integrate vae detection and safety policy`

### Phase A5: Response Orchestration and Groq Resilience
1. Review `app/services/prompt_builder.py` for strategy templates.
2. Ensure prompts include:
   - Emotion context
   - Crisis flag context
   - Safety constraints
3. Add resilient Groq call behavior in response flow:
   - Retry with capped attempts
   - Backoff delay
   - Safe fallback response when API fails
4. Ensure no secrets are logged.
5. Add focused tests/mocks for failure and fallback cases.
6. Commit:
   - `feat(response): add resilient groq orchestration and fallback`

### Phase A6: API Endpoints End-to-End
1. Finalize route behavior in:
   - `app/api/routes/emotion.py`
   - `app/api/routes/crisis.py`
   - `app/api/routes/response.py`
2. Ensure consistent HTTP status codes and error structures.
3. Ensure `app/main.py` wiring is clean and route registration deterministic.
4. Expand `tests/test_api.py` with integration coverage for all three endpoints.
5. Run:
   - `pytest -q tests/test_api.py`
6. Commit:
   - `feat(api): complete endpoint orchestration and integration tests`

### Phase A7: UI Integration and Safety Clarity
1. Connect UI in `app/ui/gradio_app.py` to backend response structure.
2. Display clearly:
   - Emotions and confidence
   - Crisis flag
   - Safety guidance when crisis-positive
3. Add non-clinical disclaimer in visible UI location.
4. Validate mobile-friendly layout behavior at basic level.
5. Commit:
   - `feat(ui): integrate api outputs with visible safety messaging`

### Phase A8: Final Branch Hardening (Before PR)
1. Run full gate:
   - `black .`
   - `ruff check .`
   - `pytest -q`
2. Update docs if API/behavior changed.
3. Rebase branch on latest `origin/dev`.
4. Prepare PR notes with:
   - Scope
   - Files changed
   - Validation evidence
   - Known risks
5. Open PR: `feature/member-a-core` -> `dev`.

## 5. Member B Task Plan (Data, Training, Evaluation)
Owner: Data pipeline, experiment execution, artifacts, and evidence for report.
Branch: `feature/member-b-data-eval`

### Phase B1: Data Pipeline Setup
1. Finalize `scripts/download_data.py` for reproducible dataset fetch/prep.
2. Ensure clear raw/processed directory paths under `data/`.
3. Add validation checks for missing columns/invalid labels.
4. Commit:
   - `feat(data): finalize reproducible dataset preparation pipeline`

### Phase B2: BERT Training Pipeline
1. Complete `scripts/train_bert.py` with configurable hyperparameters.
2. Add deterministic seed handling.
3. Add checkpoint save policy and best-model export path.
4. Start MLflow logging (params, metrics, artifacts, run id).
5. Commit:
   - `feat(train): add bert training pipeline with mlflow logging`

### Phase B3: VAE Training Pipeline
1. Complete `scripts/train_vae.py` for non-crisis-only training regime.
2. Add threshold estimation from validation distribution.
3. Export reconstruction error statistics and selected threshold evidence.
4. Log to MLflow.
5. Commit:
   - `feat(train): add vae crisis pipeline and threshold logging`

### Phase B4: Evaluation and Baselines
1. Build notebook(s) in `notebooks/` for:
   - Emotion evaluation
   - Crisis evaluation vs keyword baseline
2. Report precision/recall/F1 and threshold rationale.
3. Save reproducible figures/tables for report insertion.
4. Commit:
   - `feat(eval): add evaluation notebooks and baseline comparison`

### Phase B5: Artifact and Handoff Package
1. Create artifact manifest (paths, versions, run ids).
2. Update `docs/MODEL_CARD.md` sections with measured results.
3. Provide handoff note for Member A integration.
4. Open PR: `feature/member-b-data-eval` -> `dev`.
5. Commit:
   - `docs(artifacts): add model metrics and handoff manifest`

## 6. Integration Sequence (Professional Merge Order)
1. Member B PR merges first into `dev` (data + artifacts available).
2. Member A rebases on updated `dev` and finishes service/API integration.
3. Member A PR merges into `dev`.
4. Joint stabilization pass on `dev`:
   - full tests
   - safety checks
   - report evidence mapping
5. Create release PR `dev` -> `main`.

## 7. Daily Execution Rhythm
1. Morning sync:
   - `git fetch origin`
   - `git rebase origin/dev`
2. Work in micro-scope (1 task block only).
3. Run local tests for touched area.
4. Commit with clear message.
5. Push branch.
6. End-of-day short status note:
   - done
   - blocked
   - next

## 8. No-Compromise Rules
- No direct pushes to `main` except reviewed release merge.
- No threshold tuning using test set.
- No untracked "final" metrics without MLflow run evidence.
- No API behavior changes without schema and test updates.
- No removal of safety disclaimer or crisis guidance behavior.
