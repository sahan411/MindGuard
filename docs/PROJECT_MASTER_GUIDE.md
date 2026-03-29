# MindGuard Project Master Guide

Version: 1.0
Date: 2026-03-29
Status: Active Implementation Guide

## 1. Purpose
This document is the single source of truth for building MindGuard correctly and consistently.
It consolidates:
- Project scope
- System architecture
- Technology stack
- Implementation standards
- Team workflow
- Validation and delivery criteria

Use this guide together with the project specification document in the repository.

## 2. Project Scope
### 2.1 Problem
MindGuard is an AI-assisted mental health support prototype for identifying emotional context and crisis risk in user text, then generating safe and empathetic responses.

### 2.2 In Scope
- Multi-label emotion classification with BERT (GoEmotions, 28 labels)
- Crisis detection using VAE anomaly detection
- Keyword baseline for crisis detection comparison
- Empathetic response generation with prompt engineering (Groq Llama)
- FastAPI backend with 3 production endpoints
- Gradio UI for demo interaction
- Evaluation notebooks and reproducible experiments
- Unit + integration tests
- Responsible AI safety behavior and disclaimers

### 2.3 Out of Scope
- Clinical diagnosis
- Medical treatment recommendations
- Production deployment as a healthcare system
- Long-term user data storage

## 3. Safety and Responsible AI Requirements
These are mandatory and non-negotiable:
- Every crisis-positive response must include professional help guidance.
- UI, API behavior, and docs must state this is a non-clinical academic prototype.
- Do not claim diagnostic capability.
- Prefer higher crisis recall to reduce missed high-risk cases.
- Do not store raw personal user text by default.

Recommended crisis resources to include:
- Sri Lanka: Lifeline 1926, CCCline 1333
- International: Crisis Text Line (text HOME to 741741)

## 4. High-Level Architecture
MindGuard is a layered, modular system:

1. Presentation Layer
- Gradio app receives user text and displays output.

2. API Layer
- FastAPI endpoints:
  - POST /predict/emotion
  - POST /predict/crisis
  - POST /generate/response

3. Service Layer
- Preprocessing service
- Emotion service
- Crisis service
- Prompt builder service

4. Model Layer
- BERT classifier module
- VAE detector module

5. Data and Artifact Layer
- Datasets (raw and processed)
- GloVe embeddings
- Trained model files
- Experiment logs (MLflow)

## 5. Request Data Flow
1. User submits text from Gradio.
2. API receives request.
3. Text normalization and preprocessing run.
4. Emotion service predicts top emotions and confidence.
5. Crisis service computes reconstruction error and crisis flag.
6. Prompt builder uses text + emotion + crisis flag + strategy to generate response via Groq.
7. API returns combined output.
8. UI renders response + crisis guidance when needed.

## 6. Repository Structure
Core folders:
- app/: main backend package
- data/: raw, processed, glove
- models/: trained model artifacts
- notebooks/: EDA, training, results analysis
- scripts/: automation and CLI training scripts
- tests/: test suite
- docs/: architecture and process documentation

## 7. Technology Stack
### 7.1 Language and Runtime
- Python 3.11.x

### 7.2 Core Libraries
- torch
- transformers
- datasets
- tokenizers
- fastapi
- uvicorn
- gradio
- groq
- spacy
- nltk
- pandas
- numpy
- scikit-learn

### 7.3 MLOps and Tooling
- mlflow
- pytest
- ruff
- black
- pre-commit
- jupyter

### 7.4 Configuration
- python-dotenv
- pydantic-settings

## 8. Environments and Execution Strategy
### 8.1 Local Development (VS Code)
Use local environment for:
- API and UI integration
- Service/module coding
- Unit and integration tests
- Inference validation

### 8.2 Colab (GPU-Accelerated Training)
Use Colab for:
- BERT fine-tuning
- VAE training when local GPU is unavailable

Colab handoff requirements:
- Save trained artifacts into repository-compatible model paths.
- Record model version, data snapshot, and metrics.
- Keep notebook deterministic and reproducible.

## 9. Coding Standards (Industry Baseline)
### 9.1 General Rules
- Follow PEP 8 and enforce with Black + Ruff.
- Use type hints for all public functions and methods.
- Write concise docstrings for modules/classes/functions.
- Keep functions small and single-purpose.
- No hardcoded secrets or credentials.
- Avoid magic numbers; centralize constants.

### 9.2 Comments Policy
- Add comments only where intent is not obvious.
- Explain why, not what.
- Keep comments short and maintainable.

Good comment examples:
- Why threshold chosen from validation percentile.
- Why fallback strategy is used when model unavailable.

Avoid:
- Repeating code literally in comments.
- Over-commenting trivial assignments.

### 9.3 API Design Rules
- Validate all input with Pydantic models.
- Return explicit response schemas.
- Use stable field names and backward-compatible changes.
- Include clear error responses and status codes.

### 9.4 Reliability Rules
- Add input guards and null checks at service boundaries.
- Fail gracefully when external LLM calls fail.
- Log actionable errors without leaking secrets.

### 9.5 Testing Rules
- Unit tests for each service and helper.
- Integration tests for API routes.
- Keep tests deterministic.
- Add regression tests for every bug fix.

## 10. Model-Specific Implementation Rules
### 10.1 Emotion Pipeline
- Multi-label setup with sigmoid outputs.
- Threshold tuned on validation set and documented.
- Report micro-F1, macro-F1, per-label metrics.

### 10.2 Crisis Pipeline
- Train VAE only on non-crisis data.
- Determine threshold from validation distribution.
- Compare VAE vs keyword baseline on same test set.
- Report precision, recall, F1, and threshold rationale.

### 10.3 Response Pipeline
- Implement strategies: zero-shot, few-shot, chain-of-thought style prompting.
- Enforce crisis-safe suffix policy.
- Evaluate with BLEU/ROUGE and human rubric (empathy, relevance, safety).

## 11. Git and Collaboration Workflow
### 11.1 Branching
- main: stable only
- dev: integration branch
- feature/*: individual workstreams

### 11.2 Commits
Use Conventional Commits:
- feat(scope): ...
- fix(scope): ...
- docs(scope): ...
- test(scope): ...
- refactor(scope): ...
- chore(scope): ...

### 11.3 Pull Requests
- Open PR to dev.
- Require review from the other team member.
- Merge only after tests pass and comments resolved.

## 12. Definition of Done (Per Feature)
A feature is done only when:
1. Code follows formatting and lint rules.
2. Type hints and docstrings are present.
3. Unit/integration tests are added and passing.
4. API/schema behavior is documented.
5. Safety rules are satisfied.
6. Changes are committed with meaningful messages.

## 13. Implementation Plan (Practical Order)
1. Finalize config + constants + schema contracts.
2. Complete preprocessing service.
3. Integrate trained BERT inference path.
4. Integrate trained VAE inference path + keyword baseline.
5. Build robust prompt generation and Groq fallback handling.
6. Complete API route orchestration.
7. Connect Gradio to API and display structured results.
8. Add comprehensive tests.
9. Run evaluation notebooks and log results.
10. Freeze report artifacts and demo script.

## 14. Risk Register
- Dataset bias and language/culture limits
- False negatives in crisis detection
- External API downtime/rate limits
- Inconsistent model artifact handoff from Colab
- Scope creep near final week

Mitigations:
- Document limitations clearly.
- Maintain fallback behavior.
- Keep reproducible artifact naming and versioning.
- Track progress weekly with acceptance checkpoints.

## 15. Quick Start Checklist
1. Create and activate virtual environment.
2. Install requirements.
3. Download spaCy model.
4. Configure .env from .env.example.
5. Run tests.
6. Start FastAPI.
7. Start Gradio.

## 16. Enforcement Checklist for Every Commit
- Black formatted
- Ruff clean
- Pytest passing
- No secrets committed
- Safety language preserved
- Docs updated if behavior changed

---
If implementation and this guide conflict, update this guide in the same PR so the team always has one consistent reference.
