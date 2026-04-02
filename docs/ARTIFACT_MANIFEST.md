# MindGuard Artifact Manifest

Version: 1.0
Date: 2026-03-31
Scope: B5 handoff package for integration on branch `feature/member-b-data-eval`

## 1. Purpose
This manifest records the exact artifact paths, run identifiers, and evidence files produced so far, so Member A can integrate without ambiguity.

## 2. Artifact Inventory

### Emotion pipeline (BERT)
- Run summary: `data/processed/bert_run_summary.json`
- Best model directory: `models/bert_emotion/smoke_best_model_tiny`
- Checkpoint directory: `models/checkpoints/bert_emotion_smoke_tiny`
- MLflow run id: `664db6558d2648ccb5481fab8848b57b`

### Crisis pipeline (VAE)
- Threshold summary: `data/processed/vae_threshold_summary_smoke.json`
- Model state dict: `models/vae_crisis_smoke/vae_state_dict.pt`
- TF-IDF vocabulary: `models/vae_crisis_smoke/tfidf_vocabulary.json`
- Reconstruction errors: `models/vae_crisis_smoke/reconstruction_errors.csv`
- MLflow experiment: `mindguard-vae-crisis`

### Evaluation outputs
- Emotion metrics: `data/processed/evaluation/emotion_metrics.json`
- Crisis baseline metrics: `data/processed/evaluation/crisis_baseline_metrics.json`
- Crisis baseline table: `data/processed/evaluation/crisis_baseline_comparison.csv`
- Crisis baseline figure: `data/processed/evaluation/crisis_baseline_comparison.png`

### Dataset evidence
- Dataset manifest: `data/processed/dataset_manifest.json`
- GoEmotions processed splits:
  - `data/raw/go_emotions/train.jsonl`
  - `data/raw/go_emotions/validation.jsonl`
  - `data/raw/go_emotions/test.jsonl`
- AMOD processed split:
  - `data/raw/amod_mh_counseling/train.jsonl`

## 3. Verified Metrics Snapshot

### Emotion
- Validation micro-F1: 0.0947075208913649
- Validation macro-F1: 0.031907886927818326
- Test micro-F1: 0.12011173184357542
- Test macro-F1: 0.037491165514548785

### Crisis
- Threshold percentile: 95.0
- Threshold value: 0.001951873884536326
- VAE precision/recall/F1: 0.0 / 0.0 / 0.0
- Keyword precision/recall/F1: 0.0 / 0.0 / 0.0

## 4. Interpretation Notes
- Current outputs are smoke-stage evidence for reproducibility and integration, not final report-grade effectiveness.
- Crisis metrics are not representative because the smoke VAE data source has no positive crisis labels.
- Final benchmark run must include positive crisis examples before release metrics are frozen.

## 5. Reproducibility Commands
- Generate evaluation artifacts:
  - `python scripts/evaluate_baselines.py --bert-summary data/processed/bert_run_summary.json --vae-summary data/processed/vae_threshold_summary_smoke.json --vae-errors models/vae_crisis_smoke/reconstruction_errors.csv --output-dir data/processed/evaluation`
- View tests used for evaluation script:
  - `python -m pytest -q tests/test_evaluate_baselines.py`
