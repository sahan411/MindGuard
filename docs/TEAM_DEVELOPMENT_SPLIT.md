# MindGuard Team Development Split

Version: 1.0
Date: 2026-03-29
Purpose: Enable parallel development with low merge risk and professional git practice.

## 1. Branch Strategy
Use this branch model:
- `main`: stable, release-quality only
- `dev`: integration branch for reviewed work
- `feature/member-a-core`: Member A feature branch
- `feature/member-b-data-eval`: Member B feature branch

Rules:
- No direct commits to `main`.
- Open PRs from feature branch to `dev`.
- Merge `dev` to `main` only after integration checks pass.

## 2. Low-Conflict Work Split

### Member A (Core App and Integration Owner)
Primary responsibility:
- API contracts and endpoint orchestration
- Model inference integration
- Final integration in app services

Own these files/folders:
- `app/main.py`
- `app/api/schemas.py`
- `app/api/routes/emotion.py`
- `app/api/routes/crisis.py`
- `app/core/*`
- `app/services/emotion_service.py`
- `app/services/crisis_service.py`
- `tests/test_api.py`

### Member B (Data, Training, and Evaluation Owner)
Primary responsibility:
- Dataset pipeline, training scripts, and evaluation outputs
- Metrics artifacts for report

Own these files/folders:
- `scripts/download_data.py`
- `scripts/train_bert.py`
- `scripts/train_vae.py`
- `notebooks/*`
- `data/processed/*`
- `models/*` (tracked metadata only, no large binary commits)
- `tests/test_emotion_service.py`
- `tests/test_crisis_service.py`

### Shared (Coordinate Before Editing)
- `app/services/nlp_preprocessor.py`
- `app/services/prompt_builder.py`
- `app/ui/gradio_app.py`
- `docs/PROJECT_MASTER_GUIDE.md`
- `report/main.tex`

If a shared file must be edited:
1. Announce in chat first.
2. Keep edits minimal and isolated.
3. Merge shared-file PRs first before other dependent PRs.

## 3. Professional Git Workflow
Use this sequence from repository root.

Initial setup:
```powershell
git checkout -b dev
git push -u origin dev
git checkout -b feature/member-a-core
git push -u origin feature/member-a-core
git checkout dev
git checkout -b feature/member-b-data-eval
git push -u origin feature/member-b-data-eval
git checkout feature/member-a-core
```

Daily sync (each member):
```powershell
git fetch origin
git checkout <your-feature-branch>
git rebase origin/dev
```

PR policy:
- Small PRs (one topic each)
- Conventional Commit messages
- Tests run before PR
- At least one reviewer approval

## 4. Definition of Done per PR
- Lint/format passes
- Tests pass for touched modules
- No secret keys or local artifacts committed
- Docs updated if behavior changed
- Clear PR description with scope and validation proof

## 5. Merge Order to Reduce Risk
1. Member B merges data/training outputs and evaluation scripts into `dev`.
2. Member A rebases on latest `dev` and merges integration/API updates.
3. Joint integration test on `dev`.
4. Merge `dev` to `main` in a single reviewed PR.

## 6. What To Avoid
- Long-lived branches without sync
- Massive mixed-purpose PRs
- Direct edits to the same shared file without coordination
- Committing model binaries or temporary experiment files