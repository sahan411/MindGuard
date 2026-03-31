# MindGuard

MindGuard is an academic AI prototype for:
- Multi-label emotion classification (BERT)
- Crisis detection (VAE anomaly detector vs keyword baseline)
- Empathetic response generation (Groq Llama-3)

## Core Project Documents
- Project architecture, scope, stack, workflow, and delivery guide: [docs/PROJECT_MASTER_GUIDE.md](docs/PROJECT_MASTER_GUIDE.md)
- Coding rules and quality baseline: [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md)
- Model transparency artifact: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- Report writing and build workflow: [report/README.md](report/README.md)
- Team parallel development split and branch workflow: [docs/TEAM_DEVELOPMENT_SPLIT.md](docs/TEAM_DEVELOPMENT_SPLIT.md)
- Full micro-task execution plan for both members: [docs/PROJECT_EXECUTION_TASKLIST.md](docs/PROJECT_EXECUTION_TASKLIST.md)
- Dataset licensing and allowed-use registry: [docs/DATASET_REGISTRY.md](docs/DATASET_REGISTRY.md)
- Artifact inventory, run IDs, and reproducibility snapshot: [docs/ARTIFACT_MANIFEST.md](docs/ARTIFACT_MANIFEST.md)
- Integration contract from data/eval track to API track (B5 -> A6): [docs/INTEGRATION_HANDOFF.md](docs/INTEGRATION_HANDOFF.md)

## Critical Safety Notice
MindGuard is a research prototype only. It is not a clinical tool and must not be used for diagnosis or emergency intervention. If someone is in immediate danger, contact local emergency services.

Sri Lanka: Lifeline 1926, CCCline 1333
International: Crisis Text Line (text HOME to 741741)

## Setup
1. Create and activate a virtual environment
2. Install dependencies
3. Download spaCy model
4. Configure environment variables
5. Run API and UI

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env

# edit .env and set GROQ_API_KEY
```

## Run
```bash
uvicorn app.main:app --reload --port 8000
python app/ui/gradio_app.py
```

## API Endpoints
- `POST /predict/emotion`
- `POST /predict/crisis`
- `POST /generate/response`

## Project Layout
Follows the specification in `MindGuard_Project_Specification.docx`.
