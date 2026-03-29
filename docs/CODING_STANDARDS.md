# MindGuard Coding Standards

This file defines practical coding rules to keep quality high and predictable.

## 1. Python Style
- Python 3.11+
- Format with Black
- Lint with Ruff
- Max line length 88

## 2. Type Hints and Docstrings
- All public functions must have type hints.
- Public modules/classes/functions require docstrings.

## 3. Commenting Rules
Use comments when code intent is non-obvious.

Do:
- Explain non-trivial algorithmic decisions.
- Explain why thresholds or constants were chosen.

Do not:
- Comment obvious lines.
- Duplicate code with prose.

## 4. Error Handling
- Validate inputs at boundaries.
- Raise clear, actionable exceptions.
- Avoid swallowing exceptions silently.

## 5. API Standards
- Pydantic request/response models for every endpoint.
- Keep response schema stable.
- Include meaningful error responses.

## 6. Testing Standards
- Unit tests for each service.
- Integration tests for endpoints.
- Add regression tests for bug fixes.

## 7. Security and Privacy
- Never commit .env or secrets.
- Avoid logging sensitive raw user text.

## 8. Responsible AI Rules
- Include non-clinical disclaimer in UI and docs.
- Attach crisis resources on positive crisis signal.
- Never present the system as a diagnostic tool.

## 9. Commit Quality Gate
Before commit:
1. Run Ruff
2. Run Black
3. Run Pytest
4. Verify safety language remains intact
