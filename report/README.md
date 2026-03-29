# MindGuard Final Report Workflow

This folder contains the final report source and build flow.

## Deliverables Alignment
- Final report: max 20 pages (excluding references and appendices)
- Video presentation: team contribution and communication quality
- Code repository: reproducible and documented

## Prerequisites (Windows)
- MiKTeX installed
- Perl installed (required by latexmk)
- `latexmk` available in PATH

## Build
From repository root:

```powershell
.\report\build.ps1
```

Output PDF is generated as:
- `report\main.pdf`

## Clean Build Artifacts

```powershell
latexmk -C -cd -outdir=report report/main.tex
```

## Notes
- Keep report content professional and evidence-driven.
- Ensure metrics in report match tracked experiment runs.
- Do not exceed page limit (references/appendix excluded).
