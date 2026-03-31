# MindGuard Dataset Registry

Version: 1.0
Date: 2026-03-31
Status: Finalized for current project scope

## 1. Purpose
This registry is the authoritative record of dataset usage, license constraints, and allowed project operations.

Use this file to verify:
- what data is approved
- what use is legally allowed
- what restrictions must be respected in implementation and reporting

## 2. Approved Scope Datasets

| Dataset | Pipeline Role | Source | License | Allowed in This Project | Mandatory Restrictions | Split/Prep Policy | Status |
|---|---|---|---|---|---|---|---|
| GoEmotions | Emotion classification (BERT multi-label) | https://huggingface.co/datasets/google-research-datasets/go_emotions | Apache-2.0 | Yes (academic/non-clinical prototype) | Keep attribution and citation in report; document known bias limitations | Use simplified split (train/val/test) where available; keep preprocessing reproducible | Approved (Primary) |
| Dreaddit (stress) | Crisis/stress external benchmark validation | https://arxiv.org/abs/1911.00133 | Refer to original dataset distribution terms | Yes for academic benchmarking after source-term verification in implementation notes | Do not claim clinical diagnosis; document domain mismatch risk (stress vs crisis) | Keep as evaluation benchmark dataset; do not mix with final test leakage | Approved (Benchmark) |
| Amod mental_health_counseling_conversations | Response quality and empathetic-response support evaluation | https://huggingface.co/datasets/Amod/mental_health_counseling_conversations | RAIL-D | Yes (non-commercial academic use) | Must comply with RAIL use restrictions; do not rewrite/delete individual QA pairs; preserve license notice; include machine-generated disclosure in public outputs where required by license terms | Create reproducible train/val/test split locally and log seed/version | Approved (Primary for response evaluation) |
| ESConv | Optional comparative response dataset | https://huggingface.co/datasets/thu-coai/esconv | CC-BY-NC-4.0 | Yes for non-commercial academic comparison | Non-commercial use only; retain attribution/citation | Use only if timeline allows after primary baseline is stable | Approved (Optional) |

## 3. Not Selected for Core Pipeline (Current Scope)

| Dataset | Reason Not Selected |
|---|---|
| Kaggle OSMI Mental Health in Tech Survey | Tabular workplace survey; not aligned to conversational text modeling pipelines |
| Aggregated Kaggle mixed mental-health sentiment datasets | Mixed provenance and harder license traceability for strict academic reproducibility |

## 4. Allowed Operations Checklist (Project Policy)

| Operation | GoEmotions | Dreaddit | Amod | ESConv |
|---|---|---|---|---|
| Download and local preprocessing | Yes | Yes | Yes | Yes |
| Train/evaluate models | Yes | Yes | Yes | Yes |
| Modify individual original records | Avoid for raw reference data | Avoid for raw reference data | No (must not rewrite/delete individual QA pairs) | Avoid for raw reference data |
| Publish derived metrics/plots in report | Yes | Yes | Yes | Yes |
| Commercial deployment usage | Yes (subject to Apache terms) | Verify source terms first | Restricted by RAIL-D donation/compliance terms | No (NC license) |

## 5. Compliance Notes for Report
- Include license type and citation for every dataset used.
- Include ethical and non-clinical limitations.
- Document data-source bias and generalization limitations.
- Keep this registry updated if any dataset changes.

## 6. Ownership and Update Rule
- Owner: Team (both members)
- Update rule: any dataset change requires same-PR update to this file and report references.
