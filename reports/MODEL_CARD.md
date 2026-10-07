# Model Card — Tamweel Lite

**Status:** READY_FOR_REVIEW — interpretability quality requires human review; not an automated grade  
**Source:** LIVE — `artifacts/final_metrics.json`, `artifacts/final_policy.json`, `artifacts/day5_*`, `artifacts/ensemble_comparison.csv`  
**Program:** DSC-211 · SDAIA Academy via Learning Space · October 2026

---

## 1. Purpose & Intended Use

Tamweel Lite is an **educational credit-risk assessment simulation**. It predicts the probability that a synthetic retail loan application will default within 90 days of origination (`default_within_90d = 1`).

**This model must NOT be used for:**
- Real-world automated credit decisions
- Actual customer loan approvals or rejections
- Any live commercial or financial deployment

All data is synthetic (tamweel-lite-1.0). Results do not represent real individuals, real loans, or any financial institution.

---

## 2. Model Details

| Parameter | Value |
|:---|:---|
| **Final model** | Logistic Regression |
| **Selection decision** | KEEP SINGLE — no ensemble passed the AP-lift gate |
| **Training features** | 22 pre-application predictors (no IDs, dates, or post-outcome columns) |
| **Target variable** | `default_within_90d` (1 = synthetic default within 90 days) |
| **Training pool size** | 6,576 rows (fit + calibration combined) |
| **Random seed** | 211 |
| **Python version** | 3.13.16 · Platform: Linux |

**Key library versions** (from `artifacts/environment.json`):

| Library | Version |
|:---|:---:|
| scikit-learn | 1.6.1 |
| lightgbm | 4.6.0 |
| xgboost | 3.4.1 |
| shap | 0.52.0 |
| optuna | 4.5.0 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |

---

## 3. Data & Validation Design

**Dataset:** tamweel-lite-1.0  
- Training: 10,000 records (2022–2024) · Challenge: 2,500 records (2025, no labels)  
- 22 strictly pre-application features; leakage controls exclude `days_past_due_60`, `collection_calls`, customer IDs, and dates

**Temporal split roles:**

| Role | Rows | Positives | Notes |
|:---|:---:|:---:|:---|
| Fit (training) | 2,516 | — | Pre-evaluation periods |
| Calibration | 584 | 40 | Reserved holdout; excluded from model fit |
| Policy selection | 589 | — | Threshold fitting only |
| Evaluation | 1,733 | 139 | 2024 Q3–Q4; not an untouched holdout |

**Validation protocol:** 3-period nested forward validation. For each outer fold:
- Validation customers are strictly excluded from training (customer-level leak prevention)
- A 90-day outcome maturity buffer is enforced before each fold boundary
- Immature labels and customer-overlap rows are removed from training

**Forward fold summary** (from `artifacts/day5_period_capacity.csv`):

| Fold | Validation Rows | Positives | Prevalence |
|:---:|:---:|:---:|:---:|
| 1 | 732 | 62 | 8.47% |
| 2 | 734 | 49 | 6.68% |
| 3 | 689 | 68 | 9.87% |

---

## 4. Model Selection & Ensemble Gate

The ensemble gate requires: **AP lift over best single > fold SD** (0.0298), Brier increase ≤ 0.005, ECE increase ≤ 0.01. No candidate passed.

**Mean AP across 3 forward folds** (from `artifacts/ensemble_comparison.csv`):

| Candidate | Mean AP | Fold SD | AP Lift vs. Logistic | Gate |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression ✓** | **0.3917** | 0.0298 | — | Selected |
| Weighted Ensemble | 0.3894 | 0.0291 | −0.0022 | ✗ Failed |
| Stack Ensemble | 0.3831 | 0.0295 | −0.0085 | ✗ Failed |
| Equal Ensemble | 0.3717 | 0.0326 | −0.0200 | ✗ Failed |
| XGBoost | 0.3526 | 0.0290 | −0.0390 | ✗ Failed |
| LightGBM | 0.3455 | 0.0435 | −0.0462 | ✗ Failed |

**Residual correlation** (from `artifacts/day5_residual_correlation.csv`): Pearson r between model error residuals ranges from **0.978 to 0.993**, confirming negligible error diversity — a key reason ensembles provide no benefit.

---

## 5. Performance Metrics

**OOF threshold selection** (2,155 rows · raw threshold 0.1689):

| Metric | Value |
|:---|:---:|
| Mean OOF Average Precision | **0.3917** (±0.0298) |
| Recall @ threshold | **46.93%** |
| Precision @ threshold | **34.29%** |
| TP / FP / FN / TN | 84 / 161 / 95 / 1,815 |
| Flag fraction | 11.37% (245 / 2,155) |
| Loss (10×FN + 1×FP) | 1,111 units |
| Loss per 10,000 (normalised) | 5,155 units |
| Capacity feasible (all folds) | ✓ True |

**Per-fold AP** (from `artifacts/day5_fold_scores.csv`):

| Fold | Logistic AP | XGBoost AP | LightGBM AP |
|:---:|:---:|:---:|:---:|
| 1 | 0.3606 | 0.3251 | 0.2963 |
| 2 | 0.4200 | 0.3498 | 0.3787 |
| 3 | 0.3944 | 0.3830 | 0.3615 |

---

## 6. Calibration

Platt (Sigmoid) scaling fit on the reserved calibration split (836 rows · 78 positives).

**Formula:** `calibrated_prob = expit(-(a × raw_prob + b))` where `a = −5.5396`, `b = 2.9070` (strictly increasing)

| Metric | Raw | Sigmoid Calibrated |
|:---|:---:|:---:|
| ECE | 0.0211 | 0.0349 |
| Brier Score | 0.0765 | 0.0781 |
| Log Loss | 0.2665 | 0.2773 |
| ROC-AUC | 0.7890 | 0.7890 *(unchanged)* |
| Average Precision | 0.2878 | 0.2878 *(unchanged)* |

ROC-AUC and AP are unchanged because Platt scaling is monotonic and rank-preserving. ECE increases slightly on this split (0.0211→0.0349); the probability-scale shift enables correct threshold policy transfer. Bins above 0.5 are empty after calibration and remain noisy due to low sample counts.

> The Brier and ECE values above are diagnostics on the calibration-fit sample, not an independent test. No performance claim is made for the unlabelled challenge batch.

---

## 7. Decision Policy & Capacity

| Parameter | Value |
|:---|:---|
| Cost matrix | FN = 10 units · FP = 1 unit |
| Loss formula | `Loss = 10 × FN + 1 × FP` |
| Raw OOF threshold | 0.1689 |
| Calibrated threshold | 0.1223 |
| Hard review capacity cap | 12% of batch volume |
| Challenge batch size | 2,500 |
| Review slots available | 300 (12% × 2,500) |
| Applications eligible (≥ 0.1223) | 330 |
| Final flagged (top 300 by score) | **300** |
| Borderline cases pruned by cap | 30 |
| Equal-score tie policy | Retain full tied block; drop boundary block if it cannot fit |

**Validation capacity — all three folds within the 12% ceiling:**

| Fold | Rows | Capacity | Flagged | Flag Rate | Within Cap |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 732 | 87 | 86 | 11.75% | ✓ |
| 2 | 734 | 88 | 84 | 11.44% | ✓ |
| 3 | 689 | 82 | 75 | 10.89% | ✓ |

---

## 8. Regional Fairness Audit (Descriptive — OOF)

Regional FPR differences are **descriptive diagnostics only**. They do not certify fairness, prove causal geographic relationships, or constitute a legal compliance statement.

| Region | Rows | Flagged | TP | FP | FPR | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Central | 537 | 63 | 24 | 39 | 7.98% | 50.00% |
| Western | 533 | 71 | 21 | 50 | **10.29%** | 44.68% |
| Eastern | 542 | 48 | 16 | 32 | **6.31%** | 45.71% |
| Other | 543 | 63 | 23 | 40 | 8.10% | 46.94% |

FPR gap (Western − Eastern): **3.98 percentage points**. Quarterly regional subgroup audits are recommended in production to flag potential geographic selection discrepancies.

---

## 9. Explainability

SHAP values are computed on 300 sampled rows in raw log-odds of the uncalibrated model (background: tree training path counts · max additivity error: 5.70e-15). They are strictly additive on the log-odds scale and must be transformed via the sigmoid to obtain probabilities.

**Top features by mean |SHAP| and permutation AP drop:**

| Feature | Mean \|SHAP\| (log-odds) | Permutation AP Drop |
|:---|:---:|:---:|
| `bureau_score` | **0.9042** | **0.1270** |
| `dti` | **0.5437** | **0.0690** |
| `loan_amount_sar` | 0.3490 | 0.0140 |
| `savings_balance_sar` | 0.2169 | 0.0094 |
| `existing_obligations_sar` | 0.2022 | — |
| `prior_defaults` | 0.1830 | 0.0065 |

> **Important:** The SHAP values and local waterfall plots in Day 4 (`artifacts/shap_beeswarm.png`, `artifacts/shap_waterfall.png`) were derived from the intermediate weighted LightGBM model on a separate training split. The final deployed model is a Logistic Regression fit on the full 6,576-row pool. SHAP contributions and individual reason codes **must be recomputed** before presenting explanations to business stakeholders.

---

## 10. Monitoring Recommendations

| Monitor | Frequency | Metric / Signal |
|:---|:---:|:---|
| Input drift (credit score distribution) | Weekly | Population Stability Index (PSI) |
| Probability quality | Rolling 30-day | Brier score, ECE on matured outcomes |
| Review volume | Daily | Flag count vs. 12% capacity ceiling |
| Regional subgroup error rates | Quarterly | FPR per region vs. overall FPR |
| Threshold recalibration trigger | On drift signal | Re-run policy design on fresh development data |

If capacity is exceeded: design a new policy on fresh development data and evaluate with new evidence. Do not raise the cap or prune cases after observing outcomes.

---

## 11. Limitations

1. **Synthetic data only** — tamweel-lite-1.0 does not represent real individuals, real loans, or any bank.
2. **OOF ≠ untouched test** — three forward folds are used for model selection; they are not a statistical significance test. Challenge batch has no labels; no AP, loss, or fairness claim is made for it.
3. **Fixed cost ratio** — FN=10, FP=1 and 12% cap are simulation parameters, not production LGD or regulatory limits.
4. **Temporal stability not guaranteed** — continuous PSI monitoring is required; thresholds may need recalibration as data distributions shift.
5. **Threshold precision** — preserve full decimal precision (0.12225843144286948); rounding may change queue size.
6. **Explainability scope** — SHAP attributions are for the intermediate LightGBM model; they are not directly transferable to the final Logistic Regression without recomputation.
7. **Regional comparison** — descriptive only; not a legal or causal fairness certification.
8. **Loss units are educational** — not real SAR, tool fees, or grade deductions.

---

## 12. Reproducibility

| Item | Value |
|:---|:---|
| Random seed | 211 |
| Python | 3.13.16 |
| Final model artifact | `artifacts/final_model/model.json` |
| Model manifest | `artifacts/final_model/model_manifest.json` |
| Policy artifact | `artifacts/final_policy.json` |
| Inference script | `scripts/inference.py` |
| Retrain script | `scripts/rebuild_final.py` |
| Replay script | `scripts/replay_final.py` |
| Lab revision | `f486fc50dd9ac8403016facc58cf6a62beb4abf4` |
| Data revision | `fe0c0204e6076a7ac2139b7336485a097343fb8a` |

`scripts/inference.py` returns one row per application with `application_id` and `calibrated_probability`. The decision policy is applied after collecting the full batch — do not apply the threshold row-by-row before ranking. `replay_final.py` verifies saved predictions; `rebuild_final.py` retrains from scratch. Do not retrain after the calibrator has been frozen.

---

## 13. Artifact References

| File | Contents |
|:---|:---|
| `artifacts/final_metrics.json` | OOF threshold selection, calibration diagnostics, challenge batch audit |
| `artifacts/final_policy.json` | Calibrated threshold, Platt mapping parameters |
| `artifacts/ensemble_comparison.csv` | Multi-model AP comparison across 3 folds |
| `artifacts/day5_fold_scores.csv` | Per-fold AP, Brier, ECE for all 6 candidates |
| `artifacts/day5_residual_correlation.csv` | Pairwise Pearson r between model residuals |
| `artifacts/day5_period_capacity.csv` | Per-fold capacity evidence |
| `artifacts/day5_region_audit.csv` | Regional FPR / recall breakdown |
| `artifacts/calibration_metrics.json` | Full calibration metrics and Platt mapping |
| `artifacts/permutation_importance.csv` | Permutation AP drop (3 repeats, 1,733 eval rows) |
| `artifacts/day4_shap_global.csv` | Global mean \|SHAP\| per feature (intermediate LightGBM) |
| `artifacts/environment.json` | Library versions and environment details |
| `artifacts/day5_run.json` | Run config, script hashes, data revision |
