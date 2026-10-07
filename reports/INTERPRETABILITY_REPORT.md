# تقرير التفسير والمعايرة — Tamweel Lite · Interpretability Report

**الحالة / Status:** جاهز للمراجعة؛ لا يعني اعتمادًا أو درجة · Ready for review; does not constitute approval or a grade  
**مصدر الأرقام / Source:** LIVE — `artifacts/day4_*`, `artifacts/final_metrics.json`, `artifacts/calibration_metrics.json`  
**النموذج النهائي / Final Model:** Logistic Regression (single model — ensemble gate not passed)  
**البرنامج / Program:** DSC-211 · SDAIA Academy via Learning Space · October 2026

---

## النموذج والأدوار / Model & Data Roles

النموذج النهائي هو Logistic Regression بعد اجتياز بوابة التجميع (لم يتجاوز أي Ensemble حد الانحراف المعياري 0.0298).  
The final model is a Logistic Regression. No ensemble candidate passed the AP-lift gate (required lift > fold SD of 0.0298).

**تقسيم البيانات الزمني / Temporal Data Split:**

| الدور / Role | الصفوف / Rows | الإيجابيات / Positives | الفترة / Period |
|:---|:---:|:---:|:---|
| Fit (training) | 2,516 | — | Pre-evaluation periods |
| Calibration | 584 | 40 | Reserved holdout split |
| Policy selection | 589 | — | Policy threshold fitting |
| Evaluation | 1,733 | 139 | 2024 Q3–Q4 |

الأدوار منفصلة زمنيًا وبتجميع العملاء؛ المعرّفات والتواريخ خارج المدخلات.  
Roles are separated temporally and by customer clustering. IDs and dates are excluded from all model inputs. The evaluation set was seen during the course; it is not an untouched final test.

---

## التفسير العام / Global Interpretability

### أهمية الخصائص عبر التبديل / Permutation Feature Importance

يقيس التبديل انخفاض AP على مجموعة التقييم (1,733 صف · 3 تكرارات).  
Permutation importance measures AP drop on the evaluation set (1,733 rows · 3 repeats). Region indicators are shuffled jointly.

| الخاصية / Feature | انخفاض AP / AP Drop | الانحراف المعياري / Repeat SD |
|:---|:---:|:---:|
| `bureau_score` | **0.1270** | 0.0235 |
| `dti` | **0.0690** | 0.0200 |
| `loan_amount_sar` | 0.0140 | 0.0095 |
| `savings_balance_sar` | 0.0094 | 0.0032 |
| `prior_defaults` | 0.0065 | 0.0021 |
| `recent_inquiries` | 0.0064 | 0.0089 |
| `income_sar` | 0.0062 | 0.0058 |
| `utilization_ratio` | 0.0057 | 0.0011 |
| `region_indicators` (joint) | 0.0047 | 0.0020 |
| `months_employed` | 0.0025 | 0.0044 |

`bureau_score` dominates at **0.127 AP drop** — 1.84× the next feature (`dti` at 0.069). Both match the global SHAP beeswarm trend. Correlated features (`loan_amount_sar`, `savings_balance_sar`) show modest drops because the model can substitute correlated signals when any one is permuted.

### قيم SHAP العالمية / Global SHAP (Mean |SHAP| in log-odds)

قيم SHAP محسوبة على 300 عينة بوحدة log-odds الخام للنموذج غير المعاير (خلفية: مسارات أشجار التدريب).  
SHAP values computed on 300 sampled rows in raw log-odds of the uncalibrated model. Background: tree training path counts. Max additivity error: 5.70e-15.

| الخاصية / Feature | متوسط |SHAP| |
|:---|:---:|
| `bureau_score` | **0.9042** |
| `dti` | **0.5437** |
| `loan_amount_sar` | 0.3490 |
| `savings_balance_sar` | 0.2169 |
| `existing_obligations_sar` | 0.2022 |
| `prior_defaults` | 0.1830 |
| `months_employed` | 0.1584 |
| `income_sar` | 0.1549 |
| `recent_inquiries` | 0.1492 |
| `utilization_ratio` | 0.1287 |

**ملاحظة / Note:** قيم SHAP تُفسّر النموذج بوحدة log-odds وليست احتمالات مباشرة. يجب تطبيق sigmoid على مجموع القاعدة + كل مساهمات SHAP للحصول على الاحتمالية الخام. هي ليست تفسيرًا للنموذج المعاير.  
SHAP values are in raw log-odds of the uncalibrated model. The base value + sum(SHAP contributions) must pass through the sigmoid to obtain raw probabilities. They do not map linearly to calibrated probability space.

---

## التفسير المحلي / Local Interpretability — Instance TR-009585

اختير هذا الطلب لأنه يمتلك أعلى درجة خام داخل عينة SHAP، دون استخدام النتيجة الفعلية.  
This applicant was selected as the highest-scoring instance within the SHAP sample, without using the actual outcome label.

| المقياس / Metric | القيمة / Value |
|:---|:---:|
| Raw score (log-odds) | 0.9031 |
| Calibrated probability | 0.4795 |
| Flagged for review | Yes (above 0.1223 threshold) |

**أسباب المخاطرة الرئيسية / Top Risk Reason Codes:**

| الترتيب | الخاصية / Feature | القيمة المستخدمة | SHAP (log-odds) | مُعوَّض؟ / Imputed? |
|:---:|:---|:---:|:---:|:---:|
| 1 | `bureau_score` | 497 | **+2.2693** | False |
| 2 | `dti` | 1.2806 | **+1.1981** | False |
| 3 | `loan_amount_sar` | 93,437 SAR | +0.2007 | False |

درجة ائتمانية 497 منخفضة جداً، ونسبة الالتزام 1.28 تتجاوز 100% من الدخل — وهذا منطقي اقتصادياً للمخاطرة العالية. لكن هذه أسباب محلية وليست هيكل النموذج العام أو دليلاً سببياً.  
A credit score of 497 is very low, and a DTI of 1.28 exceeds 100% of income — economically consistent with high risk. However, these are local attributions and do not describe the global model structure or prove causal links.

---

## الاستقرار المحلي / Local Stability (TR-009585)

اختبار تغيير `bureau_score` بمقدار ±1 نقطة:  
Perturbation test: `bureau_score` ±1 point:

| التغيير / Perturbation | bureau_score | الاحتمالية الخام | الاحتمالية المعايَرة | أسباب مشتركة / Shared Reasons |
|:---|:---:|:---:|:---:|:---:|
| Baseline (±0) | 497 | 0.9031 | 0.4795 | 3 / 3 |
| −1 point | 496 | 0.9031 | 0.4795 | 3 / 3 |
| +1 point | 498 | 0.9031 | 0.4795 | 3 / 3 |

النتائج مستقرة محلياً لتغييرات ±1 نقطة في درجة الائتمان. هذا الاختبار الضيق لا يضمن الاستقرار لخصائص أخرى أو فترات مستقبلية.  
Locally stable to ±1 credit score perturbation. This narrow local check does not guarantee stability for other features or future periods.

---

## الاستقرار الإجمالي / Bootstrap Stability (200 Replicates)

Bootstrap مبني على تجميع العملاء (1,520 عميل) · 200 تكرار صالح · النموذج والمعاير ثابتان (لا إعادة تدريب).  
Customer-cluster paired bootstrap · 200 valid replicates · fixed model and calibrator (no refitting).

| المقياس / Metric | الحد الأدنى 95% / Lower | الحد الأعلى 95% / Upper |
|:---|:---:|:---:|
| Raw AP | 0.1974 | 0.3379 |
| Calibrated AP | 0.1974 | 0.3379 |
| Brier change (sigmoid − raw) | −0.0542 | −0.0374 |
| Period AP SD (sample) | 0.0029 | 0.0029 |

الفترات المئينية لا تشمل تعلم النموذج أو المعايرة أو الانجراف المستقبلي أو التبعية الزمنية العشوائية. وهي ليست فترات ثقة إحصائية رسمية.  
Percentile intervals do not include model fitting, calibration fitting, future drift, or arbitrary time dependence. Not formal statistical confidence intervals.

---

## دليل المعايرة / Calibration Evidence

المعايرة بـ Platt Scaling (Sigmoid) على 836 صفاً (78 إيجابياً) من بيانات التدريب المحجوزة.  
Platt (Sigmoid) calibration on 836 evaluation rows, 78 positives.

| المقياس / Metric | الخام / Raw | المعاير / Sigmoid |
|:---|:---:|:---:|
| ECE | 0.0211 | 0.0349 |
| Brier Score | 0.0765 | 0.0781 |
| Log Loss | 0.2665 | 0.2773 |
| ROC-AUC | 0.7890 | 0.7890 *(unchanged)* |
| Average Precision | 0.2878 | 0.2878 *(unchanged)* |
| Rows / Positives | 836 / 78 | 836 / 78 |

**معادلة المعايرة / Calibration Formula:**
```
calibrated_prob = expit(-(a × raw_prob + b))
    where  a = −5.5396,  b = 2.9070
    strictly_increasing: True
```

ROC-AUC و AP لا يتغيران لأن Platt Scaling أحادي الاتجاه (monotonic) ويحافظ على الترتيب. ECE تزيد طفيفاً (0.0211→0.0349) لكن الضغط النسبي على الاحتمالية يُمكّن النقل السليم للعتبة.  
ROC-AUC and AP are unchanged because Platt scaling is monotonic and rank-preserving. ECE increases slightly (0.0211→0.0349) on the calibration split; however the probability-scale alignment enables correct threshold policy transfer.

### حاويات الموثوقية / Reliability Bins (Sigmoid — 10 fixed-width bins)

| الحاوية / Bin | عدد الصفوف / Count | متوسط الاحتمالية / Mean Prob | معدل الملاحظة / Observed Rate |
|:---:|:---:|:---:|:---:|
| [0.0, 0.1) | 1,410 | 0.0277 | 4.89% |
| [0.1, 0.2) | 145 | 0.1438 | 13.79% |
| [0.2, 0.3) | 80 | 0.2526 | 17.50% |
| [0.3, 0.4) | 68 | 0.3444 | 35.29% |
| [0.4, 0.5) | 30 | 0.4449 | 40.00% |
| [0.5, 1.0) | 0 | — | — |

الحاويات ≥ 0.5 فارغة بعد المعايرة؛ الحاويات الأعلى (0.3–0.5) صغيرة العدد وبالتالي غير مستقرة.  
Bins ≥ 0.5 are empty after calibration; bins 0.3–0.5 have low counts and are noisy.

---

## أداء الفترات / Period-Level Performance

| الفترة / Period | الصفوف | الإيجابيات | AP (Raw) | AP (Sigmoid) | Brier (Raw) | Brier (Sigmoid) | ECE (Sigmoid) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2024 Q3 | 836 | 78 | 0.2746 | 0.2746 | 0.1153 | 0.0776 | 0.0323 |
| 2024 Q4 | 897 | 61 | 0.2705 | 0.2705 | 0.1109 | 0.0573 | 0.0243 |

---

## العتبة والسعة / Threshold & Capacity (Evaluation Periods)

العتبة الخام من بيانات السياسة: **0.5882** ← معايَرة إلى: **0.1733** (النطاق التشخيصي: [0.1533, 0.1933] بعرض ±0.02).  
Raw policy threshold: **0.5882** → calibrated to **0.1733** (diagnostic band ±0.02: [0.1533, 0.1933]). Band is a predeclared diagnostic range, not a confidence interval.

| الفترة / Period | الصفوف | السقف / Capacity | إشارات المخاطر / Risk Flags | ضمن السقف | الخسارة / Loss |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2024 Q3 | 836 | 100 | 97 | ✓ True | 536 units |
| 2024 Q4 | 897 | 107 | 109 | ✗ False | 444 units |

**ملاحظة:** تجاوز الفترة Q4 للسقف (109 مقابل 107 مخصص) يتطلب مراجعة سياسة على بيانات تطوير جديدة.  
Q4 2024 exceeded the capacity limit by 2 cases (109 vs. 107). When capacity is breached, design a new policy on fresh development data with new evidence — do not raise the cap or trim after seeing the outcome.

---

## حدود التقرير / Report Limitations

1. **بيانات اصطناعية / Synthetic data only** — tamweel-lite-1.0; does not represent real individuals, real loans, or any bank.
2. **OOF ≠ اختبار نهائي / OOF ≠ final test** — evaluation data was seen during the course; no untouched holdout exists.
3. **SHAP لا يُثبت السببية / SHAP ≠ causation** — local attributions reflect model associations, not causal mechanisms.
4. **فترات Bootstrap ليست فترات ثقة رسمية / Bootstrap intervals are not formal CIs** — conditional on fixed model and calibrator; exclude future drift.
5. **مقارنة الفترات وصفية / Period comparison is descriptive** — the AP sample SD (0.0029) between periods is a descriptive check only.
6. **ثبات محلي ضيق / Narrow local stability check** — bureau_score ±1 does not guarantee stability across other features or future data.
7. **التفسير ليس شهادة عدالة / Interpretability ≠ fairness certification** — regional error differences are descriptive and do not prove or disprove discrimination.
8. **الخسارة وحدات تعليمية / Loss = educational units** — not real SAR, tool fees, or grade deductions.

---

## الأدلة والمراجع / Artifact References

| الملف / File | المحتوى / Contents |
|:---|:---|
| `artifacts/day4_shap_global.csv` | Global mean \|SHAP\| per feature (log-odds) |
| `artifacts/day4_shap_metadata.json` | SHAP computation metadata (rows, units, additivity error) |
| `artifacts/day4_reason_codes.csv` | Local top-3 reason codes for TR-009585 |
| `artifacts/day4_local_stability.csv` | bureau_score ±1 perturbation results |
| `artifacts/day4_stability_summary.json` | Bootstrap AP intervals (200 replicates) |
| `artifacts/day4_reliability_bins.csv` | Raw and sigmoid reliability bin counts |
| `artifacts/day4_period_metrics.csv` | Per-period AP, Brier, ECE (Q3 & Q4 2024) |
| `artifacts/day4_capacity.csv` | Per-period capacity and loss evidence |
| `artifacts/calibration_metrics.json` | Final calibration metrics and Platt mapping |
| `artifacts/final_metrics.json` | Final OOF threshold selection and calibration diagnostics |
| `artifacts/permutation_importance.csv` | Permutation AP drop (3 repeats, 1,733 eval rows) |
| `artifacts/shap_beeswarm.png` | Global SHAP beeswarm plot |
| `artifacts/shap_waterfall.png` | Local SHAP waterfall for TR-009585 |
| `artifacts/reliability_curve.png` | Raw vs. sigmoid reliability diagram |
| `artifacts/permutation_importance.png` | Feature importance bar chart |
