# بطاقة القرار — Tamweel Lite · Decision Card

**الحالة / Status:** جاهزة للمراجعة؛ لا تعني اعتمادًا أو درجة · Ready for review; does not constitute approval or a grade  
**مصدر الأرقام / Source:** LIVE — `artifacts/final_metrics.json`, `artifacts/final_policy.json`, `artifacts/day5_*`  
**الاستراتيجية / Strategy:** Logistic Regression (single model — ensemble gate not passed) · **OOF Rows:** 2,155  
**البرنامج / Program:** DSC-211 · SDAIA Academy via Learning Space · October 2026

---

## المهمة / Task

الفئة الموجبة `default_within_90d=1` تعني حدث تعثر اصطناعي خلال 90 يومًا بعد الطلب.  
كل طية تحقق طلبات لاحقة، وتستبعد عملاءها من التدريب، وتشترط نضج نتيجة التدريب قبل بدايتها. المعرّفات والتاريخ خارج المدخلات.

The positive class `default_within_90d=1` is a synthetic default event within 90 days of application. Each fold validates on forward-looking applications, excludes their customers from training, and requires outcome maturity before fold start. IDs and dates are excluded from all model inputs.

---

## النتائج الرئيسية / Key Results

| المقياس / Metric | القيمة / Value |
|:---|---:|
| **النموذج المختار / Selected Model** | **Logistic Regression** |
| **متوسط AP من OOF (3 طيات) / Mean OOF AP** | **0.3917** (±0.0298 fold SD) |
| **عتبة التدقيق الخام / Raw OOF Threshold** | **0.1689** |
| **عتبة التدقيق المعايَرة / Calibrated Threshold (Platt)** | **0.1223** |
| **Recall عند العتبة / Recall @ Threshold** | **46.93%** |
| **Precision عند العتبة / Precision @ Threshold** | **34.29%** |
| **TP / FP / FN / TN (OOF)** | 84 / 161 / 95 / 1,815 |
| **الخسارة التعليمية / Loss (10×FN + 1×FP)** | 1,111 وحدة / units |
| **الخسارة لكل 10,000 طلب / Loss per 10,000** | 5,155 وحدة / units |
| **السعة القصوى / Max Review Capacity** | 12.0% (300 / 2,500) |
| **الطلبات المؤهلة ≥ 0.1223 / Eligible (Challenge)** | 330 من / of 2,500 |
| **المُشار إليها بعد السقف / Flagged after Cap** | **300** (30 pruned) |
| **ECE الخام / Raw ECE** | 0.0211 |
| **ECE المعايَرة / Sigmoid ECE** | 0.0349 |
| **فجوة FPR بين المناطق / FPR Gap (Western − Eastern)** | **3.98 نقطة مئوية / pp** |

---

## القاعدة التشغيلية / Decision Rule

**العربية:** درجة معايَرة ≥ 0.1223 تعني إشارة مراجعة. رتّب المرشحين تنازليًا حسب الاحتمالية، وخذ أعلى 300 ضمن سقف 12%. احفظ الدقة الكاملة؛ تقريب العتبة قد يغيّر حجم الطابور.

**English:** A calibrated score ≥ 0.1223 marks an application for review. Rank flagged applications by descending risk probability and retain the top 300 within the 12% hard cap. Preserve full threshold precision — rounding may change queue size.

```
Calibrated Score ≥ 0.1223 ?
    No  → Standard automated processing
    Yes → Rank by probability (descending)
              Queue ≤ 12% cap (300 slots) ?
                  Yes → Flag for manual review
                  No  → Cap at top 300; drop boundary block
```

---

## السياسة / Policy Parameters

- **تكلفة FN / FN Cost:** 10 وحدات تعليمية / educational units (missed default)
- **تكلفة FP / FP Cost:** 1 وحدة تعليمية / educational unit (unnecessary review)
- **سقف المراجعة / Capacity Ceiling:** 12% per validation period (hard constraint)
- **معادلة الخسارة / Loss Formula:** `Loss = 10 × FN + 1 × FP`

ليست ريالات فعلية أو رسوم أدوات أو خصمًا من الدرجة · Not real SAR, tool fees, or grade deductions.

---

## دليل السعة — Capacity Evidence (3 Forward Folds)

| الطية / Fold | الصفوف / Rows | السعة / Capacity | المُشار إليها / Flagged | نسبة الإشارة / Flag Rate | ضمن السقف / Within Cap |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 732 | 87 | 86 | 11.75% | ✓ |
| 2 | 734 | 88 | 84 | 11.44% | ✓ |
| 3 | 689 | 82 | 75 | 10.89% | ✓ |

جميع الفترات ضمن السقف التشغيلي · All periods within the 12% operational ceiling.

---

## التحقق الإقليمي / Regional Audit (Descriptive — OOF)

| المنطقة / Region | الصفوف | الإشارات | TP | FP | FPR | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Central | 537 | 63 | 24 | 39 | 7.98% | 50.00% |
| Western | 533 | 71 | 21 | 50 | 10.29% | 44.68% |
| Eastern | 542 | 48 | 16 | 32 | 6.31% | 45.71% |
| Other | 543 | 63 | 23 | 40 | 8.10% | 46.94% |

فجوة FPR (Western − Eastern): **3.98 نقطة مئوية** · هذا فرق وصفي لا يُثبت أو ينفي التحيز · Descriptive gap; does not prove or disprove bias.

---

## لماذا اخترت هذه العتبة؟ / Why This Threshold?

تم اختيار العتبة لأنها تعطي أقل خسارة مع استيفاء قيد السعة في كل فترة تحقق.  
The selected threshold was chosen because it yields the lowest cost-weighted loss while satisfying the 12% capacity constraint across all three forward validation periods.

## لماذا Logistic Regression وليس Ensemble؟ / Why Logistic, Not an Ensemble?

بوابة الاختيار تشترط أن يتجاوز رفع AP حد الانحراف المعياري للطيات (0.0298). لم يجتز أي تجميع هذه البوابة.  
The ensemble gate requires AP lift > fold SD (0.0298). No ensemble candidate passed. Decision: **KEEP SINGLE — Logistic Regression** (Mean AP 0.3917 vs. Weighted 0.3894, XGBoost 0.3526, LightGBM 0.3455).

## لماذا قد تخدعك Accuracy؟ / Why Accuracy Misleads?

الدقة الإجمالية تكون مرتفعة حتى حين يُخفق النموذج في رصد جميع حالات التعثر لأنها فئة أقلية. لذا يُعتمد على Recall وPrecision وAP والخسارة والسعة.  
Accuracy can be high even when the model misses all defaults because defaults are a minority class (~8–10% prevalence). Use Recall, Precision, AP, loss, and capacity instead.

## حدود النتيجة / Limitations

هذه نتيجة تطوير مبنية على بيانات اصطناعية ونتائج OOF، لا اختبار نهائي غير ملموس.  
This is a development result on synthetic data using OOF estimates — not an untouched holdout test.

1. **Synthetic data only** — tamweel-lite-1.0; does not represent real individuals or loans.
2. **OOF ≠ final test** — threshold selection and loss estimates use the same OOF targets; the challenge batch has no labels.
3. **Fixed cost ratio** — FN=10, FP=1 and 12% cap are simulation parameters, not production LGD.
4. **Temporal stability** — Continuous PSI monitoring required; thresholds may need recalibration on data shift.
5. **Regional comparison** — Descriptive only; not a legal or causal fairness certification.
6. **Calibration** — Sigmoid (Platt) scaling slightly increases ECE (0.0211 → 0.0349) but enables probability-scale policy transfer.

---

## الأدلة والمراجع / Artifact References

| الملف / File | المحتوى / Contents |
|:---|:---|
| `artifacts/final_metrics.json` | OOF threshold selection, calibration diagnostics, challenge batch audit |
| `artifacts/final_policy.json` | Calibrated threshold, Platt mapping parameters, batch audit |
| `artifacts/ensemble_comparison.csv` | Multi-model AP comparison across 3 folds |
| `artifacts/day5_period_capacity.csv` | Per-fold capacity evidence |
| `artifacts/day5_region_audit.csv` | Regional FPR / recall breakdown |
| `artifacts/permutation_importance.csv` | Feature importance (AP drop, 3 repeats) |
| `artifacts/day5_calibration_fit.png` | Calibration reliability diagram |
| `artifacts/cost_curve.png` | Cost vs. threshold sweep |
| `artifacts/day5_policy_regions.png` | Capacity regions across periods |
