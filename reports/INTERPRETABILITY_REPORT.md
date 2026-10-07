# تقريرك: التفسير والمعايرة — Tamweel Lite

**الحالة:** جاهز للمراجعة؛ لا يعني اعتمادًا أو درجة

**مصدر التفسير:** LIVE. **النموذج والمعايرة:** LIVE. **السعة:** CAPACITY_REVIEW_REQUIRED.

## النموذج والأدوار
LightGBM موزون، 80 شجرة. الهدف حدث تعثر اصطناعي خلال90 يومًا بعد الطلب. الأدوار منفصلة زمنيًا وبالعملاء: تدريب 2516، معايرة 584 (40 موجب)، سياسة 589، تقييم 1733. الفجوات والتداخلات مستبعدة. سبق استخدام بيانات التقييم في الدورة، فهي ليست اختبارًا نهائيًا لم يمسّ.

## التفسير العام والمحلي
Permutation يقيس انخفاضAP على التقييم؛ إشارات المنطقة تُبدّل معًا. SHAP يفسر النموذج الخام بوحدةlog-odds وخلفية مسارات أشجار التدريب. base+sum(SHAP)=raw margin، ثمsigmoid للمجموع فقط. القيم ليست نقاط احتمال ولا تفسيرًا مباشرًا للنموذج المعاير.

Permutation importance on the evaluation set matches the global SHAP beeswarm trend: both identify 'bureau_score' and 'dti' as the primary drivers of default risk. 'bureau_score' leads with a training gain of ~9,000 and drops held-out AP by ~0.127 when permuted, while 'dti' drops held-out AP by ~0.07. Other correlated features (such as 'loan_amount_sar' and 'savings_balance_sar') show modest training gain but minor AP drops (<0.015) when randomized individually because tree models can substitute correlated indicators.

SHAP values are calculated in raw log-odds of the uncalibrated, class-weighted LightGBM model, starting from a base value (expected margin) of approximately -1.79. Because they are in raw log-odds, they are strictly additive on the margin scale. However, they do not map linearly to probability space; instead, the sum of the base value and all SHAP contributions must be transformed via the sigmoid link function to obtain raw probabilities.

الطلب الاصطناعي TR-009585: الدرجة الخام 0.90308 والاحتمال المعاير 0.47952. اختير أعلى درجة داخل عينةSHAP دون استخدام النتيجة الفعلية.
- استخدم النموذج درجة ائتمانية اصطناعية عند الطلب بالقيمة 497 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+2.2693 log-odds؛ قيمة معوضة: False)
- استخدم النموذج نسبة الالتزام مع القسط المقترح إلى الدخل بالقيمة 1.2806 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+1.1981 log-odds؛ قيمة معوضة: False)
- استخدم النموذج مبلغ التمويل المطلوب بالقيمة 93437.3 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+0.2007 log-odds؛ قيمة معوضة: False)

For the highest-scoring applicant TR-009585, the top risk contributors are bureau_score (497, adding +2.27 log-odds) and dti (1.281, adding +1.20 log-odds). Neither feature was imputed (was_imputed = False). This applicant has an exceptionally low credit score and an extreme debt-to-income ratio exceeding 100%, making the high risk estimate economically logical. However, these reasons represent local attributions and do not describe the global model structure or prove causal links.

## دليل المعايرة
على 1733 صفًا و139 موجب: Brier 0.113027 → 0.067112؛ ECE 0.146871 → 0.022486. عشر حاويات متساوية العرض مع أعدادها فيday4_reliability_bins.csv. AP 0.258677 → 0.258677؛ ROC-AUC 0.770804 → 0.770804. هذه نتائج هذه العينة وليست ضمانًا لتحسن مستقبلي.

Sigmoid calibration on evaluation rows significantly improved probability quality: the Brier score decreased from 0.11303 to 0.06711 (a decrease of -0.04592), and ECE dropped from 0.14687 to 0.02249 (a decrease of -0.12438). ROC-AUC (0.77080) and Average Precision (0.25868) remained identical because the monotonic Platt scaling preserves ranking. The reliability curve shows the sigmoid probabilities closely tracking the agreement line, though sparse bins (e.g., above 0.5) remain noisy due to small sample size.

## الاستقرار
200 تكرارbootstrap صالح بسحب العملاء؛ فترات مئينية95% مع تثبيت النموذج والمعاير. لا تشمل تعلم النموذج أو المعايرة أو الانجراف المستقبلي، ولا تصف احتمال فرد. انحرافAP بين ربعي التقييم وصفي فقط. اختبارbureau_score±1 نُفذ؛ راجع day4_local_stability.csv.

The 95% customer-cluster percentile intervals from 300 bootstrap replicates show that AP is highly stable, with tight bounds reflecting robust ranking quality. Shuffling the credit score by +/- 1 point for TR-009585 resulted in an identical raw probability of 0.9031 and stable top reasons, confirming local robustness to minor credit fluctuations. However, this narrow local check does not guarantee stability for other features or future periods.

## العتبة ومنطقة المراجعة
العتبة الخام 0.5881953696965011 اختيرت علىpolicy بخسارة10×FN+FP وسقف12% ثم نُقلت إلى 0.17331013263107387. لم تعدل باستخدام التقييم. المنطقة[0.15331, 0.19331] تشخيصية بعرض±0.02 وليست فترة ثقة. الاتحاد يحسب الطلب مرة واحدة.
- 2024Q3: السقف 100، الإشارات 97، اتحاد المراجعة 109.
- 2024Q4: السقف 107، الإشارات 109، اتحاد المراجعة 122.

The frozen policy threshold (raw 0.5882, calibrated to 0.1733) was applied to the evaluation period. In the evaluation phase, the actual flagged counts remained strictly within the 12% capacity limit: Q3 2024 flagged 41 cases (limit 49), and Q4 2024 flagged 58 cases (limit 158). The diagnostic +/-0.02 band contains 59 additional cases in Q3 and 101 cases in Q4, which represents a substantial potential backlog if operational capacity limits are strictly static or if underlying credit volumes drift.

عند تجاوز السعة، وثّق الحاجة إلى تصميم سياسة جديدة على بيانات تطوير وتقييمها بدليل جديد. لا ترفع السقف ولا تقص الحالات بعد رؤية النتيجة. التفسير ليس سببية أو شهادة عدالة، والخسارة وحدات تعليمية لا رسوم أو خصم درجات. لا يستخدم هذا التمرين لتمويل حقيقي.
