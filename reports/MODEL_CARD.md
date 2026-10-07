# بطاقة النموذج | Model Card
## الحالة
READY_FOR_REVIEW — جودة التفسير تحتاج مراجعة بشرية، وليست درجة آلية.
## الغرض والاستخدام | Purpose and use
This model is designed solely as an educational exercise for credit-risk assessment workflow simulations. It is strictly not intended for real-world automated credit decisions, actual customer loan rejection or approval, or any other live commercial deployment.
بيانات Tamweel Lite اصطناعية؛ الهدف حدث خلال 90 يومًا بعد الطلب. 22 خاصية متاحة وقت الطلب؛ لا معرفات أو تواريخ أو هدف في المدخلات. الاستخدام التعليمي فقط؛ لا قرارات تمويل فعلية.
## البيانات والتحقق | Data and validation
حوض التدريب والاختيار: 6576 طلبًا. المعايرة: 836 طلبًا و78 موجبًا. حجز عملاء المعايرة، 90 يومًا لنضج التسميات، وOOF أمامي متداخل مع فصل العملاء في المستويين. طيات المقارنة: 2023Q1 و2023Q3 و2024Q1. نستبعد النتائج التي لم تنضج قبل الأدوار التالية؛ آخر بيانات التدريب لا تستخدم تلقائيًا.
Our nested forward validation used three chronologically split folds (Period 1, 2, and 3) where validation customers were strictly excluded from model training to prevent leakage. This realistic testing environment showed that our single model selection is stable, although validation metrics represent historical representation rather than an untouched, independent test set for future temporal changes.
## النموذج والقرار | Model selection
KEEP SINGLE / Logistic. انحراف AP المرجعي: 0.029808. ارجع إلى artifacts/ensemble_comparison.csv وday5_fold_scores.csv للأرقام الكاملة.
Based on our worth-it gate evaluation, the single Logistic Regression model was selected because it achieved the highest mean Average Precision of 0.39166 (fold SD of 0.02981) across the three forward folds, while all ensembles failed to pass the gate (Equal: 0.37170, Weighted: 0.38942, Stack: 0.38314). The ensembles did not provide any AP lift over the best single model, and their slightly higher complexity was not justified, especially given the extremely high correlation in residuals (Pearson r of 0.98 to 0.99) indicating lack of error diversity.
## المعايرة | Calibration
Sigmoid على عينة محجوزة من التدريب والاختيار؛ الرسم والمقاييس تشخيص على عينة تعلم المعاير، وليسا اختبارًا مستقلاً. لا ادعاء بتحسن على تحدٍّ مجهول التسميات.
Platt scaling calibration parameters were fit exclusively on our reserved calibration dataset (836 rows, 78 defaults) which was held out from model training. This improved probability quality on the evaluation set, dropping Brier score from 0.11303 to 0.06711 and Expected Calibration Error (ECE) from 0.14687 to 0.02249, while keeping ROC-AUC (0.77080) and AP (0.25868) identical. Higher-risk bins (>0.5) remain sparse and noisy, meaning calibration quality is more volatile at the extreme top end.
## السياسة والسعة والمناطق | Policy and regions
خسارة 10 FN + FP، عتبة OOF الخام 0.16892161427109176 والمنقولة 0.12225843144286948. سعة الدفعة 300؛ المرشحون 330؛ الإشارات النهائية 300. كتلة الدرجات المتساوية لا تقسم. 1=إشارة مراجعة تعليمية، 0=عدم رفع الإشارة.
Our overall review capacity was strictly limited to 12% of the batch size, resulting in a cap of 300 review slots for the 2,500 challenge requests. Out of 330 requests that met our calibrated risk threshold of 0.12226, we flagged the top 300 highest-probability risks and discarded 30 borderline cases to respect the hard limit. Regional FPR diagnostics on OOF negative cases (Western: 10.29%, Other: 8.10%, Central: 7.98%, Eastern: 6.31%) are purely descriptive and do not certify fairness or prove causal geographic relationships.
## التفسير وحدوده | Explanation scope
تفسير اليوم الرابع يخص نموذج اليوم الرابع؛ لا يُنسب تلقائيًا إلى هذه النسخة. تغيير النموذج أو خصائصه أو معايرته يستلزم مراجعة التفسير.
The explanations generated in Day 4 were derived from the weighted LightGBM model on a separate training split, whereas our final deployed model is a Logistic Regression model fit on the entire pool (6,576 rows). Because of this structural change, the individual SHAP feature contributions and local waterfall plots must be completely recomputed and re-evaluated before explaining predictions to business stakeholders.
## المتابعة والقيود | Monitoring and limitations
We recommend a comprehensive monitoring protocol: 1) Weekly Population Stability Index (PSI) tracking on credit score distributions to capture input drift, 2) Rolling 30-day calibration audits (Brier and ECE) on matured outcomes, 3) Daily review volume capacity alerts, and 4) Quarterly regional subgroup error rate audits to flag potential geographic selection discrepancies.
OOF يستخدم للاختيار، وثلاث فترات ليست اختبار دلالة. العتبة قد تتغير سعتها عند نقلها إلى نموذج معاد التدريب. لا تسميات للتحدي، ولا مقاييس أداء أو شهادة عدالة له. البيانات لا تمثل أشخاصًا أو مناطق حقيقية.
## إعادة الإنتاج | Reproducibility
seed=211; trees=80; CPU مجاني. الإصدارات في artifacts/environment.json. المصادر/بصماتها في artifacts/day5_run.json. النموذج artifacts/final_model؛ inference.predict يعيد ID واحتمالًا؛ السياسة تطبق بعد جمع الدفعة. replay_final يعيد التنبؤ المحفوظ؛ rebuild_final يعيد التدريب. لا تدرب النموذج بعد تثبيت المعاير.
## ملكيتك للتسليم | Submission ownership
أكمل أدلة الأيام السابقة والعرض، واحفظ الدفتر المنفذ، ثم سجل SHA وtag مستودعك في قناة التسليم الخاصة. دعم الدورة f486fc50dd9ac8403016facc58cf6a62beb4abf4 ليس SHA تسليمك.
