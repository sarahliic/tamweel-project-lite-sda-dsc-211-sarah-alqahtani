# قرار التجميع

KEEP SINGLE — Logistic

Based on our worth-it gate evaluation, the single Logistic Regression model was selected because it achieved the highest mean Average Precision of 0.39166 (fold SD of 0.02981) across the three forward folds, while all ensembles failed to pass the gate (Equal: 0.37170, Weighted: 0.38942, Stack: 0.38314). The ensembles did not provide any AP lift over the best single model, and their slightly higher complexity was not justified, especially given the extremely high correlation in residuals (Pearson r of 0.98 to 0.99) indicating lack of error diversity.

Our nested forward validation used three chronologically split folds (Period 1, 2, and 3) where validation customers were strictly excluded from model training to prevent leakage. This realistic testing environment showed that our single model selection is stable, although validation metrics represent historical representation rather than an untouched, independent test set for future temporal changes.

الدليل: artifacts/ensemble_comparison.csv وday5_ensemble_gate.json. SD وصفي، وليس اختبار دلالة.
