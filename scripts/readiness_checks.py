"""Small compatibility checks, not the assessed day-one modelling exercise."""
from __future__ import annotations

import json
from pathlib import Path
import time


def check_runtime(root: Path) -> dict:
    started = time.perf_counter()
    import os
    os.environ['MPLCONFIGDIR'] = str(Path(root).resolve() / '.cache/matplotlib')
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from threadpoolctl import threadpool_limits
    import shap
    import optuna

    x, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               weights=[.8, .2], random_state=211)
    x = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(6)])
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=.25, random_state=211)
    models = {
        'logistic': make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=250, random_state=211)),
        'xgboost': XGBClassifier(n_estimators=12, max_depth=2, n_jobs=2, random_state=211, tree_method='hist'),
        'lightgbm': LGBMClassifier(n_estimators=12, max_depth=2, num_leaves=4, n_jobs=2, random_state=211, verbosity=-1),
    }
    checks = {}
    with threadpool_limits(limits=2):
        for name, model in models.items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            if not (np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()):
                raise ValueError(f'{name}: invalid probabilities.')
            checks[name] = 'PASS'
        explanation = shap.TreeExplainer(models['xgboost'])(x_test.iloc[:8])
        margin = models['xgboost'].predict(x_test.iloc[:8], output_margin=True)
        np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), margin, atol=1e-5)
        checks['shap_margin_additivity'] = 'PASS'
        calibrated = CalibratedClassifierCV(LogisticRegression(max_iter=250), method='sigmoid', cv=3)
        calibrated.fit(x_train, y_train)
        value = brier_score_loss(y_test, calibrated.predict_proba(x_test)[:, 1])
        if not np.isfinite(value):
            raise ValueError('Calibration compatibility check failed.')
        checks['sigmoid_calibration'] = 'PASS'
        sampler = optuna.samplers.RandomSampler(seed=211)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(sampler=sampler)
        study.optimize(lambda trial: (trial.suggest_float('x', -1, 1) - .2) ** 2, n_trials=1)
        checks['optuna'] = 'PASS'
    report = {'checks': checks, 'seconds': round(time.perf_counter() - started, 2),
              'sample_rows': 300, 'purpose': 'Software compatibility only; not project model results.'}
    target = Path(root) / 'artifacts/runtime_checks.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, status in checks.items():
        print(f'{name}: {status}')
    return report
