"""Reusable, inspectable operations for the Day 1 teaching comparison."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, record_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ("Logistic Regression", "XGBoost", "LightGBM")


@dataclass(frozen=True)
class LabConfig:
    seed: int = 211
    n_jobs: int = 2
    max_trees: int = 300
    patience: int = 30
    learning_rate: float = 0.05
    xgb_depth: int = 3
    lgb_leaves: int = 15
    min_child_samples: int = 50

    def __post_init__(self):
        if not (0 < self.learning_rate <= 1 and 1 <= self.n_jobs <= 2):
            raise ValueError("Use a learning rate in (0, 1] and 1 or 2 CPU threads.")
        if not (1 <= self.patience < self.max_trees <= 1200):
            raise ValueError("Use 1 <= patience < max_trees <= 1200.")
        if not (1 <= self.xgb_depth <= 8 and 2 <= self.lgb_leaves <= 63 and self.min_child_samples >= 2):
            raise ValueError("Keep tree complexity within the CPU lab limits.")


def make_splits(frame, contract, seed=211):
    """Random stratification is a Day 1 demonstration, not final validation."""
    target = contract["target"]["name"]
    features = [item["name"] for item in contract["features"]]
    if not frame.index.is_unique or not frame.application_id.is_unique:
        raise ValueError("Rows and application IDs must be unique.")
    forbidden = set(contract["metadata"]) | {target}
    if forbidden.intersection(features):
        raise ValueError("Metadata and target cannot be predictors.")
    X, y = frame[features].copy(), frame[target].copy()
    if set(y.unique()) != {0, 1} or y.value_counts().min() < 10:
        raise ValueError("Both binary classes need at least 10 examples.")
    development, comparison = train_test_split(
        frame.index.to_numpy(), test_size=0.2, stratify=y, random_state=seed)
    inner_fit, inner_stop = train_test_split(
        development, test_size=0.25, stratify=y.loc[development], random_state=seed + 1)
    splits = {"development": development, "comparison": comparison,
              "inner_fit": inner_fit, "inner_stop": inner_stop}
    return X, y, splits


def make_booster(name, config, n_estimators, stopping=False):
    common = dict(n_estimators=int(n_estimators), learning_rate=config.learning_rate,
                  random_state=config.seed, n_jobs=config.n_jobs, subsample=0.8,
                  colsample_bytree=0.8)
    if name == "XGBoost":
        params = dict(tree_method="hist", max_depth=config.xgb_depth,
                      objective="binary:logistic", eval_metric="logloss", **common)
        if stopping:
            params["early_stopping_rounds"] = config.patience
        return XGBClassifier(**params)
    if name == "LightGBM":
        return LGBMClassifier(objective="binary", metric="binary_logloss",
                              num_leaves=config.lgb_leaves,
                              min_child_samples=config.min_child_samples,
                              subsample_freq=1, deterministic=True,
                              force_col_wise=True, verbosity=-1, **common)
    raise ValueError(f"Unknown boosting model: {name}")


def select_tree_count(name, X_fit, y_fit, X_stop, y_stop, config):
    """Only inner rows enter this operation; comparison data are never accepted."""
    if set(X_fit.index).intersection(X_stop.index):
        raise ValueError("Early-stopping rows must be separate from fitting rows.")
    started = perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fitted = pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns, index=X_fit.index)
    stopped = pd.DataFrame(imputer.transform(X_stop), columns=X_fit.columns, index=X_stop.index)
    model = make_booster(name, config, config.max_trees, stopping=True)
    if name == "XGBoost":
        model.fit(fitted, y_fit, eval_set=[(fitted, y_fit), (stopped, y_stop)], verbose=False)
        history = model.evals_result()
        curves = {"fit": history["validation_0"]["logloss"],
                  "stop": history["validation_1"]["logloss"]}
        selected_trees = int(model.best_iteration + 1)  # XGBoost starts at zero.
    else:
        history = {}
        model.fit(fitted, y_fit, eval_set=[(fitted, y_fit), (stopped, y_stop)],
                  eval_names=["fit", "stop"], eval_metric="binary_logloss",
                  callbacks=[early_stopping(config.patience, first_metric_only=True, verbose=False),
                             record_evaluation(history)])
        curves = {part: history[part]["binary_logloss"] for part in ("fit", "stop")}
        selected_trees = int(model.best_iteration_)
    return {"model": name, "selected_trees": selected_trees,
            "selection_seconds": perf_counter() - started,
            "rounds_evaluated": len(curves["stop"]), "curves": curves,
            "inner_imputer_medians": imputer.statistics_.tolist(),
            "hit_tree_budget": selected_trees == config.max_trees}


def fit_model(name, X_development, y_development, config, selection=None):
    """Fit a final candidate on development rows after freezing tree count."""
    started = perf_counter()
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "Logistic Regression":
        steps += [("scale", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000, random_state=config.seed))]
    elif name in MODEL_NAMES[1:]:
        if selection is None or selection["model"] != name:
            raise ValueError("Freeze this model's tree count using the inner split first.")
        # DataFrame output preserves feature names across LightGBM fit/predict.
        steps[0][1].set_output(transform="pandas")
        steps.append(("model", make_booster(name, config, selection["selected_trees"])))
    else:
        raise ValueError(f"Unknown model: {name}")
    pipeline = Pipeline(steps).fit(X_development, y_development)
    refit_seconds = perf_counter() - started
    timing = {"selection_seconds": 0.0 if selection is None else selection["selection_seconds"],
              "refit_seconds": refit_seconds}
    timing["train_seconds"] = timing["selection_seconds"] + refit_seconds
    return pipeline, timing


def compare_models(models, timings, selections, X_comparison, y_comparison):
    """Score frozen models once. AP is average precision, not trapezoidal PR area."""
    if set(models) != set(MODEL_NAMES) or set(y_comparison.unique()) != {0, 1}:
        raise ValueError("Comparison needs all three models and both target classes.")
    rows, probabilities = [], pd.DataFrame(index=X_comparison.index)
    for name in MODEL_NAMES:
        started = perf_counter()
        p = models[name].predict_proba(X_comparison)[:, 1]
        predict_seconds = perf_counter() - started
        if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
            raise ValueError("Predictions must be finite probabilities in [0, 1].")
        probabilities[name] = p
        selected = selections.get(name)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_comparison, p),
                     "average_precision": average_precision_score(y_comparison, p),
                     **timings[name], "predict_seconds": predict_seconds,
                     "selected_trees": None if selected is None else selected["selected_trees"],
                     "comparison_rows": len(y_comparison),
                     "positive_rate": float(y_comparison.mean())})
    return pd.DataFrame(rows), probabilities


def learner_checkpoint(candidate, notes, problem_statement, exit_answers):
    """Completeness reminder only. It neither judges reasoning nor awards marks."""
    missing = []
    if candidate not in MODEL_NAMES:
        missing.append("Choose one of the three model names.")
    for key in ("evidence", "limitation", "next_test"):
        if not isinstance(notes.get(key), str) or not notes[key].strip():
            missing.append(f"Write your {key} note.")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        missing.append("Write your problem statement.")
    if len(exit_answers) != 2 or any(not isinstance(a, str) or not a.strip() for a in exit_answers):
        missing.append("Answer both exit questions.")
    return {"status": "LEARNER_WORK_REQUIRED" if missing else "READY_FOR_REVIEW",
            "missing": missing, "automatic_grade": None}


def config_dict(config):
    return asdict(config)
