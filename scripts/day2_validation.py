"""Time, customer and label-maturity guards for the Day 2 teaching lab."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

OUTER_WINDOWS = (("2023-07-01", "2024-01-01"), ("2024-01-01", "2024-07-01"),
                 ("2024-07-01", "2025-01-01"))
TUNING_WINDOWS = (("2022-10-01", "2023-01-01"), ("2023-01-01", "2023-03-01"),
                  ("2023-03-01", "2023-04-02"))


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 211
    n_jobs: int = 2
    max_trees: int = 200
    patience: int = 20
    trials: int = 8
    timeout_seconds: float = 120.0

    def __post_init__(self):
        if not (1 <= self.n_jobs <= 2 and 0 <= self.trials <= 8 and 0 <= self.timeout_seconds <= 120):
            raise ValueError("Use up to 8 trials, 120 seconds and 2 CPU threads.")
        if not (1 <= self.patience < self.max_trees <= 600):
            raise ValueError("Use 1 <= patience < max_trees <= 600.")


BASE_PARAMS = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30}


def leakage_audit(frame, dictionary):
    """Audit availability metadata, not a guessed correlation threshold."""
    if dictionary.name.duplicated().any():
        raise ValueError("Dictionary feature names must be unique.")
    lookup = dictionary.set_index("name")
    rows = []
    for name in frame.columns:
        if name not in lookup.index:
            role, available, action = "unknown", "unknown", "BLOCK_UNKNOWN"
        else:
            role, available = lookup.loc[name, ["role", "available_at"]]
            action = ("KEEP" if role == "predictor" and available == "application_date"
                      else "DROP_POST_OUTCOME" if role == "leak" or available == "after_application"
                      else "EXCLUDE_METADATA_OR_TARGET")
        rows.append({"feature": name, "role": role, "available_at": available, "action": action})
    return pd.DataFrame(rows)


def deduplicate_applications(frame):
    """Keep one identical event; refuse conflicting copies of the same ID."""
    required = ["application_id", "customer_id", "application_date"]
    if frame[required].isna().any().any():
        raise ValueError("Application ID, customer and date must be complete.")
    dates = pd.to_datetime(frame.application_date, errors="raise")
    if dates.isna().any():
        raise ValueError("Application dates must be valid.")
    repeated = frame[frame.application_id.duplicated(keep=False)]
    for application_id, group in repeated.groupby("application_id", sort=False):
        if len(group.drop_duplicates()) != 1:
            raise ValueError(f"Conflicting duplicate application: {application_id}")
    clean = frame.sort_values(["application_date", "application_id"], kind="stable").copy()
    clean = clean.drop_duplicates().drop_duplicates(subset=[c for c in frame if c != "application_id"])
    return clean.reset_index(drop=True), {"input_rows": len(frame), "retained_rows": len(clean),
                                          "duplicate_rows_removed": len(frame) - len(clean)}


def feature_frame(frame, contract):
    """Require the exact prediction schema, so leaked columns cannot slip in."""
    names = [item["name"] for item in contract["features"]]
    if set(frame.columns) != set(names) or len(frame.columns) != len(names):
        raise ValueError("Use only the approved application-time predictors; extra or missing columns rejected.")
    return frame.loc[:, names]


def safe_predict(model, frame, contract):
    return model.predict_proba(feature_frame(frame, contract))[:, 1]


def _enough(frame, target, label, minimum):
    counts = frame[target].value_counts()
    if len(frame) < minimum or set(counts.index) != {0, 1} or counts.min() < 5:
        raise ValueError(f"{label}: too few rows or class examples (need {minimum} rows and 5 per class).")


def _frame_guard(frame):
    if not frame.index.is_unique or not frame.application_id.is_unique:
        raise ValueError("Deduplicate application IDs before splitting.")
    if frame[["customer_id", "application_date"]].isna().any().any():
        raise ValueError("Customer and date must be complete.")
    return pd.to_datetime(frame.application_date, errors="raise")


def make_time_group_split(frame, windows=OUTER_WINDOWS, horizon_days=90,
                         target="default_within_90d"):
    """Expanding past, strict mature labels, purge current validation customers."""
    dates = _frame_guard(frame)
    if horizon_days <= 0:
        raise ValueError("The outcome horizon must be positive.")
    folds, seen, previous_end = [], set(), None
    for number, (start, end) in enumerate(windows, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start >= end or (previous_end is not None and start < previous_end):
            raise ValueError("Validation windows must be ordered, nonempty and nonoverlapping.")
        valid = frame[(dates >= start) & (dates < end)]
        past = dates < start
        mature = dates + pd.Timedelta(days=horizon_days) < start
        overlap = frame.customer_id.isin(valid.customer_id)
        train = frame[mature & ~overlap]
        _enough(train, target, "Training fold", 100)
        _enough(valid, target, "Validation fold", 40)
        if seen.intersection(valid.index):
            raise ValueError("A validation row cannot receive two OOF predictions.")
        seen.update(valid.index)
        info = {"fold": number, "validation_start": str(start.date()), "validation_end_exclusive": str(end.date()),
                "train_rows": len(train), "validation_rows": len(valid),
                "train_positives": int(train[target].sum()), "validation_positives": int(valid[target].sum()),
                "immature_past_rows_removed": int((past & ~mature).sum()),
                "customer_overlap_rows_removed": int((mature & overlap).sum()),
                "shared_customers": 0,
                "latest_training_label_available": str((pd.to_datetime(train.application_date) + pd.Timedelta(days=horizon_days)).max().date())}
        folds.append({"train": train.index.to_numpy(), "valid": valid.index.to_numpy(), "audit": info})
        previous_end = end
    return folds


def make_inner_stop(train, horizon_days=90, target="default_within_90d"):
    """Choose an inner date boundary using training metadata only."""
    dates = _frame_guard(train)
    boundary = dates.quantile(0.75).normalize()
    stop = train[dates >= boundary]
    mature = dates + pd.Timedelta(days=horizon_days) < boundary
    fit = train[mature & ~train.customer_id.isin(stop.customer_id)]
    _enough(fit, target, "Inner fitting set", 100)
    _enough(stop, target, "Inner stopping set", 40)
    return fit.index.to_numpy(), stop.index.to_numpy(), str(boundary.date())


def reserve_tuning_pool(frame, outer_folds, horizon_days=90):
    """Freeze search before outer periods, with no evaluated customer in search."""
    first = pd.Timestamp(outer_folds[0]["audit"]["validation_start"])
    validation_ids = np.concatenate([fold["valid"] for fold in outer_folds])
    excluded_customers = set(frame.loc[validation_ids, "customer_id"])
    mature = pd.to_datetime(frame.application_date) + pd.Timedelta(days=horizon_days) < first
    pool = frame[mature & ~frame.customer_id.isin(excluded_customers)].copy()
    return pool


class BudgetExpired(RuntimeError):
    """A cooperative CPU search deadline has been reached."""


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise BudgetExpired("Search time budget reached; retain only completed trials.")


def _booster(params, config, trees):
    return LGBMClassifier(n_estimators=int(trees), objective="binary", metric="binary_logloss",
                          **params, random_state=config.seed, n_jobs=config.n_jobs,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          deterministic=True, force_col_wise=True, verbosity=-1)


def fit_fixed_model(train, contract, params, config, trees=80):
    """A fixed tree budget for comparisons that change only the data protocol."""
    started = perf_counter()
    features = [f["name"] for f in contract["features"]]
    target = contract["target"]["name"]
    if set(train) != set(features) | set(contract["metadata"]) | {target}:
        raise ValueError("Training schema contains unapproved or missing fields.")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                         ("model", _booster(params, config, trees))])
    pipeline.fit(feature_frame(train[features], contract), train[target])
    return pipeline, {"selected_trees": trees, "train_seconds": perf_counter() - started,
                      "refit_imputer_medians": pipeline.named_steps["imputer"].statistics_.tolist(),
                      "tree_selection": "fixed_in_advance"}


def fit_honest_model(train, contract, params, config, deadline=None):
    """Accept training rows only; select trees internally, then refit all training."""
    _check_deadline(deadline)
    started = perf_counter()
    target = contract["target"]["name"]
    features = [f["name"] for f in contract["features"]]
    allowed = set(features) | set(contract["metadata"]) | {target}
    if set(train.columns) != allowed:
        raise ValueError("Training schema contains unapproved or missing fields.")
    fit_ids, stop_ids, boundary = make_inner_stop(train, contract["target"]["horizon_days"], target)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")
    X_fit = imputer.fit_transform(feature_frame(train.loc[fit_ids, features], contract))
    X_stop = imputer.transform(feature_frame(train.loc[stop_ids, features], contract))
    def deadline_callback(environment):
        _check_deadline(deadline)
    selection = _booster(params, config, config.max_trees)
    selection.fit(X_fit, train.loc[fit_ids, target], eval_set=[(X_stop, train.loc[stop_ids, target])],
                  callbacks=[early_stopping(config.patience, first_metric_only=True, verbose=False), deadline_callback])
    trees = int(selection.best_iteration_)
    _check_deadline(deadline)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
                         ("model", _booster(params, config, trees))])
    pipeline.fit(feature_frame(train[features], contract), train[target], model__callbacks=[deadline_callback])
    details = {"selected_trees": trees, "train_seconds": perf_counter() - started,
               "inner_fit_rows": len(fit_ids), "inner_stop_rows": len(stop_ids), "inner_stop_start": boundary,
               "inner_fit_ids": train.loc[fit_ids, "application_id"].tolist(),
               "inner_stop_ids": train.loc[stop_ids, "application_id"].tolist(),
               "inner_imputer_medians": imputer.statistics_.tolist(),
               "refit_imputer_medians": pipeline.named_steps["imputer"].statistics_.tolist()}
    return pipeline, details


def score_probabilities(y, probabilities):
    p = np.asarray(probabilities)
    if set(pd.Series(y).unique()) != {0, 1} or not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Scores require both target classes and finite probabilities in [0, 1].")
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p))}


def tune_on_reserved_pool(pool, contract, config=ValidationConfig()):
    """Three forward folds; search labels and customers never enter outer evaluation."""
    started = perf_counter()
    deadline = started + config.timeout_seconds
    folds = make_time_group_split(pool, TUNING_WINDOWS, contract["target"]["horizon_days"])
    features = [f["name"] for f in contract["features"]]
    target = contract["target"]["name"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    def objective(trial):
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.025, 0.12, log=True),
                  "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31]),
                  "min_child_samples": trial.suggest_categorical("min_child_samples", [10, 20, 40])}
        scores, selected_trees = [], []
        for step, fold in enumerate(folds):
            _check_deadline(deadline)
            model, info = fit_honest_model(pool.loc[fold["train"]], contract, params, config, deadline)
            p = safe_predict(model, pool.loc[fold["valid"], features], contract)
            scores.append(score_probabilities(pool.loc[fold["valid"], target], p)["average_precision"])
            selected_trees.append(info["selected_trees"])
            trial.set_user_attr("fold_ap", scores.copy())
            trial.set_user_attr("selected_trees", selected_trees.copy())
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    if config.timeout_seconds > 0 and config.trials > 0:
        study.optimize(objective, n_trials=config.trials, timeout=config.timeout_seconds, n_jobs=1,
                       catch=(BudgetExpired,), show_progress_bar=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = []
    for trial in study.trials:
        rows.append({"trial": trial.number, "state": trial.state.name,
                     "mean_ap": trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None,
                     "seconds": trial.duration.total_seconds(), **trial.params,
                     **{f"fold_{n+1}_ap": value for n, value in enumerate(trial.user_attrs.get("fold_ap", []))}})
    history = pd.DataFrame(rows, columns=["trial", "state", "mean_ap", "seconds", "learning_rate", "num_leaves",
        "min_child_samples", "fold_1_ap", "fold_2_ap", "fold_3_ap"])
    elapsed = perf_counter() - started
    result = {"status": "COMPLETE" if completed else "NO_COMPLETED_TRIAL", "source": "LIVE",
              "best_params": study.best_params if completed else None,
              "best_tuning_ap": float(study.best_value) if completed else None,
              "completed_trials": len(completed), "attempted_trials": len(study.trials),
              "elapsed_seconds": elapsed, "seed": config.seed, "max_trials": config.trials,
              "timeout_seconds": config.timeout_seconds, "folds": 3,
              "stopping_reason": "TIME_BUDGET" if elapsed >= config.timeout_seconds else "TRIAL_LIMIT",
              "tuning_rows": len(pool), "tuning_application_ids": pool.application_id.tolist(),
              "fold_audits": [f["audit"] for f in folds]}
    return result, history


def negative_controls(clean, dirty, contract, config=ValidationConfig()):
    """Intentionally unsafe comparison; never use these predictions downstream."""
    features = [f["name"] for f in contract["features"]]
    leaked = features + [f["name"] for f in contract["leakage_features"]]
    dirty = dirty.set_index("application_id").loc[clean.application_id].reset_index()
    y = clean[contract["target"]["name"]]
    splits = list(StratifiedKFold(3, shuffle=True, random_state=config.seed).split(clean, y))
    # Deliberate negative control: all-row imputation AND post-outcome features.
    unsafe = SimpleImputer(strategy="median").set_output(transform="pandas").fit_transform(dirty[leaked])
    rows = []
    for number, (training, validation) in enumerate(splits, 1):
        for scheme in ("leaky_random_control", "clean_random_control"):
            started = perf_counter()
            if scheme == "leaky_random_control":
                model = _booster(BASE_PARAMS, config, 80).fit(unsafe.iloc[training], y.iloc[training])
                train_seconds = perf_counter() - started
                p = model.predict_proba(unsafe.iloc[validation])[:, 1]
            else:
                model = Pipeline([("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                                  ("model", _booster(BASE_PARAMS, config, 80))])
                model.fit(clean.iloc[training][features], y.iloc[training])
                train_seconds = perf_counter() - started
                p = safe_predict(model, clean.iloc[validation][features], contract)
            rows.append({"scheme": scheme, "fold": number, **score_probabilities(y.iloc[validation], p),
                         "validation_rows": len(validation), "train_rows": len(training),
                         "selected_trees": 80, "train_seconds": train_seconds,
                         "shared_customers": len(set(clean.iloc[training].customer_id) & set(clean.iloc[validation].customer_id)),
                         "prediction_source": "NEGATIVE_CONTROL_NOT_FOR_DECISIONS"})
    return pd.DataFrame(rows)


def evaluate_outer(frame, contract, folds, params, config, scheme, fixed_trees=None):
    features = [f["name"] for f in contract["features"]]
    target = contract["target"]["name"]
    rows, predictions, provenance = [], [], []
    for fold in folds:
        train, valid = frame.loc[fold["train"]], frame.loc[fold["valid"]]
        if fixed_trees is None:
            model, info = fit_honest_model(train, contract, params, config)
        else:
            model, info = fit_fixed_model(train, contract, params, config, fixed_trees)
        p = safe_predict(model, valid[features], contract)
        rows.append({"scheme": scheme, **fold["audit"], **score_probabilities(valid[target], p),
                     "selected_trees": info["selected_trees"], "train_seconds": info["train_seconds"],
                     "prediction_source": "LIVE_OUT_OF_FOLD"})
        predictions.append(valid[["application_id", "customer_id", "application_date", target]].assign(
            fold=fold["audit"]["fold"], scheme=scheme, probability=p))
        provenance.append({"scheme": scheme, "fold": fold["audit"]["fold"],
                           "training_application_ids": train.application_id.tolist(),
                           "validation_application_ids": valid.application_id.tolist(), **info})
    predictions = pd.concat(predictions, ignore_index=True)
    if predictions.application_id.duplicated().any():
        raise ValueError("Duplicate OOF predictions are not allowed.")
    return pd.DataFrame(rows), predictions, provenance


def summarize_folds(report):
    return report.groupby("scheme", sort=False).agg(
        folds=("fold", "count"), validation_rows=("validation_rows", "sum"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        ap_mean=("average_precision", "mean"), ap_sd=("average_precision", "std")).reset_index()


def reflection_check(responses):
    required = ("leak_example", "split_reason", "observed_gap", "imputer_reason", "limitations")
    missing = [key for key in required if not isinstance(responses.get(key), str) or not responses[key].strip()]
    return {"status": "LEARNER_WORK_REQUIRED" if missing else "READY_FOR_REVIEW",
            "missing": missing, "automatic_grade": None}
