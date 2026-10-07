"""Out-of-fold imbalance experiments and explicit educational decision rules."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from day2_validation import BASE_PARAMS, feature_frame, make_time_group_split, safe_predict

STRATEGIES = ("unweighted", "weighted", "oversampled")
REGIONS = ("central", "western", "eastern", "other")
TARGET = "default_within_90d"
NO_FLAGS_THRESHOLD = float(np.nextafter(1.0, 2.0))


@dataclass(frozen=True)
class DecisionConfig:
    seed: int = 211
    n_jobs: int = 2
    trees: int = 80
    timeout_seconds: float = 120.0

    def __post_init__(self):
        if not (1 <= self.n_jobs <= 2 and self.trees in (80, 160)
                and np.isfinite(self.timeout_seconds) and 0 <= self.timeout_seconds <= 120):
            raise ValueError("Use 80 or 160 trees, up to 2 CPU threads and 0–120 seconds.")


@dataclass(frozen=True)
class CostPolicy:
    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    max_flag_fraction: float = 0.12

    def __post_init__(self):
        values = [self.false_negative_cost, self.false_positive_cost, self.max_flag_fraction]
        if not np.isfinite(values).all() or min(values) < 0 or self.max_flag_fraction > 1:
            raise ValueError("Costs must be finite and nonnegative; capacity must be in [0, 1].")


def capacity_limit(n, fraction):
    if not isinstance(n, (int, np.integer)) or n < 0 or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Capacity needs a nonnegative integer volume and fraction in [0, 1].")
    return int((Decimal(int(n)) * Decimal(str(fraction))).to_integral_value(rounding=ROUND_FLOOR))


def region_labels(frame):
    columns = [f"region_{name}" for name in REGIONS[:3]]
    values = frame[columns]
    if not values.isin([0, 1]).all().all() or (values.sum(axis=1) > 1).any():
        raise ValueError("Region indicators must be binary and mutually exclusive.")
    result = pd.Series("other", index=frame.index, name="region")
    for name in REGIONS[:3]:
        result.loc[frame[f"region_{name}"].eq(1)] = name
    return result


def _labels(y):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Use a nonempty, one-dimensional binary target without missing values.")
    return values.astype(int)


def positive_weight(y):
    y = _labels(y)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("Training must contain both classes.")
    return (len(y) - positives) / positives


def oversample_positions(y, seed=211):
    """Return original positions plus sampled minority positions; never touch validation."""
    y = _labels(y)
    positive_weight(y)
    zeros, ones = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    minority = ones if len(ones) < len(zeros) else zeros
    count = abs(len(zeros) - len(ones))
    extra = np.random.default_rng(seed).choice(minority, size=count, replace=True)
    return np.concatenate([np.arange(len(y)), extra])


class TrainingBudgetExpired(RuntimeError):
    """No partial experiment is accepted as a complete OOF result."""


def _deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise TrainingBudgetExpired("CPU budget reached. Restart later or view the labeled educational example.")


def fit_strategy(train, contract, strategy, config=DecisionConfig(), deadline=None):
    """Fit imputation on original training rows, then weight or resample training only."""
    _deadline(deadline)
    if strategy not in STRATEGIES:
        raise ValueError("Choose unweighted, weighted or oversampled.")
    features = [f["name"] for f in contract["features"]]
    target = contract["target"]["name"]
    if set(train) != set(features) | set(contract["metadata"]) | {target}:
        raise ValueError("Use the approved training schema without post-outcome fields.")
    if not train.application_id.is_unique or not train.index.is_unique:
        raise ValueError("Deduplicate training before resampling.")
    started = perf_counter()
    y = _labels(train[target])
    ratio = positive_weight(y)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")
    X = imputer.fit_transform(feature_frame(train[features], contract))
    positions = oversample_positions(y, config.seed) if strategy == "oversampled" else np.arange(len(y))
    weight = ratio if strategy == "weighted" else 1.0
    model = LGBMClassifier(**BASE_PARAMS, n_estimators=config.trees, scale_pos_weight=weight,
        objective="binary", random_state=config.seed, n_jobs=config.n_jobs, subsample=.8,
        subsample_freq=1, colsample_bytree=.8, deterministic=True, force_col_wise=True, verbosity=-1)
    def budget_callback(environment):
        _deadline(deadline)
    model.fit(X.iloc[positions], y[positions], callbacks=[budget_callback])
    _deadline(deadline)
    pipeline = Pipeline([("imputer", imputer), ("model", model)])
    info = {"strategy": strategy, "original_training_rows": len(train), "fitted_rows": len(positions),
        "training_positive_rate": float(y.mean()), "fitted_positive_rate": float(y[positions].mean()),
        "negative_positive_ratio": float(ratio), "scale_pos_weight": float(weight),
        "imputer_medians": imputer.statistics_.tolist(), "trees": config.trees,
        "tree_selection": "fixed_before_comparison", "train_seconds": perf_counter() - started,
        "resampled_application_ids": train.iloc[positions].application_id.tolist() if strategy == "oversampled" else []}
    return pipeline, info


def generate_oof(frame, contract, config=DecisionConfig()):
    """Fresh predictions on mature forward/group folds; fail rather than export partial OOF."""
    started = perf_counter()
    deadline = started + config.timeout_seconds
    _deadline(deadline)
    folds = make_time_group_split(frame, horizon_days=contract["target"]["horizon_days"])
    features = [f["name"] for f in contract["features"]]
    predictions, reports, provenance = [], [], []
    regions = region_labels(frame)
    for fold in folds:
        train, valid = frame.loc[fold["train"]], frame.loc[fold["valid"]]
        for strategy in STRATEGIES:
            model, info = fit_strategy(train, contract, strategy, config, deadline)
            p = safe_predict(model, valid[features], contract)
            predictions.append(valid[["application_id", "customer_id", "application_date", TARGET]].assign(
                region=regions.loc[valid.index], fold=fold["audit"]["fold"], strategy=strategy,
                probability=p, source="LIVE", split_role="OOF"))
            reports.append({**fold["audit"], **{k: v for k, v in info.items()
                if k not in ("imputer_medians", "resampled_application_ids")},
                "roc_auc": float(roc_auc_score(valid[TARGET], p)),
                "average_precision": float(average_precision_score(valid[TARGET], p)), "source": "LIVE"})
            provenance.append({**info, "fold": fold["audit"]["fold"],
                "training_application_ids": train.application_id.tolist(),
                "validation_application_ids": valid.application_id.tolist()})
    oof = pd.concat(predictions, ignore_index=True)
    validate_oof(oof, frame, folds)
    return oof, pd.DataFrame(reports), {"source": "LIVE", "elapsed_seconds": perf_counter() - started,
        "folds": [f["audit"] for f in folds], "models": provenance}


def validate_oof(oof, frame, folds):
    required = {"application_id", "customer_id", "application_date", TARGET, "region", "fold",
                "strategy", "probability", "source", "split_role"}
    if not required <= set(oof) or oof[list(required)].isna().any().any():
        raise ValueError("OOF requires complete IDs, fold, target, scores and source metadata.")
    if (set(oof.strategy) != set(STRATEGIES) or oof.duplicated(["strategy", "application_id"]).any()
            or set(oof.split_role) != {"OOF"} or len(set(oof.source)) != 1
            or not set(oof.source) <= {"LIVE", "EDUCATIONAL_EXAMPLE"}):
        raise ValueError("Use one complete, labeled OOF run with unique IDs for all three strategies.")
    if not np.isfinite(oof.probability).all() or not oof.probability.between(0, 1).all():
        raise ValueError("OOF probabilities must be finite and in [0, 1].")
    expected_ids = set(frame.loc[np.concatenate([f["valid"] for f in folds]), "application_id"])
    lookup = frame.set_index("application_id")
    region_lookup = region_labels(frame).set_axis(frame.application_id)
    for _, group in oof.groupby("strategy"):
        if set(group.application_id) != expected_ids or set(group.fold) != {f["audit"]["fold"] for f in folds}:
            raise ValueError("OOF coverage must match the eligible validation rows and folds.")
        actual = lookup.loc[group.application_id]
        for name in ["customer_id", "application_date", TARGET]:
            if not np.array_equal(group[name].to_numpy(), actual[name].to_numpy()):
                raise ValueError("OOF labels or metadata do not match the source data.")
        if not np.array_equal(group.region.to_numpy(), region_lookup.loc[group.application_id].to_numpy()):
            raise ValueError("OOF regions do not match application-time indicators.")
        for fold in folds:
            ids = set(group.loc[group.fold == fold["audit"]["fold"], "application_id"])
            if ids != set(frame.loc[fold["valid"], "application_id"]):
                raise ValueError("An OOF row is assigned to the wrong validation period.")
    return {"eligible_rows": len(expected_ids), "all_rows": len(frame),
            "whole_data_coverage": len(expected_ids) / len(frame), "eligible_coverage": 1.0,
            "warmup_without_oof": len(frame) - len(expected_ids), "source": oof.source.iloc[0]}


def _prediction_arrays(predictions):
    required = {"application_id", "fold", TARGET, "probability", "source", "split_role"}
    if not required <= set(predictions) or predictions.empty or predictions[list(required)].isna().any().any():
        raise ValueError("Provide nonempty, labeled OOF predictions, never the holdout or training predictions.")
    if (predictions.application_id.duplicated().any() or set(predictions.split_role) != {"OOF"}
            or len(set(predictions.source)) != 1 or not set(predictions.source) <= {"LIVE", "EDUCATIONAL_EXAMPLE"}
            or ("strategy" in predictions and predictions.strategy.nunique() != 1)):
        raise ValueError("Choose one strategy from one complete OOF source, without duplicate IDs.")
    y, p = _labels(predictions[TARGET]), predictions.probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return y, p


def threshold_sweep(predictions, policy=CostPolicy()):
    """All distinct >= score rules plus 0.5 and the no-flags sentinel; ties stay together."""
    y, p = _prediction_arrays(predictions)
    thresholds = np.unique(np.r_[0.0, .5, p, NO_FLAGS_THRESHOLD])
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    fn, tn, flags = len(pos)-tp, len(neg)-fp, tp+fp
    feasible = flags <= capacity_limit(len(p), policy.max_flag_fraction)
    worst_fraction = np.zeros(len(thresholds))
    for _, group in predictions.groupby("fold", sort=True):
        scores = np.sort(group.probability.to_numpy())
        count = len(scores) - np.searchsorted(scores, thresholds, side="left")
        feasible &= count <= capacity_limit(len(scores), policy.max_flag_fraction)
        worst_fraction = np.maximum(worst_fraction, count / len(scores))
    loss = policy.false_negative_cost * fn + policy.false_positive_cost * fp
    ratio = lambda a, b: np.divide(a, b, out=np.full(len(thresholds), np.nan), where=np.asarray(b) != 0)
    return pd.DataFrame({"threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": flags, "flag_fraction": flags/len(p), "recall": ratio(tp, len(pos)),
        "precision": ratio(tp, flags), "false_positive_rate": ratio(fp, len(neg)),
        "accuracy": (tp+tn)/len(p), "loss_units": loss, "loss_units_per_10000": loss/len(p)*10000,
        "capacity_feasible": feasible, "max_period_flag_fraction": worst_fraction,
        "rows": len(p), "source": predictions.source.iloc[0]})


def select_threshold(sweep, constrained=True):
    candidates = sweep[sweep.capacity_feasible] if constrained else sweep
    if candidates.empty:
        raise ValueError("No feasible threshold; check the no-flags sentinel and policy.")
    # Preserve equal-score applicants as a block; never use ID to split a tied score.
    return candidates.sort_values(["loss_units", "flagged", "threshold"],
        ascending=[True, True, False], kind="stable").iloc[0].to_dict()


def flag_at(probabilities, threshold):
    p = np.asarray(probabilities, dtype=float)
    if (not np.isfinite(threshold) or not 0 <= threshold <= NO_FLAGS_THRESHOLD
            or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Use valid scores and an exact threshold between 0 and the no-flags sentinel.")
    return p >= threshold


def period_audit(predictions, threshold, policy=CostPolicy()):
    _prediction_arrays(predictions)
    rows = []
    for fold, group in predictions.groupby("fold", sort=True):
        flagged = int(flag_at(group.probability, threshold).sum())
        limit = capacity_limit(len(group), policy.max_flag_fraction)
        rows.append({"fold": fold, "rows": len(group), "capacity": limit, "flagged": flagged,
                     "flag_fraction": flagged/len(group), "within_capacity": flagged <= limit,
                     "source": group.source.iloc[0]})
    return pd.DataFrame(rows)


def region_audit(predictions, threshold):
    """Descriptive good-customer flag rate, not a declaration of fairness or causality."""
    y, p = _prediction_arrays(predictions)
    if "region" not in predictions or not set(predictions.region) <= set(REGIONS):
        raise ValueError("Use validated synthetic region groups.")
    flagged = flag_at(p, threshold)
    rows = []
    for region in REGIONS:
        mask = predictions.region.eq(region).to_numpy()
        yy, ff = y[mask], flagged[mask]
        negatives, positives = int((yy == 0).sum()), int(yy.sum())
        fp, tp = int(((yy == 0) & ff).sum()), int(((yy == 1) & ff).sum())
        rows.append({"region": region, "rows": int(mask.sum()), "negatives": negatives,
            "positives": positives, "flagged": int(ff.sum()), "false_positives": fp, "true_positives": tp,
            "false_positive_rate": fp/negatives if negatives else np.nan,
            "recall": tp/positives if positives else np.nan,
            "low_negative_support": negatives < 30, "source": predictions.source.iloc[0]})
    table = pd.DataFrame(rows)
    supported = table.loc[table.negatives > 0, "false_positive_rate"]
    gap = float((supported.max()-supported.min())*100) if len(supported) >= 2 else None
    return table, {"fpr_gap_percentage_points": gap, "groups_with_negative_support": len(supported),
                   "interpretation": "Descriptive; no confidence interval or fairness certification."}


def reflection_check(responses, source="LIVE"):
    required = ("threshold_reason", "cost_capacity_tradeoff", "regional_gap", "limitations",
                "why_accuracy_misleads", "why_oof")
    missing = [key for key in required if not isinstance(responses.get(key), str) or not responses[key].strip()]
    return {"status": "EXAMPLE_ONLY_NOT_SUBMITTABLE" if source != "LIVE" else
            "LEARNER_WORK_REQUIRED" if missing else "READY_FOR_REVIEW", "missing": missing, "automatic_grade": None}


def load_example(csv_path, metadata_path, frame, folds, assessment_mode=False):
    import hashlib, json
    if assessment_mode:
        raise ValueError("Educational examples are disabled in assessment mode; run live CPU training.")
    info = json.loads(metadata_path.read_text(encoding="utf-8"))
    if info.get("source") != "EDUCATIONAL_EXAMPLE" or hashlib.sha256(csv_path.read_bytes()).hexdigest() != info["csv_sha256"]:
        raise ValueError("The educational example label or checksum does not match.")
    oof = pd.read_csv(csv_path, float_precision="round_trip")
    if set(oof.source) != {"EDUCATIONAL_EXAMPLE"}:
        raise ValueError("Example rows must all carry the educational source label.")
    validate_oof(oof, frame, folds)
    return oof, info
