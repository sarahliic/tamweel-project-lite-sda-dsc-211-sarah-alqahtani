"""Separated calibration, auditable tree explanations and conditional stability checks."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from time import perf_counter
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

from day2_validation import feature_frame, safe_predict
from day3_decision import DecisionConfig, CostPolicy, fit_strategy, capacity_limit

TARGET = "default_within_90d"
ROLES = ("fit", "calibration", "policy", "evaluation")
BOUNDARIES = {"fit": (None, "2023-07-01"), "calibration": ("2023-07-01", "2023-10-01"),
              "policy": ("2024-01-01", "2024-04-01"), "evaluation": ("2024-07-01", "2025-01-01")}


@dataclass(frozen=True)
class TrustConfig:
    seed: int = 211
    trees: int = 80
    shap_rows: int = 300
    shap_seconds: float = 60.0
    permutation_repeats: int = 3
    bootstrap_repeats: int = 200
    review_half_width: float = .02

    def __post_init__(self):
        if (self.trees not in (80, 160) or not 1 <= self.shap_rows <= 600
                or not np.isfinite(self.shap_seconds) or not 0 <= self.shap_seconds <= 120
                or not 2 <= self.permutation_repeats <= 5 or not 20 <= self.bootstrap_repeats <= 500
                or not np.isfinite(self.review_half_width) or not 0 <= self.review_half_width <= .1):
            raise ValueError("Use bounded CPU settings and a predeclared review-band width.")


def split_roles(frame):
    """Reserve later-role customers first; no label is used to allocate a role."""
    if not frame.application_id.is_unique or not frame.index.is_unique or frame.customer_id.isna().any():
        raise ValueError("Use unique application IDs/index and complete customer IDs.")
    dates = pd.to_datetime(frame.application_date, errors="raise")
    if dates.isna().any():
        raise ValueError("Application dates cannot be missing.")
    masks = {"fit": dates + pd.Timedelta(days=90) < pd.Timestamp(BOUNDARIES['fit'][1])}
    masks.update({r: (dates >= start) & (dates < end) for r, (start, end) in BOUNDARIES.items() if r != "fit"})
    parts, seen = {}, set()
    for role in reversed(ROLES):
        part = frame.loc[masks[role] & ~frame.customer_id.isin(seen)].copy()
        if part.empty or set(part[TARGET]) != {0, 1}:
            raise ValueError(f"{role} needs both classes. Do not alter the split after seeing evaluation results.")
        seen.update(part.customer_id)
        parts[role] = part
    parts = {r: parts[r] for r in ROLES}
    assert_roles(parts)
    assignment = frame[["application_id", "customer_id", "application_date"]].copy()
    assignment['role'] = 'excluded_gap_or_customer'
    for role, part in parts.items():
        assignment.loc[part.index, 'role'] = role
    return parts, assignment


def assert_roles(parts):
    if set(parts) != set(ROLES):
        raise ValueError("Four explicit roles are required.")
    for a, b in combinations(ROLES, 2):
        if set(parts[a].application_id) & set(parts[b].application_id) or set(parts[a].customer_id) & set(parts[b].customer_id):
            raise ValueError("Fit, calibration, policy and evaluation must not share applications or customers.")
    for a, b in zip(ROLES[:-1], ROLES[1:]):
        ready = pd.to_datetime(parts[a].application_date) + pd.Timedelta(days=90)
        if not (ready < pd.Timestamp(BOUNDARIES[b][0])).all():
            raise ValueError("Every earlier role's labels must mature before the next role starts.")


def fit_and_calibrate(parts, contract, config=TrustConfig()):
    assert_roles(parts)
    model, info = fit_strategy(parts['fit'], contract, 'weighted', DecisionConfig(trees=config.trees, seed=config.seed))
    features = [f['name'] for f in contract['features']]
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid', n_jobs=1)
    calibrated.fit(feature_frame(parts['calibration'][features], contract), parts['calibration'][TARGET])
    # Pinned sklearn 1.6.1: a single sigmoid over the frozen pipeline's positive probability.
    sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
    mapping = {'a': float(sigmoid.a_), 'b': float(sigmoid.b_),
               'input': 'uncalibrated positive probability', 'formula': 'expit(-(a * raw_probability + b))',
               'strictly_increasing': bool(sigmoid.a_ < 0)}
    if not mapping['strictly_increasing']:
        raise ValueError("Calibration is not increasing. Stop threshold transport and investigate the calibration sample.")
    tables = []
    for role in ('calibration', 'policy', 'evaluation'):
        part = parts[role]
        raw = safe_predict(model, part[features], contract)
        cal = calibrated.predict_proba(feature_frame(part[features], contract))[:, 1]
        if not np.allclose(cal, map_probability(raw, mapping), rtol=0, atol=1e-14):
            raise ValueError("Calibration mapping does not match the fitted estimator.")
        tables.append(part[['application_id', 'customer_id', 'application_date', TARGET]].assign(
            role=role, raw_probability=raw, calibrated_probability=cal, source='LIVE'))
    info.update({'model_sha256': model_fingerprint(model), 'calibration_mapping': mapping,
                 'application_ids': {r: parts[r].application_id.tolist() for r in ROLES},
                 'customer_counts': {r: int(parts[r].customer_id.nunique()) for r in ROLES},
                 'feature_names': features, 'source': 'LIVE'})
    return model, calibrated, pd.concat(tables, ignore_index=True), info


def map_probability(p, mapping):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Raw probabilities must be finite in [0, 1].")
    return expit(-(mapping['a'] * p + mapping['b']))


def model_fingerprint(model):
    payload = model.named_steps['model'].booster_.model_to_string()
    payload += json.dumps(model.named_steps['imputer'].statistics_.tolist())
    return sha256(payload.encode()).hexdigest()


def checked_arrays(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or len(y) == 0 or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError("Use aligned nonempty binary labels and finite probabilities in [0, 1].")
    return y.astype(int), p


def reliability_table(y, p, bins=10):
    y, p = checked_arrays(y, p)
    if not isinstance(bins, int) or not 2 <= bins <= 20:
        raise ValueError("Choose 2–20 fixed-width bins before inspecting outcomes.")
    index = np.minimum((p * bins).astype(int), bins - 1)
    records = []
    for b in range(bins):
        keep = index == b
        records.append({'bin': b, 'lower': b/bins, 'upper': (b+1)/bins, 'count': int(keep.sum()),
            'mean_probability': float(p[keep].mean()) if keep.any() else np.nan,
            'observed_rate': float(y[keep].mean()) if keep.any() else np.nan,
            'low_support': bool(0 < keep.sum() < 30)})
    return pd.DataFrame(records)


def probability_metrics(y, p, bins=10):
    y, p = checked_arrays(y, p)
    table = reliability_table(y, p, bins)
    ece = float(((table['mean_probability'] - table.observed_rate).abs().fillna(0) * table['count']).sum()/len(y))
    both = len(np.unique(y)) == 2
    return {'rows': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, p)) if both else None,
            'average_precision': float(average_precision_score(y, p)) if both else None,
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'ece': ece, 'bins': bins, 'binning': 'fixed width, [lower, upper); final bin includes 1'}


def choose_policy_threshold(policy_rows, policy=CostPolicy()):
    if set(policy_rows.role) != {'policy'} or not policy_rows.application_id.is_unique:
        raise ValueError("Choose the threshold only on unique policy-development applications.")
    y, p = checked_arrays(policy_rows[TARGET], policy_rows.raw_probability)
    thresholds = np.unique(np.r_[p, 0., .5, np.nextafter(1., 2.)])
    rows = []
    limit = capacity_limit(len(y), policy.max_flag_fraction)
    for t in thresholds:
        flags = p >= t
        fn, fp = int(((y == 1) & ~flags).sum()), int(((y == 0) & flags).sum())
        rows.append({'threshold': float(t), 'flagged': int(flags.sum()), 'fn': fn, 'fp': fp,
                     'loss_units': fn*policy.false_negative_cost + fp*policy.false_positive_cost,
                     'capacity': limit, 'feasible': bool(flags.sum() <= limit)})
    sweep = pd.DataFrame(rows)
    chosen = sweep[sweep.feasible].sort_values(['loss_units', 'flagged', 'threshold'], ascending=[True, True, False]).iloc[0].to_dict()
    return chosen, sweep


def transport_threshold(raw_threshold, mapping):
    if not np.isfinite(raw_threshold) or raw_threshold < 0 or not mapping['strictly_increasing']:
        raise ValueError("Threshold transport requires an increasing mapping.")
    if raw_threshold > 1:
        return float(np.nextafter(1., 2.))
    return float(map_probability(raw_threshold, mapping))


def review_audit(evaluation, raw_threshold, calibrated_threshold, half_width=.02, policy=CostPolicy()):
    if set(evaluation.role) != {'evaluation'} or not 0 <= half_width <= .1:
        raise ValueError("Review audit requires evaluation rows and a predeclared band width.")
    raw_flags = evaluation.raw_probability.to_numpy() >= raw_threshold
    cal_flags = evaluation.calibrated_probability.to_numpy() >= calibrated_threshold
    if not np.array_equal(raw_flags, cal_flags):
        raise ValueError("Threshold mapping changed the flagged set. Do not silently round or retune.")
    lower, upper = max(0., calibrated_threshold-half_width), min(1., calibrated_threshold+half_width)
    near = evaluation.calibrated_probability.between(lower, upper).to_numpy()
    records = evaluation.copy()
    records['risk_flag'], records['near_threshold'] = cal_flags, near
    records['candidate_review'] = cal_flags | near
    records['period'] = pd.to_datetime(records.application_date).dt.to_period('Q').astype(str)
    audit = []
    for period, part in records.groupby('period'):
        flag = part.risk_flag.to_numpy();y = part[TARGET].to_numpy()
        capacity = capacity_limit(len(part), policy.max_flag_fraction)
        audit.append({'period': period, 'rows': len(part), 'capacity': capacity,
            'risk_flags': int(flag.sum()), 'near_threshold': int(part.near_threshold.sum()),
            'candidate_review': int(part.candidate_review.sum()), 'risk_within_capacity': bool(flag.sum() <= capacity),
            'candidate_within_capacity': bool(part.candidate_review.sum() <= capacity),
            'loss_units': int(((y == 1) & ~flag).sum())*policy.false_negative_cost + int(((y == 0) & flag).sum())*policy.false_positive_cost})
    return records, pd.DataFrame(audit), {'lower': lower, 'upper': upper, 'half_width': half_width,
        'interpretation': 'Predeclared diagnostic band, not a confidence interval or approved queue.'}


def permutation_report(model, evaluation, contract, config=TrustConfig()):
    features = [f['name'] for f in contract['features']]
    X = evaluation[features].copy(); y = evaluation[TARGET]
    baseline = average_precision_score(y, safe_predict(model, X, contract))
    regions = [f for f in features if f.startswith('region_')]
    groups = [(f, [f]) for f in features if f not in regions] + [('region_indicators_joint', regions)]
    gain = dict(zip(features, model.named_steps['model'].booster_.feature_importance(importance_type='gain')))
    rng = np.random.default_rng(config.seed);rows = []
    for name, columns in groups:
        drops = []
        for _ in range(config.permutation_repeats):
            shuffled = X.copy();order = rng.permutation(len(X))
            shuffled.loc[:, columns] = X[columns].to_numpy()[order]
            drops.append(float(baseline-average_precision_score(y, safe_predict(model, shuffled, contract))))
        rows.append({'feature_group': name, 'columns': '|'.join(columns), 'mean_ap_drop': float(np.mean(drops)),
            'repeat_sd': float(np.std(drops, ddof=1)), 'training_gain': float(sum(gain[c] for c in columns)),
            'baseline_ap': float(baseline), 'repeats': len(drops), 'rows': len(X), 'source': 'LIVE',
            **{f'drop_{i+1}': d for i, d in enumerate(drops)}})
    return pd.DataFrame(rows).sort_values('mean_ap_drop', ascending=False).reset_index(drop=True)


class ShapBudgetExpired(RuntimeError):
    pass


def explain_sample(model, evaluation, contract, config=TrustConfig()):
    started = perf_counter()
    if config.shap_seconds <= 0:
        raise ShapBudgetExpired('SHAP time budget reached; view a labeled example or rerun on CPU.')
    import shap
    features = [f['name'] for f in contract['features']]
    sample = evaluation.sample(n=min(config.shap_rows, len(evaluation)), random_state=config.seed).copy()
    X = model.named_steps['imputer'].transform(feature_frame(sample[features], contract))
    explainer = shap.TreeExplainer(model.named_steps['model'], model_output='raw', feature_perturbation='tree_path_dependent')
    batches = []
    for begin in range(0, len(X), 50):
        if perf_counter()-started >= config.shap_seconds:
            raise ShapBudgetExpired('SHAP time budget reached; no partial explanation is accepted.')
        batches.append(explainer(X.iloc[begin:begin+50], check_additivity=True).values)
    values = np.concatenate(batches)
    raw = model.named_steps['model'].predict(X, raw_score=True)
    base = np.full(len(X), float(np.asarray(explainer.expected_value).reshape(-1)[0]))
    if values.shape != X.shape or not np.allclose(base+values.sum(axis=1), raw, rtol=0, atol=1e-8):
        raise ValueError('SHAP must sum to the raw log-odds, not probability points.')
    if not np.allclose(expit(raw), safe_predict(model, sample[features], contract), atol=1e-12, rtol=0):
        raise ValueError('Raw margin and original probability do not reconcile.')
    if perf_counter()-started >= config.shap_seconds:
        raise ShapBudgetExpired('SHAP time budget reached after the final batch.')
    return {'values': values, 'base_values': base, 'data': X.to_numpy(), 'raw_margin': raw,
        'application_ids': sample.application_id.to_numpy(dtype=str), 'feature_names': np.array(features, dtype=str)}, {
        'source': 'LIVE', 'model_sha256': model_fingerprint(model), 'rows': len(sample),
        'units': 'raw log-odds of uncalibrated weighted model', 'background': 'tree training path counts',
        'elapsed_seconds': perf_counter()-started, 'max_additivity_error': float(np.abs(base+values.sum(axis=1)-raw).max())}


def load_shap_example(npz_path, metadata_path, model, evaluation, contract, assessment_mode=False):
    if assessment_mode:
        raise ValueError('ASSESSMENT_MODE forbids educational SHAP examples.')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['source'] != 'EDUCATIONAL_EXAMPLE' or sha256(npz_path.read_bytes()).hexdigest() != metadata['npz_sha256']:
        raise ValueError('Use an intact, labeled SHAP example.')
    if metadata['model_sha256'] != model_fingerprint(model):
        raise ValueError('Example model differs. Restart with FAST settings or complete live SHAP; never mix explanations.')
    with np.load(npz_path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    features = [f['name'] for f in contract['features']]
    if list(result['feature_names']) != features or len(set(result['application_ids'])) != len(result['application_ids']):
        raise ValueError('Example feature order and unique IDs must match.')
    sample = evaluation.set_index('application_id').loc[result['application_ids']]
    transformed = model.named_steps['imputer'].transform(sample[features])
    raw = model.named_steps['model'].predict(transformed, raw_score=True)
    if not (np.array_equal(transformed.to_numpy(), result['data']) and np.allclose(raw, result['raw_margin'], atol=1e-10, rtol=0)
            and np.allclose(result['base_values']+result['values'].sum(axis=1), raw, atol=1e-8, rtol=0)):
        raise ValueError('Example does not reconcile with this model and these held-out applications.')
    return result, metadata


def reason_codes(values, feature_names, feature_values, descriptions, limit=3):
    values = np.asarray(values)
    positive = [i for i in np.argsort(-values, kind='stable') if values[i] > 0][:limit]
    return pd.DataFrame([{'rank': rank+1, 'feature': feature_names[i], 'value_used': float(feature_values[i]),
        'contribution_log_odds': float(values[i]),
        'reason_ar': f"استخدم النموذج {descriptions[feature_names[i]]} بالقيمة {feature_values[i]:g} لرفع درجته الخام؛ لا يثبت ذلك السببية."}
        for rank, i in enumerate(positive)], columns=['rank','feature','value_used','contribution_log_odds','reason_ar'])


def local_stability(model, case, contract, mapping):
    import shap
    features = [f['name'] for f in contract['features']]
    variants = [];labels = []
    bounds = next(f['range'] for f in contract['features'] if f['name'] == 'bureau_score')
    for delta in (0, -1, 1):
        row = case[features].copy()
        row.loc[:, 'bureau_score'] = np.clip(row.bureau_score+delta, *bounds)
        variants.append(row);labels.append(f'bureau_score {delta:+d}')
    X = pd.concat(variants, ignore_index=True)
    transformed = model.named_steps['imputer'].transform(X)
    explainer = shap.TreeExplainer(model.named_steps['model'], model_output='raw', feature_perturbation='tree_path_dependent')
    values = explainer(transformed).values
    top = lambda row: [features[i] for i in np.argsort(-row, kind='stable') if row[i] > 0][:3]
    reference = set(top(values[0]));raw = model.predict_proba(X)[:, 1]
    return pd.DataFrame([{'perturbation': labels[i], 'bureau_score_used': float(X.iloc[i].bureau_score),
        'raw_probability': float(raw[i]), 'calibrated_probability': float(map_probability(raw[i], mapping)),
        'top_positive_features': '|'.join(top(values[i])), 'shared_top_reasons': len(reference & set(top(values[i]))),
        'reference_reasons': len(reference)} for i in range(len(X))])


def cluster_bootstrap(evaluation, config=TrustConfig()):
    if set(evaluation.role) != {'evaluation'} or evaluation.customer_id.isna().any():
        raise ValueError('Bootstrap requires evaluation rows with customer clusters.')
    customers = evaluation.customer_id.unique()
    groups = [np.flatnonzero(evaluation.customer_id.to_numpy() == c) for c in customers]
    rng = np.random.default_rng(config.seed);rows = [];skipped = 0
    for repeat in range(config.bootstrap_repeats):
        positions = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        part = evaluation.iloc[positions];y = part[TARGET]
        if y.nunique() != 2:
            skipped += 1
            continue
        raw, cal = part.raw_probability.to_numpy(), part.calibrated_probability.to_numpy()
        rows.append({'repeat': repeat+1, 'rows': len(part), 'raw_ap': float(average_precision_score(y, raw)),
            'calibrated_ap': float(average_precision_score(y, cal)), 'raw_brier': float(np.mean((raw-y)**2)),
            'calibrated_brier': float(np.mean((cal-y)**2)), 'brier_change': float(np.mean((cal-y)**2-(raw-y)**2))})
    table = pd.DataFrame(rows)
    summary = {'method': 'paired customer-cluster percentile bootstrap', 'level': .95,
        'requested': config.bootstrap_repeats, 'valid': len(rows), 'skipped_one_class': skipped,
        'customers': len(customers), 'conditional_on': 'fixed model, calibrator and observed evaluation period; no refitting',
        'limitation': 'Does not include model fitting, calibration fitting, future drift or arbitrary time dependence.'}
    for metric in ('raw_ap','calibrated_ap','brier_change'):
        summary[metric] = {'lower': float(table[metric].quantile(.025)), 'upper': float(table[metric].quantile(.975))} if len(rows) >= 20 else None
    return table, summary


def reflection_check(responses, explanation_source):
    required = ('global_local', 'output_units', 'reason_limits', 'calibration_evidence', 'stability', 'review_capacity')
    missing = [key for key in required if not isinstance(responses.get(key), str) or not responses[key].strip()]
    status = 'EXAMPLE_ONLY_NOT_SUBMITTABLE' if explanation_source != 'LIVE' else ('LEARNER_WORK_REQUIRED' if missing else 'READY_FOR_REVIEW')
    return {'status': status, 'missing': missing, 'no_automatic_grade': True}
