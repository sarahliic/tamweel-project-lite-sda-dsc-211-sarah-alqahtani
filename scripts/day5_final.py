"""Nested forward stacking, bounded decisions and portable course model artifacts."""
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, Booster
from xgboost import XGBClassifier

from day2_validation import feature_frame, make_time_group_split, BASE_PARAMS
from day3_decision import TARGET, CostPolicy, capacity_limit, flag_at, region_labels, threshold_sweep, select_threshold
from day4_trust import probability_metrics, map_probability, transport_threshold

BASE = ('LightGBM', 'XGBoost', 'Logistic')
ENSEMBLES = ('Equal', 'Weighted', 'Stack')
WINDOWS = (('2023-01-01', '2023-04-01'), ('2023-07-01', '2023-10-01'), ('2024-01-01', '2024-04-01'))
CAL_START, CAL_END, DEPLOY_DATE = '2024-07-01', '2024-10-01', '2025-01-01'


@dataclass(frozen=True)
class FinalConfig:
    seed: int = 211
    trees: int = 80
    seconds: float = 120.0
    max_brier_increase: float = .005
    max_ece_increase: float = .01

    def __post_init__(self):
        if (self.trees not in (80, 160) or not np.isfinite(self.seconds) or not 0 <= self.seconds <= 240
                or not 0 <= self.max_brier_increase <= .02 or not 0 <= self.max_ece_increase <= .05):
            raise ValueError('Use bounded CPU settings and predeclared comparison tolerances.')


class EnsembleBudgetExpired(RuntimeError):
    """A partial nested OOF run is never a complete comparison."""


def deadline_check(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise EnsembleBudgetExpired('CPU budget reached. Retry later or inspect the labeled example.')


def split_final_roles(frame):
    if (not frame.index.is_unique or not frame.application_id.is_unique
            or frame[['application_id', 'customer_id', 'application_date']].isna().any().any()):
        raise ValueError('Unique complete IDs, customers and dates are required.')
    dates = pd.to_datetime(frame.application_date, errors='raise')
    cal = frame.loc[(dates >= CAL_START) & (dates < CAL_END)].copy()
    fit = frame.loc[(dates + pd.Timedelta(days=90) < CAL_START) & ~frame.customer_id.isin(cal.customer_id)].copy()
    for part in (fit, cal):
        if len(part) < 100 or set(part[TARGET]) != {0, 1}:
            raise ValueError('Both classes and at least 100 rows are required in each role.')
    if not (pd.to_datetime(cal.application_date) + pd.Timedelta(days=90) < DEPLOY_DATE).all():
        raise ValueError('Calibration outcomes must mature before the challenge starts.')
    assignments = frame[['application_id', 'customer_id', 'application_date']].copy()
    assignments['role'] = 'excluded_gap_customer_or_unavailable_label'
    assignments.loc[fit.index, 'role'] = 'historical_fit_and_selection'
    assignments.loc[cal.index, 'role'] = 'calibration_only'
    return fit, cal, assignments


def fit_bases(train, contract, config, deadline=None):
    deadline_check(deadline)
    features = [f['name'] for f in contract['features']]
    if set(train) != set(features) | set(contract['metadata']) | {TARGET}:
        raise ValueError('Use only the approved training schema.')
    if set(train[TARGET]) != {0, 1}:
        raise ValueError('Training needs both classes.')
    common = dict(n_estimators=config.trees, learning_rate=.05, random_state=config.seed,
                  n_jobs=2, subsample=.8, colsample_bytree=.8)
    estimators = {
        'LightGBM': LGBMClassifier(**{k:v for k,v in BASE_PARAMS.items() if k != 'learning_rate'},
            **common, subsample_freq=1, deterministic=True, force_col_wise=True, verbosity=-1),
        'XGBoost': XGBClassifier(**common, tree_method='hist', max_depth=3, objective='binary:logistic', eval_metric='logloss'),
        'Logistic': LogisticRegression(C=1., max_iter=1000, solver='lbfgs', random_state=config.seed)}
    models = {}
    for name, estimator in estimators.items():
        deadline_check(deadline)
        steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True).set_output(transform='pandas'))]
        if name == 'Logistic':
            steps.append(('scale', StandardScaler().set_output(transform='pandas')))
        steps.append(('model', estimator))
        model = Pipeline(steps).fit(feature_frame(train[features], contract), train[TARGET])
        models[name] = model
    deadline_check(deadline)
    return models


def base_predictions(models, X):
    return np.column_stack([models[name].predict_proba(X)[:, 1] for name in BASE])


def inner_windows(train):
    last = pd.to_datetime(train.application_date).max().normalize() + pd.Timedelta(days=1)
    start, windows = pd.Timestamp('2022-07-01'), []
    while start < last:
        end = min(start + pd.DateOffset(months=3), last)
        # A sliver at a maturity boundary is not an additional validation period.
        if (end - start).days >= 30:
            windows.append((str(start.date()), str(end.date())))
        start += pd.DateOffset(months=3)
    if not windows:
        raise ValueError('Not enough earlier history for forward OOF meta-training.')
    return windows


def inner_oof(train, contract, config, deadline=None):
    folds = make_time_group_split(train, windows=inner_windows(train))
    names = [f['name'] for f in contract['features']]
    rows, audit = [], []
    for fold in folds:
        fit, valid = train.loc[fold['train']], train.loc[fold['valid']]
        models = fit_bases(fit, contract, config, deadline)
        rows.append(valid[['application_id', 'customer_id', 'application_date', TARGET]].assign(
            **dict(zip(BASE, base_predictions(models, valid[names]).T))))
        audit.append({**fold['audit'], 'training_ids': fit.application_id.tolist(), 'validation_ids': valid.application_id.tolist()})
    return pd.concat(rows, ignore_index=True), audit


def train_combiner(oof, seed=211):
    X, y = oof[list(BASE)].to_numpy(), oof[TARGET].to_numpy()
    if (oof.application_id.duplicated().any() or not np.isfinite(X).all()
            or (X < 0).any() or (X > 1).any() or set(y) != {0, 1}):
        raise ValueError('Meta-training requires complete, unique OOF rows and valid probabilities.')
    # Small predeclared simplex grid; pure singles are already compared separately.
    weights = [np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a) if max(a,b,4-a-b) < 4]
    scores = [probability_metrics(y, X @ w)['average_precision'] for w in weights]
    best = max(range(len(weights)), key=lambda i: (scores[i], -float(np.sum((weights[i]-1/3)**2)), -i))
    meta = LogisticRegression(C=1., max_iter=1000, random_state=seed).fit(X, y)
    return weights[best], meta, {'weight_grid_size':len(weights), 'weights':weights[best].tolist(),
        'weight_selection':'inner OOF AP; tie: nearest equal weights, then fixed grid order',
        'meta_rows':len(oof), 'meta_coefficients':meta.coef_[0].tolist(), 'meta_intercept':float(meta.intercept_[0])}


def candidate_predictions(matrix, weights, meta):
    p = dict(zip(BASE, np.asarray(matrix).T))
    p.update(Equal=matrix.mean(axis=1), Weighted=matrix @ weights, Stack=meta.predict_proba(matrix)[:,1])
    return p


def nested_comparison(pool, contract, config=FinalConfig()):
    started = perf_counter(); deadline = started + config.seconds
    deadline_check(deadline)
    folds = make_time_group_split(pool, windows=WINDOWS)
    features = [f['name'] for f in contract['features']]
    records, scores, provenance = [], [], []
    for fold in folds:
        train, valid = pool.loc[fold['train']], pool.loc[fold['valid']]
        inner, inner_audit = inner_oof(train, contract, config, deadline)
        weights, meta, combination = train_combiner(inner, config.seed)
        models = fit_bases(train, contract, config, deadline)
        candidates = candidate_predictions(base_predictions(models, valid[features]), weights, meta)
        records.append(valid[['application_id','customer_id','application_date',TARGET]].assign(
            region=region_labels(valid), fold=fold['audit']['fold'], source='LIVE', split_role='OOF', **candidates))
        for name, p in candidates.items():
            scores.append({'fold':fold['audit']['fold'], 'candidate':name, **probability_metrics(valid[TARGET],p)})
        provenance.append({**fold['audit'], 'training_ids':train.application_id.tolist(),
            'validation_ids':valid.application_id.tolist(), 'inner_folds':inner_audit,
            'meta_training_ids':inner.application_id.tolist(), **combination})
    result = pd.concat(records, ignore_index=True)
    validate_comparison(result, pool)
    return result, pd.DataFrame(scores), {'source':'LIVE','config':asdict(config),'folds':provenance,
        'elapsed_seconds':perf_counter()-started, 'protocol':'nested forward time, mature labels, customer purge at both levels'}


def validate_comparison(oof, pool):
    folds = make_time_group_split(pool, windows=WINDOWS)
    required = ['application_id','customer_id','application_date',TARGET,'region','fold','source','split_role',*BASE,*ENSEMBLES]
    if (not set(required) <= set(oof) or oof[required].isna().any().any() or oof.application_id.duplicated().any()
            or set(oof.split_role) != {'OOF'} or len(set(oof.source)) != 1
            or not set(oof.source) <= {'LIVE','EDUCATIONAL_EXAMPLE'}):
        raise ValueError('A complete, uniquely identified and labeled comparison is required.')
    p=oof[[*BASE,*ENSEMBLES]].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('OOF probabilities must be finite in [0,1].')
    lookup=pool.set_index('application_id')
    expected=set(pool.loc[np.concatenate([f['valid'] for f in folds]),'application_id'])
    if set(oof.application_id)!=expected:
        raise ValueError('OOF coverage does not match the forward folds.')
    actual=lookup.loc[oof.application_id]
    for column in ['customer_id','application_date',TARGET]:
        if not np.array_equal(oof[column].to_numpy(),actual[column].to_numpy()):
            raise ValueError('OOF metadata/labels do not match source data.')
    if not np.array_equal(oof.region.to_numpy(),region_labels(actual).to_numpy()):
        raise ValueError('OOF regions do not match source data.')
    for fold in folds:
        if set(oof.loc[oof.fold==fold['audit']['fold'],'application_id']) != set(pool.loc[fold['valid'],'application_id']):
            raise ValueError('OOF row assigned to wrong period.')


def comparison_scores(oof):
    return pd.DataFrame([{'fold':fold,'candidate':name,**probability_metrics(g[TARGET],g[name])}
        for fold,g in oof.groupby('fold') for name in (*BASE,*ENSEMBLES)])


def worth_it(scores, config=FinalConfig()):
    if set(scores.candidate) != set(BASE+ENSEMBLES) or scores.duplicated(['fold','candidate']).any():
        raise ValueError('Compare all candidates on the same complete three folds.')
    if any(set(g.fold) != {1,2,3} for _,g in scores.groupby('candidate')):
        raise ValueError('Three common folds are required.')
    if not np.isfinite(scores[['average_precision','brier','ece']].to_numpy()).all():
        raise ValueError('Comparison scores must be finite.')
    summary=scores.groupby('candidate').agg(mean_ap=('average_precision','mean'),fold_sd=('average_precision','std'),
        mean_brier=('brier','mean'),mean_ece=('ece','mean')).reindex(BASE+ENSEMBLES)
    best=summary.loc[list(BASE)].sort_values('mean_ap',ascending=False,kind='stable').index[0]
    baseline=summary.loc[best]
    summary['lift_vs_single']=summary.mean_ap-baseline.mean_ap
    summary['passes_gate']=(summary.lift_vs_single>baseline.fold_sd) & (summary.mean_brier<=baseline.mean_brier+config.max_brier_increase) & (summary.mean_ece<=baseline.mean_ece+config.max_ece_increase)
    passing=[name for name in ENSEMBLES if summary.loc[name,'passes_gate']]
    chosen=passing[0] if passing else best
    return summary.reset_index(), {'decision':'SHIP ENSEMBLE' if passing else 'KEEP SINGLE', 'chosen':chosen,
        'best_single':best,'fold_sd_reference':float(baseline.fold_sd),'selection_rule':'simplest passing ensemble: Equal, Weighted, Stack; otherwise best single',
        'gate':f'AP lift > single fold SD; Brier increase <={config.max_brier_increase}; ECE increase <={config.max_ece_increase}',
        'brier_tolerance':config.max_brier_increase,'ece_tolerance':config.max_ece_increase,
        'scope':'selection evidence; three dependent forward folds, descriptive SD, not a significance test or untouched holdout'}


def selected_oof(oof, chosen):
    if chosen not in BASE+ENSEMBLES:
        raise ValueError('Unknown candidate.')
    return oof[['application_id','customer_id','application_date',TARGET,'region','fold','source','split_role']].assign(probability=oof[chosen])


class FrozenCandidate(ClassifierMixin, BaseEstimator):
    def __init__(self, models, weights, meta, chosen):
        self.models=models; self.weights=weights; self.meta=meta; self.chosen=chosen
        self.classes_=np.array([0,1]); self.fitted_=True

    def fit(self, X, y=None):
        raise RuntimeError('Already fitted; use FrozenEstimator for calibration.')

    def predict_proba(self, X):
        p=candidate_predictions(base_predictions(self.models,X),self.weights,self.meta)[self.chosen]
        return np.column_stack([1-p,p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1]>=.5).astype(int)


def fit_final(pool, calibration, contract, chosen, config=FinalConfig()):
    cal_dates=pd.to_datetime(calibration.application_date,errors='raise')
    if (chosen not in BASE+ENSEMBLES or not calibration.application_id.is_unique
            or calibration.customer_id.isna().any() or cal_dates.isna().any()
            or not ((cal_dates>=CAL_START)&(cal_dates<CAL_END)).all()
            or not (cal_dates+pd.Timedelta(days=90)<DEPLOY_DATE).all()
            or set(calibration[TARGET])!={0,1}):
        raise ValueError('Use the reserved calibration period with unique IDs, both classes and mature labels.')
    if (set(pool.customer_id)&set(calibration.customer_id) or
        not (pd.to_datetime(pool.application_date)+pd.Timedelta(days=90)<CAL_START).all()):
        raise ValueError('Final fitting must precede calibration with no shared customers.')
    names=[f['name'] for f in contract['features']]
    started=perf_counter(); deadline=started+config.seconds
    inner,audit=inner_oof(pool,contract,config,deadline)
    weights,meta,details=train_combiner(inner,config.seed)
    models=fit_bases(pool,contract,config,deadline)
    candidate=FrozenCandidate(models,weights,meta,chosen)
    calibrated=CalibratedClassifierCV(FrozenEstimator(candidate),method='sigmoid',n_jobs=1)
    calibrated.fit(calibration[names],calibration[TARGET])
    sigmoid=calibrated.calibrated_classifiers_[0].calibrators[0]
    mapping={'a':float(sigmoid.a_),'b':float(sigmoid.b_),'input':'uncalibrated positive probability',
        'formula':'expit(-(a * raw_probability + b))','strictly_increasing':bool(sigmoid.a_<0)}
    if not mapping['strictly_increasing']:
        raise ValueError('Non-increasing calibration cannot transport the frozen decision rule.')
    raw=candidate.predict_proba(calibration[names])[:,1]
    cal=calibrated.predict_proba(calibration[names])[:,1]
    if not np.allclose(cal,map_probability(raw,mapping),atol=1e-14,rtol=0):
        raise ValueError('Sigmoid export must match the frozen estimator.')
    details.update(training_ids=pool.application_id.tolist(),calibration_ids=calibration.application_id.tolist(),
        inner_folds=audit,meta_training_ids=inner.application_id.tolist(),mapping=mapping,
        elapsed_seconds=perf_counter()-started,chosen=chosen)
    predictions=calibration[['application_id','customer_id','application_date',TARGET]].assign(raw_probability=raw,probability=cal)
    return candidate,mapping,predictions,details


def apply_batch_policy(predictions, threshold, policy=CostPolicy()):
    if (list(predictions)!=['application_id','probability'] or predictions.empty
            or predictions.application_id.isna().any() or not predictions.application_id.is_unique):
        raise ValueError('Provide one complete batch with unique IDs and probabilities only.')
    p=predictions.probability.to_numpy(float); eligible=flag_at(p,threshold)
    cap=capacity_limit(len(p),policy.max_flag_fraction); decision=eligible.copy()
    boundary=None
    if eligible.sum()>cap:
        if cap==0:
            decision[:]=False
        else:
            boundary=float(np.sort(p[eligible])[::-1][cap-1])
            decision=eligible & (p>=boundary)
            if decision.sum()>cap:
                decision=eligible & (p>boundary)
    result=predictions.assign(decision=decision.astype(int))
    return result, {'rows':len(p),'capacity':cap,'threshold_eligible':int(eligible.sum()),'flagged':int(decision.sum()),
        'removed_by_cap':int(eligible.sum()-decision.sum()),'boundary_score':boundary,
        'within_capacity':bool(decision.sum()<=cap),'tie_policy':'retain entire equal-score blocks; drop boundary block if it cannot fit',
        'batch_policy':'threshold first, then probability-descending full-batch cap; no labels or ID tie-breaking'}


def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def export_model(candidate,mapping,contract,destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    used=list(BASE) if candidate.chosen in ENSEMBLES else [candidate.chosen]
    specs={}
    for name in used:
        pipe=candidate.models[name]; model=pipe.named_steps['model']
        spec={'medians':pipe.named_steps['imputer'].statistics_.tolist()}
        if name=='LightGBM':
            filename='lightgbm.txt'; model.booster_.save_model(destination/filename); spec['file']=filename
        elif name=='XGBoost':
            filename='xgboost.ubj'; model.save_model(destination/filename); spec['file']=filename
        else:
            scale=pipe.named_steps['scale']
            spec.update(mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))
        specs[name]=spec
    metadata={'format_version':1,'chosen':candidate.chosen,'features':[f['name'] for f in contract['features']],
        'contract':contract,'models':specs,'weights':candidate.weights.tolist(),
        'meta_coef':candidate.meta.coef_[0].tolist(),'meta_intercept':float(candidate.meta.intercept_[0]),'calibration':mapping}
    write_json(destination/'model.json',metadata)
    filenames=['model.json']+[s['file'] for s in specs.values() if 'file' in s]
    manifest={f:sha256((destination/f).read_bytes()).hexdigest() for f in filenames}
    write_json(destination/'model_manifest.json',manifest)
    return manifest


def load_model(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'model_manifest.json').read_text(encoding='utf-8'))
    allowed={'model.json','lightgbm.txt','xgboost.ubj'}
    if 'model.json' not in manifest or not set(manifest)<=allowed:
        raise ValueError('Unexpected model manifest paths.')
    for filename,digest in manifest.items():
        if sha256((directory/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Model checksum mismatch.')
    metadata=json.loads((directory/'model.json').read_text(encoding='utf-8'))
    chosen=metadata['chosen']; expected=set(BASE) if chosen in ENSEMBLES else {chosen}
    if chosen not in BASE+ENSEMBLES or set(metadata['models'])!=expected:
        raise ValueError('Unsupported candidate/model set.')
    loaded={}
    for name,spec in metadata['models'].items():
        if 'file' in spec and (spec['file'] not in manifest or spec['file']=='model.json'):
            raise ValueError('Model file is not in the verified manifest.')
        if name=='LightGBM': loaded[name]=Booster(model_file=str(directory/spec['file']))
        elif name=='XGBoost':
            loaded[name]=XGBClassifier(n_jobs=2); loaded[name].load_model(directory/spec['file'])
    return metadata,loaded


def predict_final(frame, artifact_dir):
    metadata,loaded=load_model(artifact_dir)
    names=metadata['features']; contract=metadata['contract']
    allowed=set(names)|set(contract['metadata'])
    if set(frame)!=allowed or frame[contract['metadata']].isna().any().any() or not frame.application_id.is_unique:
        raise ValueError('Provide unique IDs, customer/date metadata and exactly the approved predictors; no target.')
    if pd.to_datetime(frame.application_date,errors='raise',format='%Y-%m-%d').isna().any():
        raise ValueError('Application dates must be complete and valid.')
    for spec in contract['features']:
        values=pd.to_numeric(frame[spec['name']],errors='raise'); present=values.dropna()
        if (not np.isfinite(present).all() or not present.between(*spec['range']).all()
                or (not spec['nullable'] and values.isna().any()) or
                (spec['dtype']=='integer' and (present%1!=0).any())):
            raise ValueError(f"Invalid feature: {spec['name']}")
    region_labels(frame)
    if not frame[['employment_government','employment_private','employment_self_employed']].sum(axis=1).eq(1).all():
        raise ValueError('Exactly one employment category is required.')
    if frame.empty:
        return pd.DataFrame({'application_id':frame.application_id,'probability':pd.Series(dtype=float)})
    matrix=frame[names].to_numpy(float); predictions={}
    for name,spec in metadata['models'].items():
        X=np.where(np.isnan(matrix),np.asarray(spec['medians']),matrix)
        if name=='LightGBM': p=loaded[name].predict(X,num_threads=2)
        elif name=='XGBoost': p=loaded[name].predict_proba(X)[:,1]
        else: p=expit(((X-np.asarray(spec['mean']))/np.asarray(spec['scale']))@np.asarray(spec['coef'])+spec['intercept'])
        predictions[name]=p
    chosen=metadata['chosen']
    if chosen in BASE: raw=predictions[chosen]
    else:
        P=np.column_stack([predictions[n] for n in BASE])
        raw=P.mean(axis=1) if chosen=='Equal' else P@np.asarray(metadata['weights']) if chosen=='Weighted' else expit(P@np.asarray(metadata['meta_coef'])+metadata['meta_intercept'])
    return pd.DataFrame({'application_id':frame.application_id.to_numpy(),'probability':map_probability(raw,metadata['calibration'])})


def load_example(csv_path,metadata_path,pool,config=FinalConfig(),assessment_mode=False):
    if assessment_mode:
        raise ValueError('Educational OOF examples are disabled in assessment mode.')
    info=json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if (info.get('source')!='EDUCATIONAL_EXAMPLE' or info.get('trees')!=config.trees or info.get('seed')!=config.seed
            or sha256(Path(csv_path).read_bytes()).hexdigest()!=info['csv_sha256']):
        raise ValueError('Example source/configuration/checksum mismatch.')
    oof=pd.read_csv(csv_path,float_precision='round_trip')
    if set(oof.source)!={'EDUCATIONAL_EXAMPLE'}:
        raise ValueError('Every example row must be labeled.')
    validate_comparison(oof,pool)
    return oof,comparison_scores(oof),info
