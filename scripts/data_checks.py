"""Validate the released dataset before any model is trained."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_files(root: Path, manifest: dict) -> None:
    root = Path(root).resolve()
    for name, entry in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Unsafe manifest path: {name}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}. Run the setup cell again.')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}. Restore the released file before continuing.')


def validate_frame(frame, contract: dict, kind: str) -> dict:
    import numpy as np
    import pandas as pd

    if kind not in ('train', 'dirty', 'challenge'):
        raise ValueError('kind must be train, dirty, or challenge')
    features = contract['features']
    target = contract['target']['name']
    expected = contract['metadata'] + [f['name'] for f in features]
    if kind != 'challenge':
        expected += [target]
    if kind == 'dirty':
        expected += [f['name'] for f in contract['leakage_features']]
    if list(frame.columns) != expected:
        raise ValueError(f'{kind}: incorrect column names/order; use the released CSV.')
    if len(frame) != contract['rows'][kind]:
        raise ValueError(f'{kind}: incorrect row count.')
    if frame['application_id'].isna().any() or not frame['application_id'].is_unique:
        raise ValueError('application_id must be unique and complete.')
    if frame[contract['metadata']].isna().any().any():
        raise ValueError('Metadata must not contain missing values.')
    dates = pd.to_datetime(frame['application_date'], errors='raise', format='%Y-%m-%d')
    start, end = contract['date_ranges']['challenge' if kind == 'challenge' else 'train']
    if not dates.between(start, end).all():
        raise ValueError('Application date outside the released interval.')
    for spec in features + (contract['leakage_features'] if kind == 'dirty' else []):
        values = pd.to_numeric(frame[spec['name']], errors='raise')
        present = values.dropna()
        if not np.isfinite(present).all() or not present.between(*spec['range']).all():
            raise ValueError(f"Invalid range or non-finite value: {spec['name']}")
        if not spec['nullable'] and values.isna().any():
            raise ValueError(f"Unexpected missing value: {spec['name']}")
        if spec['dtype'] == 'integer' and not (present % 1 == 0).all():
            raise ValueError(f"Expected whole numbers: {spec['name']}")
        if spec.get('allowed') and not present.isin(spec['allowed']).all():
            raise ValueError(f"Unexpected category: {spec['name']}")
    if not (frame[['employment_government', 'employment_private', 'employment_self_employed']].sum(axis=1) == 1).all():
        raise ValueError('Exactly one employment category is required.')
    if not (frame[['region_central', 'region_western', 'region_eastern']].sum(axis=1) <= 1).all():
        raise ValueError('Region indicators are mutually exclusive; all zero means other.')
    for column in ['months_employed', 'credit_history_months']:
        available = frame[column].notna()
        if not (frame.loc[available, column] <= (frame.loc[available, 'age'] - 18) * 12).all():
            raise ValueError(f'{column} exceeds the age-based limit.')
    result = {'rows': len(frame), 'predictors': len(features), 'missing_feature_values': int(frame[[f['name'] for f in features]].isna().sum().sum())}
    if kind != 'challenge':
        if frame[target].isna().any() or not frame[target].isin([0, 1]).all():
            raise ValueError('The training target must be complete and binary.')
        rate = float(frame[target].mean())
        if not .07 <= rate <= .10:
            raise ValueError('Training positive rate outside the released 7–10% range.')
        result['positive_rate'] = rate
    return result


def load_course_data(root: Path):
    import pandas as pd

    root = Path(root)
    contract = json.loads((root / 'data/data_contract.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'data/data_manifest.json').read_text(encoding='utf-8'))
    verify_files(root, manifest)
    frames, summaries = {}, {}
    for kind in ['train', 'dirty', 'challenge']:
        frames[kind] = pd.read_csv(root / f'data/tamweel_{kind}.csv')
        summaries[kind] = validate_frame(frames[kind], contract, kind)
    if not frames['dirty'][frames['train'].columns].equals(frames['train']):
        raise ValueError('Dirty data must match the clean training rows before the two extra columns.')
    if set(frames['train']['application_id']) & set(frames['challenge']['application_id']):
        raise ValueError('Training and challenge application IDs must be separate.')
    if set(frames['train']['customer_id']) & set(frames['challenge']['customer_id']):
        raise ValueError('Training and challenge customers must be separate.')
    return frames, contract, summaries
