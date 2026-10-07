"""Rebuild the public practice data. This generator does not create challenge data."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import math
from pathlib import Path
import random

SEED = 211
TARGET = 'default_within_90d'
METADATA = ['application_id', 'customer_id', 'application_date']
FEATURES = [
    'age', 'income_sar', 'loan_amount_sar', 'tenor_months', 'bureau_score',
    'dti', 'months_employed', 'recent_inquiries', 'prior_defaults',
    'existing_obligations_sar', 'savings_balance_sar', 'salary_transfer',
    'credit_history_months', 'utilization_ratio', 'num_open_accounts',
    'residence_years', 'region_central', 'region_western', 'region_eastern',
    'employment_government', 'employment_private', 'employment_self_employed',
]
LEAKS = ['days_past_due_60', 'collection_calls']


def make_features(n: int, seed: int, *, prefix: str = 'TR',
                  start: date = date(2022, 1, 1), days: int = 1096,
                  customer_pool: int = 6500) -> list[dict]:
    """Create fictional applications; no real customers or external data are used."""
    rng = random.Random(seed)
    profiles = {}
    rows = []
    for i in range(n):
        customer = rng.randrange(customer_pool)
        if customer not in profiles:
            age = rng.randint(21, 64)
            profiles[customer] = (age, rng.gauss(9.0, 0.47), rng.randrange(4),
                                  rng.choices([0, 1, 2], [0.25, 0.6, 0.15])[0], rng.gauss(0, 1))
        age, income_log, region, employment, latent = profiles[customer]
        income = round(min(42000, max(2500, math.exp(income_log + rng.gauss(0, .12)))), 2)
        obligations = round(income * rng.uniform(.01, .47), 2)
        tenor = rng.choice([12, 24, 36, 48, 60])
        loan = round(min(350000, max(10000, income * rng.uniform(3, 22))), 2)
        score = round(min(850, max(300, 665 + 65 * latent + rng.gauss(0, 48))))
        age_months = (age - 18) * 12
        row = dict(zip(METADATA, [f'{prefix}-{i + 1:06d}', f'{prefix}-C{customer:05d}',
                                  (start + timedelta(days=rng.randrange(days))).isoformat()]))
        row.update({
            'age': age, 'income_sar': income, 'loan_amount_sar': loan,
            'tenor_months': tenor, 'bureau_score': score,
            'dti': round((obligations + loan / tenor) / income, 4),
            'months_employed': rng.randint(0, min(age_months, 360)),
            'recent_inquiries': min(10, int(rng.expovariate(1 / 1.6))),
            'prior_defaults': rng.choices([0, 1, 2], [.86, .12, .02])[0],
            'existing_obligations_sar': obligations,
            'savings_balance_sar': round(income * rng.expovariate(1 / 2.0), 2),
            'salary_transfer': int(rng.random() < (.82 if employment != 2 else .28)),
            'credit_history_months': rng.randint(0, min(age_months, 420)),
            'utilization_ratio': round(rng.betavariate(2.0, 2.8), 4),
            'num_open_accounts': rng.randint(0, 12),
            'residence_years': rng.randint(0, min(age - 18, 35)),
            'region_central': int(region == 0), 'region_western': int(region == 1),
            'region_eastern': int(region == 2),
            'employment_government': int(employment == 0),
            'employment_private': int(employment == 1),
            'employment_self_employed': int(employment == 2),
        })
        rows.append(row)
    return rows


def risk_score(row: dict) -> float:
    """A fictional, nonlinear relationship for a teaching example, not a lending rule."""
    return ((650 - row['bureau_score']) / 87 + 1.8 * max(row['dti'] - .42, 0)
            + .7 * row['prior_defaults'] + .13 * row['recent_inquiries']
            + 1.3 * max(row['utilization_ratio'] - .55, 0)
            - .35 * row['salary_transfer'] - .25 * math.log1p(row['savings_balance_sar'] / row['income_sar'])
            + .45 * (row['dti'] > .65 and row['bureau_score'] < 620))


def make_labels(rows: list[dict], seed: int, expected_rate: float = .08) -> list[int]:
    rng = random.Random(seed)
    scores = [risk_score(row) for row in rows]
    low, high = -12.0, 3.0
    for _ in range(60):
        intercept = (low + high) / 2
        mean = sum(1 / (1 + math.exp(-(s + intercept))) for s in scores) / len(scores)
        if mean < expected_rate:
            low = intercept
        else:
            high = intercept
    return [int(rng.random() < 1 / (1 + math.exp(-(s + (low + high) / 2)))) for s in scores]


def mask_features(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = [dict(row) for row in rows]
    for row in result:
        for feature in ['bureau_score', 'months_employed', 'savings_balance_sar']:
            if rng.random() < .025:
                row[feature] = ''
    return result


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def generate_public(output: Path, n: int = 10000, seed: int = SEED) -> dict:
    if n < 100:
        raise ValueError('Use at least 100 rows for this teaching dataset.')
    raw = make_features(n, seed)
    labels = make_labels(raw, seed + 1)
    rows = mask_features(raw, seed + 2)
    for row, label in zip(rows, labels):
        row[TARGET] = label
    rows.sort(key=lambda row: (row['application_date'], row['application_id']))
    write_csv(output / 'tamweel_train.csv', rows, METADATA + FEATURES + [TARGET])
    rng = random.Random(seed + 3)
    dirty = [dict(row, days_past_due_60=max(0, round(rng.gauss(48 if row[TARGET] else 4, 12))),
                  collection_calls=rng.randint(4, 15) if row[TARGET] else rng.randint(0, 4)) for row in rows]
    write_csv(output / 'tamweel_dirty.csv', dirty, METADATA + FEATURES + [TARGET] + LEAKS)
    return {'rows': n, 'positive_rate': sum(labels) / n, 'seed': seed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    print(generate_public(args.output, args.rows, args.seed))
