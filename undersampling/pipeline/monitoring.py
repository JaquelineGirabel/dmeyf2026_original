"""Monitoring metrics of the data and of the model (run with monitor.py; read by the dashboard).

Everything here is descriptive: nothing feeds back into the experiment (no stage reads it).

- Clients and churn by month: clients, BAJA+1, BAJA+2 and CONTINUA per month of the features file.
- Broken variables: a variable is "broken" in a month when it takes a single value (all zeros, or all
  empty) that month, while in most other months it takes two or more. Checked on each raw variable
  and on its lags: lag k in month m shows the variable in month m - k, so the lags also reveal the months
  that are not in the features file (e.g. 201907 through the lag 1 of 201908). A lag that is entirely
  empty is a month absent from the raw data (e.g. before its first month), not a broken variable.
- Drift: PSI of each raw variable in each month against the training months pooled. Bins = deciles of
  the training values, plus one bin for empty values. Usual reading: < 0.10 stable, 0.10-0.25 moderate
  change, > 0.25 large change.
- Importance: gain of each feature in the baseline's best configuration (and of each experiment's
  winner), trained once on the training months with the first final seed, as the final models are.
- Predictions: with that baseline model, average predicted probability against the real share of
  positives, per month (validation and test), plus the distribution of the scores.
"""
import json
import re
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq

from . import model
from .data import META_COLS
from .metrics import evaluate
from .runs import winner_configs
from .search import BASELINE, top_configs
from .utils import log, threads, write_json

LAG_RE = re.compile(r'^(?P<var>.+)_lag(?P<k>\d+)$')
DERIVED_RE = re.compile(r'^(?P<var>.+)_(lag|delta)\d+$')
PSI_EPS = 1e-4
N_BINS = 10


def _month_add(yyyymm, k):
    i = (yyyymm // 100) * 12 + yyyymm % 100 - 1 + k
    return (i // 12) * 100 + i % 12 + 1


def feature_type(feature):
    """'original', 'lag1', 'lag2', 'delta1', 'delta2', ..."""
    m = re.match(r'^.+_(lag|delta)(\d+)$', feature)
    return f'{m.group(1)}{m.group(2)}' if m else 'original'


def base_variable(feature):
    m = DERIVED_RE.match(feature)
    return m.group('var') if m else feature


# ---------------------------------------------------------------- clients and churn
def class_table(clase, months):
    rows = []
    for m in sorted(set(months.tolist())):
        c = clase[months == m]
        n = len(c)
        b1, b2 = int((c == 'BAJA+1').sum()), int((c == 'BAJA+2').sum())
        rows.append({'month': int(m), 'clients': n, 'baja1': b1, 'baja2': b2, 'continua': n - b1 - b2,
                     'baja1_share': b1 / n, 'baja2_share': b2 / n, 'churn_share': (b1 + b2) / n})
    return rows


# ---------------------------------------------------------------- broken variables
def _column_stats(x):
    """Share of empty values, share of zeros, number of distinct non-empty values (0, 1, 2, or 3 = three or more)."""
    n = len(x)
    empty = np.isnan(x)
    vals = x[~empty]
    if len(vals) == 0:
        distinct = 0
    elif vals.min() == vals.max():
        distinct = 1
    else:
        distinct = 3 if len(np.unique(vals)) >= 3 else 2
    return {'empty': float(empty.mean()) if n else 1.0, 'zero': float((vals == 0).sum() / n) if n else 0.0,
            'distinct': distinct}


def broken_variables(table, months, raw_vars):
    """(variable, real month) pairs where the variable takes a single value; see module docstring."""
    month_list = sorted(set(months.tolist()))
    masks = {m: months == m for m in month_list}
    # stats[(variable, real_month)] -> list of observations (one from the raw column, one per lag)
    obs, per_column = {}, {}
    columns = [(v, v, 0) for v in raw_vars]
    for name in table.column_names:
        lag = LAG_RE.match(name)
        if lag and lag.group('var') in raw_vars:
            columns.append((name, lag.group('var'), int(lag.group('k'))))
    for col, var, k in columns:
        x = table.column(col).to_numpy(zero_copy_only=False).astype(np.float64)
        stats = {m: _column_stats(x[masks[m]]) for m in month_list}
        per_column[col] = stats
        for m, s in stats.items():
            if k > 0 and s['empty'] == 1.0:
                continue                                  # month absent from the raw data
            obs.setdefault((var, _month_add(m, -k)), []).append((col, s))
    # A variable "normally" takes several values if most of its real months do.
    normal = {}
    for (var, real), items in obs.items():
        normal.setdefault(var, []).append(min(s['distinct'] for _, s in items))
    broken = []
    for (var, real), items in sorted(obs.items()):
        single = [(col, s) for col, s in items if s['distinct'] <= 1]
        others = normal[var]
        if not single or np.median(others) < 2 or sum(d >= 2 for d in others) < 3:
            continue
        col, s = single[0]
        kind = 'all_empty' if s['distinct'] == 0 else ('all_zero' if s['zero'] > 0.999 else 'constant')
        broken.append({'variable': var, 'month': int(real), 'kind': kind,
                       'seen_in': sorted(c for c, _ in single), 'in_file': real in month_list})
    return broken


# ---------------------------------------------------------------- drift (PSI)
def _psi(ref_share, cur_share):
    a, e = np.maximum(cur_share, PSI_EPS), np.maximum(ref_share, PSI_EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(table, months, raw_vars, reference_months):
    month_list = sorted(set(months.tolist()))
    ref_mask = np.isin(months, reference_months)
    out = {}
    for var in raw_vars:
        x = table.column(var).to_numpy(zero_copy_only=False).astype(np.float64)
        ref = x[ref_mask]
        ref_vals = ref[~np.isnan(ref)]
        if len(ref_vals) == 0:
            continue
        edges = np.unique(np.quantile(ref_vals, np.linspace(0, 1, N_BINS + 1)[1:-1]))

        def shares(v):
            empty = np.isnan(v)
            bins = np.searchsorted(edges, v[~empty], side='right')
            counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
            return np.append(counts, empty.sum()) / max(len(v), 1)

        ref_share = shares(ref)
        out[var] = {str(m): _psi(ref_share, shares(x[months == m])) for m in month_list}
    return out


# ---------------------------------------------------------------- importance and predictions
def _importance(booster, feature_names):
    gain = booster.feature_importance(importance_type='gain')
    total = gain.sum() or 1.0
    by_feature = {f: float(g / total) for f, g in zip(feature_names, gain) if g > 0}
    by_variable = {}
    for f, share in by_feature.items():
        by_variable[base_variable(f)] = by_variable.get(base_variable(f), 0.0) + share
    by_type = {}
    for f, share in by_feature.items():
        by_type[feature_type(f)] = by_type.get(feature_type(f), 0.0) + share
    top = lambda d, n: dict(sorted(d.items(), key=lambda kv: -kv[1])[:n])
    return {'by_feature': top(by_feature, 60), 'by_variable': top(by_variable, 60), 'by_type': by_type,
            'n_features_used': len(by_feature)}


def _train(cfg, data, conf, seed):
    booster, seconds, _, _, _ = model.fit(data, cfg, conf['hp'], conf['r'], seed, threads(cfg), conf['best_iter'])
    return booster, seconds


def _predictions(cfg, booster, data):
    bins = np.concatenate([[0], np.geomspace(1e-4, 1, 41)])
    out = {'bins': bins.tolist()}
    for label, split, month in [('valid', data.valid, cfg.valid_month), ('test', data.test, cfg.test_month)]:
        scores, _ = model.predict(booster, split.X)
        m = evaluate(split, scores, cfg)
        hist = np.histogram(scores, bins=bins)[0] / len(scores)
        out[label] = {'month': month, 'mean_score': float(scores.mean()), 'positive_share': float(split.y.mean()),
                      'auc': float(m['auc']), 'profit_band': float(m['profit_band']),
                      'hist': hist.tolist(), 'hist_pos': (np.histogram(scores[split.y == 1], bins=bins)[0]
                                                          / max(split.y.sum(), 1)).tolist()}
    return out


def experiment_winners(cfg):
    """(experiment name, winner configuration) for each experiment folder whose choice found a winner."""
    found = []
    for summary_path in sorted(cfg.base_path.glob('*/summary.json')):
        s = json.loads(summary_path.read_text())
        choice = s.get('choice') or {}
        row = next((r for r in choice.get('table', []) if r.get('winner')), None)
        if row is None or row['r'] >= 1:
            continue
        exp_cfg = replace(cfg, r_mode=s['r_mode'], r_apply=s['r_apply'])
        conf = winner_configs(exp_cfg, {'study': row['study'], 'r': row['r']}, 1)
        if conf:
            found.append((summary_path.parent.name, exp_cfg, conf[0]))
    return found


# ---------------------------------------------------------------- everything
def monitoring_path(cfg):
    return cfg.base_path / 'monitoring' / 'monitoring.json'


def run_monitoring(cfg, parquet_path, data):
    schema = pq.read_schema(parquet_path)
    features = [c for c in schema.names if c not in META_COLS]
    raw_vars = [c for c in features if not DERIVED_RE.match(c)]
    lag_cols = [c for c in features if LAG_RE.match(c) and LAG_RE.match(c).group('var') in raw_vars]
    table = pq.read_table(parquet_path, columns=['foto_mes', 'clase_ternaria'] + raw_vars + lag_cols)
    months = table.column('foto_mes').to_numpy()
    clase = table.column('clase_ternaria').to_numpy(zero_copy_only=False)
    log.info(f'[monitor] {table.num_rows:,} rows, {len(raw_vars)} raw variables, {len(lag_cols)} lags')

    out = {'months': sorted(int(m) for m in set(months.tolist())), 'train_months': cfg.train_months,
           'valid_month': cfg.valid_month, 'test_month': cfg.test_month, 'n_features': len(features),
           'n_raw_variables': len(raw_vars), 'psi_levels': [0.10, 0.25],
           'feature_types': {t: sum(feature_type(f) == t for f in features)
                             for t in sorted({feature_type(f) for f in features})}}
    meta_path = parquet_path.with_suffix('.json')
    if meta_path.exists():
        out['text_columns_converted'] = json.loads(meta_path.read_text()).get('text_columns_converted', [])
    out['classes'] = class_table(clase, months)
    out['broken'] = broken_variables(table, months, raw_vars)
    log.info(f"[monitor] {len(out['broken'])} broken (variable, month) pairs")
    out['psi'] = drift(table, months, raw_vars, cfg.train_months)
    del table

    base_conf = top_configs(cfg, BASELINE, 1)
    if base_conf:
        seed = cfg.test_seeds[0]
        booster, seconds = _train(cfg, data, base_conf[0], seed)
        out['importance'] = {'baseline': {**_importance(booster, data.feature_names), 'trial': base_conf[0]['trial'],
                                          'seed': seed, 'train_seconds': seconds}}
        out['predictions'] = _predictions(cfg, booster, data)
        for exp_name, exp_cfg, conf in experiment_winners(cfg):
            booster, seconds = _train(exp_cfg, data, conf, seed)
            out['importance'][exp_name] = {**_importance(booster, data.feature_names), 'trial': conf['trial'],
                                           'study': conf['study'], 'r': conf['r'], 'seed': seed,
                                           'train_seconds': seconds}
        log.info(f"[monitor] importance for: {list(out['importance'])}")
    else:
        log.info('[monitor] no baseline search yet: importance and predictions skipped')
    path = monitoring_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    log.info(f'[monitor] written to {path}')
    return out
