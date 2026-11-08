"""Stage 'analyze': applies the confirmation/refutation rules of README section 6.

Outputs in results/<name>/: summary.json (everything, for the dashboard) and summary.md (readable).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .search import SEARCHES, top_configs
from .utils import log, read_json, read_rows, write_json

METRICS = ['profit_band', 'profit_peak', 'auc']


def bootstrap_ci(diffs, n_samples, seed=0, level=0.95):
    """Percentile bootstrap interval of the mean of paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    means = rng.choice(diffs, size=(n_samples, len(diffs)), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def wilcoxon_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if np.allclose(diffs, 0):
        return 1.0
    return float(wilcoxon(diffs, zero_method='wilcox', alternative='two-sided').pvalue)


def paired_comparison(diffs, reference_mean, cfg, seed=0):
    low, high = bootstrap_ci(diffs, cfg.bootstrap_samples, seed=seed)
    margin = cfg.delta * abs(reference_mean)
    return {
        'n_pairs': int(len(diffs)),
        'mean_diff': float(np.mean(diffs)),
        'ci_low': low,
        'ci_high': high,
        'wilcoxon_p': wilcoxon_p(diffs),
        'margin': margin,
        'non_inferior': bool(low > -margin),
    }


def compare_searches(cfg, runs):
    base = runs[runs['search'] == 'baseline'].set_index(['rank', 'seed'])
    under = runs[runs['search'] == 'undersampling'].set_index(['rank', 'seed'])
    pairs = base.index.intersection(under.index)
    out = {}
    for metric in METRICS:
        diffs = (under.loc[pairs, metric] - base.loc[pairs, metric]).to_numpy()
        out[metric] = {
            'baseline_mean': float(base.loc[pairs, metric].mean()),
            'undersampling_mean': float(under.loc[pairs, metric].mean()),
            **paired_comparison(diffs, base.loc[pairs, metric].mean(), cfg),
        }
    return out


def time_comparison(cfg, runs, timings):
    prep = timings.get('features_seconds', 0.0) + timings.get('load_seconds', 0.0)
    out = {}
    for search in SEARCHES:
        trials = read_rows(cfg.results_path / f'trials_{search}.csv')
        rank1 = runs[(runs['search'] == search) & (runs['rank'] == 1)]
        search_s = float(trials['trial_seconds'].sum())
        final_s = float(rank1['train_seconds'].mean())
        predict_s = float(rank1['predict_seconds'].mean())
        out[search] = {
            'prep_seconds': prep,
            'search_seconds': search_s,
            'final_train_seconds': final_s,
            'predict_seconds': predict_s,
            'total_seconds': prep + search_s + final_s + predict_s,
            'mean_trial_seconds': float(trials['trial_seconds'].mean()),
            'mean_trees_per_trial': float(trials['trees_built'].mean()),
            'seconds_per_tree': float(trials['train_seconds'].sum() / trials['trees_built'].sum()),
            'n_trials': int(len(trials)),
        }
    reduction = 1 - out['undersampling']['total_seconds'] / out['baseline']['total_seconds']
    search_reduction = 1 - out['undersampling']['search_seconds'] / out['baseline']['search_seconds']
    out['reduction'] = float(reduction)
    out['search_only_reduction'] = float(search_reduction)
    out['per_tree_reduction'] = float(1 - out['undersampling']['seconds_per_tree'] / out['baseline']['seconds_per_tree'])
    out['meets_threshold'] = bool(reduction >= cfg.time_reduction)
    return out


def curve_analysis(cfg, curve):
    by_r = curve.groupby('r')
    timing = ['train_seconds', 'seconds_per_tree']
    table = by_r.agg(**{f'{m}_mean': (m, 'mean') for m in METRICS + timing},
                     **{f'{m}_std': (m, 'std') for m in METRICS + timing}).reset_index()
    table = table.sort_values('r', ascending=False).reset_index(drop=True)
    ref = curve[curve['r'] == 1.0].set_index('seed')
    comparisons = []
    for r in table['r']:
        cur = curve[curve['r'] == r].set_index('seed')
        seeds = ref.index.intersection(cur.index)
        diffs = (cur.loc[seeds, 'profit_band'] - ref.loc[seeds, 'profit_band']).to_numpy()
        comparisons.append({'r': float(r), **paired_comparison(diffs, ref.loc[seeds, 'profit_band'].mean(), cfg)})
    comp = pd.DataFrame(comparisons)
    table = table.merge(comp[['r', 'mean_diff', 'ci_low', 'ci_high', 'wilcoxon_p', 'non_inferior']], on='r')

    # r_g: going down from r = 1, the last r before profit first drops more than delta.
    r_g = None
    for _, row in table.iterrows():
        if not row['non_inferior']:
            break
        r_g = float(row['r'])
    # r_t: the largest r such that every smaller r still takes >= 90% of its training time.
    times = table.set_index('r')['train_seconds_mean']
    r_t = None
    for r in table['r']:
        smaller = times[times.index < r]
        if smaller.empty or (smaller >= 0.9 * times[r]).all():
            r_t = float(r)
            break
    smallest = float(table['r'].min())
    time_flattens = r_t is not None and r_t > smallest
    useful_zone = bool(time_flattens and r_g is not None and r_g <= r_t)
    return {
        'table': table.to_dict(orient='records'),
        'r_g': r_g,
        'r_t': r_t,
        'time_flattens': bool(time_flattens),
        'useful_zone': useful_zone,
        'useful_zone_range': [r_g, r_t] if useful_zone else None,
    }


def analyze(cfg):
    path = cfg.results_path
    timings = read_json(path / 'timings.json', {})
    summary = {'experiment': cfg.experiment_name, 'config': cfg.to_dict(),
               'environment': read_json(path / 'environment.json', {}), 'timings': timings}

    runs = read_rows(path / 'runs_top.csv')
    if not runs.empty:
        summary['searches'] = compare_searches(cfg, runs)
        summary['time'] = time_comparison(cfg, runs, timings)
        summary['per_config'] = (runs.groupby(['search', 'rank'])
                                 .agg(trial=('trial', 'first'), r=('r', 'first'),
                                      **{f'{m}_mean': (m, 'mean') for m in METRICS},
                                      **{f'{m}_std': (m, 'std') for m in METRICS},
                                      train_seconds_mean=('train_seconds', 'mean'),
                                      seconds_per_tree_mean=('seconds_per_tree', 'mean'))
                                 .reset_index().to_dict(orient='records'))
        best_under = top_configs(cfg, 'undersampling', 1)[0]
        summary['undersampling_best_r'] = best_under['r']

    curve = read_rows(path / 'runs_curve.csv')
    if not curve.empty:
        summary['curve'] = curve_analysis(cfg, curve)
        if 'undersampling_best_r' in summary and summary['curve']['useful_zone']:
            low, high = summary['curve']['useful_zone_range']
            summary['optuna_r_in_zone'] = bool(low <= summary['undersampling_best_r'] <= high)

    summary['verdict'] = verdict(summary)
    write_json(path / 'summary.json', summary)
    (path / 'summary.md').write_text(to_markdown(summary, cfg))
    log.info(f"Analysis written to {path / 'summary.md'}")
    return summary


def verdict(summary):
    v = {}
    if 'searches' in summary:
        v['profit_about_the_same'] = summary['searches']['profit_band']['non_inferior']
        v['time_clearly_less'] = summary['time']['meets_threshold']
    if 'curve' in summary:
        v['time_flattens'] = summary['curve']['time_flattens']
        v['useful_zone'] = summary['curve']['useful_zone']
    if v:
        v['hypothesis_confirmed'] = all(v.values())
    return v


def _m(x):
    return f'{x / 1e6:,.2f}M'


def to_markdown(s, cfg):
    lines = [f"# Results: {s['experiment']}", '']
    env = s.get('environment', {})
    if env:
        lines += [f"Machine: {env.get('platform')}, {env.get('cpu_count')} cores, {env.get('ram_gb')} GB RAM. "
                  f"Peak memory of the run: {s['timings'].get('peak_memory_gb', float('nan')):.2f} GB.", '']
    if 'searches' in s:
        lines += ['## Undersampling search vs baseline search', '',
                  f"Top {cfg.top_k} configurations × {cfg.n_seeds} seeds, paired by rank and seed. "
                  f"Margin δ = {cfg.delta:.0%} of the baseline mean.", '',
                  '| Metric | Baseline | Undersampling | Mean diff | 95% interval | Wilcoxon p | Not worse than −δ |',
                  '|---|---|---|---|---|---|---|']
        for metric in METRICS:
            c = s['searches'][metric]
            fmt = (lambda x: f'{x:.5f}') if metric == 'auc' else _m
            lines.append(f"| {metric} | {fmt(c['baseline_mean'])} | {fmt(c['undersampling_mean'])} | "
                         f"{fmt(c['mean_diff'])} | [{fmt(c['ci_low'])}, {fmt(c['ci_high'])}] | "
                         f"{c['wilcoxon_p']:.3g} | {'yes' if c['non_inferior'] else 'no'} |")
        t = s['time']
        lines += ['', '## Time', '', '| | Baseline | Undersampling |', '|---|---|---|']
        for key in ['prep_seconds', 'search_seconds', 'final_train_seconds', 'predict_seconds', 'total_seconds',
                    'mean_trial_seconds', 'mean_trees_per_trial']:
            lines.append(f"| {key} | {t['baseline'][key]:,.1f} | {t['undersampling'][key]:,.1f} |")
        lines.append(f"| seconds_per_tree | {t['baseline']['seconds_per_tree']:.4f} | "
                     f"{t['undersampling']['seconds_per_tree']:.4f} |")
        lines += ['', f"Total time reduction: **{t['reduction']:.1%}** (search only: {t['search_only_reduction']:.1%}); "
                      f"required: {cfg.time_reduction:.0%}. Time per tree: {t['per_tree_reduction']:.1%} less.",
                  f"r chosen by the undersampling search: {s['undersampling_best_r']:.3f}.", '']
    if 'curve' in s:
        c = s['curve']
        lines += ['## r curve (best baseline configuration, only r changes)', '',
                  '| r | Band profit | Diff vs r=1 | 95% interval | AUC | Train s | s per tree |',
                  '|---|---|---|---|---|---|---|']
        for row in c['table']:
            lines.append(f"| {row['r']:g} | {_m(row['profit_band_mean'])} | {_m(row['mean_diff'])} | "
                         f"[{_m(row['ci_low'])}, {_m(row['ci_high'])}] | {row['auc_mean']:.5f} | "
                         f"{row['train_seconds_mean']:,.1f} | {row['seconds_per_tree_mean']:.4f} |")
        lines += ['', f"r_g (profit holds down to): {c['r_g']}. r_t (time flattens at): {c['r_t']}. "
                      f"Useful zone: {c['useful_zone_range'] if c['useful_zone'] else 'none'}.", '']
    if s.get('verdict'):
        lines += ['## Verdict', ''] + [f'- {k}: **{v}**' for k, v in s['verdict'].items()]
    return '\n'.join(lines) + '\n'
