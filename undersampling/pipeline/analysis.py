"""Stage 'analyze': choosing r on validation, confirming it on test, the r curve and the verdict (README sections 6-7).

The profit of a model (a configuration, whose seeds give its distribution):
1. On the validation month, its profit curves (profit contacting 0.1%, 0.2%, ..., 20% of the clients) are
   averaged over its n_seeds_valid seeds. The share where that average curve is highest is the model's
   share of clients to contact: it defines its list.
2. Its profit on a month, for each seed, is the profit of contacting that share of the clients. On the test
   month the share is the one fixed on validation.
AUC does not use any share.

Outputs in results/<base>/<mode>_<r_apply>/: summary.json (everything, for the dashboard) and summary.md.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import LIST_GRID
from .runs import candidates, load_curves, runs_path, winner_configs
from .search import BASELINE, read_trials, search_seconds, studies, top_configs
from .utils import log, read_json, read_rows, write_json

TEST_METRICS = ['profit', 'auc', 'profit_peak']
DECISIVE = ['profit', 'auc']          # "about the same AUC and profit"


# ---------- statistics ----------
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


def paired_comparison(new, ref, cfg, level=0.95):
    """new - ref, paired. Non-inferior if the interval's lower end is above -delta x |reference mean|."""
    diffs = np.asarray(new, dtype=float) - np.asarray(ref, dtype=float)
    low, high = bootstrap_ci(diffs, cfg.bootstrap_samples, level=level)
    margin = cfg.delta * abs(float(np.mean(ref)))
    return {'n_pairs': int(len(diffs)), 'reference_mean': float(np.mean(ref)), 'new_mean': float(np.mean(new)),
            'mean_diff': float(np.mean(diffs)), 'ci_low': low, 'ci_high': high, 'wilcoxon_p': wilcoxon_p(diffs),
            'margin': margin, 'non_inferior': bool(low > -margin)}


# ---------- helpers ----------
def _runs_of(runs, conf, seeds):
    """Runs of one configuration, indexed by seed, in the order of seeds (only the seeds present)."""
    df = runs[(runs['study'] == conf['study']) & (runs['trial'] == conf['trial'])].set_index('seed')
    return df.loc[[s for s in seeds if s in df.index]]


def list_share(cfg, conf):
    """The model's share of clients: where its validation profit curve, averaged over its validation seeds, peaks."""
    curves = [load_curves(cfg, conf['study'], conf['trial'], seed) for seed in cfg.valid_seeds]
    if any(c is None for c in curves):
        raise RuntimeError(f"Profit curves missing for {conf['study']} trial {conf['trial']}: run the 'validate' "
                           "and 'test' stages again (they store the curves without changing the stored runs)")
    k = int(np.argmax(np.mean([c[0] for c in curves], axis=0)))
    return k, float(LIST_GRID[k])


def _with_profit(cfg, runs, conf, seeds):
    """Runs of one model with its profit on each month at its share (columns valid_profit, test_profit)."""
    df = _runs_of(runs, conf, seeds).copy()
    k, share = list_share(cfg, conf)
    for month, j in [('valid', 0), ('test', 1)]:
        vals = []
        for seed in df.index:
            c = load_curves(cfg, conf['study'], conf['trial'], seed)
            vals.append(float(c[j][k]) if c is not None else np.nan)
        df[f'{month}_profit'] = vals
    return df, share


def _search_part(cfg, conf):
    """Search time charged to a candidate when choosing r and on the curve: its own search in grid mode;
    none in optuna mode, where one search serves every r (so all r, baseline included, compare by mean
    training time, and the search's total is reported once)."""
    return search_seconds(cfg, conf['study']) if cfg.r_mode == 'grid' else 0.0


def _n_trials_with_r(cfg, conf):
    trials = read_trials(cfg, conf['study'])
    return int(((trials['r'] - conf['r']).abs() < 1e-12).sum())


# ---------- choosing r (validation only) ----------
def choose_r(cfg, runs):
    """Applies the choice rule on validation scores. Never reads test scores.

    1. Every r is judged on its own: it passes if its validation profit (at its share of clients) is not worse
       than the baseline's beyond the margin (paired by seed, interval at cfg.choice_confidence).
    2. Among the r that pass, the one with the least time (search + mean training in grid mode; mean
       training in optuna mode).
    3. If several are within near_fastest x baseline time of the fastest, the largest r among them.
    """
    cands = candidates(cfg)
    base = cands[0]
    base_runs, _ = _with_profit(cfg, runs, base, cfg.valid_seeds)
    rows = []
    for conf in cands:
        cur, share = _with_profit(cfg, runs, conf, cfg.valid_seeds)
        seeds = [s for s in cur.index if s in base_runs.index]
        if len(seeds) < 2:
            raise RuntimeError("Validation runs missing. Run the 'validate' stage first.")
        train_s = float(cur['train_seconds'].mean())
        search_s = _search_part(cfg, conf)
        comp = paired_comparison(cur.loc[seeds, 'valid_profit'], base_runs.loc[seeds, 'valid_profit'], cfg,
                                 level=cfg.choice_confidence)
        rows.append({'r': conf['r'], 'study': conf['study'], 'trial': conf['trial'],
                     'n_trials_with_r': _n_trials_with_r(cfg, conf), 'list_share': share,
                     'valid_profit': float(cur['valid_profit'].mean()),
                     'valid_auc': float(cur['valid_auc'].mean()),
                     'diff': comp['mean_diff'], 'ci_low': comp['ci_low'], 'ci_high': comp['ci_high'],
                     'margin': comp['margin'], 'non_inferior': comp['non_inferior'],
                     'search_seconds': search_s, 'train_seconds': train_s, 'choice_seconds': search_s + train_s})
    table = pd.DataFrame(rows)
    base_time = float(table.loc[0, 'choice_seconds'])
    passing = [i for i in range(1, len(table)) if table.loc[i, 'non_inferior']]
    table['passes'] = table.index.isin(passing)
    winner = None
    if passing:
        fastest = float(table.loc[passing, 'choice_seconds'].min())
        tied = [i for i in passing if table.loc[i, 'choice_seconds'] <= fastest + cfg.near_fastest * base_time]
        win = max(tied, key=lambda i: table.loc[i, 'r'])
        winner = cands[win]
        table['winner'] = table.index == win
    else:
        table['winner'] = False
    return {'table': table.to_dict(orient='records'), 'winner': winner, 'baseline_seconds': base_time,
            'confidence': cfg.choice_confidence,
            'rule': f'every r judged on its own ({cfg.choice_confidence:.1%} interval): least time among the r not '
                    f'worse on validation profit; ties within {cfg.near_fastest:.0%} of baseline time -> largest r'}


# ---------- confirming on test ----------
def test_comparison(cfg, runs, winner, timings):
    seeds = cfg.test_seeds
    w_confs = winner_configs(cfg, winner, cfg.top_k)
    b_confs = top_configs(cfg, BASELINE, cfg.top_k)
    k = min(len(w_confs), len(b_confs))
    new, ref = {m: [] for m in TEST_METRICS}, {m: [] for m in TEST_METRICS}
    per_config = []
    for rank in range(k):
        (w, w_share), (b, b_share) = (_with_profit(cfg, runs, w_confs[rank], seeds),
                                      _with_profit(cfg, runs, b_confs[rank], seeds))
        common = [s for s in w.index if s in b.index]
        for m in TEST_METRICS:
            new[m] += list(w.loc[common, f'test_{m}'])
            ref[m] += list(b.loc[common, f'test_{m}'])
        for label, conf, df, share in [('baseline', b_confs[rank], b, b_share), ('winner', w_confs[rank], w, w_share)]:
            per_config.append({'side': label, 'rank': rank + 1, 'study': conf['study'], 'trial': conf['trial'],
                               'r': conf['r'], 'list_share': share, **{f'{m}_mean': float(df[f'test_{m}'].mean()) for m in TEST_METRICS},
                               **{f'{m}_std': float(df[f'test_{m}'].std()) for m in TEST_METRICS},
                               'train_seconds_mean': float(df['train_seconds'].mean()),
                               'seeds': [int(x) for x in df.index],
                               'values': {m: [float(x) for x in df[f'test_{m}']] for m in TEST_METRICS},
                               'valid_profit': [float(x) for x in df['valid_profit']]})
    metrics = {m: paired_comparison(new[m], ref[m], cfg) for m in TEST_METRICS}

    prep = timings.get('features_seconds', 0.0) + timings.get('load_seconds', 0.0)

    def time_of(conf, search_study):
        df = _runs_of(runs, conf, seeds)
        trials = read_trials(cfg, search_study)
        mine = trials if search_study == BASELINE or cfg.r_mode == 'grid' else trials[(trials['r'] - conf['r']).abs() < 1e-12]
        return {'prep_seconds': prep, 'search_seconds': search_seconds(cfg, search_study),
                'final_train_seconds': float(df['train_seconds'].mean()),
                'predict_seconds': float(df['test_predict_seconds'].mean()),
                'total_seconds': prep + search_seconds(cfg, search_study) + float(df['train_seconds'].mean())
                + float(df['test_predict_seconds'].mean()),
                'n_trials': int(len(trials)), 'mean_trial_seconds': float(trials['trial_seconds'].mean()),
                'seconds_per_tree': float(mine['train_seconds'].sum() / mine['trees_built'].sum())}

    base_t = time_of(b_confs[0], BASELINE)
    win_t = time_of(w_confs[0], winner['study'])
    reduction = 1 - win_t['total_seconds'] / base_t['total_seconds']
    return {'winner_r': winner['r'], 'top_k': k, 'metrics': metrics, 'per_config': per_config,
            'time': {'baseline': base_t, 'winner': win_t, 'reduction': float(reduction),
                     'search_only_reduction': float(1 - win_t['search_seconds'] / base_t['search_seconds']),
                     'final_train_reduction': float(1 - win_t['final_train_seconds'] / base_t['final_train_seconds']),
                     'meets_threshold': bool(reduction >= cfg.time_reduction)},
            'about_the_same': bool(all(metrics[m]['non_inferior'] for m in DECISIVE))}


# ---------- the r curve (test) ----------
def curve_analysis(cfg, runs):
    seeds = cfg.test_seeds
    cands = candidates(cfg)
    base_runs, _ = _with_profit(cfg, runs, cands[0], seeds)
    rows = []
    for conf in cands:
        cur, share = _with_profit(cfg, runs, conf, seeds)
        common = [s for s in cur.index if s in base_runs.index]
        comps = {m: paired_comparison(cur.loc[common, f'test_{m}'], base_runs.loc[common, f'test_{m}'], cfg)
                 for m in DECISIVE}
        search_s = _search_part(cfg, conf)
        train_s = float(cur['train_seconds'].mean())
        rows.append({'r': conf['r'], 'study': conf['study'], 'trial': conf['trial'], 'n_seeds': len(cur),
                     'n_trials_with_r': _n_trials_with_r(cfg, conf), 'list_share': share,
                     **{f'{m}_{s}': float(getattr(cur[f'test_{m}'], s)()) for m in TEST_METRICS
                        for s in ['mean', 'std', 'min', 'max']},
                     'profit_diff': comps['profit']['mean_diff'],
                     'profit_ci_low': comps['profit']['ci_low'],
                     'profit_ci_high': comps['profit']['ci_high'],
                     'auc_diff': comps['auc']['mean_diff'], 'auc_ci_low': comps['auc']['ci_low'],
                     'non_inferior': all(c['non_inferior'] for c in comps.values()),
                     'search_seconds': search_s, 'train_seconds': train_s,
                     'train_seconds_min': float(cur['train_seconds'].min()),
                     'train_seconds_max': float(cur['train_seconds'].max()),
                     'seconds_per_tree': float(cur['seconds_per_tree'].mean()),
                     'time_seconds': search_s + train_s})
    table = pd.DataFrame(rows)          # r from 1 down
    base_time = float(table.loc[0, 'time_seconds'])

    # Every r on its own: which r keep profit and AUC within the tolerance on the test month.
    holding = [float(r) for r, ok in zip(table['r'], table['non_inferior']) if r < 1 and ok]
    # r_flat: going down, the first r below which no smaller r saves more than flat_share x baseline time.
    times = table['time_seconds'].to_numpy()
    r_flat = float(table['r'].iloc[-1])
    for i in range(len(table) - 1):
        if times[i] - times[i + 1:].min() < cfg.flat_share * base_time:
            r_flat = float(table.loc[i, 'r'])
            break
    time_flattens = r_flat > float(table['r'].min())
    # Useful zone: r where time has already flattened (r <= r_flat) and profit and AUC still hold.
    zone = sorted(r for r in holding if r <= r_flat)
    useful_zone = bool(time_flattens and zone)
    return {'table': table.to_dict(orient='records'), 'baseline_seconds': base_time,
            'holding_r': holding, 'r_flat': r_flat, 'time_flattens': bool(time_flattens),
            'useful_zone': useful_zone, 'useful_zone_r': zone,
            'useful_zone_range': [zone[0], zone[-1]] if useful_zone else None}


def exploration_seconds(cfg, runs):
    """Everything this experiment ran: all its searches plus all its final trainings."""
    names = [s for s, _ in studies(cfg)]
    searches = sum(search_seconds(cfg, s) for s in names)
    mine = runs[runs['study'].isin(names)]
    return {'searches_seconds': float(searches), 'trainings_seconds': float(mine['train_seconds'].sum()),
            'n_trainings': int(len(mine)), 'total_seconds': float(searches + mine['train_seconds'].sum())}


# ---------- main entry ----------
def analyze(cfg):
    out = cfg.results_path
    timings = read_json(out / 'timings.json', {})
    summary = {'experiment': cfg.experiment_name, 'base': cfg.base_name, 'r_mode': cfg.r_mode,
               'r_apply': cfg.r_apply, 'config': cfg.to_dict(),
               'environment': read_json(out / 'environment.json', {}), 'timings': timings}
    runs = read_rows(runs_path(cfg))
    if runs.empty:
        log.info('No runs yet: nothing to analyze')
        return summary
    choice = choose_r(cfg, runs)
    summary['choice'] = {k: v for k, v in choice.items() if k != 'winner'}
    summary['choice']['winner_r'] = choice['winner']['r'] if choice['winner'] else None
    if choice['winner'] is not None and _has_test(runs, cfg):
        if _winner_tested(runs, cfg, choice['winner']):
            summary['test'] = test_comparison(cfg, runs, choice['winner'], timings)
        else:
            log.info("The winner's test trainings are missing (settings changed since the 'test' stage?): "
                     "run the 'test' stage again")
    if _has_test(runs, cfg):
        summary['curve'] = curve_analysis(cfg, runs)
    summary['exploration'] = exploration_seconds(cfg, runs)
    final = read_json(out / 'final_list.json')
    if final:
        summary['final_list'] = final      # additional result: not part of the verdict
    summary['verdict'] = verdict(summary)
    write_json(out / 'summary.json', summary)
    (out / 'summary.md').write_text(to_markdown(summary, cfg))
    log.info(f"Analysis written to {out / 'summary.md'}")
    return summary


def _has_test(runs, cfg):
    base = top_configs(cfg, BASELINE, 1)[0]
    return len(_runs_of(runs, base, cfg.test_seeds)) == cfg.n_seeds


def _winner_tested(runs, cfg, winner):
    return all(len(_runs_of(runs, c, cfg.test_seeds)) == cfg.n_seeds for c in winner_configs(cfg, winner, cfg.top_k))


def verdict(s):
    v = {'r_passed_validation': s['choice']['winner_r'] is not None}
    if 'test' in s:
        v['about_the_same_auc_and_profit'] = s['test']['about_the_same']
        v['clearly_less_time'] = s['test']['time']['meets_threshold']
    elif v['r_passed_validation'] is False:
        v['about_the_same_auc_and_profit'] = False
    if 'curve' in s:
        v['time_flattens'] = s['curve']['time_flattens']
        v['useful_zone'] = s['curve']['useful_zone']
    complete = all(k in v for k in ['about_the_same_auc_and_profit', 'clearly_less_time', 'time_flattens'])
    if complete or v['r_passed_validation'] is False:
        v['hypothesis_confirmed'] = bool(all(v.values()))
    return v


def _m(x):
    return f'{x / 1e6:,.2f}M'


def _h(seconds):
    return f'{seconds / 3600:,.2f} h' if seconds >= 3600 else f'{seconds:,.1f} s'


def to_markdown(s, cfg):
    lines = [f"# Results: {s['base']} / {s['experiment']}", '',
             f"Mode **{s['r_mode']}**, r applied **{s['r_apply']}**. r values: {', '.join(f'{r:g}' for r in cfg.r_values)}.", '']
    env = s.get('environment', {})
    if env:
        lines += [f"Machine: {env.get('platform')}, {env.get('cpu_count')} cores, {env.get('ram_gb')} GB RAM. "
                  f"Peak memory: {s['timings'].get('peak_memory_gb', float('nan')):.2f} GB.", '']
    c = s['choice']
    lines += ['## 1. Choosing r (validation month, '
              f'{cfg.n_seeds_valid} seeds)', '', c['rule'] + '.', '',
              '| r | trials with r | Share contacted | Profit | Diff vs r=1 | 95% interval | Passes | Time used to choose | Winner |',
              '|---|---|---|---|---|---|---|---|---|']
    for row in c['table']:
        lines.append(f"| {row['r']:g} | {row['n_trials_with_r']} | {row['list_share']:.1%} | {_m(row['valid_profit'])} | {_m(row['diff'])} | "
                     f"[{_m(row['ci_low'])}, {_m(row['ci_high'])}] | {'—' if row['r'] == 1 else ('yes' if row['passes'] else 'no')} | "
                     f"{_h(row['choice_seconds'])} | {'**yes**' if row['winner'] else ''} |")
    lines += ['', f"Winner: **r = {c['winner_r']:g}**." if c['winner_r'] is not None else 'No r passed: no winner.', '']
    if 'test' in s:
        t = s['test']
        lines += [f"## 2. Winner (r = {t['winner_r']:g}) vs baseline on the test month", '',
                  f"Top {t['top_k']} configurations × {cfg.n_seeds} seeds each side, paired by rank and seed. "
                  f"Margin δ = {cfg.delta:.0%} of the baseline mean. Decisive: profit (at each model's share, fixed on validation) and AUC.", '',
                  '| Metric | Baseline | Winner | Mean diff | 95% interval | Wilcoxon p | Not worse than −δ |',
                  '|---|---|---|---|---|---|---|']
        for metric in TEST_METRICS:
            m = t['metrics'][metric]
            fmt = (lambda x: f'{x:.5f}') if metric == 'auc' else _m
            lines.append(f"| {metric} | {fmt(m['reference_mean'])} | {fmt(m['new_mean'])} | {fmt(m['mean_diff'])} | "
                         f"[{fmt(m['ci_low'])}, {fmt(m['ci_high'])}] | {m['wilcoxon_p']:.3g} | "
                         f"{'yes' if m['non_inferior'] else 'no'} |")
        tm = t['time']
        lines += ['', '| Time | Baseline | Winner |', '|---|---|---|']
        for key in ['prep_seconds', 'search_seconds', 'final_train_seconds', 'predict_seconds', 'total_seconds']:
            lines.append(f"| {key} | {_h(tm['baseline'][key])} | {_h(tm['winner'][key])} |")
        lines.append(f"| seconds_per_tree (search) | {tm['baseline']['seconds_per_tree']:.4f} | "
                     f"{tm['winner']['seconds_per_tree']:.4f} |")
        lines += ['', f"Total time reduction: **{tm['reduction']:.1%}** (required {cfg.time_reduction:.0%}); "
                      f"search only {tm['search_only_reduction']:.1%}; final training {tm['final_train_reduction']:.1%}.", '']
    if 'curve' in s:
        cv = s['curve']
        lines += ['## 3. r curve (test month, best configuration of each r)', '',
                  '| r | Share | Profit | Diff vs r=1 | 95% interval | AUC | Not worse | Time | Train s |',
                  '|---|---|---|---|---|---|---|---|---|']
        for row in cv['table']:
            lines.append(f"| {row['r']:g} | {row['list_share']:.1%} | {_m(row['profit_mean'])} | {_m(row['profit_diff'])} | "
                         f"[{_m(row['profit_ci_low'])}, {_m(row['profit_ci_high'])}] | {row['auc_mean']:.5f} | "
                         f"{'—' if row['r'] == 1 else ('yes' if row['non_inferior'] else 'no')} | "
                         f"{_h(row['time_seconds'])} | {row['train_seconds']:,.1f} |")
        held = ', '.join(f'{r:g}' for r in cv['holding_r']) or 'none'
        lines += ['', f"r that keep profit and AUC: {held}. r_flat (time flattens at): {cv['r_flat']:g}. "
                      f"Useful zone: {cv['useful_zone_range'] if cv['useful_zone'] else 'none'}.", '']
    e = s['exploration']
    lines += [f"Whole exploration of this experiment: {_h(e['total_seconds'])} "
              f"(searches {_h(e['searches_seconds'])}, {e['n_trainings']} final trainings {_h(e['trainings_seconds'])}).", '']
    lines += ['## Verdict', ''] + [f'- {k}: **{v}**' for k, v in s['verdict'].items()]
    return '\n'.join(lines) + '\n'
