"""Stage 'analyze': choosing r on validation, confirming it on test, the r curve and the verdict (README sections 6-7).

A model is one configuration trained with n_seeds seeds: its seeds give its distribution. Every comparison is
between two models, distribution against distribution (seeds are not paired, configurations are never mixed).

The profit of a model:
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

from .metrics import LIST_GRID
from .runs import candidates, load_curves, runs_path, winner_configs
from .search import BASELINE, read_trials, search_seconds, studies, top_configs
from .utils import log, read_json, read_rows, write_json

TEST_METRICS = ['profit', 'auc', 'profit_peak']
DECISIVE = ['profit', 'auc']          # "about the same AUC and profit"


# ---------- statistics ----------
def bootstrap_diff_ci(new, ref, n_samples, seed=0, level=0.95):
    """Percentile bootstrap interval of mean(new) - mean(ref), two independent samples (each one resampled on
    its own: seeds of different models are not paired)."""
    rng = np.random.default_rng(seed)
    new, ref = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    diffs = (rng.choice(new, size=(n_samples, len(new)), replace=True).mean(axis=1)
             - rng.choice(ref, size=(n_samples, len(ref)), replace=True).mean(axis=1))
    alpha = (1 - level) / 2
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))


def compare_models(new, ref, cfg, share, level=0.95):
    """Two models (their seed distributions): mean(new) - mean(ref). Not worse if the interval's lower end, which
    accounts for the spread of both distributions, is above -share x |reference mean|."""
    new, ref = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    low, high = bootstrap_diff_ci(new, ref, cfg.bootstrap_samples, level=level)
    margin = share * abs(float(ref.mean()))
    return {'n_new': int(len(new)), 'n_ref': int(len(ref)), 'reference_mean': float(ref.mean()),
            'new_mean': float(new.mean()), 'mean_diff': float(new.mean() - ref.mean()), 'ci_low': low, 'ci_high': high,
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


# ---------- tolerance ----------
def tolerances(cfg, runs):
    """The tolerance of each decisive measure: how much the baseline itself varies between seeds (standard
    deviation over mean), with its best configuration and the validation seeds. It uses no undersampling result
    and no test score, so it is fixed before choosing r and before confirming. The same share serves for the
    choice, the test verdict and the r curve; profit_peak, informative, uses profit's. --delta replaces both by a
    fixed share."""
    base, _ = _with_profit(cfg, runs, candidates(cfg)[0], cfg.valid_seeds)
    out = {'source': 'baseline', 'n_seeds': int(len(base))} if cfg.delta is None else {'source': 'fixed'}
    for m, col in [('profit', 'valid_profit'), ('auc', 'valid_auc')]:
        x = base[col].to_numpy(dtype=float)
        sd, mean = float(np.std(x, ddof=1)), float(np.mean(x))
        out[m] = {'mean': mean, 'sd': sd, 'share': (sd / abs(mean)) if cfg.delta is None else float(cfg.delta)}
    out['profit_peak'] = out['profit']
    return out


# ---------- choosing r (validation only) ----------
def choose_r(cfg, runs, tol):
    """Applies the choice rule on validation scores. Never reads test scores.

    1. Every r is judged on its own: it passes if its validation profit (at its share of clients) is not worse
       than the baseline's beyond the margin (its seed distribution against the baseline's, interval at
       cfg.choice_confidence).
    2. Among the r that pass, the one with the least time (search + mean training in grid mode; mean
       training in optuna mode).
    """
    cands = candidates(cfg)
    base = cands[0]
    base_runs, _ = _with_profit(cfg, runs, base, cfg.valid_seeds)
    rows = []
    for conf in cands:
        cur, share = _with_profit(cfg, runs, conf, cfg.valid_seeds)
        if len(cur) < 2 or len(base_runs) < 2:
            raise RuntimeError("Validation runs missing. Run the 'validate' stage first.")
        train_s = float(cur['train_seconds'].mean())
        search_s = _search_part(cfg, conf)
        comp = compare_models(cur['valid_profit'], base_runs['valid_profit'], cfg, tol['profit']['share'],
                              level=cfg.choice_confidence)
        rows.append({'r': conf['r'], 'study': conf['study'], 'trial': conf['trial'],
                     'n_trials_with_r': _n_trials_with_r(cfg, conf), 'list_share': share,
                     'valid_profit': float(cur['valid_profit'].mean()),
                     'valid_auc': float(cur['valid_auc'].mean()),
                     'diff': comp['mean_diff'], 'ci_low': comp['ci_low'], 'ci_high': comp['ci_high'],
                     # each seed of this model against the baseline's average (for the charts; nothing is paired)
                     'seed_vs_base_mean': [float(x - base_runs['valid_profit'].mean()) for x in cur['valid_profit']],
                     'margin': comp['margin'], 'non_inferior': comp['non_inferior'],
                     'search_seconds': search_s, 'train_seconds': train_s, 'choice_seconds': search_s + train_s})
    table = pd.DataFrame(rows)
    base_time = float(table.loc[0, 'choice_seconds'])
    passing = [i for i in range(1, len(table)) if table.loc[i, 'non_inferior']]
    table['passes'] = table.index.isin(passing)
    winner = None
    if passing:
        win = min(passing, key=lambda i: table.loc[i, 'choice_seconds'])
        winner = cands[win]
        table['winner'] = table.index == win
    else:
        table['winner'] = False
    return {'table': table.to_dict(orient='records'), 'winner': winner, 'baseline_seconds': base_time,
            'confidence': cfg.choice_confidence,
            'rule': f'every r judged on its own ({cfg.choice_confidence:.1%} interval): least time among the r not '
                    f'worse on validation profit'}


# ---------- confirming on test ----------
def test_comparison(cfg, runs, winner, timings, tol):
    """The chosen model (the winner's best configuration) against the baseline model (its best configuration), on
    the test month: each with its n_seeds seeds, distribution against distribution."""
    seeds = cfg.test_seeds
    w_conf, b_conf = winner_configs(cfg, winner, 1)[0], top_configs(cfg, BASELINE, 1)[0]
    (w, w_share), (b, b_share) = _with_profit(cfg, runs, w_conf, seeds), _with_profit(cfg, runs, b_conf, seeds)
    models = [{'side': label, 'study': conf['study'], 'trial': conf['trial'], 'r': conf['r'], 'list_share': share,
               **{f'{m}_mean': float(df[f'test_{m}'].mean()) for m in TEST_METRICS},
               **{f'{m}_std': float(df[f'test_{m}'].std()) for m in TEST_METRICS},
               'train_seconds_mean': float(df['train_seconds'].mean()), 'seeds': [int(x) for x in df.index],
               'values': {m: [float(x) for x in df[f'test_{m}']] for m in TEST_METRICS},
               'valid_profit': [float(x) for x in df['valid_profit']]}
              for label, conf, df, share in [('baseline', b_conf, b, b_share), ('winner', w_conf, w, w_share)]]
    metrics = {m: compare_models(w[f'test_{m}'], b[f'test_{m}'], cfg, tol[m]['share']) for m in TEST_METRICS}

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

    base_t = time_of(b_conf, BASELINE)
    win_t = time_of(w_conf, winner['study'])
    reduction = 1 - win_t['total_seconds'] / base_t['total_seconds']
    return {'winner_r': winner['r'], 'metrics': metrics, 'models': models,
            'time': {'baseline': base_t, 'winner': win_t, 'reduction': float(reduction),
                     'search_only_reduction': float(1 - win_t['search_seconds'] / base_t['search_seconds']),
                     'final_train_reduction': float(1 - win_t['final_train_seconds'] / base_t['final_train_seconds']),
                     'meets_threshold': bool(reduction >= cfg.time_reduction)},
            'about_the_same': bool(all(metrics[m]['non_inferior'] for m in DECISIVE))}


# ---------- the r curve (test) ----------
def curve_analysis(cfg, runs, tol):
    seeds = cfg.test_seeds
    cands = candidates(cfg)
    base_runs, _ = _with_profit(cfg, runs, cands[0], seeds)
    rows = []
    for conf in cands:
        cur, share = _with_profit(cfg, runs, conf, seeds)
        comps = {m: compare_models(cur[f'test_{m}'], base_runs[f'test_{m}'], cfg, tol[m]['share']) for m in DECISIVE}
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
    # r_flat (the time floor): the fastest r of all (least search + training time). Below it, time does not go down.
    und = table[table['r'] < 1]
    r_flat = float(und.loc[und['time_seconds'].idxmin(), 'r'])
    time_flattens = r_flat > float(table['r'].min())
    return {'table': table.to_dict(orient='records'), 'baseline_seconds': base_time,
            'holding_r': holding, 'r_flat': r_flat, 'time_flattens': bool(time_flattens)}


def below_floor(choice, curve):
    """Part 3 of the hypothesis: does going below the time floor (r < r_flat) make the result worse? Described in
    both months: which of those r keep profit on validation (the choice rule) and profit and AUC on test (the r
    curve). 'worse' is True only if none of them holds in either month, False only if all hold in both, and None
    otherwise: then no consistent effect is observed."""
    r_flat = curve['r_flat']
    below = sorted((float(row['r']) for row in curve['table'] if row['r'] < r_flat - 1e-12), reverse=True)
    passed = [row['r'] for row in choice['table'] if row['passes']]
    valid_ok = [r for r in below if any(abs(r - x) < 1e-9 for x in passed)]
    test_ok = [r for r in below if any(abs(r - x) < 1e-9 for x in curve['holding_r'])]
    worse = None
    if curve['time_flattens'] and below:
        if not valid_ok and not test_ok:
            worse = True
        elif len(valid_ok) == len(test_ok) == len(below):
            worse = False
    return {'r_flat': r_flat, 'below_r': below, 'hold_valid_r': valid_ok, 'hold_test_r': test_ok, 'worse': worse}


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
    tol = tolerances(cfg, runs)
    summary['tolerance'] = tol
    choice = choose_r(cfg, runs, tol)
    summary['choice'] = {k: v for k, v in choice.items() if k != 'winner'}
    summary['choice']['winner_r'] = choice['winner']['r'] if choice['winner'] else None
    if choice['winner'] is not None and _has_test(runs, cfg):
        if _winner_tested(runs, cfg, choice['winner']):
            summary['test'] = test_comparison(cfg, runs, choice['winner'], timings, tol)
        else:
            log.info("The winner's test trainings are missing (settings changed since the 'test' stage?): "
                     "run the 'test' stage again")
    if _has_test(runs, cfg):
        summary['curve'] = curve_analysis(cfg, runs, tol)
        summary['below_floor'] = below_floor(summary['choice'], summary['curve'])
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
    return len(_runs_of(runs, winner_configs(cfg, winner, 1)[0], cfg.test_seeds)) == cfg.n_seeds


def verdict(s):
    """One result per part of the hypothesis:
    1. about the same profit and AUC in clearly less time (the winner against the baseline on test);
    2. below some r the time stops going down;
    3. going below that point makes the result worse: True, False, or None when no consistent effect is observed."""
    v = {'r_passed_validation': s['choice']['winner_r'] is not None}
    if 'test' in s:
        v['about_the_same_auc_and_profit'] = s['test']['about_the_same']
        v['clearly_less_time'] = s['test']['time']['meets_threshold']
    elif v['r_passed_validation'] is False:
        v['about_the_same_auc_and_profit'] = False
    if v['r_passed_validation'] is False:
        v['part1_same_with_less_time'] = False
    elif 'about_the_same_auc_and_profit' in v and 'clearly_less_time' in v:
        v['part1_same_with_less_time'] = bool(v['about_the_same_auc_and_profit'] and v['clearly_less_time'])
    if 'curve' in s:
        v['part2_time_floor'] = s['curve']['time_flattens']
        v['part3_below_floor_worse'] = s['below_floor']['worse']
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
    tol = s['tolerance']
    lines += [('Tolerance δ (what the baseline itself varies between seeds: standard deviation / mean of its best '
               f"configuration over the {tol['n_seeds']} validation seeds): profit **{tol['profit']['share']:.2%}** "
               f"({_m(tol['profit']['sd'])}), AUC **{tol['auc']['share']:.3%}** ({tol['auc']['sd']:.5f})."
               if tol['source'] == 'baseline' else
               f"Tolerance δ (fixed with --delta): **{tol['profit']['share']:.2%}** for profit and AUC."), '']
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
                  f"Best configuration of each side × {cfg.n_seeds} seeds, distribution against distribution. "
                  "Margin δ of each measure in the 'Tolerance' line above. Decisive: profit (at each model's share, fixed on validation) and AUC.", '',
                  '| Metric | Baseline | Winner | Mean diff | 95% interval | Not worse than −δ |',
                  '|---|---|---|---|---|---|']
        for metric in TEST_METRICS:
            m = t['metrics'][metric]
            fmt = (lambda x: f'{x:.5f}') if metric == 'auc' else _m
            lines.append(f"| {metric} | {fmt(m['reference_mean'])} | {fmt(m['new_mean'])} | {fmt(m['mean_diff'])} | "
                         f"[{fmt(m['ci_low'])}, {fmt(m['ci_high'])}] | "
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
        bf = s['below_floor']
        rs = lambda xs: ', '.join(f'{r:g}' for r in xs) or 'none'
        lines += ['', f"r that keep profit and AUC: {held}. Time floor r_flat (time flattens at): {cv['r_flat']:g}.", '',
                  f"Below the floor ({rs(bf['below_r'])}): keep profit on validation {rs(bf['hold_valid_r'])} "
                  f"({len(bf['hold_valid_r'])} of {len(bf['below_r'])}); keep profit and AUC on test {rs(bf['hold_test_r'])} "
                  f"({len(bf['hold_test_r'])} of {len(bf['below_r'])}).", '']
    e = s['exploration']
    lines += [f"Whole exploration of this experiment: {_h(e['total_seconds'])} "
              f"(searches {_h(e['searches_seconds'])}, {e['n_trainings']} final trainings {_h(e['trainings_seconds'])}).", '']
    lines += ['## Verdict', ''] + [f"- {k}: **{'not observed' if v is None else v}**" for k, v in s['verdict'].items()]
    return '\n'.join(lines) + '\n'
