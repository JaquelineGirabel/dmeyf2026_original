"""Stages 'validate' and 'test': final models, trained with several seeds.

A final model trains on the training months with the number of trees its search trial found (no early
stopping), then predicts both the validation and the test month. A training with a given configuration and
seed is the same model whichever month it is scored on, so it is done once and stored with both scores in
results/<base>/runs.csv (shared by every experiment on the same data; rows are keyed by study, trial, seed).

- 'validate': every candidate (the baseline's best configuration and the best configuration of each r)
  with the first n_seeds_valid seeds. Only the validation scores are used, to choose r.
- 'test': every candidate (the r curve; the winner's and the baseline's best configurations are candidates),
  with all n_seeds seeds. The test scores are read only by the final comparison and the curve.
Runs are appended as soon as they finish; on a rerun, runs already stored are skipped.

Every training also stores its profit curve on each month (profit contacting 0.1%, 0.2%, ..., 20% of the
clients) in results/<base>/curves/. The profit measure of the experiment is built from these curves (see
pipeline/analysis.py). A run stored before curves existed is trained again only to get its curves: the
model is the same (same configuration, seed and threads), and its row and timing are kept as they were.
"""
import numpy as np

from . import model
from .metrics import LIST_GRID, evaluate, profit_at_shares
from .search import BASELINE, optuna_study, grid_study, top_configs
from .utils import append_row, log, read_rows, threads


def runs_path(cfg):
    return cfg.base_path / 'runs.csv'


def candidates(cfg):
    """The configuration that represents each r: baseline first, then r from largest to smallest."""
    cands = [top_configs(cfg, BASELINE, 1)[0]]
    for r in cfg.r_values:
        if cfg.r_mode == 'grid':
            best = top_configs(cfg, grid_study(cfg, r), 1)
        else:
            best = top_configs(cfg, optuna_study(cfg), 1, r=r)
        if best:
            cands.append(best[0])
    return cands


def winner_configs(cfg, winner, k):
    """The winner's top k configurations (in optuna mode: the k best trials with the winning r)."""
    if winner['study'] == BASELINE:
        return top_configs(cfg, BASELINE, k)
    if cfg.r_mode == 'grid':
        return top_configs(cfg, winner['study'], k)
    return top_configs(cfg, winner['study'], k, r=winner['r'])


def _done(cfg):
    df = read_rows(runs_path(cfg))
    if df.empty:
        return set()
    return {(s, int(t), int(seed)) for s, t, seed in df[['study', 'trial', 'seed']].itertuples(index=False)}


def curves_path(cfg, study, trial, seed):
    return cfg.base_path / 'curves' / f'{study}__t{int(trial)}__s{int(seed)}.npz'


def scores_path(cfg, study, trial, seed):
    return cfg.base_path / 'scores' / f'{study}__t{int(trial)}__s{int(seed)}.npz'


def load_curves(cfg, study, trial, seed):
    """(validation curve, test curve) of one training, or None if it has not been stored."""
    path = curves_path(cfg, study, trial, seed)
    if not path.exists():
        return None
    z = np.load(path)
    return z['valid'], z['test']


def _train_one(cfg, data, conf, seed, num_threads):
    booster, train_s, _, trees, rows = model.fit(data, cfg, conf['hp'], conf['r'], seed, num_threads,
                                                 conf['best_iter'])
    row = {'study': conf['study'], 'trial': conf['trial'], 'seed': seed, 'r': conf['r'],
           'best_iter': conf['best_iter'], 'trees_built': trees, 'train_rows': rows,
           'train_seconds': train_s, 'seconds_per_tree': train_s / max(trees, 1)}
    curves, all_scores = {}, {}
    for label, split in [('valid', data.valid), ('test', data.test)]:
        scores, predict_s = model.predict(booster, split.X)
        row[f'{label}_predict_seconds'] = predict_s
        row.update({f'{label}_{k}': v for k, v in evaluate(split, scores, cfg).items()})
        curves[label] = profit_at_shares(split.baja2, scores, cfg.profit_hit, cfg.profit_cost)
        all_scores[label] = scores.astype(np.float32)
    path = curves_path(cfg, conf['study'], conf['trial'], seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, valid=curves['valid'], test=curves['test'], grid=LIST_GRID)
    # The client scores too, so the final list (stage 'final') reuses them instead of training again.
    spath = scores_path(cfg, conf['study'], conf['trial'], seed)
    spath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(spath, valid=all_scores['valid'], test=all_scores['test'])
    return row


def train_all(cfg, data, confs, seeds, stage):
    done = _done(cfg)
    num_threads = threads(cfg)
    todo = [(c, s) for c in _unique(confs) for s in seeds
            if (c['study'], c['trial'], s) not in done or not curves_path(cfg, c['study'], c['trial'], s).exists()]
    log.info(f'[{stage}] {len(todo)} trainings to run ({len(_unique(confs)) * len(seeds) - len(todo)} already done)')
    stored = read_rows(runs_path(cfg)) if any(k in done for k in [(c['study'], c['trial'], s) for c, s in todo]) else None
    for conf, seed in todo:
        row = _train_one(cfg, data, conf, seed, num_threads)
        if (conf['study'], conf['trial'], seed) in done:
            old = stored[(stored['study'] == conf['study']) & (stored['trial'] == conf['trial'])
                         & (stored['seed'] == seed)].iloc[0]
            same = all(abs(row[k] - old[k]) <= 1e-9 * max(1.0, abs(old[k])) for k in ['valid_auc', 'test_auc'])
            log.info(f"[{stage}] {conf['study']} trial {conf['trial']} seed {seed}: curves stored (row kept)"
                     + ('' if same else ' - WARNING: the retrained model differs from the stored run'))
            continue
        append_row(runs_path(cfg), row)
        log.info(f"[{stage}] {conf['study']} trial {conf['trial']} (r={conf['r']:g}) seed {seed}: "
                 f"valid band profit {row['valid_profit_band']:,.0f}, {row['train_seconds']:,.1f} s")


def _unique(confs):
    seen, out = set(), []
    for c in confs:
        if (c['study'], c['trial']) not in seen:
            seen.add((c['study'], c['trial']))
            out.append(c)
    return out


def run_validate(cfg, data):
    train_all(cfg, data, candidates(cfg), cfg.valid_seeds, 'validate')


def run_test(cfg, data):
    from .analysis import choose_r, tolerances   # the choice reads only validation scores
    runs = read_rows(runs_path(cfg))
    choice = choose_r(cfg, runs, tolerances(cfg, runs))
    winner = choice['winner']
    confs = list(candidates(cfg))
    if winner is not None:
        log.info(f"[test] winner chosen on validation: r = {winner['r']:g}")
        confs += winner_configs(cfg, winner, 1)
    else:
        log.info('[test] no r passed on validation; only the r curve is measured')
    train_all(cfg, data, confs, cfg.test_seeds, 'test')
