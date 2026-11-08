"""Stages 2 and 3: final models trained with several seeds and measured on the test month.

Stage 2 ('seeds'): the top_k configurations of each search, each with n_seeds seeds.
Stage 3 ('curve'): the best baseline configuration, changing only r along r_grid, with n_seeds seeds.

Final models train on the training months with the number of trees found by early stopping in the
search (no early stopping on test). Each run is appended to a CSV as soon as it finishes; on a
rerun, runs already in the CSV are skipped.
"""
import json

from . import model
from .metrics import evaluate
from .search import SEARCHES, top_configs
from .utils import append_row, log, read_rows, threads


def _done_keys(path, keys):
    df = read_rows(path)
    if df.empty:
        return set()
    return {tuple(row) for row in df[keys].itertuples(index=False, name=None)}


def _run_one(cfg, data, hp, r, best_iter, seed, num_threads):
    params = model.lgb_params(hp, cfg, seed=seed, num_threads=num_threads, r=r)
    booster, train_s, _, trees_built = model.train(params, data.dtrain, best_iter)
    scores, predict_s = model.predict(booster, data.test.X)
    return {**evaluate(data.test, scores, cfg), 'train_seconds': train_s, 'predict_seconds': predict_s,
            'seconds_per_tree': train_s / max(trees_built, 1)}


def run_seeds(cfg, data):
    path = cfg.results_path / 'runs_top.csv'
    done = _done_keys(path, ['search', 'rank', 'seed'])
    num_threads = threads(cfg)
    for search in SEARCHES:
        for conf in top_configs(cfg, search, cfg.top_k):
            for seed in cfg.seeds:
                if (search, conf['rank'], seed) in done:
                    continue
                result = _run_one(cfg, data, conf['hp'], conf['r'], conf['best_iter'], seed, num_threads)
                append_row(path, {'search': search, 'rank': conf['rank'], 'trial': conf['trial'], 'seed': seed,
                                  'r': conf['r'], 'best_iter': conf['best_iter'],
                                  'hp': json.dumps(conf['hp']), **result})
                log.info(f"[seeds] {search} #{conf['rank']} seed {seed}: band profit {result['profit_band']:,.0f}, "
                         f"AUC {result['auc']:.5f}, {result['train_seconds']:,.1f} s")


def run_curve(cfg, data):
    path = cfg.results_path / 'runs_curve.csv'
    done = _done_keys(path, ['r', 'seed'])
    best = top_configs(cfg, 'baseline', 1)[0]
    num_threads = threads(cfg)
    # r = 1 is exactly the best baseline configuration: reuse its stage-2 runs when the seeds match.
    reusable = read_rows(cfg.results_path / 'runs_top.csv')
    if not reusable.empty:
        reusable = reusable[(reusable['search'] == 'baseline') & (reusable['rank'] == 1)].set_index('seed')
    for r in sorted(cfg.r_grid, reverse=True):
        for seed in cfg.seeds:
            if (float(r), seed) in done:
                continue
            if r == 1.0 and not reusable.empty and seed in reusable.index:
                prev = reusable.loc[seed]
                result = {k: prev[k] for k in ['auc', 'auc_baja2', 'profit_peak', 'profit_peak_n', 'profit_band',
                                               'train_seconds', 'predict_seconds', 'seconds_per_tree']}
            else:
                result = _run_one(cfg, data, best['hp'], r, best['best_iter'], seed, num_threads)
            append_row(path, {'r': float(r), 'seed': seed, 'trial': best['trial'], 'best_iter': best['best_iter'],
                              **result})
            log.info(f"[curve] r={r} seed {seed}: band profit {result['profit_band']:,.0f}, "
                     f"{result['train_seconds']:,.1f} s")
