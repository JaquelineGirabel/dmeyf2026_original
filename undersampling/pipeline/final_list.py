"""Stage 'final': the list of clients to contact in the test month, for the chosen model and for the baseline.

The list uses the same rule as the profit measure of the experiment (pipeline/analysis.py):
- The model is a configuration (its best one), and its seeds give its distribution. Its scores on the test
  month are averaged over its n_seeds seeds (the mean model of that configuration).
- It contacts its share of clients: the share where its validation profit curve, averaged over its
  validation seeds, is highest. Nothing about the test month is used to build the list.
The list (client numbers) is written to final_list_<side>.csv, and its profit, counted with the real
classes of the test month, to final_list.json. The verdict is still decided per seed (analysis.py).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import model
from .analysis import list_share
from .features import features_paths
from .metrics import profit_curve
from .runs import winner_configs
from .search import BASELINE, top_configs
from .utils import log, threads, write_json


def scores_path(cfg, conf, seed):
    from .runs import scores_path as _path
    return _path(cfg, conf['study'], conf['trial'], seed)


def mean_test_scores(cfg, data, conf, seeds):
    """Test-month scores of one configuration averaged over its seeds; trains and saves the ones not saved yet."""
    total, num_threads = np.zeros(data.test.n), threads(cfg)
    for seed in seeds:
        path = scores_path(cfg, conf, seed)
        if path.exists():
            stt = np.load(path)['test']
        else:
            booster, *_ = model.fit(data, cfg, conf['hp'], conf['r'], seed, num_threads, conf['best_iter'])
            sv, _ = model.predict(booster, data.valid.X)
            stt, _ = model.predict(booster, data.test.X)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, valid=sv.astype(np.float32), test=stt.astype(np.float32))
            log.info(f"[final] {conf['study']} trial {conf['trial']} seed {seed}: scores saved")
        total += stt
    return total / len(seeds)


def run_final(cfg, data):
    from .analysis import choose_r, tolerances     # validation only
    from .runs import runs_path
    from .utils import read_rows
    runs = read_rows(runs_path(cfg))
    choice = choose_r(cfg, runs, tolerances(cfg, runs))
    winner = choice['winner']
    sides = [('baseline', top_configs(cfg, BASELINE, 1)[0])]
    if winner is not None:
        sides.append(('winner', winner_configs(cfg, winner, 1)[0]))
    ids = pq.read_table(features_paths(cfg)[0], columns=['numero_de_cliente'],
                        filters=[('foto_mes', 'in', [cfg.test_month])]).column('numero_de_cliente').to_numpy()
    gains = np.where(data.test.baja2 == 1, cfg.profit_hit, -cfg.profit_cost)
    out = {'rule': "best configuration; test scores averaged over its seeds; contact its share of clients (where its "
                   "validation profit curve, averaged over its validation seeds, is highest)",
           'test_month': cfg.test_month, 'n_seeds': cfg.n_seeds, 'n_seeds_share': cfg.n_seeds_valid, 'sides': {}}
    for side, conf in sides:
        _, share = list_share(cfg, conf)
        scores = mean_test_scores(cfg, data, conf, cfg.test_seeds)
        order = np.argsort(-scores, kind='stable')
        n = max(int(round(share * data.test.n)), 1)
        chosen = order[:n]
        curve = profit_curve(data.test.baja2, scores, cfg.profit_hit, cfg.profit_cost)
        best = int(np.argmax(curve))
        pd.DataFrame({'numero_de_cliente': ids[chosen]}).to_csv(cfg.results_path / f'final_list_{side}.csv',
                                                                index=False)
        out['sides'][side] = {'study': conf['study'], 'trial': conf['trial'], 'r': conf['r'], 'share': share,
                              'n_contacted': n, 'baja2_contacted': int(data.test.baja2[chosen].sum()),
                              'baja2_total': int(data.test.baja2.sum()), 'profit': float(gains[chosen].sum()),
                              'best_share': (best + 1) / data.test.n, 'best_profit': float(curve[best])}
        log.info(f"[final] {side} (r={conf['r']:g}): contacts {n:,} ({share:.1%}), profit {gains[chosen].sum():,.0f} "
                 f"(best possible {curve[best]:,.0f} contacting {(best + 1) / data.test.n:.1%})")
    write_json(cfg.results_path / 'final_list.json', out)
    log.info(f"[final] written to {cfg.results_path / 'final_list.json'}")
    return out
