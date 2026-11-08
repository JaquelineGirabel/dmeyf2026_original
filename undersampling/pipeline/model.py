"""LightGBM parameters for the baseline and for undersampling, and timed training/prediction.

Bagging switch (cfg.bagging_baseline), see README section 3:
- on:  baseline -> bagging_fraction (all rows, every class alike)
       undersampling -> pos_bagging_fraction = bagging_fraction, neg_bagging_fraction = bagging_fraction * r
- off: baseline -> no row sampling
       undersampling -> pos_bagging_fraction = 1, neg_bagging_fraction = r
r = 1 always gives exactly the baseline.
"""
import time

import lightgbm as lgb

TUNED = ['num_leaves', 'learning_rate', 'min_data_in_leaf', 'feature_fraction', 'bagging_fraction']


def lgb_params(hp: dict, cfg, seed: int, num_threads: int, r: float = 1.0):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'num_threads': num_threads,
        'seed': seed,                      # also seeds the bagging draws and feature_fraction
        'deterministic': True,
        'force_col_wise': True,
        'num_leaves': int(hp['num_leaves']),
        'learning_rate': hp['learning_rate'],
        'min_data_in_leaf': int(hp['min_data_in_leaf']),
        'feature_fraction': hp['feature_fraction'],
    }
    bagging = hp.get('bagging_fraction', 1.0) if cfg.bagging_baseline else 1.0
    if r >= 1.0:
        if bagging < 1.0:
            params.update(bagging_fraction=bagging, bagging_freq=cfg.bagging_freq)
    else:
        params.update(pos_bagging_fraction=bagging, neg_bagging_fraction=bagging * r,
                      bagging_freq=cfg.bagging_freq)
    return params


def train(params, dtrain, num_rounds, dvalid=None, early_stopping_rounds=None):
    """Trains and returns (booster, seconds, best_iteration, trees_built).

    trees_built counts every boosting round actually run, including the rounds after the best one that
    early stopping waits through. (After early stopping LightGBM keeps only best_iteration trees, so
    booster.current_iteration() would undercount the work done.)
    """
    built = {'n': 0}

    def count_rounds(env):
        built['n'] = env.iteration - env.begin_iteration + 1

    callbacks, valid_sets = [count_rounds], []
    if dvalid is not None:
        valid_sets = [dvalid]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    start = time.perf_counter()
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=valid_sets,
                        valid_names=['valid'] if valid_sets else None, callbacks=callbacks)
    seconds = time.perf_counter() - start
    best_iter = booster.best_iteration if dvalid is not None and booster.best_iteration > 0 else num_rounds
    return booster, seconds, best_iter, built['n']


def predict(booster, X, num_iteration=None):
    start = time.perf_counter()
    scores = booster.predict(X, num_iteration=num_iteration)
    return scores, time.perf_counter() - start
