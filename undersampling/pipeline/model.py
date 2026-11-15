"""LightGBM parameters for the baseline and for undersampling, and timed training/prediction.

How r is applied (cfg.r_apply, see README section 4):
- 'per_tree' (default): every tree draws its own sample of CONTINUA, inside LightGBM
  (pos_bagging_fraction / neg_bagging_fraction). Across the trees, the model sees almost all CONTINUA.
- 'before': one fixed random sample of r of the CONTINUA (plus every positive) is taken before training;
  LightGBM then trains on it as the baseline does, with its usual bagging.

Bagging switch (cfg.bagging_baseline):
- on:  baseline -> bagging_fraction (all rows, every class alike)
       per_tree -> pos_bagging_fraction = bagging_fraction, neg_bagging_fraction = bagging_fraction * r
       before   -> bagging_fraction on the sample
- off: no bagging; per_tree -> neg_bagging_fraction = r; before -> the sample, no bagging
Either way, each tree ends up with the same expected share of each class; r = 1 is exactly the baseline.
"""
import time

import lightgbm as lgb
import numpy as np


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
    if r < 1.0 and cfg.r_apply == 'per_tree':
        params.update(pos_bagging_fraction=bagging, neg_bagging_fraction=bagging * r,
                      bagging_freq=cfg.bagging_freq)
    elif bagging < 1.0:
        params.update(bagging_fraction=bagging, bagging_freq=cfg.bagging_freq)
    return params


def training_set(data, cfg, r: float, seed: int):
    """The Dataset a training uses. For r_apply='before', a fixed sample: every positive and r of the CONTINUA.

    The sample reuses the bins of the full training Dataset (no re-binning), and it is drawn with the
    training's own seed, so each seed also changes which CONTINUA are kept.
    """
    if r >= 1.0 or cfg.r_apply == 'per_tree':
        return data.dtrain
    y = data.train.y
    neg = np.flatnonzero(y == 0)
    keep = np.random.default_rng(seed).choice(neg, size=max(int(round(r * len(neg))), 1), replace=False)
    rows = np.sort(np.concatenate([np.flatnonzero(y == 1), keep])).astype(np.int32)
    return data.dtrain.subset(rows).construct()


def fit(data, cfg, hp, r, seed, num_threads, num_rounds, early_stopping=False):
    """Builds the training set and trains. Returns (booster, seconds, best_iteration, trees_built, rows).

    seconds covers drawing the sample ('before') plus training, so both ways of applying r pay their own cost.
    With early_stopping, the validation month stops training after cfg.early_stopping_rounds without
    improvement in AUC.
    """
    start = time.perf_counter()
    dtrain = training_set(data, cfg, r, seed)
    params = lgb_params(hp, cfg, seed=seed, num_threads=num_threads, r=r)
    built = {'n': 0}

    def count_rounds(env):
        built['n'] = env.iteration - env.begin_iteration + 1

    callbacks, valid_sets = [count_rounds], []
    if early_stopping:
        valid_sets = [data.dvalid]
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=valid_sets,
                        valid_names=['valid'] if valid_sets else None, callbacks=callbacks)
    seconds = time.perf_counter() - start
    # trees_built counts every round actually run, including the ones early stopping waits through
    # (afterwards LightGBM keeps only best_iteration trees, so current_iteration() would undercount).
    best_iter = booster.best_iteration if early_stopping and booster.best_iteration > 0 else num_rounds
    return booster, seconds, best_iter, built['n'], dtrain.num_data()


def predict(booster, X, num_iteration=None):
    start = time.perf_counter()
    scores = booster.predict(X, num_iteration=num_iteration)
    return scores, time.perf_counter() - start
