"""Stage 1: two Optuna searches with the same number of trials.

- 'baseline': the baseline's hyperparameters.
- 'undersampling': the same plus r (share of CONTINUA each tree uses).

Each trial trains on the training months with early stopping on validation AUC, and is scored by
validation AUC. Validation profit and times are recorded per trial but never used to choose.
Studies live in a SQLite file, so an interrupted search resumes where it stopped.
"""
import time

import optuna

from . import model
from .metrics import evaluate
from .utils import log, read_rows, threads

SEARCHES = ['baseline', 'undersampling']


def storage_url(cfg):
    return f"sqlite:///{(cfg.results_path / 'optuna.db').as_posix()}"


def suggest(trial, cfg, search):
    hp = {}
    for name, (low, high, kind, log_scale) in cfg.search_space.items():
        if name == 'bagging_fraction' and not cfg.bagging_baseline:
            continue
        if kind == 'int':
            hp[name] = trial.suggest_int(name, int(low), int(high), log=log_scale)
        else:
            hp[name] = trial.suggest_float(name, low, high, log=log_scale)
    r = 1.0
    if search == 'undersampling':
        r = trial.suggest_float('r', cfg.r_range[0], cfg.r_range[1], log=True)
    return hp, r


def make_objective(cfg, data, search, num_threads):
    def objective(trial):
        start = time.perf_counter()
        hp, r = suggest(trial, cfg, search)
        params = model.lgb_params(hp, cfg, seed=cfg.train_seed, num_threads=num_threads, r=r)
        booster, train_s, best_iter, trees_built = model.train(params, data.dtrain, cfg.max_rounds,
                                                               data.dvalid, cfg.early_stopping_rounds)
        scores, predict_s = model.predict(booster, data.valid.X, num_iteration=best_iter)
        metrics = evaluate(data.valid, scores, cfg)
        trial_s = time.perf_counter() - start
        for key, value in {'best_iter': best_iter, 'trees_built': trees_built,
                           'seconds_per_tree': train_s / max(trees_built, 1),
                           'train_seconds': train_s, 'predict_seconds': predict_s,
                           'trial_seconds': trial_s, 'r': r, **{f'valid_{k}': v for k, v in metrics.items()}}.items():
            trial.set_user_attr(key, value)
        log.info(f"[{search}] trial {trial.number}: valid AUC {metrics['auc']:.5f}, r={r:.3f}, "
                 f"best {best_iter} of {trees_built} trees, {trial_s:,.1f} s")
        return metrics['auc']
    return objective


def run_search(cfg, data, search):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=search, storage=storage_url(cfg), direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=cfg.optuna_seed), load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = cfg.n_trials - done
    if remaining <= 0:
        log.info(f'[{search}] already has {done} trials, nothing to do')
    else:
        log.info(f'[{search}] running {remaining} trials ({done} already done)')
        study.optimize(make_objective(cfg, data, search, threads(cfg)), n_trials=remaining)
    export_trials(cfg, study, search)
    return study


def export_trials(cfg, study, search):
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({'search': search, 'trial': t.number, 'valid_auc_score': t.value, **t.params, **t.user_attrs})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_path / f'trials_{search}.csv', index=False)
    return df


def top_configs(cfg, search, k):
    """The k best completed trials of a search, by validation AUC."""
    df = read_rows(cfg.results_path / f'trials_{search}.csv')
    if df.empty:
        raise RuntimeError(f"No trials for '{search}'. Run the 'search' stage first.")
    df = df.sort_values(['valid_auc_score', 'trial'], ascending=[False, True]).head(k)
    configs = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        hp = {name: row[name] for name in cfg.search_space if name in row and row[name] == row[name]}
        configs.append({'search': search, 'rank': rank, 'trial': int(row['trial']), 'hp': hp,
                        'r': float(row.get('r', 1.0)), 'best_iter': int(row['best_iter'])})
    return configs
