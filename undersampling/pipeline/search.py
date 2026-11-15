"""Stage 'search': the Optuna searches.

Studies (all in results/<base>/optuna.db, shared by every experiment on the same data):
- 'baseline': r = 1, the baseline's hyperparameters. Run once, reused by both modes.
- grid mode:   'grid_<r_apply>_r<r>', one per r of r_list, r fixed during the search.
- optuna mode: 'optuna_<r_apply>', Optuna picks r from r_list together with the other hyperparameters.
  Its first len(r_list) trials are forced to try each r once (the other hyperparameters still come from
  Optuna), so every r has at least one trial; after that Optuna chooses freely.

Every search uses the same number of trials and the same Optuna seed, so their first random trials propose
the same hyperparameters. Each trial trains on the training months with early stopping on validation AUC
and is scored by validation AUC. Validation profit and times are recorded per trial but never used to choose.
"""
import time
import warnings

import optuna
import pandas as pd

from . import model
from .metrics import evaluate
from .utils import log, read_rows, threads

BASELINE = 'baseline'

# The heartbeat features are stable in practice but still labelled "experimental" by Optuna.
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


def storage_url(cfg):
    return f"sqlite:///{(cfg.base_path / 'optuna.db').as_posix()}"


def storage(cfg):
    """The SQLite database, with a heartbeat so a trial cut off by a crash isn't left 'running' forever.

    Each running trial writes a heartbeat every 60 s. On the next run, a trial whose heartbeat stopped more
    than 3 minutes ago is marked failed and queued again with the same hyperparameters (and the same r), so
    the forced one-trial-per-r of optuna mode survives a crash too.
    """
    if hasattr(optuna.storages, 'RetryHeartbeatStaleTrialCallback'):      # Optuna >= 4.9
        retry = {'heartbeat_stale_trial_callback': optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=3)}
    else:
        retry = {'failed_trial_callback': optuna.storages.RetryFailedTrialCallback(max_retry=3)}
    return optuna.storages.RDBStorage(storage_url(cfg), heartbeat_interval=60, grace_period=180, **retry)


def grid_study(cfg, r):
    return f'grid_{cfg.r_apply}_r{r:g}'


def optuna_study(cfg):
    return f'optuna_{cfg.r_apply}'


def studies(cfg):
    """[(study name, fixed r or None)] this experiment needs, baseline first."""
    if cfg.r_mode == 'grid':
        return [(BASELINE, 1.0)] + [(grid_study(cfg, r), r) for r in cfg.r_values]
    return [(BASELINE, 1.0), (optuna_study(cfg), None)]


def trials_path(cfg, study):
    return cfg.base_path / 'trials' / f'{study}.csv'


def suggest(trial, cfg, fixed_r):
    hp = {}
    for name, (low, high, kind, log_scale) in cfg.search_space.items():
        if name == 'bagging_fraction' and not cfg.bagging_baseline:
            continue
        if kind == 'int':
            hp[name] = trial.suggest_int(name, int(low), int(high), log=log_scale)
        else:
            hp[name] = trial.suggest_float(name, low, high, log=log_scale)
    r = fixed_r if fixed_r is not None else trial.suggest_categorical('r', cfg.r_values)
    return hp, float(r)


def make_objective(cfg, data, study_name, fixed_r, num_threads):
    def objective(trial):
        start = time.perf_counter()
        hp, r = suggest(trial, cfg, fixed_r)
        booster, train_s, best_iter, trees_built, rows = model.fit(
            data, cfg, hp, r, cfg.seeds['search'], num_threads, cfg.max_rounds, early_stopping=True)
        scores, predict_s = model.predict(booster, data.valid.X, num_iteration=best_iter)
        metrics = evaluate(data.valid, scores, cfg)
        trial_s = time.perf_counter() - start
        for key, value in {'r': r, 'best_iter': best_iter, 'trees_built': trees_built, 'train_rows': rows,
                           'seconds_per_tree': train_s / max(trees_built, 1),
                           'train_seconds': train_s, 'predict_seconds': predict_s, 'trial_seconds': trial_s,
                           **{f'valid_{k}': v for k, v in metrics.items()}}.items():
            trial.set_user_attr(key, value)
        log.info(f"[{study_name}] trial {trial.number}: valid AUC {metrics['auc']:.5f}, r={r:g}, "
                 f"best {best_iter} of {trees_built} trees, {trial_s:,.1f} s")
        return metrics['auc']
    return objective


def run_search(cfg, data, study_name, fixed_r):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name, storage=storage(cfg), direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=cfg.seeds['optuna']), load_if_exists=True)
    optuna.storages.fail_stale_trials(study)       # trials left 'running' by a crash -> failed and queued again
    if fixed_r is None:
        _prepare_optuna_study(cfg, study)
    interrupted = [t.number for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    if interrupted:
        log.info(f'[{study_name}] trials cut off by a crash so far: {interrupted} (each one is rerun with the same values)')
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = cfg.n_trials - done
    if remaining <= 0:
        log.info(f'[{study_name}] already has {done} trials, nothing to do')
    else:
        log.info(f'[{study_name}] running {remaining} trials ({done} already done)')
        study.optimize(make_objective(cfg, data, study_name, fixed_r, threads(cfg)), n_trials=remaining)
    export_trials(cfg, study, study_name)
    return study


def _prepare_optuna_study(cfg, study):
    """First time: remember the r list and queue one trial per r. Later: check the r list didn't change."""
    r_values = list(cfg.r_values)
    if 'r_values' in study.user_attrs:
        if study.user_attrs['r_values'] != r_values:
            raise ValueError(f"Study '{study.study_name}' was created with r values {study.user_attrs['r_values']}; "
                             f"use the same --r-list or another --name.")
        return
    if cfg.n_trials < len(r_values):
        raise ValueError(f'n_trials ({cfg.n_trials}) must be at least the number of r values ({len(r_values)})')
    study.set_user_attr('r_values', r_values)
    for r in r_values:
        study.enqueue_trial({'r': r})


def export_trials(cfg, study, study_name):
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        params = {k: v for k, v in t.params.items() if k != 'r'}
        rows.append({'study': study_name, 'trial': t.number, 'valid_auc_score': t.value, **params, **t.user_attrs})
    df = pd.DataFrame(rows)
    path = trials_path(cfg, study_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def read_trials(cfg, study):
    df = read_rows(trials_path(cfg, study))
    if df.empty:
        raise RuntimeError(f"No trials for '{study}'. Run the 'search' stage first.")
    return df


def search_seconds(cfg, study):
    return float(read_trials(cfg, study)['trial_seconds'].sum())


def top_configs(cfg, study, k, r=None):
    """The k best completed trials of a study by validation AUC (only trials with that r, if given)."""
    df = read_trials(cfg, study)
    if r is not None:
        df = df[(df['r'] - r).abs() < 1e-12]
    df = df.sort_values(['valid_auc_score', 'trial'], ascending=[False, True]).head(k)
    configs = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        hp = {name: row[name] for name in cfg.search_space if name in row and row[name] == row[name]}
        configs.append({'study': study, 'rank': rank, 'trial': int(row['trial']), 'hp': hp,
                        'r': float(row['r']), 'best_iter': int(row['best_iter'])})
    return configs
