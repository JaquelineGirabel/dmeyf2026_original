"""Runs the undersampling experiment end to end (see README.md).

Examples:
    python main.py                                   # full data, grid mode, r per tree (README defaults)
    python main.py --sample-clients 0.1              # local test dataset (10% of clients)
    python main.py --r-mode optuna                   # Optuna picks r from the list
    python main.py --r-apply before                  # r applied once, before training
    python main.py --stages analyze                  # recompute the summary from saved results
    python main.py --stages monitor                  # recompute only the dashboard's monitoring metrics

Results: results/<base>/ holds what every experiment on the same data shares (Optuna studies, trials,
trainings); results/<base>/<mode>_<r_apply>/ holds this experiment's log, times and summary.
<base> is 'full', 'sample010', or --name. Rerunning the same command resumes where it stopped.
"""
import sys
import time

from config import load_config
from pipeline.analysis import analyze
from pipeline.data import load_data
from pipeline.features import build_features
from pipeline.final_list import run_final
from pipeline.monitoring import run_monitoring
from pipeline.runs import run_test, run_validate
from pipeline.search import run_search, studies
from pipeline.utils import (environment_info, log, peak_memory_gb, read_json, setup_logging, threads, timed,
                            update_json, write_json)


def check_shared_config(cfg):
    """Every experiment in one results folder must agree on what the shared studies and trainings mean."""
    path = cfg.base_path / 'shared_config.json'
    current = cfg.shared_signature()
    saved = read_json(path)
    if saved is None:
        write_json(path, current)
        return
    changed = [k for k in current if saved.get(k) != current[k]]
    if changed:
        raise SystemExit(f"{cfg.base_path} was created with different values of {changed}. "
                         f"Use the same values, or another results folder with --name.")


def main(argv=None):
    cfg = load_config(argv)
    out = cfg.results_path
    setup_logging(out)
    check_shared_config(cfg)
    write_json(out / 'config.json', cfg.to_dict())
    write_json(out / 'environment.json', environment_info())
    timings_path = out / 'timings.json'
    timings = read_json(timings_path, {})
    log.info(f"Experiment '{cfg.base_name}/{cfg.experiment_name}', stages {cfg.stages}, {threads(cfg)} threads")
    start = time.perf_counter()

    needs_data = any(s in cfg.stages for s in ['search', 'validate', 'test', 'final', 'monitor'])
    if 'features' in cfg.stages or needs_data:
        parquet_path, meta = build_features(cfg, timings)
        timings['features_peak_memory_gb'] = meta.get('peak_memory_gb')

    if needs_data:
        load = {}
        with timed('load_seconds', load):
            data = load_data(cfg, parquet_path, threads(cfg))
        # The load time is part of the measured total time: keep the one from the run that did the search,
        # so rerunning a later stage (on a busier machine) does not change it.
        if 'search' in cfg.stages or 'load_seconds' not in timings:
            timings['load_seconds'] = load['load_seconds']
        timings['n_features'] = len(data.feature_names)
        if 'search' in cfg.stages:
            for study_name, fixed_r in studies(cfg):
                run_search(cfg, data, study_name, fixed_r)
        if 'validate' in cfg.stages:
            run_validate(cfg, data)
        if 'test' in cfg.stages:
            run_test(cfg, data)
        if 'final' in cfg.stages:
            run_final(cfg, data)

    timings['peak_memory_gb'] = max(peak_memory_gb(), timings.get('peak_memory_gb', 0.0))
    update_json(timings_path, **timings)

    if 'analyze' in cfg.stages:
        analyze(cfg)
    # Last, after every timed piece and after the summary (it trains each experiment's winner, read from it).
    # Not part of the experiment's times or peak memory.
    if 'monitor' in cfg.stages:
        run_monitoring(cfg, parquet_path, data)
    log.info(f'Done in {time.perf_counter() - start:,.0f} s. Peak memory {peak_memory_gb():.2f} GB.')


if __name__ == '__main__':
    sys.exit(main())
