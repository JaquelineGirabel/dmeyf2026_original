"""Runs the undersampling experiment end to end (see README.md).

Examples:
    python main.py                                   # full experiment, README defaults
    python main.py --sample-clients 0.1              # local test dataset (10% of clients)
    python main.py --sample-clients 0.1 --n-trials 5 --n-seeds 3 --top-k 2 --r-grid 1 0.5 0.1
    python main.py --stages analyze                  # recompute the summary from saved results

Every stage saves its results as it goes; rerunning the same command resumes where it stopped.
Results go to results/<name>/ (name: 'full', 'sample010', or --name).
"""
import sys
import time

from config import load_config
from pipeline.analysis import analyze
from pipeline.data import load_data
from pipeline.features import build_features
from pipeline.runs import run_curve, run_seeds
from pipeline.search import SEARCHES, run_search
from pipeline.utils import (environment_info, log, peak_memory_gb, read_json, setup_logging, threads, timed,
                            update_json, write_json)


def main(argv=None):
    cfg = load_config(argv)
    out = cfg.results_path
    setup_logging(out)
    write_json(out / 'config.json', cfg.to_dict())
    write_json(out / 'environment.json', environment_info())
    timings_path = out / 'timings.json'
    timings = read_json(timings_path, {})
    log.info(f"Experiment '{cfg.experiment_name}', stages {cfg.stages}, {threads(cfg)} threads")
    start = time.perf_counter()

    needs_data = any(s in cfg.stages for s in ['search', 'seeds', 'curve'])
    if 'features' in cfg.stages or needs_data:
        parquet_path, meta = build_features(cfg, timings)
        timings['features_peak_memory_gb'] = meta.get('peak_memory_gb')

    if needs_data:
        with timed('load_seconds', timings):
            data = load_data(cfg, parquet_path, threads(cfg))
        timings['n_features'] = len(data.feature_names)
        if 'search' in cfg.stages:
            for search in SEARCHES:
                run_search(cfg, data, search)
        if 'seeds' in cfg.stages:
            run_seeds(cfg, data)
        if 'curve' in cfg.stages:
            run_curve(cfg, data)

    timings['peak_memory_gb'] = max(peak_memory_gb(), timings.get('peak_memory_gb', 0.0))
    update_json(timings_path, **timings)

    if 'analyze' in cfg.stages:
        analyze(cfg)
    log.info(f'Done in {time.perf_counter() - start:,.0f} s. Peak memory {peak_memory_gb():.2f} GB.')


if __name__ == '__main__':
    sys.exit(main())
