"""Shared helpers: logging, timing, peak memory, environment info and resumable CSV results."""
import json
import logging
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import pandas as pd
import psutil

log = logging.getLogger('undersampling')


def setup_logging(results_path: Path):
    results_path.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(results_path / 'run.log')):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def peak_memory_gb():
    """Peak resident memory of this process so far (DuckDB runs in-process, so it's included)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return peak / 1024**3 if sys.platform == 'darwin' else peak / 1024**2


def current_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


@contextmanager
def timed(label, store=None):
    """Measures wall time of a block; logs it and optionally saves it in store[label]."""
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    if store is not None:
        store[label] = seconds
    log.info(f'{label}: {seconds:,.1f} s (peak memory {peak_memory_gb():.2f} GB)')


def threads(cfg):
    return cfg.threads if cfg.threads > 0 else os.cpu_count()


def environment_info():
    packages = ['duckdb', 'lightgbm', 'optuna', 'numpy', 'pandas', 'pyarrow', 'scipy', 'psutil']
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'ram_gb': round(psutil.virtual_memory().total / 1024**3, 1),
        'packages': versions,
    }


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def update_json(path: Path, **values):
    data = read_json(path, {})
    data.update(values)
    write_json(path, data)


def append_row(path: Path, row: dict):
    """Appends one result row to a CSV right away, so nothing is lost if the run stops.

    The line is written in a single call and forced to disk (fsync), so a crash leaves either the whole
    row or nothing; read_rows also drops any incomplete last line, just in case.
    """
    header = not path.exists() or path.stat().st_size == 0
    text = pd.DataFrame([row]).to_csv(header=header, index=False)
    with open(path, 'a') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: Path):
    """Reads a results CSV. A last line cut short by a crash (missing values) is ignored."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df) and df.iloc[-1].isna().any() and not df.iloc[:-1].isna().any(axis=None):
        df = df.iloc[:-1]
    return df
