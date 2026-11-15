"""Monitoring metrics for the dashboard's "Monitoreo" tab (see pipeline/monitoring.py and README).

Takes the same flags as main.py, so it reads the same features file and the same results folder:

    python monitor.py --sample-clients 0.1       # results/sample010/monitoring/monitoring.json
    python monitor.py                            # results/full/monitoring/monitoring.json

It trains one model with the baseline's best configuration (and one per experiment winner), so run it
after the search stage, and not while another run is being timed on the same machine.
"""
import sys

from config import load_config
from pipeline.data import load_data
from pipeline.features import build_features
from pipeline.monitoring import monitoring_path, run_monitoring
from pipeline.utils import setup_logging, threads


def main(argv=None):
    cfg = load_config(argv)
    setup_logging(monitoring_path(cfg).parent)
    parquet_path, _ = build_features(cfg, {})
    data = load_data(cfg, parquet_path, threads(cfg))
    run_monitoring(cfg, parquet_path, data)


if __name__ == '__main__':
    sys.exit(main())
