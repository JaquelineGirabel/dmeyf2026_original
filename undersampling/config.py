"""Experiment configuration: README defaults, overridable by a JSON file and by command-line flags.

Order of precedence: defaults < --config file.json < command-line flags.
Relative paths are resolved against this folder, so the code runs the same from any working directory.
"""
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ALL_STAGES = ['features', 'search', 'seeds', 'curve', 'analyze']


@dataclass
class Config:
    # Data
    data_file: str = 'data/competencia_02_crudo.csv.gz'
    features_dir: str = 'data/features'
    results_dir: str = 'results'
    name: str = ''                      # experiment name (results subfolder); empty = automatic
    sample_clients: float = 1.0         # share of clients kept (1.0 = full dataset; 0.1 = local test dataset)

    # Months
    train_months: list = field(default_factory=lambda: [201901, 201902, 201903, 201904, 201905, 201906])
    valid_month: int = 201908
    test_month: int = 201910

    # Target and profit
    positive_classes: list = field(default_factory=lambda: ['BAJA+1', 'BAJA+2'])
    profit_hit: float = 1_072_500       # each BAJA+2 contacted
    profit_cost: float = 27_500         # each other client contacted
    band: list = field(default_factory=lambda: [0.05, 0.06, 0.07, 0.08, 0.09])

    # Undersampling
    bagging_baseline: bool = True       # baseline tunes bagging_fraction; undersampling = bagging_fraction x r on CONTINUA
    bagging_freq: int = 1
    r_range: list = field(default_factory=lambda: [0.01, 1.0])
    r_grid: list = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01])

    # Optuna
    n_trials: int = 100
    optuna_seed: int = 12345
    train_seed: int = 17                # LightGBM seed used inside the searches
    search_space: dict = field(default_factory=lambda: {
        # name: [low, high, type, log]
        'num_leaves': [8, 256, 'int', True],
        'learning_rate': [0.01, 0.3, 'float', True],
        'min_data_in_leaf': [20, 5000, 'int', True],
        'feature_fraction': [0.1, 1.0, 'float', False],
        'bagging_fraction': [0.2, 1.0, 'float', False],
    })
    max_rounds: int = 3000
    early_stopping_rounds: int = 100

    # Comparison
    top_k: int = 3
    n_seeds: int = 20
    seed_base: int = 100                # seeds are seed_base, seed_base + 1, ...
    delta: float = 0.01                 # "about the same": loses at most 1%
    time_reduction: float = 0.30        # "clearly less time": at least 30% less
    bootstrap_samples: int = 10_000

    # Machine
    threads: int = 0                    # 0 = all cores
    duckdb_memory_limit: str = '8GB'

    stages: list = field(default_factory=lambda: list(ALL_STAGES))

    # ---- derived values ----
    @property
    def seeds(self):
        return [self.seed_base + i for i in range(self.n_seeds)]

    @property
    def experiment_name(self):
        if self.name:
            return self.name
        return 'full' if self.sample_clients >= 1 else f'sample{round(self.sample_clients * 100):03d}'

    def path(self, value):
        p = Path(value)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def results_path(self):
        return self.path(self.results_dir) / self.experiment_name

    def to_dict(self):
        return asdict(self)


def _parse_months(values):
    return [int(v) for v in values]


def build_parser():
    p = argparse.ArgumentParser(description='Undersampling experiment (see README.md).')
    p.add_argument('--config', help='JSON file with any Config fields to override')
    p.add_argument('--stages', nargs='+', choices=ALL_STAGES, help='stages to run (default: all)')
    p.add_argument('--name', help='experiment name (results subfolder)')
    p.add_argument('--data-file')
    p.add_argument('--sample-clients', type=float, help='share of clients to keep, e.g. 0.1 for local tests')
    p.add_argument('--train-months', nargs='+')
    p.add_argument('--valid-month', type=int)
    p.add_argument('--test-month', type=int)
    p.add_argument('--bagging-baseline', dest='bagging_baseline', action='store_true', default=None)
    p.add_argument('--no-bagging-baseline', dest='bagging_baseline', action='store_false')
    p.add_argument('--bagging-freq', type=int)
    p.add_argument('--n-trials', type=int)
    p.add_argument('--top-k', type=int)
    p.add_argument('--n-seeds', type=int)
    p.add_argument('--r-grid', nargs='+', type=float)
    p.add_argument('--r-range', nargs=2, type=float)
    p.add_argument('--delta', type=float)
    p.add_argument('--time-reduction', type=float)
    p.add_argument('--max-rounds', type=int)
    p.add_argument('--early-stopping-rounds', type=int)
    p.add_argument('--threads', type=int)
    p.add_argument('--duckdb-memory-limit')
    return p


def load_config(argv=None):
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config field in {args.config}: '{key}'")
            setattr(cfg, key, value)
    for key, value in vars(args).items():
        if key == 'config' or value is None:
            continue
        if key == 'train_months':
            value = _parse_months(value)
        setattr(cfg, key, value)
    _validate(cfg)
    return cfg


def _validate(cfg):
    if not 0 < cfg.sample_clients <= 1:
        raise ValueError('sample_clients must be in (0, 1]')
    last_train = max(cfg.train_months)
    for label, month in [('valid_month', cfg.valid_month), ('test_month', cfg.test_month)]:
        if _month_index(month) - _month_index(last_train) < 2:
            raise ValueError(f'{label} {month} breaks the 2-month gap after the last training month {last_train}')
    if any(not 0 < r <= 1 for r in cfg.r_grid) or 1.0 not in cfg.r_grid:
        raise ValueError('r_grid values must be in (0, 1] and include 1.0 (the baseline point)')
    if cfg.top_k < 1 or cfg.n_seeds < 2:
        raise ValueError('top_k must be >= 1 and n_seeds >= 2')


def _month_index(yyyymm):
    return (yyyymm // 100) * 12 + yyyymm % 100
