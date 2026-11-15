"""Experiment configuration: README defaults, overridable by a JSON file and by command-line flags.

Order of precedence: defaults < --config file.json < command-line flags.
Relative paths are resolved against this folder, so the code runs the same from any working directory.
"""
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

ALL_STAGES = ['features', 'search', 'validate', 'test', 'final', 'analyze']
R_MODES = ['grid', 'optuna']
R_APPLY = ['per_tree', 'before']

# Fields that change what the shared results (the Optuna studies and the trainings) mean. Every
# experiment in one results folder must agree on them; see Config.shared_signature.
SHARED_FIELDS = ['data_file', 'sample_clients', 'train_months', 'valid_month', 'test_month', 'positive_classes',
                 'profit_hit', 'profit_cost', 'band', 'bagging_baseline', 'bagging_freq', 'n_trials',
                 'master_seed', 'search_space', 'max_rounds', 'early_stopping_rounds']


@dataclass
class Config:
    # Data
    data_file: str = 'data/competencia_02_crudo.csv.gz'
    features_dir: str = 'data/features'
    results_dir: str = 'results'
    name: str = ''                      # results folder name; empty = 'full' or 'sample010'
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
    r_mode: str = 'grid'                # 'grid': one Optuna search per r of r_list; 'optuna': Optuna picks r from r_list
    r_apply: str = 'per_tree'           # 'per_tree': each tree draws its own CONTINUA; 'before': one fixed sample
    r_list: list = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07,
                                                  0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8])  # r = 1 (baseline) is implicit
    bagging_baseline: bool = True       # baseline tunes bagging_fraction; undersampling = bagging_fraction x r on CONTINUA
    bagging_freq: int = 1

    # Optuna
    n_trials: int = 100                 # per search
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

    # Seeds: every seed of the experiment is derived from this one number (see README section 5, "Seeds")
    master_seed: int = 6389

    # Choosing r (validation) and confirming it (test)
    n_seeds_valid: int = 10             # seeds per candidate on validation (the first n_seeds_valid of the test seeds)
    n_seeds: int = 20                   # seeds per configuration on test
    top_k: int = 3                      # configurations of the winner and of the baseline compared on test
    delta: float = 0.01                 # "about the same": loses at most 1%
    time_reduction: float = 0.30        # "clearly less time": at least 30% less
    choice_confidence: float = 0.95     # interval level when judging each r on validation (0.997 = Bonferroni for 15 r)
    near_fastest: float = 0.05          # r values this close to the fastest (share of baseline time) count as tied
    flat_share: float = 0.05            # time "flattens" when going lower saves less than this share of baseline time
    bootstrap_samples: int = 10_000

    # Machine
    threads: int = 0                    # 0 = all cores
    duckdb_memory_limit: str = '8GB'

    stages: list = field(default_factory=lambda: list(ALL_STAGES))

    # ---- derived values ----
    @property
    def seeds(self):
        """Derived seeds: Optuna's, the one used inside search trials, and the final models' (n_seeds)."""
        optuna_ss, search_ss, final_ss = np.random.SeedSequence(self.master_seed).spawn(3)
        to_int = lambda ss, n: [int(x % (2**31 - 1)) for x in ss.generate_state(n)]
        return {'optuna': to_int(optuna_ss, 1)[0], 'search': to_int(search_ss, 1)[0],
                'final': to_int(final_ss, self.n_seeds)}

    @property
    def valid_seeds(self):
        return self.seeds['final'][:self.n_seeds_valid]

    @property
    def test_seeds(self):
        return self.seeds['final']

    @property
    def base_name(self):
        if self.name:
            return self.name
        return 'full' if self.sample_clients >= 1 else f'sample{round(self.sample_clients * 100):03d}'

    @property
    def experiment_name(self):
        return f'{self.r_mode}_{self.r_apply}'

    def path(self, value):
        p = Path(value)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def base_path(self):
        """Shared by every experiment on the same data: Optuna studies, trials and trainings."""
        return self.path(self.results_dir) / self.base_name

    @property
    def results_path(self):
        """This experiment's own outputs (mode and r_apply): log, times, summary."""
        return self.base_path / self.experiment_name

    @property
    def r_values(self):
        """The undersampling r values, largest first (r = 1 is the baseline and is not included)."""
        return sorted({float(r) for r in self.r_list if r < 1}, reverse=True)

    def shared_signature(self):
        return {k: getattr(self, k) for k in SHARED_FIELDS}

    def to_dict(self):
        return {**asdict(self), 'derived_seeds': self.seeds}


def build_parser():
    p = argparse.ArgumentParser(description='Undersampling experiment (see README.md).')
    p.add_argument('--config', help='JSON file with any Config fields to override')
    p.add_argument('--stages', nargs='+', choices=ALL_STAGES, help='stages to run (default: all)')
    p.add_argument('--name', help="results folder name (default 'full' or 'sample010')")
    p.add_argument('--data-file')
    p.add_argument('--sample-clients', type=float, help='share of clients to keep, e.g. 0.1 for local tests')
    p.add_argument('--train-months', nargs='+', type=int)
    p.add_argument('--valid-month', type=int)
    p.add_argument('--test-month', type=int)
    p.add_argument('--r-mode', choices=R_MODES, help="'grid' (one search per r, default) or 'optuna' (r inside the search)")
    p.add_argument('--r-apply', choices=R_APPLY, help="'per_tree' (default) or 'before' (one fixed sample of CONTINUA)")
    p.add_argument('--r-list', nargs='+', type=float, help='r values to try (r = 1, the baseline, is always added)')
    p.add_argument('--bagging-baseline', dest='bagging_baseline', action='store_true', default=None)
    p.add_argument('--no-bagging-baseline', dest='bagging_baseline', action='store_false')
    p.add_argument('--bagging-freq', type=int)
    p.add_argument('--n-trials', type=int)
    p.add_argument('--master-seed', type=int)
    p.add_argument('--n-seeds-valid', type=int)
    p.add_argument('--n-seeds', type=int)
    p.add_argument('--top-k', type=int)
    p.add_argument('--delta', type=float)
    p.add_argument('--time-reduction', type=float)
    p.add_argument('--choice-confidence', type=float, help='interval level when choosing r (default 0.95)')
    p.add_argument('--near-fastest', type=float)
    p.add_argument('--flat-share', type=float)
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
    if cfg.r_mode not in R_MODES or cfg.r_apply not in R_APPLY:
        raise ValueError(f'r_mode must be one of {R_MODES} and r_apply one of {R_APPLY}')
    if any(not 0 < r <= 1 for r in cfg.r_list) or not cfg.r_values:
        raise ValueError('r_list needs at least one value in (0, 1)')
    if cfg.top_k < 1 or cfg.n_seeds_valid < 2 or cfg.n_seeds < cfg.n_seeds_valid:
        raise ValueError('top_k must be >= 1, n_seeds_valid >= 2 and n_seeds >= n_seeds_valid')


def _month_index(yyyymm):
    return (yyyymm // 100) * 12 + yyyymm % 100
