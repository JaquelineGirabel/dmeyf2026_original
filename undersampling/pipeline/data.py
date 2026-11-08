"""Loading the features into memory and building LightGBM's datasets once.

The training Dataset is binned a single time and reused by every Optuna trial and every seed run.
Sampling (bagging / undersampling) happens inside LightGBM, tree by tree, so the full training data
stays in memory and is never copied per trial.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from .utils import log

META_COLS = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']


@dataclass
class Split:
    X: np.ndarray            # float32, column-major (what LightGBM reads without copying)
    y: np.ndarray            # training target: 1 = positive classes (BAJA+1 + BAJA+2 by default)
    baja2: np.ndarray        # 1 = BAJA+2 (what profit counts)
    months: list

    @property
    def n(self):
        return len(self.y)


@dataclass
class Data:
    train: Split
    valid: Split
    test: Split
    feature_names: list
    dtrain: lgb.Dataset = None
    dvalid: lgb.Dataset = None


def _read_split(parquet_path, months, feature_names, positive_classes):
    table = pq.read_table(parquet_path, filters=[('foto_mes', 'in', months)])
    n = table.num_rows
    if n == 0:
        raise ValueError(f'No rows for months {months} in {parquet_path.name}')
    clase = table.column('clase_ternaria').to_numpy(zero_copy_only=False)
    if any(v is None for v in clase):
        raise ValueError(f'Months {months} have rows without class: they cannot be used')
    X = np.empty((n, len(feature_names)), dtype=np.float32, order='F')
    for j, name in enumerate(feature_names):
        X[:, j] = table.column(name).to_numpy(zero_copy_only=False)
    del table
    y = np.isin(clase, positive_classes).astype(np.int8)
    baja2 = (clase == 'BAJA+2').astype(np.int8)
    return Split(X=X, y=y, baja2=baja2, months=list(months))


def load_data(cfg, parquet_path, num_threads):
    schema = pq.read_schema(parquet_path)
    feature_names = [c for c in schema.names if c not in META_COLS]
    train = _read_split(parquet_path, cfg.train_months, feature_names, cfg.positive_classes)
    valid = _read_split(parquet_path, [cfg.valid_month], feature_names, cfg.positive_classes)
    test = _read_split(parquet_path, [cfg.test_month], feature_names, cfg.positive_classes)
    for label, s in [('train', train), ('valid', valid), ('test', test)]:
        log.info(f'{label}: {s.n:,} rows, {s.y.sum():,} positives ({s.y.mean():.2%}), {s.baja2.sum():,} BAJA+2')

    ds_params = {'max_bin': 255, 'feature_pre_filter': False, 'verbosity': -1, 'num_threads': num_threads}
    dtrain = lgb.Dataset(train.X, label=train.y, feature_name=feature_names, params=ds_params,
                         free_raw_data=True).construct()
    dvalid = lgb.Dataset(valid.X, label=valid.y, reference=dtrain, params=ds_params,
                         free_raw_data=False).construct()
    # The binned Dataset has everything training needs; the raw training matrix can go.
    train.X = None
    return Data(train=train, valid=valid, test=test, feature_names=feature_names, dtrain=dtrain, dvalid=dvalid)
