"""Stage 'features': raw CSV -> parquet with target, lags and deltas 1 and 2.

- Target (clase_ternaria) with the same logic as clusters/z101_target_sql.ipynb, computed on all months
  of the file, so the class of the last useful month (e.g. 201910) sees the two months after it.
- Lags are calendar-exact: lag k is the client's value exactly k months earlier (NULL if the client
  was absent that month), via joins on a month index, not "the previous row".
- Text columns holding numeric codes are converted with TRY_CAST (non-numbers become NULL).
- Rows are written sorted by (foto_mes, numero_de_cliente) and feature columns sorted by name, so the
  output never depends on the order of rows or columns in the input file.
- Optional client sample (local test dataset): a client is kept or not by a fixed arithmetic hash
  of numero_de_cliente, so it's the same clients on every machine and DuckDB version.
- Only the months the experiment uses are written (train + validation + test).
"""
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from .utils import log, peak_memory_gb, threads, timed

KEY_COLS = ['numero_de_cliente', 'foto_mes']
HASH_MULT = 2654435761            # Knuth multiplicative hash
HASH_MOD = 2**32


def features_paths(cfg):
    folder = cfg.path(cfg.features_dir)
    stem = 'features_full' if cfg.sample_clients >= 1 else f'features_sample{round(cfg.sample_clients * 100):03d}'
    months = sorted(set(cfg.train_months) | {cfg.valid_month, cfg.test_month})
    tag = f'{months[0]}_{months[-1]}_{len(months)}m'
    return folder / f'{stem}_{tag}.parquet', folder / f'{stem}_{tag}.json'


def _spec(cfg, source: Path):
    """What the features file depends on. If any of this changes, the file is rebuilt."""
    stat = source.stat()
    return {
        'source': source.name,
        'source_bytes': stat.st_size,
        'sample_clients': cfg.sample_clients,
        'months': sorted(set(cfg.train_months) | {cfg.valid_month, cfg.test_month}),
        'version': 3,
    }


def build_features(cfg, timings: dict):
    parquet_path, meta_path = features_paths(cfg)
    source = cfg.path(cfg.data_file)
    spec = _spec(cfg, source)
    if parquet_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('spec') == spec:
            log.info(f'Features already built, reusing {parquet_path.name}')
            timings['features_seconds'] = meta['seconds']
            return parquet_path, meta
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    con = duckdb.connect()
    tmp_dir = parquet_path.parent / 'duckdb_tmp'
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f'SET threads={threads(cfg)}')
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")
    con.execute('SET preserve_insertion_order=true')
    con.execute('SET enable_progress_bar=false')

    columns = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{source.as_posix()}')").fetchall()
    text_cols = [name for name, dtype, *_ in columns if dtype == 'VARCHAR']
    # Sorted by name, so a CSV with the columns in another order gives the same output.
    feature_cols = sorted(name for name, *_ in columns if name not in KEY_COLS)
    if text_cols:
        log.info(f'Text columns converted to numbers: {text_cols}')

    replace = ''
    if text_cols:
        replace = ' REPLACE (' + ', '.join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in text_cols) + ')'
    sample_filter = ''
    if cfg.sample_clients < 1:
        threshold = int(cfg.sample_clients * HASH_MOD)
        sample_filter = (f'WHERE (CAST(numero_de_cliente AS HUGEINT) * {HASH_MULT}) % {HASH_MOD} < {threshold}')

    with timed('features: load raw CSV'):
        con.execute(f"""
            CREATE TABLE crudo AS
            SELECT *{replace}, (foto_mes // 100) * 12 + foto_mes % 100 AS mes_idx
            FROM read_csv_auto('{source.as_posix()}')
            {sample_filter}
        """)

    with timed('features: target'):
        con.execute("""
            CREATE TABLE clase AS
            WITH periodos AS (SELECT DISTINCT foto_mes FROM crudo),
            clientes AS (SELECT DISTINCT numero_de_cliente FROM crudo),
            todo AS (SELECT numero_de_cliente, foto_mes FROM clientes CROSS JOIN periodos),
            marcado AS (
                SELECT t.numero_de_cliente, t.foto_mes,
                       IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0
                FROM todo t
                LEFT JOIN (SELECT numero_de_cliente, foto_mes FROM crudo) c USING (numero_de_cliente, foto_mes)
            ),
            siguientes AS (
                SELECT *,
                       lead(mes_0, 1) OVER w AS mes_1,
                       lead(mes_0, 2) OVER w AS mes_2
                FROM marcado
                WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
            )
            SELECT numero_de_cliente, foto_mes,
                   CASE
                       WHEN mes_2 = 1 THEN 'CONTINUA'
                       WHEN mes_1 = 0 AND (mes_2 = 0 OR mes_2 IS NULL) THEN 'BAJA+1'
                       WHEN mes_1 = 1 AND mes_2 = 0 THEN 'BAJA+2'
                       ELSE NULL
                   END AS clase_ternaria
            FROM siguientes
            WHERE mes_0 = 1
        """)

    months = spec['months']
    select_cols = []
    for c in feature_cols:
        q = f'"{c}"'
        select_cols += [
            f'CAST(c.{q} AS FLOAT) AS {q}',
            f'CAST(l1.{q} AS FLOAT) AS "{c}_lag1"',
            f'CAST(l2.{q} AS FLOAT) AS "{c}_lag2"',
            f'CAST(c.{q} - l1.{q} AS FLOAT) AS "{c}_delta1"',
            f'CAST(c.{q} - l2.{q} AS FLOAT) AS "{c}_delta2"',
        ]
    with timed('features: lags, deltas and parquet'):
        con.execute(f"""
            COPY (
                SELECT c.numero_de_cliente, c.foto_mes, k.clase_ternaria,
                       {', '.join(select_cols)}
                FROM crudo c
                JOIN clase k USING (numero_de_cliente, foto_mes)
                LEFT JOIN crudo l1 ON l1.numero_de_cliente = c.numero_de_cliente AND l1.mes_idx = c.mes_idx - 1
                LEFT JOIN crudo l2 ON l2.numero_de_cliente = c.numero_de_cliente AND l2.mes_idx = c.mes_idx - 2
                WHERE c.foto_mes IN ({', '.join(str(m) for m in months)})
                ORDER BY c.foto_mes, c.numero_de_cliente
            ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """)

    counts = con.execute(f"""
        SELECT foto_mes, clase_ternaria, count(*) FROM read_parquet('{parquet_path.as_posix()}')
        GROUP BY ALL ORDER BY ALL
    """).fetchall()
    con.close()
    seconds = time.perf_counter() - start

    meta = {
        'spec': spec,
        'seconds': seconds,
        'peak_memory_gb': peak_memory_gb(),
        'rows': pq.ParquetFile(parquet_path).metadata.num_rows,
        'n_features': len(feature_cols) * 5,
        'text_columns_converted': text_cols,
        'class_counts': [{'foto_mes': m, 'clase_ternaria': k, 'n': n} for m, k, n in counts],
        'content_sha256': content_hash(parquet_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    timings['features_seconds'] = seconds
    log.info(f"Features: {meta['rows']:,} rows, {meta['n_features']} features, {seconds:,.0f} s -> {parquet_path.name}")
    return parquet_path, meta


def content_hash(parquet_path: Path):
    """Hash of the data itself (not of the file bytes, which include library versions).
    Two machines that produce the same features produce the same hash."""
    digest = hashlib.sha256()
    pf = pq.ParquetFile(parquet_path)
    digest.update(','.join(pf.schema_arrow.names).encode())
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i)
        for col in table.columns:
            values = col.to_numpy()
            if values.dtype == object:
                digest.update('\x1f'.join('' if v is None else str(v) for v in values).encode())
            else:
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
