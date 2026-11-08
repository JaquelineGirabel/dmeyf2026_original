"""Checks the features stage on the local test dataset (10% of clients). Run from the undersampling folder:

    python tests/check_features.py

1. Target: identical to the course notebook's query (clusters/z101_target_sql.ipynb), for every row.
2. Lags: lag k equals the client's value exactly k months earlier, and is empty if the client was absent.
3. Determinism: an input file with rows and columns shuffled gives exactly the same features.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402
from pipeline.features import HASH_MOD, HASH_MULT, build_features  # noqa: E402

NOTEBOOK_TARGET = """
with periodos as (select distinct foto_mes from crudo),
clientes as (select distinct numero_de_cliente from crudo),
todo as (select numero_de_cliente, foto_mes from clientes cross join periodos),
clase_ternaria as (
    select c.*, if(c.numero_de_cliente is null, 0, 1) as mes_0,
        lead(mes_0, 1) over (partition by t.numero_de_cliente order by foto_mes) as mes_1,
        lead(mes_0, 2) over (partition by t.numero_de_cliente order by foto_mes) as mes_2,
        case when mes_2 = 1 then 'CONTINUA'
             when mes_1 = 0 and (mes_2 = 0 or mes_2 is null) then 'BAJA+1'
             when mes_1 = 1 and mes_2 = 0 then 'BAJA+2'
             else null end as clase_ternaria
    from todo t left join crudo c using (numero_de_cliente, foto_mes))
select numero_de_cliente, foto_mes, clase_ternaria from clase_ternaria where mes_0 = 1
"""


def main():
    cfg = Config(sample_clients=0.1)
    parquet, meta = build_features(cfg, {})
    source = cfg.path(cfg.data_file).as_posix()
    sample = f'(CAST(numero_de_cliente AS HUGEINT) * {HASH_MULT}) % {HASH_MOD} < {int(0.1 * HASH_MOD)}'
    con = duckdb.connect()
    con.execute('SET enable_progress_bar=false')
    con.execute(f"CREATE TABLE crudo AS SELECT * FROM read_csv_auto('{source}') WHERE {sample}")
    con.execute(f"CREATE TABLE feats AS SELECT * FROM read_parquet('{parquet.as_posix()}')")
    ok = True

    # 1. Target
    con.execute(f'CREATE TABLE ref AS {NOTEBOOK_TARGET}')
    mismatches = con.execute("""
        SELECT count(*) FROM feats f JOIN ref r USING (numero_de_cliente, foto_mes)
        WHERE f.clase_ternaria IS DISTINCT FROM r.clase_ternaria
    """).fetchone()[0]
    missing = con.execute("""SELECT count(*) FROM feats f ANTI JOIN ref r USING (numero_de_cliente, foto_mes)""").fetchone()[0]
    print(f'1. Target: {mismatches} mismatches, {missing} rows missing in the notebook query')
    ok &= mismatches == 0 and missing == 0

    # 2. Calendar-exact lags (mrentabilidad as example)
    bad = con.execute("""
        WITH c AS (SELECT numero_de_cliente, foto_mes, mrentabilidad,
                          (foto_mes // 100) * 12 + foto_mes % 100 AS idx FROM crudo)
        SELECT count(*) FROM feats f
        JOIN c cur USING (numero_de_cliente, foto_mes)
        LEFT JOIN c p1 ON p1.numero_de_cliente = cur.numero_de_cliente AND p1.idx = cur.idx - 1
        LEFT JOIN c p2 ON p2.numero_de_cliente = cur.numero_de_cliente AND p2.idx = cur.idx - 2
        WHERE f.mrentabilidad_lag1 IS DISTINCT FROM CAST(p1.mrentabilidad AS FLOAT)
           OR f.mrentabilidad_lag2 IS DISTINCT FROM CAST(p2.mrentabilidad AS FLOAT)
           OR f.mrentabilidad_delta1 IS DISTINCT FROM CAST(cur.mrentabilidad - p1.mrentabilidad AS FLOAT)
    """).fetchone()[0]
    gaps = con.execute("""
        WITH c AS (SELECT numero_de_cliente, (foto_mes // 100) * 12 + foto_mes % 100 AS idx FROM crudo),
        g AS (SELECT idx - lag(idx) OVER (PARTITION BY numero_de_cliente ORDER BY idx) AS step FROM c)
        SELECT count(*) FROM g WHERE step > 1
    """).fetchone()[0]
    print(f'2. Lags: {bad} wrong values (clients with a gap month in the data: {gaps})')
    ok &= bad == 0

    # 3. Determinism: shuffled rows and columns
    tmp = Path(tempfile.mkdtemp())
    try:
        cols = [r[0] for r in con.execute('DESCRIBE crudo').fetchall()]
        shuffled_cols = ', '.join(f'"{c}"' for c in reversed(cols))
        shuffled_csv = tmp / 'shuffled.csv.gz'
        con.execute(f"COPY (SELECT {shuffled_cols} FROM crudo ORDER BY hash(numero_de_cliente, foto_mes, 7)) "
                    f"TO '{shuffled_csv.as_posix()}' (FORMAT CSV, HEADER)")
        cfg2 = Config(sample_clients=0.1, data_file=str(shuffled_csv), features_dir=str(tmp / 'features'))
        _, meta2 = build_features(cfg2, {})
        same = meta2['content_sha256'] == meta['content_sha256']
        print(f'3. Determinism: shuffled input gives {"the same" if same else "DIFFERENT"} features '
              f"({meta['content_sha256'][:12]} vs {meta2['content_sha256'][:12]})")
        ok &= same
    finally:
        shutil.rmtree(tmp)

    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
