"""Read-only, bounded-memory PostgreSQL export with complete snapshot readback validation."""
import argparse
import csv
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import time
import uuid

import psycopg
from psycopg import sql
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TABLES = ('train_clean', 'test_clean')


def metadata(conn, table):
    return conn.execute("""SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position""", (table,)).fetchall()


def relation(table):
    return sql.Identifier('public', table)


def batches(conn, query, size):
    with conn.cursor(name='export_'+uuid.uuid4().hex) as cur:
        cur.execute(query)
        while rows := cur.fetchmany(size):
            yield rows


def arrow_schema(conn, table, columns, size):
    types = {'integer': pa.int32(), 'smallint': pa.int16(), 'bigint': pa.int64(),
             'double precision': pa.float64(), 'real': pa.float32(), 'text': pa.string(),
             'character varying': pa.string(), 'boolean': pa.bool_()}
    fields, mappings = [], []
    for name, dtype in columns:
        if dtype == 'numeric':
            # Unconstrained PostgreSQL NUMERIC: determine exact scale from this snapshot.
            scale, integral = 0, 0
            q = sql.SQL('SELECT {} FROM {} WHERE {} IS NOT NULL').format(
                sql.Identifier(name), relation(table), sql.Identifier(name))
            for rows in batches(conn, q, size):
                for (value,) in rows:
                    if not value.is_finite():
                        raise ValueError(f'{table}.{name}: nonfinite NUMERIC is unsupported; no conversion performed')
                    _, digits, exponent = value.as_tuple()
                    scale = max(scale, -exponent)
                    integral = max(integral, len(digits)+exponent)
            precision = max(1, integral+scale)
            if precision > 76:
                raise ValueError(f'{table}.{name}: precision {precision} exceeds decimal256; export stopped')
            atype = pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)
        elif dtype in types:
            atype = types[dtype]
        else:
            raise ValueError(f'Unsupported PostgreSQL type {table}.{name}: {dtype}')
        fields.append(pa.field(name, atype))
        mappings.append({'column': name, 'postgresql_type': dtype, 'parquet_arrow_type': str(atype)})
    return pa.schema(fields), mappings


def to_arrow(rows, schema):
    arrays = []
    for i, field in enumerate(schema):
        values = [row[i] for row in rows]
        array = pa.array(values, type=field.type, safe=True)
        if pa.types.is_decimal(field.type):
            if array.to_pylist() != values:
                raise ValueError(f'Loss of decimal precision in {field.name}')
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=schema)


def export_table(conn, table, temporary, batch_size):
    started = time.perf_counter()
    columns = metadata(conn, table)
    if not columns:
        raise ValueError(f'public.{table} does not exist or has no columns')
    names = [c[0] for c in columns]
    ids = [n for n in names if n.lower() == 'transactionid']
    if len(ids) != 1:
        raise ValueError(f'{table}: expected one TransactionID column')
    idcol = ids[0]
    count, distinct, nonnull = conn.execute(sql.SQL('SELECT count(*),count(DISTINCT {}),count({}) FROM {}').format(
        sql.Identifier(idcol), sql.Identifier(idcol), relation(table))).fetchone()
    if count != distinct or count != nonnull:
        raise ValueError(f'{table}: source IDs are duplicate or NULL; export stopped without deleting rows')
    label = next((n for n in names if n.lower() == 'isfraud'), None)
    if table == 'train_clean':
        if not label:
            raise ValueError('Training label missing')
        invalid = conn.execute(sql.SQL('SELECT count(*) FROM {} WHERE {} IS NULL OR {} NOT IN (0,1)').format(
            relation(table), sql.Identifier(label), sql.Identifier(label))).fetchone()[0]
        if invalid:
            raise ValueError(f'{invalid} invalid training labels')
    schema, mappings = arrow_schema(conn, table, columns, batch_size)
    null_exprs = [sql.SQL('count(*) FILTER (WHERE {} IS NULL)').format(sql.Identifier(n)) for n in names]
    pg_nulls = conn.execute(sql.SQL('SELECT {} FROM {}').format(sql.SQL(',').join(null_exprs), relation(table))).fetchone()
    query = sql.SQL('SELECT {} FROM {} ORDER BY {}').format(
        sql.SQL(',').join(map(sql.Identifier, names)), relation(table), sql.Identifier(idcol))
    written = 0
    with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        for rows in batches(conn, query, batch_size):
            arrow = to_arrow(rows, schema)
            writer.write_table(arrow, row_group_size=batch_size)
            written += len(rows)
            logging.info('%s: exported %s/%s rows', table, written, count)
    # Compare every value with a fresh stream from the SAME repeatable-read snapshot.
    pf = pq.ParquetFile(temporary)
    try:
        if pf.metadata.num_rows != count or written != count or not pf.schema_arrow.equals(schema):
            raise ValueError(f'{table}: row count or schema mismatch')
        parquet_nulls = [0]*len(names)
        read_rows, groups, previous_id = 0, 0, None
        for group, rows in enumerate(batches(conn, query, batch_size)):
            actual = pf.read_row_group(group)
            expected = to_arrow(rows, schema)
            if not actual.equals(expected):
                raise ValueError(f'{table}: full-value readback mismatch in row group {group}')
            for i, col in enumerate(actual.columns):
                parquet_nulls[i] += col.null_count
            values = actual[idcol].to_pylist()
            for value in values:
                if previous_id is not None and value <= previous_id:
                    raise ValueError(f'{table}: duplicate or unordered output ID')
                previous_id = value
            read_rows += len(rows)
            groups += 1
            logging.info('%s: verified every value in %s/%s rows', table, read_rows, count)
        if groups != pf.num_row_groups or read_rows != count or tuple(parquet_nulls) != pg_nulls:
            raise ValueError(f'{table}: row group/row count/NULL mismatch')
        codecs = {pf.metadata.row_group(i).column(j).compression for i in range(pf.num_row_groups) for j in range(len(names))}
        if count and codecs != {'ZSTD'}:
            raise ValueError(f'{table}: unexpected compression {codecs}')
    finally:
        pf.close()
    for mapping, nulls in zip(mappings, pg_nulls):
        mapping['postgresql_nulls'] = nulls
        mapping['parquet_nulls'] = nulls
    result = {'table_name': 'public.'+table, 'postgresql_row_count': count,
              'parquet_row_count': read_rows, 'column_count': len(names),
              'parquet_file_size_mb': temporary.stat().st_size/1_000_000,
              'compression': 'zstd', 'export_duration_seconds': time.perf_counter()-started,
              'validation_status': 'PASS'}
    return result, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=50_000)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files only after both exports pass validation')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    load_dotenv(ROOT/'.env')
    output, reports = ROOT/'data/processed', ROOT/'reports/parquet_export'
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(reports/'execution.log')])
    destinations = [output/(t+'.parquet') for t in TABLES]
    if not args.overwrite and any(p.exists() for p in destinations):
        raise FileExistsError('Existing Parquet files preserved. Use --overwrite to export and validate replacements.')
    temporary = [p.with_suffix('.parquet.'+uuid.uuid4().hex+'.tmp') for p in destinations]
    results, details = [], {}
    try:
        with psycopg.connect(dbname=os.getenv('PGDATABASE', 'fraud_detection')) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            snapshot = conn.execute('SELECT current_database(),current_timestamp::text,txid_current_snapshot()::text').fetchone()
            for table, temp in zip(TABLES, temporary):
                result, mappings = export_table(conn, table, temp, args.batch_size)
                results.append(result)
                details[table] = mappings
        for temp, dest in zip(temporary, destinations):
            os.replace(temp, dest)
        with (reports/'export_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        (reports/'schema_and_null_validation.json').write_text(json.dumps(details, indent=2)+'\n')
        lines = ['# PostgreSQL to Parquet validation: PASS', '',
                 f'Database: `{snapshot[0]}`, schema: `public`. Snapshot timestamp: {snapshot[1]}; snapshot ID: `{snapshot[2]}`.',
                 f'Batch size: {args.batch_size:,}. Read-only repeatable-read transaction; source tables unchanged.', '']
        for result, path in zip(results, destinations):
            lines += [f'## {result["table_name"]}', f'- File: `{path.relative_to(ROOT)}`',
                      f'- PostgreSQL and Parquet rows: {result["parquet_row_count"]:,}',
                      f'- Columns: {result["column_count"]}; names and order match exactly.',
                      f'- Size: {result["parquet_file_size_mb"]:.2f} MB (decimal); compression: ZSTD.',
                      f'- Export plus validation: {result["export_duration_seconds"]:.2f} seconds.',
                      '- All values compared against a second ordered PostgreSQL stream from the same snapshot: PASS.',
                      '- Every column NULL count: PASS. TransactionID values, uniqueness and count: PASS.', '']
        lines += ['## Types and integrity',
                  'PostgreSQL smallint/integer/bigint map to Arrow int16/int32/int64; double precision maps to float64; text maps to UTF-8 string. NUMERIC stays exact decimal, with precision and scale inferred from all non-NULL values in the snapshot. Decimal conversion is explicitly checked for equality. No float32 downcast, imputation, normalization, feature selection or new label is applied.',
                  'Training labels verified non-NULL and binary. All source columns are preserved, including ID, time and label where present. Per-column types and NULL counts are recorded in `schema_and_null_validation.json`.',
                  'Validation compares every value, so no probabilistic checksum or sampling is used. Records are ordered by unique TransactionID for reproducibility; SQL tables have no inherent row order.',
                  'Warnings: inferred decimal types describe this export snapshot; a later export may require different precision/scale. Both temporary exports are validated before publication; each destination replacement is atomic, but the pair is not a filesystem-wide atomic transaction.',
                  '', '**Conclusion: PASS.**']
        (reports/'export_validation_report.md').write_text('\n'.join(lines)+'\n')
        for path in destinations:
            logging.info('PASS: %s', path)
    except Exception as error:
        # Avoid logging connection strings, environment variables or secrets.
        message = str(error).replace(os.getenv('PGPASSWORD') or '\0', '[REDACTED]')
        (reports/'last_failure.md').write_text(f'# Export failed\n\n{type(error).__name__}: {message}\n')
        logging.error('Export failed (%s); see reports/parquet_export/last_failure.md', type(error).__name__)
        raise SystemExit(1) from None
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
