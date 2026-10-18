"""Optional read-only Original export, reusing the validated chunk exporter.

Not invoked by Phase 3.5. Requires existing public.train_merge/test_merge.
"""
import json
import os
from pathlib import Path
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from scripts.export_postgres_to_parquet import export_table, metadata
from rl_environment.temporal_protocol import verify_alignment


def main():
    load_dotenv(ROOT/'.env')
    pairs = [('train_merge', 'train_original', 'train_clean'), ('test_merge', 'test_original', 'test_clean')]
    destinations = [ROOT/'data/processed'/f'{name}.parquet' for _, name, _ in pairs]
    if any(p.exists() for p in destinations):
        raise FileExistsError('Original output exists; no overwrite')
    temporary = [p.with_suffix('.parquet.'+uuid.uuid4().hex+'.tmp') for p in destinations]
    results = []
    try:
        with psycopg.connect(dbname=os.getenv('PGDATABASE', 'fraud_detection')) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            for (table, name, clean), tmp in zip(pairs, temporary):
                columns = {c[0] for c in metadata(conn, table)}
                schema = json.loads((ROOT/'reports/lightgbm_baseline/experiments/A_original/feature_schema.json').read_text())
                expected = {f['name'] for f in schema['features']} | {'transactionid', 'transactiondt'}
                if table == 'train_merge':
                    expected.add('isfraud')
                    invalid = conn.execute(sql.SQL('SELECT count(*) FROM {} WHERE isfraud IS NULL OR isfraud NOT IN (0,1)').format(sql.Identifier('public', table))).fetchone()[0]
                    if invalid:raise ValueError('Invalid Original training labels')
                if columns != expected:raise ValueError('Original schema does not match Original Defender artifacts')
                result, mapping = export_table(conn, table, tmp, 50000)
                verify_alignment(ROOT/'data/processed'/f'{clean}.parquet', tmp)
                results.append({'export': result, 'schema_and_null_validation': mapping})
        for tmp, dest in zip(temporary, destinations):os.replace(tmp, dest)
        report = ROOT/'reports/parquet_export_original';report.mkdir(parents=True, exist_ok=True)
        (report/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
        for dest in destinations:print(dest)
    finally:
        for tmp in temporary:tmp.unlink(missing_ok=True)


if __name__ == '__main__':main()
