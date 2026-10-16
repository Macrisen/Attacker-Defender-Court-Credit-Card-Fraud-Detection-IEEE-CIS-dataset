# Export cleaned PostgreSQL tables

From the project root:

```bash
.venv_defender/bin/python -m pip install -r requirements.txt
.venv_defender/bin/python scripts/export_postgres_to_parquet.py
```

The script reads `PGDATABASE` (default `fraud_detection`), `PGUSER`, `PGHOST`,
`PGPORT` and `PGPASSWORD` from the environment or a project `.env` file.
Libpq defaults and `.pgpass` are also supported. Use `.env.example` as a template;
no `.env` or password is required when local PostgreSQL authentication already works.

Outputs:

- `data/processed/train_clean.parquet`
- `data/processed/test_clean.parquet`
- `reports/parquet_export/export_summary.csv`
- `reports/parquet_export/export_validation_report.md`
- `reports/parquet_export/schema_and_null_validation.json`

Both tables are read from one read-only repeatable-read snapshot. Server cursors
stream batches without OFFSET. Exact decimals remain decimals, floating-point
columns retain their original precision, and column order and NULLs are preserved.
Each Parquet row group is compared with every source value in a second ordered
stream from the same snapshot. SQL aggregate NULL counts are checked separately.

Existing Parquet outputs are protected by default. To produce validated replacements:

```bash
.venv_defender/bin/python scripts/export_postgres_to_parquet.py --overwrite --batch-size 50000
```

For lower memory use, choose `--batch-size 10000`. Peak memory depends on Python
objects and the number of columns, not only the compressed file size. All temporary
files are validated before publication. Each individual file is replaced atomically;
publication of two files is not one atomic filesystem operation.

Read an exported dataset without changing its Arrow types:

```python
import pyarrow.parquet as pq
table = pq.read_table('data/processed/train_clean.parquet')
# For streaming:
for batch in pq.ParquetFile('data/processed/train_clean.parquet').iter_batches(batch_size=10000):
    pass  # downstream training preparation
```

Parquet retains the exact decimal amount column. Any conversion to model-specific
floating-point inputs belongs in the later model pipeline, not this exporter.
