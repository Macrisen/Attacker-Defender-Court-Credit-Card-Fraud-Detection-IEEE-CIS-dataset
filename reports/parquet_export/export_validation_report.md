# PostgreSQL to Parquet validation: PASS

Database: `fraud_detection`, schema: `public`. Snapshot timestamp: 2026-10-16 18:22:12.030628+07; snapshot ID: `1331:1331:`.
Batch size: 50,000. Read-only repeatable-read transaction; source tables unchanged.

## public.train_clean
- File: `data/processed/train_clean.parquet`
- PostgreSQL and Parquet rows: 590,540
- Columns: 421; names and order match exactly.
- Size: 64.20 MB (decimal); compression: ZSTD.
- Export plus validation: 53.98 seconds.
- All values compared against a second ordered PostgreSQL stream from the same snapshot: PASS.
- Every column NULL count: PASS. TransactionID values, uniqueness and count: PASS.

## public.test_clean
- File: `data/processed/test_clean.parquet`
- PostgreSQL and Parquet rows: 506,691
- Columns: 420; names and order match exactly.
- Size: 56.10 MB (decimal); compression: ZSTD.
- Export plus validation: 46.58 seconds.
- All values compared against a second ordered PostgreSQL stream from the same snapshot: PASS.
- Every column NULL count: PASS. TransactionID values, uniqueness and count: PASS.

## Types and integrity
PostgreSQL smallint/integer/bigint map to Arrow int16/int32/int64; double precision maps to float64; text maps to UTF-8 string. NUMERIC stays exact decimal, with precision and scale inferred from all non-NULL values in the snapshot. Decimal conversion is explicitly checked for equality. No float32 downcast, imputation, normalization, feature selection or new label is applied.
Training labels verified non-NULL and binary. All source columns are preserved, including ID, time and label where present. Per-column types and NULL counts are recorded in `schema_and_null_validation.json`.
Validation compares every value, so no probabilistic checksum or sampling is used. Records are ordered by unique TransactionID for reproducibility; SQL tables have no inherent row order.
Warnings: inferred decimal types describe this export snapshot; a later export may require different precision/scale. Both temporary exports are validated before publication; each destination replacement is atomic, but the pair is not a filesystem-wide atomic transaction.

**Conclusion: PASS.**
