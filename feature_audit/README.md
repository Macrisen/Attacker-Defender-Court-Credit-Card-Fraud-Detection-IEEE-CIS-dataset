# IEEE-CIS feature audit

This pipeline reads `fraud_detection.public.train_merge` and `public.test_merge`.
It creates a separate audit schema and report files. It does **not** alter source
tables, remove features, remove transactions, impute values, or train a model.

## Run the audit

From the project directory, with PostgreSQL running and `psql` on PATH:

```bash
python3 feature_audit/run_audit.py \
  --database fraud_detection --user macrisen \
  --source-schema public --train-table train_merge --test-table test_merge
```

The default creates a new timestamped audit schema. Supply `--host` / `--port` if
needed. Authentication uses PostgreSQL's normal environment/`.pgpass`; no passwords
are stored in scripts. Python dependencies are in `requirements.txt`; the exact
versions used in this run are recorded in `requirements-lock.txt` and the manifest.

Outputs: `feature_audit/reports/<audit_schema>/`. The SQL snapshot is permanent for
reproduction and consumes additional database disk space. No source writes occur.
Run names must be new: existing audit schemas are never overwritten.

To regenerate reports from the **already executed** snapshot:

```bash
python3 feature_audit/run_audit.py --reuse-sql \
  --audit-schema feature_audit_20261014_v1
```

This rewrites generated reports in that output directory; preserve reviewed files
or use `--output <new-directory>` if retaining multiple report versions.

The SQL stage can also be executed independently:

```bash
psql -X -U macrisen -d fraud_detection \
  -v audit_schema=feature_audit_new_run \
  -v source_schema=public -v train_table=train_merge -v test_table=test_merge \
  -f feature_audit/audit.sql
```

Then run the Python script with `--reuse-sql --audit-schema feature_audit_new_run`.
The script stops on SQL errors; a failed SQL transaction rolls back all its audit
objects. The source tables are never changed. Test data is snapshotted for later
application, but not used to fit feature decisions.

## Report files

- `preprocessing_summary.md`: research summary, calculated counts, recommendations,
  group analysis, limitations, and linked PNG charts.
- `data_quality_report.csv`: model-training NULL/unique/dominant statistics and status.
- `data_quality_all_training.csv`: descriptive statistics for all labeled rows;
  **not** used to fit selection rules.
- `high_correlation_pairs.csv`: abs(Pearson r) >=0.98, shared counts, window checks,
  missing-pattern differences, and globally consistent removal proposals.
- `constant_features.csv`: all-NULL, constant non-NULL, and near-constant candidates.
- `feature_selection_report.csv`: all columns, group/role, KEEP/DROP_CANDIDATE/REVIEW,
  reasons, rule names, retained representative, and temporal-availability caveat.
- `proposed_drops.csv`: concise review list, not an instruction to delete.
- `duplicate_features.csv`, `redundant_feature_groups.csv`, `fraud_associations.csv`,
  `chronological_feature_stability.csv`, `feature_group_summary.csv`: supporting evidence.
- `integrity.csv`, `schema_comparison.csv`, `partition_summary.csv`: integrity and split.
- `correlation_validation.csv`: PostgreSQL checks of all flagged correlations plus
  additional ordinary and undefined pairs.
- `window_correlation_validation.csv`: direct PostgreSQL correlations and paired
  observation counts for every flagged pair in each fitting window.
- `low_variance_correlation_checks.csv`: direct PostgreSQL recomputation of
  numerically ill-conditioned pairwise variances, including truly constant overlaps.
- `all_numeric_correlations.csv`, `correlation_matrices.npz`: complete finite correlation results.
- `selection_manifest.json`: configuration, versions, script hashes, selected feature
  proposals, exact split cutoff, snapshot identity, and approval status.

All primary decisions use only the first approximately 80% of labeled rows ordered
by `transactiondt`. Equal timestamps remain on one side of every boundary.
Validation is held out. Full-training descriptive statistics are clearly separated.

`TransactionID`/`isFraud` are technical/target columns, never predictive features.
`TransactionDT` is retained as metadata for chronological evaluation. The default
numeric-role list excludes categorical codes; review semantic assignments before modeling.
Optional LightGBM experiments and feature importance are **not executed** by this pipeline.

## Review, then apply approved changes

No final removal has been executed. Review `proposed_drops.csv` and its supporting
reports with the user. After explicit approval, fill `approved_drops.json` with
only the approved candidate names. The generated template has an empty list;
empty means no columns are approved for removal.

First prepare reviewable SQL without changing the database:

```bash
python3 feature_audit/apply_approved.py \
  --report-dir feature_audit/reports/feature_audit_20261014_v1 \
  --approval-file feature_audit/reports/feature_audit_20261014_v1/approved_drops.json \
  --output-schema fraud_selected_v1
```

Only **after approval**, execute by adding `--apply-approved` to that command.
The script creates `fraud_selected_v1.train_clean`, `test_clean`, `model_train`
and `validation` views, and a database preprocessing log. The original datasets
and snapshots remain intact. Every row is retained. The two feature sets match
except for the training-only target, and metadata columns are protected. A primary
key and row-count assertions stop an inconsistent application.

The generated datasets are selected raw features with native NULLs. Before LightGBM,
fit categorical mappings on `model_train` only and reuse them on validation/test;
unseen categories need an explicit missing/unknown policy. Do not infer separate
category code mappings from each partition. Numeric imputation/scaling is not
performed. RL states must omit `isfraud`; rewards and action semantics need a
separate design.

Optional Parquet export: install `pyarrow` and add `--parquet` when applying approved
changes. It streams SQL data in batches, preserves text/NULL distinction, uses
integer/double/string types and decimal128(38,12) for SQL numeric, and fails rather
than silently accepting incompatible decimal precision. Parquet is deferred until
approval and is not a current deliverable. PyArrow is not installed in this run.

## Checks

```bash
psql -X -U macrisen -d fraud_detection -f feature_audit/test_audit.sql
python3 feature_audit/verify_reports.py \
  --report-dir feature_audit/reports/feature_audit_20261014_v1
```

This creates isolated fixture schemas, tests missing/constant features, numeric
and categorical duplicates and tied chronological boundaries, then removes only
those fixtures. Python's pairwise-correlation implementation is tested against
independent NumPy/Pandas examples and PostgreSQL aggregate functions during every
run. The numerical implementation explicitly allocates squared-value buffers to
avoid a temporary-array reuse issue observed in this Python/NumPy environment.
The report verifier rechecks selected full-training and fitting statistics against
PostgreSQL, source-script hashes, row counts, feature coverage and retained
representatives. Its actual results are written to `verification.json`.

`*.csv` is ignored by the project's existing `.gitignore`; generated CSV reports
exist locally regardless. If versioning results, deliberately add selected report
files with `git add -f`, without adding the large original dataset CSVs.
