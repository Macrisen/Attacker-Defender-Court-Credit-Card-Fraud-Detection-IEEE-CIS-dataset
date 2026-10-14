-- Small isolated fixtures to verify NULL, constants, duplicates, and split ties.
\set ON_ERROR_STOP on
CREATE SCHEMA feature_audit_fixture_source;
CREATE TABLE feature_audit_fixture_source.train_merge AS
SELECT i AS transactionid, (i/2)::integer AS transactiondt, (i%11=0)::integer AS isfraud,
       i::numeric AS transactionamt, NULL::double precision AS all_null,
       7::integer AS constant_value, CASE WHEN i%3=0 THEN NULL ELSE 1 END AS missing_constant,
       i::double precision AS v1, i::double precision AS v2,
       CASE WHEN i%3=0 THEN NULL ELSE 'category' END::text AS category_a,
       CASE WHEN i%3=0 THEN NULL ELSE 'category' END::text AS category_b
FROM generate_series(1,1000) i;
CREATE TABLE feature_audit_fixture_source.test_merge AS
SELECT transactionid+2000 AS transactionid,transactiondt+2000 AS transactiondt,transactionamt,
       all_null,constant_value,missing_constant,v1,v2,category_a,category_b
FROM feature_audit_fixture_source.train_merge;
\set audit_schema feature_audit_fixture_results
\set source_schema feature_audit_fixture_source
\set train_table train_merge
\set test_table test_merge
\ir audit.sql
DO $$ BEGIN
  IF (SELECT count(*) FROM feature_audit_fixture_results.profile) <> 22 THEN
    RAISE EXCEPTION 'Profile coverage failed'; END IF;
  IF NOT (SELECT missing_count=total_rows AND unique_count=0 FROM feature_audit_fixture_results.profile
     WHERE feature='all_null' AND scope='model_train') THEN RAISE EXCEPTION 'All-NULL failed'; END IF;
  IF NOT (SELECT unique_count=1 AND missing_count=0 AND dominant_count=total_rows
     FROM feature_audit_fixture_results.profile WHERE feature='constant_value' AND scope='model_train')
     THEN RAISE EXCEPTION 'Constant failed'; END IF;
  IF NOT (SELECT unique_count=1 AND missing_count>0 FROM feature_audit_fixture_results.profile
     WHERE feature='missing_constant' AND scope='model_train') THEN RAISE EXCEPTION 'Missing constant failed'; END IF;
  IF NOT (SELECT identical FROM feature_audit_fixture_results.duplicate_columns WHERE feature_a='v1' AND feature_b='v2')
     THEN RAISE EXCEPTION 'Numeric duplicate failed'; END IF;
  IF NOT (SELECT identical FROM feature_audit_fixture_results.duplicate_columns WHERE feature_a='category_a' AND feature_b='category_b')
     THEN RAISE EXCEPTION 'Categorical duplicate failed'; END IF;
  IF (SELECT max(transactiondt) FROM feature_audit_fixture_results.train_snapshot WHERE _audit_window BETWEEN 1 AND 4)
     >= (SELECT min(transactiondt) FROM feature_audit_fixture_results.train_snapshot WHERE _audit_window=5)
     THEN RAISE EXCEPTION 'Chronological split leaked tied timestamps'; END IF;
END $$;
DROP SCHEMA feature_audit_fixture_results CASCADE;
DROP SCHEMA feature_audit_fixture_source CASCADE;
SELECT 'SQL fixture checks passed' AS result;
