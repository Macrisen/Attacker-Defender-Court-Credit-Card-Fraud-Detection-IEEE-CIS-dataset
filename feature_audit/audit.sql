-- PostgreSQL 17. Run through run_audit.py; all identifiers are safely quoted.
-- Required psql variables: audit_schema, source_schema, train_table, test_table.
\set ON_ERROR_STOP on
\pset pager off
SELECT set_config('audit.schema', :'audit_schema', false);
SELECT set_config('audit.source_schema', :'source_schema', false);
SELECT set_config('audit.train_table', :'train_table', false);
SELECT set_config('audit.test_table', :'test_table', false);
BEGIN ISOLATION LEVEL REPEATABLE READ;
SET LOCAL work_mem = '96MB';
SET LOCAL jit = off;
DO $$
DECLARE
    a text := current_setting('audit.schema');
    s text := current_setting('audit.source_schema');
    tr text := current_setting('audit.train_table');
    te text := current_setting('audit.test_table');
    cuts double precision[];
    fitcuts double precision[];
    r record;
    k integer := 0;
    numeric_col boolean;
    canonical text;
    mean_expr text;
    std_expr text;
    nonfinite_expr text;
BEGIN
    EXECUTE format('CREATE SCHEMA %I', a);
    PERFORM set_config('search_path', format('%I, pg_catalog', a), true);
    CREATE TABLE run_metadata AS SELECT clock_timestamp() AS started_at,
        version() AS postgres_version, s AS source_schema, tr AS train_table, te AS test_table,
        txid_current_snapshot()::text AS database_snapshot;
    CREATE TABLE columns AS
        SELECT table_name, column_name, ordinal_position, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema=s AND table_name IN (tr,te);
    IF (SELECT count(*) FROM columns WHERE table_name=tr AND column_name IN
        ('transactionid','transactiondt','isfraud')) <> 3 THEN
        RAISE EXCEPTION 'Training table requires transactionid, transactiondt and isfraud';
    END IF;
    IF EXISTS (SELECT 1 FROM columns WHERE column_name LIKE '\_audit\_%' ESCAPE '\') THEN
        RAISE EXCEPTION 'Source columns conflict with reserved _audit_ prefix';
    END IF;

    CREATE TABLE integrity(table_name text, row_count bigint, missing_ids bigint,
        duplicate_id_excess bigint, missing_time bigint, nonfinite_time bigint,
        missing_labels bigint, invalid_labels bigint, missing_amount bigint,
        negative_amount bigint, min_time double precision, max_time double precision);
    FOR r IN SELECT unnest(ARRAY[tr,te]) AS name LOOP
        EXECUTE format($q$INSERT INTO integrity
          SELECT %L, count(*), count(*) FILTER(WHERE transactionid IS NULL),
            count(transactionid)-count(DISTINCT transactionid),
            count(*) FILTER(WHERE transactiondt IS NULL),
            count(*) FILTER(WHERE transactiondt::text IN ('NaN','Infinity','-Infinity')),
            %s, %s, count(*) FILTER(WHERE transactionamt IS NULL),
            count(*) FILTER(WHERE transactionamt < 0), min(transactiondt), max(transactiondt)
          FROM %I.%I$q$, r.name,
          CASE WHEN r.name=tr THEN 'count(*) FILTER(WHERE isfraud IS NULL)' ELSE 'NULL::bigint' END,
          CASE WHEN r.name=tr THEN 'count(*) FILTER(WHERE isfraud NOT IN (0,1))' ELSE 'NULL::bigint' END,
          s,r.name);
    END LOOP;
    CREATE TABLE critical_gate AS SELECT NOT EXISTS (
        SELECT 1 FROM integrity WHERE missing_ids>0 OR duplicate_id_excess>0 OR
          missing_time>0 OR nonfinite_time>0 OR missing_labels>0 OR invalid_labels>0
    ) AS passed;
    IF NOT (SELECT passed FROM critical_gate) THEN
        RAISE NOTICE 'Integrity gate failed. Reports can inspect integrity; no selection should be applied.';
    END IF;

    EXECUTE format($q$SELECT percentile_disc(ARRAY[0.8]) WITHIN GROUP (ORDER BY transactiondt)
        FROM %I.%I WHERE transactiondt IS NOT NULL
        AND transactiondt::text NOT IN ('NaN','Infinity','-Infinity')$q$,s,tr) INTO cuts;
    IF cuts[1] IS NULL THEN RAISE EXCEPTION 'No valid chronological split available'; END IF;
    EXECUTE format($q$SELECT percentile_disc(ARRAY[0.25,0.5,0.75]) WITHIN GROUP (ORDER BY transactiondt)
        FROM %I.%I WHERE transactiondt <= $1$q$,s,tr) INTO fitcuts USING cuts[1];
    CREATE TABLE split_definition(cutoff double precision, window1_end double precision,
        window2_end double precision, window3_end double precision, requested_fit_fraction numeric);
    INSERT INTO split_definition VALUES(cuts[1],fitcuts[1],fitcuts[2],fitcuts[3],0.8);
    -- Snapshot preserves every row. Invalid time gets window_id 0, never silently removed.
    EXECUTE format($q$CREATE TABLE train_snapshot AS SELECT t.*,
       CASE WHEN transactiondt IS NULL OR transactiondt::text IN ('NaN','Infinity','-Infinity') THEN 0
            WHEN transactiondt > $1 THEN 5
            WHEN transactiondt <= $2 THEN 1
            WHEN transactiondt <= $3 THEN 2
            WHEN transactiondt <= $4 THEN 3 ELSE 4 END::smallint AS _audit_window
       FROM %I.%I AS t$q$,s,tr) USING cuts[1],fitcuts[1],fitcuts[2],fitcuts[3];
    CREATE INDEX ON train_snapshot (_audit_window);
    ANALYZE train_snapshot;
    -- Preserve test rows for reproducible future application; never fit rules on them.
    EXECUTE format('CREATE TABLE test_snapshot AS TABLE %I.%I',s,te);
    CREATE TABLE partition_summary AS SELECT _audit_window AS window_id,
        count(*) AS rows, count(*) FILTER(WHERE isfraud=1) AS fraud_rows,
        min(transactiondt) AS min_time, max(transactiondt) AS max_time
        FROM train_snapshot GROUP BY 1 ORDER BY 1;
    EXECUTE format('CREATE TABLE id_overlap AS SELECT count(*) AS overlapping_ids FROM (SELECT transactionid FROM %I.%I INTERSECT SELECT transactionid FROM %I.%I) q',s,tr,s,te);
    CREATE TABLE profile (
        scope text, feature text, data_type text, total_rows bigint, missing_count bigint,
        non_null_count bigint, unique_count bigint, dominant_value text, dominant_count bigint,
        dominant_fraud bigint, missing_fraud bigint, total_fraud bigint,
        nonfinite_count bigint, numeric_mean double precision, numeric_std double precision,
        fingerprint numeric, PRIMARY KEY(scope,feature)
    );
    CREATE TABLE window_profile(feature text, window_id integer, total_rows bigint,
        missing_count bigint, unique_count bigint, missing_fraud bigint,
        total_fraud bigint, numeric_mean double precision, numeric_std double precision,
        PRIMARY KEY(feature,window_id));
    FOR r IN SELECT * FROM columns WHERE table_name=tr ORDER BY ordinal_position LOOP
        numeric_col := r.data_type IN ('smallint','integer','bigint','numeric','real','double precision');
        -- Canonical numeric serialization is only a duplicate-candidate screen.
        -- Every candidate is subsequently verified using exact SQL value comparison.
        canonical := CASE WHEN numeric_col THEN format('trim_scale(%I::numeric)::text',r.column_name)
                          ELSE format('%I::text',r.column_name) END;
        EXECUTE format($q$CREATE TEMP TABLE feature_freq ON COMMIT DROP AS
            SELECT %I::text AS value, _audit_window AS window_id, count(*) AS n,
                   count(*) FILTER(WHERE isfraud=1) AS fraud,
                   sum(hashtextextended(jsonb_build_array(transactionid,%s)::text,0)::numeric)
                       FILTER(WHERE _audit_window BETWEEN 1 AND 4) AS fingerprint
            FROM train_snapshot GROUP BY 1,2$q$,r.column_name,canonical);
        CREATE INDEX ON feature_freq(window_id);
        mean_expr := CASE WHEN numeric_col THEN
            'sum(value::double precision*n) FILTER(WHERE value NOT IN (''NaN'',''Infinity'',''-Infinity'')) / NULLIF(sum(n) FILTER(WHERE value NOT IN (''NaN'',''Infinity'',''-Infinity'')),0)'
            ELSE 'NULL::double precision' END;
        -- Means/std are used for numerical conditioning, not imputation.
        std_expr := CASE WHEN numeric_col THEN
            'sqrt(greatest(0, sum((value::double precision)^2*n) FILTER(WHERE value NOT IN (''NaN'',''Infinity'',''-Infinity'')) / NULLIF(sum(n) FILTER(WHERE value NOT IN (''NaN'',''Infinity'',''-Infinity'')),0) - (' || mean_expr || ')^2))'
            ELSE 'NULL::double precision' END;
        nonfinite_expr := CASE WHEN numeric_col THEN
            'COALESCE(sum(n) FILTER(WHERE value IN (''NaN'',''Infinity'',''-Infinity'')),0)'
            ELSE '0::bigint' END;
        EXECUTE format($q$INSERT INTO profile
          WITH scoped AS (
            SELECT 'all_train'::text AS scope,value,n,fraud,fingerprint FROM feature_freq
            UNION ALL SELECT 'model_train',value,n,fraud,fingerprint FROM feature_freq WHERE window_id BETWEEN 1 AND 4
          ), vals AS (
            SELECT scope,value,sum(n)::bigint AS n,sum(fraud)::bigint AS fraud,sum(fingerprint) AS fingerprint
            FROM scoped GROUP BY scope,value
          ), dominant AS (
            SELECT DISTINCT ON(scope) scope,value,n,fraud FROM vals
            WHERE value IS NOT NULL ORDER BY scope,n DESC,value
          ), totals AS (
            SELECT scope,sum(n)::bigint AS total_rows,
              COALESCE(sum(n) FILTER(WHERE value IS NULL),0)::bigint AS missing,
              COALESCE(sum(n) FILTER(WHERE value IS NOT NULL),0)::bigint AS nonnull,
              count(value) AS uniques,
              COALESCE(sum(fraud) FILTER(WHERE value IS NULL),0)::bigint AS missing_fraud,
              sum(fraud)::bigint AS total_fraud,%s AS nonfinite,%s AS mean,%s AS std,
              sum(fingerprint) AS fingerprint
            FROM vals GROUP BY scope
          ) SELECT t.scope,%L,%L,t.total_rows,t.missing,t.nonnull,t.uniques,d.value,
              COALESCE(d.n,0),COALESCE(d.fraud,0),t.missing_fraud,t.total_fraud,
              t.nonfinite,t.mean,t.std,t.fingerprint
            FROM totals t LEFT JOIN dominant d USING(scope)$q$,
            nonfinite_expr,mean_expr,std_expr,r.column_name,r.data_type);
        EXECUTE format($q$INSERT INTO window_profile
            SELECT %L,window_id,sum(n),COALESCE(sum(n) FILTER(WHERE value IS NULL),0),count(value),
                COALESCE(sum(fraud) FILTER(WHERE value IS NULL),0),sum(fraud),%s,%s
            FROM feature_freq WHERE window_id BETWEEN 1 AND 4 GROUP BY window_id$q$,
            r.column_name,mean_expr,std_expr);
        DROP TABLE feature_freq;
        k := k+1;
        IF k % 20 = 0 THEN RAISE NOTICE 'Profiled % columns',k; END IF;
    END LOOP;
    CREATE TABLE duplicate_columns(feature_a text, feature_b text, identical boolean,
        comparison_scope text, PRIMARY KEY(feature_a,feature_b));
    FOR r IN
        SELECT p.feature AS a,q.feature AS b,p.data_type AS ta,q.data_type AS tb
        FROM profile p JOIN profile q ON p.scope=q.scope AND p.feature<q.feature
          AND p.fingerprint=q.fingerprint AND p.missing_count=q.missing_count
          AND p.unique_count=q.unique_count
        WHERE p.scope='model_train' AND p.feature NOT IN ('transactionid','isfraud','transactiondt')
          AND q.feature NOT IN ('transactionid','isfraud','transactiondt')
    LOOP
        IF r.ta=r.tb THEN
            EXECUTE format($q$INSERT INTO duplicate_columns
                SELECT %L,%L,NOT EXISTS(SELECT 1 FROM train_snapshot WHERE _audit_window BETWEEN 1 AND 4
                    AND %I IS DISTINCT FROM %I),'model_train'$q$,r.a,r.b,r.a,r.b);
        ELSE
            EXECUTE format($q$INSERT INTO duplicate_columns
                SELECT %L,%L,NOT EXISTS(SELECT 1 FROM train_snapshot WHERE _audit_window BETWEEN 1 AND 4
                    AND %I::text IS DISTINCT FROM %I::text),'model_train'$q$,r.a,r.b,r.a,r.b);
        END IF;
    END LOOP;
    ALTER TABLE run_metadata ADD COLUMN completed_at timestamptz;
    UPDATE run_metadata SET completed_at=clock_timestamp();
END $$;
COMMIT;
