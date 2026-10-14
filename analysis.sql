-- Run in psql while connected to the database containing the four source tables.
-- Stop on errors; existing merge tables are not overwritten.
\set ON_ERROR_STOP on
\pset pager off

BEGIN;

-- Keep all transactions. USING includes transactionid only once.
CREATE TABLE public.train_merge AS
SELECT *
FROM public.train_transaction AS t
LEFT JOIN public.train_identity AS i USING (transactionid);

CREATE TABLE public.test_merge AS
SELECT *
FROM public.test_transaction AS t
LEFT JOIN public.test_identity AS i USING (transactionid);

COMMIT;

-- Verify column counts: train = 394 + 41 - 1; test = 393 + 41 - 1.
-- This checks the merged schema, not the number of matching identity records.
WITH expected(table_name, expected_columns) AS (
    VALUES ('train_merge', 434), ('test_merge', 433)
)
SELECT e.table_name,
       COUNT(c.column_name) AS actual_columns,
       e.expected_columns,
       COUNT(c.column_name) = e.expected_columns AS columns_match
FROM expected AS e
LEFT JOIN information_schema.columns AS c
    ON c.table_schema = 'public' AND c.table_name = e.table_name
GROUP BY e.table_name, e.expected_columns
ORDER BY e.table_name;
