#!/usr/bin/env python3
"""Prepare SQL; execute only with --apply-approved after explicit user approval."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import re
from run_audit import DB, qi, ql

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--approval-file',type=Path,required=True)
    p.add_argument('--output-schema',required=True)
    p.add_argument('--user',default='macrisen');p.add_argument('--host');p.add_argument('--port',type=int)
    p.add_argument('--apply-approved',action='store_true')
    p.add_argument('--parquet',action='store_true',help='Requires pyarrow; used only after --apply-approved')
    args=p.parse_args()
    manifest=json.loads((args.report_dir/'selection_manifest.json').read_text())
    for name in [args.output_schema,manifest['audit_schema']]:
        if not re.fullmatch(r'[a-z][a-z0-9_]*',name):
            raise ValueError('Audit/output schema names must use lowercase letters, digits and underscores')
    approval=json.loads(args.approval_file.read_text())
    drops=approval['approved_drops']
    if not isinstance(drops,list) or not all(isinstance(c,str) for c in drops):
        raise ValueError('approved_drops must be a list of column names')
    if len(set(drops))!=len(drops): raise ValueError('Duplicate approval entries')
    if not set(drops)<=set(manifest['proposed_drops']):
        raise ValueError('Only audited DROP_CANDIDATE names may be approved by this script')
    if set(drops)&set(manifest['protected_columns']): raise ValueError('Cannot remove protected metadata')
    # Prevent approving a representative while dropping another column that needs it.
    import csv
    with (args.report_dir/'feature_selection_report.csv').open() as f:
        selected=list(csv.DictReader(f))
    with (args.report_dir/'schema_comparison.csv').open() as f:
        if any(r['status']=='REVIEW' for r in csv.DictReader(f)):
            raise ValueError('Resolve train/test schema differences before applying selection')
    for r in selected:
        if r['feature'] in drops and r.get('representative') in drops:
            raise ValueError(f"Keep representative {r['representative']} for {r['feature']}")
    args.database=manifest['database']; db=DB(args)
    a=qi(manifest['audit_schema']); target=qi(args.output_schema)
    if args.output_schema in ('public',manifest['source_schema'],manifest['audit_schema']):
        raise ValueError('Choose a NEW output schema, not a source or audit schema')
    if args.parquet and args.apply_approved:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    traincols=[c for c in manifest['source_columns'][manifest['train_table']] if c not in drops]
    testcols=[c for c in manifest['source_columns'][manifest['test_table']] if c not in drops]
    if set(traincols)-{'isfraud'}!=set(testcols): raise ValueError('Train/test selected feature sets differ')
    cutoff=manifest['fit_cutoff']
    lines=['\\set ON_ERROR_STOP on','BEGIN ISOLATION LEVEL REPEATABLE READ;',
        f"DO $$ BEGIN IF NOT (SELECT passed FROM {a}.critical_gate) THEN RAISE EXCEPTION 'Audit integrity gate failed'; END IF; END $$;",
        f'CREATE SCHEMA {target};',
        f'CREATE TABLE {target}.train_clean AS SELECT '+','.join(map(qi,traincols))+f' FROM {a}.train_snapshot;',
        f'CREATE TABLE {target}.test_clean AS SELECT '+','.join(map(qi,testcols))+f' FROM {a}.test_snapshot;',
        f'ALTER TABLE {target}.train_clean ADD PRIMARY KEY (transactionid);',
        f'ALTER TABLE {target}.test_clean ADD PRIMARY KEY (transactionid);',
        f'ALTER TABLE {target}.train_clean ALTER COLUMN isfraud SET NOT NULL;',
        f'ALTER TABLE {target}.train_clean ADD CHECK (isfraud IN (0,1));',
        f'CREATE VIEW {target}.model_train AS SELECT * FROM {target}.train_clean WHERE transactiondt <= {float(cutoff)!r};',
        f'CREATE VIEW {target}.validation AS SELECT * FROM {target}.train_clean WHERE transactiondt > {float(cutoff)!r};',
        f'CREATE TABLE {target}.preprocessing_log(created_at timestamptz DEFAULT clock_timestamp(), operation text, details jsonb);',
        f'INSERT INTO {target}.preprocessing_log(operation,details) VALUES (\'approved_feature_selection\', '+ql(json.dumps({
            'audit_schema':manifest['audit_schema'],'approved_drops':drops,'cutoff':cutoff,
            'transforms':'Column selection only; preserve all rows, values, native NULLs, identifiers/time/target',
            'approval_file_sha256':hashlib.sha256(args.approval_file.read_bytes()).hexdigest()}))+'::jsonb);',
        f"DO $$ BEGIN IF (SELECT count(*) FROM {target}.train_clean) <> (SELECT count(*) FROM {a}.train_snapshot) OR (SELECT count(*) FROM {target}.test_clean) <> (SELECT count(*) FROM {a}.test_snapshot) THEN RAISE EXCEPTION 'Row counts changed'; END IF; END $$;",
        'COMMIT;']
    sqlpath=args.report_dir/'apply_approved.sql'
    sqlpath.write_text('\n'.join(lines)+'\n')
    print(f'Prepared {sqlpath}; {len(drops)} approved drops; {len(traincols)} train columns; {len(testcols)} test columns.')
    if not args.apply_approved:
        print('DRY RUN: no database changes. Review generated SQL before using --apply-approved.');return
    subprocess.run(db.base+['-f',str(sqlpath)],check=True)
    log={'applied_at':datetime.now(timezone.utc).isoformat(),'output_schema':args.output_schema,
         'approved_drops':drops,'sql_sha256':hashlib.sha256(sqlpath.read_bytes()).hexdigest(),
         'parquet':[]}
    if args.parquet:
        types={r['feature']:r['data_type'] for r in selected}
        for name in ('train_clean','test_clean'):
            csvpath=args.report_dir/(name+'.export.csv'); output=args.report_dir/(name+'.parquet')
            db.export(f'SELECT * FROM {target}.{qi(name)} ORDER BY transactiondt,transactionid',csvpath)
            selectedcols=traincols if name=='train_clean' else testcols
            arrow_types={c: pa.int64() if types[c] in ('smallint','integer','bigint') else
                         pa.float64() if types[c] in ('real','double precision') else
                         pa.decimal128(38,12) if types[c]=='numeric' else pa.string()
                         for c in selectedcols}
            reader=pacsv.open_csv(csvpath,convert_options=pacsv.ConvertOptions(column_types=arrow_types,
                strings_can_be_null=True,quoted_strings_can_be_null=False,null_values=['']))
            with pq.ParquetWriter(output,reader.schema,compression='zstd') as writer:
                for batch in reader: writer.write_batch(batch)
            csvpath.unlink();log['parquet'].append(str(output))
        log['parquet_numeric_representation']='SQL numeric -> decimal128(38,12); export fails if values cannot be represented exactly'
    (args.report_dir/'preprocessing_log.json').write_text(json.dumps(log,indent=2))
    print('Approved datasets created; original tables preserved.')

if __name__=='__main__':main()
