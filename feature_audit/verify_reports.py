#!/usr/bin/env python3
"""Validate generated files and selected statistics against the immutable audit snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_audit import DB, qi, META, test_moments

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--user',default='macrisen');p.add_argument('--host');p.add_argument('--port',type=int)
    args=p.parse_args(); out=args.report_dir
    manifest=json.loads((out/'selection_manifest.json').read_text())
    args.database=manifest['database']; db=DB(args); a=qi(manifest['audit_schema'])
    test_moments()
    quality=pd.read_csv(out/'data_quality_report.csv').set_index('feature')
    selection=pd.read_csv(out/'feature_selection_report.csv').set_index('feature')
    full=pd.read_csv(out/'data_quality_all_training.csv').set_index('feature')
    pairs=pd.read_csv(out/'high_correlation_pairs.csv')
    checks=pd.read_csv(out/'correlation_validation.csv')
    temporal=pd.read_csv(out/'window_correlation_validation.csv')
    assert set(quality.index)==set(selection.index)==set(full.index)
    assert (quality.missing_count+quality.non_null_count==quality.total_rows).all()
    assert (quality.unique_count<=quality.non_null_count).all()
    assert set(selection.decision)<={'KEEP','REVIEW','DROP_CANDIDATE'}
    assert not set(manifest['proposed_drops'])&META
    assert set(manifest['proposed_drops'])==set(selection.query("decision == 'DROP_CANDIDATE'").index)
    assert (pairs.correlation.abs()>=.98).all()
    assert checks.absolute_error.max()<1e-8
    assert len(temporal)==4*len(pairs)
    assert np.allclose(quality.missing_percentage,quality.missing_count/quality.total_rows*100)
    for r in selection.itertuples():
        if r.decision=='DROP_CANDIDATE' and pd.notna(r.representative):
            assert selection.loc[r.representative,'decision']!='DROP_CANDIDATE'
    verified=[]
    for scope,where,reference in [('model_train','WHERE _audit_window BETWEEN 1 AND 4',quality),('all_train','',full)]:
        for feature in ['id_24','c3','transactionamt','deviceinfo','v1','isfraud']:
            if feature not in reference.index: continue
            col=qi(feature)
            row=db.frame(f'SELECT count(*) AS total, count(*) FILTER(WHERE {col} IS NULL) AS missing, count(DISTINCT {col}) AS uniques FROM {a}.train_snapshot {where}').iloc[0]
            expected=reference.loc[feature]
            assert int(row.total)==int(expected.total_rows)
            assert int(row.missing)==int(expected.missing_count)
            assert int(row.uniques)==int(expected.unique_count)
            verified.append({'scope':scope,'feature':feature,'total_rows':int(row.total),
                             'missing_count':int(row.missing),'unique_count':int(row.uniques)})
    for name,digest in manifest['script_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    table_counts=db.frame(f'SELECT (SELECT count(*) FROM {a}.train_snapshot) AS train_rows, (SELECT count(*) FROM {a}.test_snapshot) AS test_rows').iloc[0]
    assert int(table_counts.train_rows)==int(full.total_rows.iloc[0])
    integrity=pd.read_csv(out/'integrity.csv').set_index('table_name')
    assert int(table_counts.test_rows)==int(integrity.loc[manifest['test_table'],'row_count'])
    result={'status':'PASS','feature_count':len(selection),'verified_statistics':verified,
        'high_correlation_pairs':len(pairs),'postgres_correlation_checks':len(checks),
        'maximum_correlation_error':float(checks.absolute_error.max()),
        'sql_window_correlations':len(temporal),'source_train_rows':int(table_counts.train_rows),
        'source_test_rows':int(table_counts.test_rows),'source_script_hashes_match':True,
        'synthetic_pairwise_tests':'PASS','report_invariants':'PASS'}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='verified_statistics'},indent=2))

if __name__=='__main__':main()
