#!/usr/bin/env python3
"""Run PostgreSQL audit and generate results. No source mutations or feature removal."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(key, '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fraud_audit_matplotlib')
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact
from scipy.linalg.blas import dgemm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NUMERIC = {'smallint','integer','bigint','numeric','real','double precision'}
META = {'transactionid','transactiondt','isfraud'}

def qi(name):
    return '"' + name.replace('"', '""') + '"'

def ql(value):
    return "'" + str(value).replace("'", "''") + "'"

class DB:
    def __init__(self,args):
        self.base=['psql','-X','-v','ON_ERROR_STOP=1','-U',args.user,'-d',args.database]
        if args.host: self.base += ['-h',args.host]
        if args.port: self.base += ['-p',str(args.port)]
    def frame(self,query):
        result=subprocess.run(self.base+['--csv','-c',query],check=True,capture_output=True,text=True)
        return pd.read_csv(io.StringIO(result.stdout),keep_default_na=False,na_values=[''])
    def execute(self,query):
        return subprocess.run(self.base+['-c',query],check=True,capture_output=True,text=True).stdout
    def export(self,query,path):
        with path.open('w') as f:
            subprocess.run(self.base+['-c',f'COPY ({query}) TO STDOUT WITH (FORMAT CSV, HEADER true)'],stdout=f,check=True)

def group(name):
    import re
    for pattern,label in [(r'card[1-6]','card1–card6'),(r'c\d+','C1–C14'),
                          (r'd\d+','D1–D15'),(r'v\d+','V1–V339'),(r'id_\d+','id_01–id_38')]:
        if re.fullmatch(pattern,name): return label
    return 'Other transaction and identity'

def role(name,dtype):
    if name in META: return 'metadata_or_target'
    if name.startswith('card') or name in {'addr1','addr2'}: return 'categorical_code'
    if name.startswith('id_') and int(name[3:]) >= 12: return 'categorical_or_unverified_code'
    if dtype not in NUMERIC: return 'categorical_text'
    return 'numeric_measurement_or_anonymized_count'

class Moments:
    """Pairwise-complete Pearson sufficient statistics, never mean-imputes observations."""
    def __init__(self,p):
        self.n=np.zeros((p,p)); self.s=self.n.copy(); self.q=self.n.copy(); self.p=self.n.copy()
    def add(self,x):
        mask=np.isfinite(x).astype(np.float64)
        z=np.where(np.isfinite(x),x,0.0)
        # Explicit output prevents temporary-buffer reuse from mutating z on
        # this Python/NumPy build. Validate against independent PostgreSQL corr.
        z_squared=np.empty_like(z)
        np.multiply(z,z,out=z_squared)
        self.n += dgemm(1.0,mask,mask,trans_a=True)
        self.s += dgemm(1.0,z,mask,trans_a=True)
        self.q += dgemm(1.0,z_squared,mask,trans_a=True)
        self.p += dgemm(1.0,z,z,trans_a=True)
    def result(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            covariance=self.p-self.s*self.s.T/self.n
            mean_term=self.s*self.s/self.n
            variance=self.q-mean_term
            tolerance=1e-10*np.maximum(np.maximum(np.abs(self.q),np.abs(mean_term)),1)
            # Floating point cancellation can yield tiny negative variances.
            variance=np.maximum(variance,0)
            corr=covariance/np.sqrt(variance*variance.T)
        corr[(self.n<3)|(variance<=tolerance)|(variance.T<=tolerance.T)]=np.nan
        self.suspect_mask=(self.n>=3)&(((variance<=tolerance)&(self.q>0))|((variance.T<=tolerance.T)&(self.q.T>0)))
        if np.any(np.abs(corr[np.isfinite(corr)])>1+1e-8):
            raise ArithmeticError('Correlation outside [-1,1]; numerical validation required')
        return np.clip(corr,-1,1),np.rint(self.n).astype(np.int64)

def test_moments():
    rng=np.random.default_rng(34)
    x=rng.normal(size=(173,6)); x[:,1]=x[:,0]*3+2
    x[rng.random(x.shape)<0.24]=np.nan
    x[:,5]=1
    m=Moments(6)
    for chunk in np.array_split(x,7): m.add(chunk)
    c,n=m.result()
    for i in range(5):
        for j in range(i+1,5):
            valid=np.isfinite(x[:,i])&np.isfinite(x[:,j])
            assert n[i,j]==valid.sum()
            assert abs(c[i,j]-np.corrcoef(x[valid,i],x[valid,j])[0,1])<1e-12
    assert np.isnan(c[5]).all()
    # Larger Fortran-layout arrays exercise the temporary-buffer issue seen on Mac.
    large=np.asfortranarray(rng.normal(size=(10000,8)))
    large[rng.random(large.shape)<.35]=np.nan
    m=Moments(8); m.add(large)
    assert np.nanmax(np.abs(m.result()[0]-pd.DataFrame(large).corr().to_numpy()))<1e-10

def association(n1,f1,n0,f0):
    n1,f1,n0,f0=map(int,[n1,f1,n0,f0])
    if min(n1,n0)==0: return (np.nan,np.nan,np.nan,np.nan,False)
    rate1=f1/n1; rate0=f0/n0
    p=float(fisher_exact([[f1,n1-f1],[f0,n0-f0]]).pvalue)
    return (rate1,rate0,rate1-rate0,p,min(n1,n0)>=100 and min(f1,f0)>=5)

def bh_adjust(values):
    arr=np.array(values,dtype=float); out=np.full(len(arr),np.nan)
    idx=np.flatnonzero(np.isfinite(arr)); order=idx[np.argsort(arr[idx])]
    if len(order): out[order]=np.minimum(1,np.minimum.accumulate((arr[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return out

def write_csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.12g')

def log(message):
    print(datetime.now().strftime('%H:%M:%S'),message,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',default='fraud_detection')
    parser.add_argument('--user',default='macrisen')
    parser.add_argument('--host'); parser.add_argument('--port',type=int)
    parser.add_argument('--source-schema',default='public')
    parser.add_argument('--train-table',default='train_merge')
    parser.add_argument('--test-table',default='test_merge')
    parser.add_argument('--audit-schema',default='feature_audit_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--reuse-sql',action='store_true',help='Use an already completed SQL audit schema')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or HERE/'reports'/args.audit_schema
    out.mkdir(parents=True,exist_ok=True)
    db=DB(args); a=qi(args.audit_schema)
    test_moments()
    if not args.reuse_sql:
        log('Running SQL snapshot and column audit')
        subprocess.run(db.base+['-v',f'audit_schema={args.audit_schema}',
            '-v',f'source_schema={args.source_schema}','-v',f'train_table={args.train_table}',
            '-v',f'test_table={args.test_table}','-f',str(HERE/'audit.sql')],check=True)
    metadata=db.frame(f'SELECT * FROM {a}.run_metadata')
    actual=metadata.iloc[0]
    for key in ('source_schema','train_table','test_table'):
        if actual[key]!=getattr(args,key): raise ValueError(f'Snapshot {key} does not match configuration')
    exported={}
    for name in ['columns','integrity','critical_gate','split_definition','partition_summary','id_overlap',
                 'profile','window_profile','duplicate_columns','run_metadata']:
        exported[name]=db.frame(f'SELECT * FROM {a}.{qi(name)}')
        write_csv(exported[name],out/(name+'.csv'))
    profile=exported['profile']; windows=exported['window_profile']
    fit=profile[profile.scope=='model_train'].copy().set_index('feature')
    full=profile[profile.scope=='all_train'].copy().set_index('feature')
    fit['missing_percentage']=fit.missing_count/fit.total_rows*100
    fit['dominant_value_percentage']=fit.dominant_count/fit.total_rows*100
    fit['feature_group']=[group(c) for c in fit.index]
    fit['role']=[role(c,str(fit.loc[c,'data_type'])) for c in fit.index]
    windows['missing_percentage']=100*windows.missing_count/windows.total_rows
    fit['window_missing_range_pp']=windows.groupby('feature').missing_percentage.agg(lambda x:x.max()-x.min())
    fit['window_mean_range_sd']=(windows.groupby('feature').numeric_mean.max()-windows.groupby('feature').numeric_mean.min())/fit.numeric_std.replace(0,np.nan)
    associations=[]
    for feature,r in fit.iterrows():
        missing=association(r.missing_count,r.missing_fraud,r.non_null_count,r.total_fraud-r.missing_fraud)
        rare_n=r.non_null_count-r.dominant_count
        rare_f=r.total_fraud-r.missing_fraud-r.dominant_fraud
        rare=association(rare_n,rare_f,r.dominant_count,r.dominant_fraud)
        row={'feature':feature,'rare_non_null_rows':rare_n,'rare_non_null_fraud':rare_f}
        for prefix,values in [('missing',missing),('rare',rare)]:
            for key,value in zip(['group_fraud_rate','reference_fraud_rate','rate_difference','p_value','adequate_support'],values): row[prefix+'_'+key]=value
        associations.append(row)
    assoc=pd.DataFrame(associations).set_index('feature')
    for prefix in ['missing','rare']:
        assoc[prefix+'_q_value']=np.nan
        eligible_mask=~assoc.index.isin(META)
        assoc.loc[eligible_mask,prefix+'_q_value']=bh_adjust(assoc.loc[eligible_mask,prefix+'_p_value'])
        assoc[prefix+'_informative']=(assoc[prefix+'_q_value']<.01)&(assoc[prefix+'_rate_difference'].abs()>=.005)&assoc[prefix+'_adequate_support']
    # Missingness association must retain direction in >=3 supported fit windows.
    stable_missing={}
    for feature,w in windows.groupby('feature'):
        sign=np.sign(assoc.loc[feature,'missing_rate_difference']); supported=0; same=0
        for row in w.itertuples():
            z=association(row.missing_count,row.missing_fraud,row.total_rows-row.missing_count,row.total_fraud-row.missing_fraud)
            if z[4]:
                supported+=1; same+=int(np.sign(z[2])==sign and abs(z[2])>=.0025)
        stable_missing[feature]=(supported,same)
    assoc['missing_supported_windows']=[stable_missing[c][0] for c in assoc.index]
    assoc['missing_same_direction_windows']=[stable_missing[c][1] for c in assoc.index]
    fit=fit.join(assoc)
    write_csv(assoc.reset_index(),out/'fraud_associations.csv')
    write_csv(windows,out/'chronological_feature_stability.csv')

    names=[c for c in fit.index if fit.loc[c,'role']=='numeric_measurement_or_anonymized_count'
           and fit.loc[c,'unique_count']>1 and fit.loc[c,'nonfinite_count']==0]
    log(f'Computing pairwise-complete correlations for {len(names)} numeric features')
    numeric_path=out/'_numeric_fit.csv'
    select=', '.join(qi(c) for c in names)
    db.export(f'SELECT _audit_window,{select} FROM {a}.train_snapshot WHERE _audit_window BETWEEN 1 AND 4',numeric_path)
    moments=[Moments(len(names)) for _ in range(4)]
    means=fit.loc[names,'numeric_mean'].to_numpy(float)
    scales=fit.loc[names,'numeric_std'].to_numpy(float)
    scales[~np.isfinite(scales)|(scales==0)]=1
    processed=0
    for chunk in pd.read_csv(numeric_path,chunksize=8192,dtype=np.float64):
        x=(chunk[names].to_numpy()-means)/scales
        for w in range(1,5):
            part=x[chunk._audit_window.to_numpy()==w]
            if len(part): moments[w-1].add(part)
        processed+=len(chunk)
        if processed%81920==0: log(f'Correlation accumulation: {processed:,} rows')
    total=Moments(len(names))
    for m in moments:
        for key in ['n','s','q','p']: setattr(total,key,getattr(total,key)+getattr(m,key))
    corr,counts=total.result()
    suspect=np.argwhere(np.triu(total.suspect_mask,1))
    log(f'Checking {len(suspect):,} low-variance pairs directly in PostgreSQL')
    numerical_checks=[]
    for start in range(0,len(suspect),100):
        batch=suspect[start:start+100]; expr=[]
        for k,(i,j) in enumerate(batch):
            expr += [f'corr({qi(names[i])}::double precision,{qi(names[j])}::double precision) AS r{k}',
                     f'regr_count({qi(names[i])}::double precision,{qi(names[j])}::double precision) AS n{k}']
        row=db.frame(f'SELECT {", ".join(expr)} FROM {a}.train_snapshot WHERE _audit_window BETWEEN 1 AND 4').iloc[0]
        for k,(i,j) in enumerate(batch):
            assert int(row[f'n{k}'])==counts[i,j]
            value=float(row[f'r{k}']); corr[i,j]=corr[j,i]=value
            numerical_checks.append({'feature_a':names[i],'feature_b':names[j],
                'postgres_correlation':value,'shared_observations':counts[i,j],
                'method':'Direct PostgreSQL recomputation for negligible pairwise variance'})
        if start%500==0: log(f'Low-variance SQL checks: {min(start+100,len(suspect)):,}/{len(suspect):,}')
    write_csv(pd.DataFrame(numerical_checks,columns=['feature_a','feature_b','postgres_correlation','shared_observations','method']),out/'low_variance_correlation_checks.csv')
    np.savez_compressed(out/'correlation_matrices.npz',features=np.array(names),correlation=corr,paired_counts=counts)
    numeric_path.unlink()
    nfit=int(fit.total_rows.iloc[0]); min_pairs=max(1000,math.ceil(.01*nfit))
    allpairs=[]; high=[]
    for i,ca in enumerate(names):
        for j in range(i+1,len(names)):
            cb=names[j]; r=corr[i,j]; n=int(counts[i,j])
            if not np.isfinite(r): continue
            allpairs.append((ca,cb,float(r),n))
            if abs(r) < .98: continue
            reliable=n>=min_pairs
            window_corr=[np.nan]*4
            window_n=[0]*4
            stable=False  # Filled by PostgreSQL window aggregates below.
            row={'feature_a':ca,'feature_b':cb,'correlation':r,'shared_observations':n,
                 'minimum_reliable_pairs':min_pairs,'reliable':reliable,'stable_across_fit_windows':stable}
            for w,(rw,nw) in enumerate(zip(window_corr,window_n),1):
                row[f'window_{w}_correlation']=rw; row[f'window_{w}_paired_rows']=nw
            high.append(row)
    pairs=pd.DataFrame(high,columns=['feature_a','feature_b','correlation','shared_observations',
        'minimum_reliable_pairs','reliable','stable_across_fit_windows']+
        [v for w in range(1,5) for v in (f'window_{w}_correlation',f'window_{w}_paired_rows')])
    # Direct PostgreSQL correlations avoid cancellation for nearly constant
    # within-window observations; only flagged pairs need this extra calculation.
    temporal_checks=[]
    for start in range(0,len(pairs),100):
        batch=pairs.iloc[start:start+100]
        expr=[]
        for k,r in batch.iterrows():
            expr += [f'corr({qi(r.feature_a)}::double precision,{qi(r.feature_b)}::double precision) AS r{k}',
                     f'regr_count({qi(r.feature_a)}::double precision,{qi(r.feature_b)}::double precision) AS n{k}']
        check=db.frame(f'SELECT _audit_window,{", ".join(expr)} FROM {a}.train_snapshot WHERE _audit_window BETWEEN 1 AND 4 GROUP BY _audit_window')
        for _,row in check.iterrows():
            w=int(row['_audit_window'])
            for k,r in batch.iterrows():
                pairs.at[k,f'window_{w}_correlation']=row[f'r{k}']
                pairs.at[k,f'window_{w}_paired_rows']=int(row[f'n{k}'])
                temporal_checks.append({'window':w,'feature_a':r.feature_a,'feature_b':r.feature_b,
                    'postgres_correlation':row[f'r{k}'],'shared_observations':int(row[f'n{k}']),
                    'method':'PostgreSQL corr/regr_count directly on audit snapshot'})
    for k,r in pairs.iterrows():
        pairs.at[k,'stable_across_fit_windows']=all(r[f'window_{w}_paired_rows']>=200 and
            pd.notna(r[f'window_{w}_correlation']) and abs(r[f'window_{w}_correlation'])>=.95 and
            np.sign(r[f'window_{w}_correlation'])==np.sign(r.correlation) for w in range(1,5))
    write_csv(pd.DataFrame(temporal_checks),out/'window_correlation_validation.csv')
    allpairs=pd.DataFrame(allpairs,columns=['feature_a','feature_b','correlation','shared_observations'])
    write_csv(allpairs,out/'all_numeric_correlations.csv')
    # Independent PostgreSQL oracle for selected high/ordinary correlations.
    oracle_pairs=pd.concat([pairs[['feature_a','feature_b']],
        allpairs[['feature_a','feature_b']].sample(min(16,len(allpairs)),random_state=20261014)]).drop_duplicates()
    if 'v11' in names and 'v42' in names:
        oracle_pairs=pd.concat([oracle_pairs,pd.DataFrame([{'feature_a':'v11','feature_b':'v42'}])]).drop_duplicates()
    expr=[]
    for k,r in enumerate(oracle_pairs.itertuples(index=False)):
        expr += [f'corr({qi(r.feature_a)}::double precision,{qi(r.feature_b)}::double precision) AS r{k}',
                 f'regr_count({qi(r.feature_a)}::double precision,{qi(r.feature_b)}::double precision) AS n{k}']
    oracle=[]
    if expr:
        result=db.frame(f'SELECT {", ".join(expr)} FROM {a}.train_snapshot WHERE _audit_window BETWEEN 1 AND 4').iloc[0]
        for k,row in enumerate(oracle_pairs.itertuples(index=False)):
            i=names.index(row.feature_a); j=names.index(row.feature_b)
            sql_r=float(result[f'r{k}']); py_r=corr[i,j]
            both_undefined=np.isnan(sql_r) and np.isnan(py_r)
            error=abs(sql_r-py_r) if not both_undefined else 0.0
            assert (both_undefined or error<1e-8) and int(result[f'n{k}'])==counts[i,j], f'SQL correlation validation failed: {row.feature_a}/{row.feature_b}: SQL={sql_r}, Python={py_r}, SQL_n={result[f"n{k}"]}, Python_n={counts[i,j]}'
            oracle.append({'feature_a':row.feature_a,'feature_b':row.feature_b,'python_correlation':corr[i,j],
                           'postgres_correlation':result[f'r{k}'],'absolute_error':error,'shared_observations':int(counts[i,j])})
    write_csv(pd.DataFrame(oracle),out/'correlation_validation.csv')
    log(f'Found {len(pairs):,} high-correlation pairs; verified {len(oracle)} against PostgreSQL')
    build_reports(args,db,out,exported,fit,full,pairs,allpairs,corr,names,min_pairs)


def build_reports(args,db,out,raw,fit,full,pairs,allpairs,corr,names,min_pairs):
    reasons={c:[] for c in fit.index}; rules={c:[] for c in fit.index}
    decision={c:'KEEP' for c in fit.index}
    for c,r in fit.iterrows():
        if c in META:
            reasons[c].append('Preserve as target/technical/time metadata; excluded from predictive inputs by default')
            rules[c].append('protected_metadata'); continue
        if r.nonfinite_count>0:
            decision[c]='REVIEW'; rules[c].append('nonfinite_numeric')
            reasons[c].append('NaN/infinite numeric values require an explicit handling rule')
        elif r.unique_count==0:
            decision[c]='DROP_CANDIDATE'; rules[c].append('all_null')
            reasons[c].append('All NULL in model-training partition; no observed training variation')
        elif r.unique_count==1:
            rules[c].append('constant_non_null')
            if r.missing_count==0:
                decision[c]='DROP_CANDIDATE'; reasons[c].append('One value in every model-training row')
            else:
                decision[c]='REVIEW'; reasons[c].append('One non-NULL value, but missingness creates a potentially informative second state')
        elif r.dominant_value_percentage>99.5:
            decision[c]='REVIEW'; rules[c].append('near_constant')
            reasons[c].append('Dominant non-NULL value exceeds 99.5% of all fitting rows; retain pending rare-value evaluation')
        if r.missing_percentage>90 and r.unique_count>0:
            rules[c].append('missing_over_90pct')
            if r.missing_informative or not r.missing_adequate_support:
                decision[c]='REVIEW'
                reasons[c].append('Missingness is fraud-associated or evidence is too sparse to dismiss it')
            elif decision[c]=='KEEP':
                decision[c]='DROP_CANDIDATE'
                reasons[c].append('Over 90% missing; missingness did not meet the prespecified association screen (not proof of no signal)')
        if r.rare_informative and (r.dominant_value_percentage>99.5 or r.missing_percentage>90):
            decision[c]='REVIEW'; rules[c].append('rare_value_fraud_signal')
            reasons[c].append('Rare non-NULL values show a fraud-rate association')
        if not rules[c]:
            rules[c].append('no_removal_rule'); reasons[c].append('No evaluated removal rule triggered')

    def quality(c):
        r=fit.loc[c]
        return (decision[c]=='DROP_CANDIDATE',float(r.missing_percentage),
                float(r.window_missing_range_pp),
                float(r.window_mean_range_sd) if pd.notna(r.window_mean_range_sd) else 0,c)

    pairs['missing_pattern_disagreements']=[int(fit.loc[r.feature_a,'non_null_count']+
        fit.loc[r.feature_b,'non_null_count']-2*r.shared_observations) for r in pairs.itertuples()]
    pairs['recommended_feature_to_remove']=['' for _ in range(len(pairs))]
    pairs['suggested_retain']=['' for _ in range(len(pairs))]
    pairs['explanation']=['' for _ in range(len(pairs))]
    safe_graph={c:set() for c in fit.index if c not in META}
    all_graph={c:set() for c in fit.index if c not in META}
    for idx,r in pairs.iterrows():
        ca,cb=r.feature_a,r.feature_b; keep,drop=sorted([ca,cb],key=quality)
        all_graph[ca].add(cb); all_graph[cb].add(ca)
        pairs.at[idx,'suggested_retain']=keep
        if not r.reliable:
            reason=f'REVIEW: fewer than {min_pairs} shared observations; no removal recommendation'
        elif not r.stable_across_fit_windows:
            reason='REVIEW: correlation lacks sufficient support or abs(r)>=0.95 with consistent sign in all four fitting windows'
        elif r.missing_pattern_disagreements>0:
            reason='REVIEW: different NULL patterns may contain distinct fraud information; no removal recommendation'
        else:
            safe_graph[ca].add(cb); safe_graph[cb].add(ca)
            reason='Eligible redundancy edge: stable correlation and identical NULL pattern; global representative selection decides removal'
        pairs.at[idx,'explanation']=reason
        if reason.startswith('REVIEW'):
            for c in [ca,cb]:
                if decision[c]=='KEEP':
                    decision[c]='REVIEW'; rules[c].append('correlation_requires_review')
                    reasons[c].append('High correlation needs further review: insufficient temporal/support evidence or different missingness')

    duplicates=raw['duplicate_columns'].copy()
    verified=duplicates[duplicates.identical.astype(str).str.lower().isin(['t','true'])]
    for r in verified.itertuples():
        safe_graph[r.feature_a].add(r.feature_b); safe_graph[r.feature_b].add(r.feature_a)
        all_graph[r.feature_a].add(r.feature_b); all_graph[r.feature_b].add(r.feature_a)

    # Greedy retained representatives: each proposed redundant drop has a DIRECT edge
    # to an actual retained feature. A-B-C chains never imply A-C equivalence.
    retained=[]; representative={}
    for c in sorted(safe_graph,key=quality):
        candidates=[k for k in retained if c in safe_graph[k]]
        if candidates:
            representative[c]=min(candidates,key=quality)
        else: retained.append(c)
    exact_edges={frozenset([r.feature_a,r.feature_b]) for r in verified.itertuples()}
    for drop,keep in representative.items():
        if decision[keep]=='DROP_CANDIDATE':
            decision[keep]='KEEP'; rules[keep].append('retained_redundancy_representative')
            reasons[keep].append('Retained to preserve a representative for its redundant features')
        if fit.loc[drop,'nonfinite_count']>0: continue
        is_exact=frozenset([drop,keep]) in exact_edges
        if not is_exact and any(r in rules[drop] for r in ['rare_value_fraud_signal','near_constant','missing_over_90pct']):
            continue
        decision[drop]='DROP_CANDIDATE'
        rules[drop].append('exact_duplicate' if is_exact else 'stable_high_correlation')
        reasons[drop].append(('Exact row-aligned duplicate of ' if is_exact else 'Stable highly correlated feature represented by ')+keep+
            '; keep representative. Importance not evaluated; this remains a proposal')
    for idx,r in pairs.iterrows():
        if representative.get(r.feature_a)==r.feature_b:
            drop,keep=r.feature_a,r.feature_b
        elif representative.get(r.feature_b)==r.feature_a:
            drop,keep=r.feature_b,r.feature_a
        else: continue
        if decision[drop]=='DROP_CANDIDATE':
            pairs.at[idx,'recommended_feature_to_remove']=drop
            pairs.at[idx,'suggested_retain']=keep
            pairs.at[idx,'explanation']+='; propose '+drop+'; retain '+keep+' (lower missingness, then temporal availability/stability, then lexical tie-break; no model importance fitted)'

    # Connected components are descriptive only; selection uses direct safe edges.
    components=[]; seen=set()
    for start in sorted(all_graph):
        if start in seen or not all_graph[start]: continue
        stack=[start]; members=set()
        while stack:
            c=stack.pop()
            if c in members: continue
            members.add(c); stack.extend(all_graph[c]-members)
        seen.update(members)
        components.append({'group_id':len(components)+1,'size':len(members),
            'features':';'.join(sorted(members)),
            'retained_or_review':';'.join(sorted(c for c in members if decision[c]!='DROP_CANDIDATE')),
            'proposed_drops':';'.join(sorted(c for c in members if decision[c]=='DROP_CANDIDATE')),
            'strategy':'Greedy direct-edge representatives; no transitive correlation assumption'})
    components=pd.DataFrame(components,columns=['group_id','size','features','retained_or_review','proposed_drops','strategy'])
    write_csv(components,out/'redundant_feature_groups.csv')
    write_csv(pairs.sort_values('correlation',key=lambda s:s.abs(),ascending=False),out/'high_correlation_pairs.csv')
    if len(duplicates):
        duplicates['recommended_feature_to_remove']=[r.feature_a if representative.get(r.feature_a)==r.feature_b else r.feature_b if representative.get(r.feature_b)==r.feature_a else '' for r in duplicates.itertuples()]
    write_csv(duplicates,out/'duplicate_features.csv')
    fit['max_correlation']=np.nan
    for c in names:
        i=names.index(c); vals=np.delete(np.abs(corr[i]),i)
        if np.isfinite(vals).any(): fit.loc[c,'max_correlation']=np.nanmax(vals)
    fit['decision']=pd.Series(decision)
    for c in fit.index:
        if len(rules[c])>1 and 'no_removal_rule' in rules[c]:
            rules[c].remove('no_removal_rule')
            reasons[c]=[r for r in reasons[c] if r!='No evaluated removal rule triggered']
    fit['reason']=pd.Series({c:'; '.join(dict.fromkeys(reasons[c])) for c in fit.index})
    fit['selection_rule']=pd.Series({c:';'.join(dict.fromkeys(rules[c])) for c in fit.index})
    fit['representative']=pd.Series(representative)
    fit['temporal_availability']='Not verifiable from anonymized CSV; confirm availability at transaction decision time'
    fit.loc[list(META),'temporal_availability']='Metadata/target; not a predictor'
    fit['predictive_importance']='NOT_EVALUATED'
    master=fit.reset_index()
    write_csv(master,out/'feature_selection_report.csv')
    data_quality=master[['feature','data_type','scope','total_rows','missing_count','missing_percentage',
                        'non_null_count','unique_count','dominant_value','dominant_count',
                        'dominant_value_percentage','decision']].rename(columns={'decision':'recommended_status'})
    write_csv(data_quality.sort_values('missing_percentage',ascending=False),out/'data_quality_report.csv')
    full['missing_percentage']=full.missing_count/full.total_rows*100
    full['dominant_value_percentage']=full.dominant_count/full.total_rows*100
    full['recommended_status']=fit.decision
    full['recommendation_source']='model_train only; all_train statistics are descriptive'
    write_csv(full.reset_index().sort_values('missing_percentage',ascending=False),out/'data_quality_all_training.csv')
    const=master[(master.unique_count<=1)|(master.dominant_value_percentage>99.5)].copy()
    const['constant_type']=np.select([const.unique_count==0,const.unique_count==1],['ALL_NULL','CONSTANT_NON_NULL'],default='NEAR_CONSTANT')
    const['recommended_action']=const.decision
    write_csv(const,out/'constant_features.csv')
    bygroup=master[~master.feature.isin(META)].groupby(['feature_group','decision']).size().unstack(fill_value=0)
    write_csv(bygroup.reset_index(),out/'feature_group_summary.csv')
    cols=raw['columns']; tr=cols[cols.table_name==args.train_table]; te=cols[cols.table_name==args.test_table]
    schema=tr[['column_name','data_type']].merge(te[['column_name','data_type']],on='column_name',how='outer',suffixes=('_train','_test'))
    schema['status']=np.where(schema.column_name=='isfraud','EXPECTED_TRAIN_ONLY_TARGET',
        np.where(schema.data_type_train==schema.data_type_test,'MATCH','REVIEW'))
    write_csv(schema,out/'schema_comparison.csv')
    eligible=master[~master.feature.isin(META)]
    drop=eligible[eligible.decision=='DROP_CANDIDATE']
    write_csv(drop[['feature','reason','selection_rule','representative']],out/'proposed_drops.csv')
    manifest={'audit_schema':args.audit_schema,'source_schema':args.source_schema,'train_table':args.train_table,
        'test_table':args.test_table,'database':args.database,'fit_cutoff':float(raw['split_definition'].cutoff.iloc[0]),
        'protected_columns':sorted(META),'proposed_drops':sorted(drop.feature.tolist()),
        'retained_or_review_features':sorted(eligible[eligible.decision!='DROP_CANDIDATE'].feature.tolist()),
        'approval_status':'PENDING_USER_REVIEW','numeric_imputation':'none','scaling':'none',
        'categorical_encoding':'deferred; fit category mappings on model-training partition only',
        'source_columns':{args.train_table:tr.sort_values('ordinal_position').column_name.tolist(),
                          args.test_table:te.sort_values('ordinal_position').column_name.tolist()},
        'thresholds':{'missing_pct':90,'near_constant_pct':99.5,'abs_correlation':.98,
          'min_pair_count':min_pairs,'window_abs_correlation':.95,'window_min_pairs':200,
          'association_bh_q':.01,'association_min_rate_difference':.005,
          'association_min_group_rows':100,'association_min_group_fraud':5},
        'sql_snapshot':str(raw['run_metadata'].database_snapshot.iloc[0]),
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'audit.sql',HERE/'run_audit.py']}}
    (out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Empty approval template: no candidates are pre-approved.
    (out/'approved_drops.json').write_text(json.dumps({'approved_drops':[]},indent=2))
    figures(out,fit,corr,names)
    make_summary(out,args,raw,fit,full,pairs,const,components,bygroup,schema,manifest)
    assert set(master.feature)==set(tr.column_name)
    assert not (set(drop.feature)&META)
    assert all(decision[keep]!='DROP_CANDIDATE' for d,keep in representative.items() if decision[d]=='DROP_CANDIDATE')
    assert int(master.total_rows.iloc[0])==int(raw['partition_summary'].query('window_id >= 1 and window_id <= 4').rows.sum())
    log(f'Reports complete: {out}')
    print(master.decision.value_counts().to_string(),flush=True)

def figures(out,fit,corr,names):
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    predictors=fit.drop(index=list(META))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].hist(predictors.missing_percentage,bins=np.arange(0,105,5),color='#3579a8',edgecolor='white')
    axes[0].axvline(90,color='#b53d3d',linestyle='--'); axes[0].set(xlabel='Missing values (%)',ylabel='Feature count',title='Missingness — model-training only')
    counts=predictors.decision.value_counts().reindex(['KEEP','REVIEW','DROP_CANDIDATE'],fill_value=0)
    axes[1].bar(counts.index,counts.values,color=['#34846c','#d49d36','#b85858'])
    axes[1].set(ylabel='Feature count',title='Proposals, pending approval')
    for i,n in enumerate(counts): axes[1].text(i,n+1,str(n),ha='center')
    fig.savefig(out/'audit_overview.png'); plt.close(fig)
    top=predictors.nlargest(20,'missing_percentage').sort_values('missing_percentage')
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained'); ax.barh(top.index,top.missing_percentage,color='#3579a8')
    ax.axvline(90,color='#b53d3d',linestyle='--'); ax.set(xlim=(0,105),xlabel='Missing values (%)',title='20 most incomplete features — model-training only')
    fig.savefig(out/'top_missing_features.png'); plt.close(fig)
    if len(names):
        scores=np.sum(np.abs(corr)>=.98,axis=1)
        chosen=np.argsort(-scores,kind='stable')[:min(24,len(names))]
        fig,ax=plt.subplots(figsize=(10,9),layout='constrained')
        im=ax.imshow(corr[np.ix_(chosen,chosen)],vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(chosen)),[names[i] for i in chosen],rotation=90)
        ax.set_yticks(range(len(chosen)),[names[i] for i in chosen])
        ax.set_title('Pairwise-complete Pearson correlation\n24 features with most high-correlation links; model-training only')
        fig.colorbar(im,ax=ax,label='Pearson r',shrink=.8); fig.savefig(out/'correlation_heatmap.png'); plt.close(fig)

def make_summary(out,args,raw,fit,full,pairs,const,components,bygroup,schema,manifest):
    eligible=fit.drop(index=list(META)); nfit=int(fit.total_rows.iloc[0]); nfull=int(full.total_rows.iloc[0])
    val=int(raw['partition_summary'].query('window_id == 5').rows.sum())
    drop=eligible[eligible.decision=='DROP_CANDIDATE']; review=eligible[eligible.decision=='REVIEW']; keep=eligible[eligible.decision=='KEEP']
    ident=raw['integrity']; test_rows=int(ident.loc[ident.table_name==args.test_table,'row_count'].iloc[0])
    lines=['# IEEE-CIS preprocessing audit','',
        f'Generated from PostgreSQL database `{args.database}`, source `{args.source_schema}.{args.train_table}` and `{args.source_schema}.{args.test_table}`.',
        f'Audit schema: `{args.audit_schema}`. **Executed; no feature removal or final cleaned datasets created. Approval pending.**','',
        '## Dataset and chronology','',
        f'- Labeled training: **{nfull:,} rows**, **{len(fit)} columns**, including 3 protected metadata/target columns and **{len(eligible)} eligible predictors**.',
        f'- Unlabeled test: **{test_rows:,} rows**, **{len(schema[schema.data_type_test.notna()])} columns**.',
        f'- Earlier model-training partition: **{nfit:,} rows**; later validation: **{val:,} rows**.',
        f'- Split cutoff: `transactiondt <= {manifest["fit_cutoff"]:.0f}` for fitting, greater for validation. The requested 80% boundary uses `percentile_disc`; timestamp ties stay together.',
        '- Four approximately equal chronological windows within the fitting partition assess stability. No validation or test values fitted selection rules.',
        f'- Fraud cases in fitting: **{int(raw["partition_summary"].query("window_id >= 1 and window_id <= 4").fraud_rows.sum()):,}**; validation: **{int(raw["partition_summary"].query("window_id == 5").fraud_rows.sum()):,}**.',
        '- Full-training profiles are descriptive only (`data_quality_all_training.csv`). The required primary reports and all decisions use the model-training partition.',
        '- `transactionid` and `isfraud` are never predictors; `transactiondt` is retained for ordering/evaluation and excluded from the default feature list.','',
        '## Calculated findings','',
        '| Finding | Model-training result |','|---|---:|',
        f'| Predictors with missing rate >90% | {int((eligible.missing_percentage>90).sum())} |',
        f'| All-NULL predictors | {int((eligible.unique_count==0).sum())} |',
        f'| Constant non-NULL predictors (may also be near-constant) | {int((eligible.unique_count==1).sum())} |',
        f'| Near-constant predictors (>99.5% of all rows; includes constants) | {int((eligible.dominant_value_percentage>99.5).sum())} |',
        f'| High-correlation numeric pairs, abs(r)>=0.98 | {len(pairs)} |',
        f'| High-correlation pairs below minimum support | {int((~pairs.reliable).sum())} |',
        f'| High-correlation pairs unstable/unsupported across windows | {int((~pairs.stable_across_fit_windows).sum())} |',
        f'| Exact duplicate feature pairs | {int(raw["duplicate_columns"].identical.astype(str).str.lower().isin(["t","true"]).sum())} |',
        f'| Connected redundancy groups | {len(components)} |',
        f'| KEEP predictors | {len(keep)} |',f'| REVIEW predictors | {len(review)} |',
        f'| DROP_CANDIDATE predictors | {len(drop)} |',
        f'| Remaining predictors if ALL proposed drops approved | {len(eligible)-len(drop)} |',
        f'| Remaining train columns including metadata/target | {len(fit)-len(drop)} |',
        f'| Remaining test columns including ID/time | {len(fit)-1-len(drop)} |','',
        f'Full labeled-training predictors above 90% missing (descriptive, not fitted): **{int((full.drop(index=list(META)).missing_percentage>90).sum())}**.','',
        '![Missingness distribution and proposed decisions](audit_overview.png)','',
        '![Top missing features](top_missing_features.png)','',
        '![Selected numeric correlation heatmap](correlation_heatmap.png)','',
        '## Methods and safeguards','',
        '- SQL inspects metadata and profiles every column. NULL counts, distinct non-NULL values, dominant non-NULL values, labels, numeric NaN/infinities, window profiles and duplicate-column candidates are calculated in PostgreSQL.',
        '- Dominant percentage uses **all rows** as denominator and a non-NULL value as numerator; NULL dominance is reported separately as missingness. All-NULL, constant, and near-constant conditions may overlap except all-NULL; do not sum overlapping counts.',
        '- High missingness is a drop candidate only when the conservative fraud-association screen does not require review. Missingness versus presence and rare non-NULL versus dominant values use Fisher exact tests with Benjamini–Hochberg adjustment within each test family, q<0.01 and absolute fraud-rate difference >=0.5 percentage points. At least 100 rows and 5 fraud cases in each group are required for supported evidence. These are exploratory screens, not causal or predictive proofs.',
        '- Missingness association direction is also reported across fitting windows. Insufficient evidence never proves irrelevance. Near-constant rare values remain REVIEW unless an exact/stable redundant representative preserves their signal.',
        '- Pearson includes numeric measurements/anonymized counts only. All card fields, address codes, text fields and id_12–id_38 are conservatively treated as categorical/unverified codes. Low integer cardinality alone is not enough to call a V/C/D measurement categorical; semantic assignments require domain review.',
        f'- Correlations use all pairwise finite/non-NULL fitting observations, no sampling or imputation. Reliable support requires at least **{manifest["thresholds"]["min_pair_count"]:,}** paired rows (max of 1,000 and 1% of fitting rows). Proposed correlation removal also requires all four windows to have >=200 shared rows, abs(r)>=0.95, matching sign, and identical NULL patterns.',
        '- Chunked, centered/scaled sufficient statistics improve numerical conditioning only; no stored data is scaled or imputed. Selected correlations and shared counts are independently verified against PostgreSQL `corr` and `regr_count` (`correlation_validation.csv`). Zero-variance pairs have undefined correlation and are excluded.',
        '- Exact duplicate detection fingerprints row-aligned values INCLUDING NULL pattern, then confirms candidate equality directly in SQL. Both numeric and categorical fields are considered. No transactions are deduplicated by feature similarity.',
        '- Group selection is deterministic: lower missingness, then smaller chronological missingness variation, smaller mean variation relative to standard deviation, then lexical tie-break. Each redundant drop points directly to a retained representative; correlation is not treated as transitive. Exact duplicates need no model importance to establish equivalence within the fitting data.',
        '- Predictive importance and LightGBM experiments A/B/C were **not executed**. Optional model comparison is deferred; there is no claim that a proposed drop improves PR-AUC or preserves predictive performance.',
        '- A candidate can be REVIEW for sparse/unstable pair evidence; master decisions are the authoritative combined recommendations. Pair suggestions never independently authorize a deletion.','',
        '## Integrity and leakage','',
        '| Table | Rows | Missing IDs | Duplicate ID excess | Missing time | Missing labels | Invalid labels | Missing amount | Negative amount |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ident.itertuples():
        fmt=lambda v:'N/A' if pd.isna(v) else str(int(v))
        lines.append('| '+ ' | '.join([r.table_name]+[fmt(getattr(r,c)) for c in ['row_count','missing_ids','duplicate_id_excess','missing_time','missing_labels','invalid_labels','missing_amount','negative_amount']])+' |')
    lines += ['',f'- Train/test overlapping transaction IDs: **{int(raw["id_overlap"].overlapping_ids.iloc[0])}**.',
        f'- Numeric NaN/infinity cells in model-training: **{int(fit.nonfinite_count.sum())}**; full labeled training: **{int(full.nonfinite_count.sum())}**.',
        f'- Train/test schema differences requiring review: **{int((schema.status=="REVIEW").sum())}**. The absent test label is expected.',
        '- Native PostgreSQL numeric columns cannot contain arbitrary invalid text; failed CSV imports or meaning-specific sentinels cannot be reconstructed from these tables. Numeric category codes are not automatically continuous features.',
        '- Transaction-time ordering is checked, but the anonymized C/D/V/id features have unverified acquisition times. Unknown meaning is NOT evidence of leakage. Confirm availability at decision time; do not construct aggregates from future records or labels.',
        '- Identity records can be absent legitimately. Preserve such transactions and native numeric NULLs. Test labels are unavailable, so test cannot supply labeled RL rewards or measured fraud performance.',
        '- Static historical CSVs do not identify counterfactual action outcomes. RL reward design and environment simulation remain separate research tasks.','',
        '## Feature groups','', '| Group | KEEP | REVIEW | DROP_CANDIDATE |','|---|---:|---:|---:|']
    for name,row in bygroup.iterrows(): lines.append(f'| {name} | {int(row.get("KEEP",0))} | {int(row.get("REVIEW",0))} | {int(row.get("DROP_CANDIDATE",0))} |')
    lines += ['', 'No entire group is removed automatically. Full group evidence is in `feature_group_summary.csv` and `feature_selection_report.csv`.', '',
        '## Recommended retention and review','',
        '**KEEP:** '+(', '.join(keep.index) or '(none)')+'.','',
        '**REVIEW (retain unless explicitly approved otherwise):** '+(', '.join(review.index) or '(none)')+'.','',
        '## Proposed removals — approval required','', '| Feature | Retained representative | Reason |','|---|---|---|']
    for c,r in drop.iterrows(): lines.append(f'| {c} | {r.representative if pd.notna(r.representative) else "—"} | {r.reason} |')
    lines += ['', '## Reproduction and next step','',
        'See `feature_audit/README.md` for execution commands. The audit snapshot, SQL outputs, package versions, thresholds and script hashes are retained. Reuse the same audit schema to regenerate from the same data snapshot; use a new schema for changed source data.',
        'For this initial run, the test snapshot was captured separately after the training audit; its capture time is recorded in `run_metadata.csv`. Test values were not used for fitting. Future fresh runs snapshot both tables in one repeatable-read transaction.',
        'Review `proposed_drops.csv`; place only approved names in `approved_drops.json`. The separate application script validates the subset and preserves original tables. It has not been run. No numeric imputation or scaling is required for the initial LightGBM baseline; categorical mappings must be fitted on the early training partition and reused for validation/test, with an explicit unknown-category policy.',
        'Evaluate optional A/B/C models chronologically using PR-AUC, ROC-AUC, recall/FPR at a prespecified threshold, Brier/log loss and calibration plots, feature count, wall time and peak memory. Fit importance decisions inside the fitting partition or an inner chronological holdout, not on the final validation partition.','',
        '## References','',
        '- PostgreSQL aggregate definitions: https://www.postgresql.org/docs/17/functions-aggregate.html',
        '- LightGBM native missing and categorical handling: https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html','']
    (out/'preprocessing_summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
