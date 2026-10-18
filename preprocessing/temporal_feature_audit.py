"""Fit-only feature audit. Full-data reads restricted to ID/time membership and file hashing."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from preprocessing.temporal_feature_selector import select

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def membership(frame):
    f=frame.sort_values(['transactiondt','transactionid'])
    return hashlib.sha256(f.transactionid.to_numpy(dtype='<i8').tobytes()).hexdigest()

def verify_windows(parquet,protocol):
    m=pq.read_table(parquet,columns=['transactionid','transactiondt']).to_pandas()
    assert m.transactionid.is_unique and not m.isna().any().any()
    count=0
    for w in protocol['windows'].values():
        f=m[m.transactiondt.between(w['min_time'],w['max_time'])]
        assert len(f)==w['rows'] and membership(f)==w['ordered_id_sha256']
        count+=len(f)
    assert count==len(m)
    return True

def run(parquet,schema,protocol,output,config):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    verify_windows(parquet,protocol)
    w=protocol['windows']['defender_fit']
    frame=ds.dataset(parquet).to_table(filter=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['max_time'])).to_pandas().sort_values(['transactiondt','transactionid'])
    assert len(frame)==w['rows'] and membership(frame)==w['ordered_id_sha256']
    names=[x['name'] for x in schema['features']]
    assert set(frame.columns)==set(names)|{'transactionid','transactiondt','isfraud'}
    y=frame.isfraud; stats=[]; duplicates={}; buckets={}
    for spec in schema['features']:
        n=spec['name']; s=frame[n]; null=s.isna(); counts=s.value_counts(dropna=True)
        fingerprint=hashlib.sha256(pd.util.hash_pandas_object(s,index=False).to_numpy().tobytes()).hexdigest()
        # Hash only shortlists; equality confirms every duplicate including null mask.
        for other in buckets.get(fingerprint,[]):
            if s.equals(frame[other]): duplicates[n]=other;break
        if n not in duplicates:buckets.setdefault(fingerprint,[]).append(n)
        dominant=counts.iloc[0] if len(counts) else 0
        stats.append(dict(feature=n,data_type=spec['postgres_type'],role=spec['role'],total_rows=len(s),missing_count=int(null.sum()),missing_percentage=float(null.mean()*100),non_null_count=int(s.notna().sum()),unique_count=len(counts),dominant_value_percentage=float(dominant/len(s)*100),fraud_rate_missing=float(y[null].mean()) if null.any() else None,fraud_rate_observed=float(y[~null].mean()) if (~null).any() else None,rare_value_fraud_rate=float(y[s.ne(counts.index[0]) & ~null].mean()) if len(counts)>1 else None,duplicate_of=duplicates.get(n),max_correlation=0.))
    stats=pd.DataFrame(stats).set_index('feature',drop=False)
    nums=[s['name'] for s in schema['features'] if s['role']=='numeric' and stats.loc[s['name'],'unique_count']>1]
    numeric=frame[nums].apply(pd.to_numeric).astype(float)
    if np.isinf(numeric.to_numpy()).any(): raise ValueError('Infinite fitting data')
    corr=numeric.corr(min_periods=2)
    valid=numeric.notna().to_numpy(dtype=np.float64); shared=valid.T@valid
    pairs=[]
    for i,a in enumerate(nums):
        stats.loc[a,'max_correlation']=float(corr[a].drop(a).abs().max())
        for j in range(i+1,len(nums)):
            b=nums[j]; r=corr.iloc[i,j]
            if abs(r)>=.98:
                pairs.append(dict(feature_a=a,feature_b=b,correlation=float(r),shared_observations=int(shared[i,j]),reliable=bool(shared[i,j]>=1000),decision='REVIEW',reason='No correlation-only drops; availability and temporal stability require validation'))
    # Chronological thirds are entirely within the allowed fitting window.
    windows=[]
    for i,sub in enumerate(np.array_split(np.arange(len(frame)),3)):
        for n in names:
            s=frame[n].iloc[sub]
            windows.append(dict(window=i,feature=n,rows=len(s),missing_percentage=float(s.isna().mean()*100),unique_count=int(s.nunique())))
    for n in names:
        dec,reason=select(stats.loc[n].to_dict(),duplicates.get(n))
        stats.loc[n,'decision']=dec;stats.loc[n,'reason']=reason
    stats.to_csv(output/'feature_selection_temporal.csv',index=False)
    pd.DataFrame(pairs,columns=['feature_a','feature_b','correlation','shared_observations','reliable','decision','reason']).to_csv(output/'correlation_pairs.csv',index=False)
    pd.DataFrame(windows).to_csv(output/'fit_window_stability.csv',index=False)
    selected=[n for n in names if stats.loc[n,'decision']!='DROP']
    manifest=dict(version='cleaned_prefix_'+str(len(selected)),fit_cutoff=w['max_time'],fit_window=w,selected_predictors=selected,dropped_predictors=[n for n in names if n not in selected],parquet_sha256=sha(parquet),config_sha256=sha(config),source=str(parquet),rules='Only all-NULL, fully observed constants, or exact duplicates removed. Correlation/high-missingness/rare-value features retained for review.',null_imputation=False)
    (output/'selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    cleaned=dict(schema);cleaned['features']=[f for f in schema['features'] if f['name'] in selected]
    (output/'cleaned_schema.json').write_text(json.dumps(cleaned,indent=2))
    original=dict(manifest,version='original_431',selected_predictors=names,dropped_predictors=[],rules='All source predictors, schema only; no statistical selection')
    (output/'original_provenance.json').write_text(json.dumps(original,indent=2))
    return manifest
