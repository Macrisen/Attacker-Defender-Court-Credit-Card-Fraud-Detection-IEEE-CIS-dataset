"""Shared feature preparation and inference. No fitting occurs during inference."""
import json
import os
from pathlib import Path
from functools import lru_cache
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

NUMERIC={'smallint','integer','bigint','numeric','real','double precision'}

def feature_role(name,dtype):
    name=name.lower()
    if name.startswith('card') or name in ('addr1','addr2'):
        return 'categorical'
    if name.startswith('id_') and name[3:].isdigit() and int(name[3:])>=12:
        return 'categorical'
    return 'numeric' if dtype in NUMERIC else 'categorical'

def category_values(series,dtype):
    if dtype in NUMERIC:
        numeric=pd.to_numeric(series,errors='raise').astype('float64')
        if np.isinf(numeric.to_numpy()).any(): raise ValueError('Infinite categorical code')
        # A single canonical representation across SQL integers, CSV and JSON floats.
        return numeric.astype('string')
    return series.astype('string')

def fit_mappings(frame,schema):
    mappings={}
    for f in schema['features']:
        if f['role']=='categorical':
            s=category_values(frame[f['name']],f['postgres_type'])
            mappings[f['name']]=sorted(s.dropna().unique().tolist())
    return mappings

def encode(frame,schema,mappings,strict=True):
    features=schema['features']; names=[f['name'] for f in features]
    missing=set(names)-set(frame.columns)
    if missing and strict: raise ValueError('Missing feature columns: '+', '.join(sorted(missing)))
    out=np.empty((len(frame),len(features)),dtype=np.float32)
    diagnostics=[]
    for j,f in enumerate(features):
        name=f['name']; s=frame[name] if name in frame else pd.Series(None,index=frame.index)
        if f['role']=='categorical':
            values=category_values(s,f['postgres_type'])
            cat=pd.Categorical(values,categories=mappings[name])
            codes=cat.codes.astype(np.int32)
            # Missing and previously unseen values both become -1 (LightGBM missing).
            out[:,j]=codes
            diagnostics.append({'feature':name,'missing_rows':int(values.isna().sum()),
                'unknown_rows':int(((codes<0)&values.notna().to_numpy()).sum()),'rows':len(frame)})
        else:
            values=pd.to_numeric(s,errors='raise').to_numpy(dtype=np.float32,na_value=np.nan)
            if np.isinf(values).any(): raise ValueError(f'Infinite or float32-overflow values in {name}')
            out[:,j]=values
    return out,diagnostics

def calibrate(probabilities,artifact):
    p=np.asarray(probabilities,dtype=np.float64)
    if artifact['method']=='raw': return p
    if artifact['method']=='sigmoid':
        clipped=np.clip(p,1e-12,1-1e-12)
        logits=np.log(clipped/(1-clipped)).reshape(-1,1)
        return artifact['model'].predict_proba(logits)[:,1]
    if artifact['method']=='isotonic': return artifact['model'].predict(p)
    raise ValueError('Unknown calibration method')

class Defender:
    def __init__(self,artifact_dir):
        directory=Path(artifact_dir)
        self.schema=json.loads((directory/'feature_schema.json').read_text())
        self.mappings=json.loads((directory/'categorical_mappings.json').read_text())
        self.config=json.loads((directory/'defender_config.json').read_text())
        self.model=lgb.Booster(model_file=str(directory/'lightgbm_model.txt'))
        self.calibrator=joblib.load(directory/'calibrator.joblib')
        if self.model.feature_name()!=[f['name'] for f in self.schema['features']]:
            raise ValueError('Model feature order differs from saved feature schema')
    def prepare(self,transactions):
        if isinstance(transactions,dict): transactions=[transactions]
        df=pd.DataFrame(transactions).copy()
        lookup={f['name'].lower():f['name'] for f in self.schema['features']}
        normalized=[]
        allowed_meta={'transactionid','transactiondt','isfraud'}
        for name in df.columns:
            if str(name).lower() not in lookup and str(name).lower() not in allowed_meta:
                raise ValueError(f'Unexpected input column: {name}')
            normalized.append(lookup.get(str(name).lower(),str(name).lower()))
        if len(normalized)!=len(set(normalized)): raise ValueError('Duplicate case-normalized column names')
        df.columns=normalized
        return encode(df,self.schema,self.mappings)[0]
    def predict(self,transactions):
        x=self.prepare(transactions)
        raw=self.model.predict(x,num_threads=self.config.get('num_threads',4))
        return calibrate(raw,self.calibrator)
    def decide(self,transaction,threshold=1/6):
        if not 0<=threshold<=1: raise ValueError('Threshold must be between zero and one')
        probability=float(self.predict(transaction)[0])
        return ('FLAG' if probability>threshold else 'APPROVE',probability)

@lru_cache(maxsize=1)
def _default_defender():
    default=Path(__file__).resolve().parents[1]/'reports/lightgbm_baseline/artifacts'
    return Defender(os.environ.get('DEFENDER_ARTIFACT_DIR',str(default)))

def predict_fraud_probability(transaction):
    """Return calibrated fraud probability using persisted preprocessing/model artifacts."""
    return float(_default_defender().predict(transaction)[0])

def defender_decision(transaction,threshold=1/6):
    """Return (APPROVE or FLAG, calibrated fraud probability); ties are approved."""
    return _default_defender().decide(transaction,threshold)
