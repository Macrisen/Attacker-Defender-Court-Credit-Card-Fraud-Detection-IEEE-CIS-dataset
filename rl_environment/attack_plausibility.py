"""Fit-prefix-only soft plausibility; not a causal validity certificate."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from preprocessing.temporal_feature_audit import membership,sha

ATTRIBUTES=('productcd','card4','card6')
def key(value):
    if value is None or pd.isna(value):return '<MISSING>'
    return str(value)

def statistics(amounts):
    x=np.asarray(amounts,dtype=float);z=np.log1p(x);median=float(np.median(z))
    return dict(rows=len(x),minimum=float(x.min()),maximum=float(x.max()),q01=float(np.quantile(x,.01)),q99=float(np.quantile(x,.99)),median_log=median,mad_log=float(np.median(np.abs(z-median))))

class AttackPlausibility:
    def __init__(self,document,protocol):
        w=protocol['windows']['defender_fit'];p=document['provenance']
        if p['window']!=w:raise ValueError('Plausibility provenance differs from exact Defender Fit window')
        self.document=document
    @classmethod
    def fit_frame(cls,frame,protocol,source_sha256=None):
        w=protocol['windows']['defender_fit']
        if len(frame)!=w['rows'] or frame.transactiondt.min()!=w['min_time'] or frame.transactiondt.max()!=w['max_time'] or membership(frame)!=w['ordered_id_sha256']:
            raise ValueError('Future data or wrong fitting membership')
        amount=pd.to_numeric(frame.transactionamt).astype(float)
        valid=np.isfinite(amount)&(amount>0)
        if not valid.any():raise ValueError('No positive finite fitting amounts')
        groups={}
        for attr in ATTRIBUTES:
            if attr not in frame:continue
            groups[attr]={}
            for category,indices in frame.groupby(frame[attr].map(key),sort=True).groups.items():
                subset=amount.loc[indices];subset=subset[np.isfinite(subset)&(subset>0)]
                if len(subset)>=200:groups[attr][category]=statistics(subset)
        doc=dict(provenance=dict(window=w,source_sha256=source_sha256,fit_uses_labels=False),global_statistics=statistics(amount[valid]),conditional_statistics=groups,minimum_group_rows=200,acceptance='Hard numeric/budget checks only; statistical diagnostics never reject',causal_validity=False)
        return cls(doc,protocol)
    @classmethod
    def fit_parquet(cls,path,protocol):
        w=protocol['windows']['defender_fit']
        table=ds.dataset(path).to_table(columns=['transactionid','transactiondt','transactionamt',*ATTRIBUTES],filter=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['max_time']))
        return cls.fit_frame(table.to_pandas(),protocol,sha(path))
    def diagnostics(self,original,modified,context):
        def score(x,s):return abs(math.log1p(float(x))-s['median_log'])/max(1.4826*s['mad_log'],1e-6)
        s=self.document['global_statistics'];a=float(original);b=float(modified)
        before=score(a,s);after=score(b,s);details={}
        for attr,groups in self.document['conditional_statistics'].items():
            g=groups.get(key(context.get(attr)))
            details[attr]=dict(supported=g is not None,rows=g['rows'] if g else 0,modified_outside_q01_q99=bool(b<g['q01'] or b>g['q99']) if g else None,original_score=score(a,g) if g else None,modified_score=score(b,g) if g else None)
        return dict(original_anomaly_score=before,modified_anomaly_score=after,anomaly_change=after-before,original_outside_fit_range=bool(a<s['minimum'] or a>s['maximum']),modified_outside_fit_range=bool(b<s['minimum'] or b>s['maximum']),original_outside_q01_q99=bool(a<s['q01'] or a>s['q99']),modified_outside_q01_q99=bool(b<s['q01'] or b>s['q99']),conditional=details,interpretation='Statistical rarity only; not proof of impossibility or causal validity')
