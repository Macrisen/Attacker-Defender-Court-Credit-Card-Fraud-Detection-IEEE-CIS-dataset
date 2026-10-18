"""Scheduled refits from permitted base rows and completed-only replay."""
import json,shutil
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.dataset as ds
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from defender.inference import encode
from rl_environment.defender import FrozenDefender
from .model_registry import ModelRegistry
ROOT=Path(__file__).resolve().parents[1]
class DefenderUpdater:
    def __init__(self,version,output,with_base=True,minimum=100,max_updates=3):
        self.baseline=ROOT/'models/temporal_defender'/version;self.original=FrozenDefender(self.baseline)
        self.registry=ModelRegistry(output);self.with_base=with_base;self.minimum=minimum;self.max_updates=max_updates
        self.schema=self.original._baseline.schema;self.mapping=self.original._baseline.mappings
        self.names=self.original.feature_names;self.count=0
        # Ordered prefix from explicitly permitted fitting rows; never development oracle labels.
        data=ds.dataset(ROOT/'data/processed/train_original.parquet')
        cutoff=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())['windows']['defender_fit']['max_time']
        table=data.to_table(columns=self.names+['transactionid','transactiondt','isfraud'],filter=ds.field('transactiondt')<=cutoff).sort_by([('transactiondt','ascending'),('transactionid','ascending')]).slice(0,20000)
        frame=table.to_pandas();self.base_x,_=encode(frame,self.schema,self.mapping);self.base_y=frame.isfraud.to_numpy()
    def update(self,replay,now):
        rows=replay.available(now)
        info=dict(time=now,available_verified=len(rows),strategy='historical_base_plus_verified' if self.with_base else 'verified_only_no_replay',updates_so_far=self.count)
        if len(rows)<self.minimum or self.count>=self.max_updates:return None,dict(info,status='INELIGIBLE',reason='minimum count or maximum update budget')
        boundary=int(len(rows)*.75);fit,cal=rows[:boundary],rows[boundary:]
        y=np.array([r['verified_label'] for r in fit]);yc=np.array([r['verified_label'] for r in cal])
        if len(np.unique(y))<2 or min(np.sum(yc==0),np.sum(yc==1))<5:return None,dict(info,status='INELIGIBLE',reason='insufficient classes in chronological verified fitting/calibration subsets')
        existing=self.registry.path/f'version_{self.count+1:03d}'
        if existing.exists():
            saved=json.loads((existing/'training_provenance.json').read_text())
            if saved['fit_verified_ids']!=[r['transaction_id'] for r in fit] or saved['calibration_verified_ids']!=[r['transaction_id'] for r in cal] or saved['deployment_time']!=now:
                raise ValueError('Resumed replay differs from saved update provenance')
            self.count+=1
            return FrozenDefender(existing),saved
        xf,_=encode(pd.DataFrame([r['features'] for r in fit]),self.schema,self.mapping);xc,_=encode(pd.DataFrame([r['features'] for r in cal]),self.schema,self.mapping)
        if self.with_base:xf=np.concatenate([self.base_x,xf]);y=np.concatenate([self.base_y,y])
        params=dict(objective='binary',learning_rate=.03,num_leaves=31,seed=42,num_threads=4,deterministic=True,force_col_wise=True,verbosity=-1)
        cats=[i for i,f in enumerate(self.schema['features']) if f['role']=='categorical']
        model=lgb.train(params,lgb.Dataset(xf,label=y,feature_name=self.names,categorical_feature=cats),num_boost_round=100)
        p=np.clip(model.predict(xc,num_threads=4),1e-12,1-1e-12);calibrator=LogisticRegression(C=1e6).fit(np.log(p/(1-p)).reshape(-1,1),yc)
        self.count+=1;dest=self.registry.allocate(self.count);model.save_model(str(dest/'lightgbm_model.txt'));joblib.dump(dict(method='sigmoid',model=calibrator),dest/'calibrator.joblib')
        for name in ('feature_schema.json','categorical_mappings.json'):shutil.copyfile(self.baseline/name,dest/name)
        (dest/'defender_config.json').write_text(json.dumps(dict(calibration_method='sigmoid',thresholds={'theoretical_reward_1_6':1/6},num_threads=4),indent=2))
        provenance=dict(**info,status='UPDATED',reason='verified-only chronological availability gate passed',deployment_time=now,fit_verified_ids=[r['transaction_id'] for r in fit],calibration_verified_ids=[r['transaction_id'] for r in cal],fit_max_completion=max(r['completion_time'] for r in fit),calibration_max_completion=max(r['completion_time'] for r in cal),base_rows=20000 if self.with_base else 0,feature_schema_source=str(self.baseline),selection_bias='Court/audit selection biased; no IPW',model_path=str(dest))
        (dest/'training_provenance.json').write_text(json.dumps(provenance,indent=2));self.registry.register(dest,now,max(r['completion_time'] for r in rows))
        return FrozenDefender(dest),provenance
