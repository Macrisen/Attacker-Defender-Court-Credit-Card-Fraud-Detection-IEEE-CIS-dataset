"""Execute two matched historical Defender fits and independent window evaluation."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix
from defender.inference import Defender
from scripts.fit_temporal_defender import fit_new_defender
from preprocessing.temporal_feature_audit import run,sha,verify_windows,membership

def main():
    out=ROOT/'reports/phase3_6';data=ROOT/'data/processed/train_original.parquet'
    protofile=ROOT/'reports/phase3_5/temporal_split_protocol.json';protocol=json.loads(protofile.read_text())
    schemafile=ROOT/'reports/lightgbm_baseline/experiments/A_original/feature_schema.json';schema=json.loads(schemafile.read_text())
    manifest=run(data,schema,protocol,out,ROOT/'configs/feature_selection_temporal.yaml')
    print('Selection:',manifest['version'],'drops',manifest['dropped_predictors'],flush=True)
    metrics=[];bins=[]
    for kind,sc,prov in [('original',schemafile,out/'original_provenance.json'),('cleaned',out/'cleaned_schema.json',out/'selection_manifest.json')]:
        target=ROOT/'models/temporal_defender'/kind
        t=time.monotonic()
        fit_new_defender(data,sc,protofile,prov,target,max_rounds=2000)
        elapsed=time.monotonic()-t
        model=Defender(target)
        names=[f['name'] for f in model.schema['features']]
        for window in ('rl_development','rl_validation'):
            w=protocol['windows'][window]
            f=ds.dataset(data).to_table(columns=names+['transactionid','transactiondt','isfraud'],filter=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['max_time'])).to_pandas().sort_values(['transactiondt','transactionid'])
            assert len(f)==w['rows'] and membership(f)==w['ordered_id_sha256']
            p=model.predict(f[names]); y=f.isfraud.to_numpy(); d=p>1/6
            tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
            metrics.append(dict(dataset_version='original_431' if kind=='original' else manifest['version'],predictor_count=len(names),defender_fit_rows=protocol['windows']['defender_fit']['rows'],calibration_rows=protocol['windows']['defender_calibration']['rows'],evaluation_window=window,rows=len(f),AP=average_precision_score(y,p),ROC_AUC=roc_auc_score(y,p),recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 0,FPR=fp/(fp+tn),Brier=brier_score_loss(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),mean_business_reward=(10*tp-20*fn-5*fp+tn)/len(f),training_duration_seconds=elapsed,model_artifact_path=str(target),calibration_method=model.calibrator['method']))
            pd.DataFrame(dict(transactionid=f.transactionid,transactiondt=f.transactiondt,probability=p,decision=d.astype(int))).to_parquet(out/f'{kind}_{window}_predictions.parquet',index=False)
            for k in range(10):
                mask=(p>=k/10)&((p<(k+1)/10) if k<9 else (p<=1))
                bins.append(dict(version=kind,window=window,bin=k,count=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,fraud_rate=float(y[mask].mean()) if mask.any() else None))
        hashes={p.name:sha(p) for p in target.iterdir() if p.is_file()}
        (target/'artifact_manifest.json').write_text(json.dumps(dict(hashes=hashes,training_seconds=elapsed,seed=42,threshold_derivation='FLAG utility=15p-5; APPROVE utility=1-21p; FLAG iff 36p>6 => p>1/6'),indent=2))
        pd.DataFrame(metrics).to_csv(out/'temporal_defender_comparison.csv',index=False)
        pd.DataFrame(bins).to_csv(out/'calibration_diagnostics.csv',index=False)
        print(kind,'completed',elapsed,flush=True)
if __name__=='__main__':main()
