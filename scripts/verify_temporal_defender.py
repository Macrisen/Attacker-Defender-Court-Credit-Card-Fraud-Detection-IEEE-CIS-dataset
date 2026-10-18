"""Independent post-fit schema/vocabulary/window verification and artifact metadata."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pyarrow.dataset as ds
from defender.inference import Defender,fit_mappings
from preprocessing.temporal_feature_audit import sha,membership,verify_windows
from rl_environment.temporal_protocol import validate_model_periods

def main():
    source=ROOT/'data/processed/train_original.parquet'
    protocol=json.loads((ROOT/'reports/phase3_5/temporal_split_protocol.json').read_text());w=protocol['windows']
    verify_windows(source,protocol)
    meta=ds.dataset(source).to_table(columns=['transactionid','transactiondt'],filter=ds.field('transactiondt')<=w['defender_calibration']['max_time']).to_pandas().sort_values(['transactiondt','transactionid'])
    fit=meta[meta.transactiondt<=w['defender_fit']['max_time']];cal=meta[meta.transactiondt>w['defender_fit']['max_time']]
    inner_cut=int(fit.transactiondt.iloc[int(len(fit)*.85)-1]);cal_cut=int(cal.transactiondt.iloc[len(cal)//2-1])
    windows={}
    for name,f in [('fit',fit),('inner_fit',fit[fit.transactiondt<=inner_cut]),('inner_early_stop',fit[fit.transactiondt>inner_cut]),('calibration',cal),('calibrator_candidate_fit',cal[cal.transactiondt<=cal_cut]),('calibrator_method_selection',cal[cal.transactiondt>cal_cut])]:
        windows[name]=dict(rows=len(f),min_time=int(f.transactiondt.min()),max_time=int(f.transactiondt.max()),ordered_id_sha256=membership(f))
    results=[]
    for kind in ('original','cleaned'):
        path=ROOT/'models/temporal_defender'/kind;model=Defender(path)
        cfg=json.loads((path/'training_config.json').read_text());p=cfg['provenance']
        validate_model_periods(protocol,windows['fit']['max_time'],windows['calibration']['max_time'],p['fit_cutoff'])
        cats=[f['name'] for f in model.schema['features'] if f['role']=='categorical']
        frame=ds.dataset(source).to_table(columns=cats,filter=ds.field('transactiondt')<=w['defender_fit']['max_time']).to_pandas()
        assert fit_mappings(frame,model.schema)==model.mappings
        assert model.model.feature_name()==[f['name'] for f in model.schema['features']]
        assert p['parquet_sha256']==sha(source)
        result=dict(version=kind,independently_verified=True,windows=windows,category_vocabulary='exact equality with vocabulary recomputed from fit prefix only',model_feature_order_verified=True,source_sha256=p['parquet_sha256'],selection_config_sha256=p['config_sha256'],label_availability='offline historical availability assumed; actual discovery timestamps unavailable',threshold_fixed=1/6,calibration_refit='entire permitted calibration window',historical_baseline_reused=False)
        (path/'validation_metadata.json').write_text(json.dumps(result,indent=2))
        artifact=json.loads((path/'artifact_manifest.json').read_text());artifact['hashes']={f.name:sha(f) for f in path.iterdir() if f.is_file() and f.name!='artifact_manifest.json'}
        (path/'artifact_manifest.json').write_text(json.dumps(artifact,indent=2));results.append(result)
    (ROOT/'reports/phase3_6/independent_defender_verification.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
