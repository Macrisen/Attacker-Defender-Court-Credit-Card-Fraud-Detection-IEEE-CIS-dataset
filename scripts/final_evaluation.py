"""Freeze gates, source-paired uncertainty and protected final evaluation."""
import json,datetime,gc
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.temporal_feature_audit import sha
ROOT=Path(__file__).resolve().parents[1]

def verify_freeze(path):
    doc=json.loads(Path(path).read_text())
    if not doc.get('all_scientific_gates_passed') or doc.get('holdout_tuning_allowed',True):raise ValueError('Final protocol is not frozen')
    for name in ('phase4a','phase4b','phase4c','phase5'):
        if json.loads((ROOT/'reports'/name/'run_manifest.json').read_text())['status']!='PASS':raise ValueError('Failed mandatory phase gate')
    for p,h in doc['frozen_artifacts'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen policy/model/config changed: '+p)
    return doc

def freeze(config,out):
    path=out/'frozen_protocol.json'
    if path.exists():verify_freeze(path);return path
    files=[p for r in ('models/rl','models/temporal_defender','models/continual_defender') for p in (ROOT/r).rglob('*') if p.is_file() and p.suffix in ('.pt','.txt','.joblib','.json')]
    files+=[ROOT/'configs/attacker_ppo.yaml',ROOT/'configs/assumption_attack.yaml',ROOT/'configs/court_rl.yaml',ROOT/'configs/continual_learning.yaml']
    files += [p for directory in ('rl_agents','rl_environment','continual_learning','scripts','defender') for p in (ROOT/directory).rglob('*.py')]
    doc=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_scientific_gates_passed=True,holdout_tuning_allowed=False,previously_baseline_evaluated_holdout=True,config=config,comparison='E1-E8; same source IDs and source-keyed randomness; three-day maturity guard; adapted models frozen after Development',frozen_artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},primary_metrics=['Defender AP','conditional attacker evasion','missed-fraud recovery','residual fraud','mean end-to-end research utility'],confidence_intervals='1000 paired daily-block resamples after averaging repeated policy seeds per source ID; exploratory on validation, historically non-pristine holdout disclosed',no_policy_or_model_updates_during_final=True)
    path.write_text(json.dumps(doc,indent=2));verify_freeze(path);return path

def paired_interval(a,b,value,seed=42,repetitions=1000):
    # One observation per source after averaging repeated seed outcomes.
    a=a.groupby(['transactionid','transactiondt'],as_index=False)[value].mean();b=b.groupby(['transactionid','transactiondt'],as_index=False)[value].mean()
    joined=a.merge(b,on=['transactionid','transactiondt'],suffixes=('_a','_b'),validate='one_to_one')
    if len(joined)!=len(a) or len(joined)!=len(b):raise ValueError('Unpaired source IDs')
    joined['delta']=joined[value+'_a']-joined[value+'_b'];joined['block']=joined.transactiondt//86400
    blocks=joined.groupby('block').delta.agg(['sum','count']);rng=np.random.default_rng(seed)
    indices=rng.integers(len(blocks),size=(repetitions,len(blocks)))
    means=blocks['sum'].to_numpy()[indices].sum(axis=1)/blocks['count'].to_numpy()[indices].sum(axis=1)
    return dict(mean_difference=float(joined.delta.mean()),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),source_count=len(joined),blocks=len(blocks),bootstrap_repetitions=repetitions,method='paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence')

def final_matrix(config,out,protocol_path):
    from rl_agents.court.trajectory_collector import StreamData,collect
    from rl_agents.court.train_court import load_policy
    from rl_environment.defender import FrozenDefender
    from scripts.remaining_later_phases import policies_for,attack_adapter,transaction_results,defender_metrics
    inventory=json.loads((ROOT/'reports/phase5/model_inventory.json').read_text())
    matrix=[('E1','none','none',None),('E2','random_uniform','none',None),('E3','ppo','none',None),('E4','none','rule',None),('E5','none','learned',None),('E6','ppo','learned',None),('E7','none','learned','U3'),('E8','ppo','learned','U2')]
    summaries=[];paths=[]
    for version in config['versions']:
        data=StreamData(version,'final_holdout',frozen_protocol=protocol_path);records=data.records();baseline=data.defender
        for seed in config['seeds']:
            a,c=policies_for(version,seed)
            for name,att,court,continual in matrix:
                runout=out/'final_predictions'/version/f'{name}_seed{seed}';runout.mkdir(parents=True,exist_ok=True)
                if (runout/'metrics.json').exists():
                    summaries.append(json.loads((runout/'metrics.json').read_text()));paths.append(runout/'transactions.parquet');continue
                data.defender=baseline;deployed_updates=0
                if continual:
                    entry=next(r for r in inventory if r['version']==version and r['scenario']==continual);data.defender=FrozenDefender(entry['model_path']);deployed_updates=entry['updates']
                data.no_investigations=court=='none';adapter=attack_adapter(att,a,seed)
                _,m,sim,memory=collect(data,records,load_policy(c) if court=='learned' else None,seed,baseline='skip' if court=='none' else court,attack_adapter=adapter)
                tx=transaction_results(sim,version,seed,name)
                # Paired original scores are evaluator-only and cannot change proposals.
                original=data.defender.predict_many([r.features for r in records]);tx['original_probability']=original
                tx['attacked']=False
                if adapter:tx['attacked']=tx.transactionid.map({tid:p['modified'] for tid,p in adapter.provenance.items()})
                detected=(tx.label_evaluation_only==1)&(tx.original_probability>1/6)&tx.attacked;flips=detected&~tx.flag
                metrics=dict(version=version,seed=seed,experiment=name,window='reserved_final_holdout',attacker=att,court=court,continual_development_updates=deployed_updates,updates_during_final=0,**m,**defender_metrics(tx),newly_evaded=int(flips.sum()),accepted_originally_detected=int(detected.sum()),conditional_evasion=float(flips.sum()/detected.sum()) if detected.any() else None)
                tx.to_parquet(runout/'transactions.parquet',index=False);(runout/'metrics.json').write_text(json.dumps(metrics,indent=2));summaries.append(metrics);paths.append(runout/'transactions.parquet')
                del sim,memory,adapter,tx;gc.collect();print('Final',version,seed,name,'complete',flush=True)
        del records,data;gc.collect()
    return pd.DataFrame(summaries),paths
