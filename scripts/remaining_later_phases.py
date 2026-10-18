"""Sequential Court, hybrid, verified-only adaptation and frozen evaluation phases."""
import gc,json,time,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyarrow.dataset as ds
from scripts.remaining_pipeline import write_report,check_protected
from preprocessing.temporal_feature_audit import sha
ROOT=Path(__file__).resolve().parents[1]

def transaction_results(sim,version,seed,experiment):
    memory={r['transaction_id']:r for r in sim.memory.get_recent_verified(sim.now,sim.memory.memory_size())}
    rows=[]
    for r,a in zip(sim._records,sim.actions):
        verified=memory.get(r.transaction_id);court=verified is not None and verified['source']=='court'
        rows.append(dict(version=version,seed=seed,experiment=experiment,transactionid=r.transaction_id,transactiondt=r.transaction_time,label_evaluation_only=r.hidden_label,probability=a['defender_probability'],flag=a['defender_decision']=='FLAG',court_action=a['court_action'],investigation_source=a['source'],verified=verified is not None,verified_label=verified['verified_label'] if verified else None,court_verified_fraud=bool(court and verified['verified_label']),court_recovered_missed=bool(court and verified['verified_label'] and a['defender_decision']=='APPROVE'),completion_time=verified['completion_time'] if verified else None,model_version=sim.defender.version))
    return pd.DataFrame(rows)

def defender_metrics(frame):
    from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix
    y=frame.label_evaluation_only.to_numpy();p=frame.probability.to_numpy();d=frame.flag.to_numpy();tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
    recovered=int(frame.court_recovered_missed.sum());cost=int(frame.investigation_source.ne('none').sum())
    return dict(AP=average_precision_score(y,p),ROC_AUC=roc_auc_score(y,p),recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 0.,FPR=fp/(fp+tn),Brier=brier_score_loss(y,p),mean_business_utility=float((10*tp-20*fn-5*fp+tn)/len(y)),initially_detected_fraud=int(tp),initially_missed_fraud=int(fn),court_missed_fraud_recovery=recovered,residual_fraud_after_defender_and_court=int(fn-recovered),investigation_cost=cost,end_to_end_utility=float((10*tp-20*fn-5*fp+tn)+10*recovered-cost),utility_definition='initial Defender utility +10 per completed Court discovery of missed fraud - all Court/audit investigation costs; diagnostic research utility')

def phase4b(config,out,resume):
    from rl_agents.court.train_court import train,load_policy
    from rl_agents.court.trajectory_collector import StreamData,collect
    from rl_environment.defender import FrozenDefender
    runs=[];training=[];metrics=[];transaction=[]
    for version in config['versions']:
        for seed in config['seeds']:
            r=train(version,seed,config,resume);runs.append(r);training.append(pd.read_csv(Path(r['best_checkpoint']).parent/'training_metrics.csv'))
        data=StreamData(version,'rl_validation');records=data.records()
        # Prespecified causal high-risk threshold learned from permitted fitting data only.
        w=data.protocol['windows']['defender_fit'];d=ds.dataset(ROOT/'data/processed/train_original.parquet')
        frame=d.to_table(columns=data.defender.feature_names,filter=ds.field('transactiondt')<=w['max_time']).to_pandas()
        scores=data.defender._baseline.predict(frame);risk_threshold=float(np.quantile(scores,.982));del frame,scores;gc.collect()
        for seed in config['seeds']:
            best=next(r['best_checkpoint'] for r in runs if r['version']==version and r['seed']==seed)
            for name in ('learned','random','rule','top_risk','threshold','skip'):
                policy=load_policy(best) if name=='learned' else None
                _,m,sim,memory=collect(data,records,policy,seed,baseline=None if policy else name,threshold=risk_threshold if name=='top_risk' else 1/6)
                tx=transaction_results(sim,version,seed,name);transaction.append(tx)
                metrics.append(dict(version=version,seed=seed,policy=name,top_risk_threshold=risk_threshold,**m,**defender_metrics(tx)))
                del sim,memory;gc.collect()
                print('Court validation',version,seed,name,m['court_policy_reward'],flush=True)
        del data,records;gc.collect()
    training=pd.concat(training,ignore_index=True);comparison=pd.DataFrame(metrics);tx=pd.concat(transaction,ignore_index=True)
    training.to_csv(out/'court_training_metrics.csv',index=False);comparison.to_csv(out/'court_policy_comparison.csv',index=False);tx.to_csv(out/'court_validation_results.csv',index=False)
    comparison[['version','seed','policy','court_investigation_count','audit_investigation_count','investigation_budget_used','transactions_processed']].to_csv(out/'court_budget_analysis.csv',index=False)
    from scripts.phase3_6_checks import credit_fixture
    checks,_,_,_=credit_fixture();pd.DataFrame(checks).to_csv(out/'court_delayed_credit_checks.csv',index=False)
    (out/'court_checkpoints.json').write_text(json.dumps(runs,indent=2))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/phase4','-v'],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Court scientific tests failed')
    assert all(r['gradient_updates']>0 for r in runs);assert (comparison.investigations_pending==0).all()
    write_report(out/'court_training_report.html','Phase4B — actual recurrent Court PPO',{'Training':training,'Frozen validation comparison':comparison},'GRU32, unchanged11 observations, fixed behavior throughout each complete10000-decision episode. Exact origin utilities and time-aware MC returns become available only after maturity. Four clipped PPO updates per episode. Causal recurrent forward carry recomputed with current parameters; gradients truncated every128 steps, optimizer updates only after full-episode accumulation. Critic units are return/100; original research rewards unchanged. Budget mask identical in acting and updates. Validation only selects saved checkpoints. Top-risk is a causal fit-prefix98.2% score threshold, not oracle future-window sorting. Common random delays/audits are keyed by seed and source ID. No unbiased prevalence claim.')
    return dict(status='PASS',trained_runs=len(runs),training_interactions=sum(r['training_interactions'] for r in runs))

def run_phase(phase,config,out,resume):
    if phase=='4b':return phase4b(config,out,resume)
    if phase=='4c':return phase4c(config,out,resume)
    if phase=='5':return phase5(config,out,resume)
    if phase=='6':return phase6(config,out,resume)
    raise ValueError('Unknown phase')

def policies_for(version,seed):
    a=json.loads((ROOT/'reports/phase4a/attacker_checkpoints.json').read_text());c=json.loads((ROOT/'reports/phase4b/court_checkpoints.json').read_text())
    return next(r['best_checkpoint'] for r in a if r['version']==version and r['seed']==seed and r['policy']=='ppo'),next(r['best_checkpoint'] for r in c if r['version']==version and r['seed']==seed)

def attack_adapter(name,checkpoint,seed):
    from rl_environment.attack_plausibility import AttackPlausibility
    from rl_environment.assumption_attack_adapter import AssumptionAttackAdapter
    from rl_environment.hybrid_policy_adapter import PPOAttackAdapter
    from rl_agents.attacker.evaluate_ppo import load_policy
    import yaml
    protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text());p=AttackPlausibility(json.loads((ROOT/'reports/phase3_7/plausibility_model.json').read_text()),protocol)
    c=yaml.safe_load((ROOT/'configs/assumption_attack.yaml').read_text())
    if name=='none':return None
    if name=='ppo':return PPOAttackAdapter(p,c,load_policy(checkpoint),seed)
    return AssumptionAttackAdapter(p,c,.05,name,seed)

def phase4c(config,out,resume):
    from rl_agents.court.trajectory_collector import StreamData,collect
    from rl_agents.court.train_court import load_policy
    matrix=[('E1','none','none'),('E2','random_uniform','none'),('E3','ppo','none'),('E4','none','rule'),('E5','none','learned'),('E6','ppo','learned'),('R1','fixed_decrease','learned'),('R2','fixed_increase','learned'),('R3','random_uniform','learned')]
    summaries=[];transactions=[]
    for version in config['versions']:
        data=StreamData(version,'rl_validation');records=data.records();original=data.defender.predict_many([r.features for r in records]);original_map={r.transaction_id:p for r,p in zip(records,original)}
        for seed in config['seeds']:
            attacker,court=policies_for(version,seed)
            for name,att,c in matrix:
                data.no_investigations=c=='none';adapter=attack_adapter(att,attacker,seed)
                _,m,sim,memory=collect(data,records,load_policy(court) if c=='learned' else None,seed,baseline='skip' if c=='none' else c,attack_adapter=adapter)
                tx=transaction_results(sim,version,seed,name);tx['original_probability']=tx.transactionid.map(original_map)
                tx['newly_evaded']=(tx.label_evaluation_only==1)&(tx.original_probability>1/6)&~tx.flag
                tx['attacked']=False;tx['attack_cost']=0.;tx['attacker_reward']=0.
                if adapter:
                    tx['attacked']=tx.transactionid.map({tid:p['modified'] for tid,p in adapter.provenance.items()})
                    tx['attack_cost']=tx.transactionid.map({tid:e.cost for tid,e in adapter.episodes.items()}).fillna(0)
                    tx['attacker_reward']=tx.transactionid.map({tid:p.get('attacker_reward',0.) for tid,p in adapter.provenance.items()})
                denominator=int(((tx.label_evaluation_only==1)&tx.attacked&(tx.original_probability>1/6)).sum())
                summaries.append(dict(version=version,seed=seed,experiment=name,attacker=att,court=c,**m,**defender_metrics(tx),newly_evaded=int(tx.newly_evaded.sum()),accepted_originally_detected=denominator,conditional_evasion=float(tx.newly_evaded.sum()/denominator) if denominator else None,mean_attacker_cost=float(tx.loc[tx.attacked,'attack_cost'].mean()) if tx.attacked.any() else 0.,mean_attacker_reward=float(tx.loc[tx.label_evaluation_only==1,'attacker_reward'].mean())))
                transactions.append(tx);del sim,memory,adapter;gc.collect()
                print('Hybrid',version,seed,name,'recovered',m['missed_fraud_recovered'],flush=True)
        del data,records;gc.collect()
    summary=pd.DataFrame(summaries);tx=pd.concat(transactions,ignore_index=True)
    summary.to_csv(out/'hybrid_experiment_matrix.csv',index=False);tx.to_csv(out/'hybrid_transaction_results.csv',index=False)
    summary[['version','seed','experiment','court_policy_reward','audit_reward','end_to_end_utility','court_missed_fraud_recovery','investigation_cost']].to_csv(out/'hybrid_reward_analysis.csv',index=False)
    assert (summary.investigations_pending==0).all();assert (summary.maturity_end<=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())['windows']['rl_validation']['max_time']).all()
    write_report(out/'hybrid_integration_report.html','Phase4C — frozen-policy hybrid integration',{'Aligned experiment matrix':summary},'All comparisons use identical chronological decision prefixes with a three-day maturity guard. Verification delays and audit draws are source-keyed common random numbers. No-Court means no Court and no audit interventions. Other policies retain180+20 budgets. Attacker and Court are frozen; no alternating-training or equilibrium claim. Source labels are evaluation-only except private attacker eligibility and delayed verification. End-to-end utility is explicitly initial Defender business utility +10 per completed Court recovery of initially missed fraud minus all investigation costs; not monetary profit.')
    return dict(status='PASS',experiments=len(summary),transactions=len(tx))

def phase5(config,out,resume):
    from continual_learning.simulation import run_development
    from rl_agents.court.trajectory_collector import StreamData,collect
    from rl_agents.court.train_court import load_policy
    from rl_environment.defender import FrozenDefender
    scenarios=[('F0','none','frozen',1000,10000,'rule'),('F1','ppo','frozen',1000,10000,'rule'),('U0','none','replay',1000,10000,'rule'),('U1','ppo','replay',1000,10000,'rule'),('U2','ppo','replay',1000,10000,'learned'),('U3','none','replay',1000,10000,'learned'),('A0','ppo','no_replay',1000,10000,'rule'),('A1','ppo','replay',200,10000,'rule'),('A2','ppo','replay',1000,20000,'rule')]
    updates=[];memoryrows=[];comparison=[];forgetting=[];drift=[];inventory=[];seed=42
    for version in config['versions']:
        attacker_path,court_path=policies_for(version,seed)
        for name,att,strategy,capacity,interval,court_kind in scenarios:
            runout=out/'runs'/version/name;runout.mkdir(parents=True,exist_ok=True)
            complete=runout/'complete.json'
            if complete.exists() and resume:
                saved=json.loads(complete.read_text());summary=saved['summary'];history=saved['history'];forget=saved['forgetting'];dr=saved['drift'];development=saved['development_metrics']
            else:
                adapter=attack_adapter(att,attacker_path,seed)
                summary,history,forget,dr,sim,memory=run_development(version,seed,adapter,court_path,ROOT/'models/continual_defender'/version/name,strategy,capacity,interval,court_kind)
                tx=transaction_results(sim,version,seed,name);tx['model_version']=[a['model_version_at_decision'] for a in sim.actions]
                development=defender_metrics(tx);tx.to_parquet(runout/'development_predictions.parquet',index=False)
                with (runout/'verified_memory.jsonl').open('w') as f:
                    for r in memory:f.write(json.dumps(r,default=str)+'\n')
                complete.write_text(json.dumps(dict(summary=summary,history=history,forgetting=forget,drift=dr,development_metrics=development),indent=2))
                del sim,memory,adapter;gc.collect()
            updates.extend([dict(version=version,scenario=name,**r) for r in history])
            forgetting.extend([dict(r,version=version,scenario=name,model_version=r.get('model_version',r.get('version'))) for r in forget])
            drift.extend([dict(version=version,scenario=name,**r) for r in dr])
            memoryrows.append(dict(version=version,scenario=name,**summary));comparison.append(dict(version=version,seed=seed,scenario=name,window='rl_development_prequential',**summary,**development))
            data=StreamData(version,'rl_validation');data.defender=FrozenDefender(summary['final_model_path']);records=data.records()
            adapter=attack_adapter(att,attacker_path,seed)
            _,m,sim,memory=collect(data,records,load_policy(court_path) if court_kind=='learned' else None,seed,baseline=None if court_kind=='learned' else 'rule',attack_adapter=adapter)
            tx=transaction_results(sim,version,seed,name);tx.to_parquet(runout/'validation_predictions.parquet',index=False)
            comparison.append(dict(version=version,seed=seed,scenario=name,window='rl_validation_frozen_after_development',updates=summary['updates'],final_model_path=summary['final_model_path'],**m,**defender_metrics(tx)))
            inventory.append(dict(version=version,scenario=name,model_path=summary['final_model_path'],updates=summary['updates']))
            del sim,memory,data,records,adapter;gc.collect()
            print('Continual',version,name,'updates',summary['updates'],flush=True)
    pd.DataFrame(updates).to_csv(out/'model_update_history.csv',index=False);pd.DataFrame(memoryrows).to_csv(out/'verified_memory_summary.csv',index=False);pd.DataFrame(comparison).to_csv(out/'continual_comparison.csv',index=False);pd.DataFrame(forgetting).to_csv(out/'forgetting_analysis.csv',index=False);pd.DataFrame(drift).to_csv(out/'drift_analysis.csv',index=False)
    (out/'model_inventory.json').write_text(json.dumps(inventory,indent=2))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/phase5','-v'],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Verified-only tests failed')
    actual=sum(r['updates'] for r in inventory)
    if actual==0:raise ValueError('No update passed verified fitting/calibration eligibility; cannot claim adaptation')
    for entry in updates:
        if entry['status']=='UPDATED':assert entry['fit_max_completion']<=entry['deployment_time'] and entry['calibration_max_completion']<=entry['deployment_time']
    write_report(out/'continual_learning_report.html','Phase5 — verified-only scheduled refits',{'Executed comparison':pd.DataFrame(comparison),'Update eligibility and provenance':pd.DataFrame(updates),'Historical reference stability':pd.DataFrame(forgetting),'Selection composition':pd.DataFrame(memoryrows)},'Development prequential updates only, seed42. Baseline Defenders remain immutable. Base replay is earliest20000 Defender Fit rows; updated model fits100 trees, sigmoid calibration uses disjoint latest25% of verified labels with≥5 each class. Insufficient labels retain existing model. First1000 Development source IDs permanently excluded from update fitting/calibration and used as historical reference. No-replay fits only permitted verified examples. Capacity200/1000 and interval10000/20000 are predeclared ablations. Policies frozen; no future labels enter updater. Validation freezes the terminal Development model, so validation does not test online updating there. Selection-biased calibration and memory are not population-unbiased; no IPW. Replaying an interrupted deterministic run reuses existing identical model versions after provenance verification.')
    return dict(status='PASS',model_updates=actual,experiments=len(inventory))

def phase6(config,out,resume):
    import unittest,datetime,platform,importlib.metadata,html
    from scripts.test_phase3_7 import Results,rows as test_rows
    from scripts.final_evaluation import freeze,final_matrix,paired_interval
    from scripts.research_figures import training_figures,final_figures
    suite=unittest.TestSuite()
    for directory in ('rl_environment','phase3_5','phase3_6','phase3_7','phase4','phase5','phase6'):suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'/directory)))
    with (out/'all_tests.log').open('w') as log:tests=unittest.TextTestRunner(stream=log,resultclass=Results,verbosity=2).run(suite)
    pd.DataFrame(test_rows).to_csv(out/'all_test_results.csv',index=False)
    if not tests.wasSuccessful():raise RuntimeError('Final scientific gate: regression/new tests failed')
    # Predeclared audit/budget/delay sensitivity, not a configuration search.
    from rl_agents.court.trajectory_collector import StreamData,collect
    from rl_agents.court.train_court import load_policy
    sensitivity=[]
    for version in config['versions']:
        data=StreamData(version,'rl_validation');records=data.records();_,court=policies_for(version,42)
        for name,total,reserve,low,high in [('default',200,20,1,3),('audit_off_same_total',200,0,1,3),('half_total_budget',100,20,1,3),('shorter_delay',200,20,.5,1.5)]:
            data.environment_budget=dict(window_size=10000,total=total,audit_reserve=reserve);data.environment_verification=dict(min_days=low,max_days=high)
            _,m,sim,memory=collect(data,records,load_policy(court),42)
            sensitivity.append(dict(version=version,ablation=name,total_budget=total,audit_reserve=reserve,min_delay_days=low,max_delay_days=high,**m))
            del sim,memory;gc.collect()
        del data,records;gc.collect()
    pd.DataFrame(sensitivity).to_csv(out/'court_sensitivity.csv',index=False)
    # All policy/model decisions are now frozen before any final source feature/label read.
    frozen=freeze(config,out);final,paths=final_matrix(config,out,frozen)
    final.to_csv(out/'final_experiment_results.csv',index=False)
    comparison=final.groupby(['version','experiment']).agg(AP_mean=('AP','mean'),AP_std=('AP','std'),utility_mean=('end_to_end_utility','mean'),utility_std=('end_to_end_utility','std'),recovered_mean=('court_missed_fraud_recovery','mean'),residual_mean=('residual_fraud_after_defender_and_court','mean')).reset_index()
    comparison.to_csv(out/'original_vs_cleaned_comparison.csv',index=False)
    a=pd.read_csv(ROOT/'reports/phase4a/attacker_baseline_comparison.csv');a.to_csv(out/'attacker_ppo_results.csv',index=False)
    c=pd.read_csv(ROOT/'reports/phase4b/court_policy_comparison.csv');c.to_csv(out/'court_rl_results.csv',index=False)
    cl=pd.read_csv(ROOT/'reports/phase5/continual_comparison.csv');cl.to_csv(out/'continual_learning_results.csv',index=False)
    ablation=pd.concat([a.assign(component='attacker',evaluation_window='rl_validation'),c.assign(component='court',evaluation_window='rl_validation'),cl.assign(component='continual'),pd.DataFrame(sensitivity).assign(component='court_sensitivity',evaluation_window='rl_validation')],ignore_index=True,sort=False);ablation.to_csv(out/'ablation_results.csv',index=False)
    # Source-paired daily block CIs; repeated seed outcomes are averaged first.
    statistical=[]
    for version in config['versions']:
        frames={}
        for experiment in ('E1','E4','E5','E6','E7','E8'):
            f=pd.concat([pd.read_parquet(p) for p in paths if f'/{version}/' in str(p) and f'/{experiment}_seed' in str(p)],ignore_index=True)
            y=f.label_evaluation_only.to_numpy();d=f.flag.to_numpy();f['utility']=np.where(y,np.where(d,10,-20),np.where(d,-5,1))+10*f.court_recovered_missed.astype(int)-f.investigation_source.ne('none').astype(int);frames[experiment]=f[['transactionid','transactiondt','utility']]
        for left,right in [('E5','E4'),('E6','E1'),('E7','E5'),('E8','E6')]:statistical.append(dict(version=version,comparison=left+' minus '+right,window='reserved_final_holdout',metric='utility per source transaction',**paired_interval(frames[left],frames[right],'utility')))
    pd.DataFrame(statistical).to_csv(out/'statistical_tests.csv',index=False)
    inventory=[]
    for root in ('models/rl/attacker','models/rl/court','models/continual_defender'):
        for p in (ROOT/root).rglob('*'):
            if p.is_file():inventory.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    (out/'final_model_inventory.json').write_text(json.dumps(inventory,indent=2))
    resources=[]
    for phase,file in [('4a','attacker_checkpoints.json'),('4b','court_checkpoints.json')]:
        for r in json.loads((ROOT/'reports'/('phase'+phase)/file).read_text()):resources.append(dict(phase=phase,version=r['version'],seed=r['seed'],policy=r.get('policy','recurrent_court'),training_interactions=r['training_interactions'],gradient_updates=r['gradient_updates'],runtime_seconds=r['elapsed_seconds'],device='CPU',threads=4,peak_rss_bytes=None,peak_rss_note='not instrumented; do not infer peak from system RAM'))
    pd.DataFrame(resources).to_csv(out/'training_resource_usage.csv',index=False)
    limitations='''# Research limitations

- Mode B modifies only amount under assumed fraud-intent preservation and fixed other fields; causal real-world manipulability remains BLOCKED.
- Historical labels for initial Defender fitting/calibration are assumed available; actual label-discovery times are unavailable. Verification delays are simulated.
- The reserved final chronological holdout was evaluated by an earlier baseline study. It is not historically pristine; it was excluded from all new fitting and selection until protocol freeze.
- Repeated Development episodes are offline simulation, not independent deployment observations. Validation checkpoint selection can be optimistic; no redesign based on final results is permitted.
- Recurrent Court uses truncated gradients over128 steps, 40 updates per100k decision budget and compact11-feature observations. Recurrent state does not prove sufficient information or optimality. Poor performance is retained.
- Continual refits use selection-biased verified labels, sparse chronological calibration and a20k-row historical base. No unbiased prevalence/IPW claim. Drift outputs are descriptive changes, not proven concept drift.
- Continual adaptation is executed only on Development, seed42. Validation/final freeze the resulting models; they do not demonstrate live continual adaptation on protected evaluation labels. Some strategies may remain ineligible and retain the baseline.
- Daily-block bootstrap averages repeated training-seed outcomes by source; a single temporal dataset and limited blocks restrict uncertainty interpretation. It does not eliminate historical holdout exposure or model-selection uncertainty.
- Removing one fitting-prefix-constant feature is a small feature ablation. Probability/utility equality must be measured and does not establish broad feature-selection benefits.
- Utilities are research assumptions, not monetary profit. End-to-end utility combines initial Defender utility,10 per completed Court discovery of initially missed fraud, and all investigation costs. Court's own+3 flagged-fraud reward is a separate training objective.
- No joint-training convergence, game equilibrium or universal hybrid superiority is claimed. Training peak RSS was not instrumented.
'''
    (out/'research_limitations.md').write_text(limitations)
    conclusions='# Executed research conclusions\n\nActual PPO and recurrent Court policies were trained for three seeds under both feature configurations. See the CSV tables for results, including failures to outperform simpler baselines. Verified-only refits were executed with explicit eligibility and calibration gates.\n\n'+comparison.to_string(index=False)+'\n\nPaired utility differences (95% daily-block bootstrap):\n'+pd.DataFrame(statistical).to_string(index=False)+'\n\nA positive point estimate alone is not evidence of superiority; inspect the paired interval and the limitations. No policy/model changes were made after the final protocol freeze.\n'
    (out/'research_conclusions.md').write_text(conclusions)
    training_figures(out);final_figures(final,paths,out)
    write_report(out/'final_research_report.html','Hybrid adversarial RL — executed final research report',{'Research design':pd.DataFrame([dict(question='Does trained adversarial/Court RL and verified-only adaptation improve over simple baselines?',data='IEEE-CIS;431 vs430 predictors',chronology='45/10/15/10/20 exact saved windows',threat_model='assumption-based amount ±5%; strict blocked',initial_defender='fit-only LightGBM plus historical calibration; threshold1/6',attacker='11-observation tanh-Gaussian PPO, one terminal decision query',court='11-observation GRU PPO, matured origin utilities,180+20 budget',continual='scheduled verified-only refits; calibration before deployment',final='frozen policies/models; historically baseline-evaluated holdout disclosed')]),'Final matrix':final,'Original/Cleaned summary':comparison,'Paired uncertainty':pd.DataFrame(statistical),'Attacker validation baselines':a,'Court validation baselines':c,'Continual comparison':cl,'Resource usage':pd.DataFrame(resources)},limitations)
    # Add actual figures and links to the newly created final report only.
    report=out/'final_research_report.html';page=report.read_text();links=''.join(f'<h2>{p.stem}</h2><img style="max-width:100%" src="{p.name}" alt="{p.stem}">' for p in out.glob('*.png'))
    links+='<h2>Artifacts</h2><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in out.iterdir() if p.suffix in ('.csv','.json','.md'))+'</ul>'
    report.write_text(page.replace('</html>',links+'</html>'))
    check_protected()
    manifest=dict(frozen_protocol_sha256=sha(frozen),tests=tests.testsRun,all_tests_passed=True,python=sys.version,platform=platform.platform(),package_versions={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','pyarrow','lightgbm','gymnasium')},git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),git_dirty=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for root in ('rl_agents','rl_environment','continual_learning','scripts','configs') for p in (ROOT/root).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')},protected_prior_artifacts_unchanged=True,artifacts={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('final_reproducibility_manifest.json','run_manifest.json')})
    (out/'final_reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2))
    return dict(status='PASS',final_experiments=len(final),tests=tests.testsRun,final_protocol_frozen=True,holdout_historically_pristine=False)
