"""Phase implementations and scientific artifact gates."""
import json,time,sys,subprocess,datetime
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing.temporal_feature_audit import sha
ROOT=Path(__file__).resolve().parents[1]

def check_protected():
    saved=json.loads((ROOT/'reports/execution_control/protected.json').read_text())
    for p,h in saved.items():
        if sha(ROOT/p)!=h:raise ValueError('Protected artifact changed: '+p)

def write_report(out,title,frames,notes):
    import html
    body='<h1>'+html.escape(title)+'</h1><p>'+html.escape(notes)+'</p>'
    for label,frame in frames.items():body+='<h2>'+html.escape(label)+'</h2><div>'+frame.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</div>'
    (out).write_text('<!doctype html><html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1250px;margin:30px auto}div{overflow:auto}td,th{padding:6px;border:1px solid #ccd}table{border-collapse:collapse;font-size:13px}h1,h2{color:#175973}</style>'+body+'</html>')

def phase4a(config,out,resume):
    from rl_agents.attacker.train_ppo import train
    from rl_agents.attacker.attacker_sampler import AttackData
    from rl_agents.attacker.evaluate_ppo import evaluate,aggregate,load_policy
    runs=[];evaluation=[];summaries=[];training=[]
    started=time.monotonic()
    for version in config['versions']:
        for seed in config['seeds']:
            for bandit in (False,True):
                if time.monotonic()-started>config['resource_budget']['per_phase_wall_minutes']*60:raise TimeoutError('Phase wall-time budget reached; checkpoints preserved')
                result=train(version,seed,config,resume=resume,bandit=bandit);runs.append(result)
                runpath=Path(result['best_checkpoint']).parent
                training.append(pd.read_csv(runpath/'training_metrics.csv'))
                data=AttackData(version,'rl_validation')
                frame=evaluate(data,load_policy(result['best_checkpoint']),result['policy'],seed,bandit)
                evaluation.append(frame);summaries.append(dict(version=version,seed=seed,policy=result['policy'],**aggregate(frame)))
            data=AttackData(version,'rl_validation')
            for policy in ('identity','random_uniform','fixed_decrease','fixed_increase'):
                frame=evaluate(data,policy=policy,seed=seed);evaluation.append(frame);summaries.append(dict(version=version,seed=seed,policy=policy,**aggregate(frame)))
    metrics=pd.concat(training,ignore_index=True);ev=pd.concat(evaluation,ignore_index=True);comparison=pd.DataFrame(summaries)
    metrics.to_csv(out/'attacker_training_metrics.csv',index=False);ev.to_csv(out/'attacker_validation_results.csv',index=False);comparison.to_csv(out/'attacker_baseline_comparison.csv',index=False)
    ev[['version','seed','policy','transactionid','final_action','actions','status']].to_csv(out/'attacker_action_distribution.csv',index=False)
    comparison.assign(ablation='unchanged Phase3.7 reward; PPO vs single-decision bandit vs fixed references').to_csv(out/'attacker_ablation.csv',index=False)
    (out/'attacker_checkpoints.json').write_text(json.dumps(runs,indent=2))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/phase4','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Phase4 tests failed')
    assert all(r['gradient_updates']>0 and Path(r['best_checkpoint']).exists() for r in runs)
    assert (ev.query_count<=1).all();check_protected()
    write_report(out/'attacker_training_report.html','Phase 4A — actual PPO and contextual-bandit training',{'Training inventory':pd.DataFrame([{k:r[k] for k in ['version','seed','policy','training_interactions','training_queries','unique_training_sources','gradient_updates','elapsed_seconds','best_checkpoint']} for r in runs]),'Independent validation':comparison},'Mode B assumptions remain. Strict causal controllability is BLOCKED. PPO uses tanh-Normal Jacobian-correct log probabilities, GAE and clipped updates. Validation selects only among predeclared saved checkpoints; no validation gradients. Identity can yield reward0 and is reported separately. Policy selection counts identities as zero flips over all originally detected fraud. Contextual bandit uses one sampled proposal then forced submission; planned terminal-query budget matches PPO. Repeated offline sources are not independent evaluation observations.')
    return dict(status='PASS',actual_trained_runs=len(runs),training_interactions=sum(r['training_interactions'] for r in runs))

def execute_phase(phase,config,resume):
    root=ROOT/'reports'/('phase'+phase if phase!='6' else 'final_research');root.mkdir(parents=True,exist_ok=True)
    statusfile=root/'run_manifest.json'
    import hashlib
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if statusfile.exists():
        previous=json.loads(statusfile.read_text())
        if previous.get('status')=='PASS':
            if not resume:raise FileExistsError('Completed phase; use --resume to verify and skip')
            if previous['config_hash']!=config_hash:raise ValueError('Changed configuration cannot reuse completed phase')
            for path,h in previous['artifacts'].items():
                if sha(ROOT/path)!=h:raise ValueError('Completed artifact changed: '+path)
            return dict(status='PASS',resumed_completed=True)
    predecessor={'4b':'phase4a','4c':'phase4b','5':'phase4c','6':'phase5'}.get(phase)
    if predecessor:
        p=ROOT/'reports'/predecessor/'run_manifest.json'
        if not p.exists() or json.loads(p.read_text()).get('status')!='PASS':raise ValueError('Required previous phase has not passed: '+predecessor)
    check_protected();(root/'executed_config.json').write_text(json.dumps(config,indent=2));start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        if phase=='4a':result=phase4a(config,root,resume)
        else:
            from scripts.remaining_later_phases import run_phase
            result=run_phase(phase,config,root,resume)
        check_protected()
    except Exception as error:
        result=dict(status='PARTIAL',error=repr(error))
        import traceback
        (root/'failure.log').write_text(traceback.format_exc())
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in root.rglob('*') if p.is_file() and p!=statusfile}
    statusfile.write_text(json.dumps(dict(**result,phase=phase,config_hash=config_hash,started_at=start,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts),indent=2))
    return result
