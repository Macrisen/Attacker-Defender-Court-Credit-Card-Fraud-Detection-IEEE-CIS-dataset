"""Build report from actual artifacts; fail on failed tests or changed protected files."""
import sys,json,html,platform,datetime,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing.temporal_feature_audit import sha
from rl_environment.feature_registry import feature_group
from rl_environment.court_observation import CourtObservationV2

def main():
    out=ROOT/'reports/phase3_6'
    selection=pd.read_csv(out/'feature_selection_temporal.csv')
    pairs=pd.read_csv(out/'correlation_pairs.csv')
    metrics=pd.read_csv(out/'temporal_defender_comparison.csv')
    tests=pd.read_csv(out/'phase3_6_test_results.csv')
    credit=pd.read_csv(out/'court_reward_credit_validation.csv')
    attacks=pd.read_csv(out/'attacker_action_reward_validation.csv')
    assert set(tests.status)=={'PASS'} and set(credit.status)=={'PASS'} and set(attacks.reward_test)=={'PASS'}
    protected=json.loads((out/'protected_before.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    manifest=json.loads((out/'selection_manifest.json').read_text())
    protocol=json.loads((ROOT/'reports/phase3_5/temporal_split_protocol.json').read_text())
    verified_protocol=dict(protocol,status='EXACT_MEMBERSHIP_VERIFIED_PHASE3_6',new_defender_status='BOTH_TEMPORAL_MODELS_FITTED_AND_VERIFIED',baseline_reuse='NOT_USED; historical baseline preserved',holdout_policy='Reserved from new fitting/tuning/evaluation. Source values read only for lossless export/projection integrity. Previously evaluated by legacy baseline.',source_protocol_sha256=sha(ROOT/'reports/phase3_5/temporal_split_protocol.json'))
    (out/'temporal_split_protocol.json').write_text(json.dumps(verified_protocol,indent=2))
    smoke=json.loads((out/'development_smoke.json').read_text())
    # Connected correlation groups are review sets, not deletion equivalence classes.
    adjacency={}
    for r in pairs[pairs.reliable].itertuples():
        adjacency.setdefault(r.feature_a,set()).add(r.feature_b);adjacency.setdefault(r.feature_b,set()).add(r.feature_a)
    groups=[];seen=set()
    for start in sorted(adjacency):
        if start in seen:continue
        queue=[start];g=[]
        while queue:
            n=queue.pop()
            if n in seen:continue
            seen.add(n);g.append(n);queue+=list(adjacency[n]-seen)
        groups.append(dict(group=len(groups)+1,count=len(g),features=sorted(g),decision='REVIEW; retain all unless exact-duplicate/constant rule independently applies',limitation='Connected correlation components are not all-pairs equivalence'))
    (out/'redundant_feature_groups.json').write_text(json.dumps(groups,indent=2))
    selection['group']=selection.feature.map(feature_group)
    group_summary=selection.groupby(['group','decision']).size().unstack(fill_value=0)
    group_summary.to_csv(out/'feature_group_summary.csv')
    settings=dict(version='phase3_6_v2',executed=dict(seed=42,defender=yaml.safe_load((ROOT/'configs/defender_temporal.yaml').read_text()),feature_selection=manifest['version'],court_observation=CourtObservationV2(0).names,court_action='Discrete(2)',court_rewards={'investigation_cost':1,'missed_fraud':10,'flagged_fraud':3,'legitimate_bonus':0},budget={'window_size':10000,'total':200,'court':180,'audit':20},verification={'distribution':'Uniform','days':[1,3],'seed':43},gamma_per_day=.99,smoke_decisions_per_version=10000,learner='one synthetic deterministic MC value-regression update; no real policy optimization',attacker={'interface':'Box(k) authorized controls only','stop':'all abs(action)<=1e-6','max_steps':3,'max_terminal_queries':1,'real_training':'BLOCKED','real_asr':None},protected_windows=protocol['windows']),proposed_not_executed=dict(court='recurrent episodic policy with fixed-policy collection and mature MC returns',continual_defender='not implemented',full_ppo_dqn_training=False,real_attacker_controls='none authorized'))
    (out/'environment_config_v2.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    readiness='''# Phase 4 readiness

| Component | Verdict | Evidence / remaining work |
|---|---|---|
| Attacker Continuous PPO, IEEE-CIS | BLOCKED | Zero verified nonidentity controls; real ASR undefined. Synthetic Box(k), STOP, budgets and reward checks pass. |
| Court RL | PARTIAL | Mature finite-episode adapter, manually verified MC targets, deterministic synthetic gradient update and independent 10k-row replays pass. Recurrent policy, policy optimization and efficacy comparison are not implemented. No claim that compact observation is Markov. |
| Temporal Defender | READY | Both fresh models fit only Defender Fit, calibrate only Defender Calibration, with fit-only vocabularies/selection. Readiness is conditional on offline label availability and unknown feature temporal semantics. |
| Original / Cleaned comparison | READY | 431 vs 430 predictors, identical IDs/windows/budgets/tuning schedule. Historical 418 preserved. |
| Continual learning | PARTIAL | Verified Memory exists; retraining schedule, delayed-label calibration, forgetting checks and versioned deployment are not implemented. |
| Scientific evaluation | PARTIAL | Independent development/validation metrics and provenance available. Real adversarial validity, actual label-arrival times and final policy evaluation remain unresolved. |

Next: implement a recurrent Court-only fixed-policy episodic learner, freeze optimization budgets and comparison rules, and evaluate on RL Validation. Use the tested maturation adapter; do not train from Phase 3 arrival rewards or use the legacy one-step DQN target as a proven Bellman target. Keep Attacker blocked until there is grounded control/dependency evidence. No full PPO/DQN was trained.

The final holdout was previously evaluated by the legacy baseline. It was not used for new selection, calibration, threshold choice or model evaluation. Lossless export/projection checks necessarily read its source values; that is data integrity processing, not tuning. No observed label-discovery timestamps exist; supervised fit/calibration assume labels available before deployment.
'''
    (out/'phase4_readiness.md').write_text(readiness)
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].hist(selection.missing_percentage,bins=20,color='#267b9f');axs[0].set(xlabel='Fit-prefix missing %',ylabel='Predictors',title='Historical prefix only')
    selection.decision.value_counts().plot.bar(ax=axs[1],color=['#267b9f','#d99b32','#a54d4d']);axs[1].set_title('Conservative selection');axs[1].tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(out/'feature_audit.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3));colors=['#267b9f','#58a3b8','#bd8631','#d3ac68','#999999']
    for i,(name,w) in enumerate(protocol['windows'].items()):
        ax.barh(i,(w['max_time']-w['min_time'])/86400,left=w['min_time']/86400,color=colors[i]);ax.text((w['min_time']+w['max_time'])/2/86400,i,f"{w['rows']:,}",ha='center',va='center',fontsize=9)
    ax.set(yticks=range(5),yticklabels=list(protocol['windows']),xlabel='TransactionDT / 86,400 (relative day)');ax.invert_yaxis();fig.tight_layout();fig.savefig(out/'temporal_partitions.png',dpi=150);plt.close(fig)
    sections=[]
    def section(title,body):sections.append('<section><h2>'+html.escape(title)+'</h2>'+body+'</section>')
    def table(df):return df.to_html(index=False,escape=True,float_format=lambda x:f'{x:.6f}')
    section('Execution and data provenance','<p>Original PostgreSQL export, fit-prefix audit, two fresh model fits, calibration, independent development/validation evaluation, paired 10,000-decision Court replays, synthetic learner update, regression tests and reports executed. No full PPO/DQN and no source table modifications.</p>'+table(pd.read_json(out/'postgres_schema_verification.json')[['table','rows','column_count']])+'<p>Original export and cleaned projection verify all values, NULLs and ID/time alignment. Protected previous artifacts remain byte-identical.</p>')
    section('Temporal protocol','<img src="temporal_partitions.png" alt="Exact chronological windows"><p>Selection and mappings: Defender Fit only. Calibration: subsequent calibration window only. RL Development: future policy optimization. RL Validation: evaluation/model-policy selection. Final holdout reserved from NEW tuning; historically baseline-evaluated.</p>')
    section('Feature selection',f'<p>Original predictors: 431. New version: {manifest["version"]}. Missingness &gt;90%: {int((selection.missing_percentage>90).sum())}. Correlation pairs |r|≥.98: {len(pairs)}. Review components: {len(groups)}. All-NULL: {int((selection.unique_count==0).sum())}; constant non-NULL: {int((selection.unique_count==1).sum())}; near-constant: {int((selection.dominant_value_percentage>99.5).sum())}. These categories may overlap.</p><img src="feature_audit.png" alt="Actual missingness and selection distributions">'+table(selection[selection.decision=='DROP'][['feature','unique_count','missing_percentage','decision','reason']])+'<p>High correlations and high missingness alone never trigger removal. Correlation groups may contain noncorrelated endpoints; they are not equivalence classes. All non-DROP features are retained, including REVIEW.</p>'+group_summary.to_html())
    section('New temporal Defender results',table(metrics)+'<p>Metrics use calibrated predictions at strict p&gt;1/6. AP is average precision, not trapezoidal PR area. Identical tuning budgets: max2000 rounds, 100-round early stopping inside fit, seed42, 4 threads. Calibration is selected/refitted inside calibration only. No RL performance improvement is claimed.</p><p>Utility derivation: FLAG=15p−5, APPROVE=1−21p; threshold=1/6.</p>')
    section('Calibration diagnostics',table(pd.read_csv(out/'calibration_diagnostics.csv')))
    section('Attacker continuous action and reward',table(attacks)+'<p>V2 Box(k) uses only authorized control dimensions. Zero-vector deadband means STOP; third step submits automatically. Legacy explicit STOP coordinate remains regression-only. Real eligible modified attacks: 0. Real ASR: undefined. Scenario B amount edits remain UNKNOWN_VALIDITY; no invalid counterfactual queries.</p>')
    section('Court state, action and reward','<p>11 finite deterministic features: baseline10 plus known decision-horizon time remaining. Full causal history prefixes retained without later labels. Discrete(2): SKIP / INVESTIGATE; budget-exhausted requests blocked without charge. Cost−1, completed missed fraud+10, flagged fraud+3, legitimate+0. Research utilities, not proven financial costs.</p><p>Compact state is not asserted Markov. Queue age/risk detail and recurrent policy integration remain future work.</p>')
    section('Delayed-credit learner semantics','<p>Reuse origin ledger; release complete fixed-policy finite episodes only after all outcomes and the full 3-day guard. MC target sums discounted originated utilities from the current action onward. No one-step max-Q bootstrap and no legacy arrival reward mixing. Truncated/incomplete episodes are censored. Earlier pending bonuses are not counted again. Synthetic update is value regression, not policy training.</p>'+table(credit)+'<pre>'+html.escape((out/'synthetic_learner_update.json').read_text())+'</pre>')
    section('Independent chronological budget/verification smoke',table(pd.DataFrame([dict(version=s['dataset'],rows=s['replay_records'],start=s['first_time'],decision_end=s['last_decision_time'],maturity_end=s['label_maturity_end'],court=s['mature']['court_investigation_count'],audit=s['mature']['audit_investigation_count'],verified_fraud=s['mature']['fraud_verified'],verified_legitimate=s['mature']['legitimate_verified'],pending=s['mature']['investigations_pending']) for s in smoke]))+'<p>Same IDs, seed42, 180 Court +20 audit reserve, Uniform(1,3)-day delays. Different model decisions can change investigation/audit eligibility. Pending work survives budget boundaries; no policy update occurs in these real-data replays. Selected verified fraud rates are not unbiased population estimates.</p>')
    section('Tests and leakage checks',f'<p>{len(tests)} executed tests PASS. Includes all Phase 3 and 3.5 regressions, manual MC targets, real data alignment/provenance, exact fit-only category maps and future-poison selection isolation.</p>'+table(tests[['test','status','seconds']]))
    section('Limitations and Phase 4 readiness','<pre>'+html.escape(readiness)+'</pre><p>Feature temporal availability cannot be established from anonymous names. Calibration labels are assumed available before RL deployment; actual discovery times are absent. No causal modification contract has been established. Real Court efficacy and continual retraining remain untested.</p>')
    files=['feature_selection_temporal.csv','temporal_defender_comparison.csv','attacker_action_reward_validation.csv','court_reward_credit_validation.csv','phase3_6_test_results.csv','phase4_readiness.md','environment_config_v2.yaml','selection_manifest.json','run_manifest.json','development_smoke.json','correlation_pairs.csv','fit_window_stability.csv','cleaned_projection_validation.json']
    section('Artifacts and reproducibility','<ul>'+''.join(f'<li><a href="{f}">{f}</a></li>' for f in files)+'</ul><p><a href="../../rl_environment/PHASE3_6.md">Protocol and reproduction instructions</a></p><p>Methods: <a href="https://www.incompleteideas.net/book/bookdraft2018mar21.pdf">Sutton and Barto: Monte Carlo</a>; <a href="https://farama.org/Gymnasium-Terminated-Truncated-Step-API">Farama: termination/truncation</a>.</p>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 3.6 Validation</title><style>body{font:16px system-ui;margin:32px auto;max-width:1250px;color:#20303b;padding:0 20px}h1,h2{color:#155d77}section{margin:30px 0;overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd6db;padding:7px;text-align:left}th{background:#edf4f6}img{max-width:100%}pre{white-space:pre-wrap;font:14px ui-monospace;background:#f4f7f8;padding:16px}</style><h1>Phase 3.6 — Executed validation report</h1>'+''.join(sections)+'</html>'
    (out/'phase3_6_validation_report.html').write_text(page)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn','pyarrow','gymnasium','psycopg','PyYAML']}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    inventory={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
    sources=[p for folder in ('preprocessing','rl_agents','rl_environment','scripts','tests/phase3_6','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.md')]
    models={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/temporal_defender').rglob('*') if p.is_file()}
    run=dict(started_at=datetime.datetime.fromtimestamp((out/'protected_before.json').stat().st_mtime,datetime.timezone.utc).isoformat(),completed_at=now,python=sys.version,platform=platform.platform(),runtime_versions=versions,seeds={'model':42,'court_policy':42,'audit':42,'verification':43},dataset_versions=['original_431',manifest['version'],'historical_cleaned_418_preserved'],dataset_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data/processed').glob('*.parquet')},split_sha256=sha(ROOT/'reports/phase3_5/temporal_split_protocol.json'),model_hashes=models,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},protected_artifacts_unchanged=True,test_suite={'tests':len(tests),'PASS':len(tests)},artifact_inventory=inventory,holdout_tuning=False,holdout_integrity_export_reads=True,full_rl_training=False)
    (out/'run_manifest.json').write_text(json.dumps(run,indent=2))
    print(f'Report complete: {len(tests)} tests PASS, {len(metrics)} model/window metric rows')
if __name__=='__main__':main()
