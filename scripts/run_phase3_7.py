"""Execute predeclared Mode B episodes on RL Development only; no PPO training."""
import sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from rl_environment.attack_plausibility import AttackPlausibility
from rl_environment.assumption_attack_adapter import engine_for,prepare_episode,proposal,AssumptionAttackAdapter,AssumptionFraudSimulator
from rl_environment.attacker_reward import assumption_reward
from rl_environment.defender import FrozenDefender
from rl_environment.transaction_stream import TransactionStream,InternalTransaction
from preprocessing.temporal_feature_audit import membership,sha

def safe_rate(n,d):return n/d if d else None

def summarize(frame,source_fraud,eligible):
    rows=[]
    for (version,epsilon,policy),g in frame.groupby(['dataset_version','epsilon','policy'],sort=True):
        modified=g.status.eq('ASSUMPTION_VALID');detected=g.original_decision.eq('FLAG');approved=g.modified_decision.eq('APPROVE');valid=modified|g.status.eq('UNCHANGED')
        denominator=int((modified&detected).sum());success=int((modified&detected&approved).sum())
        rows.append(dict(dataset_version=version,epsilon=epsilon,policy=policy,source_fraud_transactions=source_fraud,eligible_fraud=eligible,attack_attempts=len(g),accepted_nonidentity=int(modified.sum()),unchanged=int(g.status.eq('UNCHANGED').sum()),assumption_acceptance_rate=safe_rate(int(modified.sum()),len(g)),rejected=int((~valid).sum()),rejection_rate=float((~valid).mean()),original_flag_rate=float(detected.mean()),modified_flag_rate=safe_rate(int((valid&g.modified_decision.eq('FLAG')).sum()),int(valid.sum())),originally_detected_fraud=int(detected.sum()),accepted_originally_detected_denominator=denominator,successful_decision_flips=success,conditional_evasion_rate=safe_rate(success,denominator),accepted_attack_approved=int((modified&approved).sum()),overall_accepted_attack_evasion_rate=safe_rate(int((modified&approved).sum()),int(modified.sum())),mean_probability_change=float(g.loc[valid,'probability_change'].mean()),mean_absolute_relative_change=float(g.loc[modified,'absolute_relative_change'].mean()) if modified.any() else 0.,mean_cost=float(g.perturbation_cost.mean()),mean_reward=float(g.reward.mean()),max_query_count=int(g.query_count.max()),mean_modified_anomaly=float(g.loc[valid,'modified_anomaly_score'].mean()),soft_outside_q01_q99=int(g.modified_outside_q01_q99.sum()),interpretation='ASSUMPTION-BASED IEEE-CIS RESULTS; not real-world validated ASR'))
    return pd.DataFrame(rows)

def equal(a,b):
    if a is None or b is None:return a is None and b is None
    if isinstance(a,float) and np.isnan(a):return isinstance(b,float) and np.isnan(b)
    return a==b

def run_experiments(out,source,protocol,config,plausibility):
    w=protocol['windows']['rl_development'];dataset=ds.dataset(source)
    filt=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['max_time'])
    table=dataset.to_table(filter=filt).sort_by([('transactiondt','ascending'),('transactionid','ascending')])
    meta=table.select(['transactionid','transactiondt']).to_pandas()
    assert len(table)==w['rows'] and membership(meta)==w['ordered_id_sha256']
    fraud=table.filter(__import__('pyarrow').compute.equal(table['isfraud'],1)).to_pylist()
    amounttype=table.schema.field('transactionamt').type
    assert str(amounttype)=='decimal128(8, 3)'
    gate=engine_for(plausibility,config,.05,amounttype.precision,amounttype.scale)
    eligible=[r for r in fraud if gate.valid_amount(r['transactionamt'])]
    pd.DataFrame([dict(transactionid=r['transactionid'],eligible=gate.valid_amount(r['transactionamt']),reason='positive finite exact Decimal' if gate.valid_amount(r['transactionamt']) else 'invalid amount') for r in fraud]).to_csv(out/'source_eligibility.csv',index=False)
    if not eligible:raise ValueError('No eligible development fraud; no thresholds loosened')
    outputs=[];checks=[];t=time.monotonic()
    for version,artifact in [('original_431','original'),('cleaned_prefix_430','cleaned')]:
        defender=FrozenDefender(ROOT/'models/temporal_defender'/artifact)
        features=[{n:r[n] for n in defender.feature_names} for r in eligible]
        original_p=defender.predict_many(features) # evaluator only, never passed to action selection
        for epsilon in config['epsilon_sensitivity']:
            for policy in config['experiment']['policies']:
                engines=[];results=[];terminal=[];positions=[];actions=[]
                for i,r in enumerate(eligible):
                    # Include protected metadata in internal engine copy for explicit checks.
                    source_record=dict(features[i],transactionid=r['transactionid'],transactiondt=r['transactiondt'],isfraud=r['isfraud'])
                    e=engine_for(plausibility,config,epsilon,amounttype.precision,amounttype.scale)
                    a=proposal(policy,r['transactionid'],config['experiment']['seed'])
                    result=prepare_episode(e,source_record,a)
                    assert all(equal(v,result['modified_transaction'][n]) for n,v in source_record.items() if n!='transactionamt')
                    engines.append(e);results.append(result);actions.append(a)
                    if result['status'] in ('ASSUMPTION_VALID','UNCHANGED'):
                        terminal.append({n:result['modified_transaction'][n] for n in defender.feature_names});positions.append(i)
                # Each proposal is fixed before the one terminal query. No search or best-of selection.
                terminal_p=defender.predict_many(terminal) if terminal else []
                by_index=dict(zip(positions,terminal_p))
                for i,(r,e,result,a) in enumerate(zip(eligible,engines,results,actions)):
                    p=by_index.get(i);decision=('FLAG' if p>defender.threshold else 'APPROVE') if p is not None else None
                    reward=e.resolve(decision) if p is not None else assumption_reward(result['status'],cost=e.cost)
                    modified=result['modified_transaction']['transactionamt'];relative=float((modified-r['transactionamt'])/r['transactionamt'])
                    diag=result['diagnostics'] or {}
                    row=dict(dataset_version=version,transactionid=r['transactionid'],transactiondt=r['transactiondt'],policy=policy,original_amount=str(r['transactionamt']),modified_amount=str(modified),continuous_action=a,action_history=json.dumps(e.history),epsilon=epsilon,modification_percentage=relative*100,absolute_relative_change=abs(relative),status=result['status'],original_decision='FLAG' if original_p[i]>defender.threshold else 'APPROVE',modified_decision=decision,original_probability_evaluator_only=float(original_p[i]),modified_probability_evaluator_only=float(p) if p is not None else None,probability_change=float(p-original_p[i]) if p is not None else None,perturbation_cost=e.cost,reward=reward,query_count=e.queries,original_anomaly_score=diag.get('original_anomaly_score'),modified_anomaly_score=diag.get('modified_anomaly_score'),modified_outside_q01_q99=diag.get('modified_outside_q01_q99',False),rejection_reason=result['reason'] if result['status'] not in ('ASSUMPTION_VALID','UNCHANGED') else '',assumption_based_validity=result['status'],dependency_status='UNKNOWN_DEPENDENCY',label_contract=config['label_contract'],causal_validity=False)
                    outputs.append(row)
                    checks.append(dict(dataset_version=version,transactionid=r['transactionid'],policy=policy,epsilon=epsilon,status=result['status'],hard_finite_positive_decimal=e.valid_amount(modified),hard_relative_budget=abs((modified-r['transactionamt'])/r['transactionamt'])<=e.epsilon,hard_non_target_and_metadata_unchanged=True,original_anomaly_score=diag.get('original_anomaly_score'),modified_anomaly_score=diag.get('modified_anomaly_score'),original_outside_fit_range=diag.get('original_outside_fit_range'),modified_outside_fit_range=diag.get('modified_outside_fit_range'),modified_outside_q01_q99=diag.get('modified_outside_q01_q99'),conditional_diagnostics=json.dumps(diag.get('conditional',{})),statistical_diagnostics_reject=False,causal_validity=False))
                print(version,epsilon,policy,'episodes',len(eligible),flush=True)
    frame=pd.DataFrame(outputs);frame.to_csv(out/'attack_results.csv',index=False,float_format='%.17g')
    pd.DataFrame(checks).to_csv(out/'plausibility_validation.csv',index=False)
    summary=summarize(frame,len(fraud),len(eligible));summary.to_csv(out/'attack_summary.csv',index=False)
    a=frame[frame.dataset_version=='original_431'];b=frame[frame.dataset_version=='cleaned_prefix_430']
    pair=a.merge(b,on=['transactionid','epsilon','policy'],suffixes=('_original','_cleaned'),validate='one_to_one')
    assert (pair.continuous_action_original==pair.continuous_action_cleaned).all() and (pair.modified_amount_original==pair.modified_amount_cleaned).all()
    diffs=[]
    for (epsilon,policy),g in pair.groupby(['epsilon','policy']):
        delta=(g.modified_probability_evaluator_only_original-g.modified_probability_evaluator_only_cleaned).abs()
        diffs.append(dict(epsilon=epsilon,policy=policy,paired_rows=len(g),max_absolute_probability_difference=float(delta.max()),mean_absolute_probability_difference=float(delta.mean()),decision_disagreements=int((g.modified_decision_original!=g.modified_decision_cleaned).sum()),identical_actions=True,identical_modified_amounts=True))
    pd.DataFrame(diffs).to_csv(out/'paired_model_comparison.csv',index=False)
    (out/'experiment_execution.json').write_text(json.dumps(dict(source_window=w,source_fraud=len(fraud),eligible_fraud=len(eligible),episodes=len(frame),elapsed_seconds=time.monotonic()-t,holdout_reads=False,validation_reads=False,amount_arrow_type=str(amounttype),defender_numeric_input='saved float32 encoding; Decimal retained until existing encode()',candidate_search=False),indent=2))

def run_smoke(out,source,protocol,config,plausibility):
    from rl_environment.court_observation import CourtObservationV2
    w=protocol['windows']['rl_development'];data=ds.dataset(source)
    filt=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['decision_cutoff'])
    ids=data.to_table(columns=['transactionid','transactiondt'],filter=filt).sort_by([('transactiondt','ascending'),('transactionid','ascending')]).slice(0,config['smoke']['decisions'])
    summaries=[]
    cfg=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text())
    for version,artifact in [('original_431','original'),('cleaned_prefix_430','cleaned')]:
        defender=FrozenDefender(ROOT/'models/temporal_defender'/artifact)
        table=data.to_table(columns=defender.feature_names+['transactionid','transactiondt','isfraud'],filter=filt&ds.field('transactionid').isin(ids['transactionid'])).sort_by([('transactiondt','ascending'),('transactionid','ascending')])
        records=[InternalTransaction(r['transactionid'],r['transactiondt'],{n:r[n] for n in defender.feature_names},r['isfraud']) for r in table.to_pylist()]
        adapter=AssumptionAttackAdapter(plausibility,config,config['smoke']['epsilon'],config['smoke']['policy'])
        sim=AssumptionFraudSimulator(TransactionStream.synthetic(records),defender,cfg,adapter)
        sim.defender_deployed_at=protocol['windows']['defender_calibration']['max_time']
        sim.observations=CourtObservationV2(records[-1].transaction_time)
        rng=np.random.default_rng(42)
        while not sim.done:
            obs=sim.observation();assert obs.shape==(11,) and np.isfinite(obs).all()
            sim.step(int(obs[3]>0 or rng.random()<.005))
        horizon=sim.summary();sim.drain_pending();mature=sim.summary()
        assert sim.now<=w['max_time']
        memory=sim.memory.get_recent_verified(sim.now,sim.memory.memory_size())
        for r in memory:
            assert 'hidden_label' not in r and r['completion_time']<=sim.now
            provenance=r['original_modified_metadata']
            assert provenance['source_transaction_id']==r['transaction_id']
            if provenance['modified']:
                assert provenance['label_contract']==config['label_contract']
                assert str(r['features']['transactionamt'])==provenance['modified_amount']
                assert provenance['source_feature_vector']['transactionamt']!=r['features']['transactionamt']
        public_rows=[]
        for tid,p in adapter.provenance.items():
            public_rows.append({k:v for k,v in p.items() if k!='source_feature_vector'})
        pd.DataFrame(public_rows).to_csv(out/f'{artifact}_smoke_attack_provenance.csv',index=False)
        pd.DataFrame(sim.actions).to_csv(out/f'{artifact}_smoke_court_actions.csv',index=False)
        # JSONL is evaluator-only completed memory, with original/modified vectors and contract.
        with (out/f'{artifact}_verified_memory.jsonl').open('w') as f:
            for r in memory:f.write(json.dumps(r,default=str,allow_nan=True)+'\n')
        summaries.append(dict(dataset_version=version,source_ids_sha256=membership(ids.to_pandas()),first_time=records[0].transaction_time,last_time=records[-1].transaction_time,maturity_end=sim.now,source_fraud=sum(r.hidden_label for r in records),horizon=horizon,mature=mature,verified_modified=sum(r['original_modified_metadata']['modified'] for r in memory),court_observation_dimension=11,attacker_queries=sum(e.queries for e in adapter.episodes.values()),provenance_verified=True))
        print(version,'end-to-end smoke done',flush=True)
    (out/'hybrid_smoke_summary.json').write_text(json.dumps(summaries,indent=2))

def main():
    out=ROOT/'reports/phase3_7';out.mkdir(parents=True,exist_ok=True)
    if not (out/'protected_before.json').exists():
        roots=['models/temporal_defender','reports/lightgbm_baseline','reports/rl_environment','reports/phase3_5','reports/phase3_6','data/processed']
        files=[p for root in roots for p in (ROOT/root).rglob('*') if p.is_file()]
        (out/'protected_before.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
    if not (out/'started_at.txt').exists():
        import datetime
        (out/'started_at.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    config=yaml.safe_load((ROOT/'configs/assumption_attack.yaml').read_text())
    protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())
    source=ROOT/'data/processed/train_original.parquet'
    plausibility=AttackPlausibility.fit_parquet(source,protocol)
    (out/'plausibility_model.json').write_text(json.dumps(plausibility.document,indent=2))
    (out/'attack_constraints.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    run_experiments(out,source,protocol,config,plausibility)
    run_smoke(out,source,protocol,config,plausibility)
if __name__=='__main__':main()
