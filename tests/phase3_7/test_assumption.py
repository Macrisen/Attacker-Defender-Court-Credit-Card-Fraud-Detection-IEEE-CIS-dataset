import copy,json,tempfile,unittest,hashlib
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from gymnasium.utils.env_checker import check_env
from rl_environment.attack_plausibility import AttackPlausibility
from rl_environment.assumption_attack_engine import AssumptionAttackEngine,AssumptionAttackerEnv,AttackStatus
from rl_environment.assumption_attack_adapter import engine_for,prepare_episode,proposal,AssumptionAttackAdapter,AssumptionFraudSimulator
from rl_environment.attacker_reward import assumption_reward
from rl_environment.transaction_stream import InternalTransaction,TransactionStream
from preprocessing.temporal_feature_audit import membership,sha
from scripts.run_phase3_7 import summarize
ROOT=Path(__file__).resolve().parents[2]
CONFIG=yaml.safe_load((ROOT/'configs/assumption_attack.yaml').read_text())

def fitting_fixture():
    f=pd.DataFrame(dict(transactionid=range(300),transactiondt=range(300),transactionamt=[Decimal(str(x)) for x in range(1,301)],productcd=['W']*300,card4=['visa']*300,card6=['debit']*300))
    protocol={'windows':{'defender_fit':dict(rows=300,min_time=0,max_time=299,ordered_id_sha256=membership(f))}}
    return f,protocol

def plausibility():
    f,p=fitting_fixture();return AttackPlausibility.fit_frame(f,p)

def engine(epsilon=.05,mode='ASSUMPTION_BASED'):
    return engine_for(plausibility(),CONFIG,epsilon,mode=mode)

def transaction():return dict(transactionamt=Decimal('100.000'),transactionid=7,transactiondt=1000,isfraud=1,productcd='W',card4='visa',card6='debit',v1=None)

class ToyDefender:
    feature_names=['transactionamt','productcd','card4','card6','v1']
    threshold=1/6;version='toy_fixture'
    def __init__(self):self.calls=0
    def decide(self,features):
        self.calls+=1;p=.8 if features['transactionamt']>=100 else .1
        return ('FLAG' if p>self.threshold else 'APPROVE',p)
    def predict_many(self,features):
        return np.array([self.decide(r)[1] for r in features])

class EngineTests(unittest.TestCase):
    def test_box_bounds_and_invalid_shape(self):
        for a in ([2.],[1.+1e-10],[np.nan],[np.inf],[0,0],['bad']):
            e=engine();e.reset(transaction());self.assertEqual(e.apply(a)['status'],'INVALID_ACTION');self.assertEqual(e.queries,0)
    def test_decimal_exact_mapping_and_rounding(self):
        e=engine();e.reset(transaction());r=e.apply([.5]);self.assertEqual(r['modified_transaction']['transactionamt'],Decimal('102.500'))
        t=transaction();t['transactionamt']=Decimal('0.011');e.reset(t);r=e.apply([1.]);self.assertEqual(r['modified_transaction']['transactionamt'],Decimal('0.011'))
        self.assertEqual(r['status'],'UNCHANGED')
    def test_absolute_not_cumulative_mapping(self):
        e=engine();e.reset(transaction());e.apply([-.5]);r=e.apply([.5]);self.assertEqual(r['modified_transaction']['transactionamt'],Decimal('102.500'))
    def test_invalid_amounts(self):
        for amount in (None,100.,Decimal('-1'),Decimal('0'),Decimal('NaN'),Decimal('Infinity'),Decimal('100000'),Decimal('1.0001')):
            t=transaction();t['transactionamt']=amount;e=engine();e.reset(t)
            self.assertEqual(e.apply([.5])['status'],'ASSUMPTION_REJECTED');self.assertEqual(e.queries,0)
    def test_precision_overflow_rejected(self):
        t=transaction();t['transactionamt']=Decimal('99999.999');e=engine();e.reset(t)
        self.assertEqual(e.apply([1])['status'],'ASSUMPTION_REJECTED')
    def test_positive_and_relative_budget_grid(self):
        for eps in (.01,.03,.05):
            for amount in ('0.001','0.011','1.234','100.001','6000.123'):
                for action in (-1,-.999,.333,1):
                    t=transaction();t['transactionamt']=Decimal(amount);e=engine(eps);e.reset(t);r=e.apply([action]);new=r['modified_transaction']['transactionamt']
                    self.assertGreater(new,0);self.assertLessEqual(abs((new-t['transactionamt'])/t['transactionamt']),Decimal(str(eps)))
    def test_stop_and_identity_reward(self):
        e=engine();e.reset(transaction());r=e.apply([0]);self.assertEqual(r['status'],'UNCHANGED');self.assertTrue(e.done);self.assertEqual(e.resolve('APPROVE'),0)
    def test_deadband_retains_current(self):
        e=engine();e.reset(transaction());e.apply([-.5]);r=e.apply([1e-7]);self.assertTrue(e.done);self.assertEqual(r['modified_transaction']['transactionamt'],Decimal('97.500'))
    def test_max_steps_and_single_query(self):
        e=engine();e.reset(transaction())
        for _ in range(3):r=e.apply([-.5])
        self.assertTrue(e.done);self.assertEqual(e.resolve('APPROVE'),.95)
        with self.assertRaises(ValueError):e.resolve('APPROVE')
        with self.assertRaises(RuntimeError):e.apply([0])
    def test_epsilon_and_cost(self):
        for eps in (.01,.03,.05):
            e=engine(eps);e.reset(transaction());r=e.apply([1]);self.assertEqual(r['modified_transaction']['transactionamt'],Decimal('100')*(1+Decimal(str(eps))));self.assertAlmostEqual(e.cost,.1)
    def test_immutable_and_label_contract(self):
        t=transaction();before=copy.deepcopy(t);e=engine();r=prepare_episode(e,t,-.5)
        self.assertEqual(t,before)
        for k in t:
            if k!='transactionamt':self.assertEqual(t[k],r['modified_transaction'][k])
        self.assertEqual(r['label_contract'],CONFIG['label_contract']);self.assertFalse(r['causal_validity']);self.assertEqual(r['dependency_status'],'UNKNOWN_DEPENDENCY')
    def test_strict_separation(self):
        e=engine(mode='STRICT');e.reset(transaction());self.assertEqual(e.apply([-.5])['status'],'STRICT_BLOCKED');self.assertEqual(e.queries,0)
        with self.assertRaises(ValueError):AssumptionAttackerEnv(e,ToyDefender(),transaction(),True)
        self.assertIn('CAUSALLY_VALIDATED',[s.value for s in AttackStatus])
    def test_contracts_required(self):
        with self.assertRaises(ValueError):AssumptionAttackEngine(plausibility())
    def test_gym_and_observation_no_oracle(self):
        e=engine();d=ToyDefender();env=AssumptionAttackerEnv(e,d,transaction(),True);check_env(env,skip_render_check=True)
        obs,_=env.reset(seed=42);self.assertEqual(obs.shape,(11,));self.assertTrue(np.isfinite(obs).all())
        _,_,done,_,info=env.step([-.5]);self.assertFalse(done);self.assertNotIn('probability',info)
        _,reward,done,_,info=env.step([0]);self.assertTrue(done);self.assertEqual(info['queries'],1);self.assertEqual(reward,.95)
        self.assertFalse(set(info)&{'isfraud','hidden_label','original_probability','modified_transaction'})
    def test_hidden_label_not_in_observation(self):
        e=engine();t=transaction();a=e.reset(t);t['isfraud']=0;b=e.reset(t);np.testing.assert_array_equal(a,b)
    def test_invalid_no_query(self):
        d=ToyDefender();env=AssumptionAttackerEnv(engine(),d,transaction(),True);env.reset();_,r,done,_,info=env.step([2]);self.assertEqual(r,-2);self.assertEqual(d.calls,0)
    def test_reward_values(self):
        self.assertEqual(assumption_reward('ASSUMPTION_VALID','APPROVE',.1),.9)
        self.assertEqual(assumption_reward('ASSUMPTION_VALID','FLAG',.1),-1.1)
        self.assertEqual(assumption_reward('INVALID_ACTION',cost=.05),-2.05)
        self.assertEqual(assumption_reward('UNCHANGED','APPROVE'),0)
    def test_same_id_seed_actions(self):
        for policy in CONFIG['experiment']['policies']:
            self.assertEqual(proposal(policy,123,42),proposal(policy,123,42))
        self.assertNotEqual(proposal('random_uniform',123),proposal('random_uniform',124))

class PlausibilityTests(unittest.TestCase):
    def test_exact_fit_and_conditional_support(self):
        p=plausibility();self.assertEqual(p.document['global_statistics']['rows'],300)
        self.assertIn('W',p.document['conditional_statistics']['productcd'])
    def test_future_contamination_detection(self):
        f,p=fitting_fixture();f.loc[299,'transactiondt']=300
        with self.assertRaises(ValueError):AttackPlausibility.fit_frame(f,p)
        f,p=fitting_fixture();f.loc[299,'transactionid']=900
        with self.assertRaises(ValueError):AttackPlausibility.fit_frame(f,p)
    def test_saved_provenance_rejected_if_changed(self):
        f,p=fitting_fixture();m=AttackPlausibility.fit_frame(f,p).document;m['provenance']['window']=dict(m['provenance']['window'],max_time=999)
        with self.assertRaises(ValueError):AttackPlausibility(m,p)
    def test_soft_anomalies_do_not_reject(self):
        t=transaction();t['transactionamt']=Decimal('9000.000');e=engine();r=prepare_episode(e,t,-.5)
        self.assertEqual(r['status'],'ASSUMPTION_VALID');self.assertTrue(r['diagnostics']['modified_outside_fit_range'])
        self.assertGreater(r['diagnostics']['modified_anomaly_score'],0)

class HybridTests(unittest.TestCase):
    def test_delayed_memory_provenance_and_label_isolation(self):
        p=plausibility();cfg=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text())
        cfg['budget']={'window_size':2,'total':2,'audit_reserve':0}
        records=[InternalTransaction(i,1000+i*100,{k:v for k,v in transaction().items() if k not in ('transactionid','transactiondt','isfraud')},int(i%2==0)) for i in range(4)]
        sim=AssumptionFraudSimulator(TransactionStream.synthetic(records),ToyDefender(),cfg,AssumptionAttackAdapter(p,CONFIG,.05,'fixed_decrease'))
        obs=sim.observation();self.assertEqual(len(obs),10);self.assertEqual(sim.memory.memory_size(),0)
        while not sim.done:
            obs,_,_,_,info=sim.step(1);self.assertNotIn('hidden_label',info)
        self.assertEqual(sim.memory.memory_size(),0);self.assertEqual(sim.verification.pending_count,4)
        sim.drain_pending();memory=sim.memory.get_recent_verified(sim.now,10)
        self.assertEqual(len(memory),4);self.assertEqual({r['verified_label'] for r in memory},{0,1})
        for r in memory:
            self.assertNotIn('hidden_label',r);self.assertGreaterEqual(r['completion_time']-r['request_time'],86400)
            a=r['original_modified_metadata'];self.assertEqual(a['source_transaction_id'],r['transaction_id'])
            if r['verified_label']:
                self.assertTrue(a['modified']);self.assertEqual(a['label_contract'],CONFIG['label_contract']);self.assertEqual(a['query_count'],1)
                self.assertEqual(r['features']['transactionamt'],Decimal('95.000'))
            else:self.assertFalse(a['modified'])
    def test_reproducible_hybrid(self):
        cfg=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text())
        records=[InternalTransaction(i,1000+i*86400,{'transactionamt':Decimal('100.000'),'productcd':'W','card4':'visa','card6':'debit','v1':None},1) for i in range(3)]
        summaries=[]
        for _ in range(2):
            sim=AssumptionFraudSimulator(TransactionStream.synthetic(records),ToyDefender(),cfg,AssumptionAttackAdapter(plausibility(),CONFIG))
            while not sim.done:sim.step(1)
            sim.drain_pending();summaries.append(sim.summary())
        self.assertEqual(summaries[0],summaries[1])

class ActualExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'reports/phase3_7'
        cls.results=pd.read_csv(cls.out/'attack_results.csv',float_precision='round_trip')
    def test_no_identity_in_asr(self):
        summary=pd.read_csv(self.out/'attack_summary.csv')
        identity=summary[summary.policy=='identity'];self.assertTrue((identity.accepted_nonidentity==0).all());self.assertTrue(identity.conditional_evasion_rate.isna().all());self.assertTrue(identity.overall_accepted_attack_evasion_rate.isna().all())
    def test_denominators_recomputed(self):
        summary=pd.read_csv(self.out/'attack_summary.csv');execution=json.loads((self.out/'experiment_execution.json').read_text())
        actual=summarize(self.results,execution['source_fraud'],execution['eligible_fraud'])
        pd.testing.assert_frame_equal(summary,actual,check_dtype=False,rtol=1e-12,atol=1e-12)
    def test_same_source_actions_and_amounts(self):
        a=self.results[self.results.dataset_version=='original_431'];b=self.results[self.results.dataset_version=='cleaned_prefix_430']
        pd.testing.assert_frame_equal(a[['transactionid','policy','epsilon','continuous_action','modified_amount']].reset_index(drop=True),b[['transactionid','policy','epsilon','continuous_action','modified_amount']].reset_index(drop=True))
    def test_development_only_and_queries(self):
        protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text());w=protocol['windows']['rl_development']
        self.assertTrue(self.results.transactiondt.between(w['min_time'],w['max_time']).all())
        self.assertTrue((self.results.query_count<=1).all());self.assertTrue((self.results.query_count[self.results.status.isin(['ASSUMPTION_VALID','UNCHANGED'])]==1).all())
    def test_fit_prefix_provenance(self):
        p=json.loads((self.out/'plausibility_model.json').read_text());protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())
        self.assertEqual(p['provenance']['window'],protocol['windows']['defender_fit']);self.assertFalse(p['provenance']['fit_uses_labels'])
        self.assertEqual(p['provenance']['source_sha256'],sha(ROOT/'data/processed/train_original.parquet'))
    def test_hard_checks_and_causal_status(self):
        checks=pd.read_csv(self.out/'plausibility_validation.csv')
        for c in ['hard_finite_positive_decimal','hard_relative_budget','hard_non_target_and_metadata_unchanged']:self.assertTrue(checks[c].all())
        self.assertFalse(checks.causal_validity.any());self.assertFalse(checks.statistical_diagnostics_reject.any())
        self.assertEqual(set(self.results.dependency_status),{'UNKNOWN_DEPENDENCY'})
    def test_prior_sources_models_reports_unchanged(self):
        protected=json.loads((self.out/'protected_before.json').read_text())
        self.assertTrue(all(sha(ROOT/p)==h for p,h in protected.items()))
    def test_actual_smoke(self):
        results=json.loads((self.out/'hybrid_smoke_summary.json').read_text())
        self.assertEqual(results[0]['source_ids_sha256'],results[1]['source_ids_sha256'])
        for r in results:
            self.assertTrue(r['provenance_verified']);self.assertGreater(r['verified_modified'],0);self.assertEqual(r['mature']['investigations_pending'],0)
    def test_real_gym_terminal_matches_batched_experiment(self):
        import pyarrow.dataset as ds
        from rl_environment.defender import FrozenDefender
        row=self.results[(self.results.dataset_version=='original_431')&(self.results.policy=='random_uniform')&(self.results.epsilon==.05)].iloc[0]
        record=ds.dataset(ROOT/'data/processed/train_original.parquet').to_table(filter=ds.field('transactionid')==int(row.transactionid)).to_pylist()[0]
        protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())
        p=AttackPlausibility(json.loads((self.out/'plausibility_model.json').read_text()),protocol)
        defender=FrozenDefender(ROOT/'models/temporal_defender/original')
        self.assertEqual(defender._baseline.prepare([{n:record[n] for n in defender.feature_names}]).dtype,np.float32)
        env=AssumptionAttackerEnv(engine_for(p,CONFIG,.05),defender,record,True)
        obs,_=env.reset(seed=42);self.assertEqual(obs.shape,(11,))
        env.step([row.continuous_action]);_,reward,done,_,info=env.step([0.])
        self.assertTrue(done);self.assertEqual(info['queries'],1);self.assertEqual(info['decision'],row.modified_decision)
        self.assertAlmostEqual(reward,row.reward)
