import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import yaml
from gymnasium.utils.env_checker import check_env
from preprocessing.temporal_feature_audit import run,membership,sha,verify_windows
from preprocessing.temporal_feature_selector import select
from scripts.phase3_6_checks import credit_fixture,attack_checks,ToyDefender
from rl_environment.court_credit_adapter import MatureEpisodeAdapter,FiniteCourtCollector
from rl_environment.court_credit_assignment import DelayedCreditLedger
from rl_environment.court_observation import CourtObservationV2
from rl_environment.court_reward import CourtReward
from rl_environment.attacker_env import ContinuousControlAttackerEnv
from rl_environment.attack_constraint_engine import AttackConstraintEngine,attack_metrics
from rl_environment.feature_registry import FeatureRegistry
from rl_environment.temporal_protocol import verify_alignment,validate_model_periods
from defender.inference import Defender,fit_mappings
ROOT=Path(__file__).resolve().parents[2]

class CreditTests(unittest.TestCase):
    def test_manual_targets_and_gradient(self):
        rows,update,records,ledger=credit_fixture()
        self.assertTrue(all(r['status']=='PASS' for r in rows))
        np.testing.assert_allclose([r['target'] for r in records],[1,-1,0])
        self.assertGreater(update['loss_before'],update['loss_after'])
    def test_causal_history_prefixes(self):
        _,_,records,_=credit_fixture()
        for i,r in enumerate(records):
            h=list(r['history']);self.assertEqual(len(h),i+1)
            self.assertTrue(all(set(x)=={'observation','action','timestamp'} for x in h))
            self.assertLessEqual(max(x['timestamp'] for x in h),i*86400)
    def test_boundary_and_truncation(self):
        with self.assertRaises(ValueError):MatureEpisodeAdapter(0,2,259201)
        _,_,_,ledger=credit_fixture();a=MatureEpisodeAdapter(0,172800,432000,.5)
        with self.assertRaises(ValueError):a.release(ledger,432000,True,truncated=True)
        self.assertEqual(a.release(ledger,432001,True),[])
    def test_pending_and_wrong_partition(self):
        l=DelayedCreditLedger(259200)
        l.register(0,1,0,[0],1,True,'rl_development');l.attach_next(0,[0],0,terminated=True)
        with self.assertRaises(ValueError):MatureEpisodeAdapter(0,0,259200).release(l,259200,True)
        l=DelayedCreditLedger(259200);l.register(0,1,0,[0],0,False,'rl_validation');l.attach_next(0,[0],0,terminated=True)
        with self.assertRaises(ValueError):MatureEpisodeAdapter(0,0,259200).release(l,259200,True)
    def test_reward_coefficients(self):
        r=CourtReward()
        self.assertEqual([r.bonus(1,'APPROVE'),r.bonus(1,'FLAG'),r.bonus(0,'FLAG')],[10,3,0])
        self.assertEqual(CourtReward(missed_fraud_reward=7).bonus(1,'APPROVE'),7)
    def test_no_double_counted_returns(self):
        _,_,r,_=credit_fixture()
        # Arrival-time return: cost -1 at day0, -1 at day1, bonus10 at day2.
        self.assertEqual(r[0]['target'],-1-.5+10*.25)
        self.assertTrue(all(x['bootstrap']==0 for x in r))
    def test_replay_cannot_be_exported_twice_through_another_adapter(self):
        _,_,_,ledger=credit_fixture()
        with self.assertRaises(ValueError):
            MatureEpisodeAdapter(0,172800,432000,.5).release(ledger,432000,True)
        self.assertEqual(ledger.ready(432000),[])
    def test_observation_has_known_horizon(self):
        obs=CourtObservationV2(100)
        self.assertEqual(obs.space.shape,(11,));self.assertIn('decision_time_remaining_scaled',obs.names)

class AttackerTests(unittest.TestCase):
    def env(self):
        spec=yaml.safe_load((ROOT/'configs/attacker_threat_model.yaml').read_text())['scenarios']['synthetic']
        return ContinuousControlAttackerEnv(AttackConstraintEngine(FeatureRegistry.synthetic(),'synthetic',spec),ToyDefender(),dict(quantity=6.,unit_price=10.,total=60.))
    def test_k_dimensions_and_gym(self):
        e=self.env();self.assertEqual(e.action_space.shape,(1,));check_env(e,skip_render_check=True)
    def test_stop_and_single_terminal_query(self):
        e=self.env();e.reset();r=e.step([-.5]);self.assertEqual(r[4]['queries'],0)
        r=e.step([0]);self.assertTrue(r[2]);self.assertEqual(r[4]['queries'],1);self.assertEqual(r[1],.5)
        with self.assertRaises(RuntimeError):e.step([0])
    def test_rewards_and_identity_denominator(self):
        rows,m=attack_checks();self.assertTrue(all(r['reward_test']=='PASS' for r in rows));self.assertEqual(m['asr_denominator'],2)
    def test_bad_shape_and_nan_reject(self):
        for a in ([0,0],[float('nan')],[1.+1e-8],['invalid']):
            e=self.env();e.reset();r=e.step(a);self.assertEqual(r[4]['status'],'INVALID_ACTION');self.assertEqual(r[4]['queries'],0)

class SelectionTests(unittest.TestCase):
    def test_missingness_not_blindly_dropped(self):
        s=dict(unique_count=1,missing_count=90,missing_percentage=90,dominant_value_percentage=10)
        self.assertEqual(select(s)[0],'REVIEW')
        s.update(unique_count=4,missing_percentage=99);self.assertEqual(select(s)[0],'REVIEW')
    def test_future_poison_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=pd.DataFrame(dict(transactionid=range(20),transactiondt=range(20),isfraud=[0,1]*10,x=[1.]*20,z=[1.]*20))
            schema={'features':[dict(name=n,postgres_type='double precision',role='numeric') for n in ('x','z')]}
            windows={}
            for name,sub in [('defender_fit',f.iloc[:10]),('future',f.iloc[10:])]:windows[name]=dict(rows=len(sub),min_time=int(sub.transactiondt.min()),max_time=int(sub.transactiondt.max()),ordered_id_sha256=membership(sub))
            protocol={'windows':windows};config=p/'config';config.write_text('fixed rules')
            manifests=[]
            for i in range(2):
                if i:f.loc[10:,'x']=np.arange(10);f.loc[10:,'z']=np.arange(10)*999;f.loc[10:,'isfraud']=1
                pq.write_table(pa.Table.from_pandas(f),p/'data.parquet')
                manifests.append(run(p/'data.parquet',schema,protocol,p/str(i),config))
            self.assertEqual(manifests[0]['selected_predictors'],manifests[1]['selected_predictors'])
            pd.testing.assert_frame_equal(pd.read_csv(p/'0/feature_selection_temporal.csv'),pd.read_csv(p/'1/feature_selection_temporal.csv'))

class ActualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((ROOT/'reports/phase3_5/temporal_split_protocol.json').read_text())
    def test_exact_partition_membership_and_alignment(self):
        for kind in ('train','test'):
            self.assertTrue(verify_alignment(ROOT/f'data/processed/{kind}_clean.parquet',ROOT/f'data/processed/{kind}_original.parquet'))
        self.assertTrue(verify_windows(ROOT/'data/processed/train_original.parquet',self.protocol))
    def test_fresh_model_provenance_and_order(self):
        for kind in ('original','cleaned'):
            path=ROOT/'models/temporal_defender'/kind;model=Defender(path)
            cfg=json.loads((path/'training_config.json').read_text());w=self.protocol['windows']
            self.assertTrue(validate_model_periods(self.protocol,w['defender_fit']['max_time'],w['defender_calibration']['max_time'],cfg['provenance']['fit_cutoff']))
            self.assertEqual(model.model.feature_name(),[f['name'] for f in model.schema['features']])
            self.assertEqual(cfg['provenance']['parquet_sha256'],sha(ROOT/'data/processed/train_original.parquet'))
            self.assertEqual(cfg['protocol']['windows'],self.protocol['windows'])
    def test_categories_fit_only_and_independent_artifacts(self):
        model=Defender(ROOT/'models/temporal_defender/original')
        cats=[f['name'] for f in model.schema['features'] if f['role']=='categorical']
        cutoff=self.protocol['windows']['defender_fit']['max_time']
        frame=ds.dataset(ROOT/'data/processed/train_original.parquet').to_table(columns=cats,filter=ds.field('transactiondt')<=cutoff).to_pandas()
        self.assertEqual(model.mappings,fit_mappings(frame,model.schema))
        clean=Defender(ROOT/'models/temporal_defender/cleaned')
        self.assertEqual(clean.mappings,{k:v for k,v in model.mappings.items() if k in clean.mappings})
    def test_no_holdout_evaluation_and_aligned_comparison(self):
        f=pd.read_csv(ROOT/'reports/phase3_6/temporal_defender_comparison.csv')
        self.assertEqual(set(f.evaluation_window),{'rl_development','rl_validation'})
        self.assertEqual(len(f),4)
        for w in f.evaluation_window.unique():
            a=pd.read_parquet(ROOT/f'reports/phase3_6/original_{w}_predictions.parquet');b=pd.read_parquet(ROOT/f'reports/phase3_6/cleaned_{w}_predictions.parquet')
            pd.testing.assert_frame_equal(a[['transactionid','transactiondt']],b[['transactionid','transactiondt']])
            self.assertEqual(membership(a),self.protocol['windows'][w]['ordered_id_sha256'])
    def test_prior_artifacts_unchanged(self):
        before=json.loads((ROOT/'reports/phase3_6/protected_before.json').read_text())
        self.assertTrue(all(sha(ROOT/p)==h for p,h in before.items()))
    def test_independent_smoke_maturation(self):
        summaries=json.loads((ROOT/'reports/phase3_6/development_smoke.json').read_text())
        for s in summaries:
            self.assertEqual(s['replay_records'],10000);self.assertFalse(s['fit_overlap'])
            self.assertLessEqual(s['label_maturity_end'],s['training_window_end'])
            self.assertEqual(s['mature']['investigations_pending'],0)
            self.assertLessEqual(s['mature']['court_investigation_count'],180)
            self.assertLessEqual(s['mature']['audit_investigation_count'],20)
