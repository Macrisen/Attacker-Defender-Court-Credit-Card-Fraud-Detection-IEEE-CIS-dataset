import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from rl_environment.attack_constraint_engine import AttackConstraintEngine, ConstrainedAttackEpisode, ConstraintAttackerEnv, attacker_reward, attack_metrics
from gymnasium.utils.env_checker import check_env
from rl_environment.feature_registry import FeatureRegistry, audit_schema
from rl_environment.feature_dependency_validator import DependencyValidator
from rl_environment.court_credit_assignment import DelayedCreditLedger, CourtCreditCollector, time_discount
from rl_environment.credit_checks import credit_checks, event
from rl_environment.temporal_protocol import build_protocol, validate_model_periods, verify_alignment
from rl_environment.transaction_stream import TransactionStream, InternalTransaction
from rl_environment.fraud_simulator import FraudSimulator
from scripts.fit_temporal_defender import fit_new_defender

ROOT = Path(__file__).resolve().parents[2]
THREAT = yaml.safe_load((ROOT/'configs/attacker_threat_model.yaml').read_text())


class ToyDefender:
    threshold = 1/6
    version = 'synthetic_only'
    def decide(self, features):
        p = .8 if features['total'] > 55 else .1
        return ('FLAG' if p > self.threshold else 'APPROVE', p)
    def predict_many(self, features):
        return np.full(len(features), .1)


def fixture():
    return {'quantity': 6., 'unit_price': 10., 'total': 60., 'transactionid': 7, 'transactiondt': 10,
            'isfraud': 1, 'partition': 'rl_development'}


def engine():
    return AttackConstraintEngine(FeatureRegistry.synthetic(), 'synthetic', THREAT['scenarios']['synthetic'])


class ConstraintTests(unittest.TestCase):
    def test_constraint_gymnasium_adapter(self):
        check_env(ConstraintAttackerEnv(engine(), ToyDefender(), fixture()), skip_render_check=True)
        registry = FeatureRegistry(audit_schema(pa.schema([('transactionamt', pa.float64())])))
        blocked = AttackConstraintEngine(registry, 'B', THREAT['scenarios']['B'])
        with self.assertRaises(ValueError):ConstraintAttackerEnv(blocked, ToyDefender(), {'transactionamt': 100})

    def test_continuous_dimensions_and_stop(self):
        e = engine(); e.reset(fixture())
        self.assertEqual(e.action_space.shape, (2,))
        r = e.apply([-.5, 1])
        self.assertTrue(r['submit']); self.assertTrue(e.done)
        self.assertEqual(r['status'], 'VALID_MODIFIED')
        self.assertTrue(e.observation_space.contains(e.observation()))
        with self.assertRaises(RuntimeError):e.apply([0, 1])

    def test_bounds_are_rejected_not_clipped(self):
        for a in ([1.01, 0], [float('inf'), 0], [0], ['bad', 0]):
            e = engine(); e.reset(fixture())
            self.assertEqual(e.apply(a)['status'], 'INVALID_ACTION')

    def test_actual_displacement_budget_and_history(self):
        e = engine(); e.reset(fixture())
        self.assertEqual(e.apply([.5, -1])['cost'], .5)
        self.assertEqual(e.apply([-.5, -1])['cost'], 0.)
        self.assertEqual(e.apply([0, -1])['status'], 'VALID_UNCHANGED')
        self.assertTrue(e.done); self.assertEqual(len(e.history), 3)
        e.reset(fixture());e.apply([1, -1])
        self.assertEqual(e.apply([1, 1])['status'], 'INVALID_ACTION')

    def test_original_and_metadata_unchanged(self):
        original = fixture(); before = copy.deepcopy(original)
        e = engine(); e.reset(original);r = e.apply([-.5, 1])
        self.assertEqual(before, original)
        for n in ('transactionid', 'transactiondt', 'isfraud', 'partition'):
            self.assertEqual(r['modified_transaction'][n], original[n])

    def test_dependency_recomputed_and_checked(self):
        e = engine();e.reset(fixture());r = e.apply([-.5, 1])
        self.assertEqual(r['modified_transaction']['total'], 50.)
        invalid = dict(r['modified_transaction']);invalid['total'] = 999
        self.assertEqual(e.dependency.validate(invalid)[0], 'INVALID')

    def test_unknown_and_immutable_controls_rejected(self):
        for name in ('v1', 'isfraud', 'partition'):
            reg = FeatureRegistry([{'feature': name, 'classification': 'UNKNOWN'}])
            spec = {'enabled': True, 'controls': [{'feature': name, 'absolute_limit': 1}]}
            with self.assertRaises(ValueError):AttackConstraintEngine(reg, 'A', spec)

    def test_label_assumption_is_tracked_not_fact(self):
        e = engine();e.reset(fixture());r = e.apply([-.5, 1])
        self.assertIsNotNone(r['label_assumption'])
        self.assertFalse(r['label_preserved_as_observed_fact'])
        spec = copy.deepcopy(THREAT['scenarios']['synthetic']);spec['label_assumption'] = None
        e = AttackConstraintEngine(FeatureRegistry.synthetic(), 'synthetic', spec);e.reset(fixture())
        self.assertEqual(e.apply([-.5, 1])['status'], 'UNKNOWN_VALIDITY')

    def test_scenario_B_unknown_dependency_never_accepted(self):
        reg = FeatureRegistry(audit_schema(pa.schema([('transactionamt', pa.decimal128(8, 3))])))
        e = AttackConstraintEngine(reg, 'B', THREAT['scenarios']['B'])
        e.reset({'transactionamt': Decimal('100.000'), 'isfraud': 1})
        result = e.apply([.5, 1])
        self.assertEqual(result['status'], 'UNKNOWN_VALIDITY')
        self.assertIsNone(result['modified_transaction'])

    def test_synthetic_or_unreferenced_evidence_cannot_promote_real_dependency(self):
        reg = FeatureRegistry(audit_schema(pa.schema([('transactionamt', pa.float64())])))
        for status in ('VERIFIED_FIXTURE', 'VERIFIED'):
            spec = copy.deepcopy(THREAT['scenarios']['B']);spec['dependency_status'] = status
            with self.assertRaises(ValueError):AttackConstraintEngine(reg, 'B', spec)

    def test_identity_is_not_an_adversarial_success(self):
        m = attack_metrics([{'status': 'VALID_UNCHANGED', 'decision': 'APPROVE', 'eligible_fraud': True}])
        self.assertIsNone(m['asr']);self.assertEqual(m['asr_denominator'], 0)

    def test_denominators_condition_on_eligible_modified_fraud(self):
        results = [{'status': 'VALID_MODIFIED', 'decision': 'APPROVE', 'original_decision': 'FLAG', 'eligible_fraud': True},
                   {'status': 'VALID_MODIFIED', 'decision': 'FLAG', 'original_decision': 'APPROVE', 'eligible_fraud': True},
                   {'status': 'VALID_MODIFIED', 'decision': 'APPROVE', 'eligible_fraud': False},
                   {'status': 'UNKNOWN_VALIDITY'}]
        m = attack_metrics(results)
        self.assertEqual(m['asr_denominator'], 2);self.assertEqual(m['asr'], .5)
        self.assertEqual(m['conditional_evasion_originally_detected'], 1.)
        self.assertEqual(m['valid_nonidentity_fraction'], .75)

    def test_query_budget_feedback_and_rewards(self):
        episode = ConstrainedAttackEpisode(engine(), ToyDefender(), THREAT['rewards'])
        episode.reset(fixture());_, reward, done, _, info = episode.step([-.5, 1])
        self.assertEqual(reward, .5);self.assertEqual(info['queries'], 1);self.assertTrue(done)
        self.assertNotIn('probability', info);self.assertNotIn('isfraud', info)
        self.assertEqual(attacker_reward('INVALID_ACTION'), -2.)
        self.assertEqual(attacker_reward('INVALID_ACTION', cost=1), -3.)
        self.assertIsNone(attacker_reward('UNKNOWN_VALIDITY'))

    def test_deterministic_replay(self):
        a, b = engine(), engine()
        a.reset(fixture());b.reset(fixture())
        for action in [[.2, -1], [-.4, -1], [.1, 1]]:
            self.assertEqual(a.apply(action), b.apply(action))

    def test_missing_control_is_unknown_without_imputation(self):
        reg = FeatureRegistry(audit_schema(pa.schema([('transactionamt', pa.float64())])))
        e = AttackConstraintEngine(reg, 'B', THREAT['scenarios']['B']);e.reset({'transactionamt': None})
        self.assertEqual(e.apply([.5, 1])['status'], 'UNKNOWN_VALIDITY')
        e.reset({'transactionamt': float('nan')})
        self.assertTrue(e.observation_space.contains(e.observation()))


class CreditTests(unittest.TestCase):
    def test_timestamp_gate_catches_incorrect_partition_tag(self):
        ledger = DelayedCreditLedger(200, training_start=100)
        ledger.register(0, 10, 0, [0], 0, False, 'rl_development')
        ledger.attach_next(0, [1], 1)
        self.assertEqual(ledger.ready(100), [])
        self.assertEqual(ledger.records()[0]['training_eligibility'], 'EXCLUDED_PARTITION')

    def test_executed_numerical_credit_cases(self):
        for row in credit_checks():
            with self.subTest(row['test_name']):self.assertEqual(row['status'], 'PASS', row)

    def test_no_future_label_and_origin_state_not_rewritten(self):
        l = DelayedCreditLedger(400000)
        obs = [0., 1.];l.register(0, 10, 0, obs, 1, True, 'rl_development')
        obs[0] = 999;l.attach_next(0, [1, 1], 1)
        with self.assertRaises(ValueError):l.complete(event(), 172799)
        self.assertEqual(l.ready(10), [])
        l.complete(event(), 172800)
        self.assertEqual(l.records()[0]['observation'], [0., 1.])

    def test_time_discount_domain(self):
        self.assertEqual(time_discount(0), 1.)
        self.assertEqual(time_discount(86400, .5), .5)
        with self.assertRaises(ValueError):time_discount(-1)

    def test_collector_excludes_audit_and_preserves_pending_across_windows(self):
        records = [InternalTransaction(i, i*86400, {'transactionamt': 10.}, 1) for i in range(4)]
        config = {'budget': {'window_size': 1, 'total': 1, 'audit_reserve': 0},
                  'verification': {'min_days': 2, 'max_days': 2}}
        sim = FraudSimulator(TransactionStream.synthetic(records), ToyDefender(), config)
        collector = CourtCreditCollector(sim, 400000)
        first = collector.step(1)
        self.assertEqual(first['ready_replay_records'], [])
        self.assertEqual(sim.verification.pending_count, 1)
        second = collector.step(0)
        origins = second['ready_replay_records']
        self.assertEqual(next(r for r in origins if r['court_action_id'] == 0)['completion_bonus'], 10)
        self.assertEqual(next(r for r in origins if r['court_action_id'] == 1)['final_attributed_reward'], 0.)

    def test_audit_logging_and_budget(self):
        records = [InternalTransaction(i, i, {'transactionamt': 10.}, 1) for i in range(3)]
        config = {'budget': {'window_size': 3, 'total': 2, 'audit_reserve': 1}}
        sim = FraudSimulator(TransactionStream.synthetic(records), ToyDefender(), config)
        sim.step(1);sim.step(1)
        self.assertFalse(sim.actions[0]['audit_eligible'])
        self.assertEqual(sim.actions[0]['audit_selection_probability'], 0.)
        self.assertEqual(sim.actions[1]['court_budget_available_before'], 0)
        self.assertTrue(sim.actions[1]['audit_eligible'])
        self.assertEqual(sim.actions[1]['audit_selection_probability'], 1/3)


class TemporalTests(unittest.TestCase):
    def create(self, directory):
        p = Path(directory)/'train.parquet'
        x = np.arange(1000)%2
        pq.write_table(pa.table({'transactionid': np.arange(1000), 'transactiondt': np.arange(1000),
                                'x': x.astype(float), 'isfraud': x}), p)
        saved = {'prior_holdout_cutoff': 799, 'hold_start': 800}
        return p, build_protocol(p, saved, 799, max_delay_seconds=3)

    def test_windows_and_maturation_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p, protocol = self.create(d);w = protocol['windows']
            self.assertEqual([v['rows'] for v in w.values()], [450, 100, 150, 100, 200])
            self.assertTrue(validate_model_periods(protocol, 449, 549, 449))
            self.assertEqual(w['rl_development']['maturity_only_rows'], 3)
            with self.assertRaises(ValueError):validate_model_periods(protocol, 649, 749, 799)
            with self.assertRaises(ValueError):validate_model_periods(protocol, 449, 549, 799)

    def test_alignment_detects_missing_transaction(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.create(d);other = Path(d)/'original.parquet'
            pq.write_table(pq.read_table(p), other)
            self.assertTrue(verify_alignment(p, other))
            pq.write_table(pq.read_table(p).slice(1), other)
            with self.assertRaises(ValueError):verify_alignment(p, other)

    def test_revised_fit_pipeline_executes_only_on_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p, protocol = self.create(d);d = Path(d)
            (d/'protocol.json').write_text(json.dumps(protocol))
            (d/'schema.json').write_text(json.dumps({'features': [{'name': 'x', 'role': 'numeric', 'postgres_type': 'double precision'}]}))
            provenance = {'fit_cutoff': 449, 'selected_predictors': ['x'], 'parquet_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            (d/'provenance.json').write_text(json.dumps(provenance))
            result = fit_new_defender(p, d/'schema.json', d/'protocol.json', d/'provenance.json', d/'new_model', max_rounds=5)
            self.assertTrue((Path(result)/'calibrator.joblib').exists())
            provenance['fit_cutoff'] = 799;(d/'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'Feature selection'):
                fit_new_defender(p, d/'schema.json', d/'protocol.json', d/'provenance.json', d/'bad_model', max_rounds=5)
            self.assertFalse((d/'bad_model').exists())


if __name__ == '__main__':unittest.main()
