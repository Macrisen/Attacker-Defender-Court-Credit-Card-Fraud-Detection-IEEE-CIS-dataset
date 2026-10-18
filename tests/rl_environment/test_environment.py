import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gymnasium.utils.env_checker import check_env

from rl_environment.transaction_stream import InternalTransaction, TransactionStream
from rl_environment.defender import FrozenDefender
from rl_environment.observations import CourtObservations
from rl_environment.attacker_env import ContinuousAttackerEnv, SyntheticQuantityThreat, identity_attack
from rl_environment.budget_manager import BudgetManager
from rl_environment.verification_simulator import VerificationSimulator, PENDING, VERIFIED_FRAUD, VERIFIED_LEGITIMATE
from rl_environment.verified_memory import VerifiedMemory
from rl_environment.fraud_simulator import FraudSimulator
from rl_environment.court_env import CourtEnv

ROOT = Path(__file__).resolve().parents[2]


class FixtureDefender:
    threshold = 1/6
    version = 'SYNTHETIC_TEST_ONLY'

    def predict_many(self, records):
        return np.array([.8 if (r.get('transactionamt') or 0) > 100 else .1 for r in records])

    def decide(self, record):
        p = .8 if record['total'] > 55 else .1
        return ('FLAG' if p > self.threshold else 'APPROVE', p)


def stream(times=(0, 1, 2), labels=(1, 0, 1), amount=10):
    return TransactionStream.synthetic([InternalTransaction(i, t, {'transactionamt': amount}, y)
                                        for i, (t, y) in enumerate(zip(times, labels))])


def config():
    return {'seed': 42, 'budget': {'window_size': 10000, 'total': 200, 'audit_reserve': 0},
            'verification': {'min_days': 1, 'max_days': 1}, 'observable_features': ['transactionamt']}


def request(label=1, tid=1, decision='APPROVE'):
    return {'transaction_id': tid, 'transaction_time': 0, 'hidden_label': label,
            'features': {'transactionamt': None}, 'defender_decision': decision,
            'source': 'court', 'action_id': tid}


class EnvironmentTests(unittest.TestCase):
    def test_identity_keeps_values_and_does_not_alias_source(self):
        original = {'transactionamt': Decimal('0.123'), 'v1': None, 'card4': 'visa'}
        modified = identity_attack(original)
        self.assertEqual(modified, original)
        modified['card4'] = 'changed'
        self.assertEqual(original['card4'], 'visa')

    def test_chronological_order_reset_and_batches(self):
        s = TransactionStream.synthetic([InternalTransaction(2, 10, {}, 0), InternalTransaction(1, 10, {}, 1),
                                         InternalTransaction(3, 9, {}, 0)])
        self.assertEqual([r.transaction_id for r in s], [3, 1, 2])
        s.reset()
        self.assertEqual(len(s.batch(2)), 2)
        self.assertEqual(len(s.batch(2)), 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            TransactionStream.synthetic([InternalTransaction(1, 0, {}, 0), InternalTransaction(1, 1, {}, 0)])

    def test_parquet_schema_holdout_boundary_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = Path(tmp)/'train.parquet', Path(tmp)/'split.json'
            pq.write_table(pa.table({'transactionid': [3, 1, 2, 4], 'transactiondt': [10, 0, 10, 30],
                                    'isfraud': [1, 0, 1, 1], 'transactionamt': [None, 2., 3., 9999.]}), p)
            b.write_text(json.dumps({'model_training_cutoff': 10, 'prior_holdout_cutoff': 20, 'train_end': 3}))
            s = TransactionStream(p, b, ['transactionamt'], limit=10)
            records = list(s)
            self.assertEqual([r.transaction_id for r in records], [1, 2, 3])
            self.assertIsNone(records[-1].features['transactionamt'])
            self.assertNotIn('isfraud', records[0].features)
            self.assertTrue(all(r.transaction_time <= 10 for r in records))
            pq.write_table(pa.table({'transactionid': [1], 'transactiondt': [0], 'transactionamt': [2.]}), p)
            with self.assertRaisesRegex(ValueError, 'Kaggle test'):
                TransactionStream(p, b, ['transactionamt'])

    def test_hidden_label_isolation_until_verification(self):
        a = FraudSimulator(stream(labels=(0, 0, 0)), FixtureDefender(), config())
        b = FraudSimulator(stream(labels=(1, 1, 1)), FixtureDefender(), config())
        np.testing.assert_array_equal(a.observation(), b.observation())
        for action in (1, 0, 0):
            oa, ra, _, _, ia = a.step(action)
            ob, rb, _, _, ib = b.step(action)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual((ra, ia), (rb, ib))
            self.assertNotIn('hidden_label', str(ia))
        self.assertEqual(a.memory.memory_size(), 0)
        self.assertEqual(b.memory.memory_size(), 0)

    def test_metadata_observation_rejected(self):
        for name in ('isfraud', 'transactionid', 'transactiondt'):
            with self.assertRaises(ValueError):
                CourtObservations([name])

    def test_missing_observation_mask(self):
        s = FraudSimulator(stream(amount=None), FixtureDefender(), config())
        obs = s.observation()
        self.assertEqual(tuple(obs[:2]), (0., 1.))
        self.assertTrue(np.isfinite(obs).all())
        self.assertIsNone(s._records[0].features['transactionamt'])

    def test_budget_exhaustion_reserve_separation(self):
        b = BudgetManager(10, 2, 1)
        b.begin(0)
        self.assertTrue(b.spend('court'))
        self.assertFalse(b.spend('court'))
        self.assertTrue(b.spend('audit'))
        self.assertFalse(b.spend('audit'))

    def test_budget_exact_window_rollover(self):
        b = BudgetManager(10000, 200, 20)
        b.begin(9999)
        for _ in range(180):
            self.assertTrue(b.spend('court'))
        self.assertFalse(b.spend('court'))
        b.begin(10000)
        self.assertEqual(b.court_used, 0)
        self.assertEqual(b.lifetime_court, 180)
        self.assertTrue(b.spend('court'))
        with self.assertRaises(ValueError):
            b.begin(9999)

    def test_delayed_verification_timing_and_outcomes(self):
        v = VerificationSimulator(min_days=1, max_days=1)
        v.request(request(1, 1), 0)
        v.request(request(0, 2), 0)
        self.assertEqual(v.status(1), PENDING)
        self.assertEqual(v.release(86399), [])
        results = v.release(86400)
        self.assertEqual([r['verified_label'] for r in results], [1, 0])
        self.assertTrue(all('hidden_label' not in r for r in results))
        self.assertEqual(v.status(1), VERIFIED_FRAUD)
        self.assertEqual(v.status(2), VERIFIED_LEGITIMATE)

    def test_no_premature_memory_and_no_duplicates(self):
        m = VerifiedMemory()
        with self.assertRaises(ValueError):
            m.add_verified(request(), 100000)
        v = VerificationSimulator(min_days=1, max_days=1)
        v.request(request(), 0)
        r = v.release(86400)[0]
        with self.assertRaises(ValueError):
            m.add_verified(r, 86399)
        m.add_verified(r, 86400)
        self.assertEqual(m.get_verified_statistics(86399)['count'], 0)
        self.assertEqual(m.get_verified_statistics(86400)['fraud_rate'], 1)
        self.assertEqual(len(m.sample_verified(10, 86400)), 1)
        with self.assertRaises(ValueError):
            m.add_verified(r, 86400)

    def test_immediate_delayed_rewards_and_attribution(self):
        s = FraudSimulator(stream(times=(0, 86400, 172800)), FixtureDefender(), config())
        _, reward, _, _, info = s.step(1)
        self.assertEqual(reward, 9)  # -1 now, +10 at next transaction one day later
        self.assertEqual(info['due_verifications'][0]['action_id'], 0)
        self.assertEqual(info['due_verifications'][0]['completion_time'], 86400)
        _, reward, _, _, _ = s.step(1)
        self.assertEqual(reward, -1)  # legitimate, no bonus
        self.assertEqual(s.policy_reward, 8)

    def test_flagged_fraud_reward(self):
        s = FraudSimulator(stream(times=(0, 86400, 172800), amount=200), FixtureDefender(), config())
        self.assertEqual(s.step(1)[1], 2)  # -1 +3

    def test_agent_info_cannot_mutate_internal_event_log(self):
        s = FraudSimulator(stream(times=(0, 86400, 172800)), FixtureDefender(), config())
        info = s.step(1)[-1]
        info['due_verifications'][0]['verified_label'] = 0
        self.assertEqual(s.events[0]['verified_label'], 1)
        self.assertEqual(s.memory.get_verified_statistics(86400)['fraud_rate'], 1.)

    def test_skip_does_not_reveal_or_charge(self):
        s = FraudSimulator(stream(times=(0, 86400, 172800)), FixtureDefender(), config())
        for _ in range(3):
            self.assertEqual(s.step(0)[1], 0)
        self.assertEqual(s.memory.memory_size(), 0)
        self.assertEqual(s.verification.pending_count, 0)

    def test_blocked_investigate_no_double_cost(self):
        c = config()
        c['budget'] = {'window_size': 10, 'total': 1, 'audit_reserve': 0}
        s = FraudSimulator(stream(), FixtureDefender(), c)
        self.assertEqual(s.step(1)[1], -1)
        _, reward, _, _, info = s.step(1)
        self.assertEqual(reward, 0)
        self.assertTrue(info['budget_blocked'])
        self.assertEqual(s.verification.pending_count, 1)

    def test_pending_survives_budget_window(self):
        c = config()
        c['budget'] = {'window_size': 1, 'total': 1, 'audit_reserve': 0}
        s = FraudSimulator(stream(), FixtureDefender(), c)
        s.step(1)
        self.assertEqual(s.budget.window, 1)
        self.assertEqual(s.verification.pending_count, 1)
        s.step(1)
        self.assertEqual(s.verification.pending_count, 2)

    def test_episode_censor_explicit_drain_and_reset_archive(self):
        s = FraudSimulator(stream(), FixtureDefender(), config())
        s.step(1)
        s.step(0)
        _, _, terminated, truncated, info = s.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['pending_censored'], 1)
        self.assertEqual(s.memory.memory_size(), 0)
        self.assertEqual(s.now, 2)
        checkpoint = s.verification.internal_checkpoint()
        s.reset(42)
        self.assertEqual(s.archived_episodes[-1]['pending_checkpoint'], checkpoint)
        for action in (1, 0, 0):
            s.step(action)
        s.drain_pending()
        self.assertEqual(s.now, 86400)
        self.assertEqual(s.memory.memory_size(), 1)
        self.assertEqual(s.verification.pending_count, 0)

    def test_random_delay_seed_reproducibility(self):
        a, b = VerificationSimulator(seed=42), VerificationSimulator(seed=42)
        for i in range(20):
            a.request(request(tid=i), 0)
            b.request(request(tid=i), 0)
        self.assertEqual(a.internal_checkpoint(), b.internal_checkpoint())
        self.assertTrue(all(86400 <= t <= 259200 for t, _, _ in a.internal_checkpoint()))

    def test_random_audits_only_approved_and_not_court_rewards(self):
        c = config()
        c['budget'] = {'window_size': 1, 'total': 1, 'audit_reserve': 1}
        s = FraudSimulator(stream(times=(0, 86400, 172800)), FixtureDefender(), c)
        self.assertEqual(s.step(0)[1], 0)
        self.assertEqual(s.budget.lifetime_audit, 1)
        self.assertEqual(s.audit_reward, 9)
        self.assertEqual(s.memory.memory_size(), 1)
        flagged = FraudSimulator(stream(amount=200), FixtureDefender(), c)
        flagged.step(0)
        self.assertEqual(flagged.budget.lifetime_audit, 0)

    def test_court_gymnasium_api(self):
        env = CourtEnv(FraudSimulator(stream(), FixtureDefender(), config()))
        check_env(env, skip_render_check=True)


class AttackerTests(unittest.TestCase):
    def env(self):
        return ContinuousAttackerEnv(FixtureDefender(), {'quantity': 5., 'unit_price': 10., 'total': 50.}, SyntheticQuantityThreat())

    def test_real_data_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'BLOCKED BY THREAT MODEL'):
            ContinuousAttackerEnv(FixtureDefender(), {'transactionamt': 10})

    def test_continuous_bounds_dependents_and_cost(self):
        e = self.env()
        e.reset(seed=42)
        _, reward, done, _, info = e.step(np.array([10., 1.], dtype=np.float32))
        self.assertEqual(e.current['quantity'], 7.)
        self.assertEqual(e.current['total'], 70.)
        self.assertEqual(e.current['unit_price'], 10.)
        self.assertEqual(reward, -2.)
        self.assertTrue(done)
        self.assertTrue(info['valid'])
        self.assertEqual(e.queries, 1)

    def test_invalid_action_and_perturbation_budget(self):
        e = self.env()
        for invalid in ([float('nan'), 0], [0], [float('inf'), 0], ['not_numeric', 0]):
            e.reset()
            _, reward, done, _, info = e.step(invalid)
            self.assertEqual(reward, -2.)
            self.assertTrue(done)
            self.assertFalse(info['valid'])
            self.assertEqual(e.queries, 0)
        e.reset()
        e.step([1, -1])
        self.assertFalse(e.step([1, -1])[-1]['valid'])
        self.assertEqual(e.queries, 0)

    def test_query_budget_stop_and_fixed_observations(self):
        e = self.env()
        obs, _ = e.reset(seed=42)
        self.assertTrue(e.observation_space.contains(obs))
        for i in range(3):
            obs, reward, done, _, info = e.step([0, -1])
            self.assertTrue(e.observation_space.contains(obs))
            self.assertEqual(e.queries, int(i == 2))
            self.assertEqual(reward, 1 if i == 2 else 0)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            e.step([0, 1])

    def test_synthetic_gymnasium_api(self):
        check_env(self.env(), skip_render_check=True)


class FrozenDefenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FrozenDefender(ROOT/'reports/lightgbm_baseline/artifacts')
        row = next(pq.ParquetFile(ROOT/'data/processed/train_clean.parquet').iter_batches(batch_size=1)).to_pylist()[0]
        cls.features = {n: row[n] for n in cls.model.feature_names}

    def test_frozen_schema_calibration_and_missing_values(self):
        p = self.model.predict_fraud_probability(self.features)
        expected = float(self.model._baseline.predict(self.features)[0])
        self.assertEqual(p, expected)
        self.assertEqual(self.model.threshold, 1/6)
        self.assertEqual(len(self.model.feature_names), 418)
        self.assertTrue(any(v is None for v in self.features.values()))
        self.assertTrue(0 <= p <= 1)

    def test_reordered_schema_is_consistent_and_extra_label_rejected(self):
        reverse = dict(reversed(list(self.features.items())))
        self.assertEqual(self.model.predict_fraud_probability(reverse), self.model.predict_fraud_probability(self.features))
        with self.assertRaises(ValueError):
            self.model.predict_fraud_probability({**self.features, 'isfraud': 1})
        incomplete = dict(self.features)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(ValueError):
            self.model.predict_fraud_probability(incomplete)

    def test_missing_artifacts_are_not_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                FrozenDefender(tmp)


if __name__ == '__main__':
    unittest.main()
