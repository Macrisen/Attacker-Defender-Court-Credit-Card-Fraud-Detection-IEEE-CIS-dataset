"""Internal causal controller. Agents receive observations and due rewards only."""
import copy
import numpy as np
from .attacker_env import identity_attack
from .budget_manager import BudgetManager
from .observations import CourtObservations
from .verification_simulator import VerificationSimulator
from .verified_memory import VerifiedMemory


class FraudSimulator:
    def __init__(self, stream, defender, config):
        self._records = stream.internal_records()
        if not self._records:
            raise ValueError('Empty stream')
        self.defender, self.config = defender, copy.deepcopy(config)
        if config.get('attacker', {}).get('real_data_mode', 'identity') != 'identity':
            raise ValueError('IEEE-CIS perturbations BLOCKED BY THREAT MODEL')
        self.observations = CourtObservations(config.get('observable_features', ['transactionamt']))
        # Vectorization is an inference optimization only. These independent frozen
        # scores stay internal and are exposed one transaction at a time.
        self._probabilities = defender.predict_many([identity_attack(r.features) for r in self._records])
        self.archived_episodes = []
        self.reset(config.get('seed', 42))

    def reset(self, seed=42):
        if hasattr(self, 'verification') and (self.index or self.verification.pending_count):
            self.archived_episodes.append({'pending_checkpoint': self.verification.internal_checkpoint(),
                                           'summary': self.summary(), 'reason': 'explicit new episode reset'})
        self.audit_rng = np.random.default_rng(seed)
        self.budget = BudgetManager(**self.config.get('budget', {}))
        self.verification = VerificationSimulator(seed=seed+1, **self.config.get('verification', {}))
        self.memory = VerifiedMemory()
        self.index, self.done = 0, False
        self.now = self._records[0].transaction_time
        # Baseline was fitted through the saved model-training cutoff, which overlaps
        # development replay. This is an offline mechanics test, NOT an online backtest.
        self.defender_deployed_at = self.now
        self.events, self.actions = [], []
        self.policy_reward = self.audit_reward = self.business_reward = 0.
        self.budget.begin(0)
        return self.observation()

    def observation(self):
        if self.done:
            return np.zeros(self.observations.space.shape, dtype=np.float32)
        record = self._records[self.index]
        p = float(self._probabilities[self.index])
        return self.observations.encode(record.features, p, 'FLAG' if p > self.defender.threshold else 'APPROVE',
                                       self.budget, self.verification.pending_count, self.memory,
                                       self.now, self.defender_deployed_at)

    def _request(self, record, p, decision, source, action_id, court_action, blocked):
        request = {'transaction_id': record.transaction_id, 'transaction_time': record.transaction_time,
                   'features': copy.deepcopy(record.features),
                   'original_modified_metadata': {'mode': 'identity', 'modified': False},
                   'defender_probability': p, 'defender_decision': decision,
                   'court_investigation_decision': 'INVESTIGATE' if court_action == 1 else 'SKIP',
                   'court_budget_blocked': blocked,
                   'source': source, 'action_id': action_id, 'hidden_label': record.hidden_label,
                   'defender_model_version': self.defender.version}
        self.verification.request(request, self.now)

    def _release(self, time):
        self.now = time
        emitted = []
        reward = 0.
        cfg = self.config.get('court', {})
        for record in self.verification.release(time):
            self.memory.add_verified(record, time)
            bonus = 0.
            if record['verified_label']:
                bonus = cfg.get('missed_fraud_reward', 10) if record['defender_decision'] == 'APPROVE' else cfg.get('flagged_fraud_reward', 3)
            event = {k: record[k] for k in ('transaction_id', 'action_id', 'source', 'request_time', 'completion_time', 'verified_label')}
            event.update({'emitted_at': time, 'reward': bonus})
            self.events.append(event)
            # Audit results are legitimate public verification; audit rewards are NOT
            # learning rewards for a Court action which did not request investigation.
            emitted.append(copy.deepcopy(event))
            if record['source'] == 'court':
                reward += bonus
                self.policy_reward += bonus
            else:
                self.audit_reward += bonus
        return reward, emitted

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has ended')
        if not isinstance(action, (int, np.integer)) or action not in (0, 1):
            raise ValueError('Court action must be 0 or 1')
        record = self._records[self.index]
        p = float(self._probabilities[self.index])
        decision = 'FLAG' if p > self.defender.threshold else 'APPROVE'
        action_id = self.index
        cost = self.config.get('court', {}).get('investigation_cost', 1)
        court_available = self.budget.court_limit-self.budget.court_used
        audit_available = self.budget.reserve-self.budget.audit_used
        reward, source, blocked = 0., None, False
        if action == 1:
            if self.budget.spend('court'):
                source = 'court'
                reward -= cost
                self.policy_reward -= cost
            else:
                blocked = True  # blocked INVESTIGATE becomes SKIP, no cost or label
        audit_eligible = source is None and decision == 'APPROVE'
        audit_probability = self.budget.reserve/self.budget.window_size if audit_eligible and audit_available > 0 else 0.
        if audit_eligible:
            # Online capped Bernoulli audit; no future approval rate/count is consulted.
            if self.audit_rng.random() < self.budget.reserve/self.budget.window_size and self.budget.spend('audit'):
                source = 'audit'
                self.audit_reward -= cost
        if source:
            self._request(record, p, decision, source, action_id, action, blocked)
        self.actions.append({'action_id': action_id, 'transaction_id': record.transaction_id,
                             'time': self.now, 'court_action': int(action), 'blocked': blocked,
                             'source': source or 'none', 'defender_probability': p, 'defender_decision': decision,
                             'immediate_court_reward': -cost if source == 'court' else 0.,
                             'budget_window': self.budget.window,
                             'court_budget_available_before': court_available,
                             'audit_budget_available_before': audit_available,
                             'audit_eligible': audit_eligible,
                             'audit_selection_probability': audit_probability,
                             'verification_status_at_request': 'PENDING' if source else 'NOT_INVESTIGATED'})
        # Evaluation-only ground truth. Never returned in observations or step info.
        if record.hidden_label:
            self.business_reward += 10 if decision == 'FLAG' else -20
        else:
            self.business_reward += -5 if decision == 'FLAG' else 1
        self.index += 1
        self.done = self.index == len(self._records)
        due = []
        if not self.done:
            next_time = self._records[self.index].transaction_time
            delayed, due = self._release(next_time)
            reward += delayed
            self.budget.begin(self.index)
        # End horizon is censored. No time jump or premature label revelation.
        info = {'action_id': action_id, 'budget_blocked': blocked,
                'pending_count': self.verification.pending_count, 'due_verifications': due}
        if self.done:
            info['pending_censored'] = self.verification.pending_count
        return self.observation(), float(reward), False, self.done, info

    def drain_pending(self):
        """Explicit post-episode diagnostic, advancing to actual simulated deadlines.

        No new policy action occurs; not added to the Gym episode's reward sequence.
        """
        if not self.done:
            raise RuntimeError('Drain only after stream horizon')
        events = []
        while self.verification.pending_count:
            _, emitted = self._release(self.verification.next_completion)
            events.extend(emitted)
        return events

    def summary(self):
        records = self._records[:self.index]
        verified = self.memory.get_recent_verified(self.now, self.memory.memory_size())
        fraud_verified = sum(r['verified_label'] for r in verified)
        investigations = self.budget.lifetime_court+self.budget.lifetime_audit
        return {'transactions_processed': self.index,
                'evaluation_only_fraud_count': sum(r.hidden_label for r in records),
                'defender_flag_count': sum(r['defender_decision'] == 'FLAG' for r in self.actions),
                'defender_approve_count': sum(r['defender_decision'] == 'APPROVE' for r in self.actions),
                'court_investigation_count': self.budget.lifetime_court,
                'audit_investigation_count': self.budget.lifetime_audit,
                'investigations_pending': self.verification.pending_count,
                'fraud_verified': fraud_verified, 'legitimate_verified': len(verified)-fraud_verified,
                'investigation_budget_used': investigations, 'audit_budget_used': self.budget.lifetime_audit,
                'investigation_rate': investigations/max(1, self.index),
                'fraud_found_per_completed_investigation': fraud_verified/max(1, len(verified)),
                'court_policy_reward': self.policy_reward, 'audit_reward': self.audit_reward,
                'evaluation_only_defender_business_reward': self.business_reward,
                'memory_size': self.memory.memory_size(),
                'identity_transactions': self.index, 'modified_attack_examples': 0,
                'attack_validity_rate': None, 'attack_success_rate': None,
                'attack_status': 'BLOCKED BY THREAT MODEL; identity is not an attack',
                'simulation_time': self.now}
