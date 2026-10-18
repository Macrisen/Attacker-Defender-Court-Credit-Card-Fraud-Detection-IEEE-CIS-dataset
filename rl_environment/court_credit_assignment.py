"""Origin-linked, maturity-gated replay records for time-discounted off-policy TD.

This is a replay adapter, NOT the legacy arrival-reward Gym stream. Do not feed
legacy step rewards to a DQN/PPO learner in addition to these attributed rewards.
"""
import copy
import math
import numpy as np


def time_discount(seconds, gamma_per_day=0.99):
    if seconds < 0 or not math.isfinite(seconds) or not 0 < gamma_per_day <= 1:
        raise ValueError('Invalid elapsed seconds or daily discount')
    return gamma_per_day**(seconds/86400.)


class DelayedCreditLedger:
    def __init__(self, training_end, gamma_per_day=0.99, training_partition='rl_development', training_start=float('-inf')):
        self.training_end = float(training_end)
        self.training_start = training_start
        self.gamma = gamma_per_day
        time_discount(0, gamma_per_day)
        self.partition = training_partition
        self._records = {}

    def register(self, action_id, transaction_id, request_time, observation, action, investigated,
                 partition, cost=1., defender_decision='APPROVE', due_timestamp=None):
        if action_id in self._records or action not in (0, 1) or (investigated and action != 1) or cost < 0:
            raise ValueError('Duplicate/invalid origin action')
        obs = np.asarray(observation, dtype=float)
        if not np.isfinite(obs).all():
            raise ValueError('Invalid historical observation')
        self._records[action_id] = {'origin_transaction_id': transaction_id, 'origin_decision_step_id': action_id,
            'court_action_id': action_id, 'request_timestamp': float(request_time), 'due_timestamp': due_timestamp,
            'completion_timestamp': None, 'observed_timestamp': None,
            'cost': float(cost) if investigated else 0., 'immediate_reward': -float(cost) if investigated else 0.,
            'completion_bonus': None if investigated else 0., 'final_attributed_reward': None if investigated else 0.,
            'partition': partition, 'training_eligibility': 'PENDING' if investigated else 'WAIT_NEXT_STATE',
            'investigated': investigated, 'action': action, 'defender_decision': defender_decision,
            'observation': obs.tolist(), 'next_observation': None, 'next_timestamp': None,
            'terminated': False, 'truncated': False, 'censored': False, 'exported': False}

    def attach_next(self, action_id, observation, timestamp, terminated=False, truncated=False):
        r = self._records[action_id]
        obs = np.asarray(observation, dtype=float)
        if r['next_timestamp'] is not None or timestamp < r['request_timestamp'] or not np.isfinite(obs).all():
            raise ValueError('Invalid/duplicate next state')
        r.update(next_observation=obs.tolist(), next_timestamp=float(timestamp),
                 terminated=bool(terminated), truncated=bool(truncated))
        self._eligibility(r)

    def complete(self, event, observed_at):
        if event['source'] != 'court':
            return False  # random audit must never earn a Court action a bonus
        r = self._records[event['action_id']]
        due = float(event['completion_time'])
        if (not r['investigated'] or r['completion_timestamp'] is not None or
            event['transaction_id'] != r['origin_transaction_id'] or
            float(event['request_time']) != r['request_timestamp'] or due <= r['request_timestamp'] or due > observed_at):
            raise ValueError('Wrong origin, duplicate completion or premature label event')
        if r['due_timestamp'] is not None and due != r['due_timestamp']:
            raise ValueError('Completion differs from registered due time')
        bonus = float(event['reward'])
        if not math.isfinite(bonus):
            raise ValueError('Nonfinite bonus')
        r.update(due_timestamp=due, completion_timestamp=due, observed_timestamp=float(observed_at),
                 completion_bonus=bonus,
                 final_attributed_reward=-r['cost']+time_discount(due-r['request_timestamp'], self.gamma)*bonus)
        self._eligibility(r)
        return True

    def _eligibility(self, r):
        if r['partition'] != self.partition or not self.training_start <= r['request_timestamp'] <= self.training_end:
            r['training_eligibility'] = 'EXCLUDED_PARTITION'
        elif r['censored']:
            r['training_eligibility'] = 'CENSORED_PENDING'
        elif r['truncated']:
            # A missing continuation is not a genuine terminal state.
            r['training_eligibility'] = 'CENSORED_TRUNCATION'
        elif (r['next_timestamp'] is not None and r['next_timestamp'] > self.training_end) or (r['observed_timestamp'] is not None and r['observed_timestamp'] > self.training_end):
            r['training_eligibility'] = 'EXCLUDED_BOUNDARY'
        elif r['next_timestamp'] is None:
            r['training_eligibility'] = 'WAIT_NEXT_STATE'
        elif r['final_attributed_reward'] is None:
            r['training_eligibility'] = 'PENDING'
        else:
            r['training_eligibility'] = 'ELIGIBLE'

    def censor_pending(self):
        for r in self._records.values():
            if r['final_attributed_reward'] is None:
                r['training_eligibility'] = 'CENSORED_PENDING'
                r['censored'] = True

    def ready(self, now):
        output = []
        for r in self._records.values():
            available_at = max(r['next_timestamp'] if r['next_timestamp'] is not None else float('inf'),
                               r['observed_timestamp'] if r['observed_timestamp'] is not None else r['request_timestamp'])
            if r['training_eligibility'] == 'ELIGIBLE' and not r['exported'] and available_at <= now <= self.training_end:
                record = copy.deepcopy(r)
                record['discount_to_next'] = time_discount(r['next_timestamp']-r['request_timestamp'], self.gamma)
                output.append(record)
                r['exported'] = True
        return output

    def records(self):
        """Evaluator/trainer ledger. Not an acting-policy observation."""
        return copy.deepcopy(list(self._records.values()))


def dqn_target(record, next_q):
    if record['training_eligibility'] != 'ELIGIBLE' or record['truncated']:
        raise ValueError('Unresolved/censored records cannot train')
    return record['final_attributed_reward'] + (0. if record['terminated'] else record['discount_to_next']*float(next_q))


class CourtCreditCollector:
    """Collects historical state/action/next-state and due events without relabeling observations."""
    def __init__(self, simulator, training_end, partition='rl_development', gamma_per_day=.99, training_start=float('-inf')):
        self.simulator = simulator
        self.partition = partition
        self.ledger = DelayedCreditLedger(training_end, gamma_per_day, training_start=training_start)

    def step(self, action):
        sim = self.simulator
        observation, timestamp, index = sim.observation().copy(), sim.now, sim.index
        tid = sim._records[index].transaction_id
        next_obs, arrival_reward, terminated, truncated, info = sim.step(action)
        log = sim.actions[-1]
        self.ledger.register(index, tid, timestamp, observation, action, log['source'] == 'court',
                             self.partition, sim.config.get('court', {}).get('investigation_cost', 1), log['defender_decision'])
        self.ledger.attach_next(index, next_obs, sim.now, terminated=terminated, truncated=truncated)
        for event in info['due_verifications']:
            self.ledger.complete(event, sim.now)
        ready = self.ledger.ready(sim.now)
        # Explicit structure prevents accidental use as a standard scalar-reward Gym API.
        return {'observation': next_obs, 'terminated': terminated, 'truncated': truncated,
                'ready_replay_records': ready, 'arrival_reward_evaluation_only': arrival_reward,
                'public_events': info['due_verifications']}
