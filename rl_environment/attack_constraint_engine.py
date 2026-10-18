"""Fail-closed continuous controls with explicit counterfactual provenance."""
import copy
from decimal import Decimal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .feature_registry import IMMUTABLE
from .feature_dependency_validator import DependencyValidator


class AttackConstraintEngine:
    def __init__(self, registry, scenario, specification, max_steps=3, budget=1.0):
        self.registry, self.scenario, self.spec = registry, scenario, copy.deepcopy(specification)
        self.controls = self.spec.get('controls', [])
        registry.authorize(self.controls, scenario)
        self.dependency = DependencyValidator(self.spec.get('dependency_status', 'UNKNOWN'), self.spec.get('dependencies', []))
        if scenario != 'synthetic' and self.dependency.status == 'VERIFIED_FIXTURE':
            raise ValueError('Synthetic dependency evidence cannot authorize IEEE-CIS edits')
        if self.dependency.status == 'VERIFIED' and not self.spec.get('dependency_evidence'):
            raise ValueError('Verified dependency status requires a reviewed evidence reference')
        for rule in self.dependency.rules:
            entry = registry.entries.get(rule['output'], {})
            if entry.get('classification') != 'DEPENDENT' or not entry.get('evidence_level'):
                raise ValueError('Dependent output requires grounded registry entry')
        self.max_steps, self.budget = max_steps, float(budget)
        if max_steps < 1 or budget <= 0:
            raise ValueError('Invalid episode limits')
        self.action_space = spaces.Box(-1, 1, shape=(len(self.controls)+1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(4*len(self.controls)+2+max_steps*(len(self.controls)+1),), dtype=np.float32)
        self.done = True

    def reset(self, transaction):
        self.original = copy.deepcopy(transaction)
        self.current = copy.deepcopy(transaction)
        self.steps, self.cost, self.done = 0, 0., False
        self.history, self.records = [], []
        self._scales = []
        for control in self.controls:
            value = transaction.get(control['feature'])
            if value is None or not np.isfinite(float(value)):
                self._scales.append(None)
            else:
                self._scales.append(float(control.get('absolute_limit', abs(float(value))*control.get('relative_limit', 0))))
        return self.observation()

    def observation(self):
        values = []
        for name in [c['feature'] for c in self.controls]:
            for record in (self.current, self.original):
                v = record.get(name)
                missing = v is None or not np.isfinite(float(v))
                values.extend([0. if missing else float(np.tanh(float(v)/10)), float(missing)])
        values += [max(0., 1-self.cost/self.budget), self.steps/self.max_steps]
        padded = self.history+[[0.]*(len(self.controls)+1)]*(self.max_steps-len(self.history))
        values += [v for action in padded for v in action]
        return np.asarray(values, dtype=np.float32)

    def apply(self, action):
        if self.done:
            raise RuntimeError('Reset required')
        result = {'status': None, 'reason': '', 'scenario': self.scenario, 'modified_transaction': None,
                  'label_assumption': self.spec.get('label_assumption'), 'label_preserved_as_observed_fact': False,
                  'cost': self.cost, 'magnitude': 0., 'submit': False}
        def finish(status, reason, terminal=True):
            result.update(status=status, reason=reason)
            self.done = terminal
            self.records.append(copy.deepcopy(result))
            return result
        if not self.spec.get('enabled') or not self.controls:
            return finish('BLOCKED', 'No authorized mutable control in this scenario')
        try:
            a = np.asarray(action, dtype=float)
        except (ValueError, TypeError):
            return finish('INVALID_ACTION', 'Non-numeric action')
        if a.shape != self.action_space.shape or not np.isfinite(a).all() or (np.abs(a) > 1).any():
            return finish('INVALID_ACTION', 'Shape, finite-value or [-1,1] bound violation; no silent clipping')
        self.steps += 1
        self.history.append(a.tolist())
        candidate = copy.deepcopy(self.current)
        cost, magnitude = 0., 0.
        for i, control in enumerate(self.controls):
            name, scale = control['feature'], self._scales[i]
            if scale is None or scale <= 0:
                return finish('UNKNOWN_VALIDITY', 'Missing/zero control scale; no imputation')
            if isinstance(candidate[name], Decimal):
                candidate[name] += Decimal(str(a[i]))*Decimal(str(scale))
            else:
                candidate[name] += float(a[i])*scale
            value = float(candidate[name])
            if not np.isfinite(value) or not control.get('minimum', -float('inf')) <= value <= control.get('maximum', float('inf')):
                return finish('INVALID_ACTION', 'Domain violation: '+name)
            delta = abs(float(candidate[name])-float(self.original[name]))
            cost += control.get('weight', 1.)*delta/scale
            magnitude += delta
        if cost > self.budget+1e-10:
            return finish('INVALID_ACTION', 'Cumulative accepted-displacement budget exceeded')
        changed = any(candidate[c['feature']] != self.original[c['feature']] for c in self.controls)
        if changed and not self.spec.get('label_assumption'):
            return finish('UNKNOWN_VALIDITY', 'Counterfactual label-preservation assumption absent')
        try:
            self.dependency.recompute(candidate)
            validity, reason = self.dependency.validate(candidate)
        except (KeyError, TypeError, ValueError):
            return finish('INVALID_ACTION', 'Dependency recomputation failed')
        allowed = {c['feature'] for c in self.controls} | {r['output'] for r in self.dependency.rules}
        for key, value in self.original.items():
            if key in IMMUTABLE or key not in allowed:
                same = candidate.get(key) == value
                if isinstance(value, float) and np.isnan(value):
                    same = isinstance(candidate.get(key), float) and np.isnan(candidate[key])
                if not same:
                    return finish('INVALID_ACTION', 'Immutable/unregistered feature changed: '+key)
        if changed and validity != 'VALID':
            return finish('UNKNOWN_VALIDITY' if validity == 'UNKNOWN' else 'INVALID_ACTION', reason)
        self.current, self.cost = candidate, cost
        result.update(modified_transaction=copy.deepcopy(candidate), cost=cost, magnitude=magnitude,
                      submit=bool(a[-1] > 0 or self.steps >= self.max_steps))
        return finish('VALID_MODIFIED' if changed else 'VALID_UNCHANGED', 'Accepted under documented scenario contract', result['submit'])


def attacker_reward(status, decision=None, cost=0., weights=None):
    weights = weights or {'fraud_approved': 1., 'fraud_flagged': -1., 'invalid': -2., 'cost_weight': 1.}
    if status == 'INVALID_ACTION':
        return weights['invalid']-weights['cost_weight']*cost
    if status not in ('VALID_MODIFIED', 'VALID_UNCHANGED'):
        return None  # blocked/unknown is not a meaningful learning transition
    if decision not in ('FLAG', 'APPROVE'):
        raise ValueError('Terminal Defender decision required')
    return weights['fraud_approved' if decision == 'APPROVE' else 'fraud_flagged']-weights['cost_weight']*cost


class ConstrainedAttackEpisode:
    """At most one final query; paired baseline evaluation belongs outside agent API."""
    def __init__(self, engine, defender, weights=None):
        self.engine, self.defender, self.weights = engine, defender, weights
        self.action_space, self.observation_space = engine.action_space, engine.observation_space

    def reset(self, transaction):
        self.queries = 0
        return self.engine.reset(transaction)

    def step(self, action):
        r = self.engine.apply(action)
        info = {k: r[k] for k in ('status', 'reason', 'cost', 'label_assumption')}
        reward = 0.
        if r['status'] == 'INVALID_ACTION':
            reward = attacker_reward(r['status'], cost=self.engine.cost, weights=self.weights)
        elif r['submit']:
            if self.queries:
                raise RuntimeError('Query budget exhausted')
            inputs = {k: v for k, v in r['modified_transaction'].items() if k in self.engine.registry.entries and k not in IMMUTABLE}
            decision, _ = self.defender.decide(inputs)
            self.queries += 1
            info['decision'] = decision
            reward = attacker_reward(r['status'], decision, r['cost'], self.weights)
        info['queries'] = self.queries
        return self.engine.observation(), reward, self.engine.done, False, info


class ConstraintAttackerEnv(gym.Env):
    """Gymnasium adapter enabled only when the engine has established controls.

    The supplied record is internal; reset observations contain authorized control
    values only. Current IEEE-CIS A/B configurations cannot instantiate this env.
    """
    metadata = {'render_modes': []}

    def __init__(self, engine, defender, transaction, weights=None):
        super().__init__()
        if not engine.controls or not engine.spec.get('enabled') or not engine.spec.get('label_assumption') or engine.dependency.status not in ('VERIFIED', 'VERIFIED_FIXTURE'):
            raise ValueError('RL training blocked: controls, dependencies and label contract must be established')
        self._original = copy.deepcopy(transaction)
        self._episode = ConstrainedAttackEpisode(engine, defender, weights)
        self.action_space, self.observation_space = engine.action_space, engine.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._episode.reset(self._original), {'scenario': self._episode.engine.scenario}

    def step(self, action):
        return self._episode.step(action)


def attack_metrics(results):
    """Evaluation-only paired metrics; identity and unknown proposals are excluded."""
    valid = [r for r in results if r['status'] in ('VALID_UNCHANGED', 'VALID_MODIFIED')]
    accepted_modified = [r for r in results if r['status'] == 'VALID_MODIFIED']
    modified = [r for r in accepted_modified if r.get('eligible_fraud', False) and r.get('decision') in ('FLAG', 'APPROVE')]
    detected = [r for r in modified if r.get('original_decision') == 'FLAG']
    return {'attempts': len(results), 'validity_rate': len(valid)/len(results) if results else None,
            'rejection_rate': sum(r['status'] == 'INVALID_ACTION' for r in results)/len(results) if results else None,
            'valid_nonidentity_fraction': len(accepted_modified)/len(results) if results else None,
            'asr_denominator': len(modified), 'asr': sum(r['decision'] == 'APPROVE' for r in modified)/len(modified) if modified else None,
            'originally_detected_denominator': len(detected),
            'conditional_evasion_originally_detected': sum(r['decision'] == 'APPROVE' for r in detected)/len(detected) if detected else None}
