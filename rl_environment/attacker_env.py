"""Continuous adapter tested only on an explicitly controlled synthetic threat model."""
from dataclasses import dataclass
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class SyntheticQuantityThreat:
    """Fixture ONLY: adversary controls quantity; unit price fixed; total recomputed.

    Fraud label represents fraudulent intent independent of purchase quantity.
    This assertion is defined for the toy generator, not for IEEE-CIS.
    """
    scope: str = 'synthetic_controlled_fixture'
    mutable: tuple = ('quantity',)
    minimum: float = 1.0
    maximum: float = 10.0
    max_delta: float = 2.0

    def transform(self, original, current, delta):
        result = copy.deepcopy(current)
        result['quantity'] += float(delta[0])*self.max_delta
        if not self.minimum <= result['quantity'] <= self.maximum:
            raise ValueError('Quantity outside synthetic domain')
        cost = abs(result['quantity']-original['quantity'])/self.max_delta
        if cost > 1+1e-9:
            raise ValueError('Total perturbation budget exceeded')
        result['total'] = result['quantity']*result['unit_price']
        if any(result[k] != original[k] for k in original if k not in ('quantity', 'total')):
            raise ValueError('Immutable fixture feature changed')
        if not np.isclose(result['total'], result['quantity']*result['unit_price']):
            raise ValueError('Dependent total inconsistent')
        return result, cost


class ContinuousAttackerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, defender, original, threat=None, max_steps=3):
        super().__init__()
        if threat is None or threat.scope != 'synthetic_controlled_fixture':
            raise ValueError('Real-data attacker BLOCKED BY THREAT MODEL; use identity-only controller')
        if max_steps < 1 or set(original) != {'quantity', 'unit_price', 'total'}:
            raise ValueError('Synthetic fixture schema required')
        if not all(np.isfinite(v) for v in original.values()) or original['unit_price'] <= 0:
            raise ValueError('Finite fixture values and positive unit price required')
        if not threat.minimum <= original['quantity'] <= threat.maximum or threat.max_delta <= 0:
            raise ValueError('Invalid initial quantity or perturbation constraint')
        if not np.isclose(original['total'], original['quantity']*original['unit_price']):
            raise ValueError('Invalid initial dependent feature')
        self.defender, self.original, self.threat = defender, copy.deepcopy(original), threat
        self.max_steps = max_steps
        # One continuous quantity dimension + one continuous submission gate.
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(4+max_steps*2,), dtype=np.float32)
        self._done = True

    def _obs(self):
        values = [self.current['quantity']/self.threat.maximum,
                  self.original['quantity']/self.threat.maximum, 1-self.cost,
                  self.steps/self.max_steps] + self.history.flatten().tolist()
        return np.asarray(values, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current, self.steps, self.cost, self.queries = copy.deepcopy(self.original), 0, 0., 0
        self.history = np.zeros((self.max_steps, 2), dtype=np.float32)
        self._done = False
        self.internal_vectors = []
        return self._obs(), {'scope': self.threat.scope}

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset after episode completion')
        try:
            action = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError):
            self._done = True
            return self._obs(), -2., True, False, {'valid': False, 'queries': self.queries}
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            self._done = True
            return self._obs(), -2., True, False, {'valid': False, 'queries': self.queries}
        action = np.clip(action, -1, 1)
        self.history[self.steps] = action
        self.steps += 1
        try:
            modified, cost = self.threat.transform(self.original, self.current, action[:-1])
        except ValueError:
            self._done = True
            return self._obs(), -2., True, False, {'valid': False, 'queries': self.queries}
        self.current, self.cost = modified, cost
        self.internal_vectors.append({'original': copy.deepcopy(self.original), 'modified': copy.deepcopy(modified),
                                      'action': action.tolist(), 'perturbation_cost': cost})
        submit = action[-1] > 0 or self.steps == self.max_steps
        if not submit:
            return self._obs(), 0., False, False, {'valid': True, 'queries': 0}
        # Synthetic fixture's intent is fraudulent; no other transaction label exists in the observation.
        decision, _ = self.defender.decide(self.current)
        self.queries += 1
        self._done = True
        reward = (1. if decision == 'APPROVE' else -1.)-cost
        return self._obs(), reward, True, False, {'valid': True, 'queries': self.queries, 'decision': decision}


def identity_attack(features):
    """No modification and no feedback query; not counted as a successful attack."""
    return copy.deepcopy(features)


class ContinuousControlAttackerEnv:
    """V2 factory: Box(k) has exactly k authorized mutation dimensions.

    A vector with every magnitude <=1e-6 means STOP (no mutation). Nonzero
    vectors mutate and continue until the third step, which submits automatically.
    The legacy explicit STOP-coordinate adapter remains for regression only.
    """
    def __new__(cls, engine, defender, transaction, weights=None):
        from .attack_constraint_engine import ConstraintAttackerEnv
        class ControlEnv(ConstraintAttackerEnv):
            def __init__(self):
                super().__init__(engine,defender,transaction,weights)
                self.action_space=spaces.Box(-1.,1.,shape=(len(engine.controls),),dtype=np.float32)
            def step(self,action):
                try:
                    a=np.asarray(action,dtype=np.float64)
                except (TypeError,ValueError):
                    return super().step(np.full(len(engine.controls)+1,2.,dtype=np.float32))
                if a.shape!=self.action_space.shape or not np.isfinite(a).all() or (np.abs(a)>1).any():
                    # Send invalid bounds to the existing reject-and-terminate engine.
                    return super().step(np.full(len(engine.controls)+1,2.,dtype=np.float32))
                stop=bool((np.abs(a)<=1e-6).all())
                mapped=np.r_[np.zeros_like(a) if stop else a,1. if stop else -1.].astype(np.float32)
                return super().step(mapped)
        return ControlEnv()
