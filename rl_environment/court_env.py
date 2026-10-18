"""Legacy Gym mechanics adapter; arrival rewards are diagnostic only.

For Phase 3.6 learner-compatible collection use FiniteCourtCollector, which
declares a finite decision horizon and waits for complete episode maturation.
Do not send this legacy scalar reward stream directly to vanilla PPO/DQN.
"""
import gymnasium as gym
from gymnasium import spaces


class CourtEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, simulator):
        super().__init__()
        self._simulator = simulator
        self.action_space = spaces.Discrete(2)
        self.observation_space = simulator.observations.space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium reset(seed=None) continues its RNG, rather than reseeding to 42.
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        return self._simulator.reset(episode_seed), {}

    def step(self, action):
        return self._simulator.step(action)
