"""Minimal deterministic value-regression update for integration tests only.

This linear history summary is a function approximation, NOT a sufficient Markov
state or a trained policy. MC targets evaluate the frozen data-collection policy.
"""
import numpy as np
class MonteCarloValueProbe:
    def __init__(self,observation_dim):
        self.weights=np.zeros(2*observation_dim+2,dtype=float)
    def features(self,record):
        observations=np.asarray([h['observation'] for h in record['history']],dtype=float)
        return np.r_[observations[-1],observations.mean(axis=0),record['action'],1.]
    def update(self,records,learning_rate=.01):
        x=np.stack([self.features(r) for r in records]); y=np.array([r['target'] for r in records])
        error=x@self.weights-y;before=float(np.mean(error**2)/2)
        gradient=x.T@error/len(y)
        self.weights-=learning_rate*gradient
        after=float(np.mean((x@self.weights-y)**2)/2)
        return dict(loss_before=before,loss_after=after,gradient_norm=float(np.linalg.norm(gradient)),weights=self.weights.tolist())
