"""Whitelisted observations with fixed analytic encoding, no fitted statistics."""
import numpy as np
from gymnasium import spaces
from .transaction_stream import METADATA


def bounded_value(value):
    if value is None:
        return (0.0, 1.0)
    x = float(value)
    if np.isnan(x):
        return (0.0, 1.0)
    if not np.isfinite(x):
        raise ValueError('Infinite observable feature')
    return (float(np.tanh(np.sign(x)*np.log1p(abs(x))/10)), 0.0)


class CourtObservations:
    def __init__(self, features=('transactionamt',)):
        if METADATA & set(features):
            raise ValueError('Hidden labels/identifiers/time are not observable feature inputs')
        self.features = tuple(features)
        self.names = [n+s for n in features for s in ('_value', '_missing')] + [
            'probability', 'flag', 'court_budget_ratio', 'window_progress', 'pending_scaled',
            'recent_verified_fraud_rate', 'recent_verified_count_scaled', 'time_since_defender_update_scaled']
        self.space = spaces.Box(-1, 1, shape=(len(self.names),), dtype=np.float32)

    def encode(self, features, probability, decision, budget, pending, memory, now, defender_updated_at):
        values = [v for name in self.features for v in bounded_value(features[name])]
        stats = memory.get_verified_statistics(now)
        values += [probability, float(decision == 'FLAG'), budget.court_ratio, budget.progress,
                   pending/(1+pending), stats['fraud_rate'], stats['count']/(1+stats['count']),
                   (now-defender_updated_at)/(86400+now-defender_updated_at)]
        out = np.asarray(values, dtype=np.float32)
        if not self.space.contains(out):
            raise ValueError('Invalid Court observation')
        return out
