"""Verified-only memory; no prepopulation, no premature labels."""
import copy
import numpy as np


class VerifiedMemory:
    def __init__(self):
        self._records = []
        self._ids = set()

    def add_verified(self, record, now):
        if 'hidden_label' in record or record.get('verified_label') not in (0, 1):
            raise ValueError('Only released binary labels may enter memory')
        if record.get('completion_time', float('inf')) > now or record['completion_time'] < record['request_time']:
            raise ValueError('Premature verification')
        if record['transaction_id'] in self._ids:
            raise ValueError('Duplicate verified transaction')
        self._records.append(copy.deepcopy(record))
        self._ids.add(record['transaction_id'])

    def get_recent_verified(self, now, limit=100):
        return copy.deepcopy([r for r in self._records if r['completion_time'] <= now][-limit:]) if limit > 0 else []

    def get_verified_statistics(self, now, limit=100):
        # Aggregate private records directly; avoid copying hundreds of predictors
        # on every agent observation. Only the two scalar aggregates leave here.
        rows = [r for r in self._records if r['completion_time'] <= now][-limit:] if limit > 0 else []
        return {'count': len(rows), 'fraud_rate': sum(r['verified_label'] for r in rows)/len(rows) if rows else 0.0}

    def sample_verified(self, size, now, seed=42):
        eligible = [r for r in self._records if r['completion_time'] <= now]
        indices = np.random.default_rng(seed).choice(len(eligible), min(size, len(eligible)), replace=False)
        return [copy.deepcopy(eligible[i]) for i in indices]

    def memory_size(self):
        return len(self._records)
