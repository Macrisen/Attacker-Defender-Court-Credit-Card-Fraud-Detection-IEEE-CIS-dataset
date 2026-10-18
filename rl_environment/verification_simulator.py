"""Labels are private until the sampled simulated completion time."""
import copy
import heapq
import numpy as np

NOT_INVESTIGATED = 'NOT_INVESTIGATED'
PENDING = 'PENDING'
VERIFIED_FRAUD = 'VERIFIED_FRAUD'
VERIFIED_LEGITIMATE = 'VERIFIED_LEGITIMATE'


class VerificationSimulator:
    def __init__(self, seed=42, min_days=1, max_days=3):
        if min_days <= 0 or max_days < min_days:
            raise ValueError('Delay must be positive')
        self.rng = np.random.default_rng(seed)
        self.low, self.high = min_days*86400, max_days*86400
        self._queue, self._states = [], {}
        self.now = float('-inf')

    def request(self, record, now):
        if now < self.now or record['transaction_id'] in self._states:
            raise ValueError('Duplicate request or time reversal')
        if record['hidden_label'] not in (0, 1) or now < record['transaction_time']:
            raise ValueError('Invalid investigation')
        self.now = now
        result = copy.deepcopy(record)
        result['request_time'] = now
        result['completion_time'] = now + float(self.rng.uniform(self.low, self.high))
        self._states[result['transaction_id']] = PENDING
        heapq.heappush(self._queue, (result['completion_time'], result['transaction_id'], result))
        return {'transaction_id': result['transaction_id'], 'status': PENDING}

    def release(self, now):
        if now < self.now:
            raise ValueError('Verification time cannot go backwards')
        self.now = now
        released = []
        while self._queue and self._queue[0][0] <= now:
            _, tid, record = heapq.heappop(self._queue)
            record['verified_label'] = record.pop('hidden_label')
            record['status'] = VERIFIED_FRAUD if record['verified_label'] else VERIFIED_LEGITIMATE
            self._states[tid] = record['status']
            released.append(record)
        return released

    def status(self, transaction_id):
        return self._states.get(transaction_id, NOT_INVESTIGATED)

    @property
    def pending_count(self):
        return len(self._queue)

    @property
    def next_completion(self):
        return self._queue[0][0] if self._queue else None

    def internal_checkpoint(self):
        """Private simulator checkpoint, not an agent info payload."""
        return copy.deepcopy(self._queue)
