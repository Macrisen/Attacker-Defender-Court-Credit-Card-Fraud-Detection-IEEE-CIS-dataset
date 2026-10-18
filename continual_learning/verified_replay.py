"""Bounded buffer containing only completed verification outcomes."""
import copy
class VerifiedReplay:
    def __init__(self,capacity=1000,excluded_ids=()):
        if capacity<1:raise ValueError('Invalid capacity')
        self.capacity=capacity;self.excluded=set(excluded_ids);self.rows=[];self.seen=set()
    def insert(self,record,now):
        if 'hidden_label' in record or record.get('verified_label') not in (0,1) or record.get('completion_time',float('inf'))>now:raise ValueError('Unverified/future label cannot enter replay')
        tid=record['transaction_id']
        if tid in self.seen:return False
        self.seen.add(tid)
        if tid in self.excluded:return False
        self.rows.append(copy.deepcopy(record));self.rows.sort(key=lambda r:(r['completion_time'],r['transaction_id']));self.rows=self.rows[-self.capacity:];return True
    def available(self,now):
        if any(r['completion_time']>now for r in self.rows):raise ValueError('Time reversed beyond available labels')
        return copy.deepcopy(self.rows)
