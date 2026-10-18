"""Mature finite-episode Monte Carlo adapter; no one-step Bellman assumption.

Reuse the origin ledger. Each episode freezes its behavior policy. The terminal
criterion is a predeclared decision horizon, followed by a <=3 day maturity tail.
No future decisions occur in that tail. Complete ordered causal observation/action
histories remain available for a future recurrent learner. Compact states alone
are not asserted Markov. No off-policy max-Q bootstrapping is used.
"""
import copy
import numpy as np
from .court_credit_assignment import time_discount

class CausalHistory:
    """Persistent linked prefix: O(n) storage for n decisions, no future nodes."""
    def __init__(self, entry, previous=None):
        self.entry=copy.deepcopy(entry)
        self.previous=previous
    def __iter__(self):
        entries=[]; node=self
        while node is not None:
            entries.append(node.entry); node=node.previous
        return iter(reversed(entries))

class MatureEpisodeAdapter:
    def __init__(self,start,decision_end,training_end,gamma=.99,max_delay=259200):
        if not start<=decision_end<=training_end or decision_end+max_delay>training_end:raise ValueError('Missing maturity guard')
        self.start,self.end,self.training_end,self.gamma=float(start),float(decision_end),float(training_end),gamma
        self.max_delay=max_delay
        self.exported=False
    def release(self,ledger,now,complete_decision_horizon=False,truncated=False):
        if self.exported: return []
        if truncated: raise ValueError('Truncated episodes are censored; no zero terminal bootstrap')
        if not complete_decision_horizon or now<self.end+self.max_delay or now>self.training_end:return []
        records=sorted(ledger.records(),key=lambda r:(r['request_timestamp'],r['court_action_id']))
        if not records: return []
        if ledger.gamma!=self.gamma:raise ValueError('Discount mismatch')
        if records[-1]['request_timestamp']!=self.end or not records[-1]['terminated']:
            raise ValueError('Declared decision horizon not completed')
        for i,r in enumerate(records):
            if r['court_action_id']!=i or (i<len(records)-1 and (r['terminated'] or r['next_timestamp']!=records[i+1]['request_timestamp'])):
                raise ValueError('Incomplete or discontinuous decision trajectory')
        for r in records:
            if r['exported']:raise ValueError('Cannot mix one-step ledger replay and episode MC replay')
            if r['partition']!='rl_development' or not self.start<=r['request_timestamp']<=self.end:raise ValueError('Wrong origin partition')
            if r['censored'] or r['truncated'] or r['final_attributed_reward'] is None or r['next_timestamp'] is None:raise ValueError('Unresolved/censored trajectory')
            if r['next_timestamp']>self.training_end or (r['observed_timestamp'] is not None and r['observed_timestamp']>now):raise ValueError('Unavailable future information')
            if r['completion_timestamp'] is not None and r['completion_timestamp']-r['request_timestamp']>self.max_delay:raise ValueError('Delay contract violated')
        output=[];history=None
        returns=np.empty(len(records));future=0.
        for i in range(len(records)-1,-1,-1):
            r=records[i]
            if i+1<len(records):future*=time_discount(records[i+1]['request_timestamp']-r['request_timestamp'],self.gamma)
            future=r['final_attributed_reward']+future;returns[i]=future
        for i,r in enumerate(records):
            # Acting history contains only observations/actions, never eventual labels.
            history=CausalHistory(dict(observation=r['observation'],action=r['action'],timestamp=r['request_timestamp']),history)
            output.append(dict(origin_action_id=r['court_action_id'],history=history,action=r['action'],target=float(returns[i]),attributed_reward=r['final_attributed_reward'],available_at=now,bootstrap=0.,terminal_horizon=self.end))
        self.exported=True
        for r in ledger._records.values():r['exported']=True
        return output

class FiniteCourtCollector:
    """Declared finite decision episode plus separate verification-only tail.

    No policy changes within the episode. The last source row is a real terminal
    only because this class declares that horizon in advance. Interruptions are
    censored. Validation episodes cannot be exported as training replay.
    """
    def __init__(self,simulator,training_end,gamma=.99,partition='rl_development'):
        from .court_credit_assignment import DelayedCreditLedger
        from .court_observation import CourtObservationV2
        self.sim=simulator
        self.partition=partition
        self.start=simulator._records[0].transaction_time
        self.end=simulator._records[-1].transaction_time
        self.ledger=DelayedCreditLedger(training_end,gamma,training_start=self.start)
        self.adapter=MatureEpisodeAdapter(self.start,self.end,training_end,gamma)
        self.sim.observations=CourtObservationV2(self.end)
    def step(self,action):
        s=self.sim;obs=s.observation().copy();time=s.now;index=s.index;tid=s._records[index].transaction_id
        next_obs,_,_,_,info=s.step(action)
        log=s.actions[-1]
        self.ledger.register(index,tid,time,obs,action,log['source']=='court',self.partition,s.config.get('court',{}).get('investigation_cost',1.),log['defender_decision'])
        self.ledger.attach_next(index,next_obs,s.now,terminated=s.done,truncated=False)
        for event in info['due_verifications']:self.ledger.complete(event,s.now)
        return dict(observation=next_obs,decision_horizon_reached=s.done,public_events=info['due_verifications'],pending=s.verification.pending_count)
    def finish(self):
        if not self.sim.done:
            self.ledger.censor_pending()
            raise ValueError('Interrupted episode censored; no MC training target')
        for event in self.sim.drain_pending():self.ledger.complete(event,event['emitted_at'])
        # This advance reads no later transactions or labels; only waits out the
        # prespecified max delay of already requested investigations.
        now=self.end+self.adapter.max_delay
        self.sim.now=now
        if self.partition!='rl_development':return []
        return self.adapter.release(self.ledger,now,complete_decision_horizon=True)
