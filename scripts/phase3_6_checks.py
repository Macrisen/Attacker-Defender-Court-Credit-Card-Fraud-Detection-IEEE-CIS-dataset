"""Executable small credit/gradient and constrained action experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import pyarrow as pa
from rl_environment.court_credit_assignment import DelayedCreditLedger
from rl_environment.court_credit_adapter import MatureEpisodeAdapter
from rl_agents.court.monte_carlo import MonteCarloValueProbe
from rl_environment.attacker_env import ContinuousControlAttackerEnv
from rl_environment.attack_constraint_engine import AttackConstraintEngine,attack_metrics
from rl_environment.feature_registry import FeatureRegistry, audit_schema
ROOT=Path(__file__).resolve().parents[1]
DAY=86400

def credit_fixture():
    ledger=DelayedCreditLedger(5*DAY,.5,training_start=0)
    for i,investigated in enumerate([True,True,False]):
        ledger.register(i,100+i,i*DAY,[.1*i,1.],int(investigated),investigated,'rl_development')
        ledger.attach_next(i,[.1*(i+1),1.],min(i+1,2)*DAY,terminated=(i==2))
    adapter=MatureEpisodeAdapter(0,2*DAY,5*DAY,.5)
    assert adapter.release(ledger,DAY,True)==[]
    assert all(r['completion_bonus'] is None for r in ledger.records()[:2])
    for i,bonus,label,delay in [(0,10.,1,2*DAY),(1,0.,0,2*DAY)]:
        event=dict(source='court',action_id=i,transaction_id=100+i,request_time=i*DAY,completion_time=i*DAY+delay,reward=bonus,verified_label=label)
        ledger.complete(event,event['completion_time'])
    assert adapter.release(ledger,3*DAY,True)==[] # full-cohort delay prevents SKIP bias
    records=adapter.release(ledger,5*DAY,True)
    expected=[1.,-1.,0.]
    rows=[]
    for r,origin,target in zip(records,ledger.records(),expected):
        rows.append(dict(origin_action_id=r['origin_action_id'],immediate_cost=origin['cost'],verification_delay_seconds=(origin['completion_timestamp']-origin['request_timestamp']) if origin['investigated'] else 0,verification_result=['VERIFIED_FRAUD','VERIFIED_LEGITIMATE','NOT_INVESTIGATED'][r['origin_action_id']],completion_bonus=origin['completion_bonus'],discounted_attributed_return=origin['final_attributed_reward'],computed_learning_target=r['target'],expected_target=target,status='PASS' if np.isclose(r['target'],target) else 'FAIL'))
    assert adapter.release(ledger,5*DAY,True)==[]
    probe=MonteCarloValueProbe(2);update=probe.update(records)
    assert update['loss_after']<update['loss_before'] and update['gradient_norm']>0
    # Exact deterministic repeat, not policy optimization.
    assert update==MonteCarloValueProbe(2).update(records)
    return rows,update,records,ledger

class ToyDefender:
    def decide(self,features):
        p=.8 if features['total']>55 else .1
        return ('FLAG' if p>1/6 else 'APPROVE',p)

def attack_checks():
    threat=yaml.safe_load((ROOT/'configs/attacker_threat_model.yaml').read_text())
    rows=[]
    for name,actions in [('modified_then_stop',[[-.5],[0.]]),('identity_stop',[[0.]]),('out_of_bounds',[[2.]]),('over_budget',[[1.],[1.]]),('three_steps',[[-.1],[-.1],[-.1]])]:
        e=AttackConstraintEngine(FeatureRegistry.synthetic(),'synthetic',threat['scenarios']['synthetic'])
        env=ContinuousControlAttackerEnv(e,ToyDefender(),dict(quantity=6.,unit_price=10.,total=60.))
        env.reset(seed=42)
        for action in actions:
            _,reward,done,_,info=env.step(action)
        rows.append(dict(action_type=name,feature_category='MUTABLE quantity; DEPENDENT total',constraint_status=info['status'],permitted_range='quantity [1,10]; displacement <=2; Box(1); zero STOP',reward=reward,queries=info['queries'],validity_status='SYNTHETIC_ONLY',limitation='No IEEE-CIS causal validity',eligible_fraud=True,decision=info.get('decision'),status=info['status']))
    expected=[.5,-1.,-2.,-3.,.7]
    for row,x in zip(rows,expected): row['reward_test']='PASS' if np.isclose(row['reward'],x) else 'FAIL'
    metrics=attack_metrics(rows)
    registry=FeatureRegistry(audit_schema(pa.schema([('transactionamt',pa.float64()),('transactionid',pa.int64())])))
    for scenario,expected in [('A','BLOCKED'),('B','UNKNOWN_VALIDITY')]:
        e=AttackConstraintEngine(registry,scenario,threat['scenarios'][scenario]);e.reset({'transactionamt':100.,'transactionid':7})
        r=e.apply([.5,1.])
        rows.append(dict(action_type='IEEE_schema_scenario_'+scenario,feature_category='UNKNOWN',constraint_status=r['status'],permitted_range='none established',reward=None,queries=0,validity_status=r['status'],limitation='Schema-only proposal probe; not a real eligible fraud attack',eligible_fraud=False,decision=None,status=r['status'],reward_test='PASS' if r['status']==expected else 'FAIL'))
    rejected=False
    try:registry.authorize([{'feature':'transactionid'}],'A')
    except ValueError:rejected=True
    rows.append(dict(action_type='immutable_identifier',feature_category='IMMUTABLE',constraint_status='REJECTED' if rejected else 'ERROR',permitted_range='no changes',reward=None,queries=0,validity_status='BLOCKED',limitation='Metadata is never an attacker control',eligible_fraud=False,decision=None,status='BLOCKED',reward_test='PASS' if rejected else 'FAIL'))
    return rows,metrics
