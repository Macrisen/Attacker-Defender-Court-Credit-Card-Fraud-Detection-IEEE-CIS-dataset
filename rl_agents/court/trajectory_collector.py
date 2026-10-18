"""Chronological fixed-policy collection with keyed common random verification."""
import copy,json,heapq
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import yaml
from rl_environment.verification_simulator import VerificationSimulator,PENDING
from rl_environment.transaction_stream import TransactionStream,InternalTransaction
from rl_environment.fraud_simulator import FraudSimulator
from rl_environment.court_credit_adapter import FiniteCourtCollector
from rl_environment.defender import FrozenDefender
ROOT=Path(__file__).resolve().parents[2]

class KeyedVerification(VerificationSimulator):
    def __init__(self,seed=42,**kwargs):super().__init__(seed,**kwargs);self.key_seed=seed
    def request(self,record,now):
        if now<self.now or record['transaction_id'] in self._states:raise ValueError('Duplicate/time reversal')
        if record['hidden_label'] not in (0,1):raise ValueError('Invalid label')
        self.now=now;r=copy.deepcopy(record);r['request_time']=now
        rng=np.random.default_rng(np.random.SeedSequence([self.key_seed,int(record['transaction_id'])]))
        r['completion_time']=now+float(rng.uniform(self.low,self.high));self._states[r['transaction_id']]=PENDING
        heapq.heappush(self._queue,(r['completion_time'],r['transaction_id'],r));return {'transaction_id':r['transaction_id'],'status':PENDING}

class KeyedAudits:
    def __init__(self,sim,seed):self.sim,self.seed=sim,seed
    def random(self):return float(np.random.default_rng(np.random.SeedSequence([self.seed,self.sim._records[self.sim.index].transaction_id,99])).random())

class StreamData:
    def __init__(self,version,window,frozen_protocol=None):
        if window not in ('rl_development','rl_validation'):
            if window!='final_holdout' or frozen_protocol is None:raise ValueError('Protected evaluation window')
            from scripts.final_evaluation import verify_freeze
            verify_freeze(frozen_protocol)
        self.version,self.window=version,window
        self.protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text());self.w=self.protocol['windows'][window]
        if window=='final_holdout':self.w=dict(self.w,decision_cutoff=self.w['max_time']-259200)
        self.defender=FrozenDefender(ROOT/'models/temporal_defender'/version)
        d=ds.dataset(ROOT/'data/processed/train_original.parquet')
        filt=(ds.field('transactiondt')>=self.w['min_time'])&(ds.field('transactiondt')<=self.w['decision_cutoff'])
        self.table=d.to_table(columns=self.defender.feature_names+['transactionid','transactiondt','isfraud'],filter=filt).sort_by([('transactiondt','ascending'),('transactionid','ascending')])
    def records(self,start=0,count=None):
        rows=self.table.slice(start,count).to_pylist()
        return [InternalTransaction(r['transactionid'],r['transactiondt'],{n:r[n] for n in self.defender.feature_names},r['isfraud']) for r in rows]

def make_sim(data,records,seed,attack_adapter=None):
    config=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text());config['seed']=seed
    if getattr(data,'no_investigations',False):config['budget']={'window_size':10000,'total':0,'audit_reserve':0}
    if getattr(data,'environment_budget',None):config['budget']=data.environment_budget
    if getattr(data,'environment_verification',None):config['verification']=data.environment_verification
    if attack_adapter:
        from rl_environment.assumption_attack_adapter import AssumptionFraudSimulator
        sim=AssumptionFraudSimulator(TransactionStream.synthetic(records),data.defender,config,attack_adapter)
    else:sim=FraudSimulator(TransactionStream.synthetic(records),data.defender,config)
    sim.verification=KeyedVerification(seed=seed+1,**config['verification']);sim.audit_rng=KeyedAudits(sim,seed)
    sim.defender_deployed_at=data.protocol['windows']['defender_calibration']['max_time']
    provenance=Path(data.defender.artifact_dir)/'training_provenance.json'
    if provenance.exists():
        deployed=float(json.loads(provenance.read_text())['deployment_time'])
        if deployed>records[0].transaction_time:raise ValueError('Defender deployed after evaluation begins')
        sim.defender_deployed_at=deployed
    return sim

def collect(data,records,policy,seed,policy_version=0,baseline=None,attack_adapter=None,threshold=1/6):
    import torch
    torch.manual_seed(seed)
    sim=make_sim(data,records,seed,attack_adapter)
    collector=FiniteCourtCollector(sim,data.w['max_time'],partition=data.window)
    obs_rows=[];actions=[];logps=[];values=[];h=policy.initial() if policy is not None else None
    rng=np.random.default_rng(seed)
    while not sim.done:
        obs=sim.observation().copy()
        if policy is not None:action,lp,v,h=policy.act(obs,h)
        else:
            lp=v=0.
            if baseline=='skip':action=0
            elif baseline=='random':action=int(rng.random()<.018)
            elif baseline=='rule':action=int(obs[3]>0 or rng.random()<.005)
            elif baseline in ('threshold','top_risk'):action=int(obs[2]>threshold)
            else:raise ValueError('Missing Court policy')
        obs_rows.append(obs);actions.append(action);logps.append(lp);values.append(v);collector.step(action)
    horizon=sim.summary();replay=collector.finish()
    # Validation finishes verification for metrics but never exports training replay.
    if data.window=='rl_development':returns=[r['target'] for r in replay]
    else:returns=[]
    memory=sim.memory.get_recent_verified(sim.now,sim.memory.memory_size())
    missed=sum(r['verified_label'] and r['defender_decision']=='APPROVE' and r['source']=='court' for r in memory)
    fraud=sum(r.hidden_label for r in records)
    metrics=dict(**sim.summary(),missed_fraud_recovered=missed,previously_flagged_fraud_verified=sum(r['verified_label'] and r['defender_decision']=='FLAG' and r['source']=='court' for r in memory),audit_fraud_verified=sum(r['verified_label'] and r['source']=='audit' for r in memory),mean_verification_delay=float(np.mean([r['completion_time']-r['request_time'] for r in memory])) if memory else None,censored_at_decision_horizon=horizon['investigations_pending'],maturity_end=sim.now,training_window=data.window,source_fraud=fraud)
    traj=dict(observations=np.asarray(obs_rows),actions=np.asarray(actions),log_probs=np.asarray(logps),values=np.asarray(values),returns=np.asarray(returns),policy_version=policy_version,mature=sim.now<=data.w['max_time'] and sim.verification.pending_count==0 and data.window=='rl_development')
    return traj,metrics,sim,memory
