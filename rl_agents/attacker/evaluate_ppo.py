"""Frozen single-query evaluation. Original scores stay evaluator-only."""
import numpy as np
import pandas as pd
import torch
from rl_environment.assumption_attack_adapter import proposal
from rl_environment.attacker_reward import assumption_reward
from .ppo_agent import SquashedActorCritic

def load_policy(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    agent=SquashedActorCritic();agent.load_state_dict(checkpoint['policy']);agent.eval()
    return agent

def evaluate(data,agent=None,policy='ppo',seed=42,bandit=False):
    torch.manual_seed(seed+91000)
    rows=[];engines=[data.engine(i) for i in range(len(data.sources))];active=list(range(len(engines)));histories=[[] for e in engines]
    while active:
        obs=np.stack([engines[i].observation() for i in active])
        if agent is not None:
            with torch.no_grad():actions=agent.sample(torch.tensor(obs))[0].numpy().reshape(-1)
        else:actions=np.array([proposal(policy,int(data.ids[i]),seed) if engines[i].steps==0 else 0. for i in active])
        for i,a in zip(active,actions):
            e=engines[i];histories[i].append(float(a));e.apply([float(a)])
            if bandit and not e.done:e.apply([0.])
        active=[i for i in active if not engines[i].done]
    original=data.probabilities(np.arange(len(engines)))
    accepted=[i for i,e in enumerate(engines) if e.result['status'] in ('ASSUMPTION_VALID','UNCHANGED')]
    modified=data.probabilities(accepted,[engines[i].current['transactionamt'] for i in accepted]);mapping=dict(zip(accepted,modified))
    for i,e in enumerate(engines):
        p=mapping.get(i);decision=('FLAG' if p>1/6 else 'APPROVE') if p is not None else None
        reward=e.resolve(decision) if p is not None else assumption_reward(e.result['status'],cost=e.cost)
        rows.append(dict(version=data.version,window=data.window,seed=seed,policy=policy,transactionid=int(data.ids[i]),transactiondt=int(data.times[i]),status=e.result['status'],original_amount=str(e.original['transactionamt']),modified_amount=str(e.current['transactionamt']),original_probability=float(original[i]),modified_probability=float(p) if p is not None else None,original_flag=bool(original[i]>1/6),modified_flag=bool(p>1/6) if p is not None else None,reward=reward,cost=e.cost,absolute_relative_change=e.cost/.1*.05,query_count=e.queries,actions=str(histories[i]),final_action=histories[i][-1]))
    return pd.DataFrame(rows)

def aggregate(frame):
    valid=frame.status.eq('ASSUMPTION_VALID');det=frame.original_flag;flips=valid&det&~frame.modified_flag.fillna(True).astype(bool)
    den=int((valid&det).sum());all_detected=int(det.sum())
    return dict(sources=len(frame),accepted_nonidentity=int(valid.sum()),identity_rate=float(frame.status.eq('UNCHANGED').mean()),invalid_rate=float((~frame.status.isin(['ASSUMPTION_VALID','UNCHANGED'])).mean()),originally_detected=all_detected,accepted_detected=den,flips=int(flips.sum()),conditional_evasion=float(flips.sum()/den) if den else None,selection_flip_fraction=float(flips.sum()/all_detected) if all_detected else 0.,overall_approved_fraction=float((~frame.modified_flag.fillna(True).astype(bool)).mean()),mean_reward=float(frame.reward.mean()),mean_cost=float(frame.cost.mean()),mean_perturbation=float(frame.absolute_relative_change.mean()),median_perturbation=float(frame.absolute_relative_change.median()),queries=int(frame.query_count.sum()))
