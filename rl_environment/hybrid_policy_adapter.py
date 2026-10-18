"""Frozen PPO chooses candidates without reading any Defender probability."""
import copy
import numpy as np
import torch
from .assumption_attack_adapter import AssumptionAttackAdapter,engine_for
from .transaction_stream import InternalTransaction
class PPOAttackAdapter(AssumptionAttackAdapter):
    def __init__(self,plausibility,config,policy,seed=42):
        super().__init__(plausibility,config,.05,'identity',seed);self.policy_model=policy
    def prepare(self,records):
        result=super().prepare(records)
        indices=[i for i,r in enumerate(records) if r.hidden_label==1]
        engines={i:engine_for(self.plausibility,self.config,.05) for i in indices}
        for i,e in engines.items():e.reset(records[i].features)
        torch.manual_seed(self.seed+91000);active=indices.copy()
        while active:
            observations=np.stack([engines[i].observation() for i in active])
            with torch.no_grad():actions=self.policy_model.sample(torch.tensor(observations))[0].numpy()
            for i,a in zip(active,actions):engines[i].apply(a)
            active=[i for i in active if not engines[i].done]
        self.episodes={}
        for i,e in engines.items():
            r=records[i];valid=e.result['status'] in ('ASSUMPTION_VALID','UNCHANGED');features=copy.deepcopy(e.current if valid else r.features)
            result[i]=InternalTransaction(r.transaction_id,r.transaction_time,features,r.hidden_label)
            if valid:self.episodes[r.transaction_id]=e
            self.provenance[r.transaction_id].update(status=e.result['status'],modified=e.result['status']=='ASSUMPTION_VALID',action_history=e.history,action=e.history[-1] if e.history else None,modified_amount=str(features['transactionamt']),policy='frozen_ppo')
        return result
