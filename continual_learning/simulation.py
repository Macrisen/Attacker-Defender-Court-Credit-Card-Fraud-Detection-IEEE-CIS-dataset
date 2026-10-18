"""Prequential Development refits; only matured memory crosses updater boundary."""
import copy,gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rl_agents.court.trajectory_collector import StreamData,KeyedVerification,KeyedAudits
from rl_agents.court.train_court import load_policy
from rl_environment.fraud_simulator import FraudSimulator
from rl_environment.assumption_attack_adapter import AssumptionFraudSimulator
from rl_environment.transaction_stream import TransactionStream
from rl_environment.court_observation import CourtObservationV2
from .verified_replay import VerifiedReplay
from .defender_updater import DefenderUpdater
from .drift_monitor import DriftMonitor
ROOT=Path(__file__).resolve().parents[1]
class AdaptiveSimulator(FraudSimulator):
    # Reuse provenance-rich request method, but resolve attack feedback only at arrival.
    _request=AssumptionFraudSimulator._request
    def summary(self):
        result=FraudSimulator.summary(self)
        modified=sum(self.attack_adapter.provenance[r.transaction_id].get('modified',False) for r in self._records[:self.index])
        result.update(identity_transactions=self.index-modified,modified_attack_examples=modified,attack_status='ASSUMPTION_BASED; causal validity not established',attack_validity_rate=None,attack_success_rate=None)
        return result
    def __init__(self,records,defender,config,adapter):
        self.attack_adapter=adapter
        prepared=adapter.prepare(records) if adapter else records
        if adapter is None:
            class Identity:
                pass
            identity=Identity();identity.provenance={r.transaction_id:dict(mode='identity',modified=False,source_transaction_id=r.transaction_id,source_transaction_time=r.transaction_time) for r in records};identity.episodes={};self.attack_adapter=identity
        super().__init__(TransactionStream.synthetic(prepared),defender,config)

def run_development(version,seed,attacker,court_checkpoint,output,strategy='replay',capacity=1000,interval=10000,court_kind='rule'):
    import yaml
    data=StreamData(version,'rl_development');records=data.records();reference=records[:1000]
    replay=VerifiedReplay(capacity,[r.transaction_id for r in reference]);cfg=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text());cfg['seed']=seed
    sim=AdaptiveSimulator(records,data.defender,cfg,attacker);sim.verification=KeyedVerification(seed+1,**cfg['verification']);sim.audit_rng=KeyedAudits(sim,seed)
    sim.observations=CourtObservationV2(records[-1].transaction_time);sim.defender_deployed_at=data.protocol['windows']['defender_calibration']['max_time']
    updater=DefenderUpdater(version,output,with_base=strategy!='no_replay') if strategy!='frozen' else None
    policy=load_policy(court_checkpoint) if court_kind=='learned' else None;h=policy.initial() if policy else None;torch.manual_seed(seed);rng=np.random.default_rng(seed)
    history=[];forgetting=[];monitor=DriftMonitor();memory_cursor=0;start=time.monotonic()
    # Reference IDs never enter update fitting or calibration.
    def reference_score(model):
        from sklearn.metrics import average_precision_score,brier_score_loss
        p=model.predict_many([r.features for r in reference]);y=[r.hidden_label for r in reference]
        return dict(AP=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)))
    forgetting.append(dict(deployment_time=sim.now,model_version=data.defender.version,**reference_score(data.defender)))
    while not sim.done:
        # _release was already called by the previous transition at this exact arrival.
        # Private controller consumes only newly released records, never pending queue labels.
        for record in sim.memory._records[memory_cursor:]:
            record['dataset_version']=version
            record['attack_mode']=record.get('original_modified_metadata',{}).get('mode','identity')
            replay.insert(record,sim.now)
        memory_cursor=sim.memory.memory_size()
        if sim.index and sim.index%interval==0:
            recent=sim.actions[-interval:];monitor.observe([r.features['transactionamt'] for r in sim._records[sim.index-interval:sim.index]],[a['defender_probability'] for a in recent],sim.now)
            if updater:
                new,info=updater.update(replay,sim.now);history.append(dict(arrivals=sim.index,**info))
                if new:
                    sim.defender=new;sim.defender_deployed_at=sim.now
                    # Independent cached inference only; no future labels/statistical fitting.
                    sim._probabilities[sim.index:]=new.predict_many([r.features for r in sim._records[sim.index:]])
                    forgetting.append(dict(deployment_time=sim.now,model_version=new.version,**reference_score(new)))
        index=sim.index;record=sim._records[index]
        engine=sim.attack_adapter.episodes.get(record.transaction_id)
        if engine:
            decision='FLAG' if sim._probabilities[index]>sim.defender.threshold else 'APPROVE';reward=engine.resolve(decision)
            sim.attack_adapter.provenance[record.transaction_id].update(query_count=1,attacker_reward=reward,terminal_decision=decision)
        obs=sim.observation()
        if policy:action,_,_,h=policy.act(obs,h)
        else:action=int(obs[3]>0 or rng.random()<.005)
        model_version=sim.defender.version;sim.step(action);sim.actions[-1]['model_version_at_decision']=model_version
    horizon=sim.summary();sim.drain_pending()
    # Tail labels populate diagnostics only; no model update after decision horizon.
    memory=sim.memory.get_recent_verified(sim.now,sim.memory.memory_size())
    for record in memory:
        record['dataset_version']=version
        record['attack_mode']=record.get('original_modified_metadata',{}).get('mode','identity')
    summary=dict(**sim.summary(),updates=updater.count if updater else 0,elapsed_seconds=time.monotonic()-start,final_model_path=str(sim.defender.artifact_dir),censored_at_horizon=horizon['investigations_pending'],strategy=strategy,capacity=capacity,update_interval=interval,verified_court=sum(r['source']=='court' for r in memory),verified_audit=sum(r['source']=='audit' for r in memory),court_verified_fraud=sum(r['source']=='court' and r['verified_label'] for r in memory),audit_verified_fraud=sum(r['source']=='audit' and r['verified_label'] for r in memory),reference_ids_excluded=len(reference))
    return summary,history,forgetting,monitor.history,sim,memory
