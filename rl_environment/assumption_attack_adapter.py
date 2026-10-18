"""Offline Mode B adapter: private eligibility, copies, one frozen terminal query.

Fraud labels are used solely by the simulator's source eligibility rule. They
never enter observations. Batch independent inference is equivalent to one
terminal query per arrival; candidate selection cannot inspect these scores.
"""
import copy
import numpy as np
from .assumption_attack_engine import AssumptionAttackEngine
from .transaction_stream import TransactionStream,InternalTransaction
from .fraud_simulator import FraudSimulator

def proposal(policy,transaction_id,seed=42):
    if policy=='identity':return 0.
    if policy=='fixed_decrease':return -1.
    if policy=='fixed_increase':return 1.
    if policy=='random_uniform':return float(np.random.default_rng(np.random.SeedSequence([seed,int(transaction_id)])).uniform(-1,1))
    raise ValueError('Unregistered reference policy')

def engine_for(plausibility,config,epsilon,precision=8,scale=3,mode='ASSUMPTION_BASED'):
    return AssumptionAttackEngine(plausibility,epsilon,precision,scale,mode=mode,lambda_cost=config['lambda_cost'],label_contract=config['label_contract'],dependency_contract=config['dependency_contract'])

def prepare_episode(engine,transaction,action):
    engine.reset(transaction)
    result=engine.apply([action])
    if not engine.done:result=engine.apply([0.])
    return result

class AssumptionAttackAdapter:
    def __init__(self,plausibility,config,epsilon=.05,policy='random_uniform',seed=42):
        self.plausibility,self.config,self.epsilon,self.policy,self.seed=plausibility,config,epsilon,policy,seed
    def prepare(self,records):
        modified=[];self.provenance={};self.episodes={}
        for record in records:
            features=copy.deepcopy(record.features)
            provenance=dict(mode='identity',modified=False,source_transaction_id=record.transaction_id,source_transaction_time=record.transaction_time,original_amount=str(features['transactionamt']),modified_amount=str(features['transactionamt']),label_contract=None,dependency_status='NOT_MODIFIED')
            if record.hidden_label==1:
                engine=engine_for(self.plausibility,self.config,self.epsilon)
                a=proposal(self.policy,record.transaction_id,self.seed)
                result=prepare_episode(engine,features,a)
                if result['status'] in ('ASSUMPTION_VALID','UNCHANGED'):
                    features=result['modified_transaction'];self.episodes[record.transaction_id]=engine
                provenance.update(mode='ASSUMPTION_BASED',modified=result['status']=='ASSUMPTION_VALID',status=result['status'],action=a,epsilon=self.epsilon,modified_amount=str(features['transactionamt']),label_contract=self.config['label_contract'],dependency_status='UNKNOWN_DEPENDENCY',causal_validity=False,source_feature_vector=copy.deepcopy(record.features))
            self.provenance[record.transaction_id]=provenance
            modified.append(InternalTransaction(record.transaction_id,record.transaction_time,features,record.hidden_label))
        return modified

class AssumptionFraudSimulator(FraudSimulator):
    def __init__(self,stream,defender,config,adapter):
        self.attack_adapter=adapter
        records=adapter.prepare(stream.internal_records())
        safe_config=copy.deepcopy(config);safe_config.setdefault('attacker',{})['real_data_mode']='identity'
        # Base simulator only evaluates already finalized per-arrival vectors.
        # Its strict perturbation path is unchanged and remains blocked.
        super().__init__(TransactionStream.synthetic(records),defender,safe_config)
        for i,r in enumerate(self._records):
            engine=adapter.episodes.get(r.transaction_id)
            if engine is not None:
                decision='FLAG' if self._probabilities[i]>defender.threshold else 'APPROVE'
                reward=engine.resolve(decision)
                adapter.provenance[r.transaction_id].update(query_count=engine.queries,attacker_reward=reward,terminal_decision=decision)
    def _request(self,record,p,decision,source,action_id,court_action,blocked):
        request=dict(transaction_id=record.transaction_id,transaction_time=record.transaction_time,features=copy.deepcopy(record.features),original_modified_metadata=copy.deepcopy(self.attack_adapter.provenance[record.transaction_id]),defender_probability=p,defender_decision=decision,court_investigation_decision='INVESTIGATE' if court_action==1 else 'SKIP',court_budget_blocked=blocked,source=source,action_id=action_id,hidden_label=record.hidden_label,defender_model_version=self.defender.version)
        self.verification.request(request,self.now)
    def summary(self):
        result=super().summary()
        attacked=[self.attack_adapter.provenance[r.transaction_id] for r in self._records[:self.index]]
        n=sum(p.get('modified',False) for p in attacked)
        result.update(identity_transactions=self.index-n,modified_attack_examples=n,attack_status='ASSUMPTION_BASED; causal validity not established',attack_validity_rate=None,attack_success_rate=None)
        return result
