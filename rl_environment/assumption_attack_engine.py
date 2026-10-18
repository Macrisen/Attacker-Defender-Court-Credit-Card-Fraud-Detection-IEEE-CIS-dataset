"""Explicit Mode B arithmetic contract; never promotes UNKNOWN causal controls."""
import copy,math
from decimal import Decimal,InvalidOperation,ROUND_HALF_EVEN,ROUND_CEILING,ROUND_FLOOR,localcontext
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .attacker_reward import assumption_reward

class AttackStatus(str,Enum):
    CAUSALLY_VALIDATED='CAUSALLY_VALIDATED'
    ASSUMPTION_VALID='ASSUMPTION_VALID'
    ASSUMPTION_REJECTED='ASSUMPTION_REJECTED'
    UNKNOWN_DEPENDENCY='UNKNOWN_DEPENDENCY'
    INVALID_ACTION='INVALID_ACTION'
    UNCHANGED='UNCHANGED'
    STRICT_BLOCKED='STRICT_BLOCKED'

class AssumptionAttackEngine:
    names=['current_amount_scaled','original_amount_scaled','relative_signed_change','remaining_budget','remaining_steps','step_index','current_missing','original_missing','action_1','action_2','action_3']
    def __init__(self,plausibility,epsilon=.05,precision=8,scale=3,mode='ASSUMPTION_BASED',max_steps=3,lambda_cost=.1,label_contract=None,dependency_contract=None):
        if mode not in ('STRICT','ASSUMPTION_BASED'):raise ValueError('Unknown mode')
        self.epsilon=Decimal(str(epsilon))
        if not self.epsilon.is_finite() or not 0<self.epsilon<1 or max_steps!=3 or precision<=scale or scale<0 or lambda_cost<0 or not math.isfinite(lambda_cost):raise ValueError('Invalid fixed contract')
        if mode=='ASSUMPTION_BASED' and (not label_contract or not dependency_contract):raise ValueError('Explicit counterfactual contracts required')
        self.mode,self.plausibility=mode,plausibility
        self.precision,self.scale=precision,scale
        self.quantum=Decimal(1).scaleb(-scale);self.maximum=Decimal(10)**(precision-scale)-self.quantum
        self.max_steps=max_steps;self.lambda_cost=lambda_cost
        self.label_contract,self.dependency_contract=label_contract,dependency_contract
        self.action_space=spaces.Box(-1,1,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-1,1,shape=(11,),dtype=np.float32)
        self.done=True
    def valid_amount(self,value):
        try:
            # Floats must not silently enter a Decimal source schema.
            if not isinstance(value,Decimal) or not value.is_finite() or not 0<value<=self.maximum:return False
            return value==value.quantize(self.quantum)
        except (InvalidOperation,TypeError,ValueError):return False
    def reset(self,transaction):
        self.original=copy.deepcopy(transaction);self.current=copy.deepcopy(transaction)
        self.steps=0;self.history=[];self.done=False;self.queries=0;self.cost=0.;self.result=None
        self.original_valid=self.valid_amount(transaction.get('transactionamt'))
        return self.observation()
    def observation(self):
        def amount(x):return float(np.tanh(math.log1p(float(x))/10)) if self.valid_amount(x) else 0.
        original=self.original.get('transactionamt');current=self.current.get('transactionamt')
        delta=float((current-original)/original/self.epsilon) if self.original_valid else 0.
        values=[amount(current),amount(original),delta,max(0.,1-abs(delta)),(self.max_steps-self.steps)/self.max_steps,self.steps/self.max_steps,float(not self.valid_amount(current)),float(not self.original_valid)]+self.history+[0.]*(self.max_steps-len(self.history))
        result=np.asarray(values,dtype=np.float32)
        if not self.observation_space.contains(result):raise ValueError('Invalid observation')
        return result
    def apply(self,action):
        if self.done:raise RuntimeError('Episode already terminated')
        def finish(status,reason,terminal=True):
            self.done=terminal
            self.result=dict(status=status,reason=reason,terminal=terminal,dependency_status='UNKNOWN_DEPENDENCY',label_contract=self.label_contract,dependency_contract=self.dependency_contract,causal_validity=False,cost=self.cost,modified_transaction=copy.deepcopy(self.current),diagnostics=None)
            if self.original_valid and status in ('ASSUMPTION_VALID','UNCHANGED'):
                self.result['diagnostics']=self.plausibility.diagnostics(self.original['transactionamt'],self.current['transactionamt'],self.original)
            return copy.deepcopy(self.result)
        if self.mode=='STRICT':return finish('STRICT_BLOCKED','UNKNOWN amount control remains forbidden under strict causal model')
        if not self.original_valid:return finish('ASSUMPTION_REJECTED','Original amount missing, non-Decimal, nonpositive, nonfinite or not representable')
        try:a=np.asarray(action,dtype=float)
        except (TypeError,ValueError):return finish('INVALID_ACTION','Action is not numeric')
        if a.shape!=(1,) or not np.isfinite(a).all() or (np.abs(a)>1).any():return finish('INVALID_ACTION','Expected finite Box(1) action in [-1,1]')
        self.steps+=1;value=float(a[0]);self.history.append(value)
        stop=abs(value)<=1e-6
        if not stop:
            original=self.original['transactionamt']
            with localcontext() as ctx:
                ctx.prec=50
                exact=original*(1+self.epsilon*Decimal(str(value)))
                lower=max(self.quantum,(original*(1-self.epsilon)).quantize(self.quantum,rounding=ROUND_CEILING))
                upper=min(self.maximum,(original*(1+self.epsilon)).quantize(self.quantum,rounding=ROUND_FLOOR))
                candidate=exact.quantize(self.quantum,rounding=ROUND_HALF_EVEN)
                # Schema overflow is rejected, never silently clipped to schema limits.
                if candidate>self.maximum:return finish('ASSUMPTION_REJECTED','Decimal precision overflow')
                candidate=min(upper,max(lower,candidate))
                if not self.valid_amount(candidate) or abs((candidate-original)/original)>self.epsilon:return finish('ASSUMPTION_REJECTED','Positive Decimal/budget invariant failed')
                self.current['transactionamt']=candidate
                self.cost=float(abs((candidate-original)/original)/self.epsilon)*self.lambda_cost
        status='ASSUMPTION_VALID' if self.current['transactionamt']!=self.original['transactionamt'] else 'UNCHANGED'
        return finish(status,'Accepted under feature-space assumptions only',stop or self.steps==self.max_steps)
    def resolve(self,decision):
        if not self.done or self.result['status'] not in ('ASSUMPTION_VALID','UNCHANGED') or self.queries:raise ValueError('No unresolved terminal query permitted')
        if decision not in ('FLAG','APPROVE'):raise ValueError('Decision-only response required')
        self.queries=1
        return assumption_reward(self.result['status'],decision,self.cost)

class AssumptionAttackerEnv(gym.Env):
    """Source is supplied by private fraud-only simulator selection, not observation."""
    metadata={'render_modes':[]}
    def __init__(self,engine,defender,transaction,eligible_fraud):
        super().__init__()
        if not eligible_fraud:raise ValueError('This attacker task requires a privately selected fraud source')
        if engine.mode!='ASSUMPTION_BASED':raise ValueError('STRICT cannot instantiate an enabled attack learner')
        self.engine,self.defender=engine,defender;self._source=copy.deepcopy(transaction)
        self.action_space,self.observation_space=engine.action_space,engine.observation_space
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return self.engine.reset(self._source),{'mode':'ASSUMPTION_BASED'}
    def step(self,action):
        r=self.engine.apply(action);reward=0.;decision=None
        if r['terminal']:
            if r['status'] in ('ASSUMPTION_VALID','UNCHANGED'):
                features={n:r['modified_transaction'][n] for n in self.defender.feature_names}
                decision,_=self.defender.decide(features)
                reward=self.engine.resolve(decision)
            else:reward=assumption_reward(r['status'],cost=self.engine.cost)
        info=dict(status=r['status'],queries=self.engine.queries,decision=decision,dependency_status='UNKNOWN_DEPENDENCY',causal_validity=False)
        return self.engine.observation(),reward,self.engine.done,False,info
