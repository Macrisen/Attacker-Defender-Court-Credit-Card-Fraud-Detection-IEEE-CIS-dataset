"""Development-only offline fraud sampling and terminal batched frozen inference."""
import copy,json
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import yaml
from rl_environment.assumption_attack_adapter import engine_for
from rl_environment.attack_plausibility import AttackPlausibility
from rl_environment.attacker_reward import assumption_reward
from rl_environment.defender import FrozenDefender
from defender.inference import encode,calibrate
ROOT=Path(__file__).resolve().parents[2]

class AttackData:
    def __init__(self,version,window,training=False):
        if training and window!='rl_development':raise ValueError('Policy optimization must use Development only')
        if window not in ('rl_development','rl_validation','final_holdout'):raise ValueError('Unknown window')
        if window=='final_holdout':raise ValueError('Final holdout requires separate frozen evaluation gate')
        self.version,self.window=version,window
        self.protocol=json.loads((ROOT/'reports/phase3_6/temporal_split_protocol.json').read_text())
        self.config=yaml.safe_load((ROOT/'configs/assumption_attack.yaml').read_text())
        self.plausibility=AttackPlausibility(json.loads((ROOT/'reports/phase3_7/plausibility_model.json').read_text()),self.protocol)
        self.defender=FrozenDefender(ROOT/'models/temporal_defender'/version)
        w=self.protocol['windows'][window];filt=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['max_time'])&(ds.field('isfraud')==1)
        names=self.defender.feature_names
        frame=ds.dataset(ROOT/'data/processed/train_original.parquet').to_table(columns=names+['transactionid','transactiondt','isfraud'],filter=filt).to_pandas().sort_values(['transactiondt','transactionid'])
        self.ids=frame.transactionid.to_numpy();self.times=frame.transactiondt.to_numpy()
        self.encoded,_=encode(frame,self.defender._baseline.schema,self.defender._baseline.mappings)
        self.amount_index=names.index('transactionamt')
        self.sources=frame[['transactionamt','productcd','card4','card6']].to_dict('records')
        gate=engine_for(self.plausibility,self.config,.05)
        if not all(gate.valid_amount(r['transactionamt']) for r in self.sources):raise ValueError('Invalid source amount; selection needs explicit audit')
        self.training=training
    def probabilities(self,indices,amounts=None):
        x=self.encoded[np.asarray(indices)].copy()
        if amounts is not None:x[:,self.amount_index]=np.asarray([float(a) for a in amounts],dtype=np.float32)
        m=self.defender._baseline
        return calibrate(m.model.predict(x,num_threads=4),m.calibrator)
    def engine(self,index):
        e=engine_for(self.plausibility,self.config,.05);e.reset(self.sources[index]);return e

class TrainingVector:
    def __init__(self,data,seed,n=32,bandit=False):
        if not data.training:raise ValueError('Training sampler requires permitted split')
        self.data,self.rng,self.n,self.bandit=data,np.random.default_rng(seed),n,bandit
        self.indices=self.rng.integers(len(data.sources),size=n);self.engines=[data.engine(i) for i in self.indices]
        self.queries=0;self.episodes=0;self.seen=set();self.finished=[]
    def observation(self):return np.stack([e.observation() for e in self.engines])
    def step(self,actions):
        rewards=np.zeros(self.n,np.float32);done=np.zeros(self.n,np.float32);terminal=[];indices=[];amounts=[]
        for j,(e,a) in enumerate(zip(self.engines,actions)):
            r=e.apply(a)
            if self.bandit and not e.done:r=e.apply([0.])
            if e.done:
                done[j]=1
                if r['status'] in ('ASSUMPTION_VALID','UNCHANGED'):
                    terminal.append(j);indices.append(self.indices[j]);amounts.append(e.current['transactionamt'])
                else:rewards[j]=assumption_reward(r['status'],cost=e.cost)
        probs=self.data.probabilities(indices,amounts) if terminal else []
        for j,p in zip(terminal,probs):rewards[j]=self.engines[j].resolve('FLAG' if p>1/6 else 'APPROVE');self.queries+=1
        for j in np.flatnonzero(done):
            e=self.engines[j];self.seen.add(int(self.indices[j]));self.episodes+=1
            self.finished.append(dict(reward=float(rewards[j]),identity=e.result['status']=='UNCHANGED',invalid=e.result['status'] not in ('UNCHANGED','ASSUMPTION_VALID'),cost=e.cost,perturbation=e.cost/.1*.05))
            self.indices[j]=self.rng.integers(len(self.data.sources));self.engines[j]=self.data.engine(self.indices[j])
        return self.observation(),rewards,done
    def state(self):
        return dict(indices=self.indices,engines=self.engines,rng=self.rng.bit_generator.state,queries=self.queries,episodes=self.episodes,seen=self.seen)
    def restore(self,state):
        self.indices=state['indices'];self.engines=state['engines'];self.rng.bit_generator.state=state['rng'];self.queries=state['queries'];self.episodes=state['episodes'];self.seen=state['seen']
