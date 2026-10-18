"""Actual recurrent PPO on complete, matured fixed-behavior episodes."""
import gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .recurrent_policy import CourtPolicy,update_mature
from .trajectory_collector import StreamData,collect
ROOT=Path(__file__).resolve().parents[2]

def load_policy(path):
    m=CourtPolicy();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy']);m.eval();return m

def train(version,seed,config,resume=False):
    c=config['court'];episodes=c['max_steps']//c['episode_decisions']
    if episodes<1:raise ValueError('Court budget must cover one declared complete episode')
    out=ROOT/'models/rl/court'/version/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        if resume:return json.loads((out/'complete.json').read_text())
        raise FileExistsError('Completed Court run exists')
    torch.set_num_threads(config['threads']);torch.manual_seed(seed)
    data=StreamData(version,'rl_development');policy=CourtPolicy();opt=torch.optim.Adam(policy.parameters(),lr=c['learning_rate']);start_episode=0;logs=[];checkpoints=[];elapsed=0.
    latest=out/'latest.pt'
    if latest.exists():
        if not resume:raise FileExistsError('Use --resume for Court checkpoint')
        cp=torch.load(latest,map_location='cpu',weights_only=False);assert cp['episodes']==episodes
        policy.load_state_dict(cp['policy']);opt.load_state_dict(cp['optimizer']);start_episode=cp['episode'];logs=cp['logs'];checkpoints=cp['checkpoints'];elapsed=cp['elapsed'];torch.set_rng_state(cp['torch_rng'])
    start=time.monotonic();block=c['episode_decisions'];available=len(data.table)//block
    for episode in range(start_episode,episodes):
        records=data.records((episode%available)*block,block)
        trajectory,metrics,sim,memory=collect(data,records,policy,seed+episode,policy_version=episode)
        assert trajectory['mature'] and len(trajectory['returns'])==block
        updates=update_mature(policy,opt,trajectory,episode,c['epochs'],c['sequence_length'])
        row=dict(version=version,seed=seed,episode=episode+1,decision_interactions=(episode+1)*block,gradient_updates=(episode+1)*c['epochs'],**metrics,**{k:float(np.mean([u[k] for u in updates])) for k in updates[0]},elapsed_seconds=elapsed+time.monotonic()-start)
        logs.append(row)
        if (episode+1)%c['checkpoint_every_episodes']==0 or episode+1==episodes:
            path=out/f'checkpoint_{episode+1}.pt';torch.save(dict(policy=policy.state_dict(),episode=episode+1,version=version,seed=seed),path);checkpoints.append(str(path))
        cp=dict(policy=policy.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),episode=episode+1,episodes=episodes,logs=logs,checkpoints=checkpoints,elapsed=row['elapsed_seconds']);tmp=out/'latest.tmp';torch.save(cp,tmp);tmp.replace(latest)
        pd.DataFrame(logs).to_csv(out/'training_metrics.csv',index=False)
        print('Court',version,seed,'episode',episode+1,'reward',metrics['court_policy_reward'],'seconds',round(row['elapsed_seconds'],1),flush=True)
        del trajectory,metrics,sim,memory,records;gc.collect()
    del data;gc.collect();validation=StreamData(version,'rl_validation');records=validation.records();rank=[]
    for path in checkpoints:
        _,metrics,sim,memory=collect(validation,records,load_policy(path),seed)
        rank.append(dict(checkpoint=path,**metrics));del sim,memory;gc.collect()
    best=max(rank,key=lambda r:(r['court_policy_reward']/r['transactions_processed'],-int(Path(r['checkpoint']).stem.split('_')[-1])))
    result=dict(status='PASS',version=version,seed=seed,algorithm='recurrent clipped PPO; mature time-aware MC advantages; GRU32; TBPTT128',training_interactions=episodes*block,gradient_updates=episodes*c['epochs'],best_checkpoint=best['checkpoint'],selection=best,checkpoints=rank,elapsed_seconds=elapsed+time.monotonic()-start)
    (out/'complete.json').write_text(json.dumps(result,indent=2));return result
