"""Resumable CPU PPO; no validation information enters gradient updates."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .ppo_agent import SquashedActorCritic,gae,optimize
from .attacker_sampler import AttackData,TrainingVector
from .evaluate_ppo import evaluate,aggregate
ROOT=Path(__file__).resolve().parents[2]

def train(version,seed,config,max_steps=None,resume=False,bandit=False):
    c=config['attacker'];budget=max_steps or (config['bandit']['training_queries'] if bandit else c['max_steps'])
    n=c['vector_envs'];rollout=c['rollout_steps'];budget=budget//rollout*rollout
    if budget<rollout:raise ValueError('Budget must allow at least one complete PPO rollout')
    torch.set_num_threads(config['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    kind='bandit' if bandit else 'ppo';output=ROOT/'models/rl/attacker'/version/f'{kind}_seed{seed}';output.mkdir(parents=True,exist_ok=True)
    finished=output/'complete.json';latest=output/'latest.pt'
    if finished.exists():
        if resume:return json.loads(finished.read_text())
        raise FileExistsError('Completed run exists; use --resume to verify/skip')
    if latest.exists() and not resume:raise FileExistsError('Interrupted run exists; use --resume')
    data=AttackData(version,'rl_development',training=True);vector=TrainingVector(data,seed,n,bandit)
    agent=SquashedActorCritic();optimizer=torch.optim.Adam(agent.parameters(),lr=c['learning_rate'])
    steps=0;logs=[];checkpoints=[];elapsed_prior=0.;updates=0
    if latest.exists():
        cp=torch.load(latest,weights_only=False,map_location='cpu');assert cp['budget']==budget and cp['kind']==kind
        agent.load_state_dict(cp['policy']);optimizer.load_state_dict(cp['optimizer']);vector.restore(cp['vector']);torch.set_rng_state(cp['torch_rng']);rng.bit_generator.state=cp['rng'];steps=cp['steps'];logs=cp['logs'];checkpoints=cp['checkpoints'];elapsed_prior=cp['elapsed'];updates=cp['gradient_updates']
    start=time.monotonic();obs=vector.observation()
    while steps<budget:
        stores=[[] for _ in range(6)];vector.finished=[]
        for _ in range(rollout//n):
            with torch.no_grad():action,z,lp,value=agent.sample(torch.tensor(obs))
            next_obs,rewards,done=vector.step(action.numpy())
            for store,item in zip(stores,[obs,z.numpy(),lp.numpy(),value.numpy(),rewards,done]):store.append(item)
            obs=next_obs
        ob,z,lp,values,rewards,done=map(np.asarray,stores)
        with torch.no_grad():last=agent.critic(torch.tensor(obs)).squeeze(-1).numpy()
        adv,returns=gae(rewards,values,done,last,c['gamma'],c['gae_lambda'])
        metrics=optimize(agent,optimizer,[ob.reshape(-1,11),z.reshape(-1,1),lp.reshape(-1),returns.reshape(-1),adv.reshape(-1)],c,rng)
        steps+=rollout;updates+=metrics['gradient_updates'];episode=pd.DataFrame(vector.finished)
        metrics.update(version=version,seed=seed,policy=kind,agent_interactions=steps,episodes=vector.episodes,defender_queries=vector.queries,unique_training_sources=len(vector.seen),total_gradient_updates=updates,elapsed_seconds=elapsed_prior+time.monotonic()-start,mean_episode_reward=float(episode.reward.mean()),identity_rate=float(episode.identity.mean()),invalid_rate=float(episode.invalid.mean()),mean_cost=float(episode.cost.mean()),mean_perturbation=float(episode.perturbation.mean()))
        logs.append(metrics)
        periodic=(steps//rollout)%c['checkpoint_every_rollouts']==0 or steps==budget
        if periodic:
            checkpoint=output/f'checkpoint_{steps}.pt';torch.save(dict(policy=agent.state_dict(),steps=steps,seed=seed,version=version,kind=kind,training_window='rl_development'),checkpoint);checkpoints.append(str(checkpoint))
            print(version,kind,seed,steps,'queries',vector.queries,'seconds',round(metrics['elapsed_seconds'],1),flush=True)
        # Resume includes optimizer, RNGs and in-flight short episodes at a rollout boundary.
        state=dict(policy=agent.state_dict(),optimizer=optimizer.state_dict(),vector=vector.state(),torch_rng=torch.get_rng_state(),rng=rng.bit_generator.state,steps=steps,logs=logs,checkpoints=checkpoints,budget=budget,kind=kind,elapsed=metrics['elapsed_seconds'],gradient_updates=updates)
        temp=output/'latest.tmp';torch.save(state,temp);temp.replace(latest)
        pd.DataFrame(logs).to_csv(output/'training_metrics.csv',index=False)
    # All candidate policies are fixed before validation selection; never update on these results.
    validation=AttackData(version,'rl_validation');rank=[]
    from .evaluate_ppo import load_policy
    for path in checkpoints:
        evaluation=evaluate(validation,load_policy(path),kind,seed,bandit);score=aggregate(evaluation)
        evaluation.to_csv(output/(Path(path).stem+'_validation.csv'),index=False)
        rank.append(dict(checkpoint=path,**score))
    best=max(rank,key=lambda r:(r['selection_flip_fraction'],-r['mean_cost'],-int(Path(r['checkpoint']).stem.split('_')[-1])))
    result=dict(status='PASS',version=version,seed=seed,policy=kind,training_interactions=steps,environment_action_calls=steps+(vector.episodes if bandit else 0),training_queries=vector.queries,training_episodes=vector.episodes,unique_training_sources=len(vector.seen),gradient_updates=updates,best_checkpoint=best['checkpoint'],selection=best,checkpoints=rank,elapsed_seconds=elapsed_prior+time.monotonic()-start,training_window='rl_development',selection_window='rl_validation',budget=budget)
    finished.write_text(json.dumps(result,indent=2));return result
