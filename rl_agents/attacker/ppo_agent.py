"""Clipped PPO with an exact tanh-Gaussian change of variables."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

class SquashedActorCritic(nn.Module):
    def __init__(self,dimension=11):
        super().__init__()
        def mlp(output):return nn.Sequential(nn.Linear(dimension,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,output))
        self.actor=mlp(1);self.critic=mlp(1);self.log_std=nn.Parameter(torch.zeros(1))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,math.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
    def distribution(self,obs):return Normal(self.actor(obs),self.log_std.clamp(-5,2).exp())
    @staticmethod
    def log_prob(dist,z):
        jacobian=2*(math.log(2)-z-F.softplus(-2*z))
        return (dist.log_prob(z)-jacobian).sum(-1)
    def sample(self,obs):
        dist=self.distribution(obs);z=dist.sample()
        return torch.tanh(z),z,self.log_prob(dist,z),self.critic(obs).squeeze(-1)
    def evaluate(self,obs,z):
        dist=self.distribution(obs);sample=dist.rsample()
        return self.log_prob(dist,z),self.critic(obs).squeeze(-1),-self.log_prob(dist,sample)

def gae(rewards,values,dones,last_values,gamma=.99,lam=.95):
    advantages=np.zeros_like(rewards);last=np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        nxt=last_values if t==len(rewards)-1 else values[t+1]
        alive=1-dones[t]
        delta=rewards[t]+gamma*nxt*alive-values[t]
        last=delta+gamma*lam*alive*last;advantages[t]=last
    return advantages,advantages+values

def clipped_loss(new_logp,old_logp,advantages,clip=.2):
    ratio=(new_logp-old_logp).exp()
    return -torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages).mean()

def optimize(agent,optimizer,batch,config,rng):
    obs,z,oldlp,returns,advantages=[torch.as_tensor(x,dtype=torch.float32) for x in batch]
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    rows=[]
    for _ in range(config['epochs']):
        indices=rng.permutation(len(obs))
        for start in range(0,len(obs),config['minibatch_size']):
            idx=indices[start:start+config['minibatch_size']]
            lp,v,entropy=agent.evaluate(obs[idx],z[idx]);delta=lp-oldlp[idx];ratio=delta.exp()
            actor=clipped_loss(lp,oldlp[idx],advantages[idx],config['clip']);value=((v-returns[idx])**2).mean()
            loss=actor+config['value_coefficient']*value-config['entropy_coefficient']*entropy.mean()
            optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(agent.parameters(),config['max_grad_norm']);optimizer.step()
            rows.append([actor.item(),value.item(),entropy.mean().item(),((ratio-1)-delta).mean().item(),((ratio-1).abs()>config['clip']).float().mean().item(),float(norm)])
    result=dict(zip(['actor_loss','value_loss','entropy','approximate_kl','clip_fraction','gradient_norm'],np.mean(rows,axis=0)))
    result['gradient_updates']=len(rows);return result
