"""Causal recurrent policy with identical budget masks during acting and updates."""
import torch
from torch import nn
from torch.distributions import Categorical

class CourtPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(11,32,batch_first=True);self.actor=nn.Linear(32,2);self.critic=nn.Linear(32,1)
        nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
        # Prior matches scarce 180/10000 budget; learnable and frozen in protocol.
        with torch.no_grad():self.actor.bias[1]=-4.
    def initial(self,batch=1):return torch.zeros(1,batch,32)
    def forward(self,obs,hidden=None):
        x,h=self.gru(obs,hidden);logits=self.actor(x)
        mask=obs[...,4]<=0
        logits=logits.clone();logits[...,1]=logits[...,1].masked_fill(mask,-1e9)
        return Categorical(logits=logits),self.critic(x).squeeze(-1),h
    def act(self,obs,hidden):
        x=torch.as_tensor(obs,dtype=torch.float32).reshape(1,1,11)
        with torch.no_grad():dist,value,h=self(x,hidden);action=dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item()),h

def update_mature(policy,optimizer,trajectory,behavior_version,epochs=4,sequence_length=128):
    if not trajectory['mature'] or trajectory['policy_version']!=behavior_version:raise ValueError('Unmatured or stale policy trajectory')
    obs=torch.tensor(trajectory['observations'],dtype=torch.float32).unsqueeze(0)
    actions=torch.tensor(trajectory['actions'],dtype=torch.int64);oldlp=torch.tensor(trajectory['log_probs'],dtype=torch.float32)
    target=torch.tensor(trajectory['returns'],dtype=torch.float32)/100.
    advantages=target-torch.tensor(trajectory['values'],dtype=torch.float32);advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    results=[];n=len(actions)
    for _ in range(epochs):
        h=policy.initial();optimizer.zero_grad();total_actor=total_value=entropy_sum=kl_sum=0.
        for start in range(0,n,sequence_length):
            end=min(n,start+sequence_length);dist,v,h=policy(obs[:,start:end],h);lp=dist.log_prob(actions[start:end].unsqueeze(0)).squeeze(0)
            ratio=(lp-oldlp[start:end]).exp();adv=advantages[start:end]
            actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean();value=((v.squeeze(0)-target[start:end])**2).mean();entropy=dist.entropy().mean()
            loss=(actor+.5*value-.01*entropy)*(end-start)/n;loss.backward();h=h.detach()
            total_actor+=float(actor.detach())*(end-start)/n;total_value+=float(value.detach())*(end-start)/n;entropy_sum+=float(entropy.detach())*(end-start)/n
            kl_sum+=float(((ratio-1)-(lp-oldlp[start:end])).mean().detach())*(end-start)/n
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step();results.append(dict(actor_loss=total_actor,value_loss=total_value,entropy=entropy_sum,approximate_kl=kl_sum,gradient_norm=float(norm)))
    return results
