import unittest,tempfile
from pathlib import Path
import numpy as np
import rl_agents  # initialize the verified LightGBM/OpenMP runtime first
import torch
from torch.distributions import Normal,TransformedDistribution,TanhTransform
from rl_agents.attacker.ppo_agent import SquashedActorCritic,gae,clipped_loss,optimize
from rl_agents.attacker.attacker_sampler import AttackData
from rl_agents.attacker.evaluate_ppo import load_policy
import yaml

class PPOTests(unittest.TestCase):
    def test_squashed_log_probability(self):
        torch.manual_seed(42);a=SquashedActorCritic();obs=torch.zeros(10,11);dist=a.distribution(obs);z=torch.linspace(-3,3,10).reshape(-1,1)
        expected=TransformedDistribution(dist,[TanhTransform(cache_size=1)]).log_prob(torch.tanh(z)).sum(-1)
        torch.testing.assert_close(a.log_prob(dist,z),expected,atol=1e-4,rtol=1e-4)
        self.assertTrue((a.sample(obs)[0].abs()<=1).all())
    def test_gae_terminal(self):
        r=np.array([[1.],[2.]],np.float32);v=np.zeros_like(r);done=np.array([[0.],[1.]],np.float32)
        adv,ret=gae(r,v,done,np.array([999.]),.5,1.)
        np.testing.assert_allclose(ret[:,0],[2,2])
    def test_clipping(self):
        loss=clipped_loss(torch.log(torch.tensor([2.,.5])),torch.zeros(2),torch.tensor([1.,-1.]),.2)
        self.assertAlmostEqual(loss.item(),-.2,places=6)
    def test_real_gradient_and_checkpoint(self):
        torch.manual_seed(42);torch.set_num_threads(4);a=SquashedActorCritic();opt=torch.optim.Adam(a.parameters(),lr=.0003);o=torch.randn(32,11)
        with torch.no_grad():_,z,lp,_=a.sample(o)
        before=torch.cat([p.flatten().detach().clone() for p in a.parameters()])
        c=yaml.safe_load(Path('configs/attacker_ppo.yaml').read_text())['attacker'];c=dict(c,epochs=1,minibatch_size=32)
        optimize(a,opt,[o.numpy(),z.numpy(),lp.numpy(),np.ones(32),np.linspace(-1,1,32)],c,np.random.default_rng(42))
        self.assertFalse(torch.equal(before,torch.cat([p.flatten().detach() for p in a.parameters()])))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.pt';torch.save({'policy':a.state_dict()},p);b=load_policy(p)
            torch.testing.assert_close(a.actor(o),b.actor(o))
    def test_split_gate(self):
        with self.assertRaises(ValueError):AttackData('original','rl_validation',training=True)
        with self.assertRaises(ValueError):AttackData('original','final_holdout')
    def test_inflight_sampler_checkpoint_resume(self):
        from rl_agents.attacker.attacker_sampler import TrainingVector
        data=AttackData('original','rl_development',training=True)
        original=TrainingVector(data,42,n=4)
        original.step(np.full((4,1),.25))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sampler.pt';torch.save(original.state(),path)
            resumed=TrainingVector(data,123,n=4)
            resumed.restore(torch.load(path,weights_only=False))
        for action in (.4,0.,-.3,0.):
            left=original.step(np.full((4,1),action))
            right=resumed.step(np.full((4,1),action))
            for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(original.indices,resumed.indices)
            self.assertEqual(original.queries,resumed.queries)
            self.assertEqual(original.episodes,resumed.episodes)
    def test_frozen_optimized_inference_matches_schema(self):
        d=AttackData('original','rl_development',training=True)
        import pyarrow.dataset as ds
        records=ds.dataset('data/processed/train_original.parquet').to_table(filter=ds.field('transactionid').isin(d.ids[:4])).to_pylist()
        records.sort(key=lambda r:(r['transactiondt'],r['transactionid']))
        amounts=[r['transactionamt']*__import__('decimal').Decimal('1.01') for r in records]
        expected=[]
        for r,amount in zip(records,amounts):
            f={n:r[n] for n in d.defender.feature_names};f['transactionamt']=amount;expected.append(f)
        np.testing.assert_allclose(d.probabilities(range(4),amounts),d.defender.predict_many(expected),rtol=0,atol=0)
