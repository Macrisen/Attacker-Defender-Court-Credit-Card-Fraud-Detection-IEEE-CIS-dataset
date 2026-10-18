import unittest
import rl_agents
import torch,numpy as np
from rl_agents.court.recurrent_policy import CourtPolicy,update_mature
from rl_agents.court.trajectory_collector import KeyedVerification
from scripts.phase3_6_checks import credit_fixture
class RecurrentTests(unittest.TestCase):
    def test_mask_and_state_reset(self):
        torch.manual_seed(42);p=CourtPolicy();o=torch.zeros(1,3,11)
        d,v,h=p(o,p.initial());self.assertTrue((d.probs[...,1]==0).all())
        d2,v2,h2=p(o,p.initial());torch.testing.assert_close(h,h2)
        o[...,4]=1;d,_,_=p(o);self.assertTrue((d.probs[...,1]>0).all())
    def test_maturity_and_policy_version_gate(self):
        p=CourtPolicy();opt=torch.optim.Adam(p.parameters())
        with self.assertRaises(ValueError):update_mature(p,opt,dict(mature=False,policy_version=0),0)
        with self.assertRaises(ValueError):update_mature(p,opt,dict(mature=True,policy_version=1),0)
    def test_recurrent_actual_gradient(self):
        torch.manual_seed(42);torch.set_num_threads(4);p=CourtPolicy();opt=torch.optim.Adam(p.parameters(),lr=.001)
        obs=np.zeros((8,11),np.float32);obs[:,4]=1
        with torch.no_grad():d,v,_=p(torch.tensor(obs).unsqueeze(0));actions=torch.ones(8,dtype=torch.int64);lp=d.log_prob(actions.unsqueeze(0)).squeeze(0)
        t=dict(mature=True,policy_version=0,observations=obs,actions=actions.numpy(),log_probs=lp.numpy(),values=v.squeeze(0).numpy(),returns=np.arange(8)*10.)
        before=p.actor.weight.detach().clone();update_mature(p,opt,t,0,epochs=1,sequence_length=4);self.assertFalse(torch.equal(before,p.actor.weight))
    def test_existing_manual_mature_credit(self):
        rows,_,_,_=credit_fixture();self.assertTrue(all(r['status']=='PASS' for r in rows))
    def test_keyed_delay_independent_request_order(self):
        def delays(order):
            v=KeyedVerification(seed=42)
            for i in order:v.request(dict(transaction_id=i,transaction_time=0,hidden_label=1),0)
            return {r['transaction_id']:r['completion_time'] for r in v.release(259200)}
        self.assertEqual(delays([1,2]),delays([2,1]))
