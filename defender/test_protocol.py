"""Small tests of chronology, reward ties, train-only mappings, and inference inputs."""
import unittest
import numpy as np
import pandas as pd
from .train import split_index,best_reward_threshold,decision_metrics,ece
from .inference import encode,fit_mappings,calibrate

class ProtocolTests(unittest.TestCase):
    def test_timestamp_ties(self):
        times=np.array([1,1,2,2,2,3,4,5])
        k,t=split_index(times,.5)
        self.assertEqual(k,5);self.assertEqual(t,2);self.assertLess(times[k-1],times[k])
    def test_reward_threshold_matches_bruteforce(self):
        y=np.array([0,0,1,1,0,1,0]);p=np.array([0.,.1,.1,.7,.7,1.,.4])
        threshold=best_reward_threshold(y,p)
        candidates=np.r_[1.,0.,[np.nextafter(v,0.) for v in np.unique(p) if v>0]]
        best=max(decision_metrics(y,p,float(t))['total_reward'] for t in candidates)
        self.assertEqual(decision_metrics(y,p,threshold)['total_reward'],best)
        self.assertEqual(decision_metrics(np.array([1,0]),np.array([1/6,1/6]),1/6)['flagged_count'],0)
    def test_encoder_does_not_learn_unknown(self):
        schema={'features':[{'name':'card1','postgres_type':'integer','role':'categorical'},
                            {'name':'device','postgres_type':'text','role':'categorical'},
                            {'name':'amount','postgres_type':'numeric','role':'numeric'}]}
        train=pd.DataFrame({'card1':[1,2,None],'device':['a','b',None],'amount':[1.,None,3.]})
        maps=fit_mappings(train,schema)
        test=pd.DataFrame({'card1':[1.,99.],'device':['a','unknown'],'amount':[None,10.]})
        x,diag=encode(test,schema,maps)
        self.assertEqual(x[0,0],0);self.assertEqual(x[1,0],-1);self.assertEqual(x[1,1],-1)
        self.assertTrue(np.isnan(x[0,2]));self.assertEqual(maps['card1'],['1.0','2.0'])
        with self.assertRaises(ValueError):encode(test.drop(columns='amount'),schema,maps)
    def test_ece_perfect(self):
        self.assertEqual(ece(np.array([0,1]),np.array([0.,1.]))[0],0.)

if __name__=='__main__':unittest.main()
