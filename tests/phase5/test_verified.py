import tempfile,unittest
from pathlib import Path
from continual_learning.verified_replay import VerifiedReplay
from continual_learning.model_registry import ModelRegistry
class VerifiedTests(unittest.TestCase):
    def row(self,i,time=1):return dict(transaction_id=i,verified_label=i%2,completion_time=time,features={'transactionamt':100},source='court')
    def test_future_and_hidden_rejected(self):
        r=VerifiedReplay()
        with self.assertRaises(ValueError):r.insert(self.row(1,10),9)
        with self.assertRaises(ValueError):r.insert(dict(self.row(1),hidden_label=1),2)
        with self.assertRaises(ValueError):r.insert({'transaction_id':1},2)
    def test_capacity_order_unique_and_exclusion(self):
        r=VerifiedReplay(2,excluded_ids=[0])
        for i in range(4):r.insert(self.row(i,i+1),10)
        self.assertEqual([v['transaction_id'] for v in r.available(10)],[2,3]);self.assertFalse(r.insert(self.row(3),10))
        with self.assertRaises(ValueError):r.available(1)
    def test_model_versions_and_deployment(self):
        with tempfile.TemporaryDirectory() as d:
            r=ModelRegistry(d);p=r.allocate(1)
            with self.assertRaises(FileExistsError):r.allocate(1)
            with self.assertRaises(ValueError):r.register(p,1,2)
            r.register(p,2,1);self.assertEqual(len(ModelRegistry(d).entries),1)
