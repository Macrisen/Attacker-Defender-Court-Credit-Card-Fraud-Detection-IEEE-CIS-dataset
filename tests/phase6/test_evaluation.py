import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from scripts.final_evaluation import paired_interval,verify_freeze
class EvaluationTests(unittest.TestCase):
    def test_paired_blocks_known_difference(self):
        a=pd.DataFrame(dict(transactionid=range(20),transactiondt=[i*86400 for i in range(20)],utility=[2.]*20));b=a.copy();b.utility=1.
        r=paired_interval(a,b,'utility');self.assertEqual(r['mean_difference'],1.);self.assertEqual(r['ci_low'],1.);self.assertEqual(r['ci_high'],1.)
    def test_unpaired_ids_rejected(self):
        a=pd.DataFrame(dict(transactionid=[1],transactiondt=[0],utility=[1.]));b=a.copy();b.transactionid=2
        with self.assertRaises(ValueError):paired_interval(a,b,'utility')
    def test_failed_freeze_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json';p.write_text(json.dumps({'all_scientific_gates_passed':False}))
            with self.assertRaises(ValueError):verify_freeze(p)
