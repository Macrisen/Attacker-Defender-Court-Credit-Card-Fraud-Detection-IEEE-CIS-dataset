import unittest
from types import SimpleNamespace
from unittest.mock import patch
from continual_learning.simulation import AdaptiveSimulator
from rl_environment.fraud_simulator import FraudSimulator


class AdaptiveSummaryTests(unittest.TestCase):
    def test_only_arrived_modified_transactions_are_counted(self):
        sim=object.__new__(AdaptiveSimulator)
        sim.index=2
        sim._records=[SimpleNamespace(transaction_id=i) for i in range(3)]
        sim.attack_adapter=SimpleNamespace(provenance={0:{'modified':False},1:{'modified':True},2:{'modified':True}})
        with patch.object(FraudSimulator,'summary',return_value={'transactions_processed':2}):
            result=sim.summary()
        self.assertEqual(result['modified_attack_examples'],1)
        self.assertEqual(result['identity_transactions'],1)
        self.assertEqual(result['transactions_processed'],2)
