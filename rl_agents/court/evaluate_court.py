"""Frozen recurrent Court evaluation; completed labels are metrics only."""
from .train_court import load_policy
from .trajectory_collector import collect

def evaluate(data,records,checkpoint,seed):return collect(data,records,load_policy(checkpoint),seed)
