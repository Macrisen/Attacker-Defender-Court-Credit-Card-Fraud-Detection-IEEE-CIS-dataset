"""Shared reward implementation; identity reward never implies adversarial success."""
from .attack_constraint_engine import attacker_reward, attack_metrics
__all__=['attacker_reward','attack_metrics']


def assumption_reward(status,decision=None,cost=0.,invalid_penalty=-2.,identity_reward=0.):
    """Mode B terminal utility; identity is a zero-reward control, never ASR success."""
    if cost<0:raise ValueError('Negative perturbation cost')
    if status=='UNCHANGED':return float(identity_reward)
    if status in ('INVALID_ACTION','ASSUMPTION_REJECTED'):return float(invalid_penalty-cost)
    if status=='STRICT_BLOCKED':return 0.
    if status!='ASSUMPTION_VALID' or decision not in ('APPROVE','FLAG'):raise ValueError('No valid terminal decision')
    return (1. if decision=='APPROVE' else -1.)-cost
