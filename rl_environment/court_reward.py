"""Research utility coefficients, not monetary estimates."""
from dataclasses import dataclass
@dataclass(frozen=True)
class CourtReward:
    investigation_cost: float=1.
    missed_fraud_reward: float=10.
    flagged_fraud_reward: float=3.
    def bonus(self,verified_label,defender_decision):
        if verified_label not in (0,1) or defender_decision not in ('APPROVE','FLAG'):raise ValueError('Invalid verified outcome')
        return (self.missed_fraud_reward if defender_decision=='APPROVE' else self.flagged_fraud_reward) if verified_label else 0.
