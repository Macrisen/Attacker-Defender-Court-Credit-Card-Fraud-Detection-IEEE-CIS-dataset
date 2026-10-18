"""Causal observation contract with known horizon. This remains a POMDP."""
import numpy as np
from gymnasium import spaces
from .observations import CourtObservations
class CourtObservationV2(CourtObservations):
    def __init__(self,decision_end):
        super().__init__()
        self.decision_end=float(decision_end)
        self.names+=['decision_time_remaining_scaled']
        self.space=spaces.Box(-1,1,shape=(11,),dtype=np.float32)
    def encode(self,features,probability,decision,budget,pending,memory,now,defender_updated_at):
        base=CourtObservations().encode(features,probability,decision,budget,pending,memory,now,defender_updated_at)
        remaining=max(0,self.decision_end-now)
        return np.append(base,remaining/(86400+remaining)).astype(np.float32)
