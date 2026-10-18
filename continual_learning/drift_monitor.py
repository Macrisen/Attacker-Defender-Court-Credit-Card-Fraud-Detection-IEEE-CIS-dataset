"""Past-window diagnostics; not a label-driven tuning or automatic drift claim."""
import numpy as np
class DriftMonitor:
    def __init__(self):self.history=[]
    def observe(self,amounts,probabilities,time):
        record=dict(time=time,rows=len(amounts),mean_log_amount=float(np.log1p(np.asarray(amounts,float)).mean()),mean_probability=float(np.mean(probabilities)))
        record['mean_probability_change']=record['mean_probability']-self.history[-1]['mean_probability'] if self.history else 0.
        self.history.append(record);return record
