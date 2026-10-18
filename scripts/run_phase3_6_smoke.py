"""Independent 10k chronological development identity replays, no policy fitting."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from rl_environment.defender import FrozenDefender
from rl_environment.transaction_stream import TransactionStream,InternalTransaction
from rl_environment.fraud_simulator import FraudSimulator
from rl_environment.court_credit_adapter import FiniteCourtCollector
from scripts.phase3_6_checks import credit_fixture,attack_checks

def main():
    out=ROOT/'reports/phase3_6'
    protocol=json.loads((ROOT/'reports/phase3_5/temporal_split_protocol.json').read_text());w=protocol['windows']['rl_development']
    config=yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text())
    summaries=[];ids=None
    for kind in ('original','cleaned'):
        defender=FrozenDefender(ROOT/'models/temporal_defender'/kind)
        # Choose IDs via metadata, then read only the prescribed decision prefix.
        source=ds.dataset(ROOT/'data/processed/train_original.parquet')
        filt=(ds.field('transactiondt')>=w['min_time'])&(ds.field('transactiondt')<=w['decision_cutoff'])
        meta=source.to_table(columns=['transactionid','transactiondt'],filter=filt).sort_by([('transactiondt','ascending'),('transactionid','ascending')]).slice(0,10000)
        if ids is None:ids=meta['transactionid']
        else:assert ids.equals(meta['transactionid'])
        table=source.to_table(columns=defender.feature_names+['transactionid','transactiondt','isfraud'],filter=filt & ds.field('transactionid').isin(ids)).sort_by([('transactiondt','ascending'),('transactionid','ascending')])
        records=[InternalTransaction(int(r['transactionid']),int(r['transactiondt']),{n:r[n] for n in defender.feature_names},int(r['isfraud'])) for r in table.to_pylist()]
        stream=TransactionStream.synthetic(records) # validated internal-record constructor, real data source logged below
        sim=FraudSimulator(stream,defender,config)
        sim.defender_deployed_at=protocol['windows']['defender_calibration']['max_time']
        collector=FiniteCourtCollector(sim,w['max_time'])
        rng=np.random.default_rng(42)
        while not sim.done:
            obs=sim.observation()
            collector.step(int(obs[3]>0 or rng.random()<.005))
        horizon=sim.summary()
        replay=collector.finish()
        assert len(replay)==10000 and all(np.isfinite(r['target']) for r in replay)
        led=collector.ledger.records()
        pd.DataFrame([{k:v for k,v in r.items() if k not in ('observation','next_observation')} for r in led]).to_csv(out/f'{kind}_court_ledger.csv',index=False)
        pd.DataFrame([{k:v for k,v in r.items() if k!='history'} for r in replay]).to_csv(out/f'{kind}_mc_targets.csv',index=False)
        pd.DataFrame(sim.actions).to_csv(out/f'{kind}_audit_action_log.csv',index=False)
        summaries.append(dict(dataset=kind,source='train_original.parquet; chronological real RL development prefix',fit_overlap=False,first_time=records[0].transaction_time,last_decision_time=records[-1].transaction_time,label_maturity_end=sim.now,training_window_end=w['max_time'],replay_records=len(replay),horizon=horizon,mature=sim.summary(),policy='fixed FLAG-or-seeded-.005; no optimization',observation_dimension=11))
        print(kind,'smoke complete',flush=True)
    (out/'development_smoke.json').write_text(json.dumps(summaries,indent=2))
    rows,update,_,_=credit_fixture();pd.DataFrame(rows).to_csv(out/'court_reward_credit_validation.csv',index=False)
    (out/'synthetic_learner_update.json').write_text(json.dumps(update,indent=2))
    rows,metrics=attack_checks();pd.DataFrame(rows).to_csv(out/'attacker_action_reward_validation.csv',index=False)
    (out/'synthetic_attack_metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
