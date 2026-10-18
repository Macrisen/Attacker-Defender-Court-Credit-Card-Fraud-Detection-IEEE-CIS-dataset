"""Fit a NEW earlier-window Defender only with independently fitted feature provenance.

Current cleaned provenance fails the gate deliberately. No override silently
converts historically selected features into a leakage-safe feature set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from defender.inference import fit_mappings, encode, calibrate
from rl_environment.temporal_protocol import validate_model_periods


def fit_new_defender(parquet, schema_file, protocol_file, provenance_file, output, max_rounds=2000):
    protocol = json.loads(Path(protocol_file).read_text())
    provenance = json.loads(Path(provenance_file).read_text())
    schema = json.loads(Path(schema_file).read_text())
    w = protocol['windows']
    validate_model_periods(protocol, w['defender_fit']['max_time'], w['defender_calibration']['max_time'], provenance['fit_cutoff'])
    names = [f['name'] for f in schema['features']]
    if set(names) & {'transactionid', 'transactiondt', 'isfraud'}:
        raise ValueError('Metadata is not a predictor')
    if set(names) != set(provenance['selected_predictors']):
        raise ValueError('Reviewed fit-only feature provenance does not match schema')
    if hashlib.sha256(Path(parquet).read_bytes()).hexdigest() != provenance['parquet_sha256']:
        raise ValueError('Provenance points to a different dataset')
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT/'reports/lightgbm_baseline'):
        raise FileExistsError('New artifact directory required; baseline overwrite prohibited')
    # No RL validation or holdout labels/features are loaded.
    frame = ds.dataset(parquet, format='parquet').to_table(
        filter=ds.field('transactiondt') <= w['defender_calibration']['max_time'],
        columns=names+['transactionid', 'transactiondt', 'isfraud']).to_pandas().sort_values(['transactiondt', 'transactionid'])
    fit = frame[frame.transactiondt <= w['defender_fit']['max_time']]
    cal = frame[frame.transactiondt > w['defender_fit']['max_time']]
    if len(fit) != w['defender_fit']['rows'] or len(cal) != w['defender_calibration']['rows']:
        raise ValueError('Partition counts differ')
    if not frame.isfraud.isin([0, 1]).all():
        raise ValueError('Invalid labels')
    inner_cut = fit.transactiondt.iloc[int(len(fit)*.85)-1]
    inner, validation = fit[fit.transactiondt <= inner_cut], fit[fit.transactiondt > inner_cut]
    mapping = fit_mappings(inner, schema)
    x, _ = encode(inner, schema, mapping)
    xv, _ = encode(validation, schema, mapping)
    cats = [i for i, f in enumerate(schema['features']) if f['role'] == 'categorical']
    params = {'objective': 'binary', 'metric': 'average_precision', 'learning_rate': .03, 'num_leaves': 31,
              'seed': 42, 'num_threads': 4, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
    train = lgb.Dataset(x, label=inner.isfraud, feature_name=names, categorical_feature=cats)
    valid = lgb.Dataset(xv, label=validation.isfraud, reference=train)
    chosen = lgb.train(params, train, num_boost_round=max_rounds, valid_sets=[valid], callbacks=[lgb.early_stopping(100)])
    iterations = chosen.best_iteration
    mapping = fit_mappings(fit, schema)
    xf, _ = encode(fit, schema, mapping)
    xc, _ = encode(cal, schema, mapping)
    model = lgb.train(params, lgb.Dataset(xf, label=fit.isfraud, feature_name=names, categorical_feature=cats), num_boost_round=iterations)
    probabilities = model.predict(xc, num_threads=4)
    cal_cut = cal.transactiondt.iloc[len(cal)//2-1]
    fit_mask = (cal.transactiondt <= cal_cut).to_numpy()
    labels = cal.isfraud.to_numpy()
    def fit_cal(method, p, y):
        if method == 'raw':
            return {'method': method}
        if method == 'isotonic':
            return {'method': method, 'model': IsotonicRegression(out_of_bounds='clip').fit(p, y)}
        p = np.clip(p, 1e-12, 1-1e-12)
        return {'method': method, 'model': LogisticRegression(C=1e6).fit(np.log(p/(1-p)).reshape(-1, 1), y)}
    candidates = [fit_cal(m, probabilities[fit_mask], labels[fit_mask]) for m in ('raw', 'sigmoid', 'isotonic')]
    best = min(candidates, key=lambda a: log_loss(labels[~fit_mask], calibrate(probabilities[~fit_mask], a), labels=[0, 1]))
    final_cal = fit_cal(best['method'], probabilities, labels)
    output.mkdir(parents=True)
    model.save_model(str(output/'lightgbm_model.txt'))
    joblib.dump(final_cal, output/'calibrator.joblib')
    documents = {'feature_schema.json': schema, 'categorical_mappings.json': mapping,
                 'defender_config.json': {'calibration_method': best['method'], 'thresholds': {'theoretical_reward_1_6': 1/6}, 'num_threads': 4},
                 'training_config.json': {'parameters': params, 'selected_iterations': iterations, 'protocol': protocol, 'provenance': provenance}}
    for name, value in documents.items():
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    return str(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', required=True)
    p.add_argument('--schema', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--provenance', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(fit_new_defender(a.parquet, a.schema, a.protocol, a.provenance, a.output))
