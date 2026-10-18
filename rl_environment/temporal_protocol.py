"""Exact timestamp partitions and provenance gate for independent policy training."""
import hashlib
import numpy as np
import pyarrow.parquet as pq


def build_protocol(parquet, saved_split, audit_fit_cutoff, max_delay_seconds=259200):
    # Only ID/time metadata is read for the final holdout. No holdout label/features.
    meta = pq.read_table(parquet, columns=['transactionid', 'transactiondt']).to_pandas().sort_values(
        ['transactiondt', 'transactionid']).reset_index(drop=True)
    if meta.isna().any().any() or not meta.transactionid.is_unique:
        raise ValueError('Invalid metadata')
    t = meta.transactiondt.to_numpy()
    hold_start = int(np.searchsorted(t, saved_split['prior_holdout_cutoff'], side='right'))
    if hold_start != saved_split['hold_start']:
        raise ValueError('Saved holdout boundary changed')
    # Fractions of the full labeled dataset: 45/10/15/10/20, with ties kept together.
    ends = [int(np.searchsorted(t, t[int(len(t)*f)-1], side='right')) for f in (.45, .55, .70)] + [hold_start, len(t)]
    names = ['defender_fit', 'defender_calibration', 'rl_development', 'rl_validation', 'final_holdout']
    windows, left = {}, 0
    for name, right in zip(names, ends):
        part = meta.iloc[left:right]
        if len(part) == 0:
            raise ValueError('Empty temporal partition')
        item = {'rows': len(part), 'min_time': int(part.transactiondt.min()), 'max_time': int(part.transactiondt.max()),
                'min_transaction_id': int(part.transactionid.min()), 'max_transaction_id': int(part.transactionid.max()),
                'first_transaction_id': int(part.transactionid.iloc[0]), 'last_transaction_id': int(part.transactionid.iloc[-1]),
                'ordered_id_sha256': hashlib.sha256(part.transactionid.to_numpy(dtype='<i8').tobytes()).hexdigest()}
        if name in ('rl_development', 'rl_validation'):
            cutoff = item['max_time']-max_delay_seconds
            item['decision_cutoff'] = cutoff
            item['eligible_origin_rows'] = int((part.transactiondt <= cutoff).sum())
            item['maturity_only_rows'] = int((part.transactiondt > cutoff).sum())
            if item['eligible_origin_rows'] == 0:
                raise ValueError('No room for full delay maturation')
        windows[name] = item
        left = right
    return {'status': 'IMPLEMENTED_AND_BOUNDARIES_VALIDATED; new model fitting gated by preprocessing provenance',
            'windows': windows, 'max_delay_seconds': max_delay_seconds,
            'historical_feature_selection_cutoff': audit_fit_cutoff,
            'legacy_holdout_previously_evaluated': True,
            'holdout_policy': 'No additional holdout labels or predictor values read in Phase 3.5; legacy baseline already evaluated this holdout, so it is reserved, not historically virgin.',
            'new_defender_status': 'NOT_FITTED: preprocessing provenance not independent of proposed RL windows',
            'baseline_reuse': 'BLOCKED: saved model/calibrator/selection exceed new fitting windows'}


def validate_model_periods(protocol, fit_end, calibration_end, feature_selection_end):
    w = protocol['windows']
    if fit_end > w['defender_fit']['max_time']:
        raise ValueError('Defender fit overlaps reserved later periods')
    if calibration_end > w['defender_calibration']['max_time'] or calibration_end >= w['rl_development']['min_time']:
        raise ValueError('Calibration must finish before RL development')
    if feature_selection_end > w['defender_fit']['max_time']:
        raise ValueError('Feature selection used future RL/calibration data; new fit alone cannot remove this leakage')
    ranges = [w[n] for n in ('defender_fit', 'defender_calibration', 'rl_development', 'rl_validation', 'final_holdout')]
    if any(a['max_time'] >= b['min_time'] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('Overlapping temporal windows')
    return True


def verify_alignment(cleaned_parquet, original_parquet):
    a = pq.read_table(cleaned_parquet, columns=['transactionid', 'transactiondt']).sort_by([('transactionid', 'ascending')])
    b = pq.read_table(original_parquet, columns=['transactionid', 'transactiondt']).sort_by([('transactionid', 'ascending')])
    if not a.equals(b):
        raise ValueError('Original/Cleaned IDs or timestamps differ')
    return True
