"""Independently verify primary saved predictions and report metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix


def verify(path=Path('reports/lightgbm_baseline')):
    data = pd.read_csv(path/'predictions_holdout.csv', float_precision='round_trip')
    metrics = pd.read_csv(path/'model_metrics.csv', float_precision='round_trip')
    config = json.loads((path/'artifacts/defender_config.json').read_text())
    binary = np.load(path/'plot_data.npz')
    assert data.transactionid.is_unique
    assert np.array_equal(data.isfraud, binary['labels'])
    checked = 0
    for method, column in [('raw', 'raw_probability'), (config['calibration_method'], 'calibrated_probability')]:
        probabilities = data[column].to_numpy()
        np.testing.assert_array_equal(probabilities, binary[column])
        rows = metrics[(metrics.experiment == 'B_clean') & (metrics.probability_method == method)]
        for _, row in rows.iterrows():
            flags = probabilities > config['thresholds'][row.threshold_policy]
            tn, fp, fn, tp = confusion_matrix(data.isfraud, flags).ravel()
            assert (tn, fp, fn, tp) == (row.tn, row.fp, row.fn, row.tp)
            assert np.isclose(average_precision_score(data.isfraud, probabilities), row.average_precision)
            assert np.isclose(roc_auc_score(data.isfraud, probabilities), row.roc_auc)
            assert 10*tp - 20*fn - 5*fp + tn == row.total_reward
            if method == config['calibration_method']:
                np.testing.assert_array_equal(data['decision_'+row.threshold_policy].eq('FLAG'), flags)
            checked += 1
    required = ['dataset_summary.csv', 'model_metrics.csv', 'threshold_comparison.csv',
                'feature_importance.csv', 'temporal_performance.csv', 'predictions_holdout.csv',
                'lightgbm_baseline_report.html']
    assert all((path/name).stat().st_size > 0 for name in required)
    result = {'status': 'PASS', 'prediction_rows': len(data), 'metric_rows_recomputed': checked,
              'checks': ['unique IDs', 'lossless probability CSV round trip', 'saved labels',
                         'AP and ROC AUC', 'confusion matrices', 'realized rewards',
                         'all primary decisions', 'required report files']}
    (path/'output_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
