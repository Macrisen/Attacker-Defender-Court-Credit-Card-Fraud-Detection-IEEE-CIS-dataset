"""Frozen adapter to the existing calibrated LightGBM inference implementation."""
import hashlib
from pathlib import Path
import numpy as np
from defender.inference import Defender as BaselineDefender
from .transaction_stream import METADATA


class FrozenDefender:
    def __init__(self, artifact_dir, threshold_policy='theoretical_reward_1_6'):
        self.load_model(artifact_dir, threshold_policy)

    def load_model(self, artifact_dir, threshold_policy):
        self.artifact_dir = Path(artifact_dir)
        required = ['lightgbm_model.txt', 'calibrator.joblib', 'feature_schema.json',
                    'categorical_mappings.json', 'defender_config.json']
        missing = [n for n in required if not (self.artifact_dir/n).exists()]
        if missing:
            raise FileNotFoundError(f'Missing calibrated Defender artifacts: {missing}')
        self._baseline = BaselineDefender(self.artifact_dir)
        self.feature_names = [f['name'] for f in self._baseline.schema['features']]
        if METADATA & set(self.feature_names):
            raise ValueError('Metadata leakage in saved model feature schema')
        if self._baseline.calibrator['method'] != self._baseline.config['calibration_method']:
            raise ValueError('Calibrator does not match saved configuration')
        self.threshold = self._baseline.config['thresholds'][threshold_policy]
        self.version = hashlib.sha256(b''.join((self.artifact_dir/n).read_bytes() for n in required)).hexdigest()

    def validate_feature_schema(self, features):
        if set(features) != set(self.feature_names):
            raise ValueError('Defender predictor keys must match saved schema exactly; no label/ID/time')

    def predict_many(self, records):
        for r in records:
            self.validate_feature_schema(r)
        # Frozen, independent per-row inference. No stream history or label is input.
        out = self._baseline.predict(records)
        if not np.isfinite(out).all() or not ((out >= 0) & (out <= 1)).all():
            raise ValueError('Invalid calibrated probability')
        return out

    def predict_fraud_probability(self, features):
        return float(self.predict_many([features])[0])

    def decide(self, features):
        p = self.predict_fraud_probability(features)
        return ('FLAG' if p > self.threshold else 'APPROVE', p)
