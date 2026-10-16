# LightGBM Defender baseline

This package trains and evaluates only the fraud Defender. It does not implement
an Attacker, Court, reinforcement-learning environment, or continual updates.
All PostgreSQL access is a read-only, repeatable-read transaction. No database
tables, columns, or rows are modified.

## Environment and execution

The executed environment is `.venv_defender/`. Packages are installed there, using
the existing system Pandas/plotting dependencies. A current NumPy wheel is local
to this environment. On this Mac, Homebrew `libomp` supplies LightGBM's OpenMP runtime.
The final pinned package versions are saved in the model artifact directory.

For a fresh environment:

```bash
python3 -m venv .venv_defender
.venv_defender/bin/python -m pip install -r reports/lightgbm_baseline/artifacts/requirements.txt
# macOS only, if LightGBM cannot find libomp:
brew install libomp
```

Database configuration uses libpq environment variables (`PGDATABASE`, `PGUSER`,
`PGHOST`, `PGPORT`, `PGPASSWORD`) or `.pgpass`. A project `.env` file is supported;
copy `.env.example` and fill only the values needed locally. Never commit passwords.
Without overrides, the database name is `fraud_detection` and libpq uses the current
operating-system user. Source tables are `public.train_clean` and `public.test_clean`.

```bash
.venv_defender/bin/python -m unittest defender.test_protocol -v
.venv_defender/bin/python -u -m defender.train
```

Default output: `reports/lightgbm_baseline/`. Existing trained artifacts are not
overwritten. To rerun, choose a fresh path:

```bash
.venv_defender/bin/python -u -m defender.train \
  --output reports/lightgbm_baseline_repeat --threads 4 --max-rounds 2000 --patience 100
```

## Leakage controls

- Discover schemas and predictor count dynamically. Validate unique IDs, binary
  training labels, compatible test fields/types, NULLs, and nonfinite numeric values.
- Preserve the earlier audit boundary: `TransactionDT <= 12192842` is pre-holdout.
- Within that approximately 80%, allocate 65/80 to model training and 15/80 to
  calibration. Timestamp ties stay together; ID breaks sort ties deterministically.
- Use the first 85% of the model-training period for inner fitting and the last
  15% for average-precision early stopping. Fit inner categorical mappings on the
  inner-fit rows only. Then refit the selected iteration count and final mappings
  on the entire model-training period.
- Fit preliminary calibrators on the first half of calibration; select raw,
  sigmoid, or isotonic by log loss on the next quarter (Brier breaks ties).
- Refit calibrators on the first three quarters of calibration. Select the empirical
  reward threshold on the final quarter, preferring fewer alerts on reward ties.
- Freeze the model, calibrator choice and thresholds before evaluating holdout.
- The earlier feature audit used the first 80%, which includes the present
  calibration period. The final holdout remains independent, but the calibration
  period is not independent of historical feature-selection decisions. This is
  explicitly documented in the research report.

No numerical imputation, scaling, target encoding, class weighting, oversampling,
or outlier removal is performed. Numeric predictors become float32 at model input.
Identifiers, target and time remain metadata and are not predictors.

Categorical roles include text, card/address codes, and id_12–id_38. Numeric-looking
codes receive the same normalization during training and inference. Final category
vocabularies use only model-training values. Missing and previously unseen categories
map to `-1`, handled by LightGBM as missing. Anonymized numeric feature semantics and
availability at transaction decision time remain a research limitation.

## Experiments

- **B_clean:** primary baseline from the actual clean tables.
- **A_original:** evaluated only if `public.train_merge` exists and its shared values
  and IDs match the clean table exactly. Extra original features are joined by ID;
  this is not a rerun of the cleaning pipeline.
- **C_correlated_candidates_removed:** run only if prior audited drop candidates
  remain present. If the clean table already excludes all of them, C is identical
  to B and is explicitly marked not distinct, without a fabricated training run.

B is the designated primary model; holdout comparisons do not select a winner.

## Saved artifacts and inference

The primary model is in `reports/lightgbm_baseline/artifacts/`:

- `lightgbm_model.txt`: final booster.
- `calibrator.joblib`: selected probability calibrator.
- `feature_schema.json`: exact predictor order, SQL types and categorical roles.
- `categorical_mappings.json`: frozen, training-only category vocabularies.
- `defender_config.json`: reward matrix, strict threshold policy and method choice.
- `training_config.json`: model parameters and split/selection protocol.
- `requirements.txt`: package versions used for the executed run.

```python
from defender.inference import predict_fraud_probability, defender_decision

# transaction is a dictionary containing all predictor names in feature_schema.json.
# Explicit None/NaN is allowed; ID/time metadata can be included but are not used.
p = predict_fraud_probability(transaction)
action, p = defender_decision(transaction, threshold=1/6)
```

For batched inference or a different artifact directory:

```python
from defender.inference import Defender
model = Defender('reports/lightgbm_baseline/artifacts')
probabilities = model.predict(dataframe_of_transactions)
```

Missing predictor keys and unexpected columns raise a schema error. Column case
is normalized only when unambiguous. Numeric parsing failures and infinities are
rejected; missing values and unseen categorical values are supported. The policy
uses strict `p > threshold` for FLAG; ties APPROVE. The default is the requested
theoretical threshold, not the empirically selected threshold. To use the latter,
read `thresholds.calibration_reward_max` from `defender_config.json` explicitly.

## Outputs and interpretation

Open `reports/lightgbm_baseline/lightgbm_baseline_report.html`; all figures are
embedded, so the HTML is portable. CSV report links require the accompanying files.
The directory also contains dataset integrity/schema checks, calibrated and raw
metrics, all threshold comparisons, predictions, temporal windows, gain and sampled
permutation importance, calibration-selection evidence, execution logs, and manifests.

Average precision is the primary PR metric. Trapezoidal PR-AUC is separately named.
ECE uses 15 equal-width probability bins. Threshold selection maximizes the given
research reward, without a capacity constraint. Theoretical `1/6` is only justified
under that reward matrix and adequately calibrated probabilities.

When reading probability CSVs with Pandas, use `float_precision='round_trip'` to
preserve strict comparisons at threshold boundaries. Verify the saved outputs with
`.venv_defender/bin/python -m defender.verify_outputs`. The executed run's
`export_precision_fix.json` records the lossless prediction-export correction;
the model, calibration, thresholds and reported metrics were unchanged.

Training and prediction timings are measured. RSS is a cumulative process peak,
not an isolated experiment allocation. Single-transaction inference timing includes
encoding and calibration, excluding one-time model loading.

The dataset is not a counterfactual simulator. Good offline metrics alone do not
establish resistance to adaptive fraud, optimal Court policy, or production readiness.
