# Phase 3: fraud simulation environment

This package runs an offline, chronological **environment mechanics test**, using
the saved calibrated LightGBM Defender. It does not train PPO/DQN, retrain Defender,
modify Parquet or connect to PostgreSQL.

## Run

From the project root:

```bash
# Existing project environment already contains these dependencies.
.venv_defender/bin/python -m pip install -r rl_environment/requirements.txt
.venv_defender/bin/python -m unittest discover -s tests/rl_environment -v
.venv_defender/bin/python scripts/run_rl_environment_smoke_test.py
```

The runner executes tests, simulates 10,000 transactions, repeats with the same
seed, checks reproducibility and input/model hashes, and writes actual reports to
`reports/rl_environment/`. To retain separate runs:

```bash
.venv_defender/bin/python scripts/run_rl_environment_smoke_test.py \
  --output reports/rl_environment_repeat
```

Configuration is in `configs/rl_environment.yaml`. Report files may be regenerated;
baseline model artifacts and source datasets are never written. A fresh Python
environment also needs the baseline's pinned inference dependencies in
`reports/lightgbm_baseline/artifacts/requirements.txt` and macOS libomp if required.

## Chronology and scientific scope

`TransactionStream` reads the actual Parquet schema, checks ID uniqueness and
NULL metadata, and uses the saved baseline model-training cutoff `9614637`.
It checks that this partition has the baseline's saved row count, then orders by
TransactionDT and TransactionID. It loads only selected training records into the
simulation. Calibration and final holdout records are excluded. Unlabeled Kaggle
test data is rejected for reward-bearing simulations.

The **frozen baseline was fitted on the entire model-training partition**. This
early-period replay is consequently in-sample and would have model-level
look-ahead if interpreted as a historical deployment. Use it only for environment
development, not independent fraud-performance or online-policy claims. Future
policy evaluation needs a Defender fitted before the evaluation stream. The
environment itself never supplies future verification labels to agents.

Frozen inference is vectorized across independent transactions for efficiency.
The cache contains no labels and is internal; only the current score is exposed.
The baseline's float32 predictor encoding, category mapping and isotonic calibrator
are reused exactly. The Parquet data remains untouched.

## Component APIs

- `TransactionStream`: `reset`, `next_transaction`, iterator, `batch`.
- `FrozenDefender`: `load_model`, `validate_feature_schema`,
  `predict_fraud_probability`, `decide`, internal batched inference.
- `CourtEnv`: Gymnasium `reset(seed=...)`, `step(action)`, `Discrete(2)`.
- `ContinuousAttackerEnv`: Gymnasium Box controls; **synthetic fixtures only**.
- `BudgetManager`: independent Court and audit tokens per transaction window.
- `VerificationSimulator`: private pending queue, positive simulated delays,
  due-time release and explicit status tracking.
- `VerifiedMemory`: verified-only insertion, recent statistics and sampling.
- `FraudSimulator`: causal controller, evaluation-only summaries, explicit drain.

`InternalTransaction`, controller records/checkpoints and evaluation summaries are
simulator-only. Policies receive the observation vector and public step results,
not the stream, hidden labels, or pending queue. This is API isolation, not a
security sandbox against malicious Python code.

## Observations and decisions

Court observes only `transactionamt` plus a missing-value indicator, current
calibrated probability/decision, available Court budget ratio, window progress,
pending count, recent verified count/fraud rate and elapsed time since simulated
Defender deployment. Missing numeric observations use a zero placeholder **and a
mask**; original feature values remain unchanged. Encodings are fixed analytic
functions with no training/calibration/holdout fitting. Vector length is 10.

The threshold is loaded from the artifact config: by default the theoretical
reward policy `p > 1/6`, with ties APPROVE. All predictor keys must match the
saved schema; ID, time and target are rejected as predictors.

## Budget, random audits and reward causality

The default 10,000-transaction window permits 200 investigations: 180 Court and
20 audit tokens. Court exhaustion converts an INVESTIGATE request to SKIP without
cost. Random audits apply only to APPROVED transactions not already investigated
by Court, including blocked requests. The seeded probability is 20/10,000 per
eligible arrival, capped at the reserve. This online policy does not guarantee
that the reserve is fully used; it never consults future counts or labels.

An accepted Court investigation incurs -1 immediately. Fraud completion returns
+10 for a missed fraud (Defender APPROVE), or +3 for Defender FLAG. Legitimate
completion returns zero additional reward. Cost is never charged twice. Audit
rewards are tracked separately and never credited to a Court action. Evaluation
business reward is also kept separate from these learning signals.

Delays are Uniform(1,3) simulated days, measured in TransactionDT seconds. Results
available by the next arrival are released before its observation. Each reward
event records the originating action/transaction and completion/emission times.
The ordinary Gym transition receives rewards that arrived during that transition;
this does **not** by itself solve delayed-action credit assignment. Before formal
PPO, define variable-time discounting and delayed reward handling explicitly.

At the last transaction, the Gym episode is truncated. Pending investigations
remain censored in the queue. The smoke runner then invokes `drain_pending()` as
a separate diagnostic, advancing to each completion time. Its results appear in
a separate summary row and are not inserted into the truncated Gym rollout.
Reset archives pending internal checkpoints and summary before starting a fresh
episode. Durable cross-process restart/recovery is not implemented.

## Attacker status and controllability

**Real-data perturbation: BLOCKED BY THREAT MODEL.** No IEEE-CIS predictor has
proven attacker controllability and label-preserving dependency rules here.
Metadata is immutable; anonymized features and amount are unknown/unproven.
`identity_attack` preserves values and generates no adversarial success claim.

The synthetic continuous fixture has one mutable quantity and a submission gate
(Box shape 2). Unit price is fixed, total is derived, quantity is bounded [1,10],
and displacement from the original is limited to 2 units. A synthetic fraud-intent
assumption preserves the fixture label, never an IEEE-CIS label. Stop gate >0 or
step 3 submits; only one final Defender query is permitted. There is no intermediate
model feedback. Reward is +1/-1 for fraud APPROVE/FLAG minus final normalized
displacement cost; invalid actions terminate with -2 and no query.

Action history, current/original quantity and remaining budget produce a fixed
10-element observation. Tests check clipping, invalid actions, dependent values,
cost, stop behavior and query limits. These tests are not real attack evaluations.

## Reports and interpretation

Open `reports/rl_environment/environment_validation_report.html`. Companion files
include executed config, simulation summaries at stream horizon and after drain,
the full feature-controllability audit, action/event logs, unit-test output,
schema NULL counts and file hashes.

Overall status is **PARTIAL**, even with passing mechanics tests, because real
attack transformations are unvalidated. The Original branch has model artifacts
but no matching Parquet dataset, so is NOT AVAILABLE for this simulation. The
loader/Defender are schema-driven and can support that branch when matching data
exists; no Original results are invented.

Verified Memory is selected, not population-representative. Court selection,
approval-only audits and delayed censoring introduce bias. No propensity
correction, production loss estimate or adversarial robustness claim is made.
