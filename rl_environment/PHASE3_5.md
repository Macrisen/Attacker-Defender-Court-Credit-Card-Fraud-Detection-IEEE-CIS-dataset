# Phase 3.5: constraints and training-readiness gates

Run all executed checks and regenerate only the new reports:

```bash
.venv_defender/bin/python scripts/run_phase3_5.py
```

Separate suites:

```bash
.venv_defender/bin/python -m unittest discover -s tests/rl_environment -v
.venv_defender/bin/python -m unittest discover -s tests/phase3_5 -v
```

Open `reports/phase3_5/phase3_5_readiness_report.html`. The report links the threat
model, per-column registry, dependency diagnostics, temporal protocol, delayed
credit derivation, numerical tests, audit logs and readiness decisions. The prior
Phase 3 report and baseline artifacts remain unchanged. Regression outputs go to
`reports/phase3_5/phase3_regression/`.

## Interfaces

- `feature_registry.py`: classifications backed by explicit evidence. UNKNOWN
  never silently becomes MUTABLE. Scenario B is a proposal-only exception for
  amount and does not change the evidence classification.
- `feature_dependency_validator.py`: supported declarative dependencies and
  fail-closed UNKNOWN validity. Current exact rules apply to synthetic data only.
- `attack_constraint_engine.py`: bounded Box controls plus STOP, immutable
  metadata, accepted-displacement cost, original-vector preservation and a
  one-query terminal wrapper. `ConstraintAttackerEnv` supplies Gymnasium's API
  only for established control/dependency/label contracts. It rejects the current
  IEEE-CIS scenarios rather than allowing PPO to train on blocked actions.
- `court_credit_assignment.py`: origin-linked pending records and historical
  observations; time-based discount, maturity and boundary checks; one-time replay
  export; a tested TD-target function. `CourtCreditCollector` returns a structured
  collection result, not a scalar-reward Gym tuple.
- `temporal_protocol.py`: exact saved-holdout preservation, 45/10/15/10/20 windows,
  timestamp ties, 3-day maturity suffix, dataset alignment and provenance checks.

Do not feed legacy `CourtEnv.step()` arrival-time reward to a learner alongside
origin-attributed rewards. They describe the same outcomes with different
accounting and would double-count them. `court_credit_protocol.md` specifies the
primary off-policy preparation path and unresolved learner/state-history work.

## New Defender protocol

`scripts/fit_temporal_defender.py` implements earlier fitting, inner chronological
early stopping, train-only category mapping, calibration-only selection/refit and
the prespecified 1/6 threshold. It writes a new directory and never overwrites the
baseline. The full pipeline was executed on a temporary synthetic fixture only.

The real current cleaned feature provenance is intentionally rejected: its feature
selection cutoff is later than the new fitting window. This is not a connection or
dependency error, and it must not be bypassed by falsifying the manifest date.

Before a real new fit, produce a reviewed earlier-window feature selection artifact
with `fit_cutoff`, `selected_predictors`, and the matching `parquet_sha256`. Match its
feature schema and use the emitted temporal protocol:

```bash
.venv_defender/bin/python scripts/fit_temporal_defender.py \
  --parquet path/to/appropriate_source.parquet \
  --schema path/to/reviewed_feature_schema.json \
  --protocol reports/phase3_5/temporal_split_protocol.json \
  --provenance path/to/reviewed_fit_only_provenance.json \
  --output models/temporal_defender_v1
```

A newly audited early-prefix feature set may not reproduce exactly 418 predictors.
Version it separately and retain the legacy Cleaned 418 representation. Keeping
the historical 418-feature representation permits a conditional comparison, not
a claim that its original feature selection was independent of the later RL data.
The dataset has no true historical label-availability timestamps; offline Defender
fitting assumes historical training labels are available. Do not interpret that
assumption as measured real-world verification timing.

## Original branch

Original model artifacts exist; matching Original Parquet was not present or
created by this phase. To explicitly export existing PostgreSQL source tables:

```bash
.venv_defender/bin/python scripts/export_original_parquet.py
```

This optional command uses environment/libpq credentials and the existing chunked
read-only exporter. It requires `public.train_merge` and `public.test_merge`, validates
the Original artifact schema, labels, full values, NULL counts, and ID/time alignment
against Cleaned, and refuses to overwrite existing Original files. No Original
experiment is claimed until the data actually exists and an experiment is run.

## Interpretation

Current real attacker training is BLOCKED. Scenario B's amount edits are hypothetical
and remain UNKNOWN_VALIDITY because the dependency graph is unresolved. Synthetic
ASR values are clearly separate; IEEE-CIS ASR is not applicable.

Court origin-credit mechanics pass, but Court training is PARTIAL, with an active
temporal/provenance gate and custom learner/cohort/history integration pending.
No PPO/DQN training is performed. The historical holdout was already evaluated in
the baseline; Phase 3.5 does not read its labels/features or use it for new tuning.
