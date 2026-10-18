# Phase 3.5 decisions and readiness

## Executed
- Phase 3 regression tests: 29 PASS.
- New Phase 3.5 tests: 24 PASS, including a temporary synthetic revised-Defender fitting pipeline test.
- Real 10,000-row identity replay, separate constrained synthetic episodes, training-only IEEE-CIS proposal checks,
  amount statistics, delayed-credit checks and protected-file integrity checks.
- Exact temporal boundaries computed from ID/time metadata, preserving the old holdout cutoff.

## Attacker PPO: BLOCKED
No validated nonidentity IEEE-CIS controls exist. Scenario B has explicit assumptions but unresolved
dependencies; no real ASR or model queries on invalid counterfactuals. Synthetic control interface is tested.

## Court RL: PARTIAL (training gate BLOCKED)
Origin attribution, time discounts, costs, censorship, reserve budgets and logging pass tests.
Standard Phase 3 arrival rewards must not be passed unchanged to a learner. A cohort-aware custom
off-policy learner and sufficient observation/history remain to be integrated. More fundamentally,
the current Defender/preprocessing provenance overlaps the proposed RL periods.

## Defender integration: PARTIAL for independent RL; frozen inference PASS
New early-fit code exists and runs on controlled synthetic fixtures. The real call was deliberately
blocked by its provenance gate: Feature selection used future RL/calibration data; new fit alone cannot remove this leakage
Refitting the same historically selected 418 features would not repair feature-selection look-ahead.
Re-audit/review a feature set using the new Defender-fit prefix, or use an externally fixed unselected
Original feature schema. Save fresh model/calibrator/category mappings in a new artifact directory.
Do not overwrite the successful baseline.

## Research evaluation: PARTIAL
The 45/10/15/10/20 proposal has validated timestamp-separated windows and 3-day maturation guards.
It is an implemented data-selection/fitting protocol, not an executed independent IEEE-CIS model result.
The final holdout was already evaluated in the earlier baseline; it is reserved from any NEW development,
not historically never-seen. No holdout labels/features are used in this phase.
Original Parquet remains unavailable unless the artifact inventory says otherwise. Use
`scripts/export_original_parquet.py` for read-only export when wanted; it checks schema, all values,
NULL counts and Original/Cleaned ID/time alignment. It was not executed here.

## Fair comparison contract
Original and Cleaned must share exact ID/time windows, maturity suffixes, threshold objective,
attack assumptions, budget 180+20, Uniform(1,3)-day delays, seed 42, and matched policy training budgets.
Each uses its own fitted model/schema and train-only category mapping. No Original scores are invented.
Fix optimization steps, environment interactions and final evaluation protocol before comparing policies.

## Audit bias
Audit probability is logged conditional on current eligibility and remaining reserve, not presented
as a marginal inclusion probability. Deterministic Court selection, exhausted budgets, uninvestigated
FLAG transactions and delayed censorship violate assumptions needed for naive inverse-propensity
prevalence correction. No unbiased estimate/IPW claim is made. Actual prevalence is evaluation-only.

## Phase 4 prerequisites
1. Resolve training-only feature provenance and fit/calibrate the new pre-RL Defender.
2. Finalize recurrent/history/cohort treatment and connect the tested ledger to a time-aware learner.
3. For Attacker PPO, establish causal controls/dependencies or explicitly redesign the research as
   an assumption-based sensitivity study with defensible dependency and label contracts.
4. Freeze evaluation and fair-comparison budgets before any reserved holdout evaluation.

Court-only training cannot yet be labeled READY on the current artifacts. A working delayed-credit
API is necessary but not sufficient. No full PPO/DQN or real revised-Defender training was run.
