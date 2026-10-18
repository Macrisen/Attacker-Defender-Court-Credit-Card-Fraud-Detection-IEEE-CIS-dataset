# Phase 3.6: temporal Defenders and mature Court episodes

## Reproduce
Run from the repository using `.venv_defender/bin/python`:

1. `scripts/export_original_parquet.py` — read-only PostgreSQL Original export; refuses existing destinations.
2. `scripts/train_temporal_defender.py` — fit-only audit, two matched fits, calibration and development/validation metrics; refuses existing model directories.
3. `scripts/export_temporal_cleaned.py` — exact-value column projection of Original into the newly selected version; refuses existing destinations.
4. `scripts/run_phase3_6_smoke.py` — two aligned real 10,000-row development simulations; synthetic action/credit/gradient checks.
5. `scripts/test_phase3_6.py` — all Phase 3, 3.5 and 3.6 unit/integration tests.
6. `scripts/report_phase3_6.py` — reports from executed results, protected-artifact verification and manifest.

Existing model/data destinations are intentionally protected. For an independent rerun, use a new checkout/output namespace; do not delete prior baseline artifacts. Reports in `reports/phase3_6` belong to this run. No PostgreSQL table updates, full PPO/DQN fitting or final-holdout model evaluation occur.

## Data and model protocol
Use saved ordered-ID SHA256s and exact 45/10/15/10/20 windows. Original schema is the complete source schema, not a statistically selected subset. SQL metadata and counts are verified against four existing tables. Original export reads all source values for lossless integrity checks, including the reserved holdout; these reads are not selection/evaluation inputs. Only ID/time metadata from holdout enters partition verification. Exporting a cleaned column projection likewise performs no statistical fitting.

The new selector fits on 265,743 rows ending at TransactionDT 6443292. Drop only all-NULL columns, complete constants, or exact duplicates including the NULL mask. Retain high-missingness, near-constant and highly correlated features for REVIEW. Numeric Pearson excludes categorical codes and metadata. Correlation is pairwise complete with shared observation counts; <1,000 is unreliable. Missing/observed and rare-value fraud rates, and chronological-third availability statistics, are descriptive fit-only evidence. Unknown semantic/temporal availability of anonymized features remains unresolved. Exact duplicates retain the first feature in source schema order. No feature-count target is imposed.

The resulting `cleaned_prefix_430` drops only `v107`, which is constant throughout the fit prefix. The historical 418-predictor files/tables remain unchanged. A prefix constant may vary later; no later data were consulted to reverse or validate its selection. Native NULL handling remains enabled. Train-only vocabulary maps unknown/missing categorical values to -1; LightGBM warnings about negative categories converting to missing are expected.

Both models have max 2,000 rounds, 100-round early stopping on the last chronological 15% inside Defender Fit, then refit on all Defender Fit. No hyperparameter search is performed. Calibration uses first-half fitting and second-half log-loss method selection, followed by calibration-window refit. Candidate methods: raw, sigmoid, isotonic. All mappings are fitted only on the fitting prefix. The operating point is fixed: FLAG has expected utility 15p−5; APPROVE has 1−21p; FLAG iff p>1/6. No evaluation data determine the threshold.

## Court observation and action
Baseline ten dimensions remain: signed-log/tanh amount, amount missing flag, calibrated probability, Defender flag, Court budget ratio, budget-window progress, pending-count transform, recent verified fraud rate/count transform, time since model deployment. V2 adds deterministically bounded time remaining to the declared decision horizon: 11 float32 dimensions. Deployment time is the end of calibration for the new smoke runs. Discrete(2) is unchanged. Exhausted INVESTIGATE becomes blocked SKIP without Court cost. Audits use a separate reserve.

Full causal prefixes of observations, actions and timestamps are retained using persistent linked histories. They contain no retrospectively inserted labels. Compact summaries, and the linear mean/last-observation synthetic probe, are not asserted to be Markov-sufficient. Queue composition/age/risk summaries are not yet implemented. A recurrent policy and controlled policy comparison remain Phase 4 work.

## Why use complete-episode Monte Carlo here
The Phase 3.5 origin ledger remains the accounting source. For origin i:

    u_i = -cost_i + gamma_day^((due_i - origin_i)/86400) * bonus_i
    G_i = sum_{j>=i} gamma_day^((origin_j-origin_i)/86400) * u_j

Each cost and bonus appears once. The discounted sum from episode start equals discounted immediate costs plus discounted completion bonuses. This equality alone does not establish a one-step Bellman recursion on compact observations. Therefore Phase 3.6 does **not** consume the legacy `dqn_target` or arrival-reward stream for learning.

The selected adapter computes full finite-episode MC targets with zero bootstrap after the predeclared last decision. A fixed collection policy runs the whole episode. After the last decision, no more Court or audit actions occur; a verification-only tail waits out all existing investigations. The maximum delay is three days. Complete episodes, including SKIP actions, are released together only after the full maturation guard, avoiding fast-SKIP versus slow-INVESTIGATE insertion bias. Partial or truncated episodes are quarantined, not assigned zero continuation value. There is no cross-partition bootstrapping.

Targets describe the originated-utility objective; previously initiated pending bonuses are not added again to later decision targets. In this simulator earlier investigation labels/delays/bonuses are unaffected by later actions. That independence must be revisited if a later design introduces action-dependent service delays or queue capacity. This is fixed-policy value regression, not off-policy control or evidence of optimized Court performance. Policy changes between episodes require fresh on-policy data or an explicit justified correction; arbitrary mixed-policy replay is not authorized.

A synthetic fixture manually checks: fraud investigation at day 0, cost1, bonus10 at day2; legitimate investigation at day1, cost1, bonus0 at day3; SKIP at day2. With gamma_day=.5, attributed utilities are [1.5,-1,0] and MC targets [1,-1,0]. One deterministic linear squared-error gradient step demonstrates integration, not full RL training.

See [Sutton and Barto, episodic Monte Carlo](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) and [Farama termination versus truncation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API). The derivation and simulator-specific assumptions above define this adapter's actual contract.

## Timeline and censoring
Defender fitting/calibration labels are assumed historically available for offline supervised training. The dataset lacks observed label-discovery timestamps, so this availability assumption cannot be empirically verified. Simulated delayed labels apply to Court, not retroactively to calibration.

RL Development ends at 10437996; latest full-window action time is 10178796. RL Validation ends at 12192842; latest action time is 11933642. Each smoke run instead declares its first-10,000-row decision horizon in advance, and matures only that cohort inside RL Development. No later transaction values are read for the maturity tail. Random audits contribute to verified memory but never Court reward. Pending investigations survive budget rollover. Unplanned termination is censored; diagnostic draining cannot revive censored training records. Validation targets cannot enter the training adapter.

## Continuous attacker V2
`ContinuousControlAttackerEnv` exposes Box(-1,1,shape=(k,),float32), where k is the number of authorized mutation controls. No extra predictor or STOP coordinate is included. An all-near-zero action (absolute values <=1e-6) means STOP without mutation; other actions continue, and step three submits automatically. The legacy k+1 explicit-STOP adapter remains available only for compatibility. This change is explicit and versioned.

Synthetic quantity has k=1, bounds [1,10], displacement budget2, fixed unit price and justified derived total. Single terminal decision query; no intermediate score feedback. Invalid actions terminate with −2 minus accepted perturbation cost. Identity outcomes never enter ASR's eligible denominator. No IEEE-CIS feature currently has a verified mutable/dependency contract, so real Attacker PPO remains BLOCKED and real ASR is undefined. Hypothetical amount edits remain UNKNOWN_VALIDITY, without Defender queries.

## Phase 4 gate
Fresh temporal Defenders and aligned Original/new Cleaned comparison are ready within the documented offline-label assumption. Court is PARTIAL: correct synthetic targets and actual mature episode integration do not establish recurrent policy effectiveness, memory sufficiency or off-policy correctness. Real Attacker PPO is BLOCKED. Continual Defender updates, forgetting controls and replay under changing model versions remain unimplemented. The final holdout was historically baseline-evaluated; it is reserved from all new tuning, not historically unseen.
