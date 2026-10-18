# Assumption-based attacker state, action and reward

## Scope and threat model
STRICT remains STRICT_BLOCKED: UNKNOWN amount control and dependencies have not been causally validated. ASSUMPTION_BASED is a separate feature-space counterfactual contract. Holding C/D/V/id and all other predictors fixed is an approximation, not a causal dependency solution. The historical fraud label is assumed preserved as intent; it is not an observed label for the modified financial transaction. No registry entry is promoted to causally MUTABLE.

Offline source eligibility uses hidden isfraud==1 only inside RL Development. The simulator owns that selection. Neither actor observations nor Court observations contain hidden labels, other-transaction labels, unreleased verification outcomes or evaluator probabilities. Independent per-row batch inference is used only after all candidate actions are fixed; no future context enters a policy.

## Action and Decimal arithmetic
Box(-1,1,shape=(1,),dtype=float32), controlling TransactionAmt only. Default epsilon .05; sensitivity .01/.03/.05 predeclared. Each non-STOP action proposes an **absolute** amount relative to the source:

    Decimal(original) * (1 + Decimal(epsilon) * Decimal(str(action)))

This is not an incremental update. Source type is decimal128(8,3). Original input must already be finite, positive and exactly representable; float inputs are rejected rather than silently converted. Compute with decimal precision50, round HALF_EVEN to .001, then choose the nearest representable grid point inside the exact ±epsilon interval if rounding alone would overshoot. Decimal precision overflow is rejected. Amount must remain positive and <=99999.999. Source Decimal values remain unchanged. At the model boundary only, the existing saved Defender preprocessing converts to float32, exactly as during fitting; conversion error and any model-input identity are explicitly measured in numeric_representation_validation.csv. Exact Decimal equality defines identity in the attack metric.

Budget is final displacement / (original * epsilon), not total path length. The attacker can revise or undo its own proposals inside the same absolute bounds without intermediate queries. Cost uses the final accepted displacement.

## STOP and episode timing
All |action|<=1e-6 means STOP with the current candidate retained; on the first step it submits identity. Otherwise apply the absolute proposal. Step3 automatically submits. No intermediate Defender query or probability feedback is allowed. A valid terminal candidate gets exactly one decision-only query. Invalid actions terminate with penalty and zero queries. Repeated resolve/step after terminal is rejected. Identity is a separate control with zero terminal reward and is excluded from accepted nonidentity ASR denominators.

The three-step API preserves compatibility with continuous PPO, but it is not a three-query black-box attack. Between actions, all new information is deterministic feedback about the actor's own edits. Evaluate a one-shot contextual-bandit baseline in Phase4; do not claim additional information-gathering power from multiple steps.

## Observation
11 float32 dimensions, all in [-1,1]. Current/original positive amounts use tanh(log1p(amount)/10), a fixed analytic transform with no future-fitted scaling. Fields in order:

1. current_amount_scaled
2. original_amount_scaled
3. relative_signed_change / epsilon
4. remaining displacement budget ratio
5. remaining_steps / 3
6. step_index / 3
7. current_missing_or_invalid indicator
8. original_missing_or_invalid indicator
9. action_1 (zero padded)
10. action_2 (zero padded)
11. action_3 (zero padded)

Other predictor values are internal to Defender only. Terminal info contains status, query count, decision, unresolved dependency status and causal-validity=false, never the full feature vector or probability. Original and modified probabilities in report files are evaluator-only.

## Reward
Predeclared lambda_cost=.1. C=.1*abs((new-original)/original)/epsilon. Intermediate valid steps have reward0. Terminal valid nonidentity fraud: APPROVE => 1-C; FLAG => -1-C. Invalid/assumption-rejected proposal => -2 minus any accepted candidate cost. Identity =>0. STRICT_BLOCKED has no attack learning episode. No bonus rewards large changes. Reward coefficient sensitivity is future work, not optimized on Validation/Holdout.

## Plausibility and acceptance
Statistics fit only the exact 265,743-row Defender Fit window ending6443292, with exact ordered-ID hash verification. No fraud labels enter plausibility fitting. Global amount min/max, q01/q99, log1p median and log-MAD; conditional statistics for ProductCD,card4,card6 only with >=200 valid fitting rows. Unsupported categories use global diagnostics, not guessed conditional evidence. Robust score=abs(log1p(amount)-median_log)/max(1.4826*MAD_log,1e-6). Report original/modified scores, conditional rarity, range/quantile support and relative displacement.

Only hard arithmetic/schema/budget checks determine acceptance. Quantile rarity and anomaly scores are soft diagnostics; rare values are not called impossible. This policy was declared before Defender evaluation. Statistical support cannot prove feasibility or label preservation.

## Experiments and denominator definitions
All 3,537 development fraud sources were eligible in this run. Identity, seeded random_uniform, fixed_decrease and fixed_increase were predeclared; each policy/epsilon pair is a separate episode. Random actions use NumPy SeedSequence([42,TransactionID]), so source actions match across models and epsilons. No best-of selection, repeated probing within an episode or adaptive epsilon selection occurs. Identity is repeated per epsilon as a labeled baseline only. Aggregate attempt counts include repeated sources across explicitly separate experiments; they are not unique fraud counts.

Conditional evasion = accepted nonidentity originally FLAG now APPROVE / accepted nonidentity originally FLAG. Report all originally detected fraud separately, including any identity outcomes. Overall accepted-attack evasion = accepted nonidentity now APPROVE / all accepted nonidentity. The latter includes already-approved fraud and must not be presented as newly achieved evasion. Zero eligible denominators mean NOT APPLICABLE, represented as empty CSV values.

## Hybrid integration
Private fraud eligibility -> finalized copied vector -> one frozen Defender inference -> normal Court observation -> rule-based INVESTIGATE/SKIP -> Uniform(1,3)-day verification -> completed-only Verified Memory. Normal non-attacked arrivals remain unchanged. Provenance carries original source ID/time, original vector, modified vector, action/epsilon, label contract, Defender model/decision, Court decision and completed outcome. Pending labels and source eligibility never enter Court state. Audit/Court budgets remain180/20 per10000. Diagnostic draining stays inside Development and is not a Court policy update. Strict FraudSimulator behavior is preserved; the separate AssumptionFraudSimulator supplies the explicitly finalized vectors.

## Run and Phase4 handoff
From repository root:

    .venv_defender/bin/python scripts/run_phase3_7.py
    .venv_defender/bin/python scripts/test_phase3_7.py
    .venv_defender/bin/python scripts/report_phase3_7.py

The runner fits plausibility, executes predeclared experiments and paired chronological smoke tests. Existing source datasets/models/earlier reports are hashed and protected. Results are written only to reports/phase3_7. PPO can use AssumptionAttackerEnv with privately selected development fraud sources and the persisted plausibility model. Build the training source sampler/orchestrator, freeze PPO budgets/seeds, and maintain one terminal query per episode. Do not feed evaluator-only files into observations. Use RL Validation only for subsequent frozen-policy evaluation/selection under a declared protocol; final holdout remains reserved.

READY_FOR_ASSUMPTION_BASED_PPO means this bounded computational task is validated, not that PPO has been trained or real-world manipulations are feasible. Court remains PARTIAL from Phase3.6; no policy efficacy or continual Defender adaptation is established here.
