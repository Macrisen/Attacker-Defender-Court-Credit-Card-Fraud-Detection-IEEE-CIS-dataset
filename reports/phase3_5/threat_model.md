# Attacker threat model

Objective: make an attacker-owned fraudulent transaction receive APPROVE in an offline research simulator.

## Knowledge and query budget
The acting attacker sees original/current authorized control values, explicit missingness,
remaining displacement budget, step and its own action history. No architecture knowledge,
probabilities, unrelated labels, future transactions or privileged verification data is provided.
One final decision-only Defender query is permitted per episode; there is no intermediate query.
An evaluator may make a separate baseline query for paired analysis; that score never goes to the agent.
STOP is the final Box coordinate >0; otherwise the episode submits on step 3.
Out-of-bound/nonfinite/malformed actions are rejected, not silently clipped.

## Scenario A: conservative real-data
No mutable IEEE-CIS control has been established. All 418 predictors retain UNKNOWN classification;
ID, time and label remain IMMUTABLE. Real-data adversarial rollout is BLOCKED. Identity is not an attack.

## Scenario B: explicit hypothetical amount sensitivity
Assume the adversary could change a positive amount within ±5% of its original value, and that
fraudulent intent would remain unchanged. Neither assumption is an observed counterfactual fact.
Changing amount while freezing anonymized features is not proven consistent. The engine therefore
allows proposal/constraint checks but returns UNKNOWN_VALIDITY and never queries Defender for a
modified transaction until dependency evidence exists. No valid IEEE-CIS ASR is reported.
This scenario must not be promoted to realistic financial fraud simply by setting a config flag.

## Synthetic controlled fixture
Quantity is MUTABLE, unit price IMMUTABLE, total DEPENDENT (quantity × unit price).
Quantity lies in [1,10], absolute displacement from original is capped at 2. Synthetic fraud intent
is defined as invariant. These are generator contracts, not IEEE-CIS evidence. Synthetic metrics
are mechanically meaningful only for that fixture and are reported separately.

## Rewards and denominators
Terminal reward: fraud APPROVE +1 / FLAG −1 minus configured weighted normalized accepted displacement.
Invalid transformation: −2 minus any previously accepted displacement cost in that episode. BLOCKED/UNKNOWN has no meaningful training reward and no Defender query.
Validity rates use all attempts only when semantic validity is established. ASR denominator is
eligible fraudulent transactions with accepted nonidentity modifications and a submitted decision.
Conditional evasion uses the subset originally FLAGGED. Unchanged natural false negatives are excluded.
Identity reversals that return to the original vector are also excluded.

## Evidence limits
The official [competition data page](https://www.kaggle.com/c/ieee-fraud-detection/data) was consulted;
the retrievable description does not establish an executable causal feature dependency graph.
Column names/types, empirical correlations and conditional ranges do not establish attacker control.
No jurisdiction-specific legal permission is inferred; authorization here means the declared simulator scenario.
