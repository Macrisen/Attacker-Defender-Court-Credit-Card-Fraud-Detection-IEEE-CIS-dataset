# Phase 4 attacker readiness

**Mode B: READY_FOR_ASSUMPTION_BASED_PPO**

The declared Box(1) task, Decimal transform, hard constraints, fixed-prefix plausibility, terminal decision-only query budget, reward, paired actual-data experiments and delayed-verification integration are validated. 106 tests PASS, including all 73 earlier regression tests. No PPO was trained.

**STRICT: BLOCKED. Real-world causal attack validity: NOT ESTABLISHED.**

No anonymous feature is promoted to causally mutable. Holding non-amount fields fixed and fraud intent unchanged are explicit counterfactual assumptions. Statistical plausibility does not establish realizability.

Next: implement the development-only source sampler and PPO training orchestration with fixed seeds/budgets; compare against the saved identity/random/directional references and a one-step contextual-bandit baseline. Keep the default epsilon=.05 and lambda=.1 unless a new predeclared development protocol is recorded. Reward/epsilon sensitivity is not a license to optimize on final holdout. Evaluate a frozen policy on RL Validation with separate denominators and one query per episode.

Court remains PARTIAL, and continual Defender updating is not implemented. Keep the Phase3.6 delayed-credit semantics; this phase only validates rule-based end-to-end integration. Prior temporal Defender/label-availability assumptions remain unchanged. Final holdout was historically baseline-evaluated, but no new Validation or Holdout feature/label data were loaded for these experiments.
