# Research limitations

- Mode B modifies only amount under assumed fraud-intent preservation and fixed other fields; causal real-world manipulability remains BLOCKED.
- Historical labels for initial Defender fitting/calibration are assumed available; actual label-discovery times are unavailable. Verification delays are simulated.
- The reserved final chronological holdout was evaluated by an earlier baseline study. It is not historically pristine; it was excluded from all new fitting and selection until protocol freeze.
- Repeated Development episodes are offline simulation, not independent deployment observations. Validation checkpoint selection can be optimistic; no redesign based on final results is permitted.
- Recurrent Court uses truncated gradients over128 steps, 40 updates per100k decision budget and compact11-feature observations. Recurrent state does not prove sufficient information or optimality. Poor performance is retained.
- Continual refits use selection-biased verified labels, sparse chronological calibration and a20k-row historical base. No unbiased prevalence/IPW claim. Drift outputs are descriptive changes, not proven concept drift.
- Continual adaptation is executed only on Development, seed42. Validation/final freeze the resulting models; they do not demonstrate live continual adaptation on protected evaluation labels. Some strategies may remain ineligible and retain the baseline.
- Daily-block bootstrap averages repeated training-seed outcomes by source; a single temporal dataset and limited blocks restrict uncertainty interpretation. It does not eliminate historical holdout exposure or model-selection uncertainty.
- Removing one fitting-prefix-constant feature is a small feature ablation. Probability/utility equality must be measured and does not establish broad feature-selection benefits.
- Utilities are research assumptions, not monetary profit. End-to-end utility combines initial Defender utility,10 per completed Court discovery of initially missed fraud, and all investigation costs. Court's own+3 flagged-fraud reward is a separate training objective.
- No joint-training convergence, game equilibrium or universal hybrid superiority is claimed. Training peak RSS was not instrumented.
