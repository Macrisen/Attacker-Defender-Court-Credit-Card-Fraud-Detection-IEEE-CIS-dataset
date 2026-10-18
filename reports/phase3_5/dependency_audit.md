# Dependency audit

All 418 cleaned predictors were inspected through their saved schema and revised fitting-window values.
The registry retains the original conservative classifications. C/D/V/id variables remain opaque;
ProductCD, card, addr, dist, email-domain, M and other identity fields are grouped by schema/name only.
No undocumented detailed semantics or deterministic relationships are asserted.

TransactionAmt type: decimal128(8, 3); fitting-window rows: 265743;
NULL count: 0; positive: 265743; zero: 0; negative: 0.
Quantiles (float64 diagnostic copies):

```json
{
  "0": 0.292,
  "0.01": 10.0,
  "0.05": 20.12,
  "0.25": 42.5655,
  "0.5": 68.95,
  "0.75": 125.0,
  "0.95": 424.95,
  "0.99": 1033.664000000013,
  "1": 6085.23
}
```

`amount_correlations.csv` reports pairwise non-NULL Pearson associations with eligible numeric predictors,
including observation counts. `conditional_amount_profile.csv` reports amount quantiles conditional on
ProductCD/card4/card6. These are descriptive plausibility checks, not dependencies, causal effects,
or proof that a counterfactual fraud label survives an edit. A candidate outside observed support is
questionable, but being inside support is not sufficient for validity.

Unresolved: which aggregate/anonymized attributes encode amount, how to recompute them, whether merchant/
product constraints would change, and whether the edited transaction would remain fraudulent.
No real dependent outputs are invented. The only tested exact dependency is the synthetic total equation.

To authorize a future control, supply reviewed semantic evidence, domain bounds, exact recomputation
rules for all affected variables, and a justified label-preservation contract. Registry and validator
must agree; UNKNOWN does not become MUTABLE automatically.
