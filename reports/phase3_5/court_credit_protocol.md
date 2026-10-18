# Court delayed-credit protocol

Primary preparation path: off-policy, time-discounted DQN-style replay with origin-linked maturation.
This is a custom replay/TD-target interface, not a plug-in replacement for standard SB3 DQN or PPO.
No neural policy is trained in Phase 3.5.

## Accounting and discounting
For an accepted investigation at time t_i, cost c_i is recorded once immediately. A completion at d_i
has bonus b_i and is linked to the immutable origin transaction/action IDs. Once observed, define

    attributed_reward_i = -c_i + gamma_day ** ((d_i - t_i)/86400) * b_i
    discount_i = gamma_day ** ((t_(i+1) - t_i)/86400)
    TD_target_i = attributed_reward_i + discount_i * (1 - terminated_i) * max Q_target(next_observation_i)

The original observation and next observation are captured when they occurred and never rewritten
with future verified statistics. No replay record is released until its next state and verification
are actually available within the allowed training window. Costs cannot be emitted as a separate
training transition and then charged again at completion. Audit bonuses are excluded from Court rewards.
The ordinary Phase 3 scalar step reward remains an arrival-time diagnostic for regression only.
`CourtCreditCollector` exposes a structured object, not a scalar-reward Gym tuple, to prevent accidental mixing.

For exponential time discounting, gamma(t_i)*gamma(d_i-t_i)=gamma(d_i). Hence the sum of origin-discounted
attributed rewards equals the sum of costs at origins and bonuses at their completion times for completed
investigations. An executed numerical test checks this identity. This is explicit known-origin redistribution,
not an implementation of learned RUDDER decomposition. See the primary
[RUDDER paper](https://arxiv.org/abs/1806.07857) for the general return-equivalence idea.

## What this does and does not establish
The numerical return identity does not prove that the 10-element Court observation is Markov-sufficient.
Pending queue composition and hidden labels create partial observability. A recurrent/history-aware
Court or validated state representation, and the custom learner integration, remain Phase 4 work.
Do not label vanilla PPO ready merely because reward records have origin IDs. On-policy PPO would
need fixed-policy collection until all relevant trajectories mature, then correct time-aware returns;
it is not the primary implemented adapter. Eligibility traces and serialized event-driven investigation
were considered but not selected: traces do not by themselves solve the boundary/availability rules,
and serializing every investigation would change concurrent investigation capacity.

## Boundaries and incomplete outcomes
- Budget rollover never resets investigation/credit records.
- Episode truncation is not a true terminal state; unresolved or truncated records are quarantined,
  never silently converted to zero-outcome rewards or terminal TD targets.
- Training/validation/holdout partitions are checked by both tag and origin time. A completion or
  next state observed beyond the training cutoff cannot enter training replay.
- Pending at termination is explicitly censored. An evaluation-only drain may reveal labels later,
  but it cannot re-enable censored training records.
- The temporal protocol reserves a 3-day maturity suffix in RL development and validation windows.
  Stop learned actions by the decision cutoff, advance the verifier to the partition end, and mature
  the origin records without introducing later labels. Define the decision-prefix finite horizon
  explicitly (including time-to-horizon in a future learner state) before treating its last action
  as truly terminal. Otherwise retain truncation and do not bootstrap across a held-out boundary.

Only mature records are available online, so maturation and sampling may bias provisional replay.
Do not assume incomplete records are legitimate or train disproportionately on immediately finalized
SKIP actions. A future trainer should release complete decision cohorts after the maximum delay.
The current adapter prepares records but does not claim that a cohort-aware neural trainer is implemented.

Termination/truncation handling follows [Farama's explanation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).
