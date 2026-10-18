"""Execute Phase 3.5 audits, regression, constraint and delayed-credit checks."""
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from rl_environment.feature_registry import FeatureRegistry, audit_schema
from rl_environment.attack_constraint_engine import AttackConstraintEngine, ConstrainedAttackEpisode, attack_metrics
from rl_environment.temporal_protocol import build_protocol, verify_alignment
from rl_environment.credit_checks import credit_checks
from rl_environment.court_credit_assignment import CourtCreditCollector
from rl_environment.defender import FrozenDefender
from rl_environment.transaction_stream import TransactionStream
from rl_environment.fraud_simulator import FraudSimulator
from scripts.fit_temporal_defender import fit_new_defender
from scripts.run_rl_environment_smoke_test import hashes, write_csv, write_json, table

OUT = ROOT/'reports/phase3_5'


def run_tests():
    results = {}
    for name, path in [('phase3_regression', 'tests/rl_environment'), ('phase3_5', 'tests/phase3_5')]:
        completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', path, '-v'], cwd=ROOT, text=True, capture_output=True)
        log = completed.stdout+completed.stderr
        (OUT/(name+'_tests.log')).write_text(log)
        if completed.returncode:
            raise RuntimeError(f'{name} tests failed; inspect log')
        results[name] = {'status': 'PASS', 'tests': int(re.search(r'Ran (\d+) tests', log).group(1))}
    return results


def amount_audit(parquet, schema, cutoff):
    logging.info('Inspecting training-only amount distribution, conditional ranges and numeric associations')
    data = ds.dataset(parquet, format='parquet').to_table(filter=ds.field('transactiondt') <= cutoff)
    amount = data['transactionamt'].to_pandas().astype(float)
    finite = np.isfinite(amount.to_numpy())
    valid = amount[finite]
    profile = {'scope': 'revised Defender fitting window ONLY', 'fit_end': cutoff, 'rows': len(data),
               'arrow_type': str(data.schema.field('transactionamt').type), 'null_count': data['transactionamt'].null_count,
               'null_percentage': 100*data['transactionamt'].null_count/len(data),
               'nonfinite_nonnull_count': int((~finite).sum()-data['transactionamt'].null_count),
               'positive': int((valid > 0).sum()), 'zero': int((valid == 0).sum()), 'negative': int((valid < 0).sum()),
               'quantiles': {str(q): float(valid.quantile(q)) for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]},
               'note': 'Descriptive calculations use float64 copies; source decimal values and Parquet are unchanged. Empirical bounds do not prove a semantic domain.'}
    correlations = []
    for entry in schema['features']:
        name = entry['name']
        if name == 'transactionamt' or entry['role'] != 'numeric':continue
        col = pd.to_numeric(data[name].to_pandas(), errors='raise').to_numpy(dtype=float, na_value=np.nan)
        mask = finite & np.isfinite(col)
        count = int(mask.sum())
        if count >= 2 and np.std(col[mask]) > 0 and np.std(amount.to_numpy()[mask]) > 0:
            correlations.append({'feature': name, 'pearson_with_amount': float(np.corrcoef(amount.to_numpy()[mask], col[mask])[0, 1]),
                                 'paired_observations': count, 'semantic_dependency': 'UNKNOWN; association is not a recomputation rule'})
    correlations.sort(key=lambda r: abs(r['pearson_with_amount']), reverse=True)
    conditional = []
    for name in ['productcd', 'card4', 'card6']:
        df = pd.DataFrame({'category': data[name].to_pylist(), 'amount': amount})
        for value, group in df.groupby('category', dropna=False):
            conditional.append({'condition_feature': name, 'condition_value': str(value), 'rows': len(group),
                                'q01': float(group.amount.quantile(.01)), 'median': float(group.amount.median()),
                                'q99': float(group.amount.quantile(.99)),
                                'interpretation': 'Descriptive conditional support only; no causal/label-preserving guarantee'})
    return data, profile, correlations, conditional


class SyntheticDefender:
    def decide(self, features):
        p = .8 if features['total'] > 55 else .1
        return ('FLAG' if p > 1/6 else 'APPROVE', p)


def synthetic_smoke(threat):
    original = {'quantity': 6., 'unit_price': 10., 'total': 60., 'transactionid': 7, 'transactiondt': 10,
                'isfraud': 1, 'partition': 'synthetic'}
    results = []
    for name, actions in [('modified', [[-.5, 1]]), ('identity', [[0, 1]]),
                          ('out_of_bounds', [[2, 1]]), ('budget_violation', [[1, -1], [1, 1]])]:
        engine = AttackConstraintEngine(FeatureRegistry.synthetic(), 'synthetic', threat['scenarios']['synthetic'])
        toy = SyntheticDefender()
        episode = ConstrainedAttackEpisode(engine, toy, threat['rewards']);episode.reset(original)
        reward = 0.
        for action in actions:
            obs, r, done, _, info = episode.step(action);reward += r
            assert episode.observation_space.contains(obs)
            if done:break
        original_decision, original_score = toy.decide(original)
        modified_score = toy.decide(engine.current)[1] if info['queries'] else None
        results.append({'fixture': name, 'scope': 'SYNTHETIC_ONLY', 'status': info['status'],
                        'eligible_fraud': True, 'original_decision': original_decision,
                        'decision': info.get('decision'), 'original_score_evaluator_only': original_score,
                        'modified_score_evaluator_only': modified_score, 'cost': engine.cost,
                        'magnitude': abs(engine.current['quantity']-original['quantity']),
                        'reward': reward, 'agent_queries': episode.queries,
                        'paired_baseline_queries_evaluator_only': 1,
                        'action_history': json.dumps(engine.history)})
    return results, attack_metrics(results)


def feasibility(data, registry, threat, count):
    fraud = data.to_pandas().query('isfraud == 1').sort_values(['transactiondt', 'transactionid'])
    selected = fraud.head(count).to_dict('records')
    results = []
    for scenario in ('A', 'B'):
        statuses = []
        for record in selected:
            engine = AttackConstraintEngine(registry, scenario, threat['scenarios'][scenario])
            engine.reset(record)
            action = [0.] if scenario == 'A' else [.5, 1.]
            result = engine.apply(action)
            statuses.append({'status': result['status'], 'reason': result['reason']})
            assert result['modified_transaction'] is None
        results.append({'scenario': scenario, 'scope': 'IEEE_CIS_training_only', 'status': 'BLOCKED' if scenario == 'A' else 'UNKNOWN_VALIDITY',
                        'candidate_fraud_in_audit_window': len(fraud), 'candidate_transactions_checked': len(selected),
                        'eligible_validated_attack_transactions': 0, 'attempted_actions': len(statuses),
                        'valid_modified_transactions': 0, 'blocked_actions': sum(r['status'] == 'BLOCKED' for r in statuses),
                        'rejected_actions': sum(r['status'] == 'INVALID_ACTION' for r in statuses),
                        'unresolved_dependency_cases': sum(r['status'] == 'UNKNOWN_VALIDITY' for r in statuses),
                        'validity_rate': None, 'asr': None, 'defender_queries': 0,
                        'reason': statuses[0]['reason'] if statuses else 'No training fraud candidates',
                        'interpretation': 'No verified nonidentity attacks; rates not applicable; no evasion claim'})
    return results


def credit_and_bias_smoke(cfg, protocol):
    defender = FrozenDefender(ROOT/cfg['baseline_artifacts'])
    stream = TransactionStream(ROOT/cfg['training_parquet'], ROOT/cfg['baseline_splits'], defender.feature_names, cfg['regression_transactions'])
    sim_config = yaml.safe_load((ROOT/'configs/rl_environment.yaml').read_text())
    sim = FraudSimulator(stream, defender, sim_config)
    window = protocol['windows']['rl_development']
    collector = CourtCreditCollector(sim, window['max_time'], partition='defender_fit_mechanics_replay',
                                    gamma_per_day=cfg['gamma_per_day'], training_start=window['min_time'])
    rng = np.random.default_rng(44)
    exports = []
    for _ in range(len(stream)):
        obs = sim.observation()
        requested_probability = 1. if obs[3] == 1 else .005
        action = int(obs[3] == 1 or rng.random() < .005)
        result = collector.step(action)
        sim.actions[-1]['court_policy_request_probability'] = requested_probability
        exports.extend(result['ready_replay_records'])
    horizon = sim.summary()
    horizon_records = sim.memory.get_recent_verified(sim.now, sim.memory.memory_size())
    pending = {tid: due for due, tid, _ in sim.verification.internal_checkpoint()}
    collector.ledger.censor_pending()
    for event in sim.drain_pending():collector.ledger.complete(event, event['emitted_at'])
    # Mechanics replay is outside the new RL development window, so no training records.
    assert not exports
    records = collector.ledger.records()
    by_action = {r['court_action_id']: r for r in records}
    for row in sim.actions:
        origin = by_action[row['action_id']]
        assert origin['immediate_reward'] == row['immediate_court_reward']
    events = {e['transaction_id']: e for e in sim.events}
    observed_at_horizon = {r['transaction_id'] for r in horizon_records}
    selection = []
    for row in sim.actions:
        event = events.get(row['transaction_id'])
        selection.append({**row, 'due_timestamp': event['completion_time'] if event else None,
                          'verification_delay_seconds': event['completion_time']-event['request_time'] if event else None,
                          'status_at_stream_horizon': ('VERIFIED' if row['transaction_id'] in observed_at_horizon else 'CENSORED_PENDING') if event else 'NOT_INVESTIGATED',
                          'observed_label_at_horizon': event['verified_label'] if row['transaction_id'] in observed_at_horizon else None,
                          'post_drain_verified_label': event['verified_label'] if event else None,
                          'label_scope': 'evaluation_log; not agent input'})
    memory = sim.memory.get_recent_verified(sim.now, sim.memory.memory_size())
    bias = []
    for stage, rows in [('stream_horizon', horizon_records), ('post_drain_diagnostic', memory)]:
        for source in ('court', 'audit'):
            subset = [r for r in rows if r['source'] == source]
            bias.append({'stage': stage, 'source': source, 'verified_count': len(subset),
                         'verified_fraud': sum(r['verified_label'] for r in subset),
                         'observed_verified_fraud_rate': sum(r['verified_label'] for r in subset)/len(subset) if subset else None,
                         'evaluation_only_actual_prevalence': horizon['evaluation_only_fraud_count']/horizon['transactions_processed'],
                         'unbiased_population_estimate': False})
    write_csv(OUT/'court_credit_ledger.csv', records)
    write_csv(OUT/'audit_selection_log.csv', selection)
    write_csv(OUT/'selection_bias_summary.csv', bias)
    return {'horizon': horizon, 'after_drain': sim.summary(), 'attributed_investigations': sum(r['investigated'] for r in records),
            'training_records_exported': len(exports), 'note': 'Old in-sample mechanics replay is excluded by the RL partition/time gates.'}, bias


def markdown_documents(cfg, protocol, profile, test_results, gate_error):
    threat = '''# Attacker threat model

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
'''
    (OUT/'threat_model.md').write_text(threat)
    dependency = f'''# Dependency audit

All {len(json.loads((ROOT/cfg['baseline_artifacts']/'feature_schema.json').read_text())['features'])} cleaned predictors were inspected through their saved schema and revised fitting-window values.
The registry retains the original conservative classifications. C/D/V/id variables remain opaque;
ProductCD, card, addr, dist, email-domain, M and other identity fields are grouped by schema/name only.
No undocumented detailed semantics or deterministic relationships are asserted.

TransactionAmt type: {profile['arrow_type']}; fitting-window rows: {profile['rows']};
NULL count: {profile['null_count']}; positive: {profile['positive']}; zero: {profile['zero']}; negative: {profile['negative']}.
Quantiles (float64 diagnostic copies):

```json
{json.dumps(profile['quantiles'], indent=2)}
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
'''
    (OUT/'dependency_audit.md').write_text(dependency)
    credit = '''# Court delayed-credit protocol

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
'''
    (OUT/'court_credit_protocol.md').write_text(credit)
    decisions = f'''# Phase 3.5 decisions and readiness

## Executed
- Phase 3 regression tests: {test_results['phase3_regression']['tests']} PASS.
- New Phase 3.5 tests: {test_results['phase3_5']['tests']} PASS, including a temporary synthetic revised-Defender fitting pipeline test.
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
blocked by its provenance gate: {gate_error}
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
'''
    (OUT/'phase3_5_decisions.md').write_text(decisions)


def html_report(manifest, protocol, profile, feasibility_rows, synthetic_rows, bias):
    status = [{'component': 'Attacker PPO (IEEE-CIS)', 'readiness': 'BLOCKED', 'reason': 'No verified mutable controls/dependencies'},
              {'component': 'Court RL', 'readiness': 'PARTIAL', 'reason': 'Credit tests PASS; independent model/provenance and custom learner integration pending'},
              {'component': 'Defender integration', 'readiness': 'PARTIAL', 'reason': 'Frozen inference PASS; revised real fit blocked by historical feature selection'},
              {'component': 'Research evaluation', 'readiness': 'PARTIAL', 'reason': 'Exact temporal protocol validated; Original unavailable; no independent RL result'}]
    sections = []
    def section(title, content):sections.append(f'<section><h2>{html.escape(title)}</h2>{content}</section>')
    section('1. Context and readiness', table(status)+'<p>Phase 3.5 implemented and tested constraints, attribution and temporal guards. No PPO/DQN training was performed. Synthetic tests are not financial attack evidence.</p>')
    section('2. Existing Phase 3 regression', table([{'suite': k, **v} for k,v in manifest['tests'].items()])+f'<p>The 10,000-transaction identity replay reproduced the prior summary exactly. Protected artifacts and source Parquet hashes were unchanged. Regression reports are under <code>phase3_regression/</code>.</p>')
    section('3. Threat model', '<p>Limited knowledge, one terminal decision-only query, at most three modification steps. Scenario A authorizes no real mutable feature. Scenario B proposes a bounded ±5% amount sensitivity study, with explicit hypothetical intent preservation; unresolved dependencies prevent acceptance and scoring.</p><p><a href="threat_model.md">Full threat model</a> · <a href="phase3_5_config.yaml">Executed configuration</a></p>')
    section('4. Feature controllability and dependencies', f'<p>{manifest["cleaned_predictors"]} predictors remain UNKNOWN; 3 metadata fields IMMUTABLE. No classification was promoted using numeric type, correlation or importance. Original dataset: {manifest["original_status"]}.</p>'+table([{'statistic': k, 'value': json.dumps(v) if isinstance(v,dict) else v} for k,v in profile.items()])+ '<p><a href="feature_controllability_v2.csv">All inspected columns</a> · <a href="dependency_audit.md">Dependency evidence and limitations</a> · <a href="amount_correlations.csv">Numeric associations</a> · <a href="conditional_amount_profile.csv">Conditional plausibility diagnostics</a></p>')
    section('5. Continuous constraint engine', '<p>Control dimensions come from authorized config/registry entries, plus a STOP gate. Immutable metadata, original vectors and partition tags are preserved. Registered dependent outputs are recomputed; unknown rules fail closed. Bounds are rejected rather than silently clipped. Normalized cost measures accepted displacement from the original.</p><p>Statuses: VALID_UNCHANGED, VALID_MODIFIED, INVALID_ACTION, UNKNOWN_VALIDITY, BLOCKED. Blocked/unknown proposals make no Defender query. Agent observation includes current/original values and missing masks, budget, step and action history.</p>')
    section('6. Synthetic attacker execution', table(synthetic_rows)+'<p>Controlled quantity/price/total fixtures only. The evaluator-only original score is not visible to the agent. Identity cases do not contribute to ASR. Query counters distinguish the agent submission from the evaluator baseline query.</p>'+table([manifest['synthetic_metrics']]))
    section('7. IEEE-CIS feasibility', table(feasibility_rows)+'<p>Source is the revised Defender-fit prefix only. Conservative actions remain blocked; hypothetical amount proposals remain UNKNOWN_VALIDITY. No modified IEEE-CIS transaction was submitted to Defender, and no real-data ASR is applicable.</p>')
    section('8. Court credit assignment', '<p>Origin-linked pending records mature into replay entries only after outcome and next-state availability. Immediate cost, eventual bonus and attributed reward remain separate. No bonus is attached to an unrelated arrival action.</p><pre>r_origin = -cost + gamma_day ** ((completion - origin)/86400) * bonus\nTD_target = r_origin + gamma_day ** ((next_time - origin)/86400) * (1 - terminated) * Q_next</pre><p>The discounted return identity is tested. This does not prove Markov sufficiency of the existing compact observation. A custom cohort-aware/recurrent learner remains future work; vanilla PPO/DQN must not consume the legacy arrival-reward stream.</p><p><a href="court_credit_protocol.md">Derivation, alternatives and boundary rules</a> · <a href="court_credit_assignment_tests.csv">Executed numerical tests</a> · <a href="court_credit_ledger.csv">Real replay origin records</a></p>')
    section('9. Chronological protocol', table([{'window': n, **w} for n,w in protocol['windows'].items()])+f'<div class="note">Historical feature selection cutoff: {protocol["historical_feature_selection_cutoff"]}. A fresh earlier model alone cannot undo this provenance issue. The new fitting code rejected current provenance before fitting. Exact code path was tested on a synthetic dataset. No real revised-model results are claimed.</div><p>The 3-day suffix is reserved for maturation, with no training labels arriving across partitions. The previous final holdout was already evaluated by the baseline; this phase reads only ID/time metadata for partition checks and no new holdout labels or predictor values.</p><p><a href="temporal_split_protocol.json">Exact boundaries and ordered-ID hashes</a></p>')
    section('10. Audit design and selection bias', table(bias)+'<p>Logs contain action, eligibility, conditional audit selection probability, budget availability, score/decision, verification delay and horizon censorship. Court and audit samples are separated. Approval-only capped audits do not yield an unbiased population prevalence estimator. No IPW correction is claimed.</p><p><a href="audit_selection_log.csv">Selection and censoring log</a></p>')
    section('11. Executed checks and artifact inventory', '<pre>'+html.escape(json.dumps(manifest['tests'],indent=2))+'</pre><p>Regression and new tests include boundary-crossing labels, wrong-origin events, immutability, STOP/query limits, time discounts, delayed credit at step 100, temporal alignment and fit-only provenance.</p><pre>'+html.escape(json.dumps(manifest['discovered_artifacts'],indent=2))+'</pre>')
    section('12. Remaining blockers', '<p>Resolve a fitting-prefix-only feature set, fit/calibrate a new Defender before RL development, integrate the credit adapter with a cohort-aware and suitably history-aware learner, and establish defensible attacker controls. Original Parquet has a reusable export script but was not created or evaluated in this phase.</p>')
    section('13. Verdict and reproduction', '<p><strong>Not yet ready for full Phase 4 IEEE-CIS training.</strong> Mechanical checks pass; semantic and chronological readiness gates remain. Court-only training is not yet marked READY with the current artifacts.</p><p><a href="phase3_5_decisions.md">All decisions and Phase 4 prerequisites</a> · <a href="run_manifest.json">Execution manifest</a></p><pre>.venv_defender/bin/python scripts/run_phase3_5.py</pre><p>Primary references: <a href="https://www.kaggle.com/c/ieee-fraud-detection/data">competition data</a>, <a href="https://arxiv.org/abs/1806.07857">RUDDER return redistribution</a>, <a href="https://farama.org/Gymnasium-Terminated-Truncated-Step-API">termination versus truncation</a>. The implemented origin accounting is derived for this simulator, not asserted to be RUDDER.</p>')
    style = 'body{margin:0;background:#f2f5f4;color:#193b35;font:16px/1.6 system-ui}header{background:#193b35;color:white;padding:42px max(24px,calc((100vw - 1160px)/2))}main{max-width:1160px;margin:auto;padding:24px}section{background:white;padding:26px;border:1px solid #d7e2dd;border-radius:12px;margin:20px 0}h1{line-height:1.2;font-size:36px}h2{font-size:24px;margin-top:0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #d7e2dd;min-width:100px}th{background:#edf4f0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf4f0;padding:18px;font-size:12px}.note{background:#fff3d9;border-left:4px solid #bd8622;padding:18px}a{color:#086d5a}'
    (OUT/'phase3_5_readiness_report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 3.5 readiness</title><style>'+style+'</style></head><body><header><p>IEEE-CIS · EXECUTED READINESS AUDIT</p><h1>Threat model, constraints and delayed credit</h1><p>Overall PARTIAL · real-data attacker BLOCKED · no PPO/DQN training</p></header><main>'+''.join(sections)+'</main></body></html>')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT/'execution.log')])
    start = time.perf_counter()
    cfg = yaml.safe_load((ROOT/'configs/phase3_5.yaml').read_text())
    threat = yaml.safe_load((ROOT/cfg['threat_model']).read_text())
    protected = [p for folder in ['data/processed', 'reports/lightgbm_baseline', 'reports/rl_environment', 'feature_audit/reports']
                 for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    before = hashes(protected)
    tests = run_tests()
    logging.info('Regression tests passed; repeating Phase 3 simulation into a NEW report directory')
    regression = subprocess.run([sys.executable, 'scripts/run_rl_environment_smoke_test.py', '--output', str(OUT/'phase3_regression')],
                                cwd=ROOT, text=True, capture_output=True)
    (OUT/'phase3_regression_execution.log').write_text(regression.stdout+regression.stderr)
    if regression.returncode:raise RuntimeError('Phase 3 smoke regression failed')
    prior = pd.read_csv(ROOT/'reports/rl_environment/simulation_summary.csv')
    replay = pd.read_csv(OUT/'phase3_regression/simulation_summary.csv')
    pd.testing.assert_frame_equal(prior, replay)
    splits = json.loads((ROOT/cfg['baseline_splits']).read_text())
    audit = json.loads((ROOT/cfg['feature_selection_manifest']).read_text())
    protocol = build_protocol(ROOT/cfg['training_parquet'], splits, audit['fit_cutoff'])
    legacy = json.loads((ROOT/cfg['baseline_artifacts']/'training_config.json').read_text())
    metadata = pq.read_table(ROOT/cfg['training_parquet'], columns=['transactionid', 'transactiondt']).sort_by([
        ('transactiondt', 'ascending'), ('transactionid', 'ascending')])
    calibration_last_index = splits['train_end']+legacy['calibration_selection_end_in_calibration']-1
    protocol['existing_baseline_periods'] = {
        'model_and_category_fit_end': splits['model_training_cutoff'],
        'final_calibrator_refit_end': metadata['transactiondt'][calibration_last_index].as_py(),
        'empirical_threshold_selection_end': splits['prior_holdout_cutoff'],
        'historical_feature_selection_end': audit['fit_cutoff'],
        'fully_unseen_pre_holdout_rows_after_all_learned_choices': 0,
        'note': 'Prespecified threshold 1/6 avoids empirical threshold fitting, but not historical feature-selection look-ahead.'}
    write_json(OUT/'temporal_split_protocol.json', protocol)
    schema = json.loads((ROOT/cfg['baseline_artifacts']/'feature_schema.json').read_text())
    inspections = []
    for name in ['training_parquet', 'test_parquet']:
        file = pq.ParquetFile(ROOT/cfg[name])
        inspections.append({'path': cfg[name], 'rows': file.metadata.num_rows,
                            'columns': [{'name': f.name, 'type': str(f.type)} for f in file.schema_arrow]})
        file.close()
    write_json(OUT/'parquet_schema_inspection.json', inspections)
    train_schema = pq.read_schema(ROOT/cfg['training_parquet'])
    entries = audit_schema(train_schema)
    original_status = 'NOT AVAILABLE'
    if (ROOT/cfg['original_parquet']).exists():
        verify_alignment(ROOT/cfg['training_parquet'], ROOT/cfg['original_parquet'])
        entries += audit_schema(pq.read_schema(ROOT/cfg['original_parquet']), 'original')
        original_status = 'AVAILABLE; schema and alignment audited; no Original policy experiment'
    write_csv(OUT/'feature_controllability_v2.csv', entries)
    data, profile, correlations, conditional = amount_audit(ROOT/cfg['training_parquet'], schema, protocol['windows']['defender_fit']['max_time'])
    write_json(OUT/'amount_profile.json', profile)
    write_csv(OUT/'amount_correlations.csv', correlations)
    write_csv(OUT/'conditional_amount_profile.csv', conditional)
    write_json(OUT/'fit_window_null_counts.json', {n: data[n].null_count for n in data.column_names})
    feasible = feasibility(data, FeatureRegistry(entries[:len(train_schema)]), threat, cfg['feasibility_candidates'])
    synthetic, synthetic_metrics = synthetic_smoke(threat)
    write_csv(OUT/'attack_feasibility_summary.csv', feasible)
    write_csv(OUT/'synthetic_attack_results.csv', synthetic)
    cases = credit_checks();assert all(r['status'] == 'PASS' for r in cases)
    write_csv(OUT/'court_credit_assignment_tests.csv', cases)
    logging.info('Validating origin-linked credit and audit logging on the real mechanics replay')
    credit_smoke, bias = credit_and_bias_smoke(cfg, protocol)
    write_json(OUT/'credit_smoke_summary.json', credit_smoke)
    provenance = {'fit_cutoff': audit['fit_cutoff'], 'selected_predictors': [f['name'] for f in schema['features']],
                  'parquet_sha256': before[cfg['training_parquet']], 'source': cfg['feature_selection_manifest']}
    write_json(OUT/'current_feature_provenance.json', provenance)
    try:
        fit_new_defender(ROOT/cfg['training_parquet'], ROOT/cfg['baseline_artifacts']/'feature_schema.json',
                         OUT/'temporal_split_protocol.json', OUT/'current_feature_provenance.json', OUT/'new_defender')
    except ValueError as error:
        gate_error = str(error)
        if 'Feature selection' not in gate_error:raise
    else:
        raise AssertionError('Known future feature selection should have blocked fitting')
    assert not (OUT/'new_defender').exists()
    assert before == hashes(protected), 'Protected prior artifacts changed'
    manifest = {'completed_at': datetime.now(timezone.utc).isoformat(), 'runtime_seconds': time.perf_counter()-start,
                'overall_status': 'PARTIAL', 'tests': tests, 'credit_numerical_cases': len(cases),
                'cleaned_predictors': len(schema['features']), 'original_status': original_status,
                'real_defender_fit': 'BLOCKED_BY_PROVENANCE', 'gate_error': gate_error,
                'synthetic_metrics': synthetic_metrics, 'protected_artifacts_unchanged': True,
                'discovered_artifacts': {k: str(ROOT/v) for k,v in cfg.items() if isinstance(v,str) and (ROOT/v).exists()},
                'protected_sha256': before,
                'source_sha256': hashes(list((ROOT/'rl_environment').glob('*.py'))+list((ROOT/'tests/phase3_5').glob('*.py'))+
                    [Path(__file__).resolve(), ROOT/'scripts/fit_temporal_defender.py', ROOT/'scripts/export_original_parquet.py', ROOT/'configs/phase3_5.yaml', ROOT/cfg['threat_model']])}
    write_json(OUT/'run_manifest.json', manifest)
    (OUT/'phase3_5_config.yaml').write_text(yaml.safe_dump({'execution': cfg, 'threat_model': threat, 'temporal_protocol': protocol}, sort_keys=False))
    markdown_documents(cfg, protocol, profile, tests, gate_error)
    html_report(manifest, protocol, profile, feasible, synthetic, bias)
    logging.info('Completed Phase 3.5 checks: %s', OUT/'phase3_5_readiness_report.html')
    print(json.dumps({'tests': tests, 'credit_cases': len(cases), 'feasibility': feasible, 'gate_error': gate_error}, indent=2))


if __name__ == '__main__':main()
