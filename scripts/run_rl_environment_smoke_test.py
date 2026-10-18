"""Execute Phase 3 tests and a reproducible, training-only mechanics simulation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
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
import yaml
from rl_environment.defender import FrozenDefender
from rl_environment.transaction_stream import TransactionStream
from rl_environment.fraud_simulator import FraudSimulator
from rl_environment.court_env import CourtEnv


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashes(paths):
    results = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(1024*1024):
                digest.update(chunk)
        results[str(path.relative_to(ROOT))] = digest.hexdigest()
    return results


def controllability(names):
    rows = []
    for name in names:
        if name in ('transactionid', 'transactiondt', 'isfraud'):
            category, reason = 'immutable', 'Technical ID, chronological metadata or hidden target; never mutable or a predictor.'
        elif name == 'transactionamt':
            category, reason = 'unknown', 'Amount controllability, preservation of fraud intent/label, and dependencies are not established. No real perturbation allowed.'
        elif re.match(r'^(v\d+|c\d+|d\d+|id_\d+)$', name):
            category, reason = 'unknown', 'Anonymized semantics and dependencies unavailable; no independent edits permitted.'
        else:
            category, reason = 'unknown', 'No documented attacker control or validated dependency/label-preservation rule; frozen in this phase.'
        rows.append({'feature': name, 'classification': category, 'perturbation_allowed': False,
                     'reason': reason, 'evidence_scope': 'IEEE-CIS cleaned schema; no proven real-data transformation'})
    return rows


def run_episode(simulator, config):
    env = CourtEnv(simulator)
    observation, _ = env.reset(seed=config['seed'])
    rng = np.random.default_rng(config['seed']+2)
    flag_index = simulator.observations.names.index('flag')
    total_reward = 0.
    for step in range(len(simulator._records)):
        if not env.observation_space.contains(observation):
            raise AssertionError('Nonfinite or wrong-shape agent observation')
        # Public signal only. No labels, future events, records or evaluations used.
        action = int(observation[flag_index] == 1 or rng.random() < config['court']['random_investigation_probability'])
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 2000 == 1999:
            logging.info('Processed %s transactions; pending=%s', step+1, info['pending_count'])
        if terminated or truncated:
            break
    horizon = {'stage': 'stream_horizon', **simulator.summary()}
    assert np.isclose(total_reward, horizon['court_policy_reward'])
    events_at_horizon = len(simulator.events)
    simulator.drain_pending()
    drained = {'stage': 'explicit_post_horizon_drain', **simulator.summary()}
    assert drained['investigations_pending'] == 0
    assert drained['memory_size'] == drained['investigation_budget_used']
    assert all(e['emitted_at'] >= e['completion_time'] >= e['request_time']+86400 for e in simulator.events)
    assert all(e['completion_time'] <= e['request_time']+3*86400 for e in simulator.events)
    assert all(r['court_used'] <= simulator.budget.court_limit and r['audit_used'] <= simulator.budget.reserve
               for r in simulator.budget.history+[simulator.budget.snapshot()])
    return [horizon, drained], events_at_horizon


def table(rows):
    if not rows:
        return '<p>No records.</p>'
    names = list(rows[0])
    header = ''.join('<th>'+html.escape(n)+'</th>' for n in names)
    body = ''.join('<tr>'+''.join('<td>'+html.escape('NOT APPLICABLE' if r.get(n) is None else str(r[n]))+'</td>' for n in names)+'</tr>' for r in rows)
    return '<div class="scroll"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def report(out, config, manifest, results, test_log, obs_names):
    h, d = results
    comparisons = [{'metric': k, 'stream_horizon': v, 'post_horizon_drain': d[k]} for k, v in h.items() if k != 'stage']
    parts = [f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 3 · Fraud simulation environment</title><style>
body{{margin:0;background:#f1f5f4;color:#173b35;font:16px/1.6 system-ui,sans-serif}}header{{background:#173b35;color:white;padding:40px max(24px,calc((100vw - 1120px)/2))}}h1{{line-height:1.2;font-size:36px}}main{{max-width:1120px;margin:auto;padding:24px}}section{{background:white;border:1px solid #d8e3df;border-radius:12px;padding:26px;margin:20px 0}}h2{{margin-top:0}}.badge{{display:inline-block;background:#ffdf9d;color:#654500;padding:8px 15px;border-radius:8px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}.card{{padding:20px;border:1px solid #d8e3df;border-radius:10px}}.card strong{{display:block;font-size:30px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d8e3df}}th{{background:#ecf4f0}}pre{{overflow:auto;font-size:12px;background:#edf3f0;padding:15px}}.note{{padding:18px;background:#fff5df;border-left:4px solid #b88223}}svg{{width:100%;height:auto}}a{{color:#09695a}}code{{background:#edf3f0;padding:2px 4px}}</style></head><body>
<header><p>IEEE-CIS · PHASE 3 · EXECUTED MECHANICS TEST</p><h1>Hybrid adversarial fraud simulation</h1><span class="badge">Overall: PARTIAL — real-data attacker blocked by threat model</span><p>{html.escape(manifest['completed_at'])}</p></header><main>
<div class="grid"><div class="card">Unit tests<strong>{manifest['tests_run']} passed</strong></div><div class="card">Transactions<strong>{h['transactions_processed']:,}</strong></div><div class="card">Investigations<strong>{h['investigation_budget_used']}</strong></div><div class="card">Predictors<strong>{manifest['predictors']}</strong></div></div>
<section><h2>1. Architecture and status</h2><p>The transaction stream, frozen calibrated Defender, Court actions, separate investigation budgets, delayed verification, and verified-only memory executed successfully. No PPO, DQN or Defender retraining was performed.</p>
<svg viewBox="0 0 1040 175" role="img" aria-label="Chronological stream passes through identity attacker, frozen Defender and Court, then delayed verification and verified memory"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#176b62"/></marker></defs>
<g fill="#edf5f1" stroke="#176b62"><rect x="5" y="10" width="180" height="55" rx="7"/><rect x="220" y="10" width="180" height="55" rx="7"/><rect x="435" y="10" width="180" height="55" rx="7"/><rect x="650" y="10" width="180" height="55" rx="7"/><rect x="650" y="110" width="180" height="55" rx="7"/><rect x="385" y="110" width="220" height="55" rx="7"/></g>
<g fill="#173b35" font-size="15" text-anchor="middle"><text x="95" y="43">Chronological stream</text><text x="310" y="43">Identity attacker</text><text x="525" y="43">Frozen Defender</text><text x="740" y="43">Court + budget</text><text x="740" y="143">Delayed verification</text><text x="495" y="143">Verified-only memory</text></g>
<g stroke="#176b62" stroke-width="2" marker-end="url(#arrow)"><path d="M185,38 H215"/><path d="M400,38 H430"/><path d="M615,38 H645"/><path d="M740,65 V105"/><path d="M650,137 H610"/></g></svg>
<p>Continuous attacker adapter: <strong>synthetic tests PASS; IEEE-CIS rollout BLOCKED BY THREAT MODEL</strong>. Original 431 Parquet: <strong>{manifest['original_dataset_status']}</strong>. No Original simulation is claimed.</p></section>
<section><h2>2. Data and frozen model</h2>{table([manifest['dataset']])}<p>Selected records are the earliest chronological transactions in the saved model-training partition, ordered by TransactionDT then TransactionID. All timestamps are ≤ {manifest['boundaries']['model_training_cutoff']}; the calibration period and final holdout remain excluded from policy development. Original NULLs remain intact.</p>
<p>Calibrator: {manifest['calibration_method']}; threshold: <code>p &gt; {manifest['threshold']}</code>. Saved feature order, categorical mappings and preprocessing are reused without fitting. Model version: <code>{manifest['model_version']}</code>.</p>
<div class="note">This is an in-sample development replay. The frozen baseline was trained on the larger model-training partition, including transactions later than some replayed transactions. This run verifies simulator causality; it is not a historically valid online backtest, an independent Defender evaluation, or evidence of absence of model-level look-ahead in this replay. No policy quality claim is made. A later online evaluation requires a model fitted strictly before its evaluation stream.</div></section>
<section><h2>3. Observations and action spaces</h2><p>Court uses a {len(obs_names)}-element float32 vector and Discrete(2): 0 SKIP, 1 INVESTIGATE.</p><pre>{html.escape(json.dumps(obs_names, indent=2))}</pre>
<p>Permitted numeric observables use fixed tanh(sign(x)·log(1+|x|)/10) plus a missing-value bit; no transformation statistics are fitted. This is an agent observation encoding, not a modification of Defender inputs. Pending counts and verified counts use n/(1+n); elapsed time uses t/(86400+t). Recent verification statistics use at most 100 already completed records. The update-time field refers to simulator deployment of the frozen model; there are no updates.</p>
<p>Internal labels, transaction identities, full simulator records, pending labels and evaluation metrics never enter agent observations. Batched inference caches independent frozen-model scores internally; only the current score is exposed.</p>
<p>Synthetic continuous attacker: Box([-1,1], shape=(2,)): one proven toy quantity control plus one submission gate. Quantity remains in [1,10], unit price is immutable, total is recomputed, and total displacement from the original quantity is capped at 2. The fixture defines fraud as intent independent of quantity. This assumption is not transferred to IEEE-CIS. Positive stop gate submits; otherwise submission occurs at step 3. One terminal Defender query per episode, decision-only feedback; no intermediate score feedback. The ten-dimensional observation holds original/current quantity, remaining budget, step and padded action history.</p></section>
<section><h2>4. Controllability audit</h2><p>All {manifest['dataset']['columns']} source columns were audited: metadata is immutable; the {manifest['predictors']} predictors remain unknown/unproven for attacker control. There are zero authorized real-data mutable features. Anonymous V/C/D/id features cannot be edited independently. TransactionAmt requires evidence of controllability, fraud-label preservation and consistent dependent features. No real dependency graph is fabricated. Mutable and dependent classifications are demonstrated only in the separately identified synthetic fixture.</p><p><a href="feature_controllability_report.csv">Full feature controllability audit</a>. Real records pass unchanged through identity_attack. Attack validity and Attack Success Rate are NOT APPLICABLE, not zero-percent attack results.</p></section>
<section><h2>5. Rewards, budgets and delayed labels</h2><p>Court receives −1 once per accepted investigation; at completion it receives +10 for verified fraud previously APPROVED, +3 for verified fraud previously FLAGGED, and no bonus for legitimate transactions. SKIP and blocked investigation actions cost zero. Audit costs and bonuses are logged separately and are not Court policy rewards. Defender business reward (+10/−20/−5/+1) is evaluation-only, distinct from Court reward.</p>
<p>Every 10,000 arrivals receive 200 tokens: 180 Court and 20 reserved audits. Court cannot spend the reserve. Audits use a seeded online Bernoulli probability 20/10,000 on APPROVED transactions not already investigated by Court, capped at 20 per window. Unused tokens expire at rollover. The cap is per full window, not a guarantee of a 2% rate for every short prefix. Pending investigations survive budget rollover.</p>
<p>Verification delays are continuous Uniform(86,400, 259,200) seconds, simulated rather than observed. Due results are released before the next transaction's observation; the step reward contains the associated completion events, each with its originating transaction/action ID. No past reward is retroactively edited. Time-aware discounting and delayed credit assignment still require design before PPO training.</p>
<p>The stream horizon truncates the Gym episode and retains pending records as censored. An explicit diagnostic drain then advances to each completion timestamp without processing additional transactions. Drain rewards are reported separately and are not injected into the truncated rollout. Reset archives pending internal checkpoints before starting an independent episode.</p></section>
<section><h2>6. Executed smoke-test results</h2><p>Rule-based Court: investigate a Defender FLAG or a seeded 0.5% random exploration draw, subject to budget. The same seed was replayed and action/event logs and summaries matched exactly. This policy was not tuned on these results.</p>{table(comparisons)}<p>Post-horizon verification is distinguished from labels available during the stream. Verified Memory begins empty and includes only requested transactions with completed verification. The reported fraud count uses internal labels for evaluation only.</p></section>
<section><h2>7. Tests and leakage checks</h2><pre>{html.escape(test_log)}</pre>
{table([{'check': k, 'status': v} for k,v in manifest['checks'].items()])}
<p>Raw input Parquet and all baseline artifacts were SHA-256 checked before and after execution and were unchanged. No database connection is opened by this phase. Full source/output provenance is in <a href="run_manifest.json">run_manifest.json</a>.</p></section>
<section><h2>8. Limitations and next research decisions</h2><ul><li>Real attack transformations and valid ASR are blocked pending a defensible threat model and dependency constraints.</li><li>Verified Memory is selected by Court and capped approval audits. It is not a representative random sample; selection bias remains, and no inverse-propensity correction is claimed.</li><li>Reward values and delay distributions are research assumptions, not measured financial loss or investigation duration.</li><li>Pending checkpoints are retained within the process; durable restart/recovery and continual model updates are not implemented.</li><li>Gym compatibility and smoke tests do not establish policy effectiveness. Delayed-credit handling, variable-time discounting and independent temporal evaluation must be settled before formal PPO training.</li><li>The Python module boundary separates observations from internal state by API convention, not by a security sandbox against malicious agent code.</li></ul></section>
<section><h2>Files and reproducibility</h2><p><a href="simulation_summary.csv">Simulation summary</a> · <a href="environment_config.json">Executed config</a> · <a href="feature_controllability_report.csv">Controllability</a> · <a href="actions.csv">Action log</a> · <a href="verification_events.csv">Verification events</a></p><pre>.venv_defender/bin/python scripts/run_rl_environment_smoke_test.py</pre><p>Gymnasium API follows the <a href="https://gymnasium.farama.org/main/introduction/create_custom_env/">official custom-environment guidance</a>. Results above come from this executed run; synthetic tests are explicitly separate from IEEE-CIS smoke results.</p></section></main></body></html>''']
    (out/'environment_validation_report.html').write_text(''.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rl_environment.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/rl_environment')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out/'execution.log')])
    start = time.perf_counter()
    config = yaml.safe_load(args.config.read_text())
    selected = config['datasets'][config['dataset_version']]
    parquet, artifacts = ROOT/selected['parquet'], ROOT/selected['artifacts']
    if not parquet.exists():
        raise FileNotFoundError(f'{config["dataset_version"]} dataset NOT AVAILABLE: {parquet}')
    baseline_files = list((ROOT/'reports/lightgbm_baseline/artifacts').glob('*')) + list((ROOT/'reports/lightgbm_baseline/experiments/A_original').glob('*'))
    protected = [p for p in baseline_files if p.is_file()] + list((ROOT/'data/processed').glob('*.parquet'))
    before = hashes(protected)
    logging.info('Running assertion-based unit and real Defender integration tests')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/rl_environment', '-v'],
                           cwd=ROOT, capture_output=True, text=True)
    test_log = tests.stdout+tests.stderr
    (out/'unit_tests.log').write_text(test_log)
    if tests.returncode:
        (out/'environment_validation_report.html').write_text('<h1>FAIL: unit tests</h1><pre>'+html.escape(test_log)+'</pre>')
        raise RuntimeError('Unit tests failed; see unit_tests.log')
    tests_run = int(re.search(r'Ran (\d+) tests', test_log).group(1))
    defender = FrozenDefender(artifacts, config['threshold_policy'])
    logging.info('Loading chronological model-training records only')
    stream = TransactionStream(parquet, ROOT/config['boundaries'], defender.feature_names, limit=config['transactions'])
    audit = controllability(stream.schema_names)
    write_csv(out/'feature_controllability_report.csv', audit)
    simulator = FraudSimulator(stream, defender, config)
    results, events_at_horizon = run_episode(simulator, config)
    actions, events = list(simulator.actions), list(simulator.events)
    logging.info('Replaying the same seed to verify identical actions, delays and rewards')
    repeated, repeated_events = run_episode(simulator, config)
    assert repeated == results and actions == simulator.actions and events == simulator.events
    assert events_at_horizon == repeated_events
    after = hashes(protected)
    assert before == after, 'Protected source/model artifacts changed'
    boundaries = json.loads((ROOT/config['boundaries']).read_text())
    manifest = {'completed_at': datetime.now(timezone.utc).isoformat(), 'overall_status': 'PARTIAL',
                'tests_run': tests_run, 'predictors': len(defender.feature_names),
                'model_version': defender.version, 'threshold': defender.threshold,
                'calibration_method': defender._baseline.config['calibration_method'],
                'original_dataset_status': 'NOT AVAILABLE' if not (ROOT/config['datasets']['original']['parquet']).exists() else ('EXECUTED' if config['dataset_version'] == 'original' else 'AVAILABLE, NOT EXECUTED'),
                'dataset': {'path': str(parquet.relative_to(ROOT)), 'source_rows': stream.source_rows,
                            'columns': len(stream.schema_names), 'selected_rows': len(stream),
                            'first_time': simulator._records[0].transaction_time,
                            'last_time': simulator._records[-1].transaction_time,
                            'source_null_values': sum(stream.null_counts.values())},
                'boundaries': boundaries, 'events_at_stream_horizon': events_at_horizon,
                'events_after_explicit_drain': len(events)-events_at_horizon,
                'runtime_seconds': time.perf_counter()-start,
                'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'pyarrow', 'gymnasium', 'PyYAML', 'lightgbm', 'scikit-learn']},
                'checks': {'unit_tests': 'PASS', 'real_frozen_defender': 'PASS', 'schema_and_null_preservation': 'PASS',
                           'training_only_stream': 'PASS', 'future_label_isolation_in_environment': 'PASS',
                           'verification_timing_and_memory': 'PASS', 'reward_attribution': 'PASS',
                           'budget_rollover_and_reserve': 'PASS', 'fixed_seed_replay': 'PASS',
                           'baseline_and_parquet_unchanged': 'PASS', 'synthetic_continuous_attacker': 'PASS',
                           'real_data_attacker': 'PARTIAL: BLOCKED BY THREAT MODEL',
                           'independent_online_performance': 'NOT EVALUATED; in-sample development replay'},
                'protected_file_sha256': before,
                'source_sha256': hashes(list((ROOT/'rl_environment').glob('*.py')) + list((ROOT/'tests/rl_environment').glob('*.py')) + [Path(__file__).resolve(), args.config.resolve()])}
    config['resolved_paths'] = {'parquet': str(parquet), 'artifacts': str(artifacts), 'output': str(out)}
    config['saved_boundaries'] = boundaries
    config['court_observation_names'] = simulator.observations.names
    config['agent_transform'] = 'fixed analytic signed-log/tanh + missing bit; no fitted statistics'
    write_json(out/'environment_config.json', config)
    write_json(out/'run_manifest.json', manifest)
    write_json(out/'source_null_counts.json', stream.null_counts)
    write_csv(out/'simulation_summary.csv', results)
    write_csv(out/'actions.csv', actions)
    write_csv(out/'verification_events.csv', events)
    report(out, config, manifest, results, test_log, simulator.observations.names)
    logging.info('COMPLETE: %s tests PASS; real-data attacker PARTIAL; report=%s', tests_run, out/'environment_validation_report.html')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
