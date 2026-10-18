"""Generate actual Mode B reports and verify all protected artifacts."""
import sys,json,html,datetime,platform,importlib.metadata,os
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/phase37-matplotlib')
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing.temporal_feature_audit import sha
from rl_environment.assumption_attack_engine import AssumptionAttackEngine

def main():
    out=ROOT/'reports/phase3_7'
    config=yaml.safe_load((out/'attack_constraints.yaml').read_text())
    tests=pd.read_csv(out/'phase3_7_test_results.csv');assert set(tests.status)=={'PASS'}
    protected=json.loads((out/'protected_before.json').read_text());assert all(sha(ROOT/p)==h for p,h in protected.items())
    results=pd.read_csv(out/'attack_results.csv',dtype={'original_amount':str,'modified_amount':str},float_precision='round_trip')
    summary=pd.read_csv(out/'attack_summary.csv');pairs=pd.read_csv(out/'paired_model_comparison.csv')
    execution=json.loads((out/'experiment_execution.json').read_text());smoke=json.loads((out/'hybrid_smoke_summary.json').read_text())
    plausibility=json.loads((out/'plausibility_model.json').read_text())
    checks=pd.read_csv(out/'plausibility_validation.csv')
    assert checks[['hard_finite_positive_decimal','hard_relative_budget','hard_non_target_and_metadata_unchanged']].all().all()
    original32=results.original_amount.map(lambda x:float(np.float32(Decimal(x))))
    modified32=results.modified_amount.map(lambda x:float(np.float32(Decimal(x))))
    numeric=results[['dataset_version','transactionid','epsilon','policy','original_amount','modified_amount']].copy()
    numeric['defender_original_amount_float32']=original32;numeric['defender_modified_amount_float32']=modified32
    numeric['original_absolute_conversion_error']=[float(abs(Decimal.from_float(f)-Decimal(a))) for f,a in zip(original32,results.original_amount)]
    numeric['modified_absolute_conversion_error']=[float(abs(Decimal.from_float(f)-Decimal(a))) for f,a in zip(modified32,results.modified_amount)]
    numeric['decimal_modified_but_float32_unchanged']=results.status.eq('ASSUMPTION_VALID')&(original32==modified32)
    numeric.to_csv(out/'numeric_representation_validation.csv',index=False,float_format='%.17g')
    doc='''# Assumption-based attacker state, action and reward

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
'''
    (out/'attacker_state_action_reward.md').write_text(doc)
    readiness=f'''# Phase 4 attacker readiness

**Mode B: READY_FOR_ASSUMPTION_BASED_PPO**

The declared Box(1) task, Decimal transform, hard constraints, fixed-prefix plausibility, terminal decision-only query budget, reward, paired actual-data experiments and delayed-verification integration are validated. {len(tests)} tests PASS, including all 73 earlier regression tests. No PPO was trained.

**STRICT: BLOCKED. Real-world causal attack validity: NOT ESTABLISHED.**

No anonymous feature is promoted to causally mutable. Holding non-amount fields fixed and fraud intent unchanged are explicit counterfactual assumptions. Statistical plausibility does not establish realizability.

Next: implement the development-only source sampler and PPO training orchestration with fixed seeds/budgets; compare against the saved identity/random/directional references and a one-step contextual-bandit baseline. Keep the default epsilon=.05 and lambda=.1 unless a new predeclared development protocol is recorded. Reward/epsilon sensitivity is not a license to optimize on final holdout. Evaluate a frozen policy on RL Validation with separate denominators and one query per episode.

Court remains PARTIAL, and continual Defender updating is not implemented. Keep the Phase3.6 delayed-credit semantics; this phase only validates rule-based end-to-end integration. Prior temporal Defender/label-availability assumptions remain unchanged. Final holdout was historically baseline-evaluated, but no new Validation or Holdout feature/label data were loaded for these experiments.
'''
    (out/'phase4_attacker_readiness.md').write_text(readiness)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,version in zip(axes,['original_431','cleaned_prefix_430']):
        for policy in ['random_uniform','fixed_decrease','fixed_increase']:
            s=summary[(summary.dataset_version==version)&(summary.policy==policy)].sort_values('epsilon')
            ax.plot(s.epsilon*100,s.conditional_evasion_rate*100,marker='o',label=policy)
        ax.set(title=version,xlabel='Predeclared epsilon (%)',xticks=[1,3,5]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Conditional evasion (%)');axes[1].legend(fontsize=8);fig.suptitle('ASSUMPTION-BASED results — accepted, originally detected fraud only');fig.tight_layout();fig.savefig(out/'conditional_evasion.png',dpi=150);plt.close(fig)
    default=results[(results.dataset_version=='original_431')&(results.policy=='random_uniform')&(results.epsilon==.05)]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].hist(default.modification_percentage,bins=30,color='#217e9b');axes[0].set(xlabel='Actual amount change (%)',ylabel='Episodes',title='Default seeded random, epsilon5%')
    axes[1].hist(default.probability_change,bins=35,color='#ac752d');axes[1].set(xlabel='Calibrated probability change',ylabel='Episodes',title='Evaluator-only score changes');fig.tight_layout();fig.savefig(out/'default_diagnostics.png',dpi=150);plt.close(fig)
    sections=[]
    def section(title,body):sections.append('<section><h2>'+html.escape(title)+'</h2>'+body+'</section>')
    def table(frame):return frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:.6f}',na_rep='NOT APPLICABLE')
    default_summary=summary[(summary.policy=='random_uniform')&(summary.epsilon==.05)]
    section('Mode B and readiness','<p><strong>READY_FOR_ASSUMPTION_BASED_PPO</strong> for the declared bounded computational task. STRICT remains BLOCKED. Real-world transaction manipulability and causal label preservation have not been established. No PPO training was run.</p><p>Only TransactionAmt changes. All other fields remain fixed under an explicit approximation; dependencies remain UNKNOWN_DEPENDENCY. Results below are ASSUMPTION-BASED IEEE-CIS RESULTS.</p>')
    section('Execution and chronological provenance',f'<p>Source fraud: {execution["source_fraud"]:,}; eligible fraud: {execution["eligible_fraud"]:,}; episode rows: {execution["episodes"]:,} across 2 models ×3 epsilons ×4 policies. Repeated sources across separate reference experiments are not unique fraud counts. No Validation or Final Holdout tuning/evaluation was performed.</p><p>Plausibility fits exactly 265,743 Defender Fit rows; exact IDs/time metadata are verified. Fresh temporal Original431 and Cleaned430 models are frozen. All prior protected files remain byte-identical.</p>')
    section('Action, state and reward','<p>Box(1), max3 steps; absolute original_amount*(1+epsilon*action). Default epsilon5%, sensitivity1/3/5%. STOP deadband1e-6 retains current candidate. One terminal decision-only query. 11 causal observation fields; no evaluator probabilities or unreleased labels. Lambda cost0.1; valid APPROVE+1−cost, FLAG−1−cost, invalid−2−accepted cost, identity0. This is a bounded multi-step interface without iterative Defender feedback.</p>')
    section('Decimal representation',f'<p>Actual source: {execution["amount_arrow_type"]}; exact Decimal proposals rounded HALF_EVEN to .001 within the budget grid. Overflow, nonfinite and nonpositive inputs reject. Saved model preprocessing uses float32. Maximum measured modified-amount conversion error: {numeric.modified_absolute_conversion_error.max():.9g}. Decimal-nonidentity episodes that become identical at float32 input: {int(numeric.decimal_modified_but_float32_unchanged.sum())}. Conversion diagnostics are saved explicitly.</p>')
    section('Default random epsilon5% results',table(default_summary[['dataset_version','source_fraud_transactions','eligible_fraud','attack_attempts','accepted_nonidentity','unchanged','assumption_acceptance_rate','originally_detected_fraud','accepted_originally_detected_denominator','successful_decision_flips','conditional_evasion_rate','overall_accepted_attack_evasion_rate','max_query_count']])+'<p>Conditional rate measures newly evaded detections. Overall approved fraction includes fraud already approved before modification. Identity rows never enter either attack-success denominator.</p><img src="default_diagnostics.png" alt="Actual perturbations and score changes">')
    section('All predeclared sensitivities',table(summary[['dataset_version','epsilon','policy','accepted_nonidentity','rejected','original_flag_rate','modified_flag_rate','accepted_originally_detected_denominator','successful_decision_flips','conditional_evasion_rate','overall_accepted_attack_evasion_rate','mean_probability_change','mean_cost','mean_reward']])+'<img src="conditional_evasion.png" alt="Conditional evasion for all predefined epsilons"><p>No best action, direction or epsilon was selected retrospectively. Each cell is a separate predeclared single-query reference experiment.</p>')
    section('Hard constraints and soft plausibility',f'<p>Hard finite/positive/Decimal/budget/metadata checks all PASS on executed episode outputs. {int(checks.modified_outside_q01_q99.sum()):,} episode outputs lie outside fit q01–q99; {int(checks.modified_outside_fit_range.sum()):,} lie outside the full fit amount range. These are repeated episode counts, not unique sources. They remain soft rarity diagnostics under the predeclared acceptance policy, not evidence of impossibility.</p>'+table(pd.DataFrame([plausibility['global_statistics']]))+'<p>Conditional ProductCD/card4/card6 distributions require≥200 fitting rows. Robust global and conditional log-MAD scores, original/modified anomalies and support flags are in plausibility_validation.csv. No statistical filter was loosened after examining attack results.</p>')
    section('Original versus Cleaned paired measurement',table(pairs)+'<p>Same source IDs, actions, epsilon and modified amounts were checked explicitly. Probability equality is a measured result for these episodes; removal of fitting-prefix-constant v107 was not assumed to guarantee equality.</p>')
    smoke_table=pd.DataFrame([dict(version=s['dataset_version'],transactions=s['mature']['transactions_processed'],source_fraud=s['source_fraud'],modified=s['mature']['modified_attack_examples'],court=s['mature']['court_investigation_count'],audit=s['mature']['audit_investigation_count'],verified_fraud=s['mature']['fraud_verified'],verified_legitimate=s['mature']['legitimate_verified'],verified_modified=s['verified_modified'],pending_after_drain=s['mature']['investigations_pending'],attacker_queries=s['attacker_queries']) for s in smoke])
    section('Hybrid chronological integration',table(smoke_table)+'<p>Frozen Defender, rule-based Court, separate180+20 budgets, seeded Uniform(1,3)-day verification. Completed-only memory links source/modified vectors, model decision, Court action, verification result and assumed label contract. Pending labels are withheld. Explicit diagnostic draining stays within Development and is not policy training.</p>')
    section('Executed tests',f'<p>{len(tests)} tests PASS, including73 prior regression tests. Covers arithmetic, precision/overflow, STOP/step/query budgets, identity exclusions, exact fit provenance/future contamination, actual saved-model Gym inference, paired models, label isolation, delayed memory provenance and artifact integrity.</p>'+table(tests[['test','status','seconds']]))
    section('Limitations and Phase 4','<pre>'+html.escape(readiness)+'</pre><p>These simple reference policies provide a baseline, not evidence that a trained PPO improves evasion. Absolute amount edits may violate unknown real dependencies. Fraud intent preservation and offline historical label availability remain assumptions. No full Court learner or continual Defender retraining is implemented.</p>')
    links=['attack_results.csv','attack_summary.csv','attacker_state_action_reward.md','attack_constraints.yaml','plausibility_validation.csv','numeric_representation_validation.csv','phase3_7_test_results.csv','phase4_attacker_readiness.md','paired_model_comparison.csv','hybrid_smoke_summary.json','plausibility_model.json','run_manifest.json']
    section('Files and reproduction','<ul>'+''.join(f'<li><a href="{f}">{f}</a></li>' for f in links)+'</ul><pre>.venv_defender/bin/python scripts/run_phase3_7.py\n.venv_defender/bin/python scripts/test_phase3_7.py\n.venv_defender/bin/python scripts/report_phase3_7.py</pre>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 3.7 — Assumption-Based Attacker</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#24333e}h1,h2{color:#165a75}section{overflow:auto;margin:30px 0}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8dd;padding:7px;text-align:left}th{background:#edf4f6}pre{white-space:pre-wrap;background:#f4f7f8;padding:16px;font:14px ui-monospace}img{max-width:100%}</style><h1>Phase 3.7 — Assumption-Based Continuous Attacker</h1>'+''.join(sections)+'</html>'
    (out/'phase3_7_attack_report.html').write_text(page)
    sources=[p for folder in ('rl_environment','scripts','tests/phase3_7','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')]
    modelhashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models/temporal_defender').rglob('*') if p.is_file()}
    inventory={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
    manifest=dict(started_at=(out/'started_at.txt').read_text(),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),config_sha256=sha(ROOT/'configs/assumption_attack.yaml'),protocol_sha256=sha(ROOT/'reports/phase3_6/temporal_split_protocol.json'),source_parquet=str(ROOT/'data/processed/train_original.parquet'),source_parquet_sha256=sha(ROOT/'data/processed/train_original.parquet'),dataset_versions=['original_431','cleaned_prefix_430'],models=modelhashes,seeds={'attack':42,'court':42,'audit':42,'verification':43},python=sys.version,platform=platform.platform(),versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','pyarrow','lightgbm','gymnasium')},source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},artifact_inventory=inventory,tests={'PASS':len(tests)},protected_artifacts_unchanged=True,validation_and_holdout_experiments=False,full_ppo_training=False,mode_b_readiness='READY_FOR_ASSUMPTION_BASED_PPO',strict_readiness='BLOCKED',real_world_causal_validity=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Reports ready: {len(tests)} tests PASS; {len(results)} episode rows; protected artifacts unchanged')
if __name__=='__main__':main()
