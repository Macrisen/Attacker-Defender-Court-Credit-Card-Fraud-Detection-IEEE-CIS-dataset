"""Build a self-contained HTML research report from executed model outputs."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score,roc_auc_score

def build_report(out):
    out=Path(out);plots=out/'figures';plots.mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'model_metrics.csv');thresholds=pd.read_csv(out/'threshold_comparison.csv')
    dataset=pd.read_csv(out/'dataset_summary.csv');importance=pd.read_csv(out/'feature_importance.csv')
    group=pd.read_csv(out/'feature_group_importance.csv');temporal=pd.read_csv(out/'temporal_performance.csv')
    calibration=pd.read_csv(out/'calibration_comparison.csv');selection=pd.read_csv(out/'calibration_selection.csv')
    runtime=pd.read_csv(out/'runtime_metrics.csv');comparisons=pd.read_csv(out/'experiment_comparison.csv')
    config=json.loads((out/'artifacts/defender_config.json').read_text())
    training=json.loads((out/'artifacts/training_config.json').read_text())
    run=json.loads((out/'run_manifest.json').read_text())
    primary=thresholds[thresholds.experiment=='B_clean'];method=config['calibration_method']
    theoretical=primary[primary.threshold_policy=='theoretical_reward_1_6'].iloc[0]
    p=np.load(out/'plot_data.npz');y=p['labels'];raw=p['raw_probability'];chosen=p['calibrated_probability']
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    palette=['#176b62','#db9345','#616f9a']
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for prob,label,color in [(raw,'Raw',palette[1]),(chosen,f'Selected: {method}',palette[0])]:
        precision,recall,_=precision_recall_curve(y,prob);fpr,tpr,_=roc_curve(y,prob)
        ax[0].plot(recall,precision,label=f'{label} · AP {average_precision_score(y,prob):.4f}',color=color)
        ax[1].plot(fpr,tpr,label=f'{label} · AUC {roc_auc_score(y,prob):.4f}',color=color)
    ax[0].axhline(y.mean(),color='gray',linestyle=':',label=f'Fraud prevalence {y.mean():.3%}')
    ax[1].plot([0,1],[0,1],':',color='gray')
    ax[0].set(xlabel='Fraud recall',ylabel='Precision',title='Precision–recall · untouched holdout',xlim=(0,1),ylim=(0,1))
    ax[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC · untouched holdout',xlim=(0,1),ylim=(0,1))
    for a in ax:a.legend(fontsize=8,loc='lower left')
    fig.savefig(plots/'ranking_curves.png');plt.close(fig)
    bins=pd.read_csv(out/'reliability_bins.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for a in axes:
        a.plot([0,1],[0,1],':',color='gray',label='Perfect calibration')
        for i,(name,rows) in enumerate(bins.groupby('method',sort=False)):
            a.plot(rows.mean_probability,rows.fraud_rate,'o-',markersize=4,label=name,color=palette[i])
        a.set(xlabel='Mean predicted probability',ylabel='Observed fraud rate');a.legend(fontsize=8)
    axes[0].set(title='Reliability · 15 fixed equal-width bins',xlim=(0,1),ylim=(0,1))
    axes[1].set(title='Low-probability region (same bins)',xlim=(0,.3),ylim=(0,.3))
    fig.savefig(plots/'calibration.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,row in zip(axes,primary.itertuples()):
        cm=np.array([[row.tn,row.fp],[row.fn,row.tp]])
        shade=np.log1p(cm)
        ax.imshow(shade,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if shade[i,j]>shade.min()+.6*(shade.max()-shade.min()) else '#19302c',fontsize=12)
        ax.set_xticks([0,1],['APPROVE','FLAG']);ax.set_yticks([0,1],['Legitimate','Fraud'])
        ax.set(title=f'{row.threshold_policy}\np > {row.threshold:.6f}',xlabel='Defender action',ylabel='Actual label')
    fig.savefig(plots/'confusion_matrices.png');plt.close(fig)
    top=importance.head(30).iloc[::-1];fig,ax=plt.subplots(figsize=(10,9),layout='constrained')
    ax.barh(top.feature,top.gain_importance,color=palette[0]);ax.set(xlabel='LightGBM gain',title='Top 30 features · cleaned Defender')
    fig.savefig(plots/'feature_importance.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(primary));width=.35
    axes[0].bar(x-width/2,primary.mean_reward,width,label='Realized reward',color=palette[0])
    axes[0].bar(x+width/2,primary.mean_expected_reward,width,label='Probability-based expected reward',color=palette[1])
    axes[0].set_xticks(x,['0.5','1/6','Calibration-selected']);axes[0].set(ylabel='Reward per transaction',title='Expected vs realized reward · holdout');axes[0].legend(fontsize=8)
    drift=temporal[temporal.threshold_policy=='theoretical_reward_1_6']
    axes[1].plot(drift.window,drift.average_precision,'o-',label='Average precision',color=palette[0])
    axes[1].plot(drift.window,drift.recall,'o-',label='Fraud recall at 1/6',color=palette[1])
    axes[1].plot(drift.window,drift.fraud_rate,'o-',label='Fraud prevalence',color=palette[2])
    axes[1].set(xticks=drift.window,xlabel='Successive chronological holdout window',ylabel='Rate / score',title='Temporal performance');axes[1].legend(fontsize=8)
    fig.savefig(plots/'reward_and_time.png');plt.close(fig)
    def image(name):
        data=base64.b64encode((plots/name).read_bytes()).decode()
        return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{data}">'
    def table(df,columns=None):
        return '<div class="table-wrap">'+df[columns].to_html(index=False,escape=True,float_format=lambda v:f'{v:.6f}',na_rep='—',border=0)+'</div>' if columns else '<div class="table-wrap">'+df.to_html(index=False,escape=True,float_format=lambda v:f'{v:.6f}',na_rep='—',border=0)+'</div>'
    splitcols=['partition','rows','predictor_count','min_transactiondt','max_transactiondt','fraud_rows','fraud_rate','missing_predictor_percentage']
    pmcols=['experiment','probability_method','threshold_policy','threshold','average_precision','roc_auc','precision','recall','f1','fpr','tnr']
    mainmetrics=metrics[(metrics.experiment=='B_clean')&(metrics.probability_method==method)]
    cal=calibration[calibration.experiment=='B_clean'];sel=selection[selection.experiment=='B_clean']
    retained=out/'retained_correlated_features.csv';ncorrelated=len(pd.read_csv(retained)) if retained.exists() else 0
    best=primary.loc[primary.mean_reward.idxmax()]
    hold_count=len(y);feature_count=int(dataset.loc[dataset.partition=='model_training','predictor_count'].iloc[0])
    cards=[('Holdout transactions',f'{hold_count:,}'),('Predictors',str(feature_count)),
           ('Average precision',f'{theoretical.average_precision:.4f}'),('ROC-AUC',f'{theoretical.roc_auc:.4f}'),
           ('Recall at 1/6',f'{theoretical.recall:.2%}'),('Mean reward at 1/6',f'{theoretical.mean_reward:.4f}')]
    kpis=''.join(f'<div class="kpi"><span>{label}</span><strong>{value}</strong></div>' for label,value in cards)
    decisioncols=['threshold_policy','threshold','tp','fp','tn','fn','flag_rate','mean_reward','total_reward','mean_expected_reward','total_expected_reward']
    comparemetrics=metrics[(metrics.selected_calibrator==True)&(metrics.threshold_policy=='theoretical_reward_1_6')]
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IEEE-CIS · LightGBM Defender baseline</title><style>
:root{{color-scheme:light;--ink:#18342f;--muted:#5f716d;--line:#dce7e2;--accent:#176b62}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f3;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}
header{{background:#123b34;color:white;padding:48px max(24px,calc((100vw - 1180px)/2));}}header p{{color:#d5e8df;max-width:850px}}h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:12px 0}}h2{{font-size:26px;margin:0 0 14px}}h3{{font-size:19px}}main{{max-width:1240px;margin:auto;padding:24px}}
.tag{{font-size:12px;letter-spacing:.13em;text-transform:uppercase}}.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}}.kpi,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px}}.kpi span{{display:block;font-size:13px;color:var(--muted)}}.kpi strong{{font-size:30px}}section{{margin:20px 0}}p{{max-width:1050px}}.note{{border-left:4px solid #d59a49;background:#fff8ed;padding:16px}}.table-wrap{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#edf4ef;color:#284d43}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto;margin:12px 0}}code{{background:#edf4ef;padding:2px 5px;border-radius:4px}}pre{{overflow:auto;background:#edf4ef;padding:18px;font-size:12px}}a{{color:var(--accent)}}.small{{font-size:13px;color:var(--muted)}}footer{{padding:20px;color:var(--muted);font-size:13px}}@media(max-width:650px){{.kpis{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}}}
</style></head><body><header><div class="tag">Research baseline · executed experiments · IEEE-CIS</div><h1>LightGBM Defender</h1><p>Chronological fraud detection with independently selected probability calibration and a configurable reward-based decision policy. Attacker, Court and reinforcement-learning components are not implemented.</p><p class="small" style="color:#d5e8df">Generated {html.escape(run['completed_at'])} · Seed 42 · PostgreSQL source tables accessed read-only</p></header><main>
<div class="kpis">{kpis}</div>
<section><h2>1. Dataset and chronological protocol</h2><p>Source tables: <code>fraud_detection.public.train_clean</code> and <code>public.test_clean</code>. Predictors are discovered from the live schema; identifiers, time and target are excluded. No database rows, columns or values were changed.</p>{table(dataset,splitcols)}
<p>The previous holdout boundary is preserved exactly: <code>TransactionDT ≤ 12192842</code> belongs to the pre-holdout period. Sorting is by time then ID; identical timestamps never cross a boundary. Rows labeled “inner”, “calibrator” and “selection” are nested subsets, not additional transactions.</p>
<ol><li>Within the first approximately 65%, use an earlier 85% for inner fitting and the later 15% for early stopping by average precision. Inner encoders are fitted only on inner-fit rows.</li><li>Refit the chosen number of trees and final categorical mappings on the entire model-training period. Calibration and holdout are excluded.</li><li>Inside the calibration period, fit candidate calibrators on its first half and select the method by log loss on its next quarter (Brier breaks ties).</li><li>Refit each calibrator on the first three quarters of calibration. Select the reward-maximizing threshold on the last quarter. Then freeze everything before holdout evaluation.</li></ol>
<div class="note">The prior feature-selection audit used the earliest 80%, including the present calibration period. The final 20% remains untouched by that selection, but calibration-period model-selection estimates are conditional on the preselected feature set and are not a completely independent feature-selection assessment.</div></section>
<section><h2>2. Model and feature preparation</h2><p>Primary Defender: experiment B (current clean tables). Numerical NULLs remain missing; there is no imputation, scaling, oversampling, class weighting or outlier deletion. Text fields, card/address codes and id_12–id_38 are categorical under a conservative metadata policy. V/C/D numeric measurements and id_01–id_11 remain numerical; anonymized semantics still require domain review.</p><p>Categorical vocabularies are learned only from model-training rows. Missing and unseen categories map to −1 for LightGBM's native missing handling. Numerical predictors use float32 consistently at training and inference; original SQL values are unchanged.</p><pre>{html.escape(json.dumps(training['parameters'],indent=2))}</pre><p>Selected iterations: <strong>{training['selected_iterations']}</strong> out of {training['max_rounds']} maximum. Early-stopping patience: {training['early_stopping_patience']}. This is a reproducible baseline, not a broad hyperparameter search.</p></section>
<section><h2>3. Final holdout discrimination</h2>{table(mainmetrics,pmcols)}{image('ranking_curves.png')}<p>Average precision is the primary PR summary; trapezoidal PR-AUC is also saved in CSV and is not treated as interchangeable with AP. Threshold-dependent metrics explicitly identify their probability method and policy.</p></section>
<section><h2>4. Probability calibration</h2><p>Selected before accessing holdout: <strong>{html.escape(method)}</strong>. Selection criterion: smallest log loss on the calibration-method-selection block, with Brier score as tie-break.</p><h3>Pre-holdout method selection</h3>{table(sel,['method','average_precision','brier_score','log_loss','ece_15_equal_width'])}<h3>Frozen calibrators evaluated on holdout</h3>{table(cal,['method','selected_before_holdout','average_precision','roc_auc','brier_score','log_loss','ece_15_equal_width'])}{image('calibration.png')}
<p>ECE uses 15 fixed equal-width bins over [0,1], weighted by bin population; empty bins contribute zero. Binning can hide local miscalibration. Isotonic calibration may introduce probability ties and alter ranking metrics. Holdout calibration results are reported for comparison only and do not change the selected method.</p></section>
<section><h2>5. Defender actions and realized reward</h2><p>Flag: +10 for fraud, −5 for legitimate; approve: −20 for fraud, +1 for legitimate. Expected rewards are <code>15p−5</code> and <code>1−21p</code>, yielding <code>FLAG iff p &gt; 1/6</code>; ties are approved.</p>{table(primary,decisioncols)}{image('confusion_matrices.png')}{image('reward_and_time.png')}
<p>The empirical threshold maximizes mean reward on the final calibration quarter, with ties preferring fewer alerts and no investigation-capacity constraint. It is not optimized on holdout. The theoretical 1/6 threshold is optimal only under the stated reward matrix and adequately calibrated probabilities. Expected–realized differences measure a combination of calibration error and sampling/time variation.</p></section>
<section><h2>6. Chronological holdout windows</h2>{table(temporal[temporal.threshold_policy=='theoretical_reward_1_6'],['window','rows','min_time','max_time','fraud_rate','average_precision','roc_auc','recall','fpr','mean_reward'])}<p>These successive windows describe temporal variation. Changes alone do not establish concept drift or its cause; no window is used to refit the model or threshold.</p></section>
<section><h2>7. Feature importance</h2>{image('feature_importance.png')}{table(group)}<p>Gain importance reflects this fitted model, not causality. Permutation importance was evaluated for the top 30 gain features, with three repeats on up to 5,000 evenly spaced pre-holdout calibration-selection rows. See <a href="permutation_importance.csv">permutation_importance.csv</a>. It is descriptive and does not trigger further removal.</p><p>{ncorrelated} pairs flagged by the earlier correlation audit remain jointly present. Correlated predictors can share or substitute importance; low marginal permutation importance is not proof of irrelevance.</p></section>
<section><h2>8. Original versus cleaned feature experiments</h2>{table(comparisons)}{table(comparemetrics,['experiment','probability_method','average_precision','roc_auc','brier_score','log_loss','recall','fpr','mean_reward'])}<p>Shared original/clean values and row IDs were checked before reusing data for experiment A. All experiments use the same chronological protocol; each chooses iterations and calibration within pre-holdout data. B remains the designated primary baseline regardless of final holdout comparison. C is not rerun when its candidate removals are already absent.</p></section>
<section><h2>9. Runtime, inference and reproducibility</h2>{table(runtime)}<p>Batch prediction time measures raw LightGBM holdout inference; single-transaction latency includes schema checks, feature encoding and calibration after loading artifacts, using two warm-ups and ten measurements. Peak RSS is the process-wide cumulative high-water mark, not an isolated per-experiment memory delta.</p><p>The <code>artifacts/</code> directory contains the booster, selected calibrator, categorical mappings, feature order/types, threshold policy, training configuration and pinned dependencies. <code>defender.inference</code> provides <code>predict_fraud_probability(transaction)</code> and <code>defender_decision(transaction, threshold=1/6)</code>. Explicit NULLs and unseen categories are supported; missing predictor keys or unexpected fields are rejected.</p><p>Saved-model predictions were checked against in-memory holdout predictions, including unknown-category and missing-value cases. Detailed execution logs, source hashes, database snapshot identifier and a hash of the extracted training snapshot are retained. Credentials come from libpq environment variables/.pgpass or a local .env, and are never saved in model artifacts.</p></section>
<section><h2>10. Conclusions and limitations</h2><p>The cleaned baseline achieved holdout AP <strong>{theoretical.average_precision:.4f}</strong> and ROC-AUC <strong>{theoretical.roc_auc:.4f}</strong>. At the prespecified 1/6 threshold it recovered <strong>{theoretical.recall:.2%}</strong> of fraud, incorrectly flagged <strong>{int(theoretical.fp):,}</strong> legitimate transactions, and approved <strong>{int(theoretical.fn):,}</strong> fraudulent transactions, with mean reward <strong>{theoretical.mean_reward:.4f}</strong>.</p><p>This is a saved, reproducible offline Defender baseline suitable for integration testing and subsequent research. It is not evidence of robustness against adaptive attackers, production readiness, or optimal investigation allocation. No RL environment, Attacker, Court, continual update loop, delayed labels or counterfactual action outcomes have been modeled.</p><ul><li>Validation is one chronological holdout; uncertainty intervals and repeated temporal backtests have not been estimated.</li><li>Calibration and thresholds can drift. Monitor probability quality, recall, false positives and reward over time.</li><li>Anonymized feature availability at decision time remains unverified. Unknown semantics are not automatically treated as leakage.</li><li>Numeric-coded categories and missing-as-unknown policy need domain validation. Test labels are unavailable and no test accuracy is claimed.</li><li>The historical Kaggle test period is later than the labeled period. Schema compatibility does not guarantee performance under that shift.</li><li>Reward values are research assumptions; no real monetary loss or operational capacity is represented.</li></ul></section>
<section><h2>Files and references</h2><p><a href="dataset_summary.csv">Dataset summary</a> · <a href="model_metrics.csv">Model metrics</a> · <a href="threshold_comparison.csv">Threshold comparison</a> · <a href="feature_importance.csv">Feature importance</a> · <a href="temporal_performance.csv">Temporal performance</a> · <a href="predictions_holdout.csv">Holdout predictions</a></p><p><a href="https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html">LightGBM early stopping</a> · <a href="https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html">LightGBM missing and categorical handling</a> · <a href="https://scikit-learn.org/stable/modules/calibration.html">scikit-learn calibration guidance</a></p></section>
</main><footer>Computed from executed PostgreSQL/LightGBM experiments. No hypothetical performance values.</footer></body></html>'''
    (out/'lightgbm_baseline_report.html').write_text(body)

if __name__=='__main__':
    import sys
    build_report(Path(sys.argv[1]) if len(sys.argv)>1 else Path('reports/lightgbm_baseline'))
