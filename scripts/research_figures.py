"""Figures from saved observations only; no smoothing or invented data."""
import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/fraud-research-matplotlib')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def training_figures(out):
    a=pd.read_csv(ROOT/'reports/phase4a/attacker_training_metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for (version,seed,policy),g in a.groupby(['version','seed','policy']):
        if version=='original' and policy=='ppo':
            axes[0].plot(g.agent_interactions,g.mean_episode_reward,label=str(seed));axes[1].plot(g.agent_interactions,g.actor_loss);axes[2].plot(g.agent_interactions,g.value_loss)
    for ax,title in zip(axes,['Attacker episode reward','PPO actor loss','Critic loss']):ax.set(title=title,xlabel='Actual interactions');ax.grid(alpha=.2)
    axes[0].legend(title='Seed');fig.tight_layout();fig.savefig(out/'attacker_training.png',dpi=150);plt.close(fig)
    inventory=json.loads((ROOT/'reports/phase4a/attacker_checkpoints.json').read_text());rows=[]
    for run in inventory:
        for cp in run['checkpoints']:rows.append(dict(version=run['version'],seed=run['seed'],policy=run['policy'],steps=int(Path(cp['checkpoint']).stem.split('_')[-1]),conditional_evasion=cp['conditional_evasion']))
    frame=pd.DataFrame(rows);frame.to_csv(out/'checkpoint_evasion_curve.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for (version,seed,policy),g in frame.groupby(['version','seed','policy']):
        if version=='original' and policy=='ppo':ax.plot(g.steps,g.conditional_evasion*100,marker='o',label=str(seed))
    ax.set(title='Frozen validation checkpoints (selection data)',xlabel='Training interactions',ylabel='Conditional evasion (%)');ax.legend(title='Seed');fig.tight_layout();fig.savefig(out/'attacker_conditional_evasion.png',dpi=150);plt.close(fig)
    c=pd.read_csv(ROOT/'reports/phase4b/court_training_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(13,4))
    for (version,seed),g in c.groupby(['version','seed']):
        if version=='original':
            axes[0].plot(g.decision_interactions,g.court_policy_reward,label=str(seed));axes[1].plot(g.decision_interactions,g.fraud_found_per_completed_investigation);axes[2].plot(g.decision_interactions,g.court_investigation_count/180)
    for ax,title in zip(axes,['Court mature episode reward','Verified-fraud investigation fraction','Court budget utilization']):ax.set(title=title,xlabel='Decision interactions');ax.grid(alpha=.2)
    axes[0].legend(title='Seed');fig.tight_layout();fig.savefig(out/'court_training.png',dpi=150);plt.close(fig)
    cl=pd.read_csv(ROOT/'reports/phase5/continual_comparison.csv');val=cl[(cl.version=='original')&(cl.window=='rl_validation_frozen_after_development')]
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(val.scenario,val.AP);axes[0].set(title='Continual strategy: frozen validation AP',ylabel='Average precision');axes[1].bar(val.scenario,val.mean_business_utility);axes[1].set(title='Frozen validation utility',ylabel='Utility / transaction');fig.tight_layout();fig.savefig(out/'continual_replay_comparison.png',dpi=150);plt.close(fig)
    forgetting=pd.read_csv(ROOT/'reports/phase5/forgetting_analysis.csv');fig,ax=plt.subplots(figsize=(9,4))
    for (version,scenario),g in forgetting.groupby(['version','scenario']):
        if version=='original' and len(g)>1:ax.plot(g.deployment_time/86400,g.AP,marker='o',label=scenario)
    ax.set(title='Historical reference excluded from all updates',xlabel='Deployment time (relative day)',ylabel='AP on fixed historical reference');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'continual_through_time.png',dpi=150);plt.close(fig)

def final_figures(frame,paths,out):
    from sklearn.metrics import precision_recall_curve,roc_curve
    from sklearn.calibration import calibration_curve
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for experiment in ('E1','E6','E8'):
        path=next(p for p in paths if '/original/' in str(p) and f'/{experiment}_seed42/' in str(p));f=pd.read_parquet(path);y=f.label_evaluation_only;p=f.probability
        precision,recall,_=precision_recall_curve(y,p);fpr,tpr,_=roc_curve(y,p);observed,predicted=calibration_curve(y,p,n_bins=10)
        axes[0].plot(recall,precision,label=experiment);axes[1].plot(fpr,tpr,label=experiment);axes[2].plot(predicted,observed,marker='o',label=experiment)
    axes[0].set(title='Reserved evaluation PR',xlabel='Recall',ylabel='Precision');axes[1].set(title='ROC',xlabel='FPR',ylabel='TPR');axes[2].set(title='Calibration',xlabel='Mean predicted p',ylabel='Observed fraud fraction');axes[2].plot([0,1],[0,1],'k--')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'defender_pr_roc_calibration.png',dpi=150);plt.close(fig)
    g=frame.groupby(['version','experiment']).end_to_end_utility.mean().unstack(0)
    ax=g.plot.bar(figsize=(10,4));ax.set(ylabel='Mean total research utility',title='Reserved evaluation: end-to-end utility');plt.tight_layout();plt.savefig(out/'end_to_end_original_cleaned.png',dpi=150);plt.close()
