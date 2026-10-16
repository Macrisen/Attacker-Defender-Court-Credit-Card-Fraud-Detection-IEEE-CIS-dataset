"""Read-only PostgreSQL training, nested chronological calibration, and reporting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/defender_matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','4')
import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import platform
import resource
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,
    log_loss,confusion_matrix,precision_recall_curve,roc_curve,auc)
from .inference import NUMERIC,feature_role,fit_mappings,encode,calibrate,Defender

ROOT=Path(__file__).resolve().parents[1]
META={'transactionid','transactiondt','isfraud'}
SEED=42
NULL_SENTINEL='__PG_FRAUD_NULL_9c729c__'

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v),allow_nan=False))

def csv(frame,path):frame.to_csv(path,index=False,float_format='%.17g')

def frame_query(conn,query,params=None):
    with conn.cursor() as cursor:
        cursor.execute(query,params)
        return pd.DataFrame(cursor.fetchall(),columns=[c.name for c in cursor.description])

def db_connection():
    load_dotenv(ROOT/'.env',override=False)
    # libpq reads PGUSER/PGPASSWORD/PGHOST/PGPORT and .pgpass securely.
    conn=psycopg.connect(dbname=os.getenv('PGDATABASE','fraud_detection'),application_name='ieee_cis_defender_readonly')
    conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
    return conn

def split_index(times,fraction):
    if len(times)<2:raise ValueError('Too few rows for chronological split')
    requested=max(1,min(len(times)-1,int(np.floor(len(times)*fraction))))
    cutoff=times[requested-1]
    stop=int(np.searchsorted(times,cutoff,side='right'))
    if stop==len(times):raise ValueError('Cannot split without splitting identical timestamps')
    return stop,float(cutoff)

def ece(y,p,bins=15):
    ids=np.minimum((np.asarray(p)*bins).astype(int),bins-1)
    result=0.;rows=[]
    for b in range(bins):
        mask=ids==b;n=int(mask.sum())
        if not n:continue
        mean_p=float(np.mean(p[mask]));rate=float(np.mean(y[mask]))
        result+=n/len(y)*abs(mean_p-rate)
        rows.append({'bin':b,'lower':b/bins,'upper':(b+1)/bins,'count':n,'mean_probability':mean_p,'fraud_rate':rate})
    return float(result),rows

def probability_metrics(y,p):
    precision,recall,_=precision_recall_curve(y,p)
    return {'average_precision':float(average_precision_score(y,p)),
        'pr_auc_trapezoidal':float(auc(recall,precision)),
        'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
        'brier_score':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
        'ece_15_equal_width':ece(y,p)[0]}

def decision_metrics(y,p,threshold):
    flag=p>threshold
    tn,fp,fn,tp=confusion_matrix(y,flag,labels=[0,1]).ravel()
    reward=np.where(flag,np.where(y==1,10,-5),np.where(y==1,-20,1))
    expected=np.where(flag,15*p-5,1-21*p)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {'threshold':float(threshold),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
        'precision':float(precision),'recall':float(recall),'f1':float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
        'fpr':float(fp/(fp+tn)) if fp+tn else 0.,'tnr':float(tn/(fp+tn)) if fp+tn else 0.,
        'flagged_count':int(flag.sum()),'flag_rate':float(flag.mean()),
        'mean_reward':float(reward.mean()),'total_reward':int(reward.sum()),
        'mean_expected_reward':float(expected.mean()),'total_expected_reward':float(expected.sum())}

def best_reward_threshold(y,p):
    order=np.argsort(-p,kind='stable');scores=p[order];labels=y[order]
    increments=np.where(labels==1,30,-6)
    gains=np.cumsum(increments,dtype=np.int64)
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    ends=ends[scores[ends]>0]
    best_gain=0;threshold=1.
    # Descending thresholds: equal reward keeps fewer flags.
    for idx in ends:
        if gains[idx]>best_gain:
            best_gain=int(gains[idx]);threshold=float(np.nextafter(scores[idx],0.))
    measured=decision_metrics(y,p,threshold)
    baseline=int(np.where(y==1,-20,1).sum())
    assert measured['total_reward']==baseline+best_gain
    return threshold

def fit_calibrator(method,p,y):
    if method=='raw':return {'method':'raw','model':None}
    if method=='sigmoid':
        clipped=np.clip(p,1e-12,1-1e-12)
        model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=SEED)
        model.fit(np.log(clipped/(1-clipped)).reshape(-1,1),y)
    else:
        model=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p,y)
    return {'method':method,'model':model}

def load_database(out):
    warnings=[];cache=out/'cache';cache.mkdir(exist_ok=True)
    with db_connection() as conn:
        metadata=frame_query(conn,"SELECT table_name,column_name,ordinal_position,data_type FROM information_schema.columns WHERE table_schema='public' AND table_name IN ('train_clean','test_clean','train_merge') ORDER BY table_name,ordinal_position")
        csv(metadata,out/'database_schema.csv')
        train_meta=metadata[metadata.table_name=='train_clean'];test_meta=metadata[metadata.table_name=='test_clean']
        if train_meta.empty or test_meta.empty:raise ValueError('Required public.train_clean/public.test_clean table missing')
        lower={c.lower():c for c in train_meta.column_name}
        if not META<=set(lower):raise ValueError('Missing ID, time, or training label columns')
        idcol,timecol,labelcol=[lower[c] for c in ['transactionid','transactiondt','isfraud']]
        train_features=[c for c in train_meta.column_name if c.lower() not in META]
        expected_test=set(train_features)|{idcol,timecol}
        if set(test_meta.column_name)!=expected_test:
            warnings.append({'severity':'BLOCKER','issue':'Train/test columns inconsistent','detail':str(set(test_meta.column_name)^expected_test)})
        types=dict(zip(train_meta.column_name,train_meta.data_type));testtypes=dict(zip(test_meta.column_name,test_meta.data_type))
        for c in expected_test:
            if c in testtypes and testtypes[c]!=types[c]:warnings.append({'severity':'BLOCKER','issue':'Data type mismatch','detail':c})
        if warnings:
            csv(pd.DataFrame(warnings),out/'schema_warnings.csv');raise ValueError('Schema inconsistencies; see schema_warnings.csv')
        integrity=[];missing=[]
        for table in ['train_clean','test_clean']:
            label=sql.SQL('count(*) FILTER(WHERE {} IS NULL OR {} NOT IN(0,1))').format(sql.Identifier(labelcol),sql.Identifier(labelcol)) if table=='train_clean' else sql.SQL('NULL::bigint')
            query=sql.SQL('SELECT count(*) AS rows,count({id})-count(DISTINCT {id}) AS duplicate_ids,count(*) FILTER(WHERE {id} IS NULL) AS missing_id,count(*) FILTER(WHERE {tm} IS NULL) AS missing_time,{lbl} AS invalid_labels,min({tm}) AS min_time,max({tm}) AS max_time FROM public.{tbl}').format(id=sql.Identifier(idcol),tm=sql.Identifier(timecol),lbl=label,tbl=sql.Identifier(table))
            row=frame_query(conn,query).iloc[0].to_dict();row['table']=table;integrity.append(row)
            if any(row[k]>0 for k in ['duplicate_ids','missing_id','missing_time']) or (table=='train_clean' and row['invalid_labels']>0):
                csv(pd.DataFrame(integrity),out/'integrity.csv');raise ValueError('Invalid IDs, timestamps or labels; no rows excluded silently')
            columns=train_meta if table=='train_clean' else test_meta
            expressions=[]
            for i,r in enumerate(columns.itertuples()):
                expressions.append(sql.SQL('count(*) FILTER(WHERE {} IS NULL) AS {}').format(sql.Identifier(r.column_name),sql.Identifier('m'+str(i))))
                if r.data_type in NUMERIC:
                    expressions.append(sql.SQL("count(*) FILTER(WHERE {}::text IN ('NaN','Infinity','-Infinity')) AS {}").format(sql.Identifier(r.column_name),sql.Identifier('bad'+str(i))))
            counts=frame_query(conn,sql.SQL('SELECT {} FROM public.{}').format(sql.SQL(',').join(expressions),sql.Identifier(table))).iloc[0]
            for i,r in enumerate(columns.itertuples()):
                invalid=int(counts.get('bad'+str(i),0))
                missing.append({'table':table,'feature':r.column_name,'postgres_type':r.data_type,'rows':int(row['rows']),
                    'missing_count':int(counts['m'+str(i)]),'missing_percentage':100*int(counts['m'+str(i)])/int(row['rows']),
                    'nonfinite_count':invalid,'role':'metadata_or_target' if r.column_name.lower() in META else feature_role(r.column_name,r.data_type)})
                if invalid:raise ValueError(f'Nonfinite numeric values in {table}.{r.column_name}; explicit policy needed')
        csv(pd.DataFrame(integrity),out/'integrity.csv');csv(pd.DataFrame(missing),out/'missing_values.csv')
        # Compare original shared columns exactly before reusing them for experiment A.
        orig=metadata[metadata.table_name=='train_merge'];extra=[];original_available=False
        if not orig.empty and set(train_meta.column_name)<=set(orig.column_name):
            original_count=frame_query(conn,'SELECT count(*) AS n,count(DISTINCT transactionid) AS u FROM public.train_merge').iloc[0]
            if int(original_count.n)==int(integrity[0]['rows']) and original_count.n==original_count.u:
                shared=list(train_meta.column_name)
                left=sql.SQL(',').join(sql.Identifier('c',c) for c in shared)
                right=sql.SQL(',').join(sql.Identifier('o',c) for c in shared)
                q=sql.SQL('SELECT count(*) AS mismatched_rows FROM public.train_clean c FULL JOIN public.train_merge o ON c.{id}=o.{id} WHERE c.{id} IS NULL OR o.{id} IS NULL OR ROW({left}) IS DISTINCT FROM ROW({right})').format(id=sql.Identifier(idcol),left=left,right=right)
                mismatches=int(frame_query(conn,q).iloc[0,0]);original_available=mismatches==0
                if not original_available:warnings.append({'severity':'WARNING','issue':'Original comparison unavailable','detail':f'{mismatches} shared rows/values differ; experiment A skipped'})
        if original_available:
            extra=[c for c in orig.column_name if c not in set(train_meta.column_name)]
            for r in orig.itertuples():types[r.column_name]=r.data_type
        else:warnings.append({'severity':'WARNING','issue':'Experiment A unavailable','detail':'Original table missing or not safely aligned; no invented A result'})
        selection=sql.SQL(',').join([sql.Identifier('c',c) for c in train_meta.column_name]+[sql.Identifier('o',c) for c in extra])
        q=sql.SQL('SELECT {} FROM public.train_clean c').format(selection)
        if extra:q+=sql.SQL(' LEFT JOIN public.train_merge o ON c.{}=o.{}').format(sql.Identifier(idcol),sql.Identifier(idcol))
        q+=sql.SQL(' ORDER BY c.{},c.{}').format(sql.Identifier(timecol),sql.Identifier(idcol))
        path=cache/'training_snapshot.csv'
        with conn.cursor() as cursor,path.open('wb') as stream:
            copy=sql.SQL('COPY ({}) TO STDOUT WITH (FORMAT CSV, HEADER true, NULL {})').format(q,sql.Literal(NULL_SENTINEL))
            with cursor.copy(copy) as blocks:
                for data in blocks:stream.write(data)
        snapshot=frame_query(conn,'SELECT txid_current_snapshot()::text AS snapshot,version() AS postgres_version').iloc[0].to_dict()
    csv(pd.DataFrame(warnings,columns=['severity','issue','detail']),out/'schema_warnings.csv')
    dtypes={c:('float32' if t in NUMERIC else 'object') for c,t in types.items() if c in list(train_meta.column_name)+extra}
    dtypes.update({idcol:'int64',timecol:'int64',labelcol:'int8'})
    df=pd.read_csv(path,dtype=dtypes,keep_default_na=False,na_values=[NULL_SENTINEL])
    # Snapshot is temporary, not a second persistent dataset export.
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    snapshot['csv_sha256']=digest.hexdigest();snapshot['captured_at']=datetime.now(timezone.utc).isoformat()
    path.unlink()
    assert len(df)==int(integrity[0]['rows'])
    return df,train_features,extra,types,integrity,missing,snapshot,original_available,(idcol,timecol,labelcol)

def experiment(name,df,features,types,bounds,meta,args,out):
    start=time.perf_counter();idcol,timecol,labelcol=meta
    directory=out/('artifacts' if name=='B_clean' else 'experiments/'+name)
    directory.mkdir(parents=True,exist_ok=True)
    schema={'features':[{'name':c,'postgres_type':types[c],'role':feature_role(c,types[c])} for c in features],
        'metadata':list(meta),'numeric_dtype':'float32','numeric_missing':'native NaN',
        'categorical_policy':'Train-only sorted vocabulary; missing and unknown -> -1; native LightGBM categorical handling',
        'required_input':'All predictor keys must be supplied; explicit null is allowed; metadata is optional'}
    cats=[i for i,f in enumerate(schema['features']) if f['role']=='categorical']
    inner_end,train_end,hold_start=bounds['inner_end'],bounds['train_end'],bounds['hold_start']
    y=df[labelcol].to_numpy(dtype=np.int8)
    params={'objective':'binary','metric':'average_precision','learning_rate':.03,'num_leaves':31,
        'seed':SEED,'verbosity':-1,'num_threads':args.threads,'deterministic':True,'force_col_wise':True,
        'feature_fraction':1.,'bagging_fraction':1.,'lambda_l2':1.,'min_data_in_leaf':50,
        'max_bin':255,'feature_pre_filter':False,'first_metric_only':True}
    logging.info('%s: inner chronological early stopping, %d predictors',name,len(features))
    maps_inner=fit_mappings(df.iloc[:inner_end],schema)
    x_inner,_=encode(df.iloc[:inner_end],schema,maps_inner)
    x_valid,_=encode(df.iloc[inner_end:train_end],schema,maps_inner)
    trainset=lgb.Dataset(x_inner,label=y[:inner_end],feature_name=features,categorical_feature=cats)
    validset=lgb.Dataset(x_valid,label=y[inner_end:train_end],reference=trainset,feature_name=features,categorical_feature=cats)
    tune_start=time.perf_counter()
    tuned=lgb.train(params,trainset,num_boost_round=args.max_rounds,valid_sets=[validset],valid_names=['inner_chronological_validation'],
        callbacks=[lgb.early_stopping(args.patience,first_metric_only=True,verbose=True),lgb.log_evaluation(100)])
    best_iteration=int(tuned.best_iteration or args.max_rounds);inner_ap=float(tuned.best_score['inner_chronological_validation']['average_precision'])
    tuning_seconds=time.perf_counter()-tune_start
    del tuned,trainset,validset,x_inner,x_valid,maps_inner;gc.collect()
    logging.info('%s: refitting %d trees on %d model-training rows',name,best_iteration,train_end)
    mappings=fit_mappings(df.iloc[:train_end],schema)
    x_train,train_diag=encode(df.iloc[:train_end],schema,mappings)
    trainset=lgb.Dataset(x_train,label=y[:train_end],feature_name=features,categorical_feature=cats)
    refit_start=time.perf_counter()
    model=lgb.train(params,trainset,num_boost_round=best_iteration,callbacks=[lgb.log_evaluation(100)])
    refit_seconds=time.perf_counter()-refit_start
    del trainset,x_train;gc.collect()
    x_cal,cal_diag=encode(df.iloc[train_end:hold_start],schema,mappings)
    p_cal=model.predict(x_cal,num_threads=args.threads);y_cal=y[train_end:hold_start]
    cal_times=df[timecol].to_numpy()[train_end:hold_start]
    cal_fit_end,_=split_index(cal_times,.5);cal_select_end,_=split_index(cal_times,.75)
    select_slice=slice(cal_fit_end,cal_select_end);policy_slice=slice(cal_select_end,None)
    calibration_scores=[];preliminary={}
    methods=['raw','sigmoid']
    if cal_fit_end>=1000 and min(np.bincount(y_cal[:cal_fit_end],minlength=2))>=20:methods.append('isotonic')
    for method in methods:
        preliminary[method]=fit_calibrator(method,p_cal[:cal_fit_end],y_cal[:cal_fit_end])
        probabilities=calibrate(p_cal[select_slice],preliminary[method])
        calibration_scores.append({'experiment':name,'method':method,'scope':'calibration_method_selection',
            **probability_metrics(y_cal[select_slice],probabilities)})
    chosen=min(calibration_scores,key=lambda r:(r['log_loss'],r['brier_score'],methods.index(r['method'])))['method']
    final_calibrators={m:fit_calibrator(m,p_cal[:cal_select_end],y_cal[:cal_select_end]) for m in methods}
    chosen_cal=final_calibrators[chosen]
    policy_p=calibrate(p_cal[policy_slice],chosen_cal)
    learned_threshold=best_reward_threshold(y_cal[policy_slice],policy_p)
    thresholds={'default_0.5':.5,'theoretical_reward_1_6':1/6,'calibration_reward_max':learned_threshold}
    policy_metrics=[{'experiment':name,'scope':'calibration_threshold_selection','threshold_policy':label,
        'probability_method':chosen,**decision_metrics(y_cal[policy_slice],policy_p,t)} for label,t in thresholds.items()]
    logging.info('%s: frozen calibration=%s; policy threshold=%.8f. Now evaluating holdout.',name,chosen,learned_threshold)
    # Model, calibration method, refitted calibrator and policy are frozen before holdout.
    x_hold,hold_diag=encode(df.iloc[hold_start:],schema,mappings)
    pred_start=time.perf_counter();p_raw=model.predict(x_hold,num_threads=args.threads)
    prediction_seconds=time.perf_counter()-pred_start
    y_hold=y[hold_start:];p_selected=calibrate(p_raw,chosen_cal)
    metrics=[];threshold_rows=[];calibration_holdout=[]
    for method in methods:
        prob=calibrate(p_raw,final_calibrators[method])
        pm=probability_metrics(y_hold,prob)
        calibration_holdout.append({'experiment':name,'method':method,'selected_before_holdout':method==chosen,**pm})
        for label,t in thresholds.items():
            dm=decision_metrics(y_hold,prob,t)
            row={'experiment':name,'scope':'final_holdout','probability_method':method,'selected_calibrator':method==chosen,
                'threshold_policy':label,**pm,**dm}
            metrics.append(row)
            if method==chosen:threshold_rows.append(row)
    model.save_model(str(directory/'lightgbm_model.txt'))
    joblib.dump(chosen_cal,directory/'calibrator.joblib')
    joblib.dump(final_calibrators,directory/'all_calibrators.joblib')
    save_json(directory/'feature_schema.json',schema);save_json(directory/'categorical_mappings.json',mappings)
    config={'experiment':name,'calibration_method':chosen,'thresholds':thresholds,'default_threshold':1/6,
        'strict_decision_rule':'FLAG iff p > threshold; ties APPROVE','num_threads':args.threads,
        'reward_matrix':{'fraud_flag':10,'fraud_approve':-20,'legitimate_flag':-5,'legitimate_approve':1},
        'threshold_selection_objective':'Maximum realized mean reward on last chronological quarter of calibration period; ties prefer fewer flags; no capacity constraint',
        'fit_partitions':{'model_rows':train_end,'calibrator_fit_rows':cal_select_end,'calibration_method_selection_rows':cal_select_end-cal_fit_end,
            'threshold_selection_rows':len(y_cal)-cal_select_end},'holdout_used_for_selection':False}
    save_json(directory/'defender_config.json',config)
    runtime={'experiment':name,'feature_count':len(features),'categorical_features':len(cats),
        'best_iteration':best_iteration,'inner_validation_average_precision':inner_ap,
        'tuning_seconds':tuning_seconds,'refit_seconds':refit_seconds,'training_seconds':tuning_seconds+refit_seconds,
        'holdout_prediction_seconds':prediction_seconds,'batch_amortized_prediction_ms':prediction_seconds/len(y_hold)*1000,
        'model_size_bytes':(directory/'lightgbm_model.txt').stat().st_size}
    training_config={'parameters':params,'max_rounds':args.max_rounds,'early_stopping_patience':args.patience,
        'selected_iterations':best_iteration,'random_seed':SEED,'splits':bounds,'calibration_fit_end_in_calibration':cal_fit_end,
        'calibration_selection_end_in_calibration':cal_select_end,'calibration_selection_metric':'log_loss, then Brier',
        'ece_binning':'15 fixed equal-width bins over [0,1]; sum(bin_fraction * abs(mean_probability - fraud_rate))',
        'imbalance_strategy':'Natural class distribution, no oversampling, no class weighting, no scale_pos_weight',
        'scaling':'none','numeric_imputation':'none','runtime':runtime}
    save_json(directory/'training_config.json',training_config)
    csv(pd.DataFrame(calibration_scores),directory/'calibration_selection.csv')
    csv(pd.DataFrame(calibration_holdout),directory/'calibration_holdout.csv')
    csv(pd.DataFrame(policy_metrics),directory/'policy_selection.csv')
    diagnostics=[]
    for scope,data in [('model_training',train_diag),('calibration',cal_diag),('holdout',hold_diag)]:
        diagnostics.extend({'partition':scope,**r} for r in data)
    csv(pd.DataFrame(diagnostics),directory/'categorical_diagnostics.csv')
    csv(pd.DataFrame(metrics),directory/'model_metrics.csv')
    csv(pd.DataFrame(threshold_rows),directory/'threshold_comparison.csv')
    gain=model.feature_importance(importance_type='gain')
    importance=pd.DataFrame({'feature':features,'gain_importance':gain,'feature_group':[feature_group(c) for c in features]})
    importance=importance.sort_values('gain_importance',ascending=False).reset_index(drop=True)
    importance['rank']=np.arange(1,len(importance)+1);importance['gain_fraction']=importance.gain_importance/importance.gain_importance.sum()
    csv(importance,directory/'feature_importance.csv')
    # Inference equivalence, unknown categories, explicit NULLs, schema validation.
    service=Defender(directory)
    sample=df.iloc[hold_start:hold_start+32][features]
    np.testing.assert_allclose(service.predict(sample),p_selected[:32],rtol=1e-7,atol=1e-9)
    modified=sample.iloc[[0]].copy()
    textcats=[f for f in schema['features'] if f['role']=='categorical' and f['postgres_type'] not in NUMERIC]
    if textcats:modified[textcats[0]['name']]='__UNSEEN_INFERENCE_TEST__'
    numeric=[f for f in schema['features'] if f['role']=='numeric']
    if numeric:modified[numeric[0]['name']]=np.nan
    assert np.isfinite(service.predict(modified)).all()
    try:service.predict(sample.drop(columns=features[0]));raise AssertionError('Missing schema field was accepted')
    except ValueError:pass
    latencies=[]
    for _ in range(12):
        t=time.perf_counter();service.predict(sample.iloc[[0]]);latencies.append((time.perf_counter()-t)*1000)
    runtime['single_transaction_pipeline_ms_median']=float(np.median(latencies[2:]))
    runtime['single_transaction_pipeline_ms_p95']=float(np.percentile(latencies[2:],95))
    if name=='B_clean':
        pred=df.iloc[hold_start:][[idcol,timecol,labelcol]].copy()
        pred['raw_probability']=p_raw;pred['calibrated_probability']=p_selected
        for label,t in thresholds.items():pred['decision_'+label]=np.where(p_selected>t,'FLAG','APPROVE')
        csv(pred,out/'predictions_holdout.csv')
        csv(importance,out/'feature_importance.csv')
        csv(importance.groupby('feature_group').agg(gain_importance=('gain_importance','sum'),feature_count=('feature','count')).reset_index(),out/'feature_group_importance.csv')
        temporal=[];times=df[timecol].to_numpy()[hold_start:]
        edges=[0]+[split_index(times,f)[0] for f in [.25,.5,.75]]+[len(times)]
        for w,(left,right) in enumerate(zip(edges[:-1],edges[1:]),1):
            for label,t in thresholds.items():
                temporal.append({'window':w,'rows':right-left,'min_time':int(times[left]),'max_time':int(times[right-1]),
                    'fraud_count':int(y_hold[left:right].sum()),'fraud_rate':float(y_hold[left:right].mean()),
                    'threshold_policy':label,'probability_method':chosen,**probability_metrics(y_hold[left:right],p_selected[left:right]),
                    **decision_metrics(y_hold[left:right],p_selected[left:right],t)})
        csv(pd.DataFrame(temporal),out/'temporal_performance.csv')
        # Descriptive permutation importance only; no additional feature removal.
        indices=np.linspace(cal_fit_end,cal_select_end-1,min(5000,cal_select_end-cal_fit_end),dtype=int)
        perm_x=x_cal[indices].copy();perm_y=y_cal[indices]
        baseline_ap=average_precision_score(perm_y,model.predict(perm_x,num_threads=args.threads))
        rng=np.random.default_rng(SEED);permutation=[]
        for feature in importance.head(30).feature:
            j=features.index(feature);original=perm_x[:,j].copy();effects=[]
            for repeat in range(3):
                perm_x[:,j]=original[rng.permutation(len(original))]
                effects.append(baseline_ap-average_precision_score(perm_y,model.predict(perm_x,num_threads=args.threads)))
            perm_x[:,j]=original
            permutation.append({'feature':feature,'average_precision_decrease_mean':float(np.mean(effects)),
                'average_precision_decrease_std':float(np.std(effects)),'repeats':3,'sample_rows':len(perm_y),
                'scope':'Evenly spaced calibration-method-selection rows; descriptive only'})
        csv(pd.DataFrame(permutation),out/'permutation_importance.csv')
        reliability=[]
        for method,cal in final_calibrators.items():
            _,bins=ece(y_hold,calibrate(p_raw,cal))
            reliability.extend({'method':method,**b} for b in bins)
        csv(pd.DataFrame(reliability),out/'reliability_bins.csv')
        np.savez_compressed(out/'plot_data.npz',labels=y_hold,raw_probability=p_raw,calibrated_probability=p_selected)
    runtime['experiment_seconds']=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runtime['process_peak_rss_mb']=float(peak/(1024**2) if sys.platform=='darwin' else peak/1024)
    runtime['peak_memory_scope']='Process cumulative high-water RSS, not isolated per-experiment increment'
    save_json(directory/'runtime.json',runtime)
    logging.info('%s complete: calibrated holdout AP=%.5f, %s, %.1f sec',name,average_precision_score(y_hold,p_selected),chosen,runtime['experiment_seconds'])
    del model,service,x_cal,x_hold;gc.collect()
    return metrics,threshold_rows,runtime,calibration_scores,calibration_holdout

def feature_group(name):
    import re
    if name.startswith('card'):return 'card'
    if name.startswith('id_') or name.lower() in ('devicetype','deviceinfo'):return 'identity'
    for prefix in ['c','d','v']:
        if re.fullmatch(prefix+r'\d+',name.lower()):return prefix.upper()
    return 'other'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/lightgbm_baseline')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-rounds',type=int,default=2000)
    parser.add_argument('--patience',type=int,default=100)
    parser.add_argument('--holdout-cutoff',type=int,default=12192842)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'artifacts/lightgbm_model.txt').exists():raise FileExistsError('Choose a fresh output directory; existing trained artifacts are protected')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(out/'execution.log'),logging.StreamHandler(sys.stdout)])
    lgb.register_logger(logging.getLogger('lightgbm'))
    start=time.perf_counter();logging.info('Beginning read-only dataset inspection')
    df,features,extras,types,integrity,missing,snapshot,original_available,meta=load_database(out)
    idcol,timecol,labelcol=meta;times=df[timecol].to_numpy()
    hold_start=int(np.searchsorted(times,args.holdout_cutoff,side='right'))
    if not 0<hold_start<len(df):raise ValueError('Prior holdout boundary not within observed times')
    train_end,train_cutoff=split_index(times[:hold_start],.65/.8)
    inner_end,inner_cutoff=split_index(times[:train_end],.85)
    bounds={'inner_end':inner_end,'train_end':train_end,'hold_start':hold_start,'total_rows':len(df),
        'inner_training_cutoff':inner_cutoff,'model_training_cutoff':train_cutoff,'prior_holdout_cutoff':args.holdout_cutoff,
        'sorting':'TransactionDT ascending, then TransactionID; identical timestamps stay together'}
    assert times[inner_end-1]<times[inner_end] and times[train_end-1]<times[train_end] and times[hold_start-1]<times[hold_start]
    summary=[]
    segments=[('all_labeled',0,len(df)),('model_training',0,train_end),('inner_fit',0,inner_end),
              ('inner_early_stopping',inner_end,train_end),('calibration',train_end,hold_start),('final_holdout',hold_start,len(df))]
    caltimes=times[train_end:hold_start];cf,_=split_index(caltimes,.5);cs,_=split_index(caltimes,.75)
    segments += [('calibrator_initial_fit',train_end,train_end+cf),('calibration_method_selection',train_end+cf,train_end+cs),
                 ('calibrator_final_fit',train_end,train_end+cs),('threshold_selection',train_end+cs,hold_start)]
    for name,left,right in segments:
        segment=df.iloc[left:right];fraud=int(segment[labelcol].sum());miss=int(segment[features].isna().sum().sum())
        summary.append({'partition':name,'rows':right-left,'predictor_count':len(features),'total_columns':len(features)+3,
            'min_transactiondt':int(times[left]),'max_transactiondt':int(times[right-1]),'fraud_rows':fraud,
            'legitimate_rows':right-left-fraud,'fraud_rate':fraud/(right-left),'missing_predictor_cells':miss,
            'missing_predictor_percentage':100*miss/((right-left)*len(features))})
    testinfo=next(r for r in integrity if r['table']=='test_clean');testmissing=sum(r['missing_count'] for r in missing if r['table']=='test_clean' and r['feature'].lower() not in META)
    summary.append({'partition':'kaggle_test_unlabeled','rows':int(testinfo['rows']),'predictor_count':len(features),'total_columns':len(features)+2,
        'min_transactiondt':int(testinfo['min_time']),'max_transactiondt':int(testinfo['max_time']),
        'missing_predictor_cells':testmissing,'missing_predictor_percentage':100*testmissing/(int(testinfo['rows'])*len(features))})
    csv(pd.DataFrame(summary),out/'dataset_summary.csv');save_json(out/'snapshot_manifest.json',snapshot)
    save_json(out/'split_boundaries.json',bounds)
    logging.info('Rows: model=%d calibration=%d holdout=%d; %d clean predictors; original extras=%s',train_end,hold_start-train_end,len(df)-hold_start,len(features),extras)
    all_metrics=[];all_thresholds=[];runtimes=[];selection=[];cal_scores=[]
    experiments=[('B_clean',features)]
    if original_available:experiments.append(('A_original',features+extras))
    for name,cols in experiments:
        m,t,r,c,h=experiment(name,df,cols,types,bounds,meta,args,out)
        all_metrics+=m;all_thresholds+=t;runtimes.append(r);selection+=c;cal_scores+=h
        # Persist each completed experiment before proceeding to comparisons.
        csv(pd.DataFrame(all_metrics),out/'model_metrics.csv');csv(pd.DataFrame(all_thresholds),out/'threshold_comparison.csv')
        csv(pd.DataFrame(runtimes),out/'runtime_metrics.csv');csv(pd.DataFrame(selection),out/'calibration_selection.csv');csv(pd.DataFrame(cal_scores),out/'calibration_comparison.csv')
    comparison=[{'experiment':name,'status':'EXECUTED','predictors':len(cols),'explanation':'Primary Defender' if name=='B_clean' else 'Original values verified equal on shared columns; same chronological protocol'} for name,cols in experiments]
    audit_manifest=ROOT/'feature_audit/reports/feature_audit_20261014_v1/selection_manifest.json'
    remaining=[]
    if audit_manifest.exists():remaining=sorted(set(json.loads(audit_manifest.read_text())['proposed_drops'])&set(features))
    if remaining:
        cols=[c for c in features if c not in remaining]
        m,t,r,c,h=experiment('C_correlated_candidates_removed',df,cols,types,bounds,meta,args,out)
        all_metrics+=m;all_thresholds+=t;runtimes.append(r);selection+=c;cal_scores+=h
        comparison.append({'experiment':'C_correlated_candidates_removed','status':'EXECUTED','predictors':len(cols),'explanation':'In-memory experiment only; database unchanged'})
    else:comparison.append({'experiment':'C_correlated_candidates_removed','status':'NOT_DISTINCT_FROM_B','predictors':len(features),'explanation':'All prior proposed correlated removals already absent from clean tables; no fabricated third run'})
    csv(pd.DataFrame(comparison),out/'experiment_comparison.csv')
    csv(pd.DataFrame(all_metrics),out/'model_metrics.csv');csv(pd.DataFrame(all_thresholds),out/'threshold_comparison.csv')
    csv(pd.DataFrame(runtimes),out/'runtime_metrics.csv');csv(pd.DataFrame(selection),out/'calibration_selection.csv');csv(pd.DataFrame(cal_scores),out/'calibration_comparison.csv')
    corrfile=ROOT/'feature_audit/reports/feature_audit_20261014_v1/high_correlation_pairs.csv'
    if corrfile.exists():
        pairs=pd.read_csv(corrfile);csv(pairs[pairs.feature_a.isin(features)&pairs.feature_b.isin(features)],out/'retained_correlated_features.csv')
    packages=['numpy','pandas','scipy','scikit-learn','lightgbm','psycopg','psycopg-binary','joblib','python-dotenv','matplotlib','psutil']
    versions={name:importlib.metadata.version(name) for name in packages}
    requirements='\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n'
    (out/'artifacts/requirements.txt').write_text(requirements)
    run={'started_at':snapshot['captured_at'],'completed_at':datetime.now(timezone.utc).isoformat(),
        'total_runtime_seconds':time.perf_counter()-start,'seed':SEED,'python':platform.python_version(),'platform':platform.platform(),
        'versions':versions,'database_access':'READ ONLY REPEATABLE READ; no writes or schema changes',
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'defender').glob('*.py')}}
    save_json(out/'run_manifest.json',run)
    from .report import build_report
    build_report(out)
    logging.info('All experiments and report generation completed: %s',out)

if __name__=='__main__':main()
