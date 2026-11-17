"""Reproducible repeated nested CV comparison with equal inner-folds and matched search trial budgets.
Research scope: classical ML / lightweight automated search. Not full AutoML or LLM agent.
"""
import argparse, json, platform, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, optuna
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
RESULTS.mkdir(parents=True, exist_ok=True)
SEEDS=[42,123]
FOLDS=3
TRIALS=12
INNER_FOLDS=3

def dataset_list():
    for name, func in [('BreastCancer',load_breast_cancer),('Wine',load_wine),('Iris',load_iris),('Digits600',load_digits)]:
        obj=func()
        X=np.asarray(obj.data,dtype=np.float64)
        y=np.asarray(obj.target,dtype=np.int64)
        if name=='Digits600':
            X,_,y,_=train_test_split(X,y,train_size=600,stratify=y,random_state=42)
        yield name,X,y

def make_model(family,par=None,seed=42):
    par = par or {}
    if family=='LogReg':
        return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=par.get('C',1.0),max_iter=700,random_state=seed))])
    if family=='SVM':
        return Pipeline([('scale',StandardScaler()),('model',SVC(kernel='rbf',C=par.get('C',1.0),gamma=par.get('gamma','scale')))])
    if family=='RF':
        return RandomForestClassifier(n_estimators=par.get('n_estimators',150),max_depth=par.get('max_depth',None),min_samples_leaf=par.get('min_samples_leaf',1),random_state=seed,n_jobs=1)
    raise ValueError(family)

def params_from_trial(trial):
    kind=trial.suggest_categorical('family',['SVM','RF'])
    if kind=='LogReg':
        p={'C':trial.suggest_float('lr_C',1e-3,1e3,log=True)}
    elif kind=='SVM':
        p={'C':trial.suggest_float('svm_C',1e-2,1e3,log=True), 'gamma':trial.suggest_categorical('svm_gamma',['scale',0.0001,0.001,0.01,0.1,1.0])}
    else:
        p={'n_estimators':trial.suggest_categorical('rf_trees',[60,100]),'max_depth':trial.suggest_categorical('rf_depth',[None,5,10]),'min_samples_leaf':trial.suggest_categorical('rf_leaf',[1,2])}
    return kind,p

def decoded(p):
    if p['family']=='LogReg': return 'LogReg',{'C':p['lr_C']}
    if p['family']=='SVM': return 'SVM',{'C':p['svm_C'],'gamma':p['svm_gamma']}
    return 'RF',{'n_estimators':p['rf_trees'],'max_depth':p['rf_depth'],'min_samples_leaf':p['rf_leaf']}

def run_search(Xtr,ytr,model_seed,inner_splits,algo,iterations,record_meta):
    studyseed=model_seed+9001
    if algo=='TPE': sampler=optuna.samplers.TPESampler(seed=studyseed,n_startup_trials=4)
    elif algo=='Random': sampler=optuna.samplers.RandomSampler(seed=studyseed)
    else: raise ValueError(algo)
    study=optuna.create_study(direction='maximize',sampler=sampler)
    alltrials=[]; stages={}
    start=time.perf_counter()
    def objective(trial):
        family,par=params_from_trial(trial)
        model=make_model(family,par,model_seed)
        value=float(np.mean(cross_val_score(model,Xtr,ytr,cv=inner_splits,scoring='balanced_accuracy',n_jobs=1,error_score='raise')))
        alltrials.append({**record_meta,'optimizer':algo,'trial':trial.number+1,'family':family,'params':json.dumps(par),'inner_cv_score':value})
        return value
    if algo=='TPE':
        study.optimize(objective,n_trials=4,show_progress_bar=False)
        stages['TPE4']={'elapsed_search_s':time.perf_counter()-start,'best_value':float(study.best_value),'best_params':study.best_params.copy()}
        study.optimize(objective,n_trials=iterations-4,show_progress_bar=False)
        stages['TPE12']={'elapsed_search_s':time.perf_counter()-start,'best_value':float(study.best_value),'best_params':study.best_params.copy()}
    else:
        study.optimize(objective,n_trials=iterations,show_progress_bar=False)
        stages['Random12']={'elapsed_search_s':time.perf_counter()-start,'best_value':float(study.best_value),'best_params':study.best_params.copy()}
    return stages,alltrials

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+',type=int,default=SEEDS)
    ap.add_argument('--trials',type=int,default=TRIALS,choices=[12])
    args=ap.parse_args()
    rows=[]; preds=[]; trials=[]; profiles=[]
    for dataset,X,y in dataset_list():
        profiles.append({'dataset':dataset,'n_samples':len(y),'n_features':X.shape[1],'n_classes':len(np.unique(y)),'min_class_count':int(np.bincount(y).min()),'data_source':'scikit-learn built-in; digits is stratified 600-sample subset','data_type':'flattened image-derived feature vectors' if dataset=='Digits600' else 'numerical tabular'})
        for seed in args.seeds:
            outer=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=seed)
            for fold,(i_train,i_test) in enumerate(outer.split(X,y),1):
                Xtr,Xte,ytr,yte=X[i_train],X[i_test],y[i_train],y[i_test]
                # The very same predefined inner-fold indices are used for SVM, TPE and random search.
                inner=list(StratifiedKFold(n_splits=INNER_FOLDS,shuffle=True,random_state=seed+fold*101).split(Xtr,ytr))
                modelseed=seed+fold*10
                meta={'dataset':dataset,'seed':seed,'outer_fold':fold}
                def score_and_log(method,model,elapsed,extra):
                    prediction=model.predict(Xte)
                    record={**meta,'method':method,'n_train':len(i_train),'n_test':len(i_test),
                      'balanced_accuracy':float(balanced_accuracy_score(yte,prediction)),
                      'macro_f1':float(f1_score(yte,prediction,average='macro',zero_division=0)),
                      'accuracy':float(accuracy_score(yte,prediction)),
                      'search_and_refit_s':float(elapsed),'method_details':json.dumps(extra,default=str)}
                    rows.append(record)
                    for idx,label,guess in zip(i_test,yte,prediction):
                        preds.append({**meta,'method':method,'row_index':int(idx),'true_label':int(label),'predicted_label':int(guess)})
                    print(f"{dataset} seed={seed} fold={fold} {method:10s} bacc={record['balanced_accuracy']:.5f} seconds={elapsed:.3f}",flush=True)
                # Untuned fixed controls.
                for label,family in [('LogReg','LogReg'),('RandomForest','RF')]:
                    start=time.perf_counter(); model=make_model(family,seed=modelseed);model.fit(Xtr,ytr)
                    score_and_log(label,model,time.perf_counter()-start,{'fixed':True})
                # Strong, systematic RBF SVM tuning; folds identical to auto search.
                grid=GridSearchCV(make_model('SVM',seed=modelseed),param_grid={'model__C':[0.1,1,10,100], 'model__gamma':['scale',0.01,0.1]},scoring='balanced_accuracy',cv=inner,n_jobs=1,refit=True,error_score='raise')
                start=time.perf_counter(); grid.fit(Xtr,ytr)
                score_and_log('TunedSVM',grid.best_estimator_,time.perf_counter()-start,{'best_params':grid.best_params_,'best_inner_score':grid.best_score_,'candidates':12})
                # Search strategies have equal candidate families and same number of trials.
                for optimizer in ['Random','TPE']:
                    stages,newtrials=run_search(Xtr,ytr,modelseed,inner,optimizer,args.trials,meta)
                    trials.extend(newtrials)
                    for label,stage in stages.items():
                        family,params=decoded(stage['best_params'])
                        start=time.perf_counter();model=make_model(family,params,modelseed);model.fit(Xtr,ytr)
                        score_and_log(label,model,stage['elapsed_search_s']+(time.perf_counter()-start),{'family':family,'params':params,'best_inner_score':stage['best_value'],'trials':4 if label=='TPE4' else 12,'optimizer':optimizer})
    output=pd.DataFrame(rows)
    pred=pd.DataFrame(preds)
    assert not pred.duplicated(['dataset','seed','outer_fold','method','row_index']).any()
    # Prevent accidental incomplete method panel.
    assert output.groupby(['dataset','seed','outer_fold']).size().eq(6).all()
    output.to_csv(RESULTS/'fold_metrics.csv',index=False)
    pred.to_csv(RESULTS/'predictions.csv',index=False)
    pd.DataFrame(profiles).to_csv(RESULTS/'dataset_profiles.csv',index=False)
    pd.DataFrame(trials).to_csv(RESULTS/'search_trial_logs.csv',index=False)
    per=output.groupby(['dataset','method'],sort=True).agg(mean_bacc=('balanced_accuracy','mean'),sd_bacc=('balanced_accuracy','std'),mean_macro_f1=('macro_f1','mean'),mean_accuracy=('accuracy','mean'),mean_seconds=('search_and_refit_s','mean'),n_test_folds=('balanced_accuracy','size')).reset_index()
    per.to_csv(RESULTS/'per_dataset_summary.csv',index=False)
    grouped=output.groupby('method',sort=True).agg(mean_bacc=('balanced_accuracy','mean'),sd_bacc=('balanced_accuracy','std'),mean_macro_f1=('macro_f1','mean'),mean_accuracy=('accuracy','mean'),mean_seconds=('search_and_refit_s','mean'),n_test_folds=('balanced_accuracy','size')).reset_index()
    grouped.to_csv(RESULTS/'overall_summary.csv',index=False)
    tab_only=output[output.dataset.isin(['BreastCancer','Wine','Iris'])]
    tab_only.groupby('method',sort=True).agg(mean_bacc=('balanced_accuracy','mean'),sd_bacc=('balanced_accuracy','std'),mean_macro_f1=('macro_f1','mean'),mean_accuracy=('accuracy','mean'),mean_seconds=('search_and_refit_s','mean'),n_test_folds=('balanced_accuracy','size')).reset_index().to_csv(RESULTS/'tabular_only_summary.csv',index=False)
    env={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'optuna':optuna.__version__,'random_seeds':args.seeds,'outer_cv_splits':FOLDS,'inner_cv_splits':INNER_FOLDS,'search_trials':TRIALS,'TPE_startup_random_trials':4,'timing_note':'Fixed models fit only; tuned models include inner CV search plus full-train refit; CPU model runtimes not equal-budget, use cautiously','no_agent_or_full_AutoML':True,'dataset_note':'Digits600 is derived from an image-recognition dataset, not conventional semantically tabular observations'}
    (RESULTS/'environment.json').write_text(json.dumps(env,indent=2))
    print('----OVERALL----\n'+grouped.to_string(index=False),flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=FutureWarning)
    main()
