"""Independent score and split validation of the CSV evidence created by benchmark.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
R=Path(__file__).resolve().parents[1]/'results'
F=pd.read_csv(R/'fold_metrics.csv')
P=pd.read_csv(R/'predictions.csv')
T=pd.read_csv(R/'search_trial_logs.csv')
D=pd.read_csv(R/'dataset_profiles.csv')
methods=['LogReg','RandomForest','TunedSVM','Random12','TPE4','TPE12']
assert set(F.method)==set(methods),f'Missing methods: {set(methods)-set(F.method)}'
assert not P.duplicated(['dataset','seed','outer_fold','method','row_index']).any()
assert not F.duplicated(['dataset','seed','outer_fold','method']).any()
assert F.groupby(['dataset','seed','outer_fold']).size().eq(6).all()
for k,p in P.groupby(['dataset','seed','outer_fold','method']):
    r=F.loc[(F.dataset==k[0])&(F.seed==k[1])&(F.outer_fold==k[2])&(F.method==k[3])]
    assert len(r)==1
    r=r.iloc[0]
    assert len(p)==r.n_test
    assert abs(balanced_accuracy_score(p.true_label,p.predicted_label)-r.balanced_accuracy)<1e-12
    assert abs(f1_score(p.true_label,p.predicted_label,average='macro',zero_division=0)-r.macro_f1)<1e-12
    assert abs(accuracy_score(p.true_label,p.predicted_label)-r.accuracy)<1e-12
for (ds,seed),g in P.groupby(['dataset','seed']):
    full_n=int(D.loc[D.dataset==ds,'n_samples'].iloc[0])
    for m,q in g.groupby('method'):
        assert len(q)==full_n,(ds,seed,m,len(q),full_n)
        assert set(q.row_index)==set(range(full_n))
    first=g[g.method==methods[0]][['outer_fold','row_index','true_label']]
    for m in methods[1:]:
        other=g[g.method==m][['outer_fold','row_index','true_label']]
        assert first.sort_values('row_index').reset_index(drop=True).equals(other.sort_values('row_index').reset_index(drop=True))
for (ds,seed,fold,optimizer),t in T.groupby(['dataset','seed','outer_fold','optimizer']):
    assert len(t)==12,(ds,seed,fold,optimizer,len(t))
    assert set(t.trial)==set(range(1,13))
assert np.isfinite(F[['balanced_accuracy','macro_f1','accuracy','search_and_refit_s']].values).all()
assert F.search_and_refit_s.ge(0).all()
report={'status':'PASS','n_dataset_tasks':int(F.dataset.nunique()),'n_seeds':int(F.seed.nunique()),'n_outer_evaluations':int(F.groupby(['dataset','seed','outer_fold']).ngroups),'n_method_evaluations':len(F),'n_test_predictions':len(P),'n_logged_search_trials':len(T),'verified':'Metric recomputation; data labels; duplicate and missing rows; held-out fold alignment across methods; all test indices covered once per seed/method; exact search-trial counts; finite values'}
(R/'verification_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
