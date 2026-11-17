from pathlib import Path
import pandas as pd,json
P=Path(__file__).resolve().parents[1]
R=P/'results'
T=P/'paper'
O=pd.read_csv(R/'overall_summary.csv').set_index('method')
B=pd.read_csv(R/'tabular_only_summary.csv').set_index('method')
D=pd.read_csv(R/'per_dataset_summary.csv')
F=pd.read_csv(R/'fold_metrics.csv')
V=json.loads((R/'verification_report.json').read_text())
methods=['LogReg','RandomForest','TunedSVM','Random12','TPE4','TPE12']
names={'LogReg':'Logistic regression','RandomForest':'Random forest','TunedSVM':'Tuned SVM','Random12':'Random 12','TPE4':'TPE prefix 4','TPE12':'TPE 12'}
dsn={'BreastCancer':'Breast Cancer','Wine':'Wine','Iris':'Iris','Digits600':'Digits600 (aux.)'}
def pct(x): return f'{x*100:.2f}\\%'
def r(x): return f'{x:.3f}'
def table_tabular():
 a=[r'\begin{table}[H]',r'\centering\caption{Primary tabular-only descriptive results over three datasets and six held-out folds per task. Standard deviation pools correlated repeated folds and is not a confidence interval.}',r'\label{tab:tabular}',r'\small\begin{tabular}{lrrrr}\toprule',r'Method & Bal. acc. & SD & Macro-F1 & Time (s)\\\midrule']
 for m in methods:
  v=B.loc[m];a.append(f"{names[m]} & {pct(v.mean_bacc)} & {pct(v.sd_bacc)} & {pct(v.mean_macro_f1)} & {r(v.mean_seconds)} " + r"\\")
 a += [r'\bottomrule\end{tabular}',r'\end{table}'];return '\n'.join(a)
def table_all():
 a=[r'\begin{table}[H]',r'\centering\caption{Four-task descriptive summary including the supplementary Digits600 feature-vector task (24 test folds per method).}',r'\label{tab:overall}',r'\small\begin{tabular}{lrrrr}\toprule',r'Method & Bal. acc. & SD & Macro-F1 & Time (s)\\\midrule']
 for m in methods:
  v=O.loc[m];a.append(f"{names[m]} & {pct(v.mean_bacc)} & {pct(v.sd_bacc)} & {pct(v.mean_macro_f1)} & {r(v.mean_seconds)} " + r"\\")
 a += [r'\bottomrule\end{tabular}',r'\end{table}'];return '\n'.join(a)
def table_dataset():
 p=D.pivot(index='dataset',columns='method',values='mean_bacc')
 a=[r'\begin{table}[H]',r'\centering\caption{Mean held-out balanced accuracy (percent) by method and task, with two seeds and three folds per seed.}',r'\label{tab:perdataset}',r'\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrrr}\toprule',r'Task & Logistic & RF & Tuned SVM & Rand12 & TPE4 & TPE12\\\midrule']
 for ds in ['BreastCancer','Wine','Iris','Digits600']:
  v=p.loc[ds]
  a.append(dsn[ds]+' & '+' & '.join(f'{v[m]*100:.2f}' for m in methods)+r'\\')
 a += [r'\bottomrule\end{tabular}',r'\end{table}'];return '\n'.join(a)
sel=F.pivot(index=['dataset','seed','outer_fold'],columns='method',values='balanced_accuracy')
tpe_minus_grid=(O.loc['TPE12'].mean_bacc-O.loc['TunedSVM'].mean_bacc)*100
random_minus=(O.loc['TPE12'].mean_bacc-O.loc['Random12'].mean_bacc)*100
bud=(O.loc['TPE12'].mean_bacc-O.loc['TPE4'].mean_bacc)*100
nplus=int((sel.TPE12>sel.TPE4).sum());nsame=int((sel.TPE12==sel.TPE4).sum());nminus=int((sel.TPE12<sel.TPE4).sum())
# Separate primary dataset winner from supplementary mixed-task winner.
tabwin=B.mean_bacc.idxmax(); fullwin=O.mean_bacc.idxmax()
sub={
 '@@NEVAL@@':str(V['n_method_evaluations']),'@@NPREDS@@':f"{V['n_test_predictions']:,}",
 '@@TABTOP@@':pct(B.loc[tabwin].mean_bacc),'@@TABWINNER@@':names[tabwin],
 '@@ALLTOP@@':pct(O.loc[fullwin].mean_bacc),'@@ALLWINNER@@':names[fullwin],
 '@@TABLETABULAR@@':table_tabular(),'@@TABLEOVERALL@@':table_all(),'@@TABLEPERDATASET@@':table_dataset(),
 '@@TABULAR_SENTENCE@@': f"the fixed logistic-regression baseline attains {pct(B.loc['LogReg'].mean_bacc)} mean balanced accuracy, with TPE12 close at {pct(B.loc['TPE12'].mean_bacc)} and grid-tuned SVM at {pct(B.loc['TunedSVM'].mean_bacc)}. The difference between the two highest tabular-only means is only {(B.loc['LogReg'].mean_bacc-B.loc['TPE12'].mean_bacc)*100:.2f} percentage points.",
 '@@SEARCH_SENTENCE@@':f"TPE12 attains {pct(O.loc['TPE12'].mean_bacc)} on the full four-task summary versus {pct(O.loc['Random12'].mean_bacc)} for Random12, a descriptive difference of {random_minus:.2f} percentage points. The difference relative to grid-tuned SVM is {tpe_minus_grid:.2f} percentage points. Both search procedures have 12 trials, but their candidate configurations and runtime realizations differ.",
 '@@BUDGET_SENTENCE@@':f"The twelve-trial TPE search improves over its four-trial prefix in {nplus} of the 24 matched test-fold comparisons, ties in {nsame}, and is lower in {nminus}; the aggregate difference is {bud:.2f} percentage points. This is a within-study observational budget comparison, not a formal proof that more trials always improve generalization.",
 '@@DATASET_SENTENCE@@':f"The primary tabular tasks are not won by the same method on every dataset: mean balanced accuracy is highest for logistic regression on Breast Cancer ({pct(D[(D.dataset=='BreastCancer')&(D.method=='LogReg')].iloc[0].mean_bacc)}) and Wine ({pct(D[(D.dataset=='Wine')&(D.method=='LogReg')].iloc[0].mean_bacc)}), while TPE12 is highest for Iris ({pct(D[(D.dataset=='Iris')&(D.method=='TPE12')].iloc[0].mean_bacc)}). On the supplementary Digits task TPE12 also leads ({pct(D[(D.dataset=='Digits600')&(D.method=='TPE12')].iloc[0].mean_bacc)}). These descriptive differences show why tabular-only and auxiliary summaries should be reported separately.",
 '@@CONCLUSION_SENTENCE@@':f"Among the three primary tabular tasks, simple fixed logistic regression is the highest-scoring method on the macro-averaged balanced-accuracy metric ({pct(B.loc['LogReg'].mean_bacc)}), narrowly above TPE12 ({pct(B.loc['TPE12'].mean_bacc)}). Over all four tasks, TPE12 attains the largest mean ({pct(O.loc['TPE12'].mean_bacc)}), but at higher observed fit/search time than tuned SVM and far higher time than the fixed logistic baseline. These rankings depend on the inclusion of the supplementary image-derived dataset and must not be presented as universal performance superiority."
}
s=(T/'manuscript_template.tex').read_text()
for k,v in sub.items():s=s.replace(k,v)
import re
left=re.findall('@@[A-Z0-9_]+@@',s)
if left:raise RuntimeError(left)
(T/'manuscript.tex').write_text(s)
print('Built',T/'manuscript.tex')
print('tabular',tabwin,round(B.loc[tabwin].mean_bacc,6),'all',fullwin,round(O.loc[fullwin].mean_bacc,6))
print('budget',nplus,nsame,nminus)
