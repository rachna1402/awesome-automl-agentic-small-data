"""Rebuilds manuscript figures only from genuine CSV results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results'
O=pd.read_csv(R/'overall_summary.csv')
P=pd.read_csv(R/'per_dataset_summary.csv')
F=pd.read_csv(R/'fold_metrics.csv')
OUT=R/'figures'; OUT.mkdir(exist_ok=True)
methods=['LogReg','RandomForest','TunedSVM','Random12','TPE4','TPE12']
short=['Logistic','Random forest','Tuned SVM','Random (12)','TPE (4)','TPE (12)']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
# Dataset comparison: separate panels, no data-scale distortions.
fig,axs=plt.subplots(2,2,figsize=(10.5,7.4),sharey=True)
axs=axs.ravel()
for ax,ds in zip(axs,['BreastCancer','Wine','Iris','Digits600']):
    p=P[P.dataset==ds].set_index('method').loc[methods]
    x=np.arange(len(methods)); ax.bar(x,100*p.mean_bacc)
    ax.errorbar(x,100*p.mean_bacc,yerr=100*p.sd_bacc,fmt='none',capsize=3,ecolor='black',lw=0.8)
    ax.set_ylim(0,105);ax.set_title({'BreastCancer':'Breast cancer','Wine':'Wine','Iris':'Iris','Digits600':'Digits (600)'}[ds]);ax.set_xticks(x,short,rotation=42,ha='right');ax.tick_params(axis='x',labelsize=9)
    ax.grid(axis='y',alpha=.18)
axs[0].set_ylabel('Balanced accuracy (%)'); axs[2].set_ylabel('Balanced accuracy (%)')
fig.suptitle('Observed mean and descriptive SD across 6 held-out folds per task',fontsize=12)
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'per_dataset_accuracy.pdf',bbox_inches='tight');fig.savefig(OUT/'per_dataset_accuracy.png',dpi=175,bbox_inches='tight');plt.close(fig)
# Resource/quality tradeoff values are per fold; fixed models have no tuning; warn in caption.
fig,ax=plt.subplots(figsize=(8.1,5.0))
o=O.set_index('method').loc[methods]
for i,(m,lab) in enumerate(zip(methods,short)):
    ax.scatter(o.loc[m,'mean_seconds'],o.loc[m,'mean_bacc']*100,s=60,label=lab)
    ax.annotate(lab,(o.loc[m,'mean_seconds'],o.loc[m,'mean_bacc']*100),xytext=(6,5),textcoords='offset points',fontsize=8)
ax.set_xlabel('Mean wall-clock search + refit time per held-out fold (seconds)')
ax.set_ylabel('Mean balanced accuracy across 4 dataset tasks (%)');ax.set_title('Accuracy–runtime trade-off (method-specific work)')
ax.set_ylim(88,102);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'runtime_tradeoff.pdf');fig.savefig(OUT/'runtime_tradeoff.png',dpi=175);plt.close(fig)
# Search budget sensitivity is matched within the same TPE optimization run, no post hoc cherry-picking.
sel=F.pivot(index=['dataset','seed','outer_fold'],columns='method',values='balanced_accuracy')
delta=(sel['TPE12']-sel['TPE4'])*100
fig,ax=plt.subplots(figsize=(8,4.6));ds_order=['BreastCancer','Wine','Iris','Digits600']
allvals=[delta.loc[d].values for d in ds_order]
ax.axhline(0,linestyle='--',linewidth=1)
ax.boxplot(allvals,tick_labels=['Breast cancer','Wine','Iris','Digits (600)'],showmeans=True)
ax.set_ylabel('TPE 12 trials – TPE 4 trials (percentage points)')
ax.set_title('Change on held-out folds after increasing the search budget')
ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'budget_effect.pdf');fig.savefig(OUT/'budget_effect.png',dpi=175);plt.close(fig)
print('Generated vector PDF and PNG figures from actual CSVs.')
