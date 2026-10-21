import sys,pandas as pd,numpy as np
from sklearn.metrics import f1_score,average_precision_score
V=sys.argv[1];R=sys.argv[2]
P=lambda sc,c: pd.read_csv(f'{V}/predictions/{sc}_{c}_test_predictions.csv')
def wr(y,s,k):
    n=int(len(y)*k/100); o=np.argsort(-s,kind='stable'); return y[o][:n].sum()/y.sum()
e2=pd.read_csv(f'{V}/results/E2_paired_bootstrap_differences.csv')
print(e2[(e2.scale=='Enron_100000')&(e2.a=='distilbert_text')&(e2.b=='tfidf_text')].round(3).to_string())
print(e2[(e2.a=='early_fusion')&(e2.b=='tfidf_text')&(e2.metric.isin(['macro_f1','pr_auc','wr10']))].round(3).to_string())
v1=pd.read_csv(f'{R}/results/tfidf_late_fusion_100k_predictions.csv')
base=P('Enron_100000','tfidf_text')[['id','y_true']]
M={'v1_late':v1,'tfidf_text':P('Enron_100000','tfidf_text'),'nosw_only':P('Enron_100000','tfidf_oof_nosw_score_only'),'distil':P('Enron_100000','distilbert_text')}
A={k:base.merge(v[['id','y_pred','score']],on='id') for k,v in M.items()}
y=base.y_true.values
for k,a in A.items(): print(k,'flag',a.y_pred.sum(),'mF1',round(f1_score(y,a.y_pred,average='macro'),4),'PR',round(average_precision_score(y,a.score),4),'WR2',round(wr(y,a.score.values,2),3),'WR10',round(wr(y,a.score.values,10),4))
# equal-flag-count comparison: flag top 360 by each score
for k,a in A.items():
    o=np.argsort(-a.score.values,kind='stable'); yp=np.zeros(len(y),int); yp[o[:360]]=1
    print('top-360',k,'macroF1',round(f1_score(y,yp,average='macro'),4),'TP',int(yp[y==1].sum()))
rng=np.random.default_rng(42); D={('v1_late','nosw_only'):[],('v1_late','tfidf_text'):[],('distil','tfidf_text'):[]}
D2={k:[] for k in D}
for _ in range(1000):
    i=rng.integers(0,len(y),len(y))
    for (a,b) in D:
        D[(a,b)].append(average_precision_score(y[i],A[a].score.values[i])-average_precision_score(y[i],A[b].score.values[i]))
        D2[(a,b)].append(wr(y[i],A[a].score.values[i],2)-wr(y[i],A[b].score.values[i],2))
for k in D: print(k,'PR-AUC diff CI',np.round(np.percentile(D[k],[2.5,97.5]),3),'WR2 diff CI',np.round(np.percentile(D2[k],[2.5,97.5]),3))
