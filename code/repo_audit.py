import sys, pandas as pd, numpy as np
R=sys.argv[1]
S={}
for sc in ['Enron_10000','Enron_50000','Enron_100000']:
    for sp in ['train','val','test']:
        d=pd.read_csv(f'{R}/splits/{sc}/{sp}_ids.csv'); S[(sc,sp)]=d
        print(sc,sp,len(d),'cols',list(d.columns),'dup ids',d.iloc[:,0].duplicated().sum())
ids={k:set(v.iloc[:,0]) for k,v in S.items()}
for sc in ['Enron_10000','Enron_50000','Enron_100000']:
    tr,va,te=ids[(sc,'train')],ids[(sc,'val')],ids[(sc,'test')]
    print(sc,'train∩test',len(tr&te),'train∩val',len(tr&va),'val∩test',len(va&te))
all_={sc:ids[(sc,'train')]|ids[(sc,'val')]|ids[(sc,'test')] for sc in ['Enron_10000','Enron_50000','Enron_100000']}
a,b,c=all_['Enron_10000'],all_['Enron_50000'],all_['Enron_100000']
print('10k∩50k',len(a&b),'10k∩100k',len(a&c),'50k∩100k',len(b&c))
# expected overlap under independent sampling from 517401
N=517401; print('expected 10k∩100k',10000*100000/N,'50k∩100k',50000*100000/N)
# cross-scale leakage: 100k test ids present in 10k/50k train?  (relevant if models reused) 
print('100k test in 50k train',len(ids[('Enron_100000','test')]&ids[('Enron_50000','train')]))
p=pd.read_csv(f'{R}/results/tfidf_late_fusion_100k_predictions.csv')
print('pred ids == 100k test ids', set(p.id)==ids[('Enron_100000','test')])
r=pd.read_csv(f'{R}/results/rule_triggers_100k.csv')
print('rule ids == 100k all', set(r.id)==c, 'dup',r.id.duplicated().sum())
m=p.merge(r,on='id'); print('merged',len(m),'y agree',(m.y_true==m.y_bin).mean())
m['tp']=(m.y_pred==1)&(m.y_true==1); m['fp']=(m.y_pred==1)&(m.y_true==0); m['fn']=(m.y_pred==0)&(m.y_true==1)
print('\nBy rule_score: n, flagged, mean score');print(m.groupby('rule_score').agg(n=('id','size'),flagged=('y_pred','sum'),mean_score=('score','mean')).round(3))
pos=m[m.y_true==1]
pos['pattern']=np.select([pos.R_core==1],['core+'],default='no-core')
pos['pattern2']=pos.apply(lambda x:'+'.join([k[2:] for k in ['R_core','R_urgency','R_action','R_url','R_url_short'] if x[k]==1]),axis=1)
print('\nPositives by pattern: n, recall');print(pos.groupby('pattern2').agg(n=('id','size'),recall=('y_pred','mean'),mean_score=('score','mean')).sort_values('n',ascending=False).round(3))
neg=m[m.y_true==0]
neg['pattern2']=neg.apply(lambda x:'+'.join([k[2:] for k in ['R_core','R_urgency','R_action','R_url','R_url_short'] if x[k]==1]) or 'none',axis=1)
print('\nFalse positives by pattern');print(neg[neg.y_pred==1].pattern2.value_counts().head(12))
print('FP rule_score dist',neg[neg.y_pred==1].rule_score.value_counts().to_dict())
# rule score as ranker (sanity: it is the label)
from sklearn.metrics import roc_auc_score,average_precision_score
print('\nAUC of model score within rule_score==2 negatives vs ==0? n/a')
# near-miss: among negatives with score 2, model mean score vs score 0/1
# how much of model ranking is explained by rule_score: Spearman
from scipy.stats import spearmanr
print('spearman(model score, rule_score)',spearmanr(m.score,m.rule_score).correlation)
# ranking top 10% composition
top=m.sort_values('score',ascending=False).head(1000); print('top10% rule_score dist',top.rule_score.value_counts().sort_index().to_dict())
# TREC
t=pd.read_csv(f'{R}/results/cross_domain_trec07_predictions.csv'); print('\ntrec dup ids',t.id.duplicated().sum())
h=pd.read_csv(f'{R}/results/human_review_annotation_template.csv'); print('\nannot template rows',len(h),h.case_type.value_counts().to_dict(), 'ids in test', h.id.isin(p.id).mean())
print('annot nonnull',h[['triage_relevance','message_type','requires_context','notes']].notna().sum().to_dict())
print('text len median',h.text.str.len().median())
hs=pd.read_csv(f'{R}/results/human_review_sample.csv'); print('sample == template ids',set(hs.id)==set(h.id))
# exact dup text in the 174
print('dup texts in sample',h.text.duplicated().sum(), 'dup subjects',h.subject.duplicated().sum())
