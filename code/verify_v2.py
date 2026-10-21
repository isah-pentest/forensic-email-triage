import sys,glob,os,pandas as pd,numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,average_precision_score,roc_auc_score
V=sys.argv[1]; R=sys.argv[2]
def wr(y,s,k):
    n=int(len(y)*k/100); o=np.argsort(-s,kind='stable'); return y[o][:n].sum()/y.sum()
rows=[]
for f in sorted(glob.glob(f'{V}/predictions/*.csv')):
    nm=os.path.basename(f).replace('_test_predictions.csv',''); sc,cfg=nm.split('_',2)[0]+'_'+nm.split('_',2)[1],nm.split('_',2)[2]
    p=pd.read_csv(f); y=p.y_true.values; s=p.score.values; yp=p.y_pred.values
    ids=set(pd.read_csv(f'{R}/splits/{sc}/test_ids.csv').id)
    rows.append(dict(scale=sc,config=cfg,n=len(p),ids_ok=set(p.id)==ids,npos=y.sum(),tp=int(((yp==1)&(y==1)).sum()),flag=int(yp.sum()),
      macro_f1=f1_score(y,yp,average='macro'),recall=recall_score(y,yp),prec=precision_score(y,yp,zero_division=0),pos_f1=f1_score(y,yp),
      pr_auc=average_precision_score(y,s),roc=roc_auc_score(y,s),wr2=wr(y,s,2),wr5=wr(y,s,5),wr10=wr(y,s,10),wr20=wr(y,s,20),
      thr_consistent=((s>=0.5).astype(int)==yp).mean()))
m=pd.DataFrame(rows); pd.set_option('display.width',250)
print(m.round(4).to_string())
m.to_csv(sys.argv[3],index=False)
# y_true consistency across configs per scale
for sc,g in m.groupby('scale'): print(sc,'npos unique',g.npos.unique())
rt=pd.read_csv(f'{R}/results/rule_triggers_100k.csv')[['id','y_bin']]
p=pd.read_csv(f'{V}/predictions/Enron_100000_tfidf_text_test_predictions.csv').merge(rt,on='id'); print('100k y_true==rule y_bin',(p.y_true==p.y_bin).mean())
# v1 late fusion vs v2
v1=pd.read_csv(f'{R}/results/tfidf_late_fusion_100k_predictions.csv')
print('v1 latefusion ids == v2 ids', set(v1.id)==set(p.id))
E2=pd.read_csv(f'{V}/results/E2_bootstrap_ci_all.csv'); print(E2.head(3))
chk=E2.merge(m.rename(columns={'config':'model'}).melt(id_vars=['scale','model'],value_vars=['macro_f1','pos_f1','pr_auc','wr10','wr20']).rename(columns={'variable':'metric','value':'mine'}),on=['scale','model','metric'])
print('E2 point vs my recompute max abs diff', (chk.point-chk.mine).abs().max())
