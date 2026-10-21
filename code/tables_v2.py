import sys,pandas as pd,numpy as np
V=sys.argv[1]; pd.set_option('display.width',250)
e=pd.read_csv(f'{V}/results/E2_bootstrap_ci_all.csv')
for (sc,mo),g in e.groupby(['scale','model'],sort=False):
    print(sc,mo,'  '.join(f"{r.metric}={r.point:.3f}[{r.ci_low:.3f},{r.ci_high:.3f}]" for r in g.itertuples()))
lc=pd.read_csv(f'{V}/results/E3_nested_learning_curve_tfidf.csv')
print(lc.groupby('n_train')[['macro_f1','pos_recall','pos_f1','pr_auc','wr10','wr2']].agg(['mean','std']).round(4).to_string())
ss=pd.read_csv(f'{V}/results/E3_split_seed_variability_tfidf.csv'); print(ss.groupby('scale')[['macro_f1','pos_recall','pr_auc','wr10']].agg(['mean','std','min','max']).round(3).to_string())
st=pd.read_csv(f'{V}/results/E4_clean_stacking.csv'); print(st[['scale','config','macro_f1','pos_recall','pos_f1','pr_auc','wr10','tp']].round(3).to_string())
def wr(y,s,k):
    n=int(len(y)*k/100); o=np.argsort(-s,kind='stable'); return y[o][:n].sum()/y.sum()
for c in ['tfidf_text','distilbert_text']:
    p=pd.read_csv(f'{V}/predictions/Enron_100000_{c}_test_predictions.csv'); y=p.y_true.values; s=p.score.values
    print(c,'curve',[(k,round(wr(y,s,k),4)) for k in [0.5,1,1.5,2,3,4,5,7.5,10,15,20,25,30]])
    rng=np.random.default_rng(42); B={k:[] for k in [2,5,10,20]}
    for _ in range(1000):
        i=rng.integers(0,len(y),len(y))
        for k in B: B[k].append(wr(y[i],s[i],k))
    print(c,{k:(round(wr(y,s,k),3),np.round(np.percentile(v,[2.5,97.5]),3).tolist()) for k,v in B.items()})
    o=np.argsort(-s,kind='stable'); cc=np.cumsum(y[o])/y.sum()
    print(c,'effort for 80/90/95/100%',[round((np.argmax(cc>=r)+1)/len(y)*100,2) for r in [.8,.9,.95,1.0]])
