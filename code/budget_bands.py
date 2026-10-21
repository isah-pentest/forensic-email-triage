import sys,pandas as pd,numpy as np
V=sys.argv[1]
ks=np.round(np.arange(0,15.01,0.25),2)
def curve(y,s):
    o=np.argsort(-s,kind='stable'); c=np.concatenate([[0],np.cumsum(y[o])])/y.sum(); n=len(y)
    return np.array([c[int(n*k/100)] for k in ks])
out={}
for c in ['tfidf_text','distilbert_text']:
    p=pd.read_csv(f'{V}/predictions/Enron_100000_{c}_test_predictions.csv'); y=p.y_true.values; s=p.score.values
    pt=curve(y,s); rng=np.random.default_rng(42); B=np.array([curve(y[i],s[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(1000))])
    lo,hi=np.percentile(B,[2.5,97.5],axis=0)
    out[c]=(pt,lo,hi)
for c,(pt,lo,hi) in out.items():
    for nm,arr in [('pt',pt),('lo',lo),('hi',hi)]:
        print(f'% {c} {nm}'); print(' '.join(f'({k:g},{v:.4f})' for k,v in zip(ks,arr)))
