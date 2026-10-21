import sys,re,yaml,pandas as pd,numpy as np
R=sys.argv[1]
rules=yaml.safe_load(open(f'{R}/rules/phishing_rules_v2.yaml'))
h=pd.read_csv(f'{R}/results/human_review_annotation_template.csv')
rt=pd.read_csv(f'{R}/results/rule_triggers_100k.csv')
h=h.merge(rt,on='id',how='left')
W=rules['weights']
def hits(t,wb=False):
    t=t.lower(); out={}
    for g in ['core','urgency','action']:
        L=rules[f'{g}_indicators']
        if wb: out[g]=[k for k in L if re.search(r'(?<![a-z0-9])'+re.escape(k)+r'(?![a-z0-9])',t)]
        else: out[g]=[k for k in L if k in t]
    out['url']=re.search(rules['regex_indicators']['url'],t) is not None
    out['short']=re.search(rules['regex_indicators']['url_shorteners'],t) is not None
    out['benign']=any(k in t for k in rules['benign_context_indicators']) if not wb else any(re.search(r'(?<![a-z0-9])'+re.escape(k)+r'(?![a-z0-9])',t) for k in rules['benign_context_indicators'])
    return out
def score(o):
    s=W['core']*bool(o['core'])+W['urgency']*bool(o['urgency'])+W['action']*bool(o['action'])+W['url']*o['url']+W['url_shortener']*o['short']
    if not o['core'] and o['benign'] and s<=2: return 0
    return s
h['text']=h.text.fillna('')
rows=[]
for _,r in h.iterrows():
    a=hits(r.text); b=hits(r.text,True)
    rows.append(dict(id=r.id,case=r.case_type,y=r.y_true,s_sub=score(a),s_wb=score(b),
      urg_sub=bool(a['urgency']),urg_wb=bool(b['urgency']),core_sub=bool(a['core']),core_wb=bool(b['core']),
      act_sub=bool(a['action']),act_wb=bool(b['action']),
      urg_terms=';'.join(a['urgency']),core_terms=';'.join(a['core']),
      urg_spurious=';'.join(sorted(set(a['urgency'])-set(b['urgency']))),
      core_spurious=';'.join(sorted(set(a['core'])-set(b['core']))),
      rule_score_repo=r.rule_score))
d=pd.DataFrame(rows)
print('recomputed substring score == repo rule_score:', (d.s_sub==d.rule_score_repo).mean())
d['y_sub']=(d.s_sub>=3).astype(int); d['y_wb']=(d.s_wb>=3).astype(int)
print('label y_sub==y_true', (d.y_sub==d.y).mean())
print('\nUrgency hit: substring',d.urg_sub.sum(),'word-boundary',d.urg_wb.sum())
print('Core hit: substring',d.core_sub.sum(),'wb',d.core_wb.sum(), ' Action',d.act_sub.sum(),d.act_wb.sum())
print('spurious urgency terms:',pd.Series(';'.join(d.urg_spurious).split(';')).value_counts().drop('',errors='ignore').to_dict())
print('spurious core terms:',pd.Series(';'.join(d.core_spurious).split(';')).value_counts().drop('',errors='ignore').to_dict())
print('\nLabel flips under word boundary, by case:'); print(pd.crosstab(d.case,[d.y_sub,d.y_wb]))
pos=d[d.y==1]; print('positives',len(pos),'lose label under wb:',(pos.y_wb==0).sum())
# trigger position (lower bound in tokens via whitespace words)
def first_pos(t):
    t=t.lower(); ps=[]
    for g in ['core','urgency','action']:
        for k in rules[f'{g}_indicators']:
            i=t.find(k)
            if i>=0: ps.append((i,g))
    for k in ['url','url_shorteners']:
        m=re.search(rules['regex_indicators'][k],t)
        if m: ps.append((m.start(),k))
    return ps
res=[]
for _,r in h[h.y_true==1].iterrows():
    t=r.text; ps=first_pos(t)
    # earliest position at which score>=3 is reached (cumulative over distinct groups, first occurrence)
    groups={}
    for i,g in sorted(ps):
        groups.setdefault(g,i)
    order=sorted(groups.items(),key=lambda x:x[1]); s=0; reach=None
    for g,i in order:
        s+= {'core':2,'urgency':1,'action':1,'url':1,'url_shorteners':1}[g]
        if s>=3: reach=i; break
    nwords=len(t[:reach].split()) if reach is not None else None
    res.append(dict(case=r.case_type,reach_char=reach,words_before=nwords,total_words=len(t.split())))
p=pd.DataFrame(res)
print('\nPositives in sample:',len(p))
print('words before label-completing trigger: median',p.words_before.median(),'; >256 words (certainly beyond 256 tokens):',(p.words_before>256).sum(),'; >180 words:',(p.words_before>180).sum())
print(p.groupby('case').words_before.describe())
d.to_csv(sys.argv[2],index=False)
g=p.assign(b=p.words_before>256).groupby('case').b.agg(['sum','count']); print(g)
tp_rate=g.loc['TP','sum']/g.loc['TP','count']; est=tp_rate*167+g.loc['FN','sum']; print('weighted estimate of test positives beyond 256 words:',round(est,1),'of 191 =',round(est/191,3))
fl=d[d.y==1].groupby('case').apply(lambda x:((x.y_wb==0).sum(),len(x))); print(fl)
tpf=fl['TP'][0]/fl['TP'][1]; estf=tpf*167+fl['FN'][0]; print('weighted estimate of positives lost under word-boundary:',round(estf,1),'/191 =',round(estf/191,3))
neg=d[d.y==0]; print('negatives: urgency substring-only hits', (neg.urg_sub & ~neg.urg_wb).sum(),'of',len(neg))
