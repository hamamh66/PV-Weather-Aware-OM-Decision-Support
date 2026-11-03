# Exact propagation of the per-seed test confusion matrices (MiT-B0 weather, Swin-Tiny PV module state) through SRM rule bases v1-v3.
# Run from the repository root:  python decision_analysis/propagate_errors.py
import json,numpy as np
V=json.load(open('decision_analysis/srm_versions.json')); R='results/'
DS=json.load(open(R+'data_summary.json'))
RISK={'LOW':0,'MEDIUM':1,'HIGH':2}
keys=list(V['v2']); W=sorted({k.split('-',1)[0] for k in keys},key=lambda x:[k.split('-',1)[0] for k in keys].index(x)); PNL=[]
[PNL.append(k.split('-',1)[1]) for k in keys if k.split('-',1)[1] not in PNL]
nw=lambda s:{x.lower():x for x in W}[s.lower()]
np_=lambda s:{x.lower():x for x in PNL}[s.lower().replace('-','_')]
cw=[nw(c) for c in DS['weather']['classes']]; cp=[np_(c) for c in DS['panel']['classes']]
DEC=lambda S,w,p:(S[f'{w}-{p}']['Risk'],S[f'{w}-{p}']['Human'],S[f'{w}-{p}']['AI'])
def mode(L): return max(L,key=L.count)  # deterministic: first among ties
def run(S,seed):
    Cw=np.array(json.load(open(R+f'runs/weather_MiT-B0_s{seed}.json'))['confusion'],float)
    Cp=np.array(json.load(open(R+f'runs/panel_Swin-Tiny_s{seed}.json'))['confusion'],float)
    Jw=Cw/Cw.sum(); Jp=Cp/Cp.sum()  # joint P(true,pred)
    md={p:mode([DEC(S,w,p) for w in W]) for p in PNL}
    o=dict(both_correct=0,decision_exact=0,risk_agree=0,risk_under=0,panel_only_decision_exact=0,panel_only_risk_under=0)
    for a in range(len(cw)):
     for b in range(len(cw)):
      for c in range(len(cp)):
       for d in range(len(cp)):
        pr=Jw[a,b]*Jp[c,d]
        if pr==0: continue
        t=DEC(S,cw[a],cp[c]); p=DEC(S,cw[b],cp[d]); po=md[cp[d]]
        o['both_correct']+=pr*(a==b and c==d); o['decision_exact']+=pr*(t==p)
        o['risk_agree']+=pr*(t[0]==p[0]); o['risk_under']+=pr*(RISK[p[0]]<RISK[t[0]])
        o['panel_only_decision_exact']+=pr*(t==po); o['panel_only_risk_under']+=pr*(RISK[po[0]]<RISK[t[0]])
    return o
out={}
for v in ('v1','v2','v3'):
    P=[run(V[v],s) for s in range(5)]
    out[v]={k:dict(mean=float(np.mean([p[k] for p in P])),sd=float(np.std([p[k] for p in P],ddof=1))) for k in P[0]}
    S=V[v]; dd=dr=dl=0
    for p in PNL:
        ds=[DEC(S,w,p) for w in W]; rs=[x[0] for x in ds]; m=mode(ds); mr=mode(rs)
        dd+=sum(x!=m for x in ds); dr+=sum(x!=mr for x in rs); dl+=sum(RISK[mr]<RISK[x] for x in rs)
    out[v]['weather_dependence']=dict(decision_differs=dd,risk_differs=dr,risk_lower=dl)
    print(v,{k:(round(x['mean'],4) if isinstance(x,dict) and 'mean' in x else x) for k,x in out[v].items()})
json.dump(out,open('decision_analysis/propagation_v1v2v3.json','w'),indent=1)
