import pickle
from multiprocessing import Pool
from aepso import run
SEEDS=list(range(1000,1030)); jobs=[]
for c in ("NA2","ADA"):
    for tag,kw in (("E6:trigger",dict(trigger=True)),("E6:trigger+tau_mem25",dict(trigger=True,tau_mem=25.0))):
        for sc in ("stationary","shift"):
            for s in SEEDS: jobs.append((f"{tag} {c}",c,sc,s,kw))
def work(j):
    tag,c,sc,s,kw=j; r=run(c,sc,s,**kw)
    return tag,sc,dict(final_regret=r["final_regret"],false_canon=r["false_canon"],early_regret=r["early_regret"],post_shift_regret=r["post_shift_regret"],diversity=r["diversity"],regret=r["regret"],err_final=r["error"][-1],ntrig=len(r["triggers"]))
if __name__=="__main__":
    with Pool() as p: res=p.map(work,jobs,chunksize=8)
    out={}
    for tag,sc,d in res: out.setdefault((sc,tag),[]).append(d)
    pickle.dump(out,open("e6_results.pkl","wb"))
    import numpy as np
    for k,v in sorted(out.items()):
        f=np.array([r['final_regret'] for r in v]); print(f"{k[0]:10s} {k[1]:32s} final {f.mean():.3f} false {int(sum(r['false_canon'] for r in v)):2d} postshift {np.mean([r['post_shift_regret'] for r in v]):.3f} triggers {np.mean([r['ntrig'] for r in v]):.1f}")
