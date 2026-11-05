"""Independent diagnostic of worker input identity and saved-node replay."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import numpy as np
from pmot.mot_multilevel.matched_fast import campaign as c
from pmot.mot_multilevel.matched_fast import config as b
from pmot.mot_multilevel.accelerated import acceleration
from pmot.mot_multilevel.matched_capture import classify_numeric


def digest(data):
    h=hashlib.sha256()
    for a in data:
        h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


def worker(sequence):
    c.worker_init();values=[]
    for index in sequence:
        data=c.worker_data(index)
        a=acceleration(np.array([.001,-.0007,.0004]),np.array([1.,-.4,.7]),np.array([0.,0.,1.]),data)
        values.append(dict(index=index,digest=digest(data),acceleration=a.tolist()))
    with np.load(b.CACHE/'pilot/configuration_045/rays/d0838_p02.npz') as z:
        r=z['position_m'].copy();direction=z['direction'].copy()
    result=classify_numeric(r,.25*direction,5e-6,.2,c.worker_data(45))
    return dict(data=values,ray=dict(code=int(result[0]),time=float(result[1]),position=result[4].tolist()))


def main():
    sequence=list(range(124));parent={}
    for i in sequence:
        data=b.prepared(b.definitions()[0][i]);a=acceleration(np.array([.001,-.0007,.0004]),np.array([1.,-.4,.7]),np.array([0.,0.,1.]),data)
        parent[i]=dict(digest=digest(data),acceleration=a.tolist())
    with ProcessPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(worker,[sequence,sequence]))
    for j,output in enumerate(outputs):
        differences=[r for r in output['data'] if r['digest']!=parent[r['index']]['digest'] or not np.allclose(r['acceleration'],parent[r['index']]['acceleration'],rtol=1e-10,atol=1e-12)]
        print('worker',j,'differences',differences,'ray',output['ray'],flush=True)
    b.write_json(b.STATISTICS/'diagnostics/worker_replay.json',dict(parent=parent,workers=outputs))


if __name__=='__main__':main()
