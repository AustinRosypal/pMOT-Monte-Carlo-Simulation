"""Disposable timing prototype, excluded from production evidence/statistics."""
from __future__ import annotations
from collections import Counter
from time import perf_counter
import numpy as np

from pmot.mot_multilevel import matched_campaign as campaign
from pmot.mot_multilevel import matched_benchmark as b
from pmot.mot_multilevel.matched_capture import RayEvaluator,solve_ray,NODE_LEVELS,states_agree,trapped
from pmot.mot_multilevel.adaptive import integrate


class HybridTimingEvaluator(RayEvaluator):
    def __init__(self,r,d,data):
        super().__init__(r,d,data);self.adaptive_search=False;self.adaptive_cache={};self.methods=Counter()

    def adaptive(self,speed,dt,duration):
        key=(float(speed),float(dt),float(duration))
        if key not in self.adaptive_cache:
            refinement=5e-6/dt
            result=integrate(self.position,speed*self.direction,duration,self.data,refinement)
            row=np.zeros(15);row[:5]=(speed,dt,duration,result[0],result[1])
            # Only classification/time are consumed. This prototype does NOT
            # publish these placeholder rows as fixed-RK4 trajectory evidence.
            self.adaptive_cache[key]=row;self.methods['adaptive']+=1
        return self.adaptive_cache[key]

    def evaluate(self,speed,dt,duration):
        if self.adaptive_search:return self.adaptive(speed,dt,duration)
        self.methods['fixed_calls']+=1
        return super().evaluate(speed,dt,duration)

    def search(self,dt,duration):
        self.adaptive_search=True
        try:return super().search(dt,duration)
        finally:self.adaptive_search=False

    def fixed_node(self,speed,start_level):
        for duration,coarse,fine in NODE_LEVELS[start_level:]:
            a=self.evaluate(speed,coarse,duration);f=self.evaluate(speed,fine,duration)
            if states_agree(a[3],f[3]):return int(trapped(f[3]))
        if speed==0 and a[3]==f[3]==-1:return -1
        raise RuntimeError('positive flux node did not resolve')

    def direct_mask(self,start_level=0):
        mask=np.empty(len(b.VELOCITIES),dtype=np.int8)
        for j,speed in enumerate(b.VELOCITIES):
            if speed==0:mask[j]=self.fixed_node(speed,start_level);continue
            for duration,coarse,fine in NODE_LEVELS[start_level:]:
                a=self.adaptive(speed,coarse,duration);f=self.adaptive(speed,fine,duration)
                if states_agree(a[3],f[3]):mask[j]=int(trapped(f[3]));break
            else:mask[j]=self.fixed_node(speed,start_level)
        checked={0}
        while True:
            indices=set()
            for j in np.flatnonzero(np.diff(mask.astype(int))!=0):
                indices.update(range(max(0,j-1),min(len(mask),j+3)))
            indices-=checked
            if not indices:break
            for j in sorted(indices):mask[j]=self.fixed_node(b.VELOCITIES[j],start_level);checked.add(j)
        if np.any(mask[-20:]>0):raise RuntimeError('tail needs extension')
        return mask


def main():
    manifest=b.read_json(b.STATISTICS/'campaign_manifest.json')
    assert manifest['source_sha256']==b.source_hashes()
    point=campaign.points_and_scans()[0][0];data=b.prepared(point)
    positions,directions,_=b.load_master_geometry();folder=b.CACHE/'production/configuration_000/rays'
    integrate(positions[0,0],directions[0,0]*10.,.001,data,1.)
    rows=[]
    for d,p in ((d,p) for d in (0,40,80,120,160,200,240,280) for p in (0,25)):
        with np.load(folder/f'd{d:04d}_p{p:02d}.npz') as z:
            assert str(z['status'])=='completed'
            reference=z['mask'].copy();prior_wall=float(z['elapsed_wall_s']);mode=str(z['mode'])
        r,direction=positions[d,p],directions[d,p];rep=p==0 and d%40==0
        # Interleave warm full-ray fixed and proposed hybrid evaluations.
        started=perf_counter();fixed=solve_ray(RayEvaluator(r,direction,data),representative_grid=rep)
        fixed_wall=perf_counter()-started
        evaluator=HybridTimingEvaluator(r,direction,data)
        started=perf_counter();hybrid=solve_ray(evaluator,representative_grid=rep)
        hybrid_wall=perf_counter()-started
        row=dict(disc=d,point=p,production_mode=mode,prior_worker_wall_s=prior_wall,
                 fixed_wall_s=fixed_wall,hybrid_wall_s=hybrid_wall,
                 fixed_reproduces_saved_mask=bool(np.array_equal(fixed['mask'],reference)),
                 hybrid_differing_positive_nodes=int(np.sum(hybrid['mask'][1:]!=reference[1:])),
                 hybrid_mode=hybrid['mode'],methods=dict(evaluator.methods))
        rows.append(row)
        result=dict(status='running',rows=rows)
        b.write_json(b.STATISTICS/'validation/hybrid_ray_timing_probe.json',result)
        print(row,flush=True)
    totals={name:sum(r[name] for r in rows) for name in ('fixed_wall_s','hybrid_wall_s')}
    totals['speedup']=totals['fixed_wall_s']/totals['hybrid_wall_s']
    result=dict(status='completed',created_utc=b.now(),purpose='Timing prototype only, excluded from production',
                rows=rows,totals=totals,source_unchanged=manifest['source_sha256']==b.source_hashes(),
                caveat='Small selected baseline-ray sample, overrepresents representative full masks. Wider matched-setting and numerical audits remain required.')
    b.write_json(b.STATISTICS/'validation/hybrid_ray_timing_probe.json',result)
    print(totals,flush=True)


if __name__=='__main__':main()
