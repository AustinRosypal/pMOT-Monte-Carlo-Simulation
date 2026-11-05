"""Hybrid classifier with separately labelled adaptive and fixed evidence."""
from collections import Counter
import numpy as np
from ..adaptive import integrate
from ..matched_capture import RayEvaluator,solve_ray,NODE_LEVELS,states_agree,trapped
from . import config as b

ADAPTIVE_COLUMNS=('speed_m_per_s','refinement','duration_s','code','elapsed_s',
                  'x_m','y_m','z_m','vx_m_per_s','vy_m_per_s','vz_m_per_s',
                  'first_trapped_s','accepted_steps','rejected_steps')


class HybridEvaluator(RayEvaluator):
    def __init__(self,position,direction,data,records=(),adaptive_records=(),checkpoint=None):
        super().__init__(position,direction,data,records,checkpoint=None)
        self.adaptive_records=[np.asarray(r,float) for r in adaptive_records]
        self.adaptive_cache={(round(float(r[0]),12),float(r[1]),float(r[2])):r for r in self.adaptive_records}
        self.searching=False;self.checkpoint=checkpoint;self.total_calls=0

    def checkpoint_if_needed(self):
        self.total_calls+=1
        if self.checkpoint is not None and self.total_calls%32==0:self.checkpoint(self)

    def fixed(self,speed,dt,duration):
        callback=self.checkpoint;self.checkpoint=None
        try:result=super().evaluate(speed,dt,duration)
        finally:self.checkpoint=callback
        self.checkpoint_if_needed()
        return result

    def adaptive(self,speed,dt,duration):
        refinement=5e-6/dt;key=(round(float(speed),12),refinement,float(duration))
        if key not in self.adaptive_cache:
            result=integrate(self.position,speed*self.direction,duration,self.data,refinement)
            row=np.asarray([speed,refinement,duration,result[0],result[1],*result[2],*result[3],result[4],result[6],result[7]],float)
            self.adaptive_cache[key]=row;self.adaptive_records.append(row)
            self.checkpoint_if_needed()
        # Search consumes only speed/step/duration/code/time. This temporary
        # view is NEVER stored as fixed-RK4/minimum-radius/core-entry evidence.
        view=np.zeros(15);view[:5]=self.adaptive_cache[key][:5];view[1]=dt
        return view

    def evaluate(self,speed,dt,duration):
        return self.adaptive(speed,dt,duration) if self.searching else self.fixed(speed,dt,duration)

    def search(self,dt,duration):
        self.searching=True
        try:return super().search(dt,duration)
        finally:self.searching=False

    def fixed_node(self,speed,start_level=0):
        for duration,coarse,fine in NODE_LEVELS[start_level:]:
            a=self.fixed(speed,coarse,duration);f=self.fixed(speed,fine,duration)
            if states_agree(a[3],f[3]):return int(trapped(f[3]))
        if speed==0. and a[3]==f[3]==-1:return -1
        raise RuntimeError(f'unresolved positive-flux node {speed:g} m/s after full fixed audit ladder')

    def direct_mask(self,start_level=0):
        mask=np.empty(len(b.VELOCITIES),dtype=np.int8)
        for j,speed in enumerate(b.VELOCITIES):
            if speed==0.:mask[j]=self.fixed_node(speed,start_level);continue
            duration,coarse,fine=NODE_LEVELS[start_level]
            a=self.adaptive(speed,coarse,duration);f=self.adaptive(speed,fine,duration)
            mask[j]=int(trapped(f[3])) if states_agree(a[3],f[3]) else self.fixed_node(speed,start_level)
        # Every transition and its neighbours receives fixed-step resolution.
        # Repeat after corrections so shifted edges are followed to closure.
        checked={0}
        while True:
            indices=set()
            for j in np.flatnonzero(np.diff(mask.astype(int))!=0):
                indices.update(range(max(0,j-1),min(len(mask),j+3)))
            indices-=checked
            if not indices:break
            for j in sorted(indices):mask[j]=self.fixed_node(float(b.VELOCITIES[j]),start_level);checked.add(j)
        if np.any(mask[1:]<0):raise RuntimeError('unresolved positive-flux hybrid capture node')
        if np.any(mask[-20:]>0):raise RuntimeError('capture approaches spectrum endpoint; dense extension required')
        return mask


def solve(evaluator,representative=False):
    return solve_ray(evaluator,representative_grid=representative)
