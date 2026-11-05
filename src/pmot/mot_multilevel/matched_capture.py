"""Fixed-RK4 capture protocol for exact-geometry multilevel comparisons.

Only the physical public compiled ``mot_multilevel.step`` evaluates forces.
Capture definitions, velocity resolution, and timeout ladder match the saved
two-level benchmark. Every attempted node is retained for audit and resumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit

from .accelerated import step
from .matched_benchmark import VELOCITIES


AUDIT_COLUMNS=("speed_m_per_s","dt_s","duration_s","code","elapsed_s",
               "minimum_radius_m","final_radius_m","x_m","y_m","z_m",
               "vx_m_per_s","vy_m_per_s","vz_m_per_s","entered_core","core_entries")
TRAPPED_CODES=(1,2)
# Complete scalar boundary re-searches stop at 400 ms. Only unresolved
# direct-grid nodes may proceed to the 1 s and 2 s levels.
SCALAR_LEVELS=((.2,5e-6,2.5e-6),(.25,5e-6,2.5e-6),(.4,2.5e-6,1.25e-6))
NODE_LEVELS=SCALAR_LEVELS+((1.,1.25e-6,.625e-6),(2.,1.25e-6,.625e-6))


@njit(cache=True,nogil=True)
def classify_numeric(initial_position,initial_velocity,dt,duration,data):
    position=initial_position.copy();velocity=initial_velocity.copy()
    axis=np.array([0.,0.,1.]);elapsed=0.
    radius=np.sqrt(np.dot(position,position));minimum=radius
    was_inside=radius<=.002;entered=was_inside;entries=int(was_inside)
    inside_since=0. if was_inside else -1.
    code=-1
    for _ in range(int(np.ceil(duration/dt))+1):
        radius=np.sqrt(np.dot(position,position));minimum=min(minimum,radius)
        inside=radius<=.002;entered=entered or inside
        if inside and not was_inside:
            entries+=1;inside_since=elapsed
        elif not inside:inside_since=-1.
        was_inside=inside
        if entries>=2:
            code=1;break
        if inside_since>=0. and elapsed-inside_since>=.005:
            code=2;break
        if radius>=.03 and np.dot(position,velocity)>0.:
            code=0;break
        if elapsed>=duration-1e-15:break
        h=min(dt,duration-elapsed)
        position,velocity,axis=step(position,velocity,axis,h,data)
        elapsed+=h
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("nonfinite physical multilevel trajectory")
    radius=np.sqrt(np.dot(position,position));minimum=min(minimum,radius)
    return code,elapsed,minimum,radius,position,velocity,entered,entries


def record_numeric(speed,dt,duration,result):
    code,elapsed,minimum,radius,r,v,entered,entries=result
    return np.asarray([speed,dt,duration,code,elapsed,minimum,radius,*r,*v,int(entered),entries],float)


def definitive(code):
    return int(code) in (0,1,2)


def trapped(code):
    return int(code) in TRAPPED_CODES


def states_agree(a,b):
    return definitive(a) and definitive(b) and trapped(a)==trapped(b)


@dataclass(frozen=True)
class Boundary:
    lower:float
    upper:float
    lower_code:int
    upper_code:int
    clean:bool
    zero:bool
    reason:str


class RayEvaluator:
    def __init__(self,position,direction,data,records=(),checkpoint=None):
        self.position=np.asarray(position,float);self.direction=np.asarray(direction,float)
        self.data=data;self.records=[np.asarray(r,float) for r in records]
        self.cache={(round(float(r[0]),12),float(r[1]),float(r[2])):r for r in self.records}
        self.checkpoint=checkpoint

    def evaluate(self,speed,dt,duration):
        key=(round(float(speed),12),float(dt),float(duration))
        if key not in self.cache:
            result=classify_numeric(self.position,float(speed)*self.direction,float(dt),float(duration),self.data)
            row=record_numeric(float(speed),dt,duration,result)
            self.cache[key]=row;self.records.append(row)
            if self.checkpoint is not None and len(self.records)%32==0:self.checkpoint(self)
        return self.cache[key]

    def search(self,dt,duration):
        zero=self.evaluate(0.,dt,duration)
        if not trapped(zero[3]):
            self.evaluate(.25,dt,duration)
            return Boundary(0.,.25,int(zero[3]),int(self.evaluate(.25,dt,duration)[3]),
                            False,True,"zero-speed lower bound is not trapped")
        lower=0.;lower_code=int(zero[3]);upper=20.
        high=self.evaluate(upper,dt,duration)
        while trapped(high[3]) and upper<80.:
            lower=upper;lower_code=int(high[3]);upper=min(80.,2*upper)
            high=self.evaluate(upper,dt,duration)
        if trapped(high[3]):
            return Boundary(lower,upper,lower_code,int(high[3]),False,False,"capture reaches velocity-domain boundary")
        if not definitive(high[3]):
            return Boundary(lower,upper,lower_code,int(high[3]),False,False,"upper bound is indeterminate")
        while upper-lower>.25:
            mid=round((lower+upper)/2,12);row=self.evaluate(mid,dt,duration)
            if not definitive(row[3]):
                return Boundary(lower,upper,lower_code,int(row[3]),False,False,"scalar search encountered timeout")
            if trapped(row[3]):lower=mid;lower_code=int(row[3])
            else:upper=mid
        return Boundary(lower,upper,lower_code,0,True,False,"definitive trapped/escaped bracket")

    def direct_mask(self,start_level=0):
        mask=np.full(len(VELOCITIES),-2,dtype=np.int8)
        for j,speed in enumerate(VELOCITIES):
            for level in range(start_level,len(NODE_LEVELS)):
                duration,coarse,fine=NODE_LEVELS[level]
                a=self.evaluate(speed,coarse,duration);b=self.evaluate(speed,fine,duration)
                if states_agree(a[3],b[3]):
                    mask[j]=int(trapped(b[3]));break
            if mask[j]<0:
                if speed==0. and a[3]==-1 and b[3]==-1:
                    # All positive speeds must still resolve before this mask
                    # is accepted. This literal zero has exactly zero flux.
                    mask[j]=-1
                else:
                    raise RuntimeError(f"unresolved positive-flux node at {speed:g} m/s after complete audit ladder")
        if mask[-1]!=0 or np.any(mask[-20:]>0):
            raise RuntimeError("capture approaches the saved velocity-domain boundary; dense extension required")
        return mask


def compatible(a,b):
    return (a.clean and b.clean and not a.zero and not b.zero
            and max(a.lower,b.lower)<min(a.upper,b.upper)
            and abs(a.lower-b.lower)<=.25 and abs(a.upper-b.upper)<=.25)


def solve_ray(evaluator,*,representative_grid=False):
    """Return authoritative grid mask plus the scalar/direct evidence type."""
    start=perf_counter()
    base=evaluator.search(5e-6,.05)
    mode="scalar_boundary";boundary=base;diagnostics=[]
    if base.zero:
        mask=evaluator.direct_mask()
        mode="direct_zero_threshold_grid"
    else:
        # The saved lower and upper boundaries are always checked at 200 ms
        # with both steps, including otherwise clean positive base searches.
        audits=[evaluator.evaluate(speed,dt,.2) for speed in (base.lower,base.upper) for dt in (5e-6,2.5e-6)]
        clean=(base.clean and all(trapped(r[3]) for r in audits[:2]) and all(r[3]==0 for r in audits[2:]))
        if not clean:
            # Once a scalar premise fails, a direct mask is authoritative even
            # if a complete longer scalar search subsequently becomes clean.
            level=0
            for level,(duration,coarse,fine) in enumerate(SCALAR_LEVELS):
                a=evaluator.search(coarse,duration);b=evaluator.search(fine,duration)
                diagnostics.append(dict(duration_s=duration,coarse=asdict_boundary(a),fine=asdict_boundary(b)))
                boundary=b
                if compatible(a,b):break
            mask=evaluator.direct_mask(start_level=level)
            mode="direct_failed_positive_boundary_grid"
        elif representative_grid:
            mask=evaluator.direct_mask()
            expected=np.asarray(VELOCITIES<=base.lower+1e-12,dtype=np.int8)
            if np.any(mask[1:]!=expected[1:]):mode="direct_representative_grid_disagrees_with_scalar"
            else:mode="direct_representative_monotonicity_grid"
        else:
            if base.upper>=75.:
                raise RuntimeError("scalar capture boundary approaches saved spectrum endpoint")
            mask=np.asarray(VELOCITIES<=base.lower+1e-12,dtype=np.int8)
    if np.any(mask[1:]<0):raise RuntimeError("unresolved positive-flux capture state")
    return dict(mask=mask,mode=mode,boundary=asdict_boundary(boundary),complete_searches=diagnostics,
                nonmonotone=bool(np.any(np.diff(mask[1:].astype(int))>0)),
                elapsed_wall_s=perf_counter()-start)


def asdict_boundary(boundary):
    return dict(lower=boundary.lower,upper=boundary.upper,lower_code=boundary.lower_code,
                upper_code=boundary.upper_code,clean=boundary.clean,zero=boundary.zero,reason=boundary.reason)
