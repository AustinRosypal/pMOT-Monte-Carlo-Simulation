"""Prospective iteration-budget extension; the public integrator body is unchanged.

The only body change is the finite attempt budget parameter. Original failures
and the frozen kernel stay untouched; tests compare the complete AST bodies.
"""
import numpy as np
from numba import njit
from pmot.mot_multilevel.adaptive import (acceleration, field_from_data,
    escaped_from_data, unlimited_step, embedded_step)
from pmot.mot_strange import accelerated_surrogate as frozen


@njit(cache=False)
def integrate_budgeted(r0,v0,duration,data,acceleration_function,
                                 field_function,escape_function,refinement=1.,retain=False,
                                 step_limit_function=unlimited_step, max_attempts=6000000):
    """Return code,time,r,v,first trap,final-window radius,accepted,rejected.

    Refinement=2 halves step caps and tightens all local error scales fourfold.
    A timeout always remains code -1. Retention does not exit at early capture.
    """
    y=np.concatenate((r0,v0))
    axis=np.array([0.,0.,1.])
    t=0.;h=5e-6
    inside=np.linalg.norm(r0)<=.002
    entries=int(inside);was_inside=inside
    inside_since=0. if inside else -1.
    first=-1.;maximum=0.
    accepted=0;rejected=0
    tolerance=refinement*refinement
    for attempt in range(max_attempts):
        radius=np.linalg.norm(y[:3])
        inside=radius<=.002
        if inside and not was_inside:
            entries+=1;inside_since=t
        elif not inside:
            inside_since=-1.
        was_inside=inside
        trapped=entries>=2 or (inside_since>=0 and t-inside_since>=.005)
        if trapped and first<0:first=t
        if trapped and not retain:
            return 1,t,y[:3],y[3:],first,maximum,accepted,rejected
        if retain and t>=duration-.025:maximum=max(maximum,radius)
        if escape_function(y[:3],y[3:],data):
            return 0,t,y[:3],y[3:],first,maximum,accepted,rejected
        if t>=duration-1e-12:
            return -1,t,y[:3],y[3:],first,maximum,accepted,rejected
        # Resolve the core boundary without imposing a micron step everywhere.
        speed=np.linalg.norm(y[3:])
        event_cap=max(.0001,abs(radius-.002)/2)/max(speed,1e-12)/refinement
        cap=(.0001 if radius<.003 else .0005)/refinement
        h=min(h,cap,event_cap,duration-t,step_limit_function(y[:3],y[3:],data)/refinement)
        if retain and t<duration-.025-1e-14:h=min(h,duration-.025-t)
        if h<1e-14:raise ValueError('adaptive trajectory step underflow')
        yn,error=embedded_step(y,axis,h,data,acceleration_function)
        norm=0.
        for i in range(6):
            atol=1e-8 if i<3 else 1e-5
            scale=(atol+1e-5*max(abs(y[i]),abs(yn[i])))/tolerance
            norm=max(norm,abs(error[i])/scale)
        if not np.isfinite(norm):raise ValueError('nonfinite adaptive trajectory')
        if norm<=1:
            y=yn;t+=h;accepted+=1
            field=field_function(y[:3],data);mag=np.linalg.norm(field)
            if mag>data[10][3]:axis=field/mag
        else:rejected+=1
        factor=5. if norm==0 else min(5.,max(.2,.9*norm**(-.2)))
        h*=factor
    raise ValueError('adaptive trajectory iteration limit')


def retained_adaptive(r, v, duration, data, refinement, max_attempts=6000000):
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 50000000:
        raise ValueError('Explicit finite attempt budget must be 1..50000000')
    return integrate_budgeted(r, v, duration, data, frozen.acceleration,
        frozen.field, frozen.escaped, refinement, True, frozen.wall_step_limit,
        max_attempts)

