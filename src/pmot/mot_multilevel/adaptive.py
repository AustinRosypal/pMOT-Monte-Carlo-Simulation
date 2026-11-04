"""Embedded Dormand--Prince trajectories using the exact population-rate force.

No interpolation or scalar capture-boundary assumption. Core events remain
sampled events, checked with independent tighter tolerances and step caps.
"""
import numpy as np
from numba import njit
from .accelerated import acceleration, magnetic_field


@njit(cache=True)
def derivative(y, axis, data, acceleration_function=acceleration):
    out=np.empty(6)
    out[:3]=y[3:]
    out[3:]=acceleration_function(y[:3],y[3:],axis,data)
    return out


@njit(cache=True)
def embedded_step(y,axis,h,data,acceleration_function=acceleration):
    k1=derivative(y,axis,data,acceleration_function)
    k2=derivative(y+h*(1/5)*k1,axis,data,acceleration_function)
    k3=derivative(y+h*(3/40*k1+9/40*k2),axis,data,acceleration_function)
    k4=derivative(y+h*(44/45*k1-56/15*k2+32/9*k3),axis,data,acceleration_function)
    k5=derivative(y+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),axis,data,acceleration_function)
    k6=derivative(y+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),axis,data,acceleration_function)
    result=y+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
    k7=derivative(result,axis,data,acceleration_function)
    error=h*((35/384-5179/57600)*k1+(500/1113-7571/16695)*k3
             +(125/192-393/640)*k4+(-2187/6784+92097/339200)*k5
             +(11/84-187/2100)*k6-1/40*k7)
    return result,error


@njit(cache=True)
def integrate(r0,v0,duration,data,refinement=1.,retain=False):
    return integrate_local_environment(r0,v0,duration,data,acceleration,
                                       field_from_data,escaped_from_data,refinement,retain)


@njit(cache=True)
def field_from_data(position,data):
    return magnetic_field(position,data[-1])


@njit(cache=True)
def escaped_from_data(position,velocity,data):
    return np.linalg.norm(position)>=.03 and np.dot(position,velocity)>0


@njit(cache=True)
def unlimited_step(position,velocity,data):
    return 1.0


@njit(cache=True)
def integrate_local_environment(r0,v0,duration,data,acceleration_function,
                                 field_function,escape_function,refinement=1.,retain=False,
                                 step_limit_function=unlimited_step):
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
    for attempt in range(2000000):
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
