"""Independent event and longer-trajectory checks for the new population kernel."""
from dataclasses import replace
from time import perf_counter

import numpy as np
from numba import njit

from ..configuration import HBAR_J_S
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import default_multilevel_mot_config
from .rate_equations import build_rate_equation_model, rate_equation_observable
from .simulation import build_multilevel_mot_beams
from .accelerated import prepare, classify, observable
from .relationship_campaign import paths, write_json


@njit(cache=True)
def event_occupations(matrix, initial, duration, replicates, seed):
    """Gillespie internal-state trajectories for a frozen phase-space point."""
    np.random.seed(seed)
    n=len(initial)
    output=np.zeros((replicates,n))
    for replicate in range(replicates):
        u=np.random.random()
        cumulative=0.
        state=n-1
        for i in range(n):
            cumulative+=initial[i]
            if u<cumulative:
                state=i
                break
        time=0.
        while time<duration:
            rate=-matrix[state,state]
            if rate<=0:
                output[replicate,state]+=duration-time
                break
            wait=-np.log(max(np.random.random(),1e-300))/rate
            output[replicate,state]+=min(wait,duration-time)
            time+=wait
            if time>=duration:
                break
            threshold=np.random.random()*rate
            cumulative=0.
            for destination in range(n):
                if destination!=state:
                    cumulative+=matrix[destination,state]
                    if threshold<cumulative:
                        state=destination
                        break
    return output/duration


def run_validation():
    model=build_rate_equation_model()
    coil=default_anti_helmholtz_config()
    cfg=default_multilevel_mot_config()
    beams=build_multilevel_mot_beams(config=cfg)
    data=prepare(cfg)
    cases=[]
    for i,(r,v) in enumerate((([0,0,0],[0,0,0]),([.001,0,0],[0,0,0]),
                              ([0,0,0],[3,0,0]),([.006,.002,-.001],[-8,.2,0]))):
        ref=rate_equation_observable(model,beams,tuple(r),tuple(v),coil,cfg,
            store_rate_matrix=True,store_beam_transition_quantities=True)
        estimates=event_occupations(ref.rate_matrix_per_s,ref.populations,.002,64,9823+i)
        sem=np.std(estimates,axis=0,ddof=1)/8
        diff=np.abs(np.mean(estimates,axis=0)-ref.populations)
        population_pass=bool(np.all(diff<=6*sem+1e-6))
        # Conditional net momentum rate of each state, time-averaged over jumps.
        reward=np.zeros((24,3))
        matrices=ref.beam_transition_quantities.stimulated_coefficients_per_s
        for b,beam in enumerate(beams):
            momentum=HBAR_J_S*2*np.pi/beam.wavelength_m*np.array(beam.direction)
            reward[:8]+=np.sum(matrices[b],axis=0)[:,None]*momentum
            reward[8:]-=np.sum(matrices[b],axis=1)[:,None]*momentum
        force_samples=estimates@reward
        fsem=np.std(force_samples,axis=0,ddof=1)/8
        fdiff=np.abs(np.mean(force_samples,axis=0)-ref.force_n)
        force_pass=bool(np.all(fdiff<=6*fsem+1e-29))
        cases.append({'position_m':r,'velocity_m_per_s':v,
            'population_pass':population_pass,'force_pass':force_pass,
            'reference_populations':ref.populations.tolist(),
            'event_populations':np.mean(estimates,axis=0).tolist(),
            'population_standard_errors':sem.tolist(),
            'reference_force_n':list(ref.force_n),
            'event_force_n':np.mean(force_samples,axis=0).tolist(),
            'force_standard_errors_n':fsem.tolist()})
    trajectory_cases=[]
    for position,speed in (([.015,0,0],5.),([.015,0,0],10.),
                            ([0,.015,0],10.),([0,0,.015],10.),
                            ([.015,.004,0],5.)):
        r=np.array(position);v=-r/np.linalg.norm(r)*speed
        coarse=classify(r,v,5e-6,.05,data)
        fine=classify(r,v,2.5e-6,.05,data)
        long_coarse=classify(r,v,5e-6,.1,data,False)
        long_fine=classify(r,v,2.5e-6,.1,data,False)
        agreeing=bool(coarse[0]==fine[0] and coarse[0]>=0)
        retained=bool(coarse[0]!=1 or (
            long_coarse[0]!=0 and long_fine[0]!=0 and
            np.linalg.norm(long_coarse[2])<=.002 and np.linalg.norm(long_fine[2])<=.002))
        endpoint_error=float(np.linalg.norm(long_coarse[2]-long_fine[2]))
        trajectory_cases.append({'position_m':position,'speed_m_per_s':speed,
            'coarse_code':int(coarse[0]),'fine_code':int(fine[0]),
            'classification_agrees':agreeing,'retained_at_100ms':retained,
            'long_final_positions_m':[long_coarse[2].tolist(),long_fine[2].tolist()],
            'long_endpoint_difference_m':endpoint_error,
            'passed':agreeing and retained and endpoint_error<1e-5})
    passed=all(c['population_pass'] and c['force_pass'] for c in cases) and all(c['passed'] for c in trajectory_cases)
    report={'passed':passed,'event_cases':cases,'trajectory_cases':trajectory_cases,
        'event_method':'64 independent 2 ms frozen-phase-space Gillespie trajectories; 6-SE regression gate',
        'limits':'frozen-point internal-state comparison; no recoil diffusion, temperature, or coherent optical dynamics validation'}
    write_json(paths()['statistics']/'validation'/'event_and_trajectory_checks.json',report)
    if not passed:
        raise RuntimeError('event/trajectory campaign gate failed; inspect validation report')
    return report
