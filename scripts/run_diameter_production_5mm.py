"""User-authorized 5--25 mm uniform-area production continuation.

Keeps the validated diameter adapter and physics sources unchanged, freezes
independent sample allocations, and records the revised scope separately.
"""
import run_population_diameter_campaign as runner
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from pmot.configuration import RB87_MASS_KG

INDICES=tuple(range(2,10))
WORKERS=16


def initialize():
    runner.rc.CAMPAIGN='cooling_diameter_20260930_fixed_intensity_baseline'
    root=runner.rc.paths()['statistics']
    assert runner.rc.source_digest()==runner.CORE_SHA
    manifest=json.loads((root/'campaign_manifest.json').read_text())
    assert manifest['runner_sha256']==hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    for name in ('geometry_and_force','event_and_trajectory_checks','fresh_worker_checks','diameter_fixed_comparison'):
        certificate=json.loads((root/'validation'/f'{name}.json').read_text())
        assert certificate['passed'] and certificate['source_sha256']==runner.CORE_SHA
    for row in certificate['rows']:
        file=root/'pilot'/row['point']/'rays'/row['ray']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['input_sha256']
    assert json.loads((root/'validation/test_status.json').read_text())==dict(exit_code=0,source_sha256=runner.CORE_SHA)
    pilot=json.loads((root/'pilot_collection_complete.json').read_text())
    allocations=[]
    for index in INDICES:
        selected=next(s for s in pilot['settings'] if s['diameter_mm']==runner.DIAMETERS[index])
        assert selected['retention_pass'] and selected['velocity_quadrature_pass']
        allocation=selected['provisional_allocation']
        assert allocation['key']==f'01_cooling_diameter/{index:03d}'
        allocations.append(allocation)
    plan={'user_decision':'Start at 5 mm; keep uniform-area sampling and fixed peak cooling intensity; collect sufficient statistics unattended.',
        'source_sha256':runner.CORE_SHA,'adapter_sha256':manifest['runner_sha256'],
        'production_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selected_indices':list(INDICES),'deferred_diameters_mm':[2.,3.],
        'allocations':allocations,'independent_fixed_sample':True,'pilot_seed':2026093001,
        'production_seed':2026093002,'workers':WORKERS,'relative_95_half_width_target':.05,
        'variance_safety_factor':1.5,'estimated_total_hours':[sum(a['estimated_hours'][i] for a in allocations) for i in (0,1)]}
    target=root/'production_5mm_plan.json'
    if target.exists():
        assert json.loads(target.read_text())==plan,'Production scope/source changed; record a revision first'
    else:
        runner.rc.write_json(target,plan)
    intensity=2*.027/(np.pi*.00635**2)
    parameters={'model':'corrected Section-12 24-state Rb87 D2 steady-state population-rate MOT',
        'source_sha256':runner.CORE_SHA,'mass_kg':RB87_MASS_KG,
        'cooling_peak_intensity_w_per_m2':intensity,'cooling_peak_intensity_mw_per_cm2':intensity/10,
        'reference_I_sat_w_per_m2':16.69,'reference_gamma_rad_per_s':2*np.pi*6.07e6,
        'reference_single_beam_s0':intensity/16.69,
        'reference_single_beam_s_eff':intensity/16.69/(1+(2*15e6/6.07e6)**2),
        'saturation_note':'s0/s_eff are reporting conventions; rates use ARC state-dependent Gamma and dipoles, with saturation emerging from coupled populations',
        'coils':asdict(runner.default_anti_helmholtz_config()),'axial_gradient_g_per_cm':10.,
        'gravity_m_per_s2':[0.,0.,-9.80665],'recoil_diffusion':False,
        'sampling':{'radius_m':.015,'launch_plane_distance_m':.015,'directions':'independent uniform full sphere',
            'points':'independent uniform area per disc','velocities':'parallel to disc normal toward MOT',
            'pilot_seed':2026093001,'production_seed':2026093002,'cluster_unit':'direction disc'},
        'capture':{'core_radius_m':.002,'residence_s':.005,'alternative':'two core entries with intervening exit',
            'escape_radius_m':.03,'escape_condition':'outward velocity','timeout':'indeterminate',
            'speed_nodes_m_per_s':np.r_[runner.ce.VELOCITIES,runner.ce.TAIL_VELOCITIES].tolist(),
            'zero_speed':'only exact weighted-integrand anchor g(0)=0; no cross-section imputation'},
        'numerics':{'adaptive':'Dormand-Prince 5(4), two independent tolerance settings',
            'initial_dt_s':5e-6,'base_max_dt_s':.0005,'base_max_dt_inside_3mm_s':.0001,
            'base_position_atol_m':1e-8,'base_velocity_atol_m_per_s':1e-5,'base_rtol':1e-5,
            'refinement':'halve step caps, divide all error scales by four; core event step cap also applies',
            'fixed_rk4_ladder_coarse_fine_duration_s':[list(x) for x in runner.ce.LADDER],
            'retention_final_window_s':.025,'retention_endpoint_agreement_m':1e-5,
            'retention_durations_s':[.1,.2,.4,1.,2.],'loading_quadrature_relative_gate':.02,
            'statistical_relative_95_half_width_gate':.05},
        'loading':{'historical_prefactor':runner.ce.LOADING_RATE_PREFACTOR,
            'thermal_scale_m2_per_s2':runner.ce.THERMAL_SCALE_M2_PER_S2,
            'vapor_normalization':'historical coefficient retained; unresolved absolute density normalization; report R/n87 as well',
            'normalized_Maxwell_equivalent_density_m_minus3':runner.ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*runner.ce.THERMAL_SCALE_M2_PER_S2**1.5))},
        'settings':[]}
    for index,a in zip(INDICES,allocations):
        diameter=runner.DIAMETERS[index]
        point=runner.DiameterPoint(index,diameter,.027*(diameter/12.7)**2,-2*np.pi*15e6)
        parameters['settings'].append({'diameter_mm':diameter,'configuration':asdict(point.config()),
            'beams':[asdict(b) for b in runner.beams_for(point.config())],
            'direction_discs':a['disc_count'],'points_per_disc':a['points_per_disc'],'total_rays':a['rays']})
        data=runner.prepare_diameter(point.config())
        file=root/'production_parameters'/f'{index:03d}_compiled_inputs.npz'
        if not file.exists():
            file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,**dict(zip(('directions','optical','dipole_strengths','frequency_offsets','ground_indices',
                'excited_indices','polarization_q','zeeman_coefficients','excited_decay_rates','spontaneous_matrix','controls'),data)))
    runner.rc.write_json(root/'production_parameters.json',parameters)
    return root,plan


def main():
    root,plan=initialize()
    runner.rc.write_json(root/'production_5mm_process.json',dict(pid=os.getpid(),command=sys.argv,workers=WORKERS,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    runner.ce.PRODUCTION_SEED=2026093002
    runner.ce.prepare=runner.prepare_diameter
    runner.ce._ray_worker=runner.diameter_ray_worker
    rows=[];issues=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index,a in zip(INDICES,plan['allocations']):
            diameter=runner.DIAMETERS[index]
            point=runner.DiameterPoint(index,diameter,.027*(diameter/12.7)**2,-2*np.pi*15e6)
            start=time.perf_counter()
            try:
                row=runner.ce.run_loading_point(point,'production',a['disc_count'],a['points_per_disc'],pool)
            except RuntimeError as error:
                issues.append({'diameter_mm':diameter,'stage':'capture','error':str(error)})
                runner.rc.write_json(root/'production_5mm_issues.json',issues)
                continue
            capture=time.perf_counter()-start
            start=time.perf_counter();retention=True
            try:
                runner.ce.duration_audits(point,'production',a['disc_count'],a['points_per_disc'],pool)
            except RuntimeError as error:
                retention=False
                issues.append({'diameter_mm':diameter,'stage':'retention','error':str(error)})
                runner.rc.write_json(root/'production_5mm_issues.json',issues)
            row['loading_precision_pass']=row['loading_relative_half_width'] is not None and row['loading_relative_half_width']<=.05
            row['retention_pass']=retention
            row['publication_status']='validated' if retention and row['loading_precision_pass'] and row['velocity_quadrature_pass'] else 'qualification pending'
            row['cooling_diameter_mm']=diameter;row['cooling_power_w_per_beam']=point.cooling_power_w
            density=runner.ce.LOADING_RATE_PREFACTOR/(4/(np.sqrt(np.pi)*runner.ce.THERMAL_SCALE_M2_PER_S2**1.5))
            row['loading_coefficient_m3_per_s']=row['loading_rate_atoms_per_s']/density
            row['loading_coefficient_95_half_width_m3_per_s']=row['loading_95_half_width_atoms_per_s']/density
            folder=root/'production'/point.key
            runner.rc.write_json(folder/'summary.json',row)
            if not (folder/'timing.json').exists():
                runner.rc.write_json(folder/'timing.json',dict(capture_wall_s=capture,retention_wall_s=time.perf_counter()-start))
            rows.append(row)
            runner.rc.write_csv(root/'production_5mm_summary.csv',rows)
            print(json.dumps(row),flush=True)
    ready=len(rows)==len(INDICES) and all(r['publication_status']=='validated' for r in rows)
    if len(rows)==len(INDICES):
        figure=runner.rc.paths()['figures']/'from_5mm';figure.mkdir(parents=True,exist_ok=True)
        for coefficient in (False,True):
            y='loading_coefficient_m3_per_s' if coefficient else 'loading_rate_atoms_per_s'
            err='loading_coefficient_95_half_width_m3_per_s' if coefficient else 'loading_95_half_width_atoms_per_s'
            fig,ax=runner.plt.subplots(figsize=(9,6),layout='constrained')
            ax.errorbar([r['cooling_diameter_mm'] for r in rows],[r[y] for r in rows],yerr=[r[err] for r in rows],fmt='o-',capsize=3)
            ax.set(xlabel='Cooling-beam Gaussian 1/e² diameter [mm]',ylabel='Loading coefficient R/n87 [m³/s]' if coefficient else 'Loading rate [atoms/s; historical vapor normalization]',
                title='Population-rate MOT · fixed peak cooling intensity · −15 MHz\n15 mm sampling radius · 95% disc-cluster intervals · 12.7 mm repump')
            ax.grid(alpha=.25)
            if not ready:ax.text(.02,.98,'Qualification pending',transform=ax.transAxes,va='top',color='darkred')
            for ext in ('png','pdf'):fig.savefig(figure/f'loading{"_coefficient" if coefficient else ""}_vs_cooling_diameter.{ext}',dpi=180)
            runner.plt.close(fig)
    runner.rc.write_json(root/'production_5mm_completion.json',dict(computed=True,publication_ready=ready,
        visual_review_pending=True,issues=issues,remaining_points=[r['cooling_diameter_mm'] for r in rows if r['publication_status']!='validated']))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        runner.rc.write_json(runner.rc.paths()['statistics']/'production_5mm_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
