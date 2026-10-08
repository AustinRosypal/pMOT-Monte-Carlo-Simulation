"""Post-run population/force snapshots and paired capture-mask explanation.

Does not change any campaign data, source physics, or trajectory settings.
Snapshots are illustrative phase-space points, not trajectory histories.
"""
import run_matched_repump_diameter_campaign as campaign
import hashlib
import json
from pathlib import Path
import numpy as np
from pmot.fields import beam_intensity_w_per_m2
from pmot.mot_multilevel.coupling import wavevector_rad_per_m
from pmot.configuration import HBAR_J_S, RB87_MASS_KG

base = campaign.base


def main():
    root = base.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1'/campaign.CAMPAIGN
    output = root/'analysis_repump_mechanism'
    output.mkdir(exist_ok=True)
    assert base.rc.source_digest() == base.CORE_SHA
    model, coil = base.build_rate_equation_model(), base.default_anti_helmholtz_config()
    ground_f = np.array([model.structure.states[i].f for i in model.ground_indices])
    rows = []
    for geometry in ('diagonal', 'x_axis'):
        unit = np.ones(3)/np.sqrt(3) if geometry == 'diagonal' else np.array([1.,0.,0.])
        for radius_mm in (0.,3.,6.,9.,12.,15.):
            position = radius_mm*.001*unit
            velocity = -20.*unit
            for repump_mm in (12.7,25.):
                cfg = base.DiameterConfig(cooling_power_w_per_beam=.027*(25/12.7)**2,
                    repump_power_w_per_beam=.0001*(repump_mm/12.7)**2,
                    cooling_diameter_mm=25.)
                beams = base.beams_for(cfg) if repump_mm == 12.7 else campaign.beams_for(cfg)
                obs = base.rate_equation_observable(model,beams,tuple(position),tuple(velocity),coil,cfg,
                    store_rate_matrix=True, store_beam_transition_quantities=True)
                ground = obs.populations[:8]
                cooling_force = np.zeros(3)
                repump_force = np.zeros(3)
                intensities = []
                for beam, rate in zip(beams,obs.beam_effective_scattering_rates_per_s):
                    force = HBAR_J_S*np.asarray(wavevector_rad_per_m(beam))*rate
                    if beam.family == 'cooling': cooling_force += force
                    else: repump_force += force
                    intensities.append(beam_intensity_w_per_m2(beam,tuple(position)))
                eigen = np.sort(np.real(np.linalg.eigvals(obs.rate_matrix_per_s)))
                assert eigen[-2] < 0 and abs(eigen[-1]) < 1e-5
                rows.append(dict(geometry=geometry,radius_mm=radius_mm,speed_m_per_s=20.,
                    repump_diameter_mm=repump_mm,F1_population=float(ground[ground_f==1].sum()),
                    F2_population=float(ground[ground_f==2].sum()),excited_population=float(obs.populations[8:].sum()),
                    scattering_per_s=obs.total_spontaneous_scattering_rate_per_s,
                    cooling_braking_acceleration_m_per_s2=float(cooling_force@unit/RB87_MASS_KG),
                    repump_braking_acceleration_m_per_s2=float(repump_force@unit/RB87_MASS_KG),
                    slowest_local_population_relaxation_s=float(-1/eigen[-2]),
                    beam_intensities_w_per_m2=intensities))
    base.rc.write_csv(output/'local_snapshots.csv',rows)
    base.rc.write_json(output/'local_snapshots.json',rows)
    point = campaign.points()[-1]
    folder = root/'production'/point.key
    old_folder = campaign.OLD/'production'/point.key
    with np.load(folder/'disc_statistics.npz') as new,np.load(old_folder/'disc_statistics.npz') as old:
        speeds = new['velocity_m_per_s']
        sigma_new = new['disc_cross_section_m2']
        sigma_old = old['disc_cross_section_m2']
        spectra=[]
        for v in (5.,10.,15.,20.,25.,30.,35.,40.):
            i = np.where(speeds==v)[0][0]
            difference,half = base.rc.cluster_interval(sigma_new[:,i]-sigma_old[:,i])
            spectra.append(dict(speed_m_per_s=v,fixed_repump_cross_section_mm2=float(sigma_old[:,i].mean()*1e6),
                matched_repump_cross_section_mm2=float(sigma_new[:,i].mean()*1e6),
                paired_difference_mm2=float(difference*1e6),paired_95_half_width_mm2=float(half*1e6)))
    base.rc.write_csv(output/'cross_sections_25mm.csv',spectra)
    bins = np.array([0.,3.,6.35,9.,12.,15.00001])
    gains = np.zeros(5)
    gains_by_disc = np.zeros((192,5))
    counts = np.zeros(5,dtype=int)
    weighting = speeds**3*np.exp(-speeds**2/base.ce.THERMAL_SCALE_M2_PER_S2)
    for file in (folder/'rays').glob('*.npz'):
        d = int(file.stem.split('_')[0][1:])
        with np.load(file) as new,np.load(old_folder/'rays'/file.name) as old:
            np.testing.assert_array_equal(new['position_m'],old['position_m'])
            assert str(new['status']) == str(old['status']) == 'resolved'
            r = new['position_m'];direction = new['direction']
            impact_mm = np.linalg.norm(r-np.dot(r,direction)*direction)*1000
            bin_id = np.searchsorted(bins,impact_mm,side='right')-1
            delta = new['codes'][:320]-old['codes'][:320]
            weighted = delta*weighting
            contribution = np.pi*.015**2*base.ce.LOADING_RATE_PREFACTOR*np.trapezoid(np.r_[0.,weighted],np.r_[0.,speeds])
            gains[bin_id] += contribution/3072
            gains_by_disc[d,bin_id] += contribution/16
            counts[bin_id] += 1
    annuli=[]
    for i in range(5):
        mean,half = base.rc.cluster_interval(gains_by_disc[:,i])
        annuli.append(dict(impact_min_mm=float(bins[i]),impact_max_mm=float(min(bins[i+1],15.)),rays=int(counts[i]),
            loading_gain_atoms_per_s=float(mean),gain_95_half_width_atoms_per_s=float(half),
            fraction_of_total_gain=float(mean/gains.sum())))
    base.rc.write_csv(output/'impact_parameter_gain_25mm.csv',annuli)
    base.rc.write_json(output/'analysis_manifest.json',dict(source_sha256=base.CORE_SHA,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Local steady-state snapshots at prescribed points, plus paired saved production masks at 25 mm; not time-dependent internal-state trajectories',
        total_gain_atoms_per_s=float(gains.sum()),fraction_gain_impact_above_6p35mm=float(gains[2:].sum()/gains.sum()),
        baseline=str(campaign.OLD),matched=str(root)))
    print(json.dumps(dict(snapshots=[r for r in rows if r['geometry']=='diagonal'],spectra=spectra,annuli=annuli),indent=2))


if __name__ == '__main__':
    main()
