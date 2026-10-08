"""Plot saved explanation diagnostics and check frozen-field pumping transients."""
import run_matched_repump_diameter_campaign as campaign
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import expm

base=campaign.base
root=base.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1'/campaign.CAMPAIGN/'analysis_repump_mechanism'
rows=json.loads((root/'local_snapshots.json').read_text())
spectra=np.genfromtxt(root/'cross_sections_25mm.csv',delimiter=',',names=True)
fig,axes=base.plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for diameter,label in ((12.7,'Repump 12.7 mm'),(25.,'Repump 25 mm')):
    selected=[r for r in rows if r['geometry']=='diagonal' and r['repump_diameter_mm']==diameter]
    axes[0].plot([r['radius_mm'] for r in selected],[100*r['F1_population'] for r in selected],'o-',label=label)
axes[0].set(xlabel='Distance from center along (1,1,1) [mm]',ylabel='Steady-state population in F=1 [%]',
    title='Local snapshots: 20 m/s toward center')
for stats_root,label,fmt in ((campaign.OLD,'Repump 12.7 mm','s--'),(root.parent,'Repump 25 mm','o-')):
    with np.load(stats_root/'production/01_cooling_diameter/009/disc_statistics.npz') as saved:
        estimates=[base.rc.cluster_interval(saved['disc_cross_section_m2'][:,np.where(saved['velocity_m_per_s']==v)[0][0]])
            for v in spectra['speed_m_per_s']]
    axes[1].errorbar(spectra['speed_m_per_s'],[1e6*x[0] for x in estimates],
        yerr=[1e6*x[1] for x in estimates],fmt=fmt,capsize=3,label=label)
axes[1].set(xlabel='Launch speed [m/s]',ylabel='Mean capture cross section [mm²]',title='Production masks; 95% disc-cluster intervals')
for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=10)
fig.suptitle('25 mm cooling beams; both peak intensities fixed',fontsize=15)
fig.savefig(root/'mechanism.png',dpi=170)
fig.savefig(root/'mechanism.pdf')
base.plt.close(fig)
model=base.build_rate_equation_model();coil=base.default_anti_helmholtz_config()
gf=np.array([model.structure.states[i].f for i in model.ground_indices])
initial=np.zeros(24);initial[:8][gf==2]=1/5
transients=[];unit=np.ones(3)/np.sqrt(3)
for radius in (12.,15.):
    for diameter in (12.7,25.):
        cfg=base.DiameterConfig(cooling_power_w_per_beam=.027*(25/12.7)**2,
            repump_power_w_per_beam=.0001*(diameter/12.7)**2,cooling_diameter_mm=25.)
        beams=base.beams_for(cfg) if diameter==12.7 else campaign.beams_for(cfg)
        obs=base.rate_equation_observable(model,beams,tuple(radius*.001*unit),tuple(-20*unit),coil,cfg,store_rate_matrix=True)
        for duration in (0.,50e-6,200e-6,1e-3):
            p=expm(obs.rate_matrix_per_s*duration)@initial
            assert abs(p.sum()-1)<1e-8 and p.min()>-1e-12
            transients.append(dict(radius_mm=radius,repump_mm=diameter,time_s=duration,
                initial='unpolarized F=2',environment='held fixed; not a moving-atom trajectory',
                F1_population=float(p[:8][gf==1].sum()),F1_steady=float(obs.populations[:8][gf==1].sum())))
base.rc.write_json(root/'frozen_environment_transients.json',transients)
base.rc.write_json(root/'plot_and_transient_provenance.json',dict(source_sha256=base.rc.source_digest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    purpose='Illustrative mechanism and steady-state applicability check; no campaign inputs or results modified'))
print(json.dumps(transients,indent=2))
