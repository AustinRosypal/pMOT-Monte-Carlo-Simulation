"""Read-only diagnostic of retained narrow-beam launch failures."""
import run_population_diameter_campaign as runner
import numpy as np
from pmot.mot_multilevel.accelerated import observable, classify
from pmot.mot_multilevel.adaptive import integrate
from pmot.mot_multilevel.campaign_execution import _resolve_node

point = runner.DiameterPoint(0, 2., .027*(2/12.7)**2, -2*np.pi*15e6)
data = runner.prepare_diameter(point.config())
root = runner.ROOT/'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
with np.load(root/'revision_history/02_compiled_cache_isolation/pilot/01_cooling_diameter/000/rays/d000_p000.npz') as saved:
    r, v = saved['position_m'], .25*saved['direction']
    next_speed=runner.ce.VELOCITIES[len(saved['codes'])]
    print('completed nodes',len(saved['codes']),'next speed',next_speed,flush=True)
print('r,v', r, v, flush=True)
for name, call in [('observable', lambda: observable(r,v,np.array([0.,0.,1.]),data)),
                   ('adaptive', lambda: integrate(r,v,.05,data,1.)),
                   ('fixed', lambda: classify(r,v,5e-6,.05,data)),
                   ('resolved', lambda: (_resolve_node(r,v/.25,next_speed,data,[],[]),))]:
    try:
        result=call()
        print(name, result[:2], flush=True)
    except Exception as error:
        print(name, repr(error), flush=True)
