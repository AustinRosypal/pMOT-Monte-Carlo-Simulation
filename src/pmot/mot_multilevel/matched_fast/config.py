"""Campaign inputs and independent sampling stages; the physical kernel is shared."""
from pathlib import Path
from dataclasses import asdict
import os
import tempfile
import numpy as np
from ...comparison_paths import COMPARISON_FIGURES as COMPARISON_FIGURE_ROOT
from ...comparison_paths import COMPARISON_NOTEBOOKS, COMPARISON_NOTEBOOK_NAMES
from scipy.stats import t
from .. import matched_benchmark as original

ROOT=original.ROOT
NAME='matched_two_level_exact_D2_fast_20261104'
BENCHMARK=original.BENCHMARK
PREVIOUS=original.STATISTICS
PREVIOUS_CACHE=original.CACHE
STATISTICS=ROOT/'outputs/statistics/mot_multilevel_population_rate_v1'/NAME
FIGURES=ROOT/'outputs/figures/mot_multilevel_population_rate_v1'/NAME
COMPARISON_FIGURES=COMPARISON_FIGURE_ROOT/NAME
CACHE=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'pmot-campaigns'/NAME
NOTEBOOK=ROOT/'notebooks/mot_multilevel'/f'{NAME}.ipynb'
COMPARISON_NOTEBOOK=COMPARISON_NOTEBOOKS/COMPARISON_NOTEBOOK_NAMES['loading_two_refined']
PILOT_DISCS=tuple(range(837,869))
VALIDATION_DISCS=tuple(range(869,901))
PILOT_POINTS=tuple(range(16))
VELOCITIES=original.VELOCITIES
WAVELENGTH_M=original.WAVELENGTH_M
REPUMP_WAVELENGTH_M=original.REPUMP_WAVELENGTH_M
REPUMP_POWERS_MW=original.REPUMP_POWERS_MW
MatchedPoint=original.MatchedPoint
now=original.now
read_json=original.read_json
write_json=original.write_json
replace_file=original.replace_file
sha256=original.sha256
configuration=original.configuration
prepared=original.prepared
definitions=original.definitions
load_master_geometry=original.load_master_geometry
model_notes=original.model_notes


def source_hashes():
    result=original.source_hashes()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        result[str(path.relative_to(ROOT)).replace('\\','/')]=sha256(path)
    return result


def available_discs(point):
    """Original coordinates only; all pilot/validation discs stay excluded."""
    maximum=original.sample_count(point)[0]
    if point.index==0:
        record=read_json(STATISTICS/'migration.json')
        excluded_through=record['original_baseline_max_checkpoint_disc']
    else:excluded_through=-1
    return [i for i in range(maximum) if i>excluded_through and i not in PILOT_DISCS+VALIDATION_DISCS]


def allocation(point,ray_rates,target=.08,variance_inflation=1.25,max_rays=12800):
    """Choose a fixed production design from excluded independent pilot discs."""
    a=np.asarray(ray_rates,float);mean=float(a.mean())
    within=float(np.var(a,axis=1,ddof=1).mean())
    between=max(0.,float(np.var(a.mean(axis=1),ddof=1))-within/a.shape[1])
    candidates=[];available=available_discs(point)
    for npoints in (16,25,32,50):
        max_discs=min(len(available),max_rays//npoints)
        choices=list(range(128,max_discs+1,16))
        if max_discs>=128 and max_discs not in choices:choices.append(max_discs)
        for discs in choices:
            half=float(t.ppf(.975,discs-1)*np.sqrt(variance_inflation*(between+within/npoints)/discs))
            relative=half/mean if mean>0 else None
            candidates.append(dict(discs=discs,points_per_disc=npoints,rays=discs*npoints,
                                   predicted_relative_half_width=relative))
    if not candidates:raise ValueError('insufficient reserved production directions')
    acceptable=[x for x in candidates if x['predicted_relative_half_width'] is not None and x['predicted_relative_half_width']<=target]
    if acceptable:chosen=min(acceptable,key=lambda x:(x['rays'],-x['discs']))
    elif mean>0:chosen=min(candidates,key=lambda x:(x['predicted_relative_half_width'],-x['discs']))
    else:chosen=max(candidates,key=lambda x:(x['rays'],x['discs']))
    return dict(**chosen,disc_indices=available[:chosen['discs']],point_indices=list(range(chosen['points_per_disc'])),
                target_relative_half_width=target,estimated_from_independent_pilot=True,
                predicted_target_met=bool(acceptable),pilot_mean_atoms_per_s=mean,
                pilot_between_disc_variance=between,pilot_within_disc_variance=within,
                variance_inflation=variance_inflation,max_rays_predeclared=max_rays,
                limitation='Planning target, not guaranteed achieved precision; production is never stopped or extended based on its observed interval.')


def sampling(point,stage='production'):
    if stage=='pilot':return list(PILOT_DISCS),list(PILOT_POINTS)
    plan=read_json(STATISTICS/'production_plan.json')['allocations'][str(point.index)]
    return plan['disc_indices'],plan['point_indices']


def sample_count(point):
    discs,points=sampling(point)
    return len(discs),len(points)
