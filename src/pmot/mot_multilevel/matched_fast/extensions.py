"""Future independent-disc extensions with explicit sequential coverage policy."""
import numpy as np
from scipy.stats import t
from ..matched_campaign import reweight_counts


def combine_independent_batches(payloads,temperature_k=296.483,pressure_torr=1e-10):
    """Combine retained production batches without naive optional-stop pooling.

    Batch k spends alpha_k=.05/[k(k+1)]. Their Student-t intervals have total
    alpha at most .05 over arbitrarily many independent-disc extensions.
    A direction-count-weighted sum of those simultaneous batch intervals is
    a conservative interval for the combined mean, under the same usual
    cluster Student-t assumptions. New batch sizes may depend on earlier data,
    but each size must be fixed before seeing that batch. This is intentionally
    more conservative than a pooled, fixed-N nominal 95% t interval.
    """
    if not payloads:raise ValueError('no production batches')
    seen=set();ids=set();results=[];configuration=None
    for k,payload in enumerate(payloads,1):
        if payload.get('stage')!='production':raise ValueError('pilot/validation data cannot enter production intervals')
        batch=payload['batch_id']
        if batch in ids:raise ValueError('duplicate batch id')
        ids.add(batch)
        if not payload.get('sample_count_fixed_before_batch',False):raise ValueError('batch count was not predeclared')
        keys={(payload['master_geometry_sha256'],d) for d in payload['disc_indices']}
        if seen&keys:raise ValueError('extension must contain new independent direction discs')
        seen|=keys
        controls={key:payload['configuration'][key] for key in ('cooling_power_mw','cooling_detuning_mhz','cooling_diameter_mm',
                    'gradient_g_per_cm','repump_power_mw','repump_diameter_mm','repump_detuning_mhz')}
        if configuration is None:configuration=controls
        if controls!=configuration:raise ValueError('physical controls differ between batches')
        result=reweight_counts(payload,temperature_k,pressure_torr)
        values=np.asarray(result['disc_loading_rates_atoms_per_s'])
        alpha=.05/(k*(k+1));half=float(t.ppf(1-alpha/2,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
        results.append(dict(batch_id=batch,discs=len(values),mean=result['mean'],lower=result['mean']-half,
                            upper=result['mean']+half,alpha_spent=alpha))
    total=sum(r['discs'] for r in results)
    combined={field:sum(r['discs']*r[field] for r in results)/total for field in ('mean','lower','upper')}
    return dict(**combined,disc_count=total,batches=results,confidence=.95,
                interval_method='alpha-spending simultaneous batch Student-t bounds, weighted across independent direction discs',
                temperature_k=temperature_k,pressure_torr=pressure_torr)
