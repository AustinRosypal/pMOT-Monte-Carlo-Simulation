"""Prospective paired geometry statistics; no trajectories or full certificates.

Numerical classification, retention and omitted-phase-space bounds remain
separate gates. Allocation inputs must come from an independent paired pilot.
"""
import numpy as np
from scipy.stats import t


def paired_inputs(reference, comparison):
    a, b = np.asarray(reference, dtype=float), np.asarray(comparison, dtype=float)
    if (a.ndim != 1 or b.shape != a.shape or len(a) < 2
        or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
        or np.any(a < 0) or np.any(b < 0) or a.mean() <= 0):
        raise ValueError('require matched finite nonnegative per-disc rates and positive reference mean')
    return a, b


def paired_equivalence(reference, comparison, *, tolerance=.02, family_size=2, alpha=.05):
    """Test relative mean equivalence through two linear paired contrasts.

    Each geometry comparison gets alpha/family_size for its intersection-union
    equivalence test. Its two one-sided contrasts share that level; they are
    not two independent tests to be combined by a union. Bonferroni across the
    geometry comparisons gives an approximate family false-acceptance rate
    <=alpha under the Student-t assumptions. This accounts for uncertainty in
    the reference mean rather than dividing a difference CI by a fixed mean.
    """
    a, b = paired_inputs(reference, comparison)
    if not 0 < tolerance < 1 or family_size < 1 or not 0 < alpha < .5:
        raise ValueError('invalid equivalence tolerance or family significance')
    n = len(a)
    critical = t.ppf(1-alpha/family_size, n-1)
    plus, minus = b-(1+tolerance)*a, b-(1-tolerance)*a
    upper = float(plus.mean()+critical*plus.std(ddof=1)/np.sqrt(n))
    lower = float(minus.mean()-critical*minus.std(ddof=1)/np.sqrt(n))
    delta = b-a
    half = float(t.ppf(.975, n-1)*delta.std(ddof=1)/np.sqrt(n))
    return dict(disc_count=n, tolerance=tolerance, family_size=family_size,
        family_alpha=alpha, mean_reference_atoms_per_s=float(a.mean()),
        mean_comparison_atoms_per_s=float(b.mean()), paired_difference_atoms_per_s=float(delta.mean()),
        paired_difference_ci95_atoms_per_s=[float(delta.mean()-half), float(delta.mean()+half)],
        upper_comparison_minus_one_plus_tolerance_reference=upper,
        lower_comparison_minus_one_minus_tolerance_reference=lower,
        paired_equivalence_pass=bool(upper < 0 and lower > 0),
        full_geometry_acceptance_certified=False,
        separate_gates=['numerical classifications', 'retention', 'omitted-phase-space bounds', 'visual review'])


def prospective_allocation(pilot_reference, pilot_comparison, *, tolerance=.02,
                           family_size=2, alpha=.05, variance_safety=1.5,
                           minimum_discs=200, maximum_discs=3975):
    """Plan a fixed new sample; never use validation data to extend its count.

    Use half of the pilot's distance to the equivalence boundary as the target
    one-sided uncertainty, with a variance safety factor. This is planning,
    not a guarantee of power, coverage or equivalence. A pilot mean outside
    the tolerance or an excessive required count must remain an explicit gate.
    """
    a, b = paired_inputs(pilot_reference, pilot_comparison)
    paired_equivalence(a, b, tolerance=tolerance, family_size=family_size, alpha=alpha)
    if variance_safety < 1 or minimum_discs < 2 or maximum_discs < minimum_discs:
        raise ValueError('invalid allocation limits or variance safety factor')
    margin = .5*(tolerance*a.mean()-abs((b-a).mean()))
    if margin <= 0:
        raise ValueError('pilot mean lies outside proposed equivalence tolerance')
    variance = max(np.var(b-(1+tolerance)*a, ddof=1), np.var(b-(1-tolerance)*a, ddof=1))
    count = int(minimum_discs)
    while t.ppf(1-alpha/family_size, count-1)**2*variance_safety*variance > count*margin**2:
        count += 25
        if count > maximum_discs:
            raise ValueError('required paired allocation exceeds available reference discs')
    return dict(prospective_disc_count=count, pilot_disc_count=len(a),
        pilot_independent_of_validation_required=True, optional_stopping=False,
        tolerance=tolerance, family_size=family_size, family_alpha=alpha,
        variance_safety_factor=variance_safety, pilot_planning_margin_atoms_per_s=float(margin),
        maximum_paired_contrast_variance=float(variance), allocation_frozen=False,
        instruction='freeze count, seeds, thresholds and source manifest before validation trajectories')
