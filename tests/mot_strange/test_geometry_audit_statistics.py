import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def statistics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_geometry_audit_statistics')


def test_paired_covariance_and_relative_reference_uncertainty(statistics):
    reference = np.linspace(5e5, 1.5e6, 400)
    comparison = reference+1000.
    row = statistics.paired_equivalence(reference, comparison)
    assert row['paired_difference_ci95_atoms_per_s'] == [1000., 1000.]
    assert row['paired_equivalence_pass']
    assert not row['full_geometry_acceptance_certified']


def test_effect_outside_tolerance_cannot_pass_with_large_sample(statistics):
    reference = np.linspace(5e5, 1.5e6, 4000)
    row = statistics.paired_equivalence(reference, 1.03*reference)
    assert not row['paired_equivalence_pass']
    with pytest.raises(ValueError, match='outside'):
        statistics.prospective_allocation(reference[:32], 1.03*reference[:32])


def test_allocation_is_prospective_and_preserves_count_limits(statistics):
    reference = np.linspace(5e5, 1.5e6, 32)
    plan = statistics.prospective_allocation(reference, reference+1000.)
    assert 200 <= plan['prospective_disc_count'] <= 3975
    assert plan['pilot_independent_of_validation_required']
    assert not plan['optional_stopping'] and not plan['allocation_frozen']
    noisy = reference+np.tile([0., 2e5], 16)-1e5
    with pytest.raises(ValueError, match='exceeds'):
        statistics.prospective_allocation(reference, noisy, maximum_discs=200)


def test_invalid_or_empty_reference_is_not_certified_as_equivalent(statistics):
    for a, b in (([0., 0.], [0., 0.]), ([1.], [1.]),
                 ([1., 2.], [1.]), ([1., 2.], [1., float('nan')]),
                 ([-1., 2.], [1., 2.])):
        with pytest.raises(ValueError, match='matched'):
            statistics.paired_equivalence(a, b)
