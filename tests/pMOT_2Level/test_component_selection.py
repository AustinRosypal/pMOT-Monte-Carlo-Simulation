"""Independent scalar/vector/tensor selection for effective two-level pMOTs."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.pMOT_2Level import load_pmot_2level_component_polarizability_table
from pmot.pMOT_2Level import normalize_pmot_stark_component_code
from pmot.pMOT_2Level import pmot_stark_component_model_label
from pmot.pmot_vec_2Level import load_vector_only_polarizability_table


@pytest.mark.parametrize(
    ("requested", "canonical", "label"),
    (
        ("svt", "svt", "2Level_pMOT_svt"),
        ("ts", "st", "2Level_pMOT_st"),
        ("sv", "sv", "2Level_pMOT_sv"),
        ("v", "v", "2Level_pMOT_v"),
        ("full", "svt", "2Level_pMOT_svt"),
        ("none", "", "2Level_pMOT_none"),
    ),
)
def test_component_codes_are_canonical(requested, canonical, label):
    assert normalize_pmot_stark_component_code(requested) == canonical
    assert pmot_stark_component_model_label(requested) == label


@pytest.mark.parametrize("invalid", ("ss", "svx", "scalar"))
def test_invalid_component_codes_are_rejected(invalid):
    with pytest.raises(ValueError):
        normalize_pmot_stark_component_code(invalid)


@pytest.mark.parametrize("code", ("", "s", "v", "t", "sv", "st", "vt", "svt"))
def test_component_tables_zero_only_unselected_ranks(code):
    table = load_pmot_2level_component_polarizability_table(code)
    arrays = {"s": table.scalar_si, "v": table.vector_si, "t": table.tensor_si}
    for letter, values in arrays.items():
        if letter in code:
            assert np.any(values != 0.0)
        else:
            assert np.all(values == 0.0)
        assert not values.flags.writeable


def test_vector_only_package_is_the_shared_v_selection():
    assert load_vector_only_polarizability_table() is (
        load_pmot_2level_component_polarizability_table("v")
    )

