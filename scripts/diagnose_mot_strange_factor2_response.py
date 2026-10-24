"""Compare old doubled and corrected mot_strange vector conversions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t


def main() -> None:
    scan_m = 1.0e-3 * np.unique(
        np.round(
            np.concatenate(
                (
                    np.linspace(-12.5, 12.5, 301),
                    np.linspace(-1.0, 1.0, 161),
                    np.linspace(-12.5, -7.5, 201),
                    np.linspace(7.5, 12.5, 201),
                )
            ),
            10,
        )
    )
    config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=27.0e-3,
        repump_power_w_per_beam=0.1e-3,
        cooling_detuning_rad_per_s=2.0 * np.pi * -15.0e6,
        include_gravity=False,
    )
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    corrected = DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    old = replace(
        corrected,
        differential_vector_polarizability_hz_per_w_per_m2=(
            2.0 * corrected.differential_vector_polarizability_hz_per_w_per_m2
        ),
    )

    for axis, component in zip("xyz", range(3), strict=True):
        for label, field_config in (("corrected", corrected), ("old_2x", old)):
            force_n = np.empty(len(scan_m))
            for index, coordinate_m in enumerate(scan_m):
                position = np.zeros(3)
                position[component] = coordinate_m
                observable = rate_equation_observable(
                    model,
                    beams,
                    tuple(position),
                    (0.0, 0.0, 0.0),
                    None,
                    config,
                    (0.0, 0.0, 1.0),
                    magnetic_field_function=lambda point, cfg=field_config: (
                        surrogate_effective_field_t(point, cfg)
                    ),
                )
                force_n[index] = observable.force_n[component]
            gradient = np.gradient(force_n, scan_m, edge_order=2)
            maximum_index = int(np.argmax(gradient))
            minimum_index = int(np.argmin(gradient))
            position = np.zeros(3)
            position[component] = scan_m[maximum_index]
            field_g = 1.0e4 * surrogate_effective_field_t(
                position, field_config
            )[component]
            center_index = int(np.argmin(np.abs(scan_m)))
            print(
                axis,
                label,
                "origin_gradient_1e19=",
                gradient[center_index] * 1.0e19,
                "max_antirestoring_1e19=",
                gradient[maximum_index] * 1.0e19,
                "at_mm=",
                scan_m[maximum_index] * 1.0e3,
                "field_g=",
                field_g,
                "force_zN=",
                force_n[maximum_index] * 1.0e21,
                "most_restoring_1e19=",
                gradient[minimum_index] * 1.0e19,
                "at_mm=",
                scan_m[minimum_index] * 1.0e3,
            )


if __name__ == "__main__":
    main()
