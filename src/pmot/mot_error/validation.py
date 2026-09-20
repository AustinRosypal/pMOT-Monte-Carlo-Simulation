"""Concise Stage A-C validation report for the new multilevel foundation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atomic_structure import build_atomic_structure
from .atomic_structure import normalized_dipole_strength
from .configuration import default_multilevel_mot_config
from .events import EventChannel
from .events import sample_channel
from .events import sample_waiting_time_s
from .events import spontaneous_channels
from .polarization import polarization_weights
from .polarization import propagation_frame_polarization
from .trajectory import absorption_velocity_kick
from .trajectory import recoil_speed_m_per_s


@dataclass(frozen=True, slots=True)
class ValidationResult:
    name: str
    passed: bool
    detail: str


def run_preliminary_validation(seed: int = 12345) -> list[ValidationResult]:
    """Run the implemented pre-trajectory checks without invoking pytest."""

    structure = build_atomic_structure()
    config = default_multilevel_mot_config()
    results: list[ValidationResult] = []

    results.append(ValidationResult("State count", len(structure.states) == 23, f"8 ground + 15 excited = {len(structure.states)}"))
    valid_rules = all(
        transition.q in (-1, 0, 1)
        and transition.excited_m_f == transition.ground_m_f + transition.q
        and abs(transition.excited_f - transition.ground_f) <= 1
        for transition in structure.absorption_transitions
    )
    results.append(ValidationResult("Selection rules", valid_rules, f"{len(structure.absorption_transitions)} cooling transitions"))
    cycling = normalized_dipole_strength(2, 2, 3, 3)
    results.append(ValidationResult("CG normalization", np.isclose(cycling, 1.0), f"cycling C^2={cycling:.16g}"))

    sigma_weights = polarization_weights(propagation_frame_polarization((0.0, 0.0, 1.0), "sigma+"), (0.0, 0.0, 1.0))
    results.append(
        ValidationResult(
            "Polarization decomposition",
            np.isclose(sigma_weights[+1], 1.0) and np.isclose(sum(sigma_weights.values()), 1.0),
            f"p(-,pi,+)={sigma_weights[-1]:.3g},{sigma_weights[0]:.3g},{sigma_weights[+1]:.3g}",
        )
    )

    expected = {1: (5.0 / 6.0, 1.0 / 6.0), 2: (0.5, 0.5), 3: (0.0, 1.0)}
    branching_ok = True
    for excited_index in structure.excited_state_indices:
        excited = structure.states[excited_index]
        f1 = sum(
            branch.branch_probability
            for branch in structure.decay_by_excited[excited_index]
            if structure.states[branch.ground_state_index].f == 1
        )
        branching_ok &= np.isclose(f1, expected[excited.f][0], atol=1.0e-12)
    results.append(ValidationResult("Hyperfine branching ratios", bool(branching_ok), "F'=1: 5/6 dark; F'=2: 1/2; F'=3: 0"))

    decay_ok = all(
        np.isclose(
            sum(channel.rate_per_s for channel in spontaneous_channels(structure, index, config.natural_linewidth_rad_per_s)),
            config.natural_linewidth_rad_per_s,
            rtol=1.0e-12,
        )
        for index in structure.excited_state_indices
    )
    results.append(ValidationResult("Spontaneous normalization", bool(decay_ok), "all excited-state rates sum to Gamma"))

    rng = np.random.default_rng(seed)
    known_rate = 2.5e6
    waits = np.asarray([sample_waiting_time_s(known_rate, rng) for _ in range(50_000)])
    wait_error = abs(float(np.mean(waits)) * known_rate - 1.0)
    results.append(ValidationResult("Gillespie waiting times", wait_error < 0.02, f"relative mean error={wait_error:.3%}"))

    channels = [EventChannel("one", 0, 1.0), EventChannel("three", 1, 3.0)]
    fraction = sum(sample_channel(channels, rng).event_type == "three" for _ in range(50_000)) / 50_000
    results.append(ValidationResult("Gillespie channel weights", abs(fraction - 0.75) < 0.02, f"observed={fraction:.4f}, expected=0.7500"))

    recoil = float(np.linalg.norm(absorption_velocity_kick((1.0, 0.0, 0.0), config.wavelength_m)))
    expected_recoil = recoil_speed_m_per_s(config.wavelength_m)
    results.append(ValidationResult("Recoil magnitude", np.isclose(recoil, expected_recoil, rtol=1.0e-12), f"dv={recoil:.6e} m/s"))
    return results


def main() -> int:
    results = run_preliminary_validation()
    print("MULTILEVEL MOT FOUNDATION VALIDATION")
    print("------------------------------------")
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")
    passed = sum(result.passed for result in results)
    print(f"\n{passed} / {len(results)} implemented foundation checks passed")
    print("Trajectory-coupled optical-pumping checks remain intentionally gated for the next stage.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
