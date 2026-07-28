from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np

from pmot.atomic_data import RB87CoolingTransition
from pmot.configuration import HBAR_J_S
from pmot.configuration import RB87_MASS_KG
from pmot.configuration import STANDARD_GRAVITY_M_PER_S2
from pmot.fields import MOTBeam
from pmot.fields import build_mot_beams
from pmot.mot import GroundState
from pmot.mot import MOTAtomState
from pmot.mot import TransitionRateSample
from pmot.mot import default_anti_helmholtz_config
from pmot.mot import simulate_mot_trajectory
from pmot.mot import transition_rate_samples
from pmot.mot.simulation import beam_scattering_rates


KB_J_PER_K = 1.380649e-23
REPORT_BAR = "=" * 60


def print_report(name: str, status: str, measured: list[str], expected: list[str], reason: str) -> None:
    print(REPORT_BAR)
    print(f"TEST: {name}")
    print(f"STATUS: {status}")
    print("Measured:")
    for item in measured:
        print(f"    {item}")
    print("Expected:")
    for item in expected:
        print(f"    {item}")
    print("Reason:")
    print(f"    {reason}")
    print(REPORT_BAR)


def zero_field_coil_config():
    config = default_anti_helmholtz_config(radius_m=40.0e-3, turns_per_coil=50, target_gradient_g_per_cm=10.0)
    return replace(config, current_a=0.0)


def beams_with_cooling_detuning(detuning_hz: float) -> list[MOTBeam]:
    beams = build_mot_beams()
    updated: list[MOTBeam] = []
    for beam in beams:
        if beam.family == "cooling":
            updated.append(replace(beam, detuning_hz=detuning_hz))
        else:
            updated.append(beam)
    return updated


def wavevector_magnitude_m_inv(beam: MOTBeam) -> float:
    return 2.0 * np.pi / beam.wavelength_m


def mean_absorption_force_n(
    beams: list[MOTBeam],
    atom_state: MOTAtomState,
    coil_config,
) -> tuple[np.ndarray, list[TransitionRateSample]]:
    samples, _ = transition_rate_samples(beams, atom_state, coil_config)
    force = np.zeros(3, dtype=float)
    for sample in samples:
        beam = next(beam for beam in beams if beam.label == sample.beam_label)
        k_vector = wavevector_magnitude_m_inv(beam) * np.asarray(beam.direction, dtype=float)
        force += HBAR_J_S * sample.scattering_rate_per_s * k_vector
    return force, samples


def molasses_beams_x_only(detuning_hz: float) -> list[MOTBeam]:
    beams = []
    for beam in beams_with_cooling_detuning(detuning_hz):
        if beam.axis_name == "horizontal_x" and beam.family == "cooling":
            beams.append(beam)
    return beams


def cooling_transition_force_scale_n(beams: list[MOTBeam]) -> float:
    transition = RB87CoolingTransition()
    representative_beam = beams[0]
    return HBAR_J_S * wavevector_magnitude_m_inv(representative_beam) * transition.linewidth_hz / 2.0


def select_beam_from_rates(
    samples: list[TransitionRateSample],
    rng: np.random.Generator,
) -> str:
    rates_by_beam = beam_scattering_rates(samples)
    labels = list(rates_by_beam)
    weights = np.asarray([rates_by_beam[label] for label in labels], dtype=float)
    cumulative = np.cumsum(weights)
    target = rng.uniform(0.0, float(cumulative[-1]))
    index = int(np.searchsorted(cumulative, target, side="right"))
    return labels[index]


def ensemble_temperature_k(velocities_m_per_s: np.ndarray) -> tuple[float, float, float, float]:
    centered = velocities_m_per_s - np.mean(velocities_m_per_s, axis=0, keepdims=True)
    tx = RB87_MASS_KG * np.var(centered[:, 0]) / KB_J_PER_K
    ty = RB87_MASS_KG * np.var(centered[:, 1]) / KB_J_PER_K
    tz = RB87_MASS_KG * np.var(centered[:, 2]) / KB_J_PER_K
    return tx, ty, tz, (tx + ty + tz) / 3.0


def run_trajectory_ensemble(
    beams: list[MOTBeam],
    coil_config,
    atom_count: int,
    duration_s: float,
    time_step_s: float,
    seed: int,
    initial_speed_std_m_per_s: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    final_velocities = []
    for atom_index in range(atom_count):
        initial_velocity = tuple(rng.normal(0.0, initial_speed_std_m_per_s, size=3))
        record = simulate_mot_trajectory(
            beams=beams,
            coil_config=coil_config,
            initial_state=MOTAtomState(
                position_m=(0.0, 0.0, 0.0),
                velocity_m_per_s=initial_velocity,
                ground_state=GroundState(f=2, m_f=2),
            ),
            duration_s=duration_s,
            time_step_s=time_step_s,
            seed=int(rng.integers(0, 2**31 - 1)) + atom_index,
        )
        final_velocities.append(record.velocities_m_per_s[-1])
    return np.asarray(final_velocities, dtype=float)


def test_zero_field_optical_molasses() -> None:
    detuning_hz = -12.0e6
    beams = molasses_beams_x_only(detuning_hz)
    coil = zero_field_coil_config()
    transition = RB87CoolingTransition()
    representative_beam = beams[0]
    v_test = 0.2 * transition.linewidth_hz / wavevector_magnitude_m_inv(representative_beam)

    state_zero = MOTAtomState(position_m=(0.0, 0.0, 0.0), velocity_m_per_s=(0.0, 0.0, 0.0), ground_state=GroundState(2, 2))
    state_plus = MOTAtomState(position_m=(0.0, 0.0, 0.0), velocity_m_per_s=(v_test, 0.0, 0.0), ground_state=GroundState(2, 2))
    state_minus = MOTAtomState(position_m=(0.0, 0.0, 0.0), velocity_m_per_s=(-v_test, 0.0, 0.0), ground_state=GroundState(2, 2))

    force_zero, _ = mean_absorption_force_n(beams, state_zero, coil)
    force_plus, _ = mean_absorption_force_n(beams, state_plus, coil)
    force_minus, _ = mean_absorption_force_n(beams, state_minus, coil)

    f_scale = cooling_transition_force_scale_n(beams)
    force_tolerance = 1.0e-8 * f_scale
    symmetry_error = abs(force_plus[0] + force_minus[0]) / max(abs(force_plus[0]), abs(force_minus[0]))
    transverse_ratio_y = abs(force_plus[1]) / max(abs(force_plus[0]), 1.0e-30)
    transverse_ratio_z = abs(force_plus[2]) / max(abs(force_plus[0]), 1.0e-30)

    passed = (
        abs(force_zero[0]) <= force_tolerance
        and abs(force_zero[1]) <= force_tolerance
        and abs(force_zero[2]) <= force_tolerance
        and force_plus[0] < 0.0
        and force_minus[0] > 0.0
        and symmetry_error < 0.02
        and transverse_ratio_y < 0.01
        and transverse_ratio_z < 0.01
    )
    print_report(
        "Zero-field optical molasses",
        "PASS" if passed else "FAIL",
        [
            f"detuning = {detuning_hz / 1e6:.3f} MHz",
            f"velocity test = {v_test:.6f} m/s",
            f"F(0) = ({force_zero[0]:.6e}, {force_zero[1]:.6e}, {force_zero[2]:.6e}) N",
            f"F(+v) = ({force_plus[0]:.6e}, {force_plus[1]:.6e}, {force_plus[2]:.6e}) N",
            f"F(-v) = ({force_minus[0]:.6e}, {force_minus[1]:.6e}, {force_minus[2]:.6e}) N",
            f"symmetry error = {symmetry_error:.6f}",
            f"transverse ratios = ({transverse_ratio_y:.6f}, {transverse_ratio_z:.6f})",
        ],
        [
            "|F(0)| below tolerance",
            "F_x(+v) < 0",
            "F_x(-v) > 0",
            "symmetry error < 0.02",
            "transverse ratios < 0.01",
        ],
        "Force should oppose motion in zero-field molasses.",
    )
    assert passed


def test_zero_velocity_restoring_force() -> None:
    beams = beams_with_cooling_detuning(-12.0e6)
    coil = default_anti_helmholtz_config()
    r_test = 1.0e-3
    axis_specs = [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]
    all_passed = True
    for axis_name, direction in axis_specs:
        plus_state = MOTAtomState(tuple(r_test * direction), (0.0, 0.0, 0.0), GroundState(2, 2))
        minus_state = MOTAtomState(tuple(-r_test * direction), (0.0, 0.0, 0.0), GroundState(2, 2))
        force_plus, _ = mean_absorption_force_n(beams, plus_state, coil)
        force_minus, _ = mean_absorption_force_n(beams, minus_state, coil)
        axis_index = int(np.argmax(np.abs(direction)))
        symmetry_error = abs(force_plus[axis_index] + force_minus[axis_index]) / max(
            abs(force_plus[axis_index]), abs(force_minus[axis_index]), 1.0e-30
        )
        kappa = -(force_plus[axis_index] - force_minus[axis_index]) / (2.0 * r_test)
        transverse = np.delete(force_plus, axis_index)
        transverse_ratio = float(np.linalg.norm(transverse)) / max(abs(force_plus[axis_index]), 1.0e-30)
        axis_passed = (
            force_plus[axis_index] < 0.0
            and force_minus[axis_index] > 0.0
            and symmetry_error < 0.05
            and kappa > 0.0
            and transverse_ratio < 0.05
        )
        all_passed = all_passed and axis_passed
        print_report(
            f"Zero-velocity restoring force ({axis_name})",
            "PASS" if axis_passed else "FAIL",
            [
                f"r_test = {r_test:.6e} m",
                f"F(+r) = ({force_plus[0]:.6e}, {force_plus[1]:.6e}, {force_plus[2]:.6e}) N",
                f"F(-r) = ({force_minus[0]:.6e}, {force_minus[1]:.6e}, {force_minus[2]:.6e}) N",
                f"kappa = {kappa:.6e} N/m",
                f"symmetry error = {symmetry_error:.6f}",
                f"transverse ratio = {transverse_ratio:.6f}",
            ],
            [
                "F_i(+r) < 0",
                "F_i(-r) > 0",
                "symmetry error < 0.05",
                "kappa > 0",
                "transverse ratio < 0.05",
            ],
            "Force should point toward the trap center.",
        )
    assert all_passed


def test_beam_selection_probability() -> None:
    beams = molasses_beams_x_only(-12.0e6) + [beam for beam in beams_with_cooling_detuning(-12.0e6) if beam.axis_name == "vertical_z" and beam.family == "cooling"]
    coil = zero_field_coil_config()
    transition = RB87CoolingTransition()
    representative_beam = beams[0]
    v_test = 0.2 * transition.linewidth_hz / wavevector_magnitude_m_inv(representative_beam)
    atom = MOTAtomState(position_m=(0.0, 0.0, 0.0), velocity_m_per_s=(0.0, 0.0, -v_test), ground_state=GroundState(2, 2))
    samples, _ = transition_rate_samples(beams, atom, coil)
    rates_by_beam = beam_scattering_rates(samples)
    total_rate = sum(rates_by_beam.values())
    labels = sorted(rates_by_beam)
    expected_p = {label: rates_by_beam[label] / total_rate for label in labels}
    plus_z_label = next(label for label in labels if label.startswith("vertical_z_incident"))
    minus_z_label = next(label for label in labels if label.startswith("vertical_z_retro"))
    assert rates_by_beam[plus_z_label] > rates_by_beam[minus_z_label]

    rng = np.random.default_rng(12345)
    draw_count = 100_000
    counts = {label: 0 for label in labels}
    for _ in range(draw_count):
        counts[select_beam_from_rates(samples, rng)] += 1

    per_beam_pass = True
    measured_lines = []
    for label in labels:
        p_value = expected_p[label]
        observed_fraction = counts[label] / draw_count
        sigma = sqrt(draw_count * p_value * (1.0 - p_value))
        z_score = abs(counts[label] - draw_count * p_value) / max(sigma, 1.0e-30)
        tolerance = max(5.0 * sqrt(p_value * (1.0 - p_value) / draw_count), 1.0e-3)
        if draw_count * p_value >= 20.0:
            per_beam_pass = per_beam_pass and (z_score < 5.0)
        per_beam_pass = per_beam_pass and (abs(observed_fraction - p_value) < tolerance)
        measured_lines.append(
            f"{label}: rate={rates_by_beam[label]:.6e}  p={p_value:.6f}  f={observed_fraction:.6f}  count={counts[label]}  z={z_score:.3f}"
        )

    passed = per_beam_pass and (counts[plus_z_label] / draw_count > counts[minus_z_label] / draw_count)
    print_report(
        "Beam-selection probability",
        "PASS" if passed else "FAIL",
        measured_lines
        + [
            f"R_(+z)/R_(-z) = {rates_by_beam[plus_z_label] / rates_by_beam[minus_z_label]:.6f}",
            f"p_(+z)/p_(-z) = {expected_p[plus_z_label] / expected_p[minus_z_label]:.6f}",
            f"N = {draw_count}",
            "seed = 12345",
        ],
        [
            "R_(+z) > R_(-z)",
            "f_(+z) > f_(-z)",
            "per-beam z-score < 5 for common categories",
            "|f_j - p_j| within binomial tolerance",
        ],
        "Beam-selection frequencies should match the fixed conditional beam probabilities.",
    )
    assert passed


def test_heating_versus_cooling_under_detuning_reversal() -> None:
    detuning_red_hz = -12.0e6
    detuning_zero_hz = 0.0
    detuning_blue_hz = +12.0e6
    coil = zero_field_coil_config()
    reference_beams = molasses_beams_x_only(detuning_red_hz)
    representative_beam = reference_beams[0]
    transition = RB87CoolingTransition()
    v_test = 0.2 * transition.linewidth_hz / wavevector_magnitude_m_inv(representative_beam)

    def alpha_for_detuning(detuning_hz: float) -> tuple[float, np.ndarray, np.ndarray]:
        beams = molasses_beams_x_only(detuning_hz)
        plus_state = MOTAtomState((0.0, 0.0, 0.0), (v_test, 0.0, 0.0), GroundState(2, 2))
        minus_state = MOTAtomState((0.0, 0.0, 0.0), (-v_test, 0.0, 0.0), GroundState(2, 2))
        force_plus, _ = mean_absorption_force_n(beams, plus_state, coil)
        force_minus, _ = mean_absorption_force_n(beams, minus_state, coil)
        alpha = -(force_plus[0] - force_minus[0]) / (2.0 * v_test)
        return alpha, force_plus, force_minus

    alpha_red, force_red_plus, force_red_minus = alpha_for_detuning(detuning_red_hz)
    alpha_zero, force_zero_plus, force_zero_minus = alpha_for_detuning(detuning_zero_hz)
    alpha_blue, force_blue_plus, force_blue_minus = alpha_for_detuning(detuning_blue_hz)
    red_blue_symmetry_error = abs(alpha_red + alpha_blue) / max(abs(alpha_red), abs(alpha_blue), 1.0e-30)
    zero_ratio = abs(alpha_zero) / max(abs(alpha_red), 1.0e-30)
    passed = (
        alpha_red > 0.0
        and alpha_blue < 0.0
        and force_red_plus[0] < 0.0
        and force_blue_plus[0] > 0.0
        and zero_ratio < 0.1
        and red_blue_symmetry_error < 0.1
    )
    print_report(
        "Heating versus cooling under detuning reversal",
        "PASS" if passed else "FAIL",
        [
            f"detunings [MHz] = ({detuning_red_hz / 1e6:.3f}, {detuning_zero_hz / 1e6:.3f}, {detuning_blue_hz / 1e6:.3f})",
            f"F_red(+v), F_red(-v) = ({force_red_plus[0]:.6e}, {force_red_minus[0]:.6e}) N",
            f"F_zero(+v), F_zero(-v) = ({force_zero_plus[0]:.6e}, {force_zero_minus[0]:.6e}) N",
            f"F_blue(+v), F_blue(-v) = ({force_blue_plus[0]:.6e}, {force_blue_minus[0]:.6e}) N",
            f"alpha_red = {alpha_red:.6e}",
            f"alpha_zero = {alpha_zero:.6e}",
            f"alpha_blue = {alpha_blue:.6e}",
            f"|alpha_zero|/|alpha_red| = {zero_ratio:.6f}",
            f"red-blue symmetry error = {red_blue_symmetry_error:.6f}",
        ],
        [
            "alpha_red > 0",
            "alpha_blue < 0",
            "F_red(+v) < 0",
            "F_blue(+v) > 0",
            "|alpha_zero| < 0.1 |alpha_red|",
            "red-blue symmetry error < 0.1",
        ],
        "Red detuning should damp, blue detuning should anti-damp, and zero detuning should have weak linear friction.",
    )
    assert passed


def test_doppler_temperature_scale(monkeypatch) -> None:
    monkeypatch.setattr("pmot.mot.simulation.GRAVITY_ACCELERATION_M_PER_S2", (0.0, 0.0, 0.0))
    beams = molasses_beams_x_only(-RB87CoolingTransition().linewidth_hz / 2.0)
    coil = zero_field_coil_config()
    linewidth_hz = RB87CoolingTransition().linewidth_hz
    t_d = HBAR_J_S * linewidth_hz / (2.0 * KB_J_PER_K)
    atom_count = 120
    duration_s = 1.0e-3
    dt_large = 2.0e-6
    dt_small = 1.0e-6
    initial_speed_std = 0.25

    final_velocities_large = run_trajectory_ensemble(
        beams=beams,
        coil_config=coil,
        atom_count=atom_count,
        duration_s=duration_s,
        time_step_s=dt_large,
        seed=12345,
        initial_speed_std_m_per_s=initial_speed_std,
    )
    final_velocities_small = run_trajectory_ensemble(
        beams=beams,
        coil_config=coil,
        atom_count=atom_count,
        duration_s=duration_s,
        time_step_s=dt_small,
        seed=12345,
        initial_speed_std_m_per_s=initial_speed_std,
    )
    initial_velocities = np.random.default_rng(12345).normal(0.0, initial_speed_std, size=(atom_count, 3))
    _, _, _, t_initial = ensemble_temperature_k(initial_velocities)
    tx_large, ty_large, tz_large, t_large = ensemble_temperature_k(final_velocities_large)
    tx_small, ty_small, tz_small, t_small = ensemble_temperature_k(final_velocities_small)
    anisotropy_ratio = max(tx_small, ty_small, tz_small) / max(min(tx_small, ty_small, tz_small), 1.0e-30)
    timestep_metric = abs(t_large - t_small) / max(t_small, 1.0e-30)

    window_atom_count = atom_count // 4
    window_temperatures = []
    for index in range(4):
        block = final_velocities_small[index * window_atom_count : (index + 1) * window_atom_count]
        _, _, _, block_t = ensemble_temperature_k(block)
        window_temperatures.append(block_t)
    stationarity_metric = (max(window_temperatures) - min(window_temperatures)) / max(np.mean(window_temperatures), 1.0e-30)

    passed = (
        t_small < t_initial
        and t_small > 0.0
        and 0.3 * t_d <= t_small <= 3.0 * t_d
        and anisotropy_ratio < 1.25
        and timestep_metric < 0.15
        and stationarity_metric < 0.15
    )
    print_report(
        "Doppler-temperature scale",
        "PASS" if passed else "FAIL",
        [
            f"atomic mass = {RB87_MASS_KG:.6e} kg",
            f"linewidth convention = Hz, Gamma = {linewidth_hz:.6e} Hz",
            f"detuning = {-linewidth_hz / 2.0 / 1e6:.6f} MHz",
            f"gravity disabled = yes, beams only in x-axis molasses",
            f"dt_large = {dt_large:.3e} s, dt_small = {dt_small:.3e} s",
            f"N_atoms = {atom_count}",
            f"T_initial = {t_initial:.6e} K",
            f"T_D = {t_d:.6e} K",
            f"T_large = {t_large:.6e} K",
            f"T_x, T_y, T_z = ({tx_small:.6e}, {ty_small:.6e}, {tz_small:.6e}) K",
            f"T_final = {t_small:.6e} K",
            f"T_final/T_D = {t_small / t_d:.6f}",
            f"stationarity metric = {stationarity_metric:.6f}",
            f"anisotropy ratio = {anisotropy_ratio:.6f}",
            f"timestep metric = {timestep_metric:.6f}",
        ],
        [
            "T_final < T_initial",
            "0.3 T_D <= T_final <= 3 T_D",
            "anisotropy ratio < 1.25",
            "stationarity metric < 0.15",
            "timestep metric < 0.15",
        ],
        "Doppler molasses should settle to a finite recoil-limited temperature scale.",
    )
    assert passed
