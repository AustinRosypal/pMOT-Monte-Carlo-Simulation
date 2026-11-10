"""Direction-cluster loading statistics for cell-clipped launch discs."""

import numpy as np
from scipy.stats import t

from ..loading import VaporLoadingParameters


def loading_by_ray(
    masks: np.ndarray,
    speeds_m_per_s: np.ndarray,
    disc_areas_m2: np.ndarray,
    vapor: VaporLoadingParameters,
) -> np.ndarray:
    """Return loading contributions before averaging points and directions."""
    masks = np.asarray(masks)
    speeds = np.asarray(speeds_m_per_s, dtype=float)
    areas = np.asarray(disc_areas_m2, dtype=float)
    if (
        masks.ndim != 3
        or speeds.ndim != 1
        or len(speeds) < 2
        or masks.shape[2] != len(speeds)
        or areas.shape != (masks.shape[0],)
        or not np.all(np.isfinite(areas) & (areas > 0))
        or not np.all(np.isfinite(speeds))
        or speeds[0] != 0
        or np.any(np.diff(speeds) <= 0)
    ):
        raise ValueError("inconsistent launch discs or velocity grid")
    if not np.all(np.isin(masks[..., 1:], (0, 1))):
        raise ValueError("unresolved positive-flux capture nodes")
    weight = speeds**3 * np.exp(-speeds**2 / vapor.thermal_scale_m2_per_s2)
    integrand = np.zeros(masks.shape, dtype=float)
    integrand[..., 1:] = masks[..., 1:] * weight[1:]
    return vapor.loading_rate_prefactor * areas[:, None] * np.trapezoid(
        integrand, speeds, axis=2
    )


def loading_interval(disc_rates: np.ndarray) -> dict:
    """Pointwise 95% Student-t interval over independent direction discs."""
    rates = np.asarray(disc_rates, dtype=float)
    if rates.ndim != 1 or len(rates) < 2 or not np.all(np.isfinite(rates) & (rates >= 0)):
        raise ValueError("require at least two finite nonnegative disc rates")
    mean = float(rates.mean())
    if not mean:
        return {
            "mean_atoms_per_s": 0.0,
            "lower_atoms_per_s": None,
            "upper_atoms_per_s": None,
            "relative_half_width": None,
            "observed_zero_capture": True,
        }
    half_width = float(t.ppf(0.975, len(rates) - 1) * rates.std(ddof=1) / np.sqrt(len(rates)))
    return {
        "mean_atoms_per_s": mean,
        "lower_atoms_per_s": mean - half_width,
        "upper_atoms_per_s": mean + half_width,
        "relative_half_width": half_width / mean,
        "observed_zero_capture": False,
    }


def plan_production(
    pilot_ray_rates: np.ndarray,
    *,
    target_relative_half_width: float = .08,
    variance_inflation: float = 1.25,
    max_rays: int = 12800,
) -> dict:
    """Select a fixed sample count using only independent pilot directions."""
    rates = np.asarray(pilot_ray_rates, dtype=float)
    if rates.ndim != 2 or min(rates.shape) < 2 or not np.all(np.isfinite(rates) & (rates >= 0)):
        raise ValueError("pilot needs finite nonnegative samples on at least two discs")
    if not 0 < target_relative_half_width < 1 or variance_inflation < 1:
        raise ValueError("invalid production precision parameters")
    if max_rays < 128 * 16:
        raise ValueError("production budget cannot cover 128 discs with 16 points each")
    mean = float(rates.mean())
    within = float(rates.var(axis=1, ddof=1).mean())
    between = max(0., float(rates.mean(axis=1).var(ddof=1)) - within / rates.shape[1])
    candidates = []
    for points_per_disc in (16, 25, 32, 50):
        limit = max_rays // points_per_disc
        counts = sorted(set(range(128, limit + 1, 16)) | ({limit} if limit >= 128 else set()))
        for discs in counts:
            variance = variance_inflation * (between + within / points_per_disc) / discs
            relative = float(t.ppf(.975, discs - 1) * np.sqrt(variance) / mean) if mean else None
            candidates.append({
                "disc_count": discs, "points_per_disc": points_per_disc,
                "ray_count": discs * points_per_disc, "predicted_relative_half_width": relative,
            })
    acceptable = [
        candidate for candidate in candidates
        if candidate["predicted_relative_half_width"] is not None
        and candidate["predicted_relative_half_width"] <= target_relative_half_width
    ]
    if acceptable:
        choice = min(acceptable, key=lambda row: (row["ray_count"], -row["disc_count"]))
    elif mean:
        choice = min(candidates, key=lambda row: (row["predicted_relative_half_width"], -row["disc_count"]))
    else:
        choice = max(candidates, key=lambda row: (row["ray_count"], row["disc_count"]))
    return {
        **choice, "predicted_target_met": bool(acceptable),
        "target_relative_half_width": target_relative_half_width,
        "pilot_mean_atoms_per_s": mean,
        "pilot_between_disc_variance": between,
        "pilot_within_disc_variance": within,
        "variance_inflation": variance_inflation, "max_rays": max_rays,
    }
