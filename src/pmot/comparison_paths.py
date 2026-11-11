"""Canonical destinations and descriptive names for cross-model reports."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_NOTEBOOKS = REPOSITORY_ROOT / "notebooks/model_comparisons"
COMPARISON_FIGURES = REPOSITORY_ROOT / "outputs/figures/model_comparisons"
COMPARISON_NOTEBOOK_NAMES = {
    "force_three": "force_and_gradient_multilevel_mot_strange_mot_2level_pmot_20261020.ipynb",
    "force_four": "force_and_gradient_multilevel_mot_strange_mot_2level_pmot_svt_vs_vector.ipynb",
    "loading_two": "loading_2level_mot_vs_multilevel_mot_exact_d2_20261104.ipynb",
    "loading_two_refined": "loading_2level_mot_vs_multilevel_mot_exact_d2_refined_20261104.ipynb",
    "loading_three": "loading_gradient_detuning_diameter_2level_mot_multilevel_mot_strange_mot_20261111.ipynb",
    "loading_four": "loading_gradient_2level_mot_multilevel_mot_strange_mot_2level_pmot_20261111.ipynb",
}
