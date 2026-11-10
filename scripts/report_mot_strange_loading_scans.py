"""Render the audited symmetric strange-MOT loading scans."""

import base64
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from math import pi, sqrt
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

from run_mot_strange_loading_pilot import NAME, ROOT


SCAN_LABELS = {
    "trap_power_isotropic": ("Equal-power trapping scan", "trap_power_mw"),
    "trap_power_axial": ("Doubled-axial trapping scan", "trap_power_mw"),
    "cooling_detuning": ("Cooling detuning scan", "cooling_detuning_mhz"),
    "cooling_diameter": ("Cooling diameter scan", "cooling_diameter_mm"),
}
SERIES = {
    "core": ("Core retention", "#236b8d", "#b9dde0", "o"),
    "offcore": ("Verified off-core restoring pocket", "#ad542e", "#efc6af", "s"),
}

REPORT_CONTEXT = (
    "These scans describe a conventional 24-state Rb-87 population-rate MOT "
    "driven by an intensity-derived magnetic-field surrogate. The local steady-state "
    "solver uses incoherent cooling and repump components, deterministic mean force "
    "and gravity.",
    "The 1529 nm light sets the surrogate field geometry and calibration. "
    "A full optical pMOT prediction additionally requires a state-resolved Stark "
    "Hamiltonian, conservative optical forces and trapping-light scattering and heating.",
    "The gradient axes show the derivative at the origin of the localized surrogate "
    "field. They do not describe a spatially uniform coil gradient across the launch "
    "aperture. Both trapping-power scans keep the 780 nm cooling and repump settings fixed.",
    "Loading refers to the stated 2 s observation and final 100 ms residence window "
    "within the finite cell and launch aperture. Core retention and verified restoring "
    "off-core capture are distinct outcomes. Localized trajectories without a verified "
    "restoring pocket and other retained trajectories remain separate audit categories. "
    "Rates use the declared 0–80 m/s velocity grid, with representative 85–120 m/s tail checks.",
    "Intervals are pointwise 95% Student-t intervals across independent direction discs. "
    "Zero observed captures have undefined Student-t precision. Sample counts were "
    "fixed before production, and the preferred 10% relative half-width is assessed "
    "from the completed sample. These intervals exclude model and velocity-grid errors. "
    "Curve maxima describe the sampled settings.",
)


def parameters(manifest, records):
    geometry, vapor = manifest["geometry"], manifest["vapor"]
    first = records[0]["point"]
    split = "(P, P, P)" if first["geometry"] == "isotropic" else "(P/2, P/2, P)"
    powers = ", ".join(f"{v:g}" for v in sorted({r["point"]["trap_power_mw"] for r in records}))
    detunings = ", ".join(f"{v:g}" for v in sorted({r["point"]["cooling_detuning_mhz"] for r in records}))
    diameters = ", ".join(f"{v:g}" for v in sorted({r["point"]["cooling_diameter_mm"] for r in records}))
    cooling_points = {
        (row["point"]["cooling_detuning_mhz"], row["point"]["cooling_diameter_mm"]): row["point"]
        for row in records
    }
    saturation = ", ".join(
        f"({p['cooling_detuning_mhz']:g}, {p['cooling_diameter_mm']:g}, "
        f"{p['cooling_s0']:.6g}, {p['cooling_s_eff']:.6g})"
        for p in cooling_points.values()
    )
    samples = ", ".join(
        f"{discs} × {points}" for discs, points in sorted({
            (row["discs"], row["points_per_disc"]) for row in records
        })
    )
    intensity_per_mw = ", ".join(
        f"{pair[0] / first['trap_power_mw']:.8g}"
        for pair in first["origin_trapping_intensity_w_per_cm2_xyz_each_direction"]
    )
    gradient_per_mw = ", ".join(
        f"{axis[i] / first['trap_power_mw']:.8g}"
        for i, axis in enumerate(first["origin_surrogate_field_jacobian_g_per_cm"])
    )
    return (
        f"Rb-87 surrogate MOT · Trap {geometry['wavelength_m'] * 1e9:.6f} nm, "
        f"waist centres {geometry['forward_waist_position_m'] * 1e3:g}/{geometry['return_waist_position_m'] * 1e3:+g} mm, "
        f"waist radius {1e6 * sqrt(geometry['wavelength_m'] * geometry['forward_rayleigh_range_m'] / pi):.7g} µm, "
        f"35 mm input diameter, 80.3 mm focal length, XYZ powers {split} and gradient signs (+,+,−).\n"
        "Trap components have equal incident and return powers with propagation-frame helicities σ+, σ+, σ−.\n"
        f"Trap P [mW/component] = {powers}\n"
        f"For the listed P [mW], central intensity XYZ per traveling component = P × "
        f"({intensity_per_mw}) W/cm²; central gradient XYZ = P × ({gradient_per_mw}) G/cm.\n"
        "Cooling 780.241209686 nm, 27 mW/component, detuning from F=2→F′=3. "
        "Repump 780.232684 nm, 0.1 mW/component, 12.7 mm, 0 MHz from ARC F=1→F′=2. "
        "Both colors have six traveling components, helicities σ+, σ+, σ− on x, y, z.\n"
        f"Cooling detuning [MHz] = {detunings}. Cooling Gaussian 1/e² diameter [mm] = {diameters}.\n"
        f"Gravity −z, no recoil · cell half-length {1e3 * geometry['cell_half_length_m']:.7g} mm · "
        f"full-sphere {1e3 * manifest['radius_m']:g} mm launch discs clipped to the cell, "
        f"plane distance {1e3 * manifest['plane_distance_m']:g} mm · "
        f"vapor {vapor['temperature_k']:g} K, {vapor['rb87_partial_pressure_torr']:g} Torr partial pressure.\n"
        f"Observation {manifest['horizon_s']:g} s, final window {1e3 * manifest['final_window_s']:g} ms · "
        "2 mm core · pointwise 95% Student-t intervals over independent direction discs.\n"
        "Core requires a 5 ms continuous core event or two entries separated by an exit, plus final-window core residence.\n"
        "Local steady-state 24-state force. Speeds sampled from the 0.125 m/s grid through 80 m/s, "
        "two draws per velocity stratum.\n"
        "Cooling s₀ uses 16.69 W/m² and s_eff uses a 6.065 MHz reference linewidth.\n"
        f"(Cooling Δ [MHz], D [mm], s₀, s_eff) = {saturation}\n"
        f"Independent discs × launch points per disc at each setting = {samples}."
    )


def draw(scan, records, manifest):
    title, control = SCAN_LABELS[scan]
    records = sorted(records, key=lambda row: row["point"][control])
    point = records[0]["point"]
    context = parameters(manifest, records)
    context_lines = "\n".join(fill(line, width=141) for line in context.splitlines())
    context_height = len(context_lines.splitlines()) * .18
    height = context_height + 5.8
    fig, ax = plt.subplots(figsize=(12.5, height))
    fig.subplots_adjust(left=.12, right=.96, top=(height - context_height - 1.3) / height,
                        bottom=1.05 / height)
    fig.text(.12, 1 - .25 / height, title, fontsize=17, weight="bold", va="top")
    fig.text(.12, 1 - .75 / height, context_lines,
             fontsize=8.5, va="top", linespacing=1.45)
    if control == "trap_power_mw":
        intensity = "origin_trapping_intensity_w_per_cm2_xyz_each_direction"
        jacobian = "origin_surrogate_field_jacobian_g_per_cm"
        values = [row["point"][intensity][2][0] for row in records]
        scale = point[jacobian][2][2] / point[intensity][2][0]
        top = ax.secondary_xaxis("top", functions=(lambda x: x * scale, lambda g: g / scale))
        top.set_xlabel("Signed central axial gradient ∂Bz/∂z [G/cm]")
        ax.set_xlabel("Central trapping intensity per axial traveling component [W/cm²]")
    else:
        values = [row["point"][control] for row in records]
        ax.set_xlabel({"cooling_detuning_mhz": "Cooling detuning [MHz]",
                       "cooling_diameter_mm": "Cooling Gaussian 1/e² diameter [mm]"}[control])
    for key, (label, color, pastel, marker) in SERIES.items():
        if key == "offcore" and all(row["counts"][key] == 0 for row in records):
            continue
        stats = [row[key] for row in records]
        means = [s["mean_atoms_per_s"] for s in stats]
        lower = [float("nan") if s["lower_atoms_per_s"] is None else s["lower_atoms_per_s"] for s in stats]
        upper = [float("nan") if s["upper_atoms_per_s"] is None else s["upper_atoms_per_s"] for s in stats]
        ax.fill_between(values, lower, upper, color=pastel, alpha=.55, linewidth=0)
        ax.plot(values, means, marker=marker, color=color, label=label, markersize=4.2, linewidth=2)
    ax.set_ylabel("Loading rate [atoms/s]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=.22)
    ax.legend()
    fig.text(.12, .20 / height,
             "Shaded bands show pointwise 95% Student-t intervals across direction discs.\n"
             "Exact per-point powers, intensities, gradients, cooling saturation, "
             "sample counts and intervals are in this figure’s data table.", fontsize=8.5)
    return fig, records


def point_table(records):
    headers = ("Power per component XYZ [mW]", "Central intensity per component XYZ [W/cm²]",
               "Central gradient XYZ [G/cm]",
               "Cooling Δ [MHz]", "Cooling D [mm]", "Cooling s₀", "Cooling s_eff",
               "Discs × points", "Core atoms/s [95% CI]", "Core relative half-width",
               "Off-core atoms/s [95% CI]", "Off-core relative half-width")

    def numbers(values):
        return ", ".join(f"{x:.8g}" for x in values)

    def interval(stats):
        mean = stats["mean_atoms_per_s"]
        low, high = stats["lower_atoms_per_s"], stats["upper_atoms_per_s"]
        return f"{mean:.8g} [{low:.8g}, {high:.8g}]" if low is not None else f"{mean:.8g} (CI undefined)"

    def relative(stats):
        half = stats["relative_half_width"]
        return f"{half:.2%}" if half is not None else "Undefined"

    rows = []
    for row in records:
        p = row["point"]
        rows.append([
            numbers(p["trapping_power_mw_xyz"]),
            numbers([pair[0] for pair in p["origin_trapping_intensity_w_per_cm2_xyz_each_direction"]]),
            numbers([axis[i] for i, axis in enumerate(p["origin_surrogate_field_jacobian_g_per_cm"])]),
            f"{p['cooling_detuning_mhz']:g}", f"{p['cooling_diameter_mm']:g}",
            f"{p['cooling_s0']:.8g}", f"{p['cooling_s_eff']:.8g}",
            f"{row['discs']} × {row['points_per_disc']}", interval(row["core"]), relative(row["core"]),
            interval(row["offcore"]), relative(row["offcore"]),
        ])
    head = "".join(f"<th>{escape(column)}</th>" for column in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    manifest = json.loads((ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "final_audit.json").read_text(encoding="utf-8"))
    results_bytes = (ROOT / "final_loading_results.json").read_bytes()
    results = json.loads(results_bytes)
    if (audit["status"] != "passed" or audit["result_sha256"] != sha256(results_bytes).hexdigest()
            or set(results) != set(map(str, range(len(manifest["points"]))))):
        raise RuntimeError("The report requires complete, audited production results")
    offcore = sum(row["counts"]["offcore"] for row in results.values())
    localized = sum(row["counts"]["localized_only"] for row in results.values())
    outcome = (
        f"The production audit found {offcore:,} verified off-core speed draws and "
        f"{localized:,} localized-only draws (including repeated draws). Off-core "
        "verification requires an admissible nearby force balance, stable three-dimensional "
        "motion and return after perturbations. A series with no verified draws is "
        "omitted from the plots; its zero estimate and undefined interval remain in the tables."
    )
    context = (*REPORT_CONTEXT, outcome)
    html_sections, cells = [], [nbformat.v4.new_markdown_cell(
        "# Symmetric strange-MOT loading scans\n\n"
        + "\n\n".join(context)
    )]
    for scan, ids in manifest["scans"].items():
        records = [dict(results[str(i)], point=manifest["points"][i]) for i in ids]
        fig, records = draw(scan, records, manifest)
        png = BytesIO()
        fig.savefig(png, dpi=150)
        plt.close(fig)
        encoded = base64.b64encode(png.getvalue()).decode("ascii")
        table = point_table(records)
        title = SCAN_LABELS[scan][0]
        section = (f"<section><h2>{escape(title)}</h2><img alt='{escape(title)}' "
                   f"src='data:image/png;base64,{encoded}'>"
                   f"<details><summary>Exact plotted values and controls</summary>{table}</details></section>")
        html_sections.append(section)
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}", attachments={
            "scan.png": {"image/png": encoded}
        }))
        cells[-1].source += f"\n\n![{title}](attachment:scan.png)\n\n{table}"
    html_file = Path("visualizations/mot_loading") / f"{NAME}.html"
    notebook_file = Path("notebooks/mot_strange") / f"{NAME}.ipynb"
    html_file.parent.mkdir(parents=True, exist_ok=True)
    notebook_file.parent.mkdir(parents=True, exist_ok=True)
    html_file.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Symmetric strange-MOT loading scans</title>"
        "<style>body{max-width:1300px;margin:2rem auto;padding:0 1rem;"
        "font:16px system-ui}img{width:100%;height:auto}details{overflow-x:auto}"
        "table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ddd}"
        "section{margin-bottom:3rem}</style><h1>Symmetric strange-MOT loading scans</h1>"
        + "".join(f"<p>{escape(paragraph)}</p>" for paragraph in context)
        + "".join(html_sections) + "</html>", encoding="utf-8",
    )
    notebook = nbformat.v4.new_notebook(cells=cells)
    if notebook_file.exists():
        notebook.metadata.update(nbformat.read(notebook_file, as_version=4).metadata)
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell(
            "## Regenerate the figures\n\n"
            "Run from this checkout with the audited production JSON available. "
            "The figures and tables above are embedded for offline reading."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "from IPython.display import display\n\n"
            "here = Path.cwd()\n"
            "root = next(p for p in (here, *here.parents)\n"
            "            if (p / 'scripts' / 'report_mot_strange_loading_scans.py').is_file())\n"
            "sys.path.insert(0, str(root / 'scripts'))\n"
            "from report_mot_strange_loading_scans import draw\n\n"
            f"data = root / 'outputs/statistics/mot_strange/{NAME}'\n"
            "manifest = json.loads((data / 'pilot_manifest.json').read_text(encoding='utf-8'))\n"
            "results = json.loads((data / 'final_loading_results.json').read_text(encoding='utf-8'))\n"
            "for scan, indices in manifest['scans'].items():\n"
            "    rows = [dict(results[str(i)], point=manifest['points'][i]) for i in indices]\n"
            "    figure, _ = draw(scan, rows, manifest)\n"
            "    display(figure)"
        ),
    ])
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_file)


if __name__ == "__main__":
    main()
