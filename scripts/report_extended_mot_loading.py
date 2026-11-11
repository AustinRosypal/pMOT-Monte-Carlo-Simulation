"""Merge audited gradient additions and build reproducible MOT comparison notebooks."""

import base64
from collections import defaultdict
from copy import deepcopy
import csv
from html import escape
from io import BytesIO
import json
from pathlib import Path
from shutil import copy2
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np

import extend_mot_strange_gradient_loading as extension


REPO = Path(__file__).resolve().parents[1]
OUTPUT = (REPO / extension.OUTPUT).resolve()
SIMPLE = REPO / "outputs/statistics/mot_simple/loading_scans_exact_D2_20261103"
MULTI = REPO / "outputs/statistics/mot_multilevel_population_rate_v1/matched_two_level_exact_D2_fast_20261104"
PMOT = REPO / "outputs/statistics/pMOT_2Level/loading_20261109/impact_stratified"
FIGURES = REPO / "outputs/figures/mot_loading/extended_comparison_20261111"
NOTEBOOKS = REPO / "notebooks/mot_comparisons"
COLORS = {"2-level MOT": "#2563eb", "Multilevel MOT": "#6d28d9",
          "Strange MOT": "#d97706", "2-level pMOT": "#15803d"}
CONTEXT = (
    "Existing campaign overlays. The fixed controls and acceptance rules differ between models.",
    "All curves use Rb-87 at 296.483 K and 10⁻¹⁰ Torr partial pressure, gravity −z, deterministic mean force.",
    "Coil MOT scans hold 10 mW/cooling beam, −15 MHz, 12.7 mm diameter and axial 20 G/cm except the swept control. "
    "Multilevel MOT includes 0.1 mW repump/beam. Their coil gradient ratio is (+G/2,+G/2,−G).",
    "Strange MOT uses 27 mW/cooling beam, −15 MHz, 12.7 mm diameter. Its detuning/diameter scans fix "
    "(+2.6113,+2.6113,−5.2226) G/cm. Repump is 0.1 mW/beam, 12.7 mm.",
    "780 nm saturation uses I_sat=16.69 W/m² and linewidth 6.065 MHz. At the fixed −15 MHz, 12.7 mm controls, "
    "single-beam (s₀,s_eff) is (9.45968,0.371448) for coil MOTs and (25.5411,1.00291) for strange MOT.",
    "Coil MOTs use a 15 mm-radius disc at 15 mm and a 30 mm escape sphere. Core event is 5 ms inside 2 mm "
    "or two entries, with their saved convergence audits. Strange MOT uses a 15 mm disc at 10 mm clipped "
    "to a ±14.142 mm cube and requires final 100 ms core residence at 2 s.",
    "Strange trapping light is 1529.268881 nm, equal returns, waists ±10 mm, 35 mm input and 80.3 mm focal length. "
    "It supplies a surrogate field to the 24-state MOT kernel.",
    "Bands are pointwise 95% Student-t direction-disc intervals. Curves join simulated settings. "
    "The log panel resolves the smaller rates. These overlays do not isolate a single physical change.",
)
PMOT_CONTEXT = (
    "2-level pMOT holds 27 mW/cooling beam, −17.5 MHz, 12.7 mm diameter, no repumper or real field. "
    "It uses 22 impact-stratified points/disc × 12 speed strata, one draw each, final 100 ms at 2 s in the core "
    "with speed ≤0.05 m/s. It includes differential Stark detuning and omits conservative Stark forces, "
    "trap-light scattering and recoil.",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merged_data():
    document = read(OUTPUT / "manifest.json")
    old_bytes = (extension.ORIGINAL / "final_loading_results.json").read_bytes()
    if extension.checksum(extension.ORIGINAL / "final_loading_results.json") != document["original_results_sha256"]:
        raise ValueError("Original data changed")
    old = json.loads(old_bytes)
    manifest = deepcopy(read(extension.ORIGINAL / "pilot_manifest.json"))
    added = read(OUTPUT / "production_results.json")
    offset = len(manifest["points"])
    if offset != document["configuration_offset"] or set(added) != set(map(str, range(offset, offset+20))):
        raise ValueError("All twenty new settings must be audited")
    results = {**old, **added}
    for i, point in enumerate(document["points"], offset):
        row = added[str(i)]
        if (row["status"] != "passed" or row["manifest_sha256"] != extension.checksum(OUTPUT / "manifest.json")
                or row["discs"] != document["discs"] or row["points_per_disc"] != document["points_per_disc"]):
            raise ValueError("Incompatible extension result")
        manifest["points"].append(point)
        key = "trap_power_isotropic" if point["geometry"] == "isotropic" else "trap_power_axial"
        manifest["scans"][key].append(i)
    if any(results[k] != value for k, value in old.items()):
        raise ValueError("Merge changed an original result")
    folder = OUTPUT / "merged"
    folder.mkdir(exist_ok=True)
    extension.write_json(folder / "pilot_manifest.json", manifest)
    extension.write_json(folder / "final_loading_results.json", results)
    extension.write_json(folder / "final_audit.json", {
        "status": "passed", "result_sha256": extension.checksum(folder / "final_loading_results.json"),
        "original_result_sha256": document["original_results_sha256"],
        "extension_result_sha256": extension.checksum(OUTPUT / "production_results.json"),
        "old_records_unchanged": len(old), "new_records": len(added),
    })
    return manifest, results


def campaign_rows(path, model, family):
    manifest = read(path / "campaign_manifest.json")
    plan = read(path / "production_plan.json")
    with (path / "aggregate.csv").open(newline="", encoding="utf-8") as stream:
        aggregate = {int(row["index"]): row for row in csv.DictReader(stream)}
    if model == "2-level MOT":
        indices = manifest["scan_configuration_indices"][family]
    else:
        indices = manifest["scans"]["cooling_"+family]
    fields = {"gradient": "gradient_g_per_cm", "detuning": "detuning_mhz", "diameter": "diameter_mm"}
    rows = []
    for index in indices:
        source = aggregate[index]
        if model == "2-level MOT":
            controls = {k: float(source[k]) for k in ("power_mw_per_beam", "detuning_mhz", "diameter_mm",
                                                     "gradient_g_per_cm", "s0", "seff")}
            points = plan["searches"][str(index)]["points_per_disc"]
        else:
            controls = {
                "power_mw_per_beam": float(source["cooling_power_mw"]),
                "detuning_mhz": float(source["cooling_detuning_mhz"]),
                "diameter_mm": float(source["cooling_diameter_mm"]),
                "gradient_g_per_cm": float(source["gradient_g_per_cm"]),
                "s0": float(source["cooling_s0"]), "seff": float(source["cooling_seff"]),
            }
            points = plan["allocations"][str(index)]["points_per_disc"]
        rows.append({
            "model": model, "x": controls[fields[family]], "source_index": index, **controls,
            **{key: float(source[key]) for key in ("mean", "lower", "upper")},
            "discs": int(source["disc_count"]), "points": points,
        })
    return rows


def strange_rows(family, geometry, manifest, results):
    scan = {"gradient": "trap_power_isotropic" if geometry == "isotropic" else "trap_power_axial",
            "detuning": "cooling_detuning", "diameter": "cooling_diameter"}[family]
    rows = []
    for i in manifest["scans"][scan]:
        control, result = manifest["points"][i], results[str(i)]
        gradient = abs(control["origin_surrogate_field_jacobian_g_per_cm"][2][2])
        x = {"gradient": gradient, "detuning": control["cooling_detuning_mhz"],
             "diameter": control["cooling_diameter_mm"]}[family]
        q = result["core"]
        rows.append({
            "model": "Strange MOT", "x": x, "source_index": i,
            "mean": q["mean_atoms_per_s"], "lower": q["lower_atoms_per_s"], "upper": q["upper_atoms_per_s"],
            "discs": result["discs"], "points": result["points_per_disc"],
            "power_mw_per_beam": 27., "detuning_mhz": control["cooling_detuning_mhz"],
            "diameter_mm": control["cooling_diameter_mm"], "gradient_g_per_cm": gradient,
            "s0": control["cooling_s0"], "seff": control["cooling_s_eff"],
            "trapping_power_mw_xyz": control["trapping_power_mw_xyz"],
        })
    return rows


def pmot_rows(spec):
    controls, design = read(PMOT / "controls.json"), read(PMOT / "production_summary.json")
    audit = read(PMOT / "production_replay_audit.json")
    if not audit["passed"] or audit["manifest_sha256"] != design["manifest_sha256"]:
        raise ValueError("pMOT audit does not match the saved rates")
    rows = []
    geometry = "equal_power" if spec["geometry"] == "isotropic" else "doubled_axial"
    for control, result in zip(controls["settings"], design["settings"], strict=True):
        if (control["ranks"], control["wavelength_nm"], control["geometry"]) != (
                spec["ranks"], spec["wavelength_nm"], geometry):
            continue
        q = result["statistics"]["core"]
        g = abs(control["signed_gradient_g_cm_xyz"][2])
        half = q["half_width"]
        rows.append({
            "model": "2-level pMOT", "x": g, "source_index": result["setting_id"],
            "mean": q["mean"], "lower": q["mean"]-half if half is not None else None,
            "upper": q["mean"]+half if half is not None else None,
            "discs": result["discs"], "points": design["points_per_disc"],
            "power_mw_per_beam": 27., "detuning_mhz": -17.5, "diameter_mm": 12.7,
            "gradient_g_per_cm": g, "s0": controls["cooling_s0_seff"][0],
            "seff": controls["cooling_s0_seff"][1], "trapping_power_mw_xyz": [1000*p for p in control["power_w_xyz"]],
        })
    return rows


def series(spec):
    manifest = read(OUTPUT / "merged/pilot_manifest.json")
    results = read(OUTPUT / "merged/final_loading_results.json")
    rows = campaign_rows(SIMPLE, "2-level MOT", spec["family"])
    rows += campaign_rows(MULTI, "Multilevel MOT", spec["family"])
    rows += strange_rows(spec["family"], spec["geometry"], manifest, results)
    if "ranks" in spec:
        rows += pmot_rows(spec)
    return rows


def figure(spec, rows):
    family = spec["family"]
    geometry = "Equal-power strange/pMOT (+G,+G,−G)" if spec["geometry"] == "isotropic" else "Doubled axial (+G/2,+G/2,−G)"
    title = f"{'Four' if 'ranks' in spec else 'Three'}-model loading comparison · {family}"
    if family == "gradient":
        title += "\n" + geometry
    if "ranks" in spec:
        ranks = {"svt": "Scalar + vector + tensor", "sv": "Scalar + vector", "v": "Vector only"}
        title += f"\npMOT {ranks[spec['ranks']]} · {spec['wavelength_nm']:.12f} nm"
    fig, axes = plt.subplots(1, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.08, right=.97, top=.80, bottom=.51, wspace=.25)
    fig.suptitle(title, y=.98, fontsize=15)
    label = {"gradient": "Axial gradient magnitude |∂Bz/∂z| (G/cm)",
             "detuning": "Cooling detuning (MHz)", "diameter": "Cooling 1/e² diameter (mm)"}[family]
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    for name, group in groups.items():
        group.sort(key=lambda row: row["x"])
        x = np.array([r["x"] for r in group])
        mean = np.array([r["mean"] for r in group])
        low = np.array([max(0., r["lower"]) if r["lower"] is not None else np.nan for r in group])
        high = np.array([r["upper"] if r["upper"] is not None else np.nan for r in group])
        for axis in axes:
            axis.plot(x, np.where(mean > 0, mean, np.nan) if axis is axes[1] else mean,
                      "o-", color=COLORS[name], markersize=3, label=name, linewidth=1.4)
            axis.fill_between(x, np.maximum(low, 1e-5) if axis is axes[1] else low,
                              high, color=COLORS[name], alpha=.16)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel(label)
        axis.set_ylabel("Core loading rate (atoms/s)")
        axis.grid(alpha=.2, which="both")
        axis.legend(fontsize=9)
        if family == "gradient":
            axis.set_xlim(0, 40)
    context = list(CONTEXT)
    if "ranks" in spec:
        context.extend(PMOT_CONTEXT)
    for name, group in groups.items():
        samples = " ".join(f"{r['x']:.3g}/{r['discs']}×{r['points']}" for r in group)
        context.append(f"{name} samples, x / independent discs × points per disc. {samples}")
    lines = [line for paragraph in context for line in wrap(paragraph, 159)]
    fig.text(.08, .45, "\n".join(lines), va="top", fontsize=8.1, linespacing=1.25)
    return fig


def figure_png(spec, rows=None):
    fig = figure(spec, series(spec) if rows is None else rows)
    buffer = BytesIO()
    try:
        fig.savefig(buffer, dpi=145)
        return buffer.getvalue()
    finally:
        plt.close(fig)


def table(rows):
    keys = ("model", "source_index", "x", "mean", "lower", "upper", "discs", "points",
            "power_mw_per_beam", "detuning_mhz", "diameter_mm", "gradient_g_per_cm", "s0", "seff",
            "trapping_power_mw_xyz")
    def value(row, key):
        v = row.get(key)
        return f"{v:.8g}" if isinstance(v, float) else str(v) if v is not None else "undefined"
    head = "".join(f"<th>{escape(k)}</th>" for k in keys)
    body = "".join("<tr>"+"".join(f"<td>{escape(value(row,k))}</td>" for k in keys)+"</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_notebook(name, specs):
    title = "Three-model MOT loading overlays" if name.startswith("three") else "Four-model MOT and pMOT loading overlays"
    cells = [nbformat.v4.new_markdown_cell("# "+title+"\n\n"+"\n\n".join(CONTEXT))]
    cells.append(nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport sys\nfrom IPython.display import Image, HTML, display\n"
        "repo = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'pyproject.toml').is_file())\n"
        "sys.path.insert(0, str(repo/'scripts'))\n"
        "from report_extended_mot_loading import figure_png, series, table\n"))
    for spec in specs:
        rows = series(spec)
        png = figure_png(spec, rows)
        slug = "_".join(str(v) for v in spec.values()).replace(".", "p")
        (FIGURES / f"{name}_{slug}.png").write_bytes(png)
        cell = nbformat.v4.new_code_cell(
            f"spec = {spec!r}\nrows = series(spec)\n"
            "display(Image(data=figure_png(spec, rows)))\ndisplay(HTML(table(rows)))")
        cell.outputs = [
            nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode()}),
            nbformat.v4.new_output("display_data", data={"text/html": table(rows)}),
        ]
        cells.append(cell)
    book = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.validate(book)
    nbformat.write(book, NOTEBOOKS / f"{name}_loading_20261111.ipynb")


def main():
    import report_mot_strange_loading_scans as report

    manifest, results = merged_data()
    if not (OUTPUT / "retention_diagnosis.json").is_file():
        raise ValueError("Retention diagnosis is required")
    backup = OUTPUT / "prior_reports"
    backup.mkdir(exist_ok=True)
    for path in (REPO / f"notebooks/mot_strange/{extension.ORIGINAL.name}.ipynb",
                 REPO / f"visualizations/mot_loading/{extension.ORIGINAL.name}.html"):
        if not (backup / path.name).exists():
            copy2(path, backup / path.name)
    report.ROOT = OUTPUT / "merged"
    report.REPORT_CONTEXT = (*report.REPORT_CONTEXT,
        "The first simulated gradient is 0.5223 G/cm. Representative captures there converge to a stable "
        "gravity-on core equilibrium about 52 µm below the origin, remain through 4 s and return after "
        "position and velocity perturbations. The matched launch without the surrogate field stays outside "
        "the core. The two extended scans retain the same production core definition.")
    report.main()
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    diagnosis = read(OUTPUT / "retention_diagnosis.json")
    bookpath = REPO / f"notebooks/mot_strange/{extension.ORIGINAL.name}.ipynb"
    book = nbformat.read(bookpath, as_version=4)
    for cell in book.cells:
        if cell.cell_type == "code" and "pilot_manifest.json" in cell.source:
            cell.source = cell.source.replace(
                f"data = root / 'outputs/statistics/mot_strange/{extension.ORIGINAL.name}'",
                f"data = root / {str((OUTPUT/'merged').relative_to(REPO).as_posix())!r}")
            cell.source = cell.source.replace(
                "from IPython.display import display",
                "from IPython.display import display, Image\nfrom io import BytesIO\nimport matplotlib.pyplot as plt")
            cell.source = cell.source.replace(
                "    display(figure)",
                "    png = BytesIO()\n    figure.savefig(png, dpi=150)\n"
                "    display(Image(data=png.getvalue()))\n    plt.close(figure)")
    book.cells.insert(1, nbformat.v4.new_markdown_cell(
        "## Extended gradients and retention check\n\n"
        "The original 69 audited records are retained. Twenty new settings extend both trapping scans "
        "with axial magnitudes 22, 24, …, 40 G/cm. The first simulated gradient is 0.5223 G/cm.\n\n"
        "The original core rule is unchanged at every gradient. Representative lowest-gradient captures "
        "return to a stable gravity-on equilibrium about 52 µm below the origin, remain there through 4 s, "
        "and return after perturbations. The same incident launch with zero surrogate field stays outside "
        "the core. These diagnostics distinguish that example from molasses localization. "
        "The saved speed decomposition attributes about 91% of the initial rate dip to 18–35 m/s launches.\n\n"
        "The original MOT benchmarks use an early core event with audited duration checks. "
        "The strange-MOT rate additionally requires final 100 ms residence at 2 s. "
        "The comparison notebooks record that acceptance difference."))
    book.cells.append(nbformat.v4.new_code_cell(
        "from report_extended_mot_loading import read, OUTPUT\n"
        "diagnosis = read(OUTPUT/'retention_diagnosis.json')\n"
        "for name, result in diagnosis['results'].items():\n"
        "    print(name, result['core_equilibrium_mm'], result['spectral_abscissae_s_inv'])"))
    nbformat.write(book, bookpath)
    for scan in ("trap_power_isotropic", "trap_power_axial"):
        rows = [dict(results[str(i)], point=manifest["points"][i]) for i in manifest["scans"][scan]]
        fig, _ = report.draw(scan, rows, manifest)
        fig.savefig(FIGURES / f"strange_{scan}_through40G.png", dpi=150)
        plt.close(fig)
    three = [{"family": f, "geometry": g} for f, g in (
        ("gradient", "isotropic"), ("gradient", "axial"), ("detuning", "axial"), ("diameter", "axial"))]
    four = [{"family": "gradient", "geometry": g, "ranks": r, "wavelength_nm": w}
            for g in ("isotropic", "axial") for r in ("svt", "sv", "v")
            for w in (1529.268881048958, 1529.273195181438)]
    build_notebook("three_model", three)
    build_notebook("four_model", four)
    extension.write_json(OUTPUT / "report_build.json", {
        "original_records_retained": 69, "added_settings": 20, "three_model_panels": len(three),
        "four_model_panels": len(four), "retention_diagnosis_status": diagnosis["status"],
        "sources": {str(path.relative_to(REPO)): extension.checksum(path) for path in (
            SIMPLE / "aggregate.csv", MULTI / "aggregate.csv", PMOT / "production_summary.json",
            OUTPUT / "production_results.json", Path(__file__).resolve())},
    })
    print("Updated strange-MOT report and built three-model and four-model comparison notebooks", flush=True)


if __name__ == "__main__":
    main()
