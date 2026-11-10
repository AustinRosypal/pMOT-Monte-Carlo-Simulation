"""Render independent effective-two-level pMOT loading results as notebook and HTML."""

import argparse
import base64
from collections import defaultdict
from html import escape
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pmot.pMOT_2Level.loading_scan import GEOMETRIES, RANKS, WAVELENGTHS_NM
from validate_pmot_2level_loading import OUTPUT, ROOT, write_json


TITLE = 'Effective two-level pMOT loading'
RUN_NAME = OUTPUT.name
RANK_NAMES = {'svt': 'Scalar + vector + tensor', 'sv': 'Scalar + vector', 'v': 'Vector only'}
WAVELENGTH_NAMES = ('Single-beam scalar–tensor cancellation', 'Differential scalar zero')
GEOMETRY_NAMES = {'equal_power': 'Equal power (+G,+G,−G)',
                  'doubled_axial': 'Doubled axial (+G/2,+G/2,−G)'}
CAPTION = (
    'Rb-87 effective F=2 → F′=3 mean force. No repumper or real magnetic field. Gravity −z. '
    'Six 780.241209686 nm beams, 27 mW each, 12.7 mm 1/e² diameter, −17.5 MHz; '
    'matched propagation-frame helicities (σ+, σ+, σ−) on both directions of the 780 and 1529 nm paths. '
    'Ideal equal returns, symmetric ±10 mm waist positions, 35 mm input diameter, 80.3 mm focal length. '
    '15 mm launch disc at 10 mm, clipped to ±14.142 mm cube; vapor 296.483 K and 10⁻¹⁰ Torr Rb-87. '
    'Bands show pointwise 95% Student-t intervals across independent direction discs. '
    'The model includes differential Stark detuning; it omits conservative Stark force, 1529 nm scattering and recoil.'
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def figure(group, rows, design, controls):
    ranks, wavelength, geometry = group
    intensity = np.array([row['center_intensity_w_m2_six_components'][4] for row in rows])
    g = np.array([row['signed_gradient_g_cm_xyz'][2] for row in rows])
    slope = g[-1] / intensity[-1]
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.subplots_adjust(left=.12, right=.96, top=.77, bottom=.34)
    names = {'core': 'Retained core', 'off_core': 'Verified restoring off-core'}
    colors = {'core': '#155e9a', 'off_core': '#b85222'}
    for key in names:
        points = [row['statistics'][key] for row in rows]
        mean = np.array([p['mean'] for p in points])
        half = np.array([p['half_width'] if p['half_width'] is not None else np.nan for p in points])
        if key == 'off_core' and not np.any(mean):
            ax.text(.02, .96, 'No verified off-core events observed', transform=ax.transAxes,
                    va='top', fontsize=9, color=colors[key])
            continue
        ax.plot(intensity, mean, 'o-', markersize=3, color=colors[key], label=names[key])
        ax.fill_between(intensity, np.maximum(0, mean-half), mean+half, color=colors[key], alpha=.2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(intensity.min(), intensity.max())
    ax.set_xlabel('1529 nm axial intensity at origin, one traveling component (W/m²)')
    ax.set_ylabel('Loading rate (atoms/s)')
    ax.grid(alpha=.18)
    ax.legend(loc='best', fontsize=9)
    top = ax.secondary_xaxis('top', functions=(lambda i: i*slope, lambda b: b/slope))
    top.set_xlabel('Signed local axial vector Stark gradient (G/cm); Bₑff = Q/μB')
    top.set_xticks(np.r_[g[::3], g[-1]])
    idx = WAVELENGTHS_NM.index(wavelength)
    fig.suptitle(f'{TITLE} | {RANK_NAMES[ranks]}\n'
                 f'{wavelength:.12f} nm · {WAVELENGTH_NAMES[idx]}\n'
                 f'{GEOMETRY_NAMES[geometry]}', y=.977, fontsize=14)
    s0, seff = controls['cooling_s0_seff']
    edges = controls.get('impact_edges_mm')
    aperture_note = ('Impact annuli (mm): ' + ', '.join(f'{edge:g}' for edge in edges) + '; area weighted.\n') if edges else ''
    controls_text = (
        '2–40 G/cm by 2 (axial magnitude); helicities (σ+, σ+, σ−) in both directions on cooling and trap paths.\n'
        '780.241209686 nm cooling: 27 mW per beam, 12.7 mm Gaussian diameter, −17.5 MHz; '
        f'center single-beam s₀={s0:.4f}, nominal sₑff={seff:.4f}.\n'
        'No applied magnetic field or repumper. Gravity −z. Trap waists ±10 mm, 35 mm input, 80.3 mm lens, equal returns.\n'
        'Vapor: 296.483 K, 10⁻¹⁰ Torr Rb-87. 15 mm discs at 10 mm clipped to ±14.142 mm cube. '
        f'{design["points_per_disc"]} points/disc × {len(design["speed_edges_m_s"])-1} strata × '
        f'{design["draws_per_stratum"]} speed draw(s).\n'
        'Bands: pointwise 95% Student-t, by direction disc. No relative CI for zero observed events. '
        'Finite 2 s retention; unverified/unresolved in the table.\n'
        + aperture_note +
        'Each entry below: |Gz| (G/cm) / Pz (mW per component) / independent direction discs. '
        + ('Px=Py=Pz.' if geometry == 'equal_power' else 'Px=Py=Pz/2.')
    )
    fig.text(.12, .245, controls_text, va='top', fontsize=8.6, linespacing=1.35)
    entries = [f'{abs(z):.0f} / {r["power_w_xyz"][2]*1000:.3f} / {r["discs"]}' for z, r in zip(g, rows, strict=True)]
    fig.text(.12, .095, '\n'.join('    '.join(f'{e:>15}' for e in entries[i:i+5]) for i in range(0, 20, 5)),
             family='monospace', fontsize=8.4, va='top')
    return fig


def figure_png(group, rows, design, controls):
    fig = figure(group, rows, design, controls)
    buffer = BytesIO()
    try:
        fig.savefig(buffer, dpi=155, facecolor='white')
        return buffer.getvalue()
    finally:
        plt.close(fig)


def table_html(rows):
    def value(row, key):
        stat = row['statistics'][key]
        if stat['half_width'] is None:
            return '0 observed'
        return f'{stat["mean"]:,.3g} ± {stat["half_width"]:,.3g} ({100*stat["relative_half_width"]:.1f}%)'

    headers = ('Signed gradients x,y,z (G/cm)', 'P x,y,z (mW each)', 'Axial center I (W/m²)',
               'Discs', 'Core (atoms/s)', 'Verified off-core (atoms/s)',
               'Localized unverified (atoms/s)', 'Unresolved (atoms/s)')
    body = []
    for row in rows:
        cells = (', '.join(f'{b:+g}' for b in row['signed_gradient_g_cm_xyz']),
                 ', '.join(f'{1000*p:.5f}' for p in row['power_w_xyz']),
                 f'{row["center_intensity_w_m2_six_components"][4]:.5f}', str(row['discs']),
                 *(value(row, k) for k in ('core', 'off_core', 'localized_unverified', 'unresolved')))
        body.append('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in cells) + '</tr>')
    return ('<details><summary>Per-point controls, rates, and intervals</summary><table><thead><tr>'
            + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join(body) + '</tbody></table></details>')


def comparison_html(rows):
    lines = []
    for row in rows:
        values = [RANK_NAMES[row['ranks']], GEOMETRY_NAMES[row['geometry']],
                  str(row['axial_magnitude_g_cm']), str(row['paired_discs'])]
        for key in ('core', 'off_core'):
            delta, width = row[key]['delta_atoms_s'], row[key]['half_width']
            values.append(f'{delta:+,.3g} ± {width:,.3g}' if width is not None else 'No observed difference')
        lines.append('<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in values) + '</tr>')
    return ('<details><summary>Matched-gradient wavelength comparisons</summary>'
            '<p>Scalar-zero minus cancellation (atoms/s), with pointwise 95% intervals on paired discs. '
            'A positive difference favors scalar zero; use the interval to judge how well it is resolved.</p>'
            '<table><thead><tr><th>Stark terms</th><th>Geometry</th><th>|Gz| (G/cm)</th><th>Paired discs</th>'
            '<th>Core difference</th><th>Verified off-core difference</th></tr></thead><tbody>'
            + ''.join(lines) + '</tbody></table></details>')


def report():
    design = read(OUTPUT / 'production_summary.json')
    audit = read(OUTPUT / 'production_replay_audit.json')
    if design['phase'] != 'production' or audit['manifest_sha256'] != design['manifest_sha256'] or not audit['passed']:
        raise ValueError('A passing replay audit of independent production is required')
    paired = read(OUTPUT / 'production_paired_wavelengths.json')
    if paired['manifest_sha256'] != design['manifest_sha256'] or len(paired['rows']) != 120:
        raise ValueError('Matched-wavelength comparison evidence differs from the audit')
    comparison = comparison_html(paired['rows'])
    controls = read(OUTPUT / 'controls.json')
    if len(design['settings']) != 240 or len(controls['settings']) != 240:
        raise ValueError('Report requires all 240 settings')
    groups = defaultdict(list)
    for index, (control, result) in enumerate(zip(controls['settings'], design['settings'], strict=True)):
        if result['setting_id'] != index or result['discs'] != design['counts'][index]:
            raise ValueError('Summary and frozen design differ')
        groups[(control['ranks'], control['wavelength_nm'], control['geometry'])].append({**control, **result})
    fig_root = ROOT / 'outputs/figures/pMOT_2Level' / RUN_NAME
    nb_root = ROOT / 'notebooks/pMOT_2Level'
    html_root = ROOT / 'visualizations/pMOT_2Level'
    for d in (fig_root, nb_root, html_root):
        d.mkdir(parents=True, exist_ok=True)
    panels, nb_cells = [], []
    precision = {key: [p['statistics'][key]['relative_half_width'] for p in design['settings']
                       if p['statistics'][key]['relative_half_width'] is not None]
                 for key in ('core', 'off_core')}
    counts = {key: sum(width < .1 for width in values) for key, values in precision.items()}
    overview = (f'{len(design["settings"])} settings and {sum(design["counts"]):,} direction-disc batches. '
                'Each setting uses independent directions, with common draws across settings for paired comparisons. '
                f'Pointwise relative half-width <10%: {counts["core"]}/240 core and '
                f'{counts["off_core"]}/240 verified off-core. Zero observed events do not establish zero rate. '
                f'Replay audited {len(audit["replays"]):,} nodes. Seed {design["seed"]}; pilot excluded.')
    if precision['core']:
        overview += (f' Median core relative half-width is {100*np.median(precision["core"]):.1f}% '
                     f'across {len(precision["core"])} settings with observed core retention.')
    overview += (' The bands describe sampling uncertainty for the fixed 2 s retention rule. '
                 'Unresolved and unverified localized flux appears separately in each table.')
    if controls.get('impact_edges_mm'):
        edges = ', '.join(f'{edge:g}' for edge in controls['impact_edges_mm'])
        overview += f' Launch points are stratified in impact-parameter annuli with edges {edges} mm and weighted by clipped area.'
    nb_cells.append({'cell_type': 'markdown', 'metadata': {},
                     'source': f'# {TITLE}\n\n{CAPTION}\n\n{overview}'})
    setup = (
        'from pathlib import Path\nfrom IPython.display import HTML, Image, display\nimport sys\n'
        'repo = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())\n'
        'sys.path.insert(0, str(repo / "scripts"))\n'
        'from report_pmot_2level_loading import figure_png, table_html, comparison_html, read\n'
        f'campaign = repo / {OUTPUT.relative_to(ROOT).as_posix()!r}\n'
        'design = read(campaign / "production_summary.json")\n'
        'controls = read(campaign / "controls.json")\n'
        'rows = [{**c, **r} for c, r in zip(controls["settings"], design["settings"], strict=True)]\n'
    )
    nb_cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': setup})
    nb_cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': 2,
                     'source': 'display(HTML(comparison_html(read(campaign / "production_paired_wavelengths.json")["rows"])))\n',
                     'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {'text/html': comparison}}]})
    for ranks in RANKS:
        for wave in WAVELENGTHS_NM:
            for geometry in GEOMETRIES:
                group = (ranks, wave, geometry)
                rows = groups[group]
                if len(rows) != 20:
                    raise ValueError(f'Expected 20 samples in {group}')
                png = figure_png(group, rows, design, controls)
                name = f'{ranks}_{WAVELENGTHS_NM.index(wave)+1}_{geometry}'
                (fig_root / f'{name}.png').write_bytes(png)
                img = base64.b64encode(png).decode('ascii')
                table = table_html(rows)
                panels.append(f'<section><img alt="{escape(name)}" src="data:image/png;base64,{img}">{table}</section>')
                source = (f'group = {group!r}\n'
                          'selected = [r for r in rows if (r["ranks"], r["wavelength_nm"], r["geometry"]) == group]\n'
                          'display(Image(data=figure_png(group, selected, design, controls)))\n'
                          'display(HTML(table_html(selected)))\n')
                nb_cells.append({'cell_type': 'code', 'metadata': {}, 'source': source,
                                 'execution_count': len(nb_cells), 'outputs': [
                    {'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': img}},
                    {'output_type': 'display_data', 'metadata': {}, 'data': {'text/html': table}},
                ]})
    for i, cell in enumerate(nb_cells):
        cell['id'] = f'loading-{i:02}'
    write_json(nb_root / f'{RUN_NAME}.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': nb_cells})
    style = ('body{font-family:system-ui;margin:2rem auto;max-width:1150px;padding:0 1rem;color:#18202b}'
             'section{padding:1rem 0;border-top:1px solid #dbe2eb}img{width:100%;height:auto}'
             'td,th{text-align:right;padding:.45rem;border-bottom:1px solid #dbe2eb}'
             'table{border-collapse:collapse;font-size:.85rem}summary{cursor:pointer}')
    (html_root / f'{RUN_NAME}.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'
        f'<style>{style}</style><main><h1>{TITLE}</h1><p>{escape(CAPTION)}</p>'
        f'<p>{escape(overview)}</p>{comparison}{"".join(panels)}</main></html>', encoding='utf-8')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()
