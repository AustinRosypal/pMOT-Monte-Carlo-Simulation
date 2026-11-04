"""Create provenance-preserving, normalization-corrected result files.

The historical products are left untouched.  Corrected JSON and CSV files are
written under a separate output root with the same relative paths.  Absolute
loading rates and their dimensional uncertainties are multiplied by the ratio
of the corrected Maxwell/vapor prefactor to the historical 9.1196e5 value.
Capture spectra, trajectories, classifications, and dimensionless relative
uncertainties are not changed.

This is deliberately a scaling migration.  It retains each historical
velocity integral, whose 5.667e4 m2/s2 thermal scale differs negligibly from
the corrected 5.6728111e4 m2/s2 scale over the saved capture-speed ranges.
New calculations use the corrected scale directly through ``pmot.loading``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmot.loading import DEFAULT_VAPOR_LOADING_PARAMETERS


HISTORICAL_LOADING_RATE_PREFACTOR = 9.1196e5
MIGRATION_NAME = "loading_normalization_corrected_20261103"
RATE_FIELD = re.compile(r"loading.*atoms_per_s$", re.IGNORECASE)
FIGURE_SUFFIXES = {".png", ".svg", ".pdf"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _correct_formula_text(value: str, new_prefactor: float) -> str:
    corrected = value.replace("9.1196e5", f"{new_prefactor:.12g}")
    corrected = corrected.replace("9.1196E5", f"{new_prefactor:.12g}")
    return corrected


def _correct_json_value(
    value: Any,
    *,
    scale: float,
    new_prefactor: float,
    counters: dict[str, int],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            if RATE_FIELD.search(key) and _is_number(item):
                output[key] = float(item) * scale
                counters["scaled_numeric_fields"] += 1
                changed = True
            elif key == "loading_rate_prefactor" and _is_number(item):
                output[key] = new_prefactor
                counters["replaced_prefactors"] += 1
                changed = True
            else:
                corrected_item, item_changed = _correct_json_value(
                    item,
                    scale=scale,
                    new_prefactor=new_prefactor,
                    counters=counters,
                )
                output[key] = corrected_item
                changed = changed or item_changed
        return output, changed
    if isinstance(value, list):
        output_list: list[Any] = []
        changed = False
        for item in value:
            corrected_item, item_changed = _correct_json_value(
                item,
                scale=scale,
                new_prefactor=new_prefactor,
                counters=counters,
            )
            output_list.append(corrected_item)
            changed = changed or item_changed
        return output_list, changed
    if isinstance(value, str):
        corrected = _correct_formula_text(value, new_prefactor)
        if corrected != value:
            counters["corrected_formula_strings"] += 1
            return corrected, True
    return value, False


def _correct_json(
    source: Path,
    destination: Path,
    *,
    scale: float,
    new_prefactor: float,
    relative_source: Path,
    apply: bool,
) -> dict[str, Any] | None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    counters = {
        "scaled_numeric_fields": 0,
        "replaced_prefactors": 0,
        "corrected_formula_strings": 0,
    }
    corrected, changed = _correct_json_value(
        payload,
        scale=scale,
        new_prefactor=new_prefactor,
        counters=counters,
    )
    if not changed:
        return None

    if isinstance(corrected, dict):
        corrected["loading_normalization_correction"] = {
            "migration": MIGRATION_NAME,
            "source_relative_path": relative_source.as_posix(),
            "historical_loading_rate_prefactor": HISTORICAL_LOADING_RATE_PREFACTOR,
            "corrected_loading_rate_prefactor": new_prefactor,
            "multiplicative_scale": scale,
            "method": (
                "Scaled absolute loading-rate and dimensional uncertainty fields; "
                "retained the saved historical velocity integral"
            ),
            "capture_and_trajectory_data_changed": False,
        }
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(corrected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {"kind": "json", **counters}


def _correct_csv(
    source: Path,
    destination: Path,
    *,
    scale: float,
    new_prefactor: float,
    apply: bool,
) -> dict[str, Any] | None:
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        fieldnames = list(reader.fieldnames)
        rate_fields = [name for name in fieldnames if RATE_FIELD.search(name)]
        prefactor_fields = [name for name in fieldnames if name == "loading_rate_prefactor"]
        if not rate_fields and not prefactor_fields:
            return None
        rows = list(reader)

    scaled_values = 0
    replaced_prefactors = 0
    for row in rows:
        for name in rate_fields:
            raw = row.get(name, "")
            if raw not in (None, ""):
                row[name] = repr(float(raw) * scale)
                scaled_values += 1
        for name in prefactor_fields:
            raw = row.get(name, "")
            if raw not in (None, ""):
                row[name] = repr(new_prefactor)
                replaced_prefactors += 1

    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "kind": "csv",
        "scaled_numeric_fields": scaled_values,
        "replaced_prefactors": replaced_prefactors,
        "corrected_formula_strings": 0,
    }


def migrate(source_root: Path, destination_root: Path, *, apply: bool) -> dict[str, Any]:
    vapor = DEFAULT_VAPOR_LOADING_PARAMETERS
    new_prefactor = vapor.loading_rate_prefactor
    scale = new_prefactor / HISTORICAL_LOADING_RATE_PREFACTOR
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if apply and destination_root.exists():
        raise FileExistsError(
            f"destination already exists; preserve it or choose another path: {destination_root}"
        )

    records: list[dict[str, Any]] = []
    skipped_errors: list[dict[str, str]] = []
    stale_figures: list[str] = []
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        try:
            source.relative_to(destination_root)
        except ValueError:
            pass
        else:
            continue
        relative = source.relative_to(source_root)
        if "loading" in source.name.lower() and source.suffix.lower() in FIGURE_SUFFIXES:
            stale_figures.append(relative.as_posix())
        if source.suffix.lower() not in {".json", ".csv"}:
            continue
        destination = destination_root / relative
        try:
            if source.suffix.lower() == ".json":
                result = _correct_json(
                    source,
                    destination,
                    scale=scale,
                    new_prefactor=new_prefactor,
                    relative_source=relative,
                    apply=apply,
                )
            else:
                result = _correct_csv(
                    source,
                    destination,
                    scale=scale,
                    new_prefactor=new_prefactor,
                    apply=apply,
                )
        except (ValueError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            skipped_errors.append({"path": relative.as_posix(), "error": str(exc)})
            continue
        if result is None:
            continue
        record = {
            "source": relative.as_posix(),
            "corrected": relative.as_posix(),
            "source_sha256": _sha256(source),
            **result,
        }
        if apply:
            record["corrected_sha256"] = _sha256(destination)
        records.append(record)

    manifest = {
        "migration": MIGRATION_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied": apply,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "historical_loading_rate_prefactor": HISTORICAL_LOADING_RATE_PREFACTOR,
        "corrected_loading_rate_prefactor": new_prefactor,
        "multiplicative_scale": scale,
        "vapor_parameters": {
            "temperature_k": vapor.temperature_k,
            "rb87_partial_pressure_torr": vapor.rb87_partial_pressure_torr,
            "rb87_partial_pressure_pa": vapor.rb87_partial_pressure_pa,
            "rb87_number_density_m3": vapor.number_density_m3,
            "rb87_mass_kg": vapor.rb87_mass_kg,
            "speed_distribution_prefactor_s3_per_m3": (
                vapor.speed_distribution_prefactor_s3_per_m3
            ),
            "thermal_scale_m2_per_s2": vapor.thermal_scale_m2_per_s2,
        },
        "method": (
            "Scale absolute loading-rate and dimensional uncertainty fields in "
            "JSON/CSV products. Preserve original products and all capture, "
            "trajectory, and dimensionless uncertainty data."
        ),
        "historical_integral_note": (
            "Corrected copies retain saved historical velocity integrals. New "
            "calculations use the exact corrected thermal scale."
        ),
        "corrected_file_count": len(records),
        "scaled_numeric_field_count": sum(
            int(record["scaled_numeric_fields"]) for record in records
        ),
        "replaced_prefactor_count": sum(
            int(record["replaced_prefactors"]) for record in records
        ),
        "corrected_formula_string_count": sum(
            int(record["corrected_formula_strings"]) for record in records
        ),
        "loading_figures_not_regenerated": sorted(stale_figures),
        "skipped_errors": skipped_errors,
        "files": records,
    }
    if apply:
        destination_root.mkdir(parents=True, exist_ok=True)
        (destination_root / "migration_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination_root / "README.md").write_text(
            "# Corrected loading-rate normalization\n\n"
            f"Absolute loading-rate values were multiplied by `{scale:.15g}`. "
            "Original simulation products remain in place and unchanged.\n\n"
            f"Default vapor assumptions: `{vapor.temperature_k}` K, "
            f"`{vapor.rb87_partial_pressure_torr}` Torr Rb-87 partial pressure, "
            f"and `{vapor.number_density_m3:.12g}` atoms/m^3.\n\n"
            "See `migration_manifest.json` for per-file hashes and the complete "
            "list of corrected products. Existing loading-rate images were not "
            "silently modified; regenerate them from the corrected data.\n",
            encoding="utf-8",
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("outputs") / MIGRATION_NAME,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write corrected copies; otherwise perform a read-only dry run",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = migrate(args.source_root, args.destination_root, apply=args.apply)
    print(
        json.dumps(
            {
                "applied": manifest["applied"],
                "corrected_file_count": manifest["corrected_file_count"],
                "scaled_numeric_field_count": manifest["scaled_numeric_field_count"],
                "replaced_prefactor_count": manifest["replaced_prefactor_count"],
                "multiplicative_scale": manifest["multiplicative_scale"],
                "destination_root": manifest["destination_root"],
                "skipped_error_count": len(manifest["skipped_errors"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
