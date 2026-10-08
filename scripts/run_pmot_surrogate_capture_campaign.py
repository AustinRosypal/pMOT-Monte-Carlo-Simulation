"""Run the time-bounded full-sphere surrogate-field capture campaign."""

from __future__ import annotations

from pathlib import Path

from pmot.pmot.surrogate_capture_campaign import (
    CampaignConfig,
    run_campaign,
    run_representative_audits,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        root
        / "outputs"
        / "diagnostics"
        / "pmot"
        / "MOT testing with a strange defined magnetic field"
        / "preliminary full-sphere capture and loading 20x10 20260930"
    )
    config = CampaignConfig()
    run_campaign(output_dir, config)
    run_representative_audits(output_dir, config)


if __name__ == "__main__":
    main()
