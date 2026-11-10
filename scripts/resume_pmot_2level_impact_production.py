"""Resume the frozen impact-stratified production plan with 16 workers."""

import multiprocessing

import run_pmot_2level_impact_loading as impact


def main():
    if not (impact.OUTPUT / "production" / "manifest.json").is_file():
        raise FileNotFoundError("The frozen production manifest is required to resume")
    impact.base.run("production", workers=16, max_discs=None)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
