# MOT loading reports

Download the HTML files from GitHub or pull them with the repository, then open
them in a browser. Each file contains its own data and needs no local server or
files from `outputs/`. The temperature control reweights the saved loading
results at fixed Rb-87 partial pressure.

| Campaign | Report |
| --- | --- |
| 2026-11-03 · Effective two-level benchmark | [Loading parameter scans](2026-11-03_two_level/loading_parameter_scans.html) |
| 2026-11-04 · Physical 24-state MOT | [Multilevel loading scans](2026-11-04_multilevel/multilevel_loading_scans.html) |
| 2026-11-04 · Matched launch samples | [Two-level vs. multilevel comparison](2026-11-04_multilevel/two_level_multilevel_comparison.html) |
| 2026-11-05 · Symmetric strange-MOT, audited production | [Strange-MOT loading scans](symmetric_exact_D2_loading_20261105.html) |

The symmetric strange-MOT production report is complete. Its separate excluded
dense diagnostic was stopped at the user's request on November 8; its unfinished
validation is recorded separately. The existing two-level/multilevel loading
comparison does not include the strange-MOT loading data.

The [archive](archive/2026-11-04_original_multilevel/) holds the two HTML
previews from the superseded full-sample run. Those previews contain no
completed multilevel configurations.

The campaign generators still render into their original `outputs/figures/`
directories for reproducibility. If you regenerate a report, move the finished
HTML into the corresponding folder above to publish the refreshed export.
