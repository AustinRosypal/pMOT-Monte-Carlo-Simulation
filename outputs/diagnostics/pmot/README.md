# pMOT diagnostic-result layout

Ordered scientific QA campaigns live in one named subdirectory per run. Each
run contains a top-level manifest and result index, one directory per executed
test, and explicit `NOT_RUN` records for tests skipped after the first failure.

```text
outputs/diagnostics/pmot/<campaign>/
  README.md
  run_manifest.json
  result_index.csv
  result_index.json
  test_00_configuration_and_units/
    README.md
    result.json
    *.csv
    *.json
  test_01_cooling_only_doppler/
    README.md
    result.json
    *.csv
    figures/*.png
  test_02_signed_vector_shift/
    README.md
    result.json
    *.csv
    *.json
    figures/*.png
  not_run_tests.csv
```

The numerical tables are the authoritative plot inputs. Figures are rendered
derivatives. Every test-level README summarizes the physical assumptions and
known limitations; exact criteria and tolerances are recorded in its
`result.json`. The runner obeys the canonical
[`DIAGNOSTIC_TESTS.md`](../../../docs/pmot/DIAGNOSTIC_TESTS.md) instruction to
stop after the first failed test; later tests are never silently treated as
passing.
