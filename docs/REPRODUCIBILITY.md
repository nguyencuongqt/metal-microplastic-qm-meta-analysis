# Reproducibility Notes

This document summarizes how reviewers can audit and rerun the analysis.

## Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate qm_meta_analysis
```

Alternative:

```bash
pip install -r requirements.txt
```

The locked environment targets Python 3.11. The local release audit also compiled successfully under Python 3.12.

## Lightweight Audit Commands

Run these before attempting the full pipeline:

```bash
python -m compileall -q run_pipeline.py validate_data_policy_compliance.py scripts tests
python validate_data_policy_compliance.py
python run_pipeline.py --dry-run
python -m pytest -q
```

The full `pytest` command requires installing the package dependencies from `requirements.txt` or `environment.yml`.

## Full Pipeline

```bash
python run_pipeline.py
```

The complete pipeline includes model fitting and sensitivity analyses. These steps may take substantial time because they run Bayesian sampling and repeated robustness checks.

Regenerate the current environmental-context figure after the primary model is available:

```bash
python scripts/08_environmental_context.py
```

## Included Data

- `data_raw/Qm data.xlsx`: compiled source workbook standardized with `Source_ID`, DOI, publication year, title, journal, and author metadata for the 50 source studies.
- `data_processed/00_raw_data_snapshot.csv`: frozen raw-data snapshot.
- `data_processed/02_cleaned_data.csv`: cleaned dataset used by the core analyses.
- `data_processed/03_enriched_data.csv`: cleaned dataset with metal descriptors and derived fields.
- `results/data_qc/qm_data_standardization_audit.csv`: source-level audit confirming row counts and DOI coverage after standardization.

## Included Outputs

The repository includes compact numerical summaries so reviewers can inspect the results without regenerating every large intermediate. Figures are generated locally by the analysis scripts and are not stored in the repository.

Large MCMC trace files (`models/*_trace.nc`) and sensitivity caches are intentionally excluded from GitHub. The compact `models/*_summary.json` files are included.

The post-model diagnostic and missingness summaries used in the current manuscript are retained under `results/additional_analyses/`. The total-water-pool EMVP comparison is retained under `results/environmental/`; obsolete regulatory-benchmark outputs are excluded.

## Release Audit Performed

The release checks were rerun after the August 2026 manuscript update:

- Python compile check for `run_pipeline.py`, `validate_data_policy_compliance.py`, `scripts/`, and `tests/`.
- Data policy validator.
- Pipeline dry run over all eight public pipeline steps.
- Smoke-test functions in `tests/test_reproducibility_smoke.py`.
- Qm source standardization audit: 316 rows, 50 sources, 50 unique DOI values, and zero missing DOI values.
