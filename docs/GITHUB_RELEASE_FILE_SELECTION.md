# GitHub Release File Selection

This repository is intentionally minimal. It includes only the files needed for reviewers to inspect and rerun the final analysis.

## Included

Core files:

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `environment.yml`
- `run_pipeline.py`
- `validate_data_policy_compliance.py`
- `config/config.yaml`
- `docs/REPRODUCIBILITY.md`
- `docs/reports/policy/data_policy.md`
- `tests/test_reproducibility_smoke.py`

Data:

- `data_raw/Qm data.xlsx`
- `data_processed/00_raw_data_snapshot.csv`
- `data_processed/02_cleaned_data.csv`
- `data_processed/03_enriched_data.csv`

Scripts:

- `scripts/utils.py`
- `scripts/00_standardize_qm_data.py`
- `scripts/01_data_loading.py`
- `scripts/02_data_cleaning.py`
- `scripts/03_descriptor_enrichment.py`
- `scripts/04_exploratory_analysis.py`
- `scripts/05_model_fitting.py`
- `scripts/06_model_diagnostics.py`
- `scripts/07_sensitivity_analysis.py`
- `scripts/08_environmental_module.py`
- `scripts/09_manuscript_materials.py`

Compact outputs:

- `models/*_summary.json`
- `results/data_qc/01_data_loading_metadata.json`
- `results/data_qc/02_cleaning_report.json`
- `results/data_qc/03_enrichment_validation.json`
- `results/data_qc/qm_data_standardization_audit.csv`
- `results/eda/04_eda_results.json`
- `results/diagnostics/06_diagnostics_results.json`
- `results/diagnostics/06_variance_partition_summary.csv`
- `results/environmental/08_emvp_summary_table.csv`
- `results/environmental/08_environmental_results.json`
- `results/environmental/08_environmental_verification.json`
- `results/environmental/08_emvp_sensitivity.json`
- `results/environmental/08_emvp_release_fraction_summary.csv`
- `results/sensitivity/*_summary.csv`

Final manuscript figures:

- `figures/Fig1_forest_plot.{png,pdf,svg}`
- `figures/Fig2_polymer_metal_heatmap.png`
- `figures/Fig3_partial_dependence.{png,pdf,svg}`
- `figures/Fig4_aging_enhancement.png`
- `figures/Fig5_EMVP_scenarios.{png,pdf,svg}`
- `figures/Fig6_sensitivity_tornado.{png,pdf,svg}`

References:

- `manuscript/REFERENCES.bib`
- `manuscript/references/QM_50_SOURCES.csv`
- `manuscript/references/QM_50_SOURCES.md`

## Excluded

The public repository excludes local environments, logs, caches, manuscript drafts, revision-history folders, high-resolution TIFFs, model traces, duplicate figure exports, and detailed intermediate audit tables.
