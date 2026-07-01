# Quantitative Hierarchical Meta-Analysis of Heavy Metal Adsorption onto Microplastics

This repository contains the data, analysis scripts, model summaries, result tables, and publication figures supporting a quantitative hierarchical meta-analysis of Langmuir maximum adsorption capacity (`Qm`) for heavy metal adsorption onto microplastics.

The reviewer-facing package is designed to make the analysis auditable without including local caches, virtual environments, manuscript lock files, or large regenerated intermediates.

## Study Scope

- Evidence base: 316 equilibrium isotherm experiments compiled from 50 studies.
- Response: Langmuir maximum adsorption capacity, `Qm`, harmonized as micrograms per gram (`ug/g`) in the cleaned analysis data.
- Predictors: metal identity, polymer identity, aging/weathering status, temperature, pH, surface area, and selected metal descriptors.
- Main analysis: hierarchical Bayesian mixed-effects models with study-level random effects.
- Additional analyses: model diagnostics, sensitivity checks, and Environmental Metal Vector Potential (EMVP) scenario screening.

## Repository Structure

```text
.
|-- config/                         Analysis configuration
|-- data_raw/                       Original compiled source workbook
|-- data_processed/                 Snapshot, cleaned, and enriched analysis data
|-- docs/                           Data policy and reproducibility documentation
|-- figures/                        Final manuscript figures
|-- manuscript/references/          Source-study reference inventories
|-- models/                         Compact model summary JSON files
|-- results/                        Compact result summaries
|-- scripts/                        Final analysis pipeline scripts
|-- tests/                          Lightweight reproducibility smoke tests
|-- run_pipeline.py                 Pipeline runner
|-- validate_data_policy_compliance.py
|-- environment.yml
`-- requirements.txt
```

Ignored local artifacts include `qm_metal_MP_env/`, `logs/`, `tmp/`, `output/`, Python caches, sensitivity caches, Word lock files, and large model trace files (`models/*_trace.nc`).

## Installation

Recommended setup with conda:

```bash
conda env create -f environment.yml
conda activate qm_meta_analysis
```

Pip-only setup:

```bash
pip install -r requirements.txt
```

The submitted analysis was prepared with Python 3.11. The scripts also compile under Python 3.12 in the local audit environment, but the locked conda environment is the preferred reproduction target.

## Reproducibility Checks

Run the lightweight checks first:

```bash
python -m compileall -q run_pipeline.py validate_data_policy_compliance.py scripts
python validate_data_policy_compliance.py
python run_pipeline.py --dry-run
```

If `pytest` is installed:

```bash
python -m pytest -q
```

Run the full pipeline:

```bash
python run_pipeline.py
```

The full Bayesian model-fitting and sensitivity steps can take substantial time. Compact model summaries and result tables are included so reviewers can inspect outputs without rerunning every MCMC trace.

## Data Handling Policy

The analysis follows the project data policy in `docs/reports/policy/data_policy.md`.

Key rules:

- pH is never globally imputed; missing pH values remain missing and are tracked by `pH_missing`.
- Only temperature (`Temp`), initial concentration (`AIC`), and rotation rate (`rpm`) may be imputed, with traceability flags.
- Contextual variables such as surface area (`SA`) and aging status (`AgS`) are preserved without global imputation.
- Cleaned `Qm` values are numeric and interpreted as `ug/g`; `log_Qm` is the natural logarithm of `Qm`.

## Key Entry Points

- `scripts/01_data_loading.py`: reads the raw workbook and freezes a raw-data snapshot.
- `scripts/02_data_cleaning.py`: applies cleaning, quality flags, and policy-compliant imputation.
- `scripts/03_descriptor_enrichment.py`: adds metal descriptors and derived variables.
- `scripts/05_model_fitting.py`: fits the primary Bayesian hierarchical models.
- `scripts/06_model_diagnostics.py`: computes convergence and posterior predictive diagnostics.
- `scripts/07_sensitivity_analysis.py`: performs leave-one-study-out, prior, perturbation, and bootstrap checks.
- `scripts/08_environmental_module.py`: runs EMVP scenario calculations.
The repository intentionally presents the final analysis package only. Internal revision-history folders, local manuscript drafts, and exploratory reviewer-response work directories are excluded to keep the public record concise.

## Citation

If using this repository before article publication, cite the manuscript associated with this GitHub release. Replace this section with the final DOI and journal citation after publication.

## License

This repository is released under the MIT License; see `LICENSE`.
