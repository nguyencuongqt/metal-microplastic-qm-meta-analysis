# Data Handling Policy

This policy defines the mandatory data-handling rules for the Qm metal-microplastic adsorption meta-analysis. It is intended to make the cleaned dataset auditable by reviewers and to prevent silent transformations that would change the scientific interpretation.

## Scope

The policy applies to:

- `data_processed/02_cleaned_data.csv`
- `data_processed/03_enriched_data.csv`
- all downstream modeling, diagnostics, sensitivity, and EMVP scripts

The cleaned `Qm` values are numeric and interpreted as micrograms per gram (`ug/g`). The derived response `log_Qm` must be the natural logarithm of `Qm`.

## Rule 1: pH Is Not Globally Imputed

Missing pH values must remain missing. The cleaned dataset must contain `pH_missing`, with value `1` for rows where pH is missing and `0` otherwise.

Required invariant:

```python
assert "pH_missing" in df.columns
assert int(df["pH_missing"].sum()) == int(df["pH"].isna().sum())
```

## Rule 2: Limited Imputation Only

Only these predictors may be imputed:

- `Temp`
- `AIC`
- `rpm`

Each imputed value must be traceable with a binary flag:

- `imputed_Temp`
- `imputed_AIC`
- `imputed_rpm`

No other analysis covariate should be silently imputed.

## Rule 3: Contextual Variables Are Preserved

Variables such as surface area (`SA`), aging status (`AgS`), and solution-agent annotations must be retained as reported. Their missingness is part of the evidence structure and should not be globally filled.

## Rule 4: Qm Unit Handling

The project treats cleaned `Qm` values as `ug/g`. Unit labels must not remain embedded in numeric cells. The `Qm` column must be numeric and `log_Qm` must equal `ln(Qm)`.

Required invariant:

```python
assert pd.api.types.is_numeric_dtype(df["Qm"])
assert np.allclose(df["log_Qm"], np.log(df["Qm"]), rtol=1e-5)
```

## Validation

Run:

```bash
python validate_data_policy_compliance.py
```

The validator checks:

- pH preservation and `pH_missing`
- allowed imputation flags
- contextual-variable preservation
- numeric `Qm`
- `log_Qm = ln(Qm)`
- quality-score and quality-flag availability

The current cleaned dataset passes all policy checks.
