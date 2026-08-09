import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "additional_analyses"
FIG = ROOT / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def parse_year(s):
    m = re.search(r"(19|20)\d{2}", str(s))
    return int(m.group(0)) if m else np.nan


def fit_ols(d, formula, label):
    if len(d) < 20:
        return {"model": label, "n": int(len(d)), "status": "too_few_rows"}
    try:
        mod = smf.ols(formula, data=d).fit(cov_type="HC3")
        rows = []
        for term in mod.params.index:
            if term in ["Hydration_z", "pH_z", "SA_z", "Temp_z"] or term.startswith("C(AgS)"):
                rows.append({
                    "model": label,
                    "n": int(mod.nobs),
                    "term": term,
                    "estimate": float(mod.params[term]),
                    "se": float(mod.bse[term]),
                    "p": float(mod.pvalues[term]),
                    "r2": float(mod.rsquared),
                    "status": "ok",
                })
        return rows
    except Exception as e:
        return {"model": label, "n": int(len(d)), "status": "error", "error": str(e)}


def chi_test(df, missing_col, cat_col):
    tab = pd.crosstab(df[cat_col], df[missing_col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, dof, _ = st.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tab.shape) - 1))) if n else np.nan
    return {"missing": missing_col, "variable": cat_col, "test": "chi_square", "p": float(p), "cramers_v": float(cramers_v), "dof": int(dof)}


def continuous_test(df, missing_col, xcol):
    a = df.loc[df[missing_col] == 1, xcol].dropna()
    b = df.loc[df[missing_col] == 0, xcol].dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    u, p = st.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "missing": missing_col,
        "variable": xcol,
        "test": "mann_whitney",
        "n_missing": int(len(a)),
        "n_observed": int(len(b)),
        "median_missing": float(np.median(a)),
        "median_observed": float(np.median(b)),
        "p": float(p),
    }


def main() -> int:
    df = pd.read_csv(ROOT / "data_processed" / "03_enriched_data.csv")
    df["Year"] = df["Study_ID"].map(parse_year)
    df["is_oxyanion"] = df["Metal_raw"].isin(["As (III)", "Cr (VI)"])
    for col in ["Hydration_Energy", "pH", "SA", "Temp"]:
        mu, sd = df[col].mean(skipna=True), df[col].std(skipna=True)
        df[col.replace("Hydration_Energy", "Hydration") + "_z"] = (df[col] - mu) / sd

    formula = "log_Qm ~ Hydration_z + pH_z + C(AgS) + C(Study_ID)"
    mech_rows = []
    for label, d in [
        ("pooled_all_species_complete_pH", df.dropna(subset=["Hydration_z", "pH_z", "AgS", "Study_ID"])),
        ("cation_only_excluding_AsIII_CrVI", df.loc[~df["is_oxyanion"]].dropna(subset=["Hydration_z", "pH_z", "AgS", "Study_ID"])),
        ("oxyanion_only_AsIII_CrVI", df.loc[df["is_oxyanion"]].dropna(subset=["Hydration_z", "pH_z", "AgS", "Study_ID"])),
    ]:
        res = fit_ols(d, formula, label)
        mech_rows.extend(res if isinstance(res, list) else [res])
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT / "03_oxyanion_mechanistic_model_sensitivity.csv", index=False)

    # Missingness tests.
    df["pH_missing_flag"] = df["pH"].isna().astype(int)
    df["SA_missing_flag"] = df["SA"].isna().astype(int)
    miss_rows = []
    for miss_col in ["pH_missing_flag", "SA_missing_flag"]:
        for cat in ["Metal_raw", "ReT", "AgS", "Study_ID"]:
            r = chi_test(df, miss_col, cat)
            if r:
                miss_rows.append(r)
        for cont in ["log_Qm", "Year", "Temp"]:
            r = continuous_test(df, miss_col, cont)
            if r:
                miss_rows.append(r)
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(OUT / "03_missingness_tests.csv", index=False)

    # Multiple-imputation sensitivity for pH and SA in a fixed-effect surrogate model.
    impute_cols = ["log_Qm", "pH", "SA", "Temp", "Hydration_Energy"]
    base = df[impute_cols].copy()
    mi_rows = []
    for i in range(20):
        imp = IterativeImputer(random_state=SEED + i, sample_posterior=True, max_iter=20)
        vals = imp.fit_transform(base)
        tmp = df.copy()
        tmp[impute_cols] = vals
        for col in ["pH", "SA", "Temp", "Hydration_Energy"]:
            tmp[col.replace("Hydration_Energy", "Hydration") + "_z"] = (tmp[col] - tmp[col].mean()) / tmp[col].std()
        mod = smf.ols("log_Qm ~ Hydration_z + pH_z + SA_z + Temp_z + C(Metal_raw) + C(ReT) + C(AgS)", data=tmp).fit()
        for term in ["Hydration_z", "pH_z", "SA_z", "Temp_z"]:
            mi_rows.append({"imputation": i + 1, "term": term, "estimate": float(mod.params[term]), "se": float(mod.bse[term]), "r2": float(mod.rsquared)})
    mi = pd.DataFrame(mi_rows)
    mi.to_csv(OUT / "03_multiple_imputation_surrogate_coefficients.csv", index=False)
    mi_summary = mi.groupby("term").agg(mean_estimate=("estimate", "mean"), sd_estimate=("estimate", "std"), mean_se=("se", "mean"), mean_r2=("r2", "mean")).reset_index()
    mi_summary.to_csv(OUT / "03_multiple_imputation_surrogate_summary.csv", index=False)

    # As audit and leave-As-out sensitivity.
    as_rows = df[df["Metal_raw"].eq("As (III)")].copy()
    raw = pd.read_excel(ROOT / "data_raw" / "Qm data.xlsx")
    raw_as = raw[raw["Metal"].astype(str).str.contains("As", na=False)].copy()
    as_rows.to_csv(OUT / "03_asIII_cleaned_records.csv", index=False)
    raw_as.to_csv(OUT / "03_asIII_raw_excel_records.csv", index=False)

    rank_all = df.groupby("Metal_raw")["log_Qm"].mean().sort_values(ascending=False).reset_index(name="mean_ln_Qm_all")
    rank_no_as = df[df["Metal_raw"] != "As (III)"].groupby("Metal_raw")["log_Qm"].mean().sort_values(ascending=False).reset_index(name="mean_ln_Qm_without_As")
    rank_all.to_csv(OUT / "03_metal_descriptive_rank_all.csv", index=False)
    rank_no_as.to_csv(OUT / "03_metal_descriptive_rank_without_As.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.boxplot(column="log_Qm", by="Metal_raw", ax=ax, rot=45)
    ax.set_title("ln(Qm) by metal")
    ax.figure.suptitle("")
    ax.set_xlabel("Metal")
    ax.set_ylabel("ln(Qm)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"03_logQm_by_metal_as_audit.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = {
        "random_seed": SEED,
        "mechanistic_models": mech_df.to_dict(orient="records"),
        "missingness_interpretation": "pH and SA missingness are associated with study/design variables if chi-square or Mann-Whitney tests are significant; this supports MAR/plausible MNAR caution rather than MCAR.",
        "missingness_tests_min_p": float(miss_df["p"].min()),
        "as_audit": {
            "n_as_cleaned": int(len(as_rows)),
            "n_as_raw_excel": int(len(raw_as)),
            "raw_vs_cleaned_qm_match": bool(np.allclose(np.sort(as_rows["Qm"].values), np.sort(pd.to_numeric(raw_as.iloc[:, 10]).values))),
            "source_identified": "Dong et al. 2020, Chemosphere 239:124792, DOI 10.1016/j.chemosphere.2019.124792 for seven PS As(III) rows; Jiang et al. 2025 for one PBS row. Numeric values still require author full-text verification.",
        },
        "mi_summary": mi_summary.to_dict(orient="records"),
    }
    (OUT / "03_mechanistic_missingness_as_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / "CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write("\n## Step 3/4 - Mechanistic, missingness, and As audit\n")
        f.write("- Ran pooled, cation-only, and oxyanion-only hydration-energy surrogate models; saved `03_oxyanion_mechanistic_model_sensitivity.csv`.\n")
        f.write("- Tested pH/SA missingness against metal, polymer, aging, study, year, temperature, and ln(Qm); saved `03_missingness_tests.csv`.\n")
        f.write("- Ran 20-imputation fixed-effect sensitivity for pH/SA; saved `03_multiple_imputation_surrogate_summary.csv`.\n")
        f.write("- Audited As(III) raw vs cleaned rows and descriptive metal ranks with/without As; saved `03_asIII_*` and rank tables.\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
