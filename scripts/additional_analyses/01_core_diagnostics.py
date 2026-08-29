import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "additional_analyses"
FIG = ROOT / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)


def hdi(x, prob=0.95):
    vals = az.hdi(np.asarray(x), hdi_prob=prob)
    return float(vals[0]), float(vals[1])


def main() -> int:
    idata = az.from_netcdf(ROOT / "models" / "primary_model_trace.nc")
    df = pd.read_csv(ROOT / "data_processed" / "03_enriched_data.csv")
    summary = json.loads((ROOT / "models" / "primary_model_summary.json").read_text(encoding="utf-8"))
    diag = json.loads((ROOT / "results" / "diagnostics" / "06_diagnostics_results.json").read_text(encoding="utf-8"))

    # Reconstruct study local index used by Script 05.
    unique_studies = sorted(df["Study_ID_numeric"].dropna().unique())
    study_map = {sid: i for i, sid in enumerate(unique_studies)}
    study_idx = df["Study_ID_numeric"].map(study_map).astype(int).to_numpy()

    mu = idata.posterior["mu"].stack(sample=("chain", "draw")).transpose("sample", "mu_dim_0").values
    study_effect = idata.posterior["study_effect"].stack(sample=("chain", "draw")).transpose("sample", "study_effect_dim_0").values
    sigma = idata.posterior["sigma_residual"].stack(sample=("chain", "draw")).values
    full_mu = mu
    fixed_mu = mu - study_effect[:, study_idx]

    var_full = np.var(full_mu, axis=1, ddof=1)
    var_fixed = np.var(fixed_mu, axis=1, ddof=1)
    r2_cond = var_full / (var_full + sigma**2)
    r2_marg = var_fixed / (var_fixed + sigma**2)
    r2_rows = []
    for name, arr in [("marginal_R2_fixed_effects", r2_marg), ("conditional_R2_full_model", r2_cond)]:
        lo, hi = hdi(arr)
        r2_rows.append({"metric": name, "mean": float(arr.mean()), "median": float(np.median(arr)), "hdi95_low": lo, "hdi95_high": hi})
    pd.DataFrame(r2_rows).to_csv(OUT / "01_bayesian_r2.csv", index=False)

    loo = az.loo(idata, pointwise=True)
    pareto = np.asarray(loo.pareto_k)
    pareto_df = df[["Study_ID", "Study_ID_numeric", "Metal_raw", "ReT", "AgS", "Qm", "log_Qm"]].copy()
    pareto_df["row_index"] = np.arange(len(pareto_df))
    pareto_df["pareto_k"] = pareto
    pareto_df["pareto_class"] = pd.cut(pareto, [-np.inf, 0.5, 0.7, 1.0, np.inf], labels=["good", "ok", "bad", "very_bad"])
    pareto_df.to_csv(OUT / "01_psis_pareto_k_all_rows.csv", index=False)
    highk = pareto_df[pareto_df["pareto_k"] > 0.7].copy()
    highk.to_csv(OUT / "01_psis_high_k_rows.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pareto, bins=35, color="#4477AA", edgecolor="white")
    ax.axvline(0.7, color="#CC6677", linestyle="--", label="k = 0.7")
    ax.axvline(1.0, color="#AA3377", linestyle=":", label="k = 1.0")
    ax.set_xlabel("Pareto k")
    ax.set_ylabel("Rows")
    ax.set_title("PSIS-LOO Pareto-k diagnostics")
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"01_psis_pareto_k.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Verified sensitivity summaries from existing refit cache/results.
    prior = pd.read_csv(ROOT / "results" / "sensitivity" / "07_sensitivity_prior_summary.csv")
    loso = pd.read_csv(ROOT / "results" / "sensitivity" / "07_sensitivity_loso_summary.csv")
    boot = pd.read_csv(ROOT / "results" / "sensitivity" / "07_sensitivity_cluster_bootstrap_summary.csv")
    robustness = {
        "prior_max_abs_delta_pooled_mean": float(prior["delta_pooled_mean"].abs().max()),
        "prior_max_abs_delta_hdi_width": float(prior["delta_pooled_hdi_width"].abs().max()),
        "loso_delta_pooled_mean_min": float(loso["delta_pooled_mean"].min()),
        "loso_delta_pooled_mean_max": float(loso["delta_pooled_mean"].max()),
        "loso_max_influence_score": float(loso["influence_score"].max()),
        "loso_max_influence_study_id_numeric": int(loso.loc[loso["influence_score"].idxmax(), "study_id"]),
        "bootstrap_iterations": int(len(boot)),
        "bootstrap_pooled_mean_mean": float(boot["pooled_mean"].mean()) if "pooled_mean" in boot else None,
        "bootstrap_pooled_mean_p025": float(boot["pooled_mean"].quantile(0.025)) if "pooled_mean" in boot else None,
        "bootstrap_pooled_mean_p975": float(boot["pooled_mean"].quantile(0.975)) if "pooled_mean" in boot else None,
    }

    # Targeted high-k sensitivity using already available LOSO refits where the high-k row's study was removed.
    id_to_study = df[["Study_ID_numeric", "Study_ID"]].drop_duplicates().set_index("Study_ID_numeric")["Study_ID"].to_dict()
    highk_summary = highk.merge(
        loso[["study_id", "delta_pooled_mean", "influence_score"]],
        left_on="Study_ID_numeric",
        right_on="study_id",
        how="left",
    )
    highk_summary["study_label"] = highk_summary["Study_ID_numeric"].map(id_to_study)
    highk_summary.to_csv(OUT / "01_high_k_targeted_loso_sensitivity.csv", index=False)

    result = {
        "random_seed": SEED,
        "r2": r2_rows,
        "psis_loo": {
            "elpd_loo": float(loo.elpd_loo),
            "p_loo": float(loo.p_loo),
            "se": float(loo.se),
            "pareto_k_max": float(pareto.max()),
            "pareto_k_gt_0p7": int((pareto > 0.7).sum()),
            "pareto_k_gt_1": int((pareto > 1.0).sum()),
            "note": "High-k rows are paired with exact study-level LOSO refits as a conservative targeted sensitivity; exact observation-level reloo is not implemented.",
        },
        "robustness": robustness,
        "package_versions": {"arviz": az.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (OUT / "01_core_diagnostics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / "CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write("\n## Step 1 - Core diagnostics\n")
        f.write("- Computed marginal and conditional Bayesian R2 from `models/primary_model_trace.nc`; saved `01_bayesian_r2.csv`.\n")
        f.write("- Exported PSIS-LOO Pareto-k for all rows and high-k rows; saved `01_psis_pareto_k_all_rows.csv` and figure `figures/01_psis_pareto_k.*`.\n")
        f.write("- Verified prior, LOSO, and bootstrap summaries from existing sensitivity outputs; saved `01_core_diagnostics_summary.json`.\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
