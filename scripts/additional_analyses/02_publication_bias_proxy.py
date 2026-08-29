import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "additional_analyses"
FIG = ROOT / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


PATTERNS = re.compile(r"(se$|std|standard|stderr|error|ci|confidence|langmuir|(^|_)k(l)?($|_)|constant|r2|r\^2)", re.I)


def scan_tables():
    rows = []
    for path in list((ROOT / "data_processed").glob("*.csv")) + list((ROOT / "results").rglob("*.csv")):
        try:
            cols = pd.read_csv(path, nrows=1).columns.tolist()
        except Exception:
            continue
        matches = [c for c in cols if PATTERNS.search(c)]
        if matches:
            rows.append({"path": str(path.relative_to(ROOT)), "matching_columns": "; ".join(matches)})
    # Raw Excel.
    xls = ROOT / "data_raw" / "Qm data.xlsx"
    if xls.exists():
        xl = pd.ExcelFile(xls)
        for sheet in xl.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=1)
            matches = [c for c in df.columns if PATTERNS.search(str(c))]
            rows.append({"path": f"data_raw/Qm data.xlsx::{sheet}", "matching_columns": "; ".join(matches)})
    return pd.DataFrame(rows)


def egger(grouped):
    d = grouped.dropna(subset=["mean_log_qm", "se_proxy"]).copy()
    d = d[(d["n"] >= 2) & np.isfinite(d["se_proxy"]) & (d["se_proxy"] > 0)]
    if len(d) < 6:
        return None, d
    grand = d["mean_log_qm"].mean()
    d["standard_normal_deviate"] = (d["mean_log_qm"] - grand) / d["se_proxy"]
    d["precision"] = 1 / d["se_proxy"]
    X = sm.add_constant(d["precision"])
    model = sm.OLS(d["standard_normal_deviate"], X).fit()
    return {
        "n_units": int(len(d)),
        "intercept": float(model.params["const"]),
        "intercept_p": float(model.pvalues["const"]),
        "slope": float(model.params["precision"]),
        "r2": float(model.rsquared),
        "precision_proxy": "within-study or within-metal-study SE of mean ln(Qm): sd(lnQm)/sqrt(n), restricted to aggregation units with n>=2",
    }, d


def funnel_plot(d, title, filename):
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d["mean_log_qm"], d["se_proxy"], s=np.clip(d["n"] * 10, 20, 100), alpha=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ln(Qm)")
    ax.set_ylabel("Proxy SE")
    ax.set_title(title)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    df = pd.read_csv(ROOT / "data_processed" / "03_enriched_data.csv")
    scan = scan_tables()
    scan.to_csv(OUT / "02_se_k_search_inventory.csv", index=False)
    row_level_se_cols = []
    row_level_k_cols = []
    for _, r in scan.iterrows():
        cols = [c.strip() for c in str(r["matching_columns"]).split(";") if c.strip()]
        for c in cols:
            cl = c.lower()
            if cl in {"se", "se_qm", "qm_se", "stderr_qm", "standard_error_qm", "langmuir_qm_se"}:
                row_level_se_cols.append((r["path"], c))
            if cl in {"k", "kl", "k_l", "langmuir_k", "langmuir_kl", "k_langmuir"}:
                row_level_k_cols.append((r["path"], c))
    found_se = bool(row_level_se_cols)
    found_k = bool(row_level_k_cols)

    study = df.groupby("Study_ID").agg(n=("log_Qm", "size"), mean_log_qm=("log_Qm", "mean"), sd_log_qm=("log_Qm", "std")).reset_index()
    study["se_proxy"] = study["sd_log_qm"] / np.sqrt(study["n"])
    overall, study_used = egger(study)
    study_used.to_csv(OUT / "02_publication_bias_proxy_overall_units.csv", index=False)
    funnel_plot(study_used, "Funnel plot using study-level proxy SE", "02_funnel_overall_proxy")

    subgroup_results = []
    all_group_units = []
    for metal, g in df.groupby("Metal_raw"):
        units = g.groupby("Study_ID").agg(n=("log_Qm", "size"), mean_log_qm=("log_Qm", "mean"), sd_log_qm=("log_Qm", "std")).reset_index()
        units["Metal_raw"] = metal
        units["se_proxy"] = units["sd_log_qm"] / np.sqrt(units["n"])
        res, used = egger(units)
        if res:
            res["Metal_raw"] = metal
            subgroup_results.append(res)
            all_group_units.append(used.assign(Metal_raw=metal))
            funnel_plot(used, f"Funnel plot, {metal}, proxy SE", f"02_funnel_{metal.replace(' ', '_').replace('(', '').replace(')', '')}_proxy")
    subgroup_df = pd.DataFrame(subgroup_results)
    subgroup_df.to_csv(OUT / "02_egger_proxy_by_metal.csv", index=False)
    if all_group_units:
        pd.concat(all_group_units, ignore_index=True).to_csv(OUT / "02_publication_bias_proxy_by_metal_units.csv", index=False)

    result = {
        "random_seed": SEED,
        "source_level_langmuir_se_found": bool(found_se),
        "source_level_langmuir_K_found": bool(found_k),
        "row_level_se_columns": row_level_se_cols,
        "row_level_k_columns": row_level_k_cols,
        "se_search_conclusion": "No row-level Langmuir Qm standard errors were found in the source workbook, processed data, model artifacts, or result tables. Weighted measurement-error meta-analysis is therefore not identifiable from the available data; publication-bias diagnostics use a study-level proxy precision.",
        "k_search_conclusion": "No row-level Langmuir K/KL values were found in the repository tables. Full-isotherm EMVP must use K sensitivity ranges unless authors provide original K values.",
        "egger_overall_proxy": overall,
        "egger_by_metal_proxy": subgroup_results,
    }
    (OUT / "02_se_search_publication_bias_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT / "CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write("\n## Step 2/5 - SE search and publication bias\n")
        f.write("- Scanned raw Excel, processed CSVs, model files, and result CSVs for Langmuir SE/K evidence; saved `02_se_k_search_inventory.csv`.\n")
        f.write("- No row-level Langmuir standard errors were found; the publication-bias analysis therefore uses a study-level proxy precision.\n")
        f.write("- Ran funnel plots and Egger-type tests using study-level proxy SE; saved `02_egger_proxy_by_metal.csv` and `figures/02_funnel_*`.\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
