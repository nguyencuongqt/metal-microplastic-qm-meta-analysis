"""Evidence-based manuscript assembly script for Qm metal adsorption project."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None
    sns = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

sys.path.append(str(Path(__file__).parent))
from utils import ProjectConfig, load_json, print_section_header, save_dataframe, save_json, setup_logging


MANUAL_FLAG = "[MANUAL CHECK NEEDED]"
RAW_QM_UNIT = "ug/g"
MODELING_QM_UNIT = "ug/g"
ENVIRONMENTAL_QM_UNIT = "mg/g"
MAIN_FIGURES_DEFAULT = [
    "Fig1_forest_plot",
    "Fig2_polymer_metal_heatmap",
    "Fig3_partial_dependence",
    "Fig4_aging_enhancement",
    "Fig5_EMVP_scenarios",
    "Fig6_sensitivity_tornado",
]
SUPPLEMENTARY_FIGURES_DEFAULT = [
    "Diagnostics_01_trace_plot",
    "Diagnostics_02_posterior_predictive",
    "Diagnostics_03_energy_plot",
    "Diagnostics_04_pareto_k_fallback",
    "EDA_01_response_distribution",
    "EDA_02_categorical_factors",
    "EDA_03_correlation_matrix",
    "EDA_03b_binary_indicator_logQm",
    "EDA_04_study_variation",
    "Environmental_01_EMVP_analysis",
    "Environmental_02_sensitivity_ci_width",
    "Sensitivity_01_perturbation",
    "Sensitivity_01_delta_posterior_perturbation",
    "Sensitivity_02_bootstrap",
    "Sensitivity_02_prior_variant_forest",
    "Sensitivity_03_influential_studies",
    "Sensitivity_03_loso_influence_tornado",
    "Sensitivity_04_cluster_bootstrap_distribution",
]


@dataclass
class ArtifactStatus:
    path: str
    exists: bool
    sha256: Optional[str]
    note: str


class WarningRegistry:
    def __init__(self) -> None:
        self.items: List[str] = []

    def add(self, text: str) -> str:
        if MANUAL_FLAG not in text:
            text = f"{text} {MANUAL_FLAG}"
        self.items.append(text)
        return text

    def unique(self) -> List[str]:
        return sorted(set(self.items))


class ProvenanceTracker:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.map: Dict[str, Dict[str, Any]] = {}

    def record(self, output_path: Path, inputs: List[Path], warnings: List[str]) -> None:
        input_records = []
        for path in inputs:
            input_records.append(
                {
                    "source_file": path.name,
                    "source_path": self._relpath(path),
                    "sha256": sha256_of_file(path),
                    "exists": path.exists(),
                }
            )
        self.map[str(output_path)] = {
            "output_file": output_path.name,
            "output_path": self._relpath(output_path),
            "output_sha256": sha256_of_file(output_path),
            "inputs": input_records,
            "warnings": warnings,
            "manual_check": len(warnings) > 0,
        }

    def _relpath(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except Exception:
            return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-based manuscript assembly")
    parser.add_argument(
        "--skip-upstream-figure-step",
        action="store_true",
        help="Deprecated compatibility flag; retained to avoid CLI breakage.",
    )
    parser.add_argument("--blueprint", default="manuscript/manuscript_blueprint.yaml")
    parser.add_argument("--output-dir", default="manuscript/generated_materials")
    parser.add_argument("--journal", default="water_research")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_of_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_results_artifact(project_root: Path, filename: str) -> Path:
    """Locate a results artifact by basename, searching the ``results/`` root and
    its immediate subdirectories (e.g. ``data_qc/``, ``eda/``, ``diagnostics/``,
    ``sensitivity/``, ``environmental/``).

    Returns the most recently modified match when several copies exist, or the
    canonical root path ``results/<filename>`` when none is found, so that
    downstream "missing artifact" warnings still report a clear path. This makes
    the assembly robust to whether upstream scripts write to ``results/`` or to a
    stage subdirectory, preventing tables from being silently stubbed out.
    """
    results_dir = project_root / "results"
    candidates = [results_dir / filename, *results_dir.glob(f"*/{filename}")]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return results_dir / filename
    return max(existing, key=lambda p: p.stat().st_mtime)


def required_inputs(project_root: Path) -> List[Path]:
    return [
        project_root / "data_processed" / "02_cleaned_data.csv",
        resolve_results_artifact(project_root, "01_data_loading_metadata.json"),
        resolve_results_artifact(project_root, "04_eda_results.json"),
        resolve_results_artifact(project_root, "06_diagnostics_results.json"),
        resolve_results_artifact(project_root, "07_sensitivity_results.json"),
        resolve_results_artifact(project_root, "07_sensitivity_perturbation_summary.csv"),
        resolve_results_artifact(project_root, "07_sensitivity_loso_summary.csv"),
        resolve_results_artifact(project_root, "07_sensitivity_prior_summary.csv"),
        resolve_results_artifact(project_root, "08_environmental_results.json"),
        resolve_results_artifact(project_root, "08_emvp_summary_table.csv"),
    ]


def ensure_layout(output_root: Path) -> Dict[str, Path]:
    layout = {
        "root": output_root,
        "drafts": output_root / "drafts",
        "tables": output_root / "tables",
        "captions": output_root / "captions",
        "evidence": output_root / "evidence",
        "evidence_manifests": output_root / "evidence" / "manifests",
        "evidence_inventories": output_root / "evidence" / "inventories",
        "fig_main_pdf": output_root / "figures" / "main" / "PDF",
        "fig_main_tiff": output_root / "figures" / "main" / "TIFF_300dpi",
        "fig_supp_pdf": output_root / "figures" / "supplementary" / "PDF",
        "fig_supp_tiff": output_root / "figures" / "supplementary" / "TIFF_300dpi",
    }
    for p in layout.values():
        p.mkdir(parents=True, exist_ok=True)
    return layout


def default_blueprint() -> Dict[str, Any]:
    return {
        "journal_target": "water_research",
        "figure_order": {
            "main": MAIN_FIGURES_DEFAULT,
            "supplementary": SUPPLEMENTARY_FIGURES_DEFAULT,
        },
        "table_order": [
            "Table_1_descriptive_stats",
            "Table_2_model_summary",
            "Table_S1_emvp_summary",
            "Table_S2_descriptors",
            "Table_S3_dataset_summary",
            "Table_S4_posterior_summary",
            "Table_S5_sensitivity_summary",
            "Table_S6_diagnostics_summary",
        ],
        "section_mapping": {
            "intro": "manual",
            "methods": "workflow+artifacts",
            "results": "artifacts_only",
            "discussion": "artifact_bounded",
            "limitations": "artifact_bounded",
            "conclusion": "artifact_bounded",
        },
    }


def _blueprint_uses_legacy_eda_main(fig_order: Dict[str, Any]) -> bool:
    main_figs = fig_order.get("main", []) or []
    return any(str(stem).startswith(("EDA_", "Environmental_")) for stem in main_figs)


def load_or_create_blueprint(path: Path, journal: str, warnings: WarningRegistry) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "journal_target" not in data:
            data["journal_target"] = journal
        fig_order = data.setdefault("figure_order", {})
        if _blueprint_uses_legacy_eda_main(fig_order):
            fig_order["main"] = MAIN_FIGURES_DEFAULT.copy()
            fig_order["supplementary"] = SUPPLEMENTARY_FIGURES_DEFAULT.copy()
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)
            warnings.add(f"Blueprint main figures were updated from legacy EDA defaults to manuscript main figures at {path}")
        else:
            main_figs = fig_order.setdefault("main", MAIN_FIGURES_DEFAULT.copy())
            supp_figs = fig_order.setdefault("supplementary", SUPPLEMENTARY_FIGURES_DEFAULT.copy())
            for stem in MAIN_FIGURES_DEFAULT:
                if stem not in main_figs:
                    main_figs.append(stem)
                    warnings.add(f"Blueprint main figures missing {stem}; appended default main figure")
            for stem in SUPPLEMENTARY_FIGURES_DEFAULT:
                if stem not in supp_figs:
                    supp_figs.append(stem)
        return data

    data = default_blueprint()
    data["journal_target"] = journal
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    warnings.add(f"Blueprint file did not exist and was auto-created at {path}")
    return data


def safe_load_json(path: Path, warnings: WarningRegistry, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning(warnings.add(f"Missing JSON artifact: {path}"))
        return None
    try:
        return load_json(path)
    except Exception as exc:  # pragma: no cover
        logger.warning(warnings.add(f"Failed to read JSON artifact {path}: {exc}"))
        return None


def safe_load_csv(path: Path, warnings: WarningRegistry, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning(warnings.add(f"Missing CSV artifact: {path}"))
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        logger.warning(warnings.add(f"Failed to read CSV artifact {path}: {exc}"))
        return None


def classify_figure(stem: str, blueprint: Dict[str, Any]) -> str:
    main = set(blueprint.get("figure_order", {}).get("main", []))
    if stem in main:
        return "main"
    return "supplementary"


def configure_plot_style() -> None:
    if plt is None:
        return
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10


def generate_fig2_polymer_metal_heatmap(cleaned: pd.DataFrame, output_png: Path) -> bool:
    if plt is None or sns is None:
        return False
    required = {"ReT", "Metal", "log_Qm"}
    if cleaned is None or not required.issubset(cleaned.columns):
        return False

    plot_df = cleaned.loc[:, ["ReT", "Metal", "log_Qm"]].dropna().copy()
    if plot_df.empty:
        return False

    polymer_order = plot_df["ReT"].value_counts().index.tolist()
    metal_order = plot_df["Metal"].value_counts().index.tolist()
    pivot = plot_df.pivot_table(index="ReT", columns="Metal", values="log_Qm", aggfunc="mean")
    pivot = pivot.reindex(index=polymer_order, columns=metal_order)

    configure_plot_style()
    fig_width = max(8.0, 1.1 * len(metal_order) + 2.0)
    fig_height = max(6.0, 0.45 * len(polymer_order) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Mean log(Qm)"},
        ax=ax,
    )
    ax.set_title("Polymer-metal adsorption heatmap", pad=12)
    ax.set_xlabel("Metal")
    ax.set_ylabel("Polymer")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png.exists()


def generate_fig4_aging_enhancement(cleaned: pd.DataFrame, output_png: Path) -> bool:
    if plt is None or sns is None:
        return False
    required = {"AgS", "Metal", "log_Qm"}
    if cleaned is None or not required.issubset(cleaned.columns):
        return False

    plot_df = cleaned.loc[:, ["AgS", "Metal", "log_Qm"]].dropna().copy()
    plot_df["AgS"] = plot_df["AgS"].astype(str).str.strip().str.title()
    plot_df = plot_df[plot_df["AgS"].isin(["Aged", "Virgin"])].copy()
    if plot_df.empty:
        return False

    summary = (
        plot_df.groupby(["Metal", "AgS"])
        .agg(mean_log_qm=("log_Qm", "mean"), n=("log_Qm", "size"))
        .reset_index()
    )
    pivot_mean = summary.pivot(index="Metal", columns="AgS", values="mean_log_qm")
    pivot_n = summary.pivot(index="Metal", columns="AgS", values="n")
    valid = pivot_mean.dropna(subset=["Aged", "Virgin"], how="any").copy()
    if valid.empty:
        return False

    valid["enhancement_ratio"] = (valid["Aged"] - valid["Virgin"]).apply(math.exp)
    valid["n_label"] = valid.index.map(
        lambda metal: f"nA={int(pivot_n.loc[metal, 'Aged'])}, nV={int(pivot_n.loc[metal, 'Virgin'])}"
    )
    valid = valid.sort_values("enhancement_ratio", ascending=True)

    overall = plot_df.groupby("AgS")["log_Qm"].mean()
    overall_ratio = math.exp(overall["Aged"] - overall["Virgin"]) if {"Aged", "Virgin"}.issubset(overall.index) else None

    configure_plot_style()
    fig_height = max(5.5, 0.55 * len(valid.index) + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    bars = ax.barh(valid.index.astype(str), valid["enhancement_ratio"], color="#2f6c8f")
    ax.axvline(1.0, color="#a61c3c", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Aged / Virgin geometric mean Qm ratio")
    ax.set_ylabel("Metal")
    title = "Aging enhancement ratios by metal"
    if overall_ratio is not None:
        title += f" (overall={overall_ratio:.2f}x)"
    ax.set_title(title, pad=12)
    ax.set_xlim(left=0)
    for bar, ratio, n_label in zip(bars, valid["enhancement_ratio"], valid["n_label"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{ratio:.2f}x | {n_label}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png.exists()


def generate_missing_main_figure_sources(
    figures_dir: Path,
    cleaned: Optional[pd.DataFrame],
    warnings: WarningRegistry,
    logger: logging.Logger,
) -> None:
    if cleaned is None:
        return

    generation_plan = {
        "Fig2_polymer_metal_heatmap": generate_fig2_polymer_metal_heatmap,
        "Fig4_aging_enhancement": generate_fig4_aging_enhancement,
    }
    for stem, builder in generation_plan.items():
        output_png = figures_dir / f"{stem}.png"
        if output_png.exists():
            continue
        try:
            ok = builder(cleaned, output_png)
        except Exception as exc:  # pragma: no cover
            logger.warning(warnings.add(f"Failed to generate derived source PNG for {stem}: {exc}"))
            continue
        if ok:
            logger.info(f"Generated derived source PNG for {stem}: {output_png}")
        else:
            logger.warning(warnings.add(f"Could not generate derived source PNG for {stem}"))


def export_figure_packages(
    figures_dir: Path,
    layout: Dict[str, Path],
    blueprint: Dict[str, Any],
    warnings: WarningRegistry,
    logger: logging.Logger,
) -> Tuple[Dict[str, int], pd.DataFrame, List[Path]]:
    order_index: Dict[str, int] = {}
    seq = blueprint.get("figure_order", {}).get("main", []) + blueprint.get("figure_order", {}).get("supplementary", [])
    for idx, stem in enumerate(seq, start=1):
        order_index[stem] = idx

    if not seq:
        logger.warning(warnings.add("No figures configured in blueprint"))
        return {"source_png_total": 0, "pdf_created": 0, "tiff_created": 0}, pd.DataFrame(), []

    pdf_created = 0
    tiff_created = 0
    rows: List[Dict[str, Any]] = []
    exported_files: List[Path] = []

    for stem in seq:
        group = classify_figure(stem, blueprint)
        pdf_out = (layout["fig_main_pdf"] if group == "main" else layout["fig_supp_pdf"]) / f"{stem}.pdf"
        tiff_out = (layout["fig_main_tiff"] if group == "main" else layout["fig_supp_tiff"]) / f"{stem}.tif"
        source_png = figures_dir / f"{stem}.png"
        source_exists = source_png.exists()

        pdf_ok = False
        tiff_ok = False
        if not source_exists:
            logger.warning(warnings.add(f"Missing source PNG for {stem}"))
        elif Image is None:
            logger.warning(warnings.add("Pillow not installed; skipped PDF/TIFF conversion"))
        else:
            try:
                with Image.open(source_png) as opened:
                    img = opened.copy()
                if img.mode in ("RGBA", "LA", "P"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(pdf_out, format="PDF", quality=95, optimize=True)
                pdf_ok = pdf_out.exists()
                if pdf_ok:
                    pdf_created += 1
                    exported_files.append(pdf_out)
                img.save(tiff_out, format="TIFF", dpi=(300, 300), compression="lzw")
                tiff_ok = tiff_out.exists()
                if tiff_ok:
                    tiff_created += 1
                    exported_files.append(tiff_out)
            except Exception as exc:  # pragma: no cover
                logger.warning(warnings.add(f"Figure conversion failed for {stem}: {exc}"))

        rows.append(
            {
                "figure_stem": stem,
                "group": group,
                "order": order_index.get(stem, 999),
                "source_png": str(source_png),
                "exported_pdf": str(pdf_out),
                "exported_tiff": str(tiff_out),
                "source_exists": source_exists,
                "pdf_exists": pdf_ok,
                "tiff300_exists": tiff_ok,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["group", "order", "figure_stem"]).reset_index(drop=True)
    return {
        "source_png_total": len(list(figures_dir.glob("*.png"))),
        "pdf_created": pdf_created,
        "tiff_created": tiff_created,
    }, df, exported_files


def infer_qm_unit_policy(
    metadata: Optional[Dict[str, Any]],
    cleaned: Optional[pd.DataFrame],
    unit_verification: Optional[Dict[str, Any]],
    env_json: Optional[Dict[str, Any]],
    warnings: WarningRegistry,
) -> Dict[str, Any]:
    metadata_unit = (((metadata or {}).get("quality_report", {}).get("qm_unit_report", {}) or {}).get("units_found_in_headers", [None])[0])
    if metadata_unit in {"\u00b5g/g", "\u03bcg/g", "ug/g"}:
        metadata_unit = RAW_QM_UNIT
    raw_rows = ((metadata or {}).get("quality_report", {}) or {}).get("total_rows")
    cleaned_rows = int(cleaned.shape[0]) if cleaned is not None else None
    verification_rows = (unit_verification or {}).get("n_rows")
    env_uses_mg_g = '"qm_mean_mg_g"' in json.dumps(env_json or {})

    policy = {
        "raw_dataset_qm_unit": metadata_unit or RAW_QM_UNIT,
        "modeled_qm_unit": MODELING_QM_UNIT,
        "modeled_log_transform": (unit_verification or {}).get("inferred_log_base_from_data", "natural_ln"),
        "environmental_qm_unit": ENVIRONMENTAL_QM_UNIT if env_uses_mg_g else None,
        "environmental_conversion_applied": bool(env_uses_mg_g),
        "scientific_rationale": "Environmental scenario summaries convert Qm from ug/g to mg/g for readability while core modeling retains Qm in ug/g before natural-log transformation.",
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "unit_verification_rows": verification_rows,
    }
    if verification_rows is not None and cleaned_rows is not None and verification_rows != cleaned_rows:
        warnings.add(f"Qm unit verification row count ({verification_rows}) != cleaned dataset rows ({cleaned_rows})")
    if metadata_unit and metadata_unit not in {"ug/g", "\u00b5g/g", "\u03bcg/g"}:
        warnings.add(f"Unexpected raw Qm unit label in metadata: {metadata_unit}")
    return policy


def build_artifact_summary(
    cleaned: Optional[pd.DataFrame],
    metadata: Optional[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]],
    unit_policy: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned_rows = int(cleaned.shape[0]) if cleaned is not None else None
    cleaned_studies = int(cleaned["Study_ID"].nunique(dropna=True)) if cleaned is not None and "Study_ID" in cleaned.columns else None
    ppc = (diagnostics or {}).get("posterior_predictive", {})
    loso = (diagnostics or {}).get("loso_validation", {})
    return {
        "n_observations_cleaned": cleaned_rows,
        "n_observations_diagnostics": ppc.get("n_observations"),
        "n_studies_cleaned": cleaned_studies,
        "n_studies_loso": loso.get("n_studies"),
        "validation_status": loso.get("status"),
        "validation_performed": loso.get("validation_performed"),
        "diagnostic_type": ppc.get("diagnotic_type"),
        "raw_qm_unit": unit_policy.get("raw_dataset_qm_unit"),
        "modeled_qm_unit": unit_policy.get("modeled_qm_unit"),
        "environmental_qm_unit": unit_policy.get("environmental_qm_unit"),
        "raw_rows": ((metadata or {}).get("quality_report", {}) or {}).get("total_rows"),
    }


def build_verification_checks(
    cleaned: Optional[pd.DataFrame],
    metadata: Optional[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]],
    unit_verification: Optional[Dict[str, Any]],
    unit_policy: Dict[str, Any],
    fig_inventory: pd.DataFrame,
    blueprint: Dict[str, Any],
    required_artifacts: List[Path],
    warnings: WarningRegistry,
) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []
    raw_rows = ((metadata or {}).get("quality_report", {}) or {}).get("total_rows")
    cleaned_rows = int(cleaned.shape[0]) if cleaned is not None else None
    diag_rows = ((diagnostics or {}).get("posterior_predictive", {}) or {}).get("n_observations")
    unit_rows = (unit_verification or {}).get("n_rows")

    row_status = len({v for v in [raw_rows, cleaned_rows, diag_rows] if v is not None}) <= 1
    checks.append({
        "check": "dataset_row_count_consistency",
        "status": "PASS" if row_status else "WARN",
        "details": f"raw={raw_rows}; cleaned={cleaned_rows}; diagnostics={diag_rows}",
    })
    if not row_status:
        warnings.add(f"Dataset row count mismatch across artifacts: raw={raw_rows}, cleaned={cleaned_rows}, diagnostics={diag_rows}")

    unit_status = unit_policy.get("raw_dataset_qm_unit") in {"ug/g", "\u00b5g/g", "\u03bcg/g"} and unit_policy.get("modeled_qm_unit") == MODELING_QM_UNIT
    checks.append({
        "check": "qm_unit_policy_consistency",
        "status": "PASS" if unit_status else "WARN",
        "details": f"raw={unit_policy.get('raw_dataset_qm_unit')}; modeled={unit_policy.get('modeled_qm_unit')}; environmental={unit_policy.get('environmental_qm_unit')}",
    })
    if not unit_status:
        warnings.add("Qm unit policy could not be reconciled across raw/model/environmental artifacts")

    unit_rows_match = unit_rows is None or cleaned_rows is None or unit_rows == cleaned_rows
    checks.append({
        "check": "qm_unit_verification_rows",
        "status": "PASS" if unit_rows_match else "WARN",
        "details": f"unit_verification={unit_rows}; cleaned={cleaned_rows}",
    })

    key_artifacts_ok = all(p.exists() for p in required_artifacts)
    checks.append({
        "check": "key_model_artifacts_exist",
        "status": "PASS" if key_artifacts_ok else "WARN",
        "details": ", ".join([p.name for p in required_artifacts if p.exists()]) or "none",
    })
    if not key_artifacts_ok:
        warnings.add("One or more key model artifacts required by manuscript assembly are missing")

    expected_main = blueprint.get("figure_order", {}).get("main", [])
    actual_main = set(fig_inventory.loc[fig_inventory["group"] == "main", "figure_stem"].tolist()) if not fig_inventory.empty else set()
    missing_main = [stem for stem in expected_main if stem not in actual_main]
    checks.append({
        "check": "figure_inventory_consistency",
        "status": "PASS" if not missing_main else "WARN",
        "details": "missing_main=" + (", ".join(missing_main) if missing_main else "none"),
    })
    if missing_main:
        warnings.add(f"Expected publication figure(s) missing from figures folder: {', '.join(missing_main)}")

    return pd.DataFrame(checks)


def summarize_consistency(
    cleaned: Optional[pd.DataFrame],
    metadata: Optional[Dict[str, Any]],
    eda: Optional[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]],
    sensitivity_json: Optional[Dict[str, Any]],
    sensitivity_perturb: Optional[pd.DataFrame],
    sensitivity_loso: Optional[pd.DataFrame],
    warnings: WarningRegistry,
) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    clean_n = cleaned.shape[0] if cleaned is not None else None
    eda_n = eda.get("response_distribution", {}).get("n") if eda else None
    ppc_n = diagnostics.get("posterior_predictive", {}).get("n_observations") if diagnostics else None
    loso_n = diagnostics.get("loso_validation", {}).get("n_studies") if diagnostics else None
    cleaned_studies = int(cleaned["Study_ID"].nunique(dropna=True)) if cleaned is not None and "Study_ID" in cleaned.columns else None

    n_set = {v for v in [clean_n, eda_n, ppc_n] if v is not None}
    n_match = len(n_set) <= 1
    if not n_match:
        warnings.add(f"N mismatch across cleaned/EDA/diagnostics: {clean_n}/{eda_n}/{ppc_n}")
    checks.append({"check": "N_consistency", "status": "PASS" if n_match else "WARN", "details": f"{clean_n}/{eda_n}/{ppc_n}"})

    study_set = {v for v in [cleaned_studies, loso_n] if v is not None}
    study_match = len(study_set) <= 1
    if not study_match:
        warnings.add(f"Study count mismatch across cleaned/diagnostics: {cleaned_studies}/{loso_n}")
    checks.append({"check": "study_count_consistency", "status": "PASS" if study_match else "WARN", "details": f"{cleaned_studies}/{loso_n}"})

    if metadata:
        total_rows = metadata.get("quality_report", {}).get("total_rows")
        same_rows = (total_rows == clean_n) if (total_rows is not None and clean_n is not None) else False
        if not same_rows:
            warnings.add(f"Metadata total_rows ({total_rows}) != cleaned rows ({clean_n})")
        checks.append({"check": "metadata_vs_cleaned_rows", "status": "PASS" if same_rows else "WARN", "details": f"{total_rows}/{clean_n}"})
    else:
        checks.append({"check": "metadata_vs_cleaned_rows", "status": "WARN", "details": warnings.add("Metadata missing for consistency check")})

    sens_ok = sensitivity_json is not None
    checks.append({"check": "sensitivity_json_present", "status": "PASS" if sens_ok else "WARN", "details": "07_sensitivity_results.json"})
    if not sens_ok:
        warnings.add("Sensitivity JSON missing")

    for name, df in [("perturbation", sensitivity_perturb), ("loso", sensitivity_loso)]:
        ok = df is not None and not df.empty
        checks.append({"check": f"sensitivity_{name}_csv", "status": "PASS" if ok else "WARN", "details": "present" if ok else "missing/empty"})
        if not ok:
            warnings.add(f"Sensitivity {name} CSV missing or empty")

    return pd.DataFrame(checks)


def make_table_from_rows(rows: List[Dict[str, Any]], fallback_note: str) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame([{"note": f"{fallback_note} {MANUAL_FLAG}"}])


def build_tables(
    cleaned: Optional[pd.DataFrame],
    emvp: Optional[pd.DataFrame],
    diagnostics: Optional[Dict[str, Any]],
    sensitivity_json: Optional[Dict[str, Any]],
    sensitivity_perturb: Optional[pd.DataFrame],
    sensitivity_prior: Optional[pd.DataFrame],
    sensitivity_loso: Optional[pd.DataFrame],
    config: ProjectConfig,
    warnings: WarningRegistry,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    tables: Dict[str, pd.DataFrame] = {}
    captions: Dict[str, str] = {}

    t1_rows: List[Dict[str, Any]] = []
    if cleaned is not None and {"Metal", "log_Qm"}.issubset(cleaned.columns):
        for metal in sorted(cleaned["Metal"].dropna().unique()):
            vals = cleaned.loc[cleaned["Metal"] == metal, "log_Qm"].dropna()
            if not vals.empty:
                t1_rows.append({
                    "Metal": metal,
                    "n": int(vals.shape[0]),
                    "Mean_log_Qm": float(vals.mean()),
                    "SD_log_Qm": float(vals.std(ddof=1)) if vals.shape[0] > 1 else None,
                    "Median_log_Qm": float(vals.median()),
                })
    else:
        warnings.add("Table_1 requires Metal and log_Qm in cleaned data")
    tables["Table_1_descriptive_stats"] = make_table_from_rows(t1_rows, "Unable to construct Table_1")
    captions["Table_1_descriptive_stats"] = "Table 1. Descriptive statistics by metal on modeled log(Qm) scale."

    t2_rows: List[Dict[str, Any]] = []
    if diagnostics:
        conv = diagnostics.get("convergence", {})
        ppc = diagnostics.get("posterior_predictive", {})
        t2_rows = [
            {"Metric": "Rhat_max", "Value": conv.get("rhat", {}).get("max", f"{MANUAL_FLAG}")},
            {"Metric": "ESS_bulk_min", "Value": conv.get("ess_bulk", {}).get("min", f"{MANUAL_FLAG}")},
            {"Metric": "ESS_tail_min", "Value": conv.get("ess_tail", {}).get("min", f"{MANUAL_FLAG}")},
            {"Metric": "Posterior_predictive_R2", "Value": ppc.get("r2", f"{MANUAL_FLAG}")},
            {"Metric": "Posterior_predictive_RMSE", "Value": ppc.get("rmse", f"{MANUAL_FLAG}")},
            {"Metric": "Validation_scope", "Value": ppc.get("diagnotic_type", f"{MANUAL_FLAG}")},
        ]
    else:
        warnings.add("Diagnostics JSON unavailable for Table_2_model_summary")
    tables["Table_2_model_summary"] = make_table_from_rows(t2_rows, "Unable to construct Table_2 model summary")
    captions["Table_2_model_summary"] = "Table 2. Model summary metrics from diagnostics artifacts only."

    if emvp is not None and not emvp.empty:
        cols = [c for c in ["metal", "scenario", "qm_mean_mg_g", "emvp_mean_ug_L", "emvp_ci_95_lower", "emvp_ci_95_upper", "ratio_mean", "interpretation"] if c in emvp.columns]
        tables["Table_S1_emvp_summary"] = emvp[cols].copy()
    else:
        warnings.add("EMVP CSV unavailable for Table_S1")
        tables["Table_S1_emvp_summary"] = make_table_from_rows([], "Unable to construct Table_S1")
    captions["Table_S1_emvp_summary"] = "Table S1. EMVP scenario outputs and ratio-to-background interpretation."

    desc_rows = []
    for metal, vals in (config.get("metal_descriptors", {}) or {}).items():
        desc_rows.append({
            "Metal": metal,
            "Hydration_energy_kJ_mol": vals.get("hydration_energy"),
            "Ionic_radius_A": vals.get("ionic_radius"),
            "Valence": vals.get("valence"),
            "Electronegativity": vals.get("electronegativity"),
            "HSAB_hardness": vals.get("hardness"),
        })
    tables["Table_S2_descriptors"] = make_table_from_rows(desc_rows, "Unable to construct Table_S2")
    captions["Table_S2_descriptors"] = "Table S2. Metal descriptors from project configuration."

    s3_rows = []
    if cleaned is not None:
        s3_rows.append({"Metric": "cleaned_rows", "Value": int(cleaned.shape[0])})
        s3_rows.append({"Metric": "cleaned_columns", "Value": int(cleaned.shape[1])})
        if "Metal" in cleaned.columns:
            s3_rows.append({"Metric": "unique_metals", "Value": int(cleaned["Metal"].nunique(dropna=True))})
    else:
        warnings.add("Cleaned data unavailable for Table_S3")
    tables["Table_S3_dataset_summary"] = make_table_from_rows(s3_rows, "Unable to construct Table_S3")
    captions["Table_S3_dataset_summary"] = "Table S3. Dataset summary from cleaned data and consistency checks."

    s4_rows = []
    coeffs = (sensitivity_json or {}).get("baseline_metrics", {}).get("coefficients", {})
    if coeffs:
        for name, v in coeffs.items():
            s4_rows.append(
                {
                    "parameter": name,
                    "mean": v.get("mean"),
                    "hdi_low": v.get("hdi_low"),
                    "hdi_high": v.get("hdi_high"),
                    "prob_gt_zero": v.get("prob_gt_zero"),
                    "contains_zero": v.get("contains_zero"),
                }
            )
    else:
        warnings.add("Posterior coefficient summary unavailable for Table_S4")
    tables["Table_S4_posterior_summary"] = make_table_from_rows(s4_rows, "Unable to construct Table_S4")
    captions["Table_S4_posterior_summary"] = "Table S4. Posterior coefficient summary from sensitivity baseline artifact."

    s5_rows = []
    if sensitivity_perturb is not None and not sensitivity_perturb.empty:
        s5_rows.append({"module": "perturbation", "rows": int(sensitivity_perturb.shape[0]), "notes": "from CSV"})
    if sensitivity_prior is not None and not sensitivity_prior.empty:
        s5_rows.append({"module": "prior", "rows": int(sensitivity_prior.shape[0]), "notes": "from CSV"})
    if sensitivity_loso is not None and not sensitivity_loso.empty:
        s5_rows.append({"module": "loso", "rows": int(sensitivity_loso.shape[0]), "notes": "from CSV"})
    if not s5_rows:
        warnings.add("Sensitivity CSV summaries unavailable for Table_S5")
    tables["Table_S5_sensitivity_summary"] = make_table_from_rows(s5_rows, "Unable to construct Table_S5")
    captions["Table_S5_sensitivity_summary"] = "Table S5. Sensitivity summary availability and row counts from authoritative CSV artifacts."

    s6_rows = []
    if diagnostics:
        adv = diagnostics.get("advanced_diagnostics", {})
        ppc = diagnostics.get("posterior_predictive", {})
        s6_rows = [
            {"check": "divergences", "value": adv.get("divergences", {}).get("count", f"{MANUAL_FLAG}")},
            {"check": "bfmi_min", "value": adv.get("bfmi", {}).get("min", f"{MANUAL_FLAG}")},
            {"check": "pareto_k_max", "value": adv.get("psis_loo", {}).get("pareto_k", {}).get("max", f"{MANUAL_FLAG}")},
            {"check": "ppc_coverage_95", "value": ppc.get("coverage_95", f"{MANUAL_FLAG}")},
        ]
    else:
        warnings.add("Diagnostics artifact unavailable for Table_S6")
    tables["Table_S6_diagnostics_summary"] = make_table_from_rows(s6_rows, "Unable to construct Table_S6")
    captions["Table_S6_diagnostics_summary"] = "Table S6. Diagnostics summary from convergence and posterior predictive artifacts."

    return tables, captions


def write_manuscript_skeleton(
    output_file: Path,
    journal: str,
    workflow_notes: List[str],
    warnings: WarningRegistry,
    inputs_used: List[Path],
    artifact_summary: Dict[str, Any],
    unit_policy: Dict[str, Any],
) -> None:
    warning_block = "\n".join([f"- {w}" for w in warnings.unique()]) or "- None"
    workflow_block = "\n".join([f"- {w}" for w in workflow_notes])
    input_block = "\n".join([f"- {p}" for p in inputs_used])
    evidence_block = "\n".join(
        [
            f"- Cleaned observations: {artifact_summary.get('n_observations_cleaned')}",
            f"- Diagnostics observations: {artifact_summary.get('n_observations_diagnostics')}",
            f"- Studies in cleaned dataset: {artifact_summary.get('n_studies_cleaned')}",
            f"- LOSO studies: {artifact_summary.get('n_studies_loso')}",
            f"- LOSO validation performed: {artifact_summary.get('validation_performed')}",
            f"- Validation status: {artifact_summary.get('validation_status')}",
            f"- Diagnostics scope: {artifact_summary.get('diagnostic_type')}",
            f"- Raw/modeling Qm unit: {artifact_summary.get('raw_qm_unit')} -> {artifact_summary.get('modeled_qm_unit')} (natural-log transformed for modeling)",
            f"- Environmental Qm unit in scenario outputs: {artifact_summary.get('environmental_qm_unit')}",
        ]
    )
    unit_block = "\n".join(
        [
            f"- Raw extraction unit: {unit_policy.get('raw_dataset_qm_unit')}",
            f"- Modeling unit retained for Qm and log(Qm): {unit_policy.get('modeled_qm_unit')}",
            f"- Environmental scenario summaries use: {unit_policy.get('environmental_qm_unit')}",
            f"- Conversion policy: {unit_policy.get('scientific_rationale')}",
        ]
    )

    text = f"""# Manuscript Skeleton (Evidence-Based)

## Journal Target
- {journal}

## Title Placeholder
- {MANUAL_FLAG} Provide title after author review.

## Abstract Skeleton
- Background: concise context from verified project scope only.
- Objective: quantify adsorption behavior and environmental scenario outputs.
- Methods: mention data-cleaning, diagnostics, sensitivity, and EMVP artifacts.
- Results: include only values traceable to artifacts.
- Conclusion: conservative, no external generalization claim.

## Methods Summary (Workflow-Derived)
{workflow_block}

## Artifact-Derived Quantitative Summary
{evidence_block}

## Qm Unit Policy
{unit_block}

## Results Skeleton (Evidence-Derived)
- Report model diagnostics from results/06_diagnostics_results.json only.
- Report sensitivity from results/07_sensitivity_*.csv and 07_sensitivity_results.json.
- Report EMVP from results/08_emvp_summary_table.csv.
- If conflicts exist, annotate {MANUAL_FLAG}.

## Discussion Placeholder
- Interpret within artifact scope.
- Do not overclaim causality or out-of-sample generalization.

## Limitations Placeholder
- Explicitly state in-sample calibration limitations and missing validations.

## Conclusion Placeholder
- Keep conclusion bounded by validated artifacts.

## Authoritative Inputs Used
{input_block}

## Open Manual Checks
{warning_block}
"""
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_captions(
    fig_inventory: pd.DataFrame,
    table_captions: Dict[str, str],
    captions_dir: Path,
    warnings: WarningRegistry,
    diagnostics: Optional[Dict[str, Any]],
    env_json: Optional[Dict[str, Any]],
    unit_policy: Dict[str, Any],
) -> None:
    def normalize_caption_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def fmt_caption_num(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{numeric:.2f}"

    ppc = (diagnostics or {}).get("posterior_predictive", {})
    loso = (diagnostics or {}).get("loso_validation", {})
    env_priority = (((env_json or {}).get("scenarios", {}) or {}).get("priority_metals", {}) or {})
    caption_map = {
        "Fig1_forest_plot": "Figure 1. Forest plot of model-estimated adsorption effect sizes with uncertainty intervals on the log(Qm) scale.",
        "Fig2_polymer_metal_heatmap": "Figure 2. Polymer x metal heatmap of mean log(Qm) values computed from the cleaned study corpus, highlighting adsorption intensity patterns across polymer types and target metals.",
        "Fig3_partial_dependence": "Figure 3. Partial dependence plots showing the modeled change in log(Qm) across key predictors while holding other terms at their fitted values.",
        "Fig4_aging_enhancement": "Figure 4. Aging enhancement ratios by metal, expressed as the geometric-mean Qm ratio for aged versus virgin microplastic conditions, with sample counts annotated for each comparison.",
        "Fig5_EMVP_scenarios": f"Figure 5. Environmental Metal Vector Potential (EMVP) scenario analysis. Scenario summaries use Qm converted to {unit_policy.get('environmental_qm_unit')} for readability; priority-metal means include Pb={fmt_caption_num(env_priority.get('Pb', {}).get('qm_mean_mg_g'))}, Cd={fmt_caption_num(env_priority.get('Cd', {}).get('qm_mean_mg_g'))}, and As={fmt_caption_num(env_priority.get('As', {}).get('qm_mean_mg_g'))} {unit_policy.get('environmental_qm_unit')}.",
        "Fig6_sensitivity_tornado": "Figure 6. Tornado summary of sensitivity analyses showing how methodological perturbations shift key model outputs.",
        "Diagnostics_01_trace_plot": "Figure S1. Trace plots for the fitted Bayesian model showing stable mixing across chains for key sampled parameters.",
        "Diagnostics_02_posterior_predictive": f"Figure S2. Posterior predictive calibration for the fitted model ({ppc.get('diagnotic_type', 'in-sample calibration').lower()}; n={ppc.get('n_observations')}, R2={ppc.get('r2')}, RMSE={ppc.get('rmse')}).",
        "Diagnostics_03_energy_plot": "Figure S3. Energy diagnostic plot used to assess HMC sampling stability and confirm no evident pathologies in the fitted chains.",
        "Diagnostics_04_pareto_k_fallback": f"Figure S4. Pareto-k diagnostic plot summarizing influential observations in approximate leave-one-out analysis; the artifact reports LOSO status as '{loso.get('status')}'.",
        "EDA_01_response_distribution": "Figure S5. Distribution of natural log-transformed adsorption capacity used for model fitting, shown to document response-scale preprocessing rather than a manuscript main result.",
        "EDA_02_categorical_factors": "Figure S6. Exploratory categorical summaries of adsorption response across study factors; retained as supplementary material because they are descriptive EDA outputs.",
        "EDA_03_correlation_matrix": "Figure S7. Exploratory correlation matrix among numeric variables and log(Qm), included in the supplement for descriptive context only.",
        "EDA_03b_binary_indicator_logQm": "Figure S8. Exploratory contrasts between binary indicators and log(Qm), included as supplementary descriptive analysis.",
        "EDA_04_study_variation": "Figure S9. Study-level variation in mean log(Qm), provided as supplementary context for between-study heterogeneity.",
        "Environmental_01_EMVP_analysis": "Figure S10. Artifact-level EMVP analysis figure used for internal review; the manuscript uses the curated EMVP scenario figure in the main text instead.",
        "Environmental_02_sensitivity_ci_width": "Figure S11. Width of environmental scenario uncertainty intervals across metals and loading conditions.",
        "Sensitivity_01_perturbation": "Figure S12. Initial perturbation-based sensitivity analysis retained as supplementary material.",
        "Sensitivity_01_delta_posterior_perturbation": "Figure S13. Posterior change under input perturbation scenarios, reported as supplementary sensitivity detail.",
        "Sensitivity_02_bootstrap": "Figure S14. Bootstrap-based stability analysis retained as supplementary methodological support.",
        "Sensitivity_02_prior_variant_forest": "Figure S15. Prior-variant sensitivity forest plot summarizing coefficient robustness to prior specification changes.",
        "Sensitivity_03_influential_studies": "Figure S16. Exploratory identification of influential studies, retained in the supplement because it supports robustness rather than the main narrative.",
        "Sensitivity_03_loso_influence_tornado": "Figure S17. Leave-one-study-out influence summary for model robustness assessment.",
        "Sensitivity_04_cluster_bootstrap_distribution": "Figure S18. Cluster-bootstrap distribution of summary effects for supplementary robustness evaluation.",
    }

    main_lines: List[str] = ["# Figure Captions (Main Manuscript)"]
    supp_lines: List[str] = ["# Figure Captions (Supplementary)"]

    if fig_inventory.empty:
        warnings.add("Figure inventory empty; figure captions generated as placeholders")
        main_lines.append(f"- {MANUAL_FLAG} No figures available")
        supp_lines.append(f"- {MANUAL_FLAG} No figures available")
    else:
        for _, row in fig_inventory.iterrows():
            stem = row["figure_stem"]
            caption = caption_map.get(stem)
            if caption is None:
                caption = f"{stem}: Artifact-derived figure requires a figure-specific caption source. {MANUAL_FLAG}"
            line = f"- {normalize_caption_text(caption)}"
            if row["group"] == "main":
                main_lines.append(line)
            else:
                supp_lines.append(line)

    table_lines = ["# Table Captions"]
    for name, cap in table_captions.items():
        table_lines.append(f"- {name}: {cap}")

    with open(captions_dir / "figure_captions_main.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(main_lines))
    with open(captions_dir / "figure_captions_supplementary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(supp_lines))
    with open(captions_dir / "table_captions.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines))


def write_readme(root: Path, conv: Dict[str, int], warnings: WarningRegistry) -> None:
    lines = [
        "Evidence-based manuscript assembly outputs",
        "",
        "Figure conversion summary:",
        f"- Source PNG discovered: {conv.get('source_png_total', 0)}",
        f"- PDF generated: {conv.get('pdf_created', 0)}",
        f"- TIFF generated (300 dpi): {conv.get('tiff_created', 0)}",
        "",
        "Warnings:",
    ]
    if warnings.unique():
        lines.extend([f"- {w}" for w in warnings.unique()])
    else:
        lines.append("- None")
    with open(root / "README_GENERATED.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    config = ProjectConfig()
    logger = setup_logging(config, "09_manuscript_materials")
    print_section_header("SCRIPT 09: EVIDENCE-BASED MANUSCRIPT ASSEMBLY", logger=logger)

    project_root = Path(__file__).parent.parent
    output_root = project_root / args.output_dir
    blueprint_path = project_root / args.blueprint
    figures_dir = project_root / "figures"

    warnings = WarningRegistry()
    provenance = ProvenanceTracker(project_root=project_root)
    layout = ensure_layout(output_root)
    blueprint = load_or_create_blueprint(blueprint_path, args.journal, warnings)

    if args.skip_upstream_figure_step:
        logger.info("Compatibility flag --skip-upstream-figure-step received; no upstream figure script is used.")

    statuses: List[ArtifactStatus] = []
    missing_inputs: List[Path] = []
    for path in required_inputs(project_root):
        exists = path.exists()
        if not exists:
            missing_inputs.append(path)
        statuses.append(
            ArtifactStatus(
                path=str(path),
                exists=exists,
                sha256=sha256_of_file(path),
                note="" if exists else warnings.add(f"Missing authoritative input: {path}"),
            )
        )

    if args.strict and missing_inputs:
        logger.error("Strict mode failed because authoritative inputs are missing.")
        return 2

    cleaned = safe_load_csv(project_root / "data_processed" / "02_cleaned_data.csv", warnings, logger)
    generate_missing_main_figure_sources(figures_dir, cleaned, warnings, logger)

    core_pngs = [figures_dir / f"{stem}.png" for stem in blueprint.get("figure_order", {}).get("main", [])]
    for pth in [pth for pth in core_pngs if not pth.exists()]:
        warnings.add(f"Missing expected main figure PNG: {pth}")

    conv, fig_inventory, exported_figure_files = export_figure_packages(figures_dir, layout, blueprint, warnings, logger)

    inv_file = layout["evidence_inventories"] / "figure_inventory.csv"
    save_dataframe(fig_inventory, inv_file, index=False)
    provenance.record(inv_file, sorted([p for p in figures_dir.glob("*.png") if p.is_file()] + [blueprint_path]), [])

    if not fig_inventory.empty:
        main_inv = layout["evidence_inventories"] / "figure_inventory_main.csv"
        supp_inv = layout["evidence_inventories"] / "figure_inventory_supplementary.csv"
        save_dataframe(fig_inventory[fig_inventory["group"] == "main"], main_inv, index=False)
        save_dataframe(fig_inventory[fig_inventory["group"] == "supplementary"], supp_inv, index=False)
        provenance.record(main_inv, [inv_file], [])
        provenance.record(supp_inv, [inv_file], [])

    metadata = safe_load_json(resolve_results_artifact(project_root, "01_data_loading_metadata.json"), warnings, logger)
    eda = safe_load_json(resolve_results_artifact(project_root, "04_eda_results.json"), warnings, logger)
    diagnostics = safe_load_json(resolve_results_artifact(project_root, "06_diagnostics_results.json"), warnings, logger)
    sensitivity_json = safe_load_json(resolve_results_artifact(project_root, "07_sensitivity_results.json"), warnings, logger)
    env_json = safe_load_json(resolve_results_artifact(project_root, "08_environmental_results.json"), warnings, logger)
    unit_verification = safe_load_json(resolve_results_artifact(project_root, "08_qm_units_log_verification.json"), warnings, logger)
    sens_perturb = safe_load_csv(resolve_results_artifact(project_root, "07_sensitivity_perturbation_summary.csv"), warnings, logger)
    sens_loso = safe_load_csv(resolve_results_artifact(project_root, "07_sensitivity_loso_summary.csv"), warnings, logger)
    sens_prior = safe_load_csv(resolve_results_artifact(project_root, "07_sensitivity_prior_summary.csv"), warnings, logger)
    emvp = safe_load_csv(resolve_results_artifact(project_root, "08_emvp_summary_table.csv"), warnings, logger)

    unit_policy = infer_qm_unit_policy(metadata, cleaned, unit_verification, env_json, warnings)
    artifact_summary = build_artifact_summary(cleaned, metadata, diagnostics, unit_policy)

    consistency = summarize_consistency(cleaned, metadata, eda, diagnostics, sensitivity_json, sens_perturb, sens_loso, warnings)
    consistency_file = layout["evidence_manifests"] / "consistency_checks.csv"
    save_dataframe(consistency, consistency_file, index=False)
    provenance.record(
        consistency_file,
        [
            project_root / "data_processed" / "02_cleaned_data.csv",
            resolve_results_artifact(project_root, "01_data_loading_metadata.json"),
            resolve_results_artifact(project_root, "04_eda_results.json"),
            resolve_results_artifact(project_root, "06_diagnostics_results.json"),
            resolve_results_artifact(project_root, "07_sensitivity_results.json"),
            resolve_results_artifact(project_root, "07_sensitivity_perturbation_summary.csv"),
            resolve_results_artifact(project_root, "07_sensitivity_loso_summary.csv"),
        ],
        [w for w in warnings.unique() if "mismatch" in w.lower() or "consistency" in w.lower()],
    )

    verification = build_verification_checks(
        cleaned,
        metadata,
        diagnostics,
        unit_verification,
        unit_policy,
        fig_inventory,
        blueprint,
        [
            resolve_results_artifact(project_root, "06_diagnostics_results.json"),
            resolve_results_artifact(project_root, "07_sensitivity_results.json"),
            resolve_results_artifact(project_root, "08_environmental_results.json"),
            resolve_results_artifact(project_root, "08_emvp_summary_table.csv"),
        ],
        warnings,
    )
    verification_file = layout["evidence_manifests"] / "lightweight_verification_checks.csv"
    save_dataframe(verification, verification_file, index=False)
    provenance.record(
        verification_file,
        [
            project_root / "data_processed" / "02_cleaned_data.csv",
            resolve_results_artifact(project_root, "01_data_loading_metadata.json"),
            resolve_results_artifact(project_root, "06_diagnostics_results.json"),
            resolve_results_artifact(project_root, "08_qm_units_log_verification.json"),
            resolve_results_artifact(project_root, "08_environmental_results.json"),
            inv_file,
            blueprint_path,
        ],
        [w for w in warnings.unique() if "figure" in w.lower() or "unit" in w.lower() or "row count" in w.lower()],
    )

    unit_policy_file = layout["evidence_manifests"] / "qm_unit_policy.json"
    save_json(unit_policy, unit_policy_file)
    provenance.record(
        unit_policy_file,
        [
            resolve_results_artifact(project_root, "01_data_loading_metadata.json"),
            project_root / "data_processed" / "02_cleaned_data.csv",
            resolve_results_artifact(project_root, "08_qm_units_log_verification.json"),
            resolve_results_artifact(project_root, "08_environmental_results.json"),
        ],
        [w for w in warnings.unique() if "unit" in w.lower()],
    )

    tables, table_caps = build_tables(
        cleaned,
        emvp,
        diagnostics,
        sensitivity_json,
        sens_perturb,
        sens_prior,
        sens_loso,
        config,
        warnings,
    )

    table_outputs: List[Path] = []
    for table_name, df in tables.items():
        out = layout["tables"] / f"{table_name}.csv"
        save_dataframe(df, out, index=False)
        table_outputs.append(out)
        provenance.record(
            out,
            [
                p
                for p in [
                    project_root / "data_processed" / "02_cleaned_data.csv",
                    resolve_results_artifact(project_root, "06_diagnostics_results.json"),
                    resolve_results_artifact(project_root, "07_sensitivity_results.json"),
                    resolve_results_artifact(project_root, "07_sensitivity_perturbation_summary.csv"),
                    resolve_results_artifact(project_root, "07_sensitivity_prior_summary.csv"),
                    resolve_results_artifact(project_root, "07_sensitivity_loso_summary.csv"),
                    resolve_results_artifact(project_root, "08_emvp_summary_table.csv"),
                    project_root / "config" / "config.yaml",
                ]
                if p.exists()
            ],
            [w for w in warnings.unique() if table_name in w],
        )

    write_captions(fig_inventory, table_caps, layout["captions"], warnings, diagnostics, env_json, unit_policy)
    provenance.record(layout["captions"] / "figure_captions_main.md", [inv_file, blueprint_path], [])
    provenance.record(layout["captions"] / "figure_captions_supplementary.md", [inv_file, blueprint_path], [])
    provenance.record(layout["captions"] / "table_captions.md", table_outputs, [])

    workflow_notes = [
        "Data source: data_processed/02_cleaned_data.csv.",
        f"Qm unit policy: raw/modeling data remain in {MODELING_QM_UNIT}; environmental scenario summaries report Qm in {ENVIRONMENTAL_QM_UNIT}.",
        "EDA source: results/04_eda_results.json.",
        "Diagnostics source: results/06_diagnostics_results.json.",
        "Sensitivity source: results/07_sensitivity_results.json and summary CSVs.",
        "Environmental source: results/08_environmental_results.json and 08_emvp_summary_table.csv.",
    ]
    skeleton_file = layout["drafts"] / "manuscript_skeleton.md"
    write_manuscript_skeleton(
        skeleton_file,
        args.journal,
        workflow_notes,
        warnings,
        required_inputs(project_root),
        artifact_summary,
        unit_policy,
    )
    provenance.record(skeleton_file, required_inputs(project_root), warnings.unique())


    evidence_manifest = {
        "generated_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "journal": args.journal,
        "strict": args.strict,
        "blueprint": str(blueprint_path),
        "authoritative_inputs": [s.__dict__ for s in statuses],
        "artifact_summary": artifact_summary,
        "unit_policy": unit_policy,
        "figure_conversion": conv,
        "warnings": warnings.unique(),
    }
    evidence_file = layout["evidence"] / "evidence_manifest.json"
    save_json(evidence_manifest, evidence_file)
    provenance.record(evidence_file, required_inputs(project_root), warnings.unique())

    provenance_file = layout["evidence"] / "provenance_map.json"
    save_json(provenance.map, provenance_file)
    provenance.record(
        provenance_file,
        [evidence_file, inv_file, consistency_file, verification_file, unit_policy_file, *exported_figure_files],
        [],
    )
    save_json(provenance.map, provenance_file)

    write_readme(layout["root"], conv, warnings)
    logger.info("Manuscript assembly completed.")
    if warnings.unique():
        logger.info("Manual review flags:")
        for warning_text in warnings.unique():
            logger.info(f"  - {warning_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())










