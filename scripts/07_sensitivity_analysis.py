"""
Script 07: Model-Based Sensitivity Analysis
===========================================

Sensitivity analyses focused on model conclusions for hierarchical Bayesian
meta-analysis of log_Qm (metal adsorption onto microplastics):
- Predictor perturbation with stochastic measurement-error style noise
- Prior/likelihood sensitivity refits
- Leave-One-Study-Out (LOSO) influence diagnostics
- Cluster bootstrap by study

Author: Manuscript authors
Date: March 2026
"""

import os

# Configure PyTensor before importing PyMC
os.environ.setdefault("PYTENSOR_FLAGS", "optimizer=fast_compile")

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils import (
    ProjectConfig,
    load_dataframe,
    print_section_header,
    save_json,
    set_random_seed,
    setup_logging,
)


def load_script05_module(logger):
    """Dynamically load Script 05 to reuse validated primary-model helpers."""
    script_path = Path(__file__).parent / "05_model_fitting.py"
    spec = importlib.util.spec_from_file_location("script05_model_fitting", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info(f"Loaded model pipeline helpers from: {script_path.name}")
    return module


def ensure_dirs(config: ProjectConfig) -> Dict[str, Path]:
    """Create output directories used by sensitivity analysis."""
    results_dir = config.get_path("results")
    figures_dir = config.get_path("figures")
    cache_dir = results_dir / "07_sensitivity_cache"
    for d in [results_dir, figures_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"results": results_dir, "figures": figures_dir, "cache": cache_dir}


def stable_hash(payload: Dict[str, Any]) -> str:
    """Build a stable short hash for cache keys."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_json_safe(obj: Any) -> Any:
    """Recursively convert nested results to JSON-safe Python types."""
    if isinstance(obj, dict):
        safe_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                safe_key = "__".join(str(part) for part in key)
            else:
                safe_key = str(key) if not isinstance(key, (str, int, float, bool)) and key is not None else key
            safe_dict[safe_key] = make_json_safe(value)
        return safe_dict
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def file_sha256(path: Path) -> str:
    """Hash file bytes for cache invalidation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Hash DataFrame contents and column order deterministically."""
    payload = {
        "columns": list(df.columns),
        "hash": pd.util.hash_pandas_object(df, index=True).astype(str).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_cache_signature(
    config: ProjectConfig,
    script05_path: Path,
    df_raw: pd.DataFrame,
    tracked_coefficients: List[str],
    run_mode: str,
) -> Dict[str, Any]:
    """Build a provenance signature for baseline and refit caches."""
    relevant_config = {
        "random_seed": config.get("random_seed", 42),
        "modeling_bayesian": config.get("modeling.bayesian", {}),
        "sensitivity_run_mode": run_mode,
        "sensitivity_perturbation": config.get("sensitivity.perturbation", {}),
        "sensitivity_prior_sensitivity": config.get("sensitivity.prior_sensitivity", {}),
        "sensitivity_loso": config.get("sensitivity.loso", {}),
        "sensitivity_bootstrap": config.get("sensitivity.bootstrap", {}),
    }
    payload = {
        "script05_sha256": file_sha256(script05_path),
        "dataset_sha256": dataframe_sha256(df_raw),
        "tracked_coefficients": tracked_coefficients,
        "relevant_config": relevant_config,
    }
    return {"signature": stable_hash(payload), "fields": payload}


def find_study_column(df: pd.DataFrame, config: ProjectConfig) -> Optional[str]:
    """Find study identifier column with config preference and fallbacks."""
    preferred = config.get("sensitivity.study_id_column", None)
    candidates = [
        preferred,
        "Study_ID_numeric",
        "Study_ID",
        "study_id",
        "Study",
        "Author_year",
        "Author_Year",
        "Reference",
    ]
    for col in candidates:
        if col and col in df.columns:
            return col
    return None


def find_predictor_column(df: pd.DataFrame, predictor_name: str) -> Optional[str]:
    """Resolve a predictor name to an existing DataFrame column (case-insensitive)."""
    if predictor_name in df.columns:
        return predictor_name
    lowered = predictor_name.lower()
    matches = [c for c in df.columns if c.lower() == lowered]
    if matches:
        return matches[0]
    fuzzy = [c for c in df.columns if lowered in c.lower()]
    return fuzzy[0] if fuzzy else None


def get_obs_like_name(idata: az.InferenceData) -> Optional[str]:
    """Get observed/log-likelihood variable name robustly."""
    for group_name in ["posterior_predictive", "log_likelihood", "observed_data"]:
        group = getattr(idata, group_name, None)
        if group is not None and hasattr(group, "data_vars"):
            vars_found = list(group.data_vars)
            if vars_found:
                for preferred in ["likelihood", "y", "y_obs"]:
                    if preferred in vars_found:
                        return preferred
                return vars_found[0]
    return None


def vector_hdi(samples: np.ndarray, hdi_prob: float = 0.95) -> Tuple[float, float]:
    """Compute HDI for 1D samples safely."""
    arr = np.asarray(samples).reshape(-1)
    if arr.size == 0:
        return np.nan, np.nan
    hdi_vals = az.hdi(arr, hdi_prob=hdi_prob)
    return float(hdi_vals[0]), float(hdi_vals[1])


def summarize_distribution(samples: np.ndarray, hdi_prob: float = 0.95) -> Dict[str, float]:
    """Summarize posterior-like samples with direction and interval diagnostics."""
    arr = np.asarray(samples).reshape(-1)
    low, high = vector_hdi(arr, hdi_prob=hdi_prob)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "hdi_low": low,
        "hdi_high": high,
        "hdi_width": float(high - low),
        "prob_gt_zero": float(np.mean(arr > 0)),
        "prob_lt_zero": float(np.mean(arr < 0)),
        "contains_zero": bool(low <= 0 <= high),
    }


def select_representative_indices(df_clean: pd.DataFrame) -> List[int]:
    """Select trace-specific representative rows without cross-trace row assumptions."""
    if len(df_clean) == 0:
        return []
    return [0, len(df_clean) // 2, len(df_clean) - 1]


def get_sampling_config(config: ProjectConfig, mode: str = "standard") -> Dict[str, Any]:
    """Sampling settings for sensitivity refits."""
    if mode == "quick":
        return {
            "chains": 2,
            "draws": 800,
            "tune": 800,
            "target_accept": 0.90,
            "cores": 1,
            "max_treedepth": 10,
        }
    if mode == "publication":
        return {
            "chains": 4,
            "draws": 2000,
            "tune": 2000,
            "target_accept": 0.95,
            "cores": min(4, int(os.cpu_count() or 1)),
            "max_treedepth": 12,
        }
    return {
        "chains": 4,
        "draws": 1500,
        "tune": 1500,
        "target_accept": 0.92,
        "cores": min(4, int(os.cpu_count() or 1)),
        "max_treedepth": 11,
    }


def check_convergence_diagnostics(
    trace: az.InferenceData, logger, run_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """Check MCMC convergence diagnostics post-sampling."""
    diags = {"run_id": run_id, "timestamp": pd.Timestamp.now().isoformat()}
    all_pass = True

    try:
        rhat_vals = az.rhat(trace)
        rhat_max = float(rhat_vals.to_array().max().values)
        diags["rhat_max"] = rhat_max
        diags["rhat_pass"] = rhat_max <= 1.01
        if not diags["rhat_pass"]:
            logger.warning(f"  Convergence FAIL: rhat_max={rhat_max:.4f} > 1.01")
            all_pass = False
        else:
            logger.info(f"  Convergence check: rhat_max={rhat_max:.4f} [PASS]")
    except Exception as e:
        logger.warning(f"  Could not compute rhat: {e}")
        diags["rhat_error"] = str(e)
        all_pass = False

    try:
        ess_bulk = az.ess(trace, method="bulk")
        ess_bulk_min = float(ess_bulk.to_array().min().values)
        diags["ess_bulk_min"] = ess_bulk_min
        diags["ess_bulk_pass"] = ess_bulk_min >= 400
        if not diags["ess_bulk_pass"]:
            logger.warning(f"  Convergence FAIL: ess_bulk_min={ess_bulk_min:.0f} < 400")
            all_pass = False
        else:
            logger.info(f"  ESS Bulk: min={ess_bulk_min:.0f} [PASS]")
    except Exception as e:
        logger.warning(f"  Could not compute ESS bulk: {e}")
        diags["ess_bulk_error"] = str(e)
        all_pass = False

    try:
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = int(trace.sample_stats.diverging.sum().values)
            diags["divergences"] = divergences
            diags["divergences_pass"] = divergences == 0
            if not diags["divergences_pass"]:
                logger.warning(f"  Convergence FAIL: {divergences} divergences detected")
                all_pass = False
            else:
                logger.info(f"  Divergences: {divergences} [PASS]")
    except Exception as e:
        logger.warning(f"  Could not check divergences: {e}")
        diags["divergences_error"] = str(e)
        all_pass = False

    diags["overall_pass"] = all_pass
    return all_pass, diags


def prepare_primary_model_inputs(
    df_raw: pd.DataFrame,
    script05,
    logger,
) -> Dict[str, Any]:
    """
    Prepare the exact primary-model inputs used by Script 05.

    This intentionally mirrors the validated primary model:
    log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)
    """
    df_model, _, _ = script05.prepare_modeling_data(df_raw.copy(), logger)

    required_predictors = {
        "metal": [col for col in df_model.columns if col == "Metal_raw"]
        or [col for col in df_model.columns if col == "Metal_raw_encoded"],
        "polymer": [col for col in df_model.columns if col == "ReT"]
        or [col for col in df_model.columns if col == "ReT_encoded"],
        "aging": [col for col in df_model.columns if col == "AgS"]
        or [col for col in df_model.columns if col == "AgS_encoded"],
        "Temp": [col for col in df_model.columns if col == "Temp"],
        "Study": [col for col in df_model.columns if col == "Study_ID_local"],
    }
    predictors = {k: (v[0] if v else None) for k, v in required_predictors.items()}

    core_keys = ["metal", "polymer", "aging", "Temp", "Study"]
    missing_core_terms = [k for k in core_keys if predictors.get(k) is None]
    if missing_core_terms:
        raise ValueError(
            "Primary core model missing required predictors: " + ", ".join(missing_core_terms)
        )

    core_required_cols = ["log_Qm"] + [predictors[k] for k in core_keys]
    df_clean = script05._log_model_row_loss(df_model, core_required_cols, "Primary model", logger)
    if len(df_clean) < 10:
        raise ValueError(f"Insufficient complete cases for primary refit: n={len(df_clean)}")

    df_scaled = df_clean.copy()
    temp_col = predictors["Temp"]
    temp_mean = df_clean[temp_col].mean()
    temp_sd = df_clean[temp_col].std()
    df_scaled[temp_col] = 0.0 if temp_sd <= 0 else (df_clean[temp_col] - temp_mean) / temp_sd
    scaling_params = {temp_col: {"mean": float(temp_mean), "std": float(temp_sd)}}

    metal_codes, metal_levels, metal_reference = script05._encode_categorical_with_reference(
        df_scaled, predictors["metal"], logger
    )
    polymer_codes, polymer_levels, polymer_reference = script05._encode_categorical_with_reference(
        df_scaled, predictors["polymer"], logger
    )
    aging_codes, aging_levels, aging_reference = script05._encode_categorical_with_reference(
        df_scaled, predictors["aging"], logger
    )

    coefficient_labels = {
        "beta_metal": [f"beta_metal[{level}_vs_{metal_reference}]" for level in metal_levels[1:]],
        "beta_polymer": [f"beta_polymer[{level}_vs_{polymer_reference}]" for level in polymer_levels[1:]],
        "beta_aging": [f"beta_aging[{level}_vs_{aging_reference}]" for level in aging_levels[1:]],
        "beta_Temp": ["beta_Temp"],
        "sigma_study": ["sigma_study"],
        "sigma_residual": ["sigma_residual"],
    }

    return {
        "df_model": df_model,
        "df_clean": df_clean,
        "df_scaled": df_scaled,
        "predictors": predictors,
        "core_keys": core_keys,
        "scaling_params": scaling_params,
        "codes": {
            "metal": metal_codes,
            "polymer": polymer_codes,
            "aging": aging_codes,
        },
        "levels": {
            "metal": metal_levels,
            "polymer": polymer_levels,
            "aging": aging_levels,
        },
        "reference_levels": {
            "metal": metal_reference,
            "polymer": polymer_reference,
            "aging": aging_reference,
        },
        "coefficient_labels": coefficient_labels,
        "formula": "log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)",
    }


def get_tracked_coefficients(config: ProjectConfig) -> List[str]:
    """Tracked terms should match manuscript-critical primary-model conclusions."""
    return config.get(
        "sensitivity.coefficients_to_track",
        ["beta_metal", "beta_polymer", "beta_aging", "beta_Temp", "sigma_study", "sigma_residual"],
    )

def validate_trace_against_primary_context(
    trace: az.InferenceData,
    summary: Optional[Dict[str, Any]],
    primary_context: Dict[str, Any],
    tracked_coefficients: List[str],
) -> Tuple[bool, List[str]]:
    """Apply conservative checks before reusing a saved Script 05 trace artifact."""
    reasons = []
    if summary is None:
        reasons.append("missing primary_model_summary.json")
        return False, reasons

    if summary.get("formula") != primary_context["formula"]:
        reasons.append("formula mismatch")
    if int(summary.get("n_observations", -1)) != int(len(primary_context["df_clean"])):
        reasons.append("n_observations mismatch")

    summary_refs = summary.get("reference_levels", {})
    for key, expected in primary_context["reference_levels"].items():
        if summary_refs.get(key) != expected:
            reasons.append(f"reference level mismatch for {key}")

    for coeff in tracked_coefficients:
        if coeff not in trace.posterior.data_vars:
            reasons.append(f"missing posterior variable: {coeff}")

    return len(reasons) == 0, reasons


def fit_primary_model_with_options(
    df_raw: pd.DataFrame,
    config: ProjectConfig,
    script05,
    logger,
    prior_scale: float = 1.0,
    likelihood_family: str = "normal",
    seed_offset: int = 0,
    run_mode: str = "standard",
    auto_retry: bool = True,
) -> Dict[str, Any]:
    """Fit the validated Script 05 primary model with optional prior/likelihood sensitivity."""
    prepared = prepare_primary_model_inputs(df_raw, script05, logger)
    df_scaled = prepared["df_scaled"]
    predictors = prepared["predictors"]

    intercept_mu = float(config.get("modeling.bayesian.priors.intercept.mu", 0))
    intercept_sigma_cfg = float(config.get("modeling.bayesian.priors.intercept.sigma", 5))
    beta_mu = float(config.get("modeling.bayesian.priors.fixed_effects.mu", 0))
    beta_sigma_cfg = float(config.get("modeling.bayesian.priors.fixed_effects.sigma", 2))
    sigma_study_prior = float(config.get("modeling.bayesian.priors.random_effects_sd.sigma", 2))
    sigma_resid_prior = float(config.get("modeling.bayesian.priors.residual_sd.sigma", 1))

    intercept_sigma = min(intercept_sigma_cfg, 5.0)
    beta_sigma = min(beta_sigma_cfg, 2.0) * prior_scale

    with pm.Model() as model:
        y_obs = df_scaled["log_Qm"].values
        intercept = pm.Normal("intercept", mu=intercept_mu, sigma=intercept_sigma)

        metal_effect = script05._build_reference_effect(
            "beta_metal",
            prepared["codes"]["metal"],
            prepared["levels"]["metal"],
            beta_mu,
            beta_sigma,
        )
        polymer_effect = script05._build_reference_effect(
            "beta_polymer",
            prepared["codes"]["polymer"],
            prepared["levels"]["polymer"],
            beta_mu,
            beta_sigma,
        )
        aging_effect = script05._build_reference_effect(
            "beta_aging",
            prepared["codes"]["aging"],
            prepared["levels"]["aging"],
            beta_mu,
            beta_sigma,
        )
        beta_Temp = pm.Normal("beta_Temp", mu=beta_mu, sigma=beta_sigma)

        study_idx = df_scaled[predictors["Study"]].astype(int).values
        n_studies = int(df_scaled[predictors["Study"]].max()) + 1
        sigma_study = pm.HalfNormal("sigma_study", sigma=sigma_study_prior)
        study_effect = pm.Normal("study_effect", mu=0, sigma=sigma_study, shape=n_studies)

        mu = intercept + metal_effect + polymer_effect + aging_effect
        mu += beta_Temp * df_scaled[predictors["Temp"]].values
        mu += study_effect[study_idx]

        pm.Deterministic("mu", mu)
        sigma_residual = pm.HalfNormal("sigma_residual", sigma=sigma_resid_prior)

        if likelihood_family.lower() == "student_t":
            nu_minus_two = pm.Exponential("nu_minus_two", lam=1 / 30)
            nu = pm.Deterministic("nu", nu_minus_two + 2)
            pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma_residual, observed=y_obs)
        else:
            pm.Normal("likelihood", mu=mu, sigma=sigma_residual, observed=y_obs)

        sample_cfg = get_sampling_config(config, mode=run_mode)
        logger.info(
            f"Sampling config (mode={run_mode}): chains={sample_cfg['chains']}, "
            f"draws={sample_cfg['draws']}, tune={sample_cfg['tune']}, "
            f"target_accept={sample_cfg['target_accept']}, cores={sample_cfg['cores']}"
        )
        seed_used = int(config.get("random_seed", 42)) + seed_offset
        logger.info(f"Random seed: {seed_used}")

        trace = pm.sample(
            draws=sample_cfg["draws"],
            tune=sample_cfg["tune"],
            chains=sample_cfg["chains"],
            cores=sample_cfg["cores"],
            target_accept=sample_cfg["target_accept"],
            max_treedepth=sample_cfg.get("max_treedepth", 10),
            return_inferencedata=True,
            random_seed=seed_used,
            progressbar=False,
        )

        with model:
            pm.sample_posterior_predictive(
                trace,
                extend_inferencedata=True,
                random_seed=seed_used,
                progressbar=False,
            )
            pm.compute_log_likelihood(trace)

    conv_pass, conv_diags = check_convergence_diagnostics(trace, logger, f"seed_{seed_used}")
    execution = {
        "status": "fresh",
        "seed_used": seed_used,
        "requested_run_mode": run_mode,
        "executed_run_mode": run_mode,
        "auto_retry_triggered": False,
    }

    if not conv_pass and auto_retry and run_mode != "publication":
        logger.warning("Convergence check failed; auto-retrying with stricter settings")
        retry_result = fit_primary_model_with_options(
            df_raw=df_raw,
            config=config,
            script05=script05,
            logger=logger,
            prior_scale=prior_scale,
            likelihood_family=likelihood_family,
            seed_offset=seed_offset + 1,
            run_mode="publication",
            auto_retry=False,
        )
        retry_result["execution"]["auto_retry_triggered"] = True
        retry_result["execution"]["initial_run_mode"] = run_mode
        retry_result["execution"]["initial_seed"] = seed_used
        return retry_result

    return {
        "trace": trace,
        "df_clean": prepared["df_clean"],
        "predictors": predictors,
        "scaling_params": prepared["scaling_params"],
        "reference_levels": prepared["reference_levels"],
        "coefficient_labels": prepared["coefficient_labels"],
        "convergence_diagnostics": conv_diags,
        "convergence_pass": conv_pass,
        "execution": execution,
    }


def extract_model_conclusion_metrics(
    trace: az.InferenceData,
    tracked_coefficients: List[str],
    coefficient_labels: Optional[Dict[str, List[str]]],
    representative_indices: List[int],
    include_information_criteria: bool = True,
) -> Dict[str, Any]:
    """Extract posterior summaries, predictive summaries, and optional information criteria."""
    posterior = trace.posterior

    coeff_metrics = {}
    for coeff in tracked_coefficients:
        if coeff not in posterior.data_vars:
            continue
        vals = posterior[coeff].values
        if vals.ndim <= 2:
            coeff_metrics[coeff] = summarize_distribution(vals.reshape(-1))
            continue

        n_terms = vals.shape[-1]
        labels = (coefficient_labels or {}).get(coeff, [f"{coeff}[{i}]" for i in range(n_terms)])
        flat = vals.reshape(-1, n_terms)
        for idx in range(n_terms):
            coeff_metrics[labels[idx]] = summarize_distribution(flat[:, idx])

    obs_name = get_obs_like_name(trace)
    pooled_summary = {}
    representative = {}

    if obs_name and hasattr(trace, "posterior_predictive") and obs_name in trace.posterior_predictive.data_vars:
        pp = trace.posterior_predictive[obs_name].values
        pp_flat = pp.reshape(-1, pp.shape[-1])
        pooled_summary = summarize_distribution(pp_flat.mean(axis=1))
        for ridx in representative_indices:
            if ridx < pp_flat.shape[1]:
                representative[f"row_{ridx}"] = summarize_distribution(pp_flat[:, ridx])

    loo_result = None
    waic_result = None
    pareto_summary = {}
    if include_information_criteria:
        try:
            loo = az.loo(trace, pointwise=True)
            loo_result = {
                "elpd_loo": float(loo.elpd_loo),
                "se": float(loo.se),
                "p_loo": float(loo.p_loo),
            }
            if hasattr(loo, "pareto_k"):
                k = loo.pareto_k.values
                pareto_summary = {
                    "good_lt_0_5": int(np.sum(k < 0.5)),
                    "ok_0_5_to_0_7": int(np.sum((k >= 0.5) & (k < 0.7))),
                    "bad_0_7_to_1": int(np.sum((k >= 0.7) & (k < 1.0))),
                    "very_bad_ge_1": int(np.sum(k >= 1.0)),
                    "max": float(np.max(k)),
                }
        except Exception:
            loo_result = None

        try:
            waic = az.waic(trace, pointwise=True)
            waic_result = {
                "elpd_waic": float(waic.elpd_waic),
                "se": float(waic.se),
                "p_waic": float(waic.p_waic),
            }
        except Exception:
            waic_result = None

    return {
        "coefficients": coeff_metrics,
        "pooled_log_qm": pooled_summary,
        "representative_predictions": representative,
        "loo": loo_result,
        "waic": waic_result,
        "pareto_k": pareto_summary,
    }


def compute_delta_vs_baseline(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    compare_information_criteria: bool = True,
) -> Dict[str, Any]:
    """Compute delta metrics for posterior summaries and, when valid, information criteria."""
    out = {"coefficients": {}, "pooled_log_qm": {}, "loo": {}, "waic": {}}

    base_coeff = baseline.get("coefficients", {})
    cand_coeff = candidate.get("coefficients", {})
    for coeff, bvals in base_coeff.items():
        if coeff not in cand_coeff:
            continue
        cvals = cand_coeff[coeff]
        delta_mean = cvals["mean"] - bvals["mean"]
        denom = abs(bvals["mean"]) if abs(bvals["mean"]) > 1e-8 else np.nan
        out["coefficients"][coeff] = {
            "delta_mean": float(delta_mean),
            "delta_mean_pct": float(100.0 * delta_mean / denom) if np.isfinite(denom) else np.nan,
            "delta_hdi_width": float(cvals["hdi_width"] - bvals["hdi_width"]),
            "baseline_sign": int(np.sign(bvals["mean"])),
            "candidate_sign": int(np.sign(cvals["mean"])),
            "sign_changed": bool(np.sign(cvals["mean"]) != np.sign(bvals["mean"])),
            "baseline_contains_zero": bool(bvals.get("contains_zero", False)),
            "candidate_contains_zero": bool(cvals.get("contains_zero", False)),
            "contains_zero_changed": bool(cvals.get("contains_zero", False) != bvals.get("contains_zero", False)),
            "baseline_prob_gt_zero": float(bvals.get("prob_gt_zero", np.nan)),
            "candidate_prob_gt_zero": float(cvals.get("prob_gt_zero", np.nan)),
        }

    if baseline.get("pooled_log_qm") and candidate.get("pooled_log_qm"):
        b = baseline["pooled_log_qm"]
        c = candidate["pooled_log_qm"]
        d = c["mean"] - b["mean"]
        denom = abs(b["mean"]) if abs(b["mean"]) > 1e-8 else np.nan
        out["pooled_log_qm"] = {
            "delta_mean": float(d),
            "delta_mean_pct": float(100.0 * d / denom) if np.isfinite(denom) else np.nan,
            "delta_hdi_width": float(c["hdi_width"] - b["hdi_width"]),
        }

    if compare_information_criteria and baseline.get("loo") and candidate.get("loo"):
        delta_elpd = candidate["loo"]["elpd_loo"] - baseline["loo"]["elpd_loo"]
        se = np.sqrt(baseline["loo"]["se"] ** 2 + candidate["loo"]["se"] ** 2)
        out["loo"] = {"delta_elpd_loo": float(delta_elpd), "delta_se_approx": float(se)}

    if compare_information_criteria and baseline.get("waic") and candidate.get("waic"):
        delta_elpd = candidate["waic"]["elpd_waic"] - baseline["waic"]["elpd_waic"]
        se = np.sqrt(baseline["waic"]["se"] ** 2 + candidate["waic"]["se"] ** 2)
        out["waic"] = {"delta_elpd_waic": float(delta_elpd), "delta_se_approx": float(se)}

    return out


def load_existing_primary_trace(config: ProjectConfig, logger) -> Tuple[Optional[az.InferenceData], Optional[Dict[str, Any]]]:
    """Load existing primary model trace artifact and summary if available."""
    models_dir = config.get_path("models")
    nc_path = models_dir / "primary_model_trace.nc"
    json_path = models_dir / "primary_model_trace.json"
    summary_path = models_dir / "primary_model_summary.json"
    summary = None

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed loading primary model summary: {exc}")

    if nc_path.exists():
        try:
            logger.info(f"Loading existing baseline trace: {nc_path.name}")
            return az.from_netcdf(str(nc_path)), summary
        except Exception as exc:
            logger.warning(f"Failed loading NetCDF baseline trace: {exc}")

    if json_path.exists():
        try:
            logger.info(f"Loading existing baseline trace: {json_path.name}")
            return az.from_json(str(json_path)), summary
        except Exception as exc:
            logger.warning(f"Failed loading JSON baseline trace: {exc}")

    return None, summary


def run_cached_refit(
    refit_id: str,
    cache_dir: Path,
    logger,
    fit_callable,
    cache_signature: str,
    diagnostics_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load refit metrics from cache or run fit and checkpoint results."""
    cache_file = cache_dir / f"{refit_id}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("cache_signature") == cache_signature:
            logger.info(f"Cache hit: {cache_file.name}")
            cached["cache_status"] = "hit"
            return cached
        logger.info(f"Cache stale: {cache_file.name}; signature mismatch, rebuilding")

    logger.info(f"Cache miss: {cache_file.name}; running refit")
    result = fit_callable()
    result["cache_signature"] = cache_signature
    result["cache_status"] = "miss"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if diagnostics_csv_path and "convergence_diagnostics" in result:
        diag_df = pd.DataFrame([result["convergence_diagnostics"]])
        mode = "a" if diagnostics_csv_path.exists() else "w"
        diag_df.to_csv(diagnostics_csv_path, mode=mode, header=(mode == "w"), index=False)

    return result

def run_parameter_perturbation(
    df_raw: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    config: ProjectConfig,
    script05,
    cache_dir: Path,
    tracked_coeffs: List[str],
    logger,
    cache_signature: str,
    run_mode: str = "standard",
) -> Dict[str, Any]:
    """Model-based continuous predictor perturbation aligned to the validated primary model."""
    if not config.get("sensitivity.perturbation.enabled", True):
        logger.info("Perturbation analysis skipped: disabled in config")
        return {"status": "skipped: disabled"}

    requested_variables = config.get("sensitivity.perturbation.variables", ["Temp"])
    valid_continuous = {"Temp"}
    variables = []
    skipped_variables = []
    for variable in requested_variables:
        if variable in valid_continuous:
            variables.append(variable)
        else:
            skipped_variables.append(variable)
            logger.info(
                f"Perturbation skipped for {variable}: not a continuous term in the validated primary model"
            )

    if not variables:
        return {
            "status": "skipped: no valid perturbation targets",
            "requested_variables": requested_variables,
            "skipped_variables": skipped_variables,
        }

    n_reps = int(config.get("sensitivity.perturbation.n_replicates", 3))
    noise_fraction = float(config.get("sensitivity.perturbation.noise_fraction", 0.05))
    if noise_fraction <= 0:
        frac = config.get("sensitivity.perturbation.perturbation_fraction", [0.9, 1.1])
        if isinstance(frac, list) and len(frac) == 2:
            noise_fraction = abs(float(frac[1]) - float(frac[0])) / 2.0
        else:
            noise_fraction = 0.05

    records: List[Dict[str, Any]] = []
    for variable in variables:
        col = find_predictor_column(df_raw, variable)
        if col is None:
            logger.info(f"Perturbation skipped for {variable}: column not found")
            continue
        if df_raw[col].dropna().empty:
            logger.info(f"Perturbation skipped for {variable}: no non-missing values")
            continue

        col_sd = float(df_raw[col].std()) if float(df_raw[col].std()) > 0 else 1.0
        col_min = float(df_raw[col].min())
        col_max = float(df_raw[col].max())

        for rep in range(n_reps):
            rng = np.random.default_rng(int(config.get("random_seed", 42)) + rep)
            noisy = df_raw.copy()
            eps = rng.normal(0, noise_fraction * col_sd, size=len(noisy))
            noisy[col] = np.clip(noisy[col].astype(float) + eps, col_min, col_max)

            cache_key = stable_hash(
                {
                    "analysis": "perturbation",
                    "signature": cache_signature,
                    "variable": variable,
                    "column": col,
                    "rep": rep,
                    "noise_fraction": noise_fraction,
                }
            )
            diag_csv = cache_dir / "diagnostics_perturbation.csv"

            def _fit_and_extract():
                fit = fit_primary_model_with_options(
                    noisy,
                    config,
                    script05,
                    logger,
                    prior_scale=1.0,
                    likelihood_family="normal",
                    seed_offset=1000 + rep,
                    run_mode=run_mode,
                    auto_retry=True,
                )
                metrics = extract_model_conclusion_metrics(
                    fit["trace"],
                    tracked_coeffs,
                    fit.get("coefficient_labels"),
                    select_representative_indices(fit["df_clean"]),
                    include_information_criteria=True,
                )
                return {
                    "metrics": metrics,
                    "convergence_diagnostics": fit.get("convergence_diagnostics", {}),
                    "convergence_pass": fit.get("convergence_pass", False),
                    "execution": fit.get("execution", {}),
                }

            candidate = run_cached_refit(
                f"perturb_{cache_key}",
                cache_dir,
                logger,
                _fit_and_extract,
                cache_signature=cache_signature,
                diagnostics_csv_path=diag_csv,
            )
            delta = compute_delta_vs_baseline(baseline_metrics, candidate["metrics"], compare_information_criteria=True)

            rec = {
                "variable": variable,
                "column": col,
                "replicate": rep,
                "cache_status": candidate.get("cache_status", "unknown"),
                "auto_retry_triggered": bool(candidate.get("execution", {}).get("auto_retry_triggered", False)),
                "convergence_pass": candidate.get("convergence_pass", False),
                "delta_pooled_mean": delta.get("pooled_log_qm", {}).get("delta_mean", np.nan),
                "delta_pooled_mean_pct": delta.get("pooled_log_qm", {}).get("delta_mean_pct", np.nan),
                "delta_pooled_hdi_width": delta.get("pooled_log_qm", {}).get("delta_hdi_width", np.nan),
                "delta_elpd_loo": delta.get("loo", {}).get("delta_elpd_loo", np.nan),
                "delta_elpd_loo_se": delta.get("loo", {}).get("delta_se_approx", np.nan),
            }
            for coeff, coeff_delta in delta.get("coefficients", {}).items():
                rec[f"delta_{coeff}_mean"] = coeff_delta.get("delta_mean", np.nan)
                rec[f"delta_{coeff}_pct"] = coeff_delta.get("delta_mean_pct", np.nan)
                rec[f"delta_{coeff}_hdi_width"] = coeff_delta.get("delta_hdi_width", np.nan)
                rec[f"sign_changed_{coeff}"] = coeff_delta.get("sign_changed", False)
                rec[f"contains_zero_changed_{coeff}"] = coeff_delta.get("contains_zero_changed", False)
            records.append(rec)

    if not records:
        return {"status": "skipped: no valid perturbation scenarios", "skipped_variables": skipped_variables}

    df_records = pd.DataFrame(records)
    n_pass = int((df_records["convergence_pass"] == True).sum())
    n_fail = len(df_records) - n_pass
    n_cached = int((df_records["cache_status"] == "hit").sum())
    n_fresh = len(df_records) - n_cached
    n_retried = int((df_records["auto_retry_triggered"] == True).sum())

    logger.info(
        f"Perturbation analysis: {n_pass}/{len(df_records)} runs passed convergence; "
        f"fresh={n_fresh}, cached={n_cached}, fresh_auto_retries={n_retried}"
    )

    numeric_cols = df_records.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "replicate"]
    summary_stats = {}
    if numeric_cols:
        summary_stats = df_records.groupby("variable")[numeric_cols].agg(["mean", "std", "max", "min"]).to_dict()

    return {
        "status": "completed",
        "n_runs": int(len(df_records)),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_cached": n_cached,
        "n_fresh": n_fresh,
        "n_fresh_auto_retried": n_retried,
        "noise_fraction": noise_fraction,
        "requested_variables": requested_variables,
        "skipped_variables": skipped_variables,
        "records": records,
        "summary": summary_stats,
    }


def run_prior_sensitivity(
    df_raw: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    config: ProjectConfig,
    script05,
    cache_dir: Path,
    tracked_coeffs: List[str],
    logger,
    cache_signature: str,
    run_mode: str = "standard",
) -> Dict[str, Any]:
    """Prior and likelihood sensitivity through exact-primary-model refits."""
    if not config.get("sensitivity.prior_sensitivity.enabled", True):
        logger.info("Prior sensitivity skipped: disabled in config")
        return {"status": "skipped: disabled"}

    scales = config.get("sensitivity.prior_sensitivity.scales", [0.5, 2.0])
    variants = [
        {"name": "stronger_regularization", "prior_scale": float(scales[0]) if len(scales) > 0 else 0.5, "likelihood": "normal"},
        {"name": "weaker_regularization", "prior_scale": float(scales[-1]) if len(scales) > 0 else 2.0, "likelihood": "normal"},
        {"name": "robust_student_t", "prior_scale": 1.0, "likelihood": "student_t"},
    ]

    rows = []
    coeff_rankings = {}
    diag_csv = cache_dir / "diagnostics_prior_sensitivity.csv"

    for i, variant in enumerate(variants):
        cache_key = stable_hash({"analysis": "prior", "signature": cache_signature, **variant})

        def _fit_and_extract():
            fit = fit_primary_model_with_options(
                df_raw,
                config,
                script05,
                logger,
                prior_scale=float(variant["prior_scale"]),
                likelihood_family=str(variant["likelihood"]),
                seed_offset=2000 + i,
                run_mode=run_mode,
                auto_retry=True,
            )
            metrics = extract_model_conclusion_metrics(
                fit["trace"],
                tracked_coeffs,
                fit.get("coefficient_labels"),
                select_representative_indices(fit["df_clean"]),
                include_information_criteria=True,
            )
            return {
                "metrics": metrics,
                "convergence_diagnostics": fit.get("convergence_diagnostics", {}),
                "convergence_pass": fit.get("convergence_pass", False),
                "execution": fit.get("execution", {}),
            }

        candidate = run_cached_refit(
            f"prior_{cache_key}",
            cache_dir,
            logger,
            _fit_and_extract,
            cache_signature=cache_signature,
            diagnostics_csv_path=diag_csv,
        )
        delta = compute_delta_vs_baseline(baseline_metrics, candidate["metrics"], compare_information_criteria=True)

        metric_row = {
            "variant": variant["name"],
            "prior_scale": variant["prior_scale"],
            "likelihood": variant["likelihood"],
            "cache_status": candidate.get("cache_status", "unknown"),
            "auto_retry_triggered": bool(candidate.get("execution", {}).get("auto_retry_triggered", False)),
            "convergence_pass": candidate.get("convergence_pass", False),
            "delta_pooled_mean": delta.get("pooled_log_qm", {}).get("delta_mean", np.nan),
            "delta_pooled_hdi_width": delta.get("pooled_log_qm", {}).get("delta_hdi_width", np.nan),
            "delta_elpd_loo": delta.get("loo", {}).get("delta_elpd_loo", np.nan),
            "delta_elpd_loo_se": delta.get("loo", {}).get("delta_se_approx", np.nan),
        }

        coeff_means = {}
        sign_changes = {}
        contains_zero_changes = {}
        for coeff, cdelta in delta.get("coefficients", {}).items():
            metric_row[f"delta_{coeff}_mean"] = cdelta.get("delta_mean", np.nan)
            metric_row[f"delta_{coeff}_hdi_width"] = cdelta.get("delta_hdi_width", np.nan)
            metric_row[f"sign_changed_{coeff}"] = cdelta.get("sign_changed", False)
            metric_row[f"contains_zero_changed_{coeff}"] = cdelta.get("contains_zero_changed", False)

            cmetrics = candidate["metrics"].get("coefficients", {}).get(coeff, {})
            coeff_means[coeff] = cmetrics.get("mean", np.nan)
            sign_changes[coeff] = cdelta.get("sign_changed", False)
            contains_zero_changes[coeff] = cdelta.get("contains_zero_changed", False)

        ordered = sorted(
            [(k, abs(v)) for k, v in coeff_means.items() if np.isfinite(v)],
            key=lambda x: x[1],
            reverse=True,
        )
        coeff_rankings[variant["name"]] = [k for k, _ in ordered]
        metric_row["any_sign_change"] = bool(any(sign_changes.values()))
        metric_row["any_zero_overlap_change"] = bool(any(contains_zero_changes.values()))
        rows.append(metric_row)

    if not rows:
        return {"status": "skipped: no prior variants executed"}

    df_variants = pd.DataFrame(rows)
    n_pass = int((df_variants["convergence_pass"] == True).sum())
    n_fail = len(df_variants) - n_pass
    n_cached = int((df_variants["cache_status"] == "hit").sum())
    n_retried = int((df_variants["auto_retry_triggered"] == True).sum())
    stability = {
        "sign_stable_all_variants": bool(all(not r.get("any_sign_change", False) for r in rows)),
        "zero_overlap_stable_all_variants": bool(all(not r.get("any_zero_overlap_change", False) for r in rows)),
        "ranking_by_abs_mean": coeff_rankings,
    }

    logger.info(
        f"Prior sensitivity: {n_pass}/{len(df_variants)} variants passed convergence; "
        f"cached={n_cached}, fresh_auto_retries={n_retried}, sign_stable={stability['sign_stable_all_variants']}"
    )

    return {
        "status": "completed",
        "variants": rows,
        "stability": stability,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_cached": n_cached,
        "n_fresh_auto_retried": n_retried,
    }

def run_loso_influence(
    df_raw: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    config: ProjectConfig,
    script05,
    cache_dir: Path,
    tracked_coeffs: List[str],
    logger,
    cache_signature: str,
    run_mode: str = "standard",
) -> Dict[str, Any]:
    """Leave-One-Study-Out influence analysis across all studies unless explicitly screening."""
    if not config.get("sensitivity.loso.enabled", True):
        logger.info("LOSO analysis skipped: disabled in config")
        return {"status": "skipped: disabled"}

    study_col = find_study_column(df_raw, config)
    if study_col is None:
        return {"status": "skipped: study_id column not found"}

    counts = df_raw[study_col].value_counts(dropna=True)
    if len(counts) < 3:
        return {"status": "skipped: too few studies"}

    if run_mode == "quick":
        max_studies = int(config.get("sensitivity.loso.max_studies", min(len(counts), 10)))
        selected = counts.head(max_studies).index.tolist()
        loso_scope = "screening_subset"
    else:
        selected = counts.index.tolist()
        loso_scope = "all_studies"

    rows = []
    diag_csv = cache_dir / "diagnostics_loso.csv"
    for i, sid in enumerate(selected):
        reduced = df_raw[df_raw[study_col] != sid].copy()
        if len(reduced) < 30:
            logger.info(f"LOSO skip study {sid}: insufficient rows after exclusion")
            continue

        cache_key = stable_hash(
            {
                "analysis": "loso",
                "signature": cache_signature,
                "study_col": study_col,
                "study": str(sid),
            }
        )

        def _fit_and_extract():
            fit = fit_primary_model_with_options(
                reduced,
                config,
                script05,
                logger,
                prior_scale=1.0,
                likelihood_family="normal",
                seed_offset=3000 + i,
                run_mode=run_mode,
                auto_retry=True,
            )
            metrics = extract_model_conclusion_metrics(
                fit["trace"],
                tracked_coeffs,
                fit.get("coefficient_labels"),
                select_representative_indices(fit["df_clean"]),
                include_information_criteria=True,
            )
            return {
                "metrics": metrics,
                "convergence_diagnostics": fit.get("convergence_diagnostics", {}),
                "convergence_pass": fit.get("convergence_pass", False),
                "execution": fit.get("execution", {}),
            }

        candidate = run_cached_refit(
            f"loso_{cache_key}",
            cache_dir,
            logger,
            _fit_and_extract,
            cache_signature=cache_signature,
            diagnostics_csv_path=diag_csv,
        )
        delta = compute_delta_vs_baseline(baseline_metrics, candidate["metrics"], compare_information_criteria=False)

        coeff_deltas = [abs(v.get("delta_mean", 0.0)) for v in delta.get("coefficients", {}).values()]
        pooled_delta = abs(delta.get("pooled_log_qm", {}).get("delta_mean", 0.0))
        influence_score = max(coeff_deltas + [pooled_delta]) if (coeff_deltas or np.isfinite(pooled_delta)) else 0.0

        row = {
            "study_col": study_col,
            "study_id": str(sid),
            "n_rows_study": int(counts.loc[sid]),
            "cache_status": candidate.get("cache_status", "unknown"),
            "auto_retry_triggered": bool(candidate.get("execution", {}).get("auto_retry_triggered", False)),
            "convergence_pass": candidate.get("convergence_pass", False),
            "delta_pooled_mean": delta.get("pooled_log_qm", {}).get("delta_mean", np.nan),
            "delta_pooled_hdi_width": delta.get("pooled_log_qm", {}).get("delta_hdi_width", np.nan),
            "influence_score": float(influence_score),
            "candidate_elpd_loo": candidate["metrics"].get("loo", {}).get("elpd_loo", np.nan),
            "candidate_waic": candidate["metrics"].get("waic", {}).get("elpd_waic", np.nan),
            "information_criteria_note": "LOSO information criteria are reported only as reduced-dataset exploratory values and are not directly comparable to the full-data baseline.",
        }
        for coeff, coeff_delta in delta.get("coefficients", {}).items():
            row[f"delta_{coeff}_mean"] = coeff_delta.get("delta_mean", np.nan)
            row[f"delta_{coeff}_hdi_width"] = coeff_delta.get("delta_hdi_width", np.nan)
            row[f"sign_changed_{coeff}"] = coeff_delta.get("sign_changed", False)
            row[f"contains_zero_changed_{coeff}"] = coeff_delta.get("contains_zero_changed", False)
        rows.append(row)

    if not rows:
        return {
            "status": "skipped: no LOSO refits completed",
            "study_col": study_col,
            "n_passed": 0,
            "n_failed": 0,
        }

    n_passed = sum(1 for r in rows if r.get("convergence_pass", False))
    n_failed = len(rows) - n_passed
    ranked = sorted(rows, key=lambda x: x["influence_score"], reverse=True)
    logger.info(
        f"LOSO analysis ({loso_scope}): {n_passed}/{len(rows)} refits passed convergence. "
        f"Top influential study={ranked[0]['study_id']}"
    )

    return {
        "status": "completed",
        "study_col": study_col,
        "loso_scope": loso_scope,
        "n_total_studies": int(len(counts)),
        "n_evaluated": len(ranked),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "ranked_studies": ranked,
    }


def run_cluster_bootstrap(
    df_raw: pd.DataFrame,
    config: ProjectConfig,
    script05,
    cache_dir: Path,
    tracked_coeffs: List[str],
    logger,
    cache_signature: str,
    run_mode: str = "standard",
) -> Dict[str, Any]:
    """Cluster bootstrap by study with exact-primary-model refits."""
    if not config.get("sensitivity.bootstrap.enabled", True):
        logger.info("Cluster bootstrap skipped: disabled in config")
        return {"status": "skipped: disabled"}

    cluster_col = config.get("sensitivity.bootstrap.cluster_column", None)
    if not cluster_col or cluster_col not in df_raw.columns:
        cluster_col = find_study_column(df_raw, config)
    if cluster_col is None:
        return {"status": "skipped: no cluster/study column"}

    clusters = [c for c in df_raw[cluster_col].dropna().unique().tolist()]
    if len(clusters) < 3:
        return {"status": "skipped: too few clusters", "cluster_col": cluster_col}

    n_iter_cfg = int(config.get("sensitivity.bootstrap.n_iterations", 1000))
    if run_mode == "quick":
        max_iter = 10
    elif run_mode == "publication":
        max_iter = 100
    else:
        max_iter = int(config.get("sensitivity.bootstrap.max_iterations", 30))
    n_iterations = min(n_iter_cfg, max_iter)
    if n_iter_cfg > max_iter:
        logger.info(
            f"Bootstrap iterations capped ({run_mode}): requested={n_iter_cfg}, running={n_iterations}."
        )

    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    rows = []
    diag_csv = cache_dir / "diagnostics_cluster_bootstrap.csv"

    for i in range(n_iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for j, cid in enumerate(sampled):
            part = df_raw[df_raw[cluster_col] == cid].copy()
            # Reindex study IDs so Script 05's local study remapping remains coherent within each bootstrap sample.
            part["Study_ID_numeric"] = j
            part["__boot_cluster_id"] = f"{cid}__{j}"
            parts.append(part)
        boot_df = pd.concat(parts, ignore_index=True)

        cache_key = stable_hash(
            {
                "analysis": "bootstrap",
                "signature": cache_signature,
                "iter": i,
                "cluster_col": cluster_col,
            }
        )

        def _fit_and_extract():
            fit = fit_primary_model_with_options(
                boot_df,
                config,
                script05,
                logger,
                prior_scale=1.0,
                likelihood_family="normal",
                seed_offset=4000 + i,
                run_mode=run_mode,
                auto_retry=True,
            )
            metrics = extract_model_conclusion_metrics(
                fit["trace"],
                tracked_coeffs,
                fit.get("coefficient_labels"),
                select_representative_indices(fit["df_clean"]),
                include_information_criteria=False,
            )
            return {
                "metrics": metrics,
                "convergence_diagnostics": fit.get("convergence_diagnostics", {}),
                "convergence_pass": fit.get("convergence_pass", False),
                "execution": fit.get("execution", {}),
            }

        candidate = run_cached_refit(
            f"bootstrap_{cache_key}",
            cache_dir,
            logger,
            _fit_and_extract,
            cache_signature=cache_signature,
            diagnostics_csv_path=diag_csv,
        )
        rec = {
            "iteration": i,
            "cache_status": candidate.get("cache_status", "unknown"),
            "auto_retry_triggered": bool(candidate.get("execution", {}).get("auto_retry_triggered", False)),
            "convergence_pass": candidate.get("convergence_pass", False),
            "pooled_mean": candidate["metrics"].get("pooled_log_qm", {}).get("mean", np.nan),
        }
        for coeff_name, coeff_metrics in candidate["metrics"].get("coefficients", {}).items():
            rec[coeff_name] = coeff_metrics.get("mean", np.nan)
        rows.append(rec)

    if not rows:
        return {
            "status": "skipped: no bootstrap refits completed",
            "cluster_col": cluster_col,
            "n_passed": 0,
            "n_failed": 0,
        }

    boot_df = pd.DataFrame(rows)
    n_passed = sum(1 for r in rows if r.get("convergence_pass", False))
    n_failed = len(rows) - n_passed
    ci = {
        "pooled_mean": {
            "lower": float(np.nanpercentile(boot_df["pooled_mean"], 2.5)),
            "upper": float(np.nanpercentile(boot_df["pooled_mean"], 97.5)),
            "mean": float(np.nanmean(boot_df["pooled_mean"])),
            "interval_type": "percentile_interval",
        },
        "coefficients": {},
    }

    excluded_cols = {"iteration", "cache_status", "auto_retry_triggered", "convergence_pass", "pooled_mean"}
    for coeff in [c for c in boot_df.columns if c not in excluded_cols]:
        if boot_df[coeff].notna().any():
            ci["coefficients"][coeff] = {
                "lower": float(np.nanpercentile(boot_df[coeff], 2.5)),
                "upper": float(np.nanpercentile(boot_df[coeff], 97.5)),
                "mean": float(np.nanmean(boot_df[coeff])),
                "interval_type": "percentile_interval",
            }

    logger.info(
        f"Cluster bootstrap completed ({run_mode}): {n_iterations} iter, {n_passed}/{n_iterations} passed convergence. "
        f"Pooled percentile interval=[{ci['pooled_mean']['lower']:.3f}, {ci['pooled_mean']['upper']:.3f}]"
    )

    return {
        "status": "completed",
        "cluster_col": cluster_col,
        "n_iterations": n_iterations,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "records": rows,
        "cluster_bootstrap_ci": ci,
    }


def save_csv_records(records: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dictionaries to CSV if non-empty."""
    if not records:
        return
    pd.DataFrame(records).to_csv(path, index=False)

def plot_perturbation_deltas(perturbation_result: Dict[str, Any], figures_dir: Path) -> Optional[str]:
    """Figure 1: delta posterior distributions across perturbation runs."""
    if perturbation_result.get("status") != "completed":
        return None

    dfp = pd.DataFrame(perturbation_result.get("records", []))
    if dfp.empty:
        return None

    metric_cols = [c for c in dfp.columns if c.startswith("delta_") and c.endswith("_mean")]
    metric_cols = ["delta_pooled_mean"] + [c for c in metric_cols if c != "delta_pooled_mean"]
    metric_cols = [c for c in metric_cols if c in dfp.columns]
    if not metric_cols:
        return None

    melted = dfp.melt(id_vars=["variable"], value_vars=metric_cols, var_name="metric", value_name="delta")
    melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta"])
    if melted.empty:
        return None

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=melted, x="variable", y="delta", hue="metric")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Delta Posterior Means Across Predictor Perturbations")
    plt.xlabel("Perturbed Predictor")
    plt.ylabel("Delta posterior mean (candidate - baseline)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = figures_dir / "Sensitivity_01_delta_posterior_perturbation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_prior_forest(prior_result: Dict[str, Any], figures_dir: Path) -> Optional[str]:
    """Figure 2: prior-variant comparison forest-like plot for pooled delta + LOO."""
    if prior_result.get("status") != "completed":
        return None

    dfp = pd.DataFrame(prior_result.get("variants", []))
    if dfp.empty:
        return None

    y = np.arange(len(dfp))
    x = dfp["delta_pooled_mean"].values

    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x,
        y,
        xerr=np.abs(dfp.get("delta_pooled_hdi_width", pd.Series(np.zeros(len(dfp))))).values / 2,
        fmt="o",
        capsize=4,
        label="Delta pooled mean (approx HDI-width/2)",
    )
    if "delta_elpd_loo" in dfp.columns:
        plt.scatter(dfp["delta_elpd_loo"].values, y, marker="s", label="Delta ELPD-LOO")

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, dfp["variant"].values)
    plt.xlabel("Delta versus baseline")
    plt.ylabel("Prior/Likelihood variant")
    plt.title("Prior Sensitivity: Variant Comparison")
    plt.legend()
    plt.tight_layout()

    out = figures_dir / "Sensitivity_02_prior_variant_forest.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_loso_tornado(loso_result: Dict[str, Any], figures_dir: Path) -> Optional[str]:
    """Figure 3: LOSO influence tornado plot."""
    if loso_result.get("status") != "completed":
        return None

    dfl = pd.DataFrame(loso_result.get("ranked_studies", []))
    if dfl.empty:
        return None

    top = dfl.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["study_id"], top["influence_score"], color="steelblue", alpha=0.85)
    plt.xlabel("Influence score (max abs delta posterior)")
    plt.ylabel("Study removed")
    plt.title("LOSO Influence Ranking")
    plt.tight_layout()

    out = figures_dir / "Sensitivity_03_loso_influence_tornado.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_cluster_bootstrap(bootstrap_result: Dict[str, Any], figures_dir: Path) -> Optional[str]:
    """Figure 4: cluster-bootstrap pooled posterior distribution."""
    if bootstrap_result.get("status") != "completed":
        return None

    dfb = pd.DataFrame(bootstrap_result.get("records", []))
    if dfb.empty or "pooled_mean" not in dfb.columns:
        return None

    plt.figure(figsize=(10, 5))
    sns.histplot(dfb["pooled_mean"].dropna(), bins=30, kde=True, color="darkcyan")
    ci = bootstrap_result.get("cluster_bootstrap_ci", {}).get("pooled_mean", {})
    if ci:
        plt.axvline(ci.get("lower", np.nan), linestyle="--", color="red", label="2.5%")
        plt.axvline(ci.get("upper", np.nan), linestyle="--", color="red", label="97.5%")
        plt.axvline(ci.get("mean", np.nan), linestyle="-", color="black", label="Mean")
    plt.xlabel("Bootstrap pooled log_Qm posterior mean")
    plt.ylabel("Count")
    plt.title("Cluster Bootstrap Distribution (Pooled log_Qm)")
    plt.legend()
    plt.tight_layout()

    out = figures_dir / "Sensitivity_04_cluster_bootstrap_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def main() -> int:
    """Main execution function."""
    try:
        config = ProjectConfig()
        logger = setup_logging(config, "07_sensitivity_analysis")
        set_random_seed(config=config)

        run_mode = config.get("sensitivity.run_mode", "standard")
        if run_mode not in ("quick", "standard", "publication"):
            logger.warning(f"Invalid run_mode '{run_mode}'; defaulting to 'standard'")
            run_mode = "standard"
        logger.info(f"Running sensitivity analysis in '{run_mode}' mode")

        print_section_header("SCRIPT 07: MODEL-CONCLUSION SENSITIVITY", logger=logger)

        out_dirs = ensure_dirs(config)
        results_dir = out_dirs["results"]
        figures_dir = out_dirs["figures"]
        cache_dir = out_dirs["cache"]

        data_path = config.get_path("processed_data") / "03_enriched_data.csv"
        df_raw = load_dataframe(data_path)
        logger.info(f"Loaded enriched data: n={len(df_raw)}, p={len(df_raw.columns)}")
        if "log_Qm" not in df_raw.columns:
            logger.error("Missing required outcome column: log_Qm")
            return 1

        script05 = load_script05_module(logger)
        tracked_coeffs = get_tracked_coefficients(config)
        primary_context = prepare_primary_model_inputs(df_raw, script05, logger)
        cache_sig = build_cache_signature(
            config,
            Path(__file__).parent / "05_model_fitting.py",
            df_raw,
            tracked_coeffs,
            run_mode,
        )
        logger.info(f"Sensitivity cache signature: {cache_sig['signature']}")

        baseline_cache = cache_dir / "baseline_metrics.json"
        baseline_metrics: Dict[str, Any]
        baseline_source: str

        cached_baseline = None
        if baseline_cache.exists():
            with open(baseline_cache, "r", encoding="utf-8") as f:
                cached_baseline = json.load(f)
            if cached_baseline.get("cache_signature") == cache_sig["signature"]:
                logger.info("Using cached baseline metrics")
                baseline_metrics = cached_baseline["metrics"]
                baseline_source = cached_baseline.get("baseline_source", "cache")
            else:
                logger.info("Baseline cache stale; rebuilding baseline metrics")
                cached_baseline = None

        if cached_baseline is None:
            existing_trace, existing_summary = load_existing_primary_trace(config, logger)
            can_use_artifact = False
            if existing_trace is not None:
                can_use_artifact, reasons = validate_trace_against_primary_context(
                    existing_trace,
                    existing_summary,
                    primary_context,
                    tracked_coeffs,
                )
                if not can_use_artifact:
                    logger.warning(
                        "Existing primary trace artifact did not pass conservative compatibility checks: "
                        + "; ".join(reasons)
                    )

            if can_use_artifact and existing_trace is not None:
                logger.info("Baseline source: validated Script 05 artifact (compatibility checks passed)")
                baseline_metrics = extract_model_conclusion_metrics(
                    existing_trace,
                    tracked_coeffs,
                    primary_context["coefficient_labels"],
                    representative_indices=[],
                    include_information_criteria=True,
                )
                baseline_source = "artifact"
            else:
                logger.info("Baseline artifact unavailable or incompatible; refitting exact Script 05 primary model")
                fit = fit_primary_model_with_options(
                    df_raw,
                    config,
                    script05,
                    logger,
                    prior_scale=1.0,
                    likelihood_family="normal",
                    seed_offset=900,
                    run_mode=run_mode,
                    auto_retry=True,
                )
                baseline_metrics = extract_model_conclusion_metrics(
                    fit["trace"],
                    tracked_coeffs,
                    fit.get("coefficient_labels"),
                    select_representative_indices(fit["df_clean"]),
                    include_information_criteria=True,
                )
                baseline_source = "fallback_refit"

            with open(baseline_cache, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cache_signature": cache_sig["signature"],
                        "baseline_source": baseline_source,
                        "metrics": baseline_metrics,
                    },
                    f,
                    indent=2,
                )

        logger.info(f"Baseline metrics prepared (source={baseline_source})")

        perturbation_result = run_parameter_perturbation(
            df_raw=df_raw,
            baseline_metrics=baseline_metrics,
            config=config,
            script05=script05,
            cache_dir=cache_dir,
            tracked_coeffs=tracked_coeffs,
            logger=logger,
            cache_signature=cache_sig["signature"],
            run_mode=run_mode,
        )

        prior_result = run_prior_sensitivity(
            df_raw=df_raw,
            baseline_metrics=baseline_metrics,
            config=config,
            script05=script05,
            cache_dir=cache_dir,
            tracked_coeffs=tracked_coeffs,
            logger=logger,
            cache_signature=cache_sig["signature"],
            run_mode=run_mode,
        )

        loso_result = run_loso_influence(
            df_raw=df_raw,
            baseline_metrics=baseline_metrics,
            config=config,
            script05=script05,
            cache_dir=cache_dir,
            tracked_coeffs=tracked_coeffs,
            logger=logger,
            cache_signature=cache_sig["signature"],
            run_mode=run_mode,
        )

        bootstrap_result = run_cluster_bootstrap(
            df_raw=df_raw,
            config=config,
            script05=script05,
            cache_dir=cache_dir,
            tracked_coeffs=tracked_coeffs,
            logger=logger,
            cache_signature=cache_sig["signature"],
            run_mode=run_mode,
        )

        fig_paths = {
            "perturbation": plot_perturbation_deltas(perturbation_result, figures_dir),
            "prior": plot_prior_forest(prior_result, figures_dir),
            "loso": plot_loso_tornado(loso_result, figures_dir),
            "cluster_bootstrap": plot_cluster_bootstrap(bootstrap_result, figures_dir),
        }

        save_csv_records(
            perturbation_result.get("records", []),
            results_dir / "07_sensitivity_perturbation_summary.csv",
        )
        save_csv_records(
            prior_result.get("variants", []),
            results_dir / "07_sensitivity_prior_summary.csv",
        )
        save_csv_records(
            loso_result.get("ranked_studies", []),
            results_dir / "07_sensitivity_loso_summary.csv",
        )
        save_csv_records(
            bootstrap_result.get("records", []),
            results_dir / "07_sensitivity_cluster_bootstrap_summary.csv",
        )

        final_results = {
            "baseline_source": baseline_source,
            "tracked_coefficients": tracked_coeffs,
            "reference_levels": primary_context["reference_levels"],
            "coefficient_labels": primary_context["coefficient_labels"],
            "cache_signature": cache_sig["signature"],
            "cache_signature_fields": cache_sig["fields"],
            "baseline_metrics": baseline_metrics,
            "perturbation": perturbation_result,
            "prior_sensitivity": prior_result,
            "loso_influence": loso_result,
            "cluster_bootstrap": bootstrap_result,
            "figures": fig_paths,
        }

        results_path = results_dir / "07_sensitivity_results.json"
        save_json(make_json_safe(final_results), results_path)
        logger.info(f"Saved sensitivity results JSON: {results_path}")

        logger.info("=" * 60)
        logger.info("MODEL-CONCLUSION SENSITIVITY ANALYSIS COMPLETED")
        logger.info("=" * 60)
        return 0

    except Exception as exc:
        if "logger" in locals():
            logger.error(f"Error in sensitivity analysis: {exc}", exc_info=True)
        else:
            print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

