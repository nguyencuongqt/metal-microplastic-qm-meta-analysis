"""
Script 08: Environmental Relevance Module
=========================================

Calculate Environmental Metal Vector Potential (EMVP) and assess
environmental implications of metal adsorption onto microplastics.

EMVP = Qm x C_MP (ug metal per L)

Includes:
- EMVP calculation across scenarios
- Monte Carlo uncertainty propagation
- Comparison with background concentrations
- Scenario analysis (polymer types, aging, MP concentrations)

Author: Manuscript authors
Date: March 2026
"""

import sys
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, save_dataframe, save_json, print_section_header)


def _qm_to_mg_g_scale_factor(unit: str) -> float:
    """Return multiplicative factor to convert Qm from `unit` to mg/g."""
    if not unit:
        return 1.0

    normalized = unit.strip().lower()
    normalized = normalized.replace("\u00c2\u00b5", "u")
    normalized = normalized.replace("\u00ce\u00bc", "u")
    normalized = normalized.replace("\u00b5", "u")
    normalized = normalized.replace("\u03bc", "u")
    normalized = normalized.replace("\u00c3\u2014", "x")
    normalized = normalized.replace(" ", "")

    # Be tolerant to encoding artifacts (e.g., "ug/g") and partial aliases.
    if "ug/g" in normalized:
        return 1.0 / 1000.0

    # Common aliases observed in config and source studies.
    if normalized in {"mg/g", "g/kg"}:
        return 1.0
    if normalized in {"ug/g", "mg/kg"}:
        return 1.0 / 1000.0

    return 1.0


def calculate_emvp(qm_values: np.ndarray, 
                   c_mp: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate Environmental Metal Vector Potential.
    
    EMVP = Qm (mg/g) x C_MP (mg/L) = ug metal / L
    
    Parameters
    ----------
    qm_values : np.ndarray
        Qm values in mg/g
    c_mp : float or np.ndarray
        Microplastic concentration in mg/L (scalar or sampled array)
        
    Returns
    -------
    np.ndarray
        EMVP values in ug/L
    """
    # Dimensional identity (C_MP in mg/L):
    #   metal[mg/L] = Qm[mg metal/g MP] x C_MP[g MP/L]
    #             = Qm[mg/g] x (C_MP[mg/L] / 1000)
    #   metal[ug/L] = metal[mg/L] x 1000 = Qm[mg/g] x C_MP[mg/L]
    # The g->mg (/1000) and mg->ug (x1000) factors cancel, so no extra factor.
    emvp = qm_values * c_mp  # ug/L

    return emvp


def monte_carlo_emvp_uncertainty(df: pd.DataFrame, 
                                 config: ProjectConfig,
                                 logger) -> tuple[dict, dict]:
    """
    Propagate uncertainty in EMVP using Monte Carlo simulation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with Qm values
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple[dict, dict]
        Monte Carlo results and verification audit data
    """
    logger.info("Performing Monte Carlo uncertainty propagation...")
    
    # Get configuration
    n_simulations = config.get('environmental.emvp.monte_carlo.n_simulations', 10000)
    qm_cv = config.get('environmental.emvp.monte_carlo.Qm_cv', 0.3)
    c_mp_cv = config.get('environmental.emvp.monte_carlo.C_MP_cv', 0.5)
    mp_concentrations = config.get('environmental.emvp.mp_concentrations', {})
    random_seed = config.get('random_seed', 42)
    
    logger.info(f"  Simulations: {n_simulations}")
    logger.info(f"  Qm CV: {qm_cv}")
    logger.info(f"  C_MP CV: {c_mp_cv}")
    
    # Back-transform log_Qm to Qm, then normalize to mg/g for EMVP formula.
    source_qm_unit = config.get(
        'data_processing.qm_units.source',
        config.get('data_processing.qm_units.target', 'mg/g')
    )
    target_qm_unit = 'mg/g'
    qm_scale = _qm_to_mg_g_scale_factor(source_qm_unit)

    # Unit conversion invariants for common aliases.
    conversion_asserts = {
        '1000_ug_g_to_1_mg_g': bool(np.isclose(1000.0 * _qm_to_mg_g_scale_factor('ug/g'), 1.0)),
        '1_mg_kg_to_0p001_mg_g': bool(np.isclose(1.0 * _qm_to_mg_g_scale_factor('mg/kg'), 0.001)),
        'mg_g_factor_is_1': bool(np.isclose(_qm_to_mg_g_scale_factor('mg/g'), 1.0)),
    }
    assert all(conversion_asserts.values())

    if qm_scale != 1.0:
        logger.info("  Qm unit config keys checked: data_processing.qm_units.source, data_processing.qm_units.target")
        logger.info(f"  Detected source Qm unit: {source_qm_unit}")
        logger.info(f"  Applying conversion {source_qm_unit} -> {target_qm_unit} (factor={qm_scale})")
    else:
        logger.info("  Qm unit config keys checked: data_processing.qm_units.source, data_processing.qm_units.target")
        logger.info(f"  Detected source Qm unit as mg/g-compatible: {source_qm_unit}")

    q_from_log_before_scaling = np.exp(df['log_Qm'])
    q_mg_g = q_from_log_before_scaling * qm_scale

    logger.info(
        "  Qm from exp(log_Qm) BEFORE scaling (source unit): "
        f"min={q_from_log_before_scaling.min():.6g}, "
        f"median={q_from_log_before_scaling.median():.6g}, "
        f"max={q_from_log_before_scaling.max():.6g}"
    )
    logger.info(
        "  Qm_mg_g AFTER scaling: "
        f"min={q_mg_g.min():.6g}, "
        f"median={q_mg_g.median():.6g}, "
        f"max={q_mg_g.max():.6g}"
    )

    id_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ['metal', 'polymer', 'study', 'source', 'ags'])
    ]
    top_id_cols = []
    for col in ['Metal', 'Polymer', 'Study_ID', 'AgS']:
        if col in df.columns and col not in top_id_cols:
            top_id_cols.append(col)
    for col in id_candidates:
        if col not in top_id_cols:
            top_id_cols.append(col)
    top_id_cols = top_id_cols[:4]
    top10_df = df[top_id_cols].copy()
    top10_df['Qm_mg_g'] = q_mg_g
    top10 = top10_df.nlargest(10, 'Qm_mg_g')[top_id_cols + ['Qm_mg_g']]
    logger.info("  Top-10 largest Qm_mg_g rows (identifier check):")
    logger.info("\n" + top10.to_string(index=False))

    verification = {
        'unit': {
            'source_qm_unit': source_qm_unit,
            'target_qm_unit': target_qm_unit,
            'applied_scale_factor': float(qm_scale),
            'conversion_asserts': conversion_asserts,
            'q_from_log_before_scaling': {
                'min': float(q_from_log_before_scaling.min()),
                'median': float(q_from_log_before_scaling.median()),
                'max': float(q_from_log_before_scaling.max()),
            },
            'q_mg_g_after_scaling': {
                'min': float(q_mg_g.min()),
                'median': float(q_mg_g.median()),
                'max': float(q_mg_g.max()),
            },
            'top10_qm_mg_g_rows': top10.to_dict(orient='records'),
        },
        'emvp_identity': {},
        'monte_carlo_moment_checks': [],
    }
    
    # Sanity check: compute example EMVP
    if len(df) > 0:
        example_qm = q_mg_g.iloc[0]
        example_cmp = mp_concentrations.get('moderate', 0.01)
        expected_emvp = example_qm * example_cmp
        computed_emvp = float(calculate_emvp(np.array([example_qm]), float(example_cmp))[0])
        emvp_abs_diff = abs(expected_emvp - computed_emvp)
        logger.info(
            "  Deterministic EMVP check: "
            f"EMVP = {example_qm:.6g} mg/g * {example_cmp:.6g} mg/L = {expected_emvp:.6g} ug/L; "
            f"computed={computed_emvp:.6g} ug/L; abs_diff={emvp_abs_diff:.3e}"
        )
        verification['emvp_identity'] = {
            'example_qm_mg_g': float(example_qm),
            'example_c_mp_mg_L': float(example_cmp),
            'expected_emvp_ug_L': float(expected_emvp),
            'computed_emvp_ug_L': float(computed_emvp),
            'abs_diff': float(emvp_abs_diff),
        }
    
    # Calculate mean Qm by metal (if available)
    metal_col = [col for col in df.columns if 'metal' in col.lower() and 'encoded' not in col.lower()]
    
    if metal_col:
        metal_col = metal_col[0]
        qm_by_metal = pd.DataFrame({metal_col: df[metal_col], 'Qm_mg_g': q_mg_g}).groupby(metal_col)['Qm_mg_g'].agg(['mean', 'std']).to_dict('index')
    else:
        # Use overall statistics
        qm_by_metal = {
            'Overall': {
                'mean': q_mg_g.mean(),
                'std': q_mg_g.std()
            }
        }
    
    # Initialize RNG once for reproducibility
    rng = np.random.default_rng(random_seed)
    
    # Convert CV to lognormal parameters
    qm_sigma = np.sqrt(np.log(1 + qm_cv**2))
    c_mp_sigma = np.sqrt(np.log(1 + c_mp_cv**2))
    
    results = {}
    
    for metal, qm_stats in qm_by_metal.items():
        qm_mean = qm_stats['mean']
        metal_results = {}
        
        for scenario_name, c_mp_mean in mp_concentrations.items():
            if not np.isfinite(qm_mean) or qm_mean <= 0:
                logger.warning(f"  Skipping {metal}-{scenario_name}: non-positive/non-finite Qm mean ({qm_mean})")
                continue
            if not np.isfinite(c_mp_mean) or c_mp_mean <= 0:
                logger.warning(f"  Skipping {metal}-{scenario_name}: non-positive/non-finite C_MP mean ({c_mp_mean})")
                continue

            # Generate distributions using correct lognormal parameterization
            # For lognormal: if X ~ lognormal(mu, sigma), then E[X] = exp(mu + sigma^2/2)
            # Given mean and CV: sigma = sqrt(log(1 + CV^2)), mu = log(mean) - sigma^2/2
            
            qm_mu = np.log(qm_mean) - 0.5 * qm_sigma**2
            qm_samples = rng.lognormal(
                mean=qm_mu,
                sigma=qm_sigma,
                size=n_simulations
            )
            
            c_mp_mu = np.log(c_mp_mean) - 0.5 * c_mp_sigma**2
            c_mp_samples = rng.lognormal(
                mean=c_mp_mu,
                sigma=c_mp_sigma,
                size=n_simulations
            )

            # Monte Carlo verification: sample moments vs targets.
            qm_sample_mean = float(np.mean(qm_samples))
            qm_sample_cv = float(np.std(qm_samples) / qm_sample_mean)
            c_mp_sample_mean = float(np.mean(c_mp_samples))
            c_mp_sample_cv = float(np.std(c_mp_samples) / c_mp_sample_mean)

            qm_mean_err_pct = 100.0 * abs(qm_sample_mean - qm_mean) / qm_mean
            qm_cv_err_pct = 100.0 * abs(qm_sample_cv - qm_cv) / qm_cv
            c_mp_mean_err_pct = 100.0 * abs(c_mp_sample_mean - c_mp_mean) / c_mp_mean
            c_mp_cv_err_pct = 100.0 * abs(c_mp_sample_cv - c_mp_cv) / c_mp_cv

            max_err_pct = max(qm_mean_err_pct, qm_cv_err_pct, c_mp_mean_err_pct, c_mp_cv_err_pct)

            logger.info(
                f"    MC verify [{metal}-{scenario_name}] Qm sample mean={qm_sample_mean:.6g}, "
                f"CV={qm_sample_cv:.4f}, %err(mean)={qm_mean_err_pct:.2f}, %err(CV)={qm_cv_err_pct:.2f}"
            )
            logger.info(
                f"    MC verify [{metal}-{scenario_name}] C_MP sample mean={c_mp_sample_mean:.6g}, "
                f"CV={c_mp_sample_cv:.4f}, %err(mean)={c_mp_mean_err_pct:.2f}, %err(CV)={c_mp_cv_err_pct:.2f}"
            )

            verification['monte_carlo_moment_checks'].append({
                'metal': str(metal),
                'scenario': str(scenario_name),
                'qm_target_mean_mg_g': float(qm_mean),
                'qm_target_cv': float(qm_cv),
                'qm_sample_mean_mg_g': float(qm_sample_mean),
                'qm_sample_cv': float(qm_sample_cv),
                'qm_mean_error_pct': float(qm_mean_err_pct),
                'qm_cv_error_pct': float(qm_cv_err_pct),
                'c_mp_target_mean_mg_L': float(c_mp_mean),
                'c_mp_target_cv': float(c_mp_cv),
                'c_mp_sample_mean_mg_L': float(c_mp_sample_mean),
                'c_mp_sample_cv': float(c_mp_sample_cv),
                'c_mp_mean_error_pct': float(c_mp_mean_err_pct),
                'c_mp_cv_error_pct': float(c_mp_cv_err_pct),
                'max_error_pct': float(max_err_pct),
                'reparameterized': bool(max_err_pct > 5.0),
            })
            if max_err_pct > 5.0:
                logger.warning(
                    f"    MC verify [{metal}-{scenario_name}] error {max_err_pct:.2f}% > 5%; "
                    "reapplying parameterization and regenerating samples"
                )
                qm_sigma = np.sqrt(np.log(1 + qm_cv**2))
                c_mp_sigma = np.sqrt(np.log(1 + c_mp_cv**2))
                qm_mu = np.log(qm_mean) - 0.5 * qm_sigma**2
                c_mp_mu = np.log(c_mp_mean) - 0.5 * c_mp_sigma**2
                qm_samples = rng.lognormal(mean=qm_mu, sigma=qm_sigma, size=n_simulations)
                c_mp_samples = rng.lognormal(mean=c_mp_mu, sigma=c_mp_sigma, size=n_simulations)
            
            # Calculate EMVP
            emvp_samples = calculate_emvp(qm_samples, c_mp_samples)
            
            # Calculate statistics
            metal_results[scenario_name] = {
                'c_mp_mg_L': c_mp_mean,
                'qm_mean_mg_g': float(qm_mean),
                'emvp_mean_ug_L': float(emvp_samples.mean()),
                'emvp_median_ug_L': float(np.median(emvp_samples)),
                'emvp_std_ug_L': float(emvp_samples.std()),
                'emvp_ci_95_lower': float(np.percentile(emvp_samples, 2.5)),
                'emvp_ci_95_upper': float(np.percentile(emvp_samples, 97.5)),
            }
            
            logger.info(f"\n  {metal} - {scenario_name}:")
            logger.info(f"    C_MP: {c_mp_mean} mg/L")
            logger.info(f"    Qm: {qm_mean:.2f} mg/g")
            logger.info(f"    EMVP: {metal_results[scenario_name]['emvp_mean_ug_L']:.3f} ug/L "
                       f"(95% CI: [{metal_results[scenario_name]['emvp_ci_95_lower']:.3f}, "
                       f"{metal_results[scenario_name]['emvp_ci_95_upper']:.3f}])")
        
        results[metal] = metal_results
    
    return results, verification


def compare_with_background(emvp_results: dict, config: ProjectConfig, logger) -> dict:
    """
    Compare EMVP with background dissolved metal concentrations.
    
    Parameters
    ----------
    emvp_results : dict
        EMVP calculation results
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Comparison results
    """
    logger.info("\nComparing EMVP with background concentrations...")
    
    background_metals = config.get('environmental.emvp.background_metals', {})
    ratio_minor_threshold = config.get('environmental.emvp.ratio_minor_threshold', 0.1)
    ratio_major_threshold = config.get('environmental.emvp.ratio_major_threshold', 1.0)
    
    comparisons = {}
    
    for metal, scenarios in emvp_results.items():
        if metal == 'Overall':
            continue
        
        background_conc = background_metals.get(metal)
        
        if not background_conc:
            logger.warning(f"  No background concentration for {metal}")
            continue
        
        metal_comparisons = {}
        
        for scenario, values in scenarios.items():
            emvp_mean = values['emvp_mean_ug_L']
            emvp_median = values['emvp_median_ug_L']
            emvp_ci_lower = values['emvp_ci_95_lower']
            emvp_ci_upper = values['emvp_ci_95_upper']
            ratio = emvp_mean / background_conc
            median_ratio = emvp_median / background_conc
            ci_ratio_lower = emvp_ci_lower / background_conc
            ci_ratio_upper = emvp_ci_upper / background_conc
            
            # Determine interpretation based on configurable thresholds
            if ratio < ratio_minor_threshold:
                interpretation = 'likely_minor'
            elif ratio < ratio_major_threshold:
                interpretation = 'potentially_relevant'
            else:
                interpretation = 'dominant'
            
            metal_comparisons[scenario] = {
                'emvp_ug_L': emvp_mean,
                'emvp_median_ug_L': emvp_median,
                'emvp_ci_95_lower_ug_L': emvp_ci_lower,
                'emvp_ci_95_upper_ug_L': emvp_ci_upper,
                'background_ug_L': background_conc,
                'emvp_background_ratio': float(ratio),
                'emvp_background_ratio_median': float(median_ratio),
                'emvp_background_ratio_ci95_lower': float(ci_ratio_lower),
                'emvp_background_ratio_ci95_upper': float(ci_ratio_upper),
                'interpretation': interpretation
            }
            
            logger.info(f"\n  {metal} - {scenario}:")
            logger.info(f"    EMVP: {emvp_mean:.3f} ug/L")
            logger.info(f"    EMVP median: {emvp_median:.3f} ug/L")
            logger.info(f"    Background: {background_conc} ug/L")
            logger.info(f"    Ratio: {ratio:.4f} ({ratio*100:.2f}%)")
            logger.info(f"    Ratio (median): {median_ratio:.4f} ({median_ratio*100:.2f}%)")
            logger.info(f"    Ratio (CI95): [{ci_ratio_lower:.4f}, {ci_ratio_upper:.4f}]")
            logger.info(f"    Interpretation: {interpretation}")
        
        comparisons[metal] = metal_comparisons
    
    return comparisons


def scenario_analysis(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Perform scenario analysis for different conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Modeling data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Scenario analysis results
    """
    logger.info("\nPerforming scenario analysis...")
    
    scenarios = config.get('environmental.scenarios', {})
    
    # Build local Qm_mg_g series (avoid mutating input dataframe).
    if 'Qm_mg_g' in df.columns:
        q_mg_g = df['Qm_mg_g']
    else:
        source_qm_unit = config.get(
            'data_processing.qm_units.source',
            config.get('data_processing.qm_units.target', 'mg/g')
        )
        qm_scale = _qm_to_mg_g_scale_factor(source_qm_unit)
        q_mg_g = np.exp(df['log_Qm']) * qm_scale
        logger.info("  Qm unit config keys checked: data_processing.qm_units.source, data_processing.qm_units.target")
        logger.info(f"  Applied unit conversion to local Qm_mg_g ({source_qm_unit} -> mg/g, factor={qm_scale})")
    
    results = {}
    
    # Analyze by aging status
    aging_scenarios = scenarios.get('aging_scenarios', [])
    aging_col = [col for col in df.columns if 'aging' in col.lower() and 'encoded' not in col.lower()]
    
    if aging_col and aging_scenarios:
        aging_col = aging_col[0]
        logger.info(f"\nAging Scenario Analysis:")
        
        aging_results = {}
        for scenario in aging_scenarios:
            mask = df[aging_col].str.contains(scenario, case=False, na=False)
            if mask.any():
                qm_values = q_mg_g.loc[mask]
                aging_results[scenario] = {
                    'n': int(mask.sum()),
                    'qm_mean_mg_g': float(qm_values.mean()),
                    'qm_std_mg_g': float(qm_values.std()),
                    'qm_median_mg_g': float(qm_values.median())
                }
                logger.info(f"  {scenario}: n={aging_results[scenario]['n']}, "
                           f"Qm={aging_results[scenario]['qm_mean_mg_g']:.2f}+/-{aging_results[scenario]['qm_std_mg_g']:.2f} mg/g")
        
        results['aging'] = aging_results
    
    # Analyze by polymer type
    polymer_scenarios = scenarios.get('polymer_scenarios', [])
    polymer_col = [col for col in df.columns if 'polymer' in col.lower() and 'encoded' not in col.lower()]
    
    if polymer_col and polymer_scenarios:
        polymer_col = polymer_col[0]
        logger.info(f"\nPolymer Scenario Analysis:")
        
        polymer_results = {}
        for scenario in polymer_scenarios:
            mask = df[polymer_col].str.contains(scenario, case=False, na=False)
            if mask.any():
                qm_values = q_mg_g.loc[mask]
                polymer_results[scenario] = {
                    'n': int(mask.sum()),
                    'qm_mean_mg_g': float(qm_values.mean()),
                    'qm_std_mg_g': float(qm_values.std()),
                    'qm_median_mg_g': float(qm_values.median())
                }
                logger.info(f"  {scenario}: n={polymer_results[scenario]['n']}, "
                           f"Qm={polymer_results[scenario]['qm_mean_mg_g']:.2f}+/-{polymer_results[scenario]['qm_std_mg_g']:.2f} mg/g")
        
        results['polymer'] = polymer_results
    
    # Analyze priority metals
    metal_priority = scenarios.get('metal_priority', [])
    metal_col = [col for col in df.columns if 'metal' in col.lower() and 'encoded' not in col.lower()]
    
    if metal_col and metal_priority:
        metal_col = metal_col[0]
        logger.info(f"\nPriority Metal Analysis:")
        
        metal_results = {}
        for metal in metal_priority:
            mask = df[metal_col] == metal
            if mask.any():
                qm_values = q_mg_g.loc[mask]
                metal_results[metal] = {
                    'n': int(mask.sum()),
                    'qm_mean_mg_g': float(qm_values.mean()),
                    'qm_std_mg_g': float(qm_values.std()),
                    'qm_median_mg_g': float(qm_values.median())
                }
                logger.info(f"  {metal}: n={metal_results[metal]['n']}, "
                           f"Qm={metal_results[metal]['qm_mean_mg_g']:.2f}+/-{metal_results[metal]['qm_std_mg_g']:.2f} mg/g")
        
        results['priority_metals'] = metal_results
    
    return results


def create_emvp_visualization(emvp_results: dict, 
                              background_comparison: dict,
                              config: ProjectConfig,
                              logger) -> None:
    """
    Create comprehensive EMVP visualization.
    
    Parameters
    ----------
    emvp_results : dict
        EMVP results
    background_comparison : dict
        Background comparison results
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
    """
    logger.info("\nCreating EMVP visualizations...")
    
    # Prepare data for plotting
    plot_data = []
    
    for metal, scenarios in emvp_results.items():
        if metal == 'Overall':
            continue
        
        for scenario, values in scenarios.items():
            plot_data.append({
                'Metal': metal,
                'Scenario': scenario,
                'EMVP': values['emvp_mean_ug_L'],
                'CI_lower': values['emvp_ci_95_lower'],
                'CI_upper': values['emvp_ci_95_upper']
            })
    
    if not plot_data:
        logger.warning("No data available for EMVP visualization")
        return
    
    df_plot = pd.DataFrame(plot_data)

    non_positive_mask = df_plot['EMVP'] <= 0
    n_non_positive = int(non_positive_mask.sum())
    if n_non_positive > 0:
        logger.warning(
            f"Found {n_non_positive} non-positive EMVP values; filtering before log-scale plotting"
        )
        df_plot = df_plot.loc[~non_positive_mask].copy()

    if df_plot.empty:
        logger.warning("No positive EMVP data available for log-scale visualization after filtering")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: EMVP by scenario with confidence intervals
    scenarios = list(df_plot['Scenario'].unique())
    metal_order = list(df_plot['Metal'].unique())
    x = np.arange(len(scenarios))
    width = 0.8 / len(metal_order)
    
    for i, metal in enumerate(metal_order):
        metal_data = (
            df_plot[df_plot['Metal'] == metal]
            .set_index('Scenario')
            .reindex(scenarios)
        )
        valid = metal_data['EMVP'].notna()
        if not valid.any():
            continue

        x_pos = x[valid.values] + i * width
        y = metal_data.loc[valid, 'EMVP']
        ci_lower = metal_data.loc[valid, 'CI_lower'].clip(lower=1e-6)
        ci_upper = metal_data.loc[valid, 'CI_upper']

        # Compute error bars (must be positive for log scale)
        yerr_lower = y - ci_lower
        yerr_upper = ci_upper - y
        axes[0].bar(x_pos, y, width, label=metal,
                   yerr=[yerr_lower, yerr_upper], capsize=3, error_kw={'linewidth': 1, 'alpha': 0.7})
    
    axes[0].set_xlabel('MP Concentration Scenario')
    axes[0].set_ylabel('EMVP (ug metal / L)')
    axes[0].set_title('Environmental Metal Vector Potential by Scenario')
    axes[0].set_xticks(x + width * (len(metal_order)-1) / 2)
    axes[0].set_xticklabels(scenarios, rotation=45)
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Comparison with background
    if background_comparison:
        comparison_data = []
        for metal, scenarios in background_comparison.items():
            for scenario, values in scenarios.items():
                if scenario == 'moderate':  # Show one representative scenario
                    comparison_data.append({
                        'Metal': metal,
                        'EMVP': values['emvp_ug_L'],
                        'Background': values['background_ug_L']
                    })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            x = np.arange(len(df_comp))
            width = 0.35
            
            axes[1].bar(x - width/2, df_comp['EMVP'], width, label='EMVP (moderate scenario)')
            axes[1].bar(x + width/2, df_comp['Background'], width, label='Background')
            axes[1].set_xlabel('Metal')
            axes[1].set_ylabel('Concentration (ug/L)')
            axes[1].set_title('EMVP vs Background Concentration')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(df_comp['Metal'])
            axes[1].legend()
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = config.get_path('figures') / 'Environmental_01_EMVP_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved EMVP visualization: {fig_path}")


def _create_summary_table(emvp_results: dict, background_comparison: dict, 
                          config: ProjectConfig, logger) -> None:
    """
    Export long-format summary table to CSV.
    
    One row per (metal, scenario) with columns: metal, scenario, c_mp_mg_L, qm_mean_mg_g,
    emvp_mean_ug_L, emvp_median_ug_L, emvp_ci_95_lower, emvp_ci_95_upper,
    background_ug_L, ratio_mean, ratio_median, ratio_ci95_lower, ratio_ci95_upper, interpretation.
    """
    logger.info("\nExporting summary table...")
    
    rows = []
    for metal, scenarios in emvp_results.items():
        if metal == 'Overall':
            continue
        
        for scenario_name, emvp_vals in scenarios.items():
            row = {
                'metal': metal,
                'scenario': scenario_name,
                'c_mp_mg_L': emvp_vals['c_mp_mg_L'],
                'qm_mean_mg_g': emvp_vals['qm_mean_mg_g'],
                'emvp_mean_ug_L': emvp_vals['emvp_mean_ug_L'],
                'emvp_median_ug_L': emvp_vals['emvp_median_ug_L'],
                'emvp_ci_95_lower': emvp_vals['emvp_ci_95_lower'],
                'emvp_ci_95_upper': emvp_vals['emvp_ci_95_upper'],
                'background_ug_L': None,
                'ratio_mean': None,
                'ratio_median': None,
                'ratio_ci95_lower': None,
                'ratio_ci95_upper': None,
                'interpretation': None,
            }
            
            # Add background comparison if available
            if metal in background_comparison and scenario_name in background_comparison[metal]:
                bg = background_comparison[metal][scenario_name]
                row['background_ug_L'] = bg['background_ug_L']
                row['ratio_mean'] = bg['emvp_background_ratio']
                row['ratio_median'] = bg['emvp_background_ratio_median']
                row['ratio_ci95_lower'] = bg['emvp_background_ratio_ci95_lower']
                row['ratio_ci95_upper'] = bg['emvp_background_ratio_ci95_upper']
                row['interpretation'] = bg['interpretation']
            
            rows.append(row)
    
    df_summary = pd.DataFrame(rows)
    summary_path = config.get_path('results') / "08_emvp_summary_table.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary table: {summary_path}")


def _sensitivity_analysis(df: pd.DataFrame, config: ProjectConfig, logger,
                          qm_scale: float) -> None:
    """
    Run lightweight sensitivity analysis on CV parameters and n_simulations.
    Grid: Qm_cv in {0.1, 0.3, 0.5}, C_MP_cv in {0.2, 0.5, 0.8}, n_simulations in {2000, 10000}
    Report for metals {Pb, Cd, As} and scenario "moderate".
    """
    logger.info("\nRunning sensitivity analysis...")
    
    # Configuration
    focus_metals = config.get('environmental.emvp.sensitivity_metals', ['Pb', 'Cd', 'As'])
    focus_scenario = config.get('environmental.emvp.sensitivity_scenario', 'moderate')
    mp_concentrations = config.get('environmental.emvp.mp_concentrations', {})
    random_seed = config.get('random_seed', 42)
    base_qm_cv = config.get('environmental.emvp.monte_carlo.Qm_cv', 0.3)
    base_c_mp_cv = config.get('environmental.emvp.monte_carlo.C_MP_cv', 0.5)
    
    # Get Metal column
    metal_col = [col for col in df.columns if 'metal' in col.lower() and 'encoded' not in col.lower()]
    if not metal_col:
        logger.warning("  No Metal column found; skipping sensitivity analysis")
        return
    
    metal_col = metal_col[0]
    q_from_log = np.exp(df['log_Qm'])
    q_mg_g = q_from_log * qm_scale
    
    # Map available metals to focus list
    available_metals = df[metal_col].unique()
    metals_to_test = [m for m in focus_metals if m in available_metals]
    
    if focus_scenario not in mp_concentrations:
        logger.warning(f"  Scenario '{focus_scenario}' not in config; using 'moderate' if available")
        focus_scenario = 'moderate' if 'moderate' in mp_concentrations else list(mp_concentrations.keys())[0]
    
    c_mp_mean = mp_concentrations[focus_scenario]
    
    # Grid parameters
    qm_cv_vals = [0.1, 0.3, 0.5]
    c_mp_cv_vals = [0.2, 0.5, 0.8]
    n_sim_vals = [2000, 10000]
    
    sensitivity_results = []
    rng = np.random.default_rng(random_seed)
    
    for metal in metals_to_test:
        mask = df[metal_col] == metal
        if not mask.any():
            continue
        
        qm_mean = q_mg_g[mask].mean()
        
        for qm_cv in qm_cv_vals:
            for c_mp_cv in c_mp_cv_vals:
                for n_sim in n_sim_vals:
                    # Parameterize and sample
                    qm_sigma = np.sqrt(np.log(1 + qm_cv**2))
                    c_mp_sigma = np.sqrt(np.log(1 + c_mp_cv**2))
                    qm_mu = np.log(qm_mean) - 0.5 * qm_sigma**2
                    c_mp_mu = np.log(c_mp_mean) - 0.5 * c_mp_sigma**2
                    
                    qm_samples = rng.lognormal(mean=qm_mu, sigma=qm_sigma, size=n_sim)
                    c_mp_samples = rng.lognormal(mean=c_mp_mu, sigma=c_mp_sigma, size=n_sim)
                    
                    emvp_samples = calculate_emvp(qm_samples, c_mp_samples)
                    ci_width = np.percentile(emvp_samples, 97.5) - np.percentile(emvp_samples, 2.5)
                    
                    sensitivity_results.append({
                        'metal': metal,
                        'scenario': focus_scenario,
                        'qm_cv': qm_cv,
                        'c_mp_cv': c_mp_cv,
                        'n_simulations': n_sim,
                        'emvp_mean_ug_L': float(emvp_samples.mean()),
                        'emvp_ci_95_width': float(ci_width),
                    })
    
    results_dict = {
        'grid_parameters': {
            'qm_cv': qm_cv_vals,
            'c_mp_cv': c_mp_cv_vals,
            'n_simulations': n_sim_vals,
        },
        'results': sensitivity_results,
    }
    
    sensitivity_path = config.get_path('results') / "08_emvp_sensitivity.json"
    save_json(results_dict, sensitivity_path)
    logger.info(f"Saved sensitivity analysis: {sensitivity_path}")
    
    # Create figure
    df_sens = pd.DataFrame(sensitivity_results)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for n_sim in n_sim_vals:
        subset = df_sens[df_sens['n_simulations'] == n_sim]
        if not subset.empty:
            ax.plot(range(len(subset)), subset['emvp_ci_95_width'].values, 
                   marker='o', label=f'n_sim={n_sim}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Grid Index (Qm_cv, C_MP_cv)')
    ax.set_ylabel('95% CI Width (ug/L)')
    ax.set_title('Sensitivity: EMVP CI Width vs CV Settings (Pb, Cd, As - moderate scenario)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = config.get_path('figures') / 'Environmental_02_sensitivity_ci_width.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sensitivity figure: {fig_path}")


def _compute_reproducibility_metadata(input_csv_path: Path, verification_dict: dict, logger) -> None:
    """
    Record reproducibility information: random_seed, library versions, and input CSV hash.
    """
    logger.info("\nRecording reproducibility metadata...")
    
    try:
        # Compute SHA256 hash of input CSV
        sha256_hash = hashlib.sha256()
        with open(input_csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        
        verification_dict['reproducibility'] = {
            'input_csv_sha256': sha256_hash.hexdigest(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
        }
        logger.info(f"  Input CSV SHA256: {verification_dict['reproducibility']['input_csv_sha256'][:16]}...")
        logger.info(f"  NumPy: {verification_dict['reproducibility']['numpy_version']}")
        logger.info(f"  Pandas: {verification_dict['reproducibility']['pandas_version']}")
    except Exception as e:
        logger.warning(f"  Could not compute reproducibility metadata: {e}")


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '08_environmental_module')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 08: ENVIRONMENTAL RELEVANCE MODULE", logger=logger)
        
        # Load data
        data_path = config.get_path('processed_data') / "03_enriched_data.csv"
        df = load_dataframe(data_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Step 1: Monte Carlo EMVP calculation
        emvp_results, verification_results = monte_carlo_emvp_uncertainty(df, config, logger)
        
        # Step 2: Compare with background
        background_comparison = compare_with_background(emvp_results, config, logger)
        
        # Step 3: Scenario analysis
        scenario_results = scenario_analysis(df, config, logger)
        
        # Step 4: Create visualizations
        create_emvp_visualization(emvp_results, background_comparison, config, logger)
        
        # Step 5: Export summary table
        _create_summary_table(emvp_results, background_comparison, config, logger)
        
        # Step 6: Run sensitivity analysis (extract qm_scale from verification)
        qm_scale = verification_results.get('unit', {}).get('applied_scale_factor', 0.001)
        _sensitivity_analysis(df, config, logger, qm_scale)
        
        # Step 7: Add reproducibility metadata
        _compute_reproducibility_metadata(data_path, verification_results, logger)
        
        # Save results
        environmental_results = {
            'emvp': emvp_results,
            'background_comparison': background_comparison,
            'scenarios': scenario_results
        }
        
        results_path = config.get_path('results') / "08_environmental_results.json"
        save_json(environmental_results, results_path)
        logger.info(f"\nSaved environmental results: {results_path}")

        verification_path = config.get_path('results') / "08_environmental_verification.json"
        save_json(verification_results, verification_path)
        logger.info(f"Saved environmental verification (with reproducibility metadata): {verification_path}")
        
        logger.info("\n" + "="*60)
        logger.info("ENVIRONMENTAL MODULE COMPLETED")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in environmental module: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
