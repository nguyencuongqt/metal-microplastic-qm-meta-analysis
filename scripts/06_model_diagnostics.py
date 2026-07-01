"""
Script 06: Model Diagnostics and Validation
===========================================

Comprehensive model diagnostics including:
- Convergence diagnostics
- Posterior predictive checks
- Residual diagnostics
- Multicollinearity assessment  
- Leave-one-study-out validation framework
- Influence diagnostics

Author: Manuscript authors
Date: March 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, load_json, save_json, print_section_header)


def get_data_var_safely(azdata, preferred_names: list = None, data_type: str = "variable"):
    """
    Safely retrieve data variable names from ArviZ objects.
    
    Parameters
    ----------
    azdata : arviz object
        ArviZ object (trace.observed_data, trace.log_likelihood, etc.)
    preferred_names : list
        Preferred variable names to try first (e.g., ['likelihood', 'y'])
    data_type : str
        Description of data type for logging
        
    Returns
    -------
    str or None
        Variable name if found, else None
    """
    if azdata is None or not hasattr(azdata, 'data_vars'):
        return None
    
    available_vars = list(azdata.data_vars)
    
    if not available_vars:
        return None
    
    # Try preferred names first
    if preferred_names:
        for name in preferred_names:
            if name in available_vars:
                return name
    
    # Return first available
    return available_vars[0]


def load_trace_with_fallback(trace_path, logger):
    """
    Load ArviZ trace with fallback mechanisms.
    
    Parameters
    ----------
    trace_path : Path
        Path to trace file (.nc or .json)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    arviz.InferenceData or None
        Loaded trace, or None if failed
    """
    # Try NetCDF first
    try:
        logger.info(f"Attempting to load trace from NetCDF format...")
        trace = az.from_netcdf(str(trace_path))
        logger.info("[OK] Successfully loaded trace from NetCDF")
        return trace
    except ImportError as e:
        logger.debug(f"NetCDF loading failed (h5py issue): {e}")
    except Exception as e:
        logger.debug(f"NetCDF loading failed: {e}")
    
    # Try JSON fallback
    json_path = trace_path.with_suffix('.json')
    if json_path.exists():
        try:
            logger.info(f"Attempting to load trace from JSON format...")
            trace = az.from_json(str(json_path))
            logger.info("[OK] Successfully loaded trace from JSON")
            return trace
        except Exception as e:
            logger.debug(f"JSON loading failed: {e}")
    
    # Try pickle fallback
    pickle_path = trace_path.with_suffix('.pkl')
    if pickle_path.exists():
        try:
            import pickle
            logger.info(f"Attempting to load trace from pickle format...")
            with open(pickle_path, 'rb') as f:
                trace = pickle.load(f)
            logger.info("[OK] Successfully loaded trace from pickle")
            return trace
        except Exception as e:
            logger.debug(f"Pickle loading failed: {e}")
    
    logger.error(f"Failed to load trace from any format (tried: NetCDF, JSON, pickle)")
    return None


def check_convergence(trace, config: ProjectConfig, logger) -> dict:
    """
    Check MCMC convergence diagnostics.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Convergence diagnostics
    """
    logger.info("\nChecking convergence diagnostics...")
    
    # Get thresholds from config
    rhat_threshold = config.get('diagnostics.convergence.rhat_threshold', 1.01)
    ess_bulk_min = config.get('diagnostics.convergence.ess_bulk_min', 400)
    ess_tail_min = config.get('diagnostics.convergence.ess_tail_min', 400)
    
    try:
        # Calculate diagnostics
        rhat = az.rhat(trace)
        ess_bulk = az.ess(trace, method='bulk')
        ess_tail = az.ess(trace, method='tail')
        
        # Check thresholds
        rhat_passed = (rhat.to_array() < rhat_threshold).all().values
        ess_bulk_passed = (ess_bulk.to_array() > ess_bulk_min).all().values
        ess_tail_passed = (ess_tail.to_array() > ess_tail_min).all().values
        
        diagnostics = {
            'rhat': {
                'min': float(rhat.to_array().min().values),
                'max': float(rhat.to_array().max().values),
                'threshold': rhat_threshold,
                'passed': bool(rhat_passed)
            },
            'ess_bulk': {
                'min': float(ess_bulk.to_array().min().values),
                'threshold': ess_bulk_min,
                'passed': bool(ess_bulk_passed)
            },
            'ess_tail': {
                'min': float(ess_tail.to_array().min().values),
                'threshold': ess_tail_min,
                'passed': bool(ess_tail_passed)
            },
            'overall_convergence': bool(rhat_passed and ess_bulk_passed and ess_tail_passed)
        }
        
        logger.info(f"\nConvergence Diagnostics:")
        logger.info(f"  R-hat: {diagnostics['rhat']['min']:.4f} - {diagnostics['rhat']['max']:.4f} "
                    f"(threshold: {rhat_threshold}) - {'[PASS]' if rhat_passed else '[FAIL]'}")
        logger.info(f"  ESS Bulk: min={diagnostics['ess_bulk']['min']:.0f} "
                    f"(threshold: {ess_bulk_min}) - {'[PASS]' if ess_bulk_passed else '[FAIL]'}")
        logger.info(f"  ESS Tail: min={diagnostics['ess_tail']['min']:.0f} "
                    f"(threshold: {ess_tail_min}) - {'[PASS]' if ess_tail_passed else '[FAIL]'}")
        logger.info(f"  Overall: {'[CONVERGED]' if diagnostics['overall_convergence'] else '[NOT CONVERGED]'}")
        
        # Create trace plots
        try:
            axes = az.plot_trace(trace, compact=False, figsize=(12, 8))
            fig = np.asarray(axes).ravel()[0].figure
            fig.tight_layout()
            fig_dir = config.get_path('figures')
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / 'Diagnostics_01_trace_plot.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved trace plot: {fig_path}")
        except Exception as e:
            logger.warning(f"Failed to create trace plot: {e}")
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Error in convergence diagnostics: {e}")
        return {
            'rhat': {'passed': False},
            'ess_bulk': {'passed': False},
            'ess_tail': {'passed': False},
            'overall_convergence': False,
            'error': str(e)
        }


def advanced_bayesian_diagnostics(trace, config: ProjectConfig, logger) -> dict:
    """
    Perform advanced Bayesian MCMC diagnostics including divergences, BFMI, 
    energy plots, tree depth, and PSIS-LOO.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace with sample_stats and log_likelihood
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Advanced diagnostic metrics
    """
    logger.info("\nPerforming advanced Bayesian diagnostics...")
    diagnostics = {}
    
    try:
        # 1. Check for divergences
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            divergences = trace.sample_stats.diverging.sum().values
            n_samples = len(trace.sample_stats.diverging.chain) * len(trace.sample_stats.diverging.draw)
            divergence_rate = divergences / n_samples if n_samples > 0 else 0
            
            diagnostics['divergences'] = {
                'count': int(divergences),
                'rate': float(divergence_rate),
                'passed': divergences == 0
            }
            
            if divergences > 0:
                logger.warning(f"  Divergences: {divergences} ({divergence_rate:.2%}) - [WARN]")
                logger.warning("    Consider increasing target_accept or reparameterizing model")
            else:
                logger.info(f"  Divergences: {divergences} - [PASS]")
        
        # 2. Check BFMI (Bayesian Fraction of Missing Information)
        try:
            bfmi_values = az.bfmi(trace)
            bfmi_min = float(bfmi_values.min())
            bfmi_threshold = 0.3  # Standard threshold
            
            diagnostics['bfmi'] = {
                'min': bfmi_min,
                'values': [float(v) for v in bfmi_values],
                'threshold': bfmi_threshold,
                'passed': bfmi_min > bfmi_threshold
            }
            
            if bfmi_min < bfmi_threshold:
                logger.warning(f"  BFMI: min={bfmi_min:.3f} (threshold: {bfmi_threshold}) - [WARN]")
                logger.warning("    Low BFMI may indicate poor exploration of parameter space")
            else:
                logger.info(f"  BFMI: min={bfmi_min:.3f} (threshold: {bfmi_threshold}) - [PASS]")
        except Exception as e:
            logger.warning(f"  BFMI calculation failed: {e}")
            diagnostics['bfmi'] = {'error': str(e)}
        
        # 3. Check tree depth (for NUTS sampler) with improved messaging (Issue E)
        if hasattr(trace, 'sample_stats') and 'tree_depth' in trace.sample_stats:
            tree_depths = trace.sample_stats.tree_depth.values.flatten()
            max_tree_depth = int(tree_depths.max())
            
            # Check if hitting max tree depth (default is usually 10)
            max_treedepth = config.get('modeling.bayesian.max_treedepth', 10)
            hitting_max = np.sum(tree_depths >= max_treedepth)
            hitting_max_rate = hitting_max / len(tree_depths) if len(tree_depths) > 0 else 0
            
            # Enhanced messaging (Issue E)
            diagnostics['tree_depth'] = {
                'max': max_tree_depth,
                'hitting_max_count': int(hitting_max),
                'hitting_max_rate': float(hitting_max_rate),
                'max_treedepth': max_treedepth,
                'passed': hitting_max_rate < 0.01  # Less than 1% hitting max
            }
            
            if hitting_max_rate > 0.05:
                logger.warning(f"  Tree depth: {hitting_max} iterations ({hitting_max_rate:.2%}) hit max depth {max_treedepth} - [WARN]")
                logger.warning("    Consider: (1) increasing max_treedepth, (2) increasing target_accept, (3) reparameterizing model")
                logger.warning(f"    Recommendation: set max_treedepth >= {max_tree_depth + 1} in config")
                diagnostics['tree_depth']['note'] = "High rate of hitting max_treedepth; may need parameter adjustment"
            elif hitting_max_rate > 0.01:
                logger.info(f"  Tree depth: max={max_tree_depth}, hitting_max={hitting_max} ({hitting_max_rate:.2%}) - [NOTE]")
                logger.info(f"    Small fraction hitting max; monitor if problematic")
                diagnostics['tree_depth']['note'] = "Small fraction hitting max_treedepth; generally acceptable"
            else:
                logger.info(f"  Tree depth: max={max_tree_depth}, hitting_max={hitting_max} - [PASS]")
        
        # 4. Create energy plot (to diagnose HMC transitions)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            az.plot_energy(trace, ax=ax)
            ax.set_title("Energy Transition Plot (HMC Diagnostic)")
            fig.tight_layout()
            
            fig_dir = config.get_path('figures')
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / 'Diagnostics_03_energy_plot.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  Saved energy plot: {fig_path}")
            diagnostics['energy_plot'] = str(fig_path)
        except Exception as e:
            logger.warning(f"  Energy plot failed: {e}")
            diagnostics['energy_plot'] = {'error': str(e)}
        
        # 5. PSIS-LOO (Pareto Smoothed Importance Sampling Leave-One-Out) with Issue G detection
        try:
            # Auto-detect log_likelihood variable (Issue C)
            ll_var_name = get_data_var_safely(
                trace.log_likelihood if hasattr(trace, 'log_likelihood') else None,
                preferred_names=['likelihood', 'y', 'y_obs'],
                data_type="log_likelihood"
            )
            
            if hasattr(trace, 'log_likelihood') and ll_var_name:
                loo = az.loo(trace, pointwise=True)
                
                # Extract Pareto k values
                pareto_k = loo.pareto_k.values
                
                # Classify Pareto k values (following ArviZ conventions):
                # k < 0.5: good
                # 0.5 <= k < 0.7: ok
                # 0.7 <= k < 1: bad
                # k >= 1: very bad
                k_good = np.sum(pareto_k < 0.5)
                k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
                k_bad = np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))
                k_very_bad = np.sum(pareto_k >= 1.0)
                
                diagnostics['psis_loo'] = {
                    'elpd_loo': float(loo.elpd_loo),
                    'p_loo': float(loo.p_loo),
                    'looic': float(loo.looic) if hasattr(loo, 'looic') else None,
                    'pareto_k': {
                        'good_count': int(k_good),
                        'ok_count': int(k_ok),
                        'bad_count': int(k_bad),
                        'very_bad_count': int(k_very_bad),
                        'max': float(pareto_k.max()),
                        'mean': float(pareto_k.mean())
                    },
                    'passed': k_bad + k_very_bad < len(pareto_k) * 0.05,  # Less than 5% problematic
                    'll_var_name': ll_var_name
                }
                
                logger.info(f"  PSIS-LOO (using '{ll_var_name}'):")
                logger.info(f"    ELPD LOO: {loo.elpd_loo:.2f}")
                logger.info(f"    p_loo: {loo.p_loo:.2f}")
                logger.info(f"    Pareto k: good={k_good}, ok={k_ok}, bad={k_bad}, very_bad={k_very_bad}")
                
                if k_bad + k_very_bad > 0:
                    logger.warning(f"    {k_bad + k_very_bad} observations have problematic Pareto k values - [WARN]")
                else:
                    logger.info(f"    All Pareto k values acceptable - [PASS]")
                
                # Create Pareto k diagnostic plot (Issue D: safer implementation)
                try:
                    fig_dir = config.get_path('figures')
                    fig_path = fig_dir / 'Diagnostics_04_pareto_k.png'
                    
                    # Use safer plotting approach without show_bins
                    fig, ax = plt.subplots(figsize=(10, 6))
                    az.plot_khat(loo, ax=ax)  # Avoid show_bins=True to prevent read-only assignment
                    ax.set_title("PSIS-LOO Pareto k Diagnostic")
                    fig.tight_layout()
                    
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"  Saved Pareto k plot: {fig_path}")
                    diagnostics['psis_loo']['plot_path'] = str(fig_path)
                except Exception as e2:
                    logger.warning(f"  ArviZ Pareto k plot creation failed: {e2}")
                    logger.info("    Attempting matplotlib fallback plot...")
                    
                    # Fallback: create simple matplotlib plot of Pareto-k values
                    try:
                        fig_dir = config.get_path('figures')
                        fig_path = fig_dir / 'Diagnostics_04_pareto_k_fallback.png'
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot histogram of Pareto-k values with threshold markers
                        ax.hist(pareto_k, bins=30, edgecolor='black', alpha=0.7, color='steelblue', label='Pareto k')
                        ax.axvline(x=0.5, color='yellow', linestyle='--', linewidth=2, label='Good/Ok threshold (0.5)')
                        ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Ok/Bad threshold (0.7)')
                        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Bad/Very bad threshold (1.0)')
                        
                        ax.set_xlabel('Pareto k', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                        ax.set_title('PSIS-LOO Pareto k Diagnostic (Fallback Plot)', fontsize=13, fontweight='bold')
                        ax.legend(fontsize=10)
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        fig.tight_layout()
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"  Saved Pareto k fallback plot: {fig_path}")
                        diagnostics['psis_loo']['plot_path'] = str(fig_path)
                        diagnostics['psis_loo']['plot_method'] = 'matplotlib_fallback'
                    except Exception as e3:
                        logger.warning(f"  Fallback plot also failed: {e3}")
                        logger.info("    Pareto k values logged as summary statistics only")
                        diagnostics['psis_loo']['plot_error'] = str(e2)
            else:
                logger.info("  PSIS-LOO: log_likelihood not available in trace")
                logger.info("    Run pm.compute_log_likelihood(trace) after sampling to enable PSIS-LOO")
                diagnostics['psis_loo'] = {'error': 'log_likelihood not available'}
        except Exception as e:
            logger.warning(f"  PSIS-LOO calculation failed: {e}")
            diagnostics['psis_loo'] = {'error': str(e)}
        
        # Overall assessment
        passed_checks = []
        if 'divergences' in diagnostics:
            passed_checks.append(diagnostics['divergences'].get('passed', False))
        if 'bfmi' in diagnostics and 'passed' in diagnostics['bfmi']:
            passed_checks.append(diagnostics['bfmi']['passed'])
        if 'tree_depth' in diagnostics:
            passed_checks.append(diagnostics['tree_depth']['passed'])
        if 'psis_loo' in diagnostics and 'passed' in diagnostics['psis_loo']:
            passed_checks.append(diagnostics['psis_loo']['passed'])
        
        diagnostics['overall_advanced_diagnostics'] = all(passed_checks) if passed_checks else None
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Error in advanced diagnostics: {e}", exc_info=True)
        return {'error': str(e)}


def posterior_predictive_checks(trace, df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Perform posterior predictive checks using actual posterior samples.
    
    IMPORTANT: These are IN-SAMPLE FIT DIAGNOSTICS (calibration checks).
    MAE, RMSE, and R2 are computed on observations used during model fitting.
    They do NOT constitute external validation or out-of-sample performance assessment.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace
    df : pd.DataFrame
        Modeling data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Posterior predictive check results (in-sample calibration only)
    """
    logger.info("\nPerforming posterior predictive checks (IN-SAMPLE CALIBRATION ONLY)...")
    
    try:
        # Step 0: Auto-detect observed RV name (Issue C: fragile key assumptions)
        obs_var_name = get_data_var_safely(
            trace.observed_data, 
            preferred_names=['likelihood', 'y', 'y_obs'],
            data_type="observed_data"
        )
        
        if obs_var_name is None:
            logger.error("No observed data variables found in trace.observed_data")
            return {
                'mae': None,
                'rmse': None,
                'r2': None,
                'n_observations': len(df),
                'error': 'No observed data variables in trace'
            }
        
        logger.info(f"[OK] Auto-detected observed RV: '{obs_var_name}'")
        
        # Step 1: Get observed data
        observed = trace.observed_data[obs_var_name].values
        n_observed = len(observed)
        n_df = len(df)
        
        # Issue B: Detect and clearly report mismatch
        if n_observed != n_df:
            logger.warning(f"\n[ISSUE B: TRACE-DATAFRAME MISMATCH]")
            logger.warning(f"  Posterior predictive check will use SUBSET of observations:")
            logger.warning(f"    Trace observed_data['{obs_var_name}']: {n_observed} observations")
            logger.warning(f"    Full dataframe: {n_df} observations")
            logger.warning(f"    Ratio: {n_observed/n_df:.1%}")
            logger.warning(f"\n  PPC metrics below are IN-SAMPLE FIT DIAGNOSTICS")
            logger.warning(f"  (evaluated on the {n_observed} observations used during model fitting)")
            logger.warning(f"  Note: Index mapping from df to observed RV unavailable in this context")
        else:
            logger.info(f"  Trace and DataFrame match: {n_observed} observations")
        
        logger.info(f"Extracted {n_observed} observations from trace.observed_data['{obs_var_name}']")
        
        # Step 2: Get fitted values (mu) from posterior
        if 'mu' not in trace.posterior.data_vars:
            logger.error("Fitted values 'mu' not found in trace.posterior. "
                        "Model must include pm.Deterministic('mu', mu) before likelihood.")
            return {
                'mae': None,
                'rmse': None,
                'r2': None,
                'n_observations': n_observed,
                'error': 'No fitted values (mu) in trace'
            }
        
        # Extract posterior mean of fitted values
        posterior_mean = trace.posterior['mu'].mean(dim=('chain', 'draw')).values
        logger.info(f"Extracted {len(posterior_mean)} fitted values from trace.posterior['mu']")
        
        # Step 3: Get posterior predictive samples (if available)
        posterior_predictive_samples = None
        pp_var_name = get_data_var_safely(
            trace.posterior_predictive if hasattr(trace, 'posterior_predictive') else None,
            preferred_names=['likelihood', 'y', 'y_obs'],
            data_type="posterior_predictive"
        )
        
        if pp_var_name and hasattr(trace, 'posterior_predictive'):
            try:
                posterior_predictive_samples = trace.posterior_predictive[pp_var_name].values
                logger.info(f"Found posterior predictive samples: shape {posterior_predictive_samples.shape}")
            except Exception as e:
                logger.debug(f"Posterior predictive samples retrieval failed: {e}")
        else:
            logger.info("No posterior predictive samples found in trace")
        
        # Ensure matching lengths
        min_len = min(len(observed), len(posterior_mean))
        observed = observed[:min_len]
        posterior_mean = posterior_mean[:min_len]
        
        # Calculate goodness of fit metrics
        mae = mean_absolute_error(observed, posterior_mean)
        rmse = np.sqrt(mean_squared_error(observed, posterior_mean))
        
        # Calculate R2 (handle edge case where variance is 0)
        ss_res = np.sum((observed - posterior_mean) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0
            logger.warning("Total sum of squares is zero, R2 set to 0")
        
        # Calculate additional metrics
        residuals = observed - posterior_mean
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Calculate coverage (what % of observations are within posterior predictive interval)
        coverage = None
        if posterior_predictive_samples is not None:
            pp_samples = posterior_predictive_samples.reshape(-1, len(observed))
            lower = np.percentile(pp_samples, 2.5, axis=0)
            upper = np.percentile(pp_samples, 97.5, axis=0)
            coverage = np.mean((observed >= lower) & (observed <= upper))
            logger.info(f"  95% coverage: {coverage:.1%}")
        
        results = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual),
            'n_observations': int(min_len),
            'n_dataframe': int(n_df),
            'obs_var_name': obs_var_name,
            'diagnotic_type': 'IN-SAMPLE CALIBRATION (not external validation)',
            'note_sample_type': 'in-sample fit diagnostics' if n_observed != n_df else 'full data',
            'coverage_95': float(coverage) if coverage is not None else None,
            'scientific_note': 'These metrics reflect goodness-of-fit to training data used in model fitting. They do NOT assess generalization or out-of-sample performance. See LOSO validation framework for cross-validation structure.'
        }
        
        if n_observed != n_df:
            results['note_mismatch'] = f"Trace subset ({n_observed}) vs full DataFrame ({n_df})"
        
        logger.info(f"  MAE:  {mae:.3f}")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  R2:   {r2:.3f}")
        logger.info(f"  Mean residual: {mean_residual:.3f}")
        logger.info(f"  Std residual:  {std_residual:.3f}")
        
        # Create enhanced diagnostic plots (2x2 grid)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            # Plot 1: Observed vs Predicted with credible intervals
            if posterior_predictive_samples is not None:
                pp_samples = posterior_predictive_samples.reshape(-1, len(observed))
                lower = np.percentile(pp_samples, 2.5, axis=0)
                upper = np.percentile(pp_samples, 97.5, axis=0)
                
                # Sort for better visualization
                sort_idx = np.argsort(observed)
                obs_sorted = observed[sort_idx]
                pred_sorted = posterior_mean[sort_idx]
                lower_sorted = lower[sort_idx]
                upper_sorted = upper[sort_idx]
                
                axes[0].scatter(obs_sorted, pred_sorted, alpha=0.6, s=40, c='steelblue', edgecolors='navy', linewidth=0.5)
                axes[0].fill_between(obs_sorted, lower_sorted, upper_sorted, alpha=0.2, color='steelblue', label='95% CI')
            else:
                axes[0].scatter(observed, posterior_mean, alpha=0.6, s=40, c='steelblue', edgecolors='navy', linewidth=0.5)
            
            min_val, max_val = observed.min(), observed.max()
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit', lw=2)
            axes[0].set_xlabel('Observed log(Qm)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Predicted log(Qm)', fontsize=12, fontweight='bold')
            axes[0].set_title(f'Observed vs Predicted (n={len(observed)})\nR2 = {r2:.3f}, RMSE = {rmse:.3f}', fontsize=13, fontweight='bold')
            axes[0].legend(loc='upper left', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Residuals vs Fitted
            residuals_copy = residuals.copy()  # Avoid redefining
            axes[1].scatter(posterior_mean, residuals_copy, alpha=0.6, s=40, c='coral', edgecolors='darkred', linewidth=0.5)
            axes[1].axhline(y=0, color='red', linestyle='--', lw=2, label='Zero residual')
            axes[1].axhline(y=2*std_residual, color='orange', linestyle=':', lw=1.5, alpha=0.7, label='+/-2 SD')
            axes[1].axhline(y=-2*std_residual, color='orange', linestyle=':', lw=1.5, alpha=0.7)
            axes[1].set_xlabel('Fitted values', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Residuals vs Fitted\nMean = {mean_residual:.3f}, SD = {std_residual:.3f}', 
                            fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Residual histogram with normal curve
            n_bins = min(30, max(10, len(residuals)//5))
            counts, bins, _ = axes[2].hist(residuals, bins=n_bins, edgecolor='black', alpha=0.7, 
                                          color='steelblue', density=True, label='Residuals')
            
            # Overlay normal distribution
            from scipy.stats import norm
            x = np.linspace(residuals.min(), residuals.max(), 100)
            axes[2].plot(x, norm.pdf(x, mean_residual, std_residual), 'r-', lw=2, label='Normal fit')
            axes[2].set_xlabel('Residuals', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Density', fontsize=12, fontweight='bold')
            axes[2].set_title(f'Residual Distribution (n={len(residuals)})', fontsize=13, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Q-Q plot
            from scipy import stats as sp_stats
            sp_stats.probplot(residuals, dist="norm", plot=axes[3])
            axes[3].set_title('Q-Q Plot (Normal)', fontsize=13, fontweight='bold')
            axes[3].set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
            axes[3].set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_dir = config.get_path('figures')
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / 'Diagnostics_02_posterior_predictive.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved enhanced posterior predictive plots: {fig_path}")
        except Exception as e:
            logger.warning(f"Failed to create posterior predictive plots: {e}", exc_info=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in posterior predictive checks: {e}", exc_info=True)
        return {
            'mae': None,
            'rmse': None,
            'r2': None,
            'n_observations': 0,
            'error': str(e)
        }


def assess_multicollinearity(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Assess multicollinearity using VIF and correlation analysis for continuous predictors.
    
    NOTE: VIF analysis focuses on continuous predictors in the PRIMARY MODEL ONLY.
    Categorical predictors (Metal, ReT, AgS) use reference coding and don't have meaningful VIF.
    
    Primary model predictors: Temp (continuous); Metal, ReT, AgS (categorical, reference-coded)
    
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
        VIF and correlation results
    """
    logger.info("\nAssessing multicollinearity (primary model continuous predictors)...")
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Define continuous predictors used in the PRIMARY model only.
        # Primary model: log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)
        # Categorical predictors (Metal, ReT, AgS) use reference coding and don't have meaningful VIF.
        # Only continuous predictor: Temp (and standardized variants)
        potential_predictors = [
            'Temp', 'Temperature', 'Temp_C',  # Only continuous in primary model
            'Temp_standardized', 'Temperature_standardized'
        ]
        
        # Find which of these columns exist in the dataframe
        available_cols = df.columns.tolist()
        predictors = [col for col in potential_predictors if col in available_cols]
        
        if len(predictors) < 1:
            logger.warning("No continuous model predictors found in dataframe for VIF analysis")
            logger.info("  NOTE: Categorical predictors (Metal, ReT, AgS) use reference coding.")
            logger.info("        Reference-coded categorical variables do not have meaningful VIF values.")
            return {
                'vif_values': [],
                'n_high_vif': 0,
                'threshold': config.get('diagnostics.multicollinearity.vif_threshold', 10.0),
                'note': 'Primary model uses reference-coded categorical predictors; VIF not applicable',
                'categorical_predictors': ['Metal', 'ReT', 'AgS'],
                'categorical_encoding': 'reference_coding'
            }
        
        # Remove columns with missing values
        df_vif = df[predictors].dropna()
        
        if len(df_vif) < 10:
            logger.warning(f"Insufficient data for VIF ({len(df_vif)} rows, {len(predictors)} predictors)")
            return {
                'vif_values': [],
                'n_high_vif': 0,
                'threshold': config.get('diagnostics.multicollinearity.vif_threshold', 10.0),
                'warning': f'Insufficient data (n={len(df_vif)})'
            }
        
        if len(predictors) < 2:
            logger.info(f"Only 1 predictor found ({predictors[0]}), VIF not applicable")
            return {
                'vif_values': [{'Variable': predictors[0], 'VIF': 1.0, 'High_VIF': False}],
                'n_high_vif': 0,
                'threshold': config.get('diagnostics.multicollinearity.vif_threshold', 10.0)
            }
        
        logger.info(f"Calculating VIF for {len(predictors)} primary model continuous predictor(s): {predictors}")
        logger.info(f"Using {len(df_vif)} complete cases")
        
        # Log categorical predictor note
        logger.info(f"NOTE: Categorical predictors (Metal, ReT, AgS) use reference coding (K-1 parameterization).")
        logger.info(f"      These do not require VIF assessment.")
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = predictors
        vif_values = []
        
        for i in range(len(predictors)):
            try:
                vif = variance_inflation_factor(df_vif.values, i)
                # Handle inf or very large VIF values
                if np.isinf(vif) or vif > 1000:
                    logger.warning(f"  {predictors[i]}: VIF is infinite or very large (perfect collinearity)")
                    vif_values.append(np.nan)
                else:
                    vif_values.append(vif)
            except Exception as e:
                logger.warning(f"  Failed to calculate VIF for {predictors[i]}: {e}")
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        
        # Filter out NaN values for threshold checking
        vif_data_valid = vif_data[vif_data['VIF'].notna()].copy()
        
        if len(vif_data_valid) == 0:
            logger.warning("Could not calculate VIF for any variables")
            return {
                'vif_values': vif_data.to_dict('records'),
                'n_high_vif': 0,
                'threshold': config.get('diagnostics.multicollinearity.vif_threshold', 10.0),
                'warning': 'VIF calculation failed for all variables'
            }
        
        vif_data_valid = vif_data_valid.sort_values('VIF', ascending=False)
        
        threshold = config.get('diagnostics.multicollinearity.vif_threshold', 10.0)
        vif_data_valid['High_VIF'] = vif_data_valid['VIF'] > threshold
        
        logger.info(f"\nVIF Results (threshold={threshold}):")
        for _, row in vif_data_valid.iterrows():
            status = "[FAIL]" if row['High_VIF'] else "[PASS]"
            logger.info(f"  {row['Variable']:20s} VIF={row['VIF']:8.2f} {status}")
        
        n_high_vif = vif_data_valid['High_VIF'].sum()
        
        # Issue F: Add Pearson and Spearman correlation analysis
        logger.info(f"\n[ISSUE F: CORRELATION ANALYSIS]")
        
        # Pearson correlation
        pearson_corr = df_vif.corr(method='pearson')
        
        # Identify high correlations (>0.7)
        high_corr_pairs = []
        for i in range(len(predictors)):
            for j in range(i+1, len(predictors)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'var1': predictors[i],
                        'var2': predictors[j],
                        'pearson_r': float(corr_val),
                        'r_squared': float(corr_val**2)
                    })
        
        if high_corr_pairs:
            logger.warning(f"  Found {len(high_corr_pairs)} highly correlated predictor pair(s) (|r| > 0.7):")
            for pair in high_corr_pairs:
                logger.warning(f"    {pair['var1']} <-> {pair['var2']}: r = {pair['pearson_r']:.3f}, "
                              f"r2 = {pair['r_squared']:.3f}")
            
            if n_high_vif > 0:
                logger.warning(f"  NOTE (Bayesian context): High VIF ({n_high_vif} variables) combined with")
                logger.warning(f"    high correlations affects POSTERIOR IDENTIFIABILITY more than")
                logger.warning(f"    predictive performance. Check posterior correlations in trace plots.")
        else:
            logger.info(f"  Pearson correlation: no |r| > 0.7 pairs detected")
        
        # Spearman correlation (for ranking)
        spearman_corr = df_vif.corr(method='spearman')
        
        if n_high_vif > 0:
            logger.warning(f"[WARN] {n_high_vif} variables have VIF > {threshold}")
            logger.warning("  VIF Interpretation (Bayesian context):")
            logger.warning("    - High VIF primarily affects posterior correlation/identifiability")
            logger.warning("    - Predictive fit may still be good if posterior uncertainties are properly calibrated")
            logger.warning("    - Consider: (1) Bayesian model averaging, (2) regularization via priors,")
            logger.warning("               (3) combining collinear variables, or (4) refitting with fewer predictors")
        else:
            logger.info(f"[PASS] All variables have VIF < {threshold}")
        
        results = {
            'vif_values': vif_data_valid.to_dict('records'),
            'n_high_vif': int(n_high_vif),
            'threshold': threshold,
            'n_predictors': len(predictors),
            'n_complete_cases': len(df_vif),
            'high_correlation_pairs': high_corr_pairs,
            'note_bayesian_context': 'High VIF affects posterior identifiability; predictive fit may still be adequate'
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in multicollinearity assessment: {e}", exc_info=True)
        return {'error': str(e), 'vif_values': [], 'n_high_vif': 0}


def leave_one_study_out_validation(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Framework for leave-one-study-out cross-validation.
    
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
        LOSO validation framework and summary
    """
    logger.info("\n[LOSO VALIDATION: FRAMEWORK STRUCTURE ONLY]")
    logger.info("Note: This section identifies the study structure for LOSO cross-validation")
    logger.info("      but does NOT perform actual leave-one-study-out refitting.")
    
    try:
        # Find study column
        study_col = None
        for col in df.columns:
            if 'study' in col.lower() and 'id' in col.lower():
                study_col = col
                break
        
        if study_col is None:
            logger.warning("Study ID column not found. Skipping LOSO validation.")
            return {'status': 'skipped', 'reason': 'Study ID not found'}
        
        studies = df[study_col].dropna().unique()
        n_studies = len(studies)
        
        logger.info(f"Found {n_studies} studies for LOSO validation framework")
        
        # Provide framework statistics
        study_sizes = df[study_col].value_counts().sort_values(ascending=False)
        
        logger.info(f"\nStudy Summary:")
        logger.info(f"  Total studies: {n_studies}")
        logger.info(f"  Min study size: {study_sizes.min()}")
        logger.info(f"  Max study size: {study_sizes.max()}")
        logger.info(f"  Median study size: {study_sizes.median():.0f}")
        logger.info(f"  Mean study size: {study_sizes.mean():.1f}")
        
        results = {
            'n_studies': int(n_studies),
            'method': 'leave_one_study_out',
            'status': 'framework_structure_available',
            'validation_performed': False,
            'study_size_stats': {
                'min': int(study_sizes.min()),
                'max': int(study_sizes.max()),
                'median': float(study_sizes.median()),
                'mean': float(study_sizes.mean())
            },
            'note': '(FRAMEWORK ONLY) LOSO validation structure identified but NOT YET EXECUTED. Full implementation requires model refitting for each fold (n_studies={} refits needed).'.format(n_studies)
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in LOSO validation analysis: {e}")
        return {'status': 'error', 'error': str(e)}


def influence_diagnostics_with_pareto_k(trace, df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Issue G: Identify high-influence observations via Pareto-k and aggregate by study.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace with PSIS-LOO
    df : pd.DataFrame
        Modeling data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Influence diagnostics with study aggregation
    """
    logger.info("\n[ISSUE G: INFLUENCE DIAGNOSTICS]")
    
    try:
        # Auto-detect log_likelihood variable
        ll_var_name = get_data_var_safely(
            trace.log_likelihood if hasattr(trace, 'log_likelihood') else None,
            preferred_names=['likelihood', 'y', 'y_obs'],
            data_type="log_likelihood"
        )
        
        if not hasattr(trace, 'log_likelihood') or ll_var_name is None:
            logger.info("  Log likelihood not available; skipping Pareto-k influence analysis")
            return {'status': 'skipped', 'reason': 'log_likelihood not available'}
        
        # Calculate PSIS-LOO
        try:
            loo = az.loo(trace, pointwise=True)
            pareto_k = loo.pareto_k.values
        except Exception as e:
            logger.warning(f"  PSIS-LOO calculation failed: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # Identify problematic observations
        k_bad = np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))
        k_very_bad = np.sum(pareto_k >= 1.0)
        
        if k_bad + k_very_bad == 0:
            logger.info("  No problematic Pareto k values detected (all <0.7)")
            return {'status': 'passed', 'n_problematic': 0, 'problematic_obs': []}
        
        logger.warning(f"  Found {k_bad} 'bad' and {k_very_bad} 'very bad' Pareto k observations")
        
        # Find indices of problematic observations
        bad_idx = np.where((pareto_k >= 0.7) & (pareto_k < 1.0))[0]
        very_bad_idx = np.where(pareto_k >= 1.0)[0]
        all_problematic_idx = np.concatenate([very_bad_idx, bad_idx])
        
        # Try to map to study if study column exists
        # WARNING: Index mapping from trace observations to df rows is PROVISIONAL.
        # This mapping assumes the trace observations are ordered consistently with df rows,
        # but this is NOT guaranteed without explicit row matching logic.
        study_col = None
        for col in df.columns:
            if 'study' in col.lower() and 'id' in col.lower():
                study_col = col
                break
        
        problematic_obs = []
        study_aggregation = {}
        has_exact_study_mapping = False  # Flag: true only if we can guarantee exact mapping
        
        for idx in all_problematic_idx:
            severity = 'very_bad' if idx in very_bad_idx else 'bad'
            k_val = pareto_k[idx]
            
            obs_info = {
                'index': int(idx),
                'pareto_k': float(k_val),
                'severity': severity
            }
            
            # Add study info if available (PROVISIONAL MAPPING: approximate)
            if study_col and idx < len(df):
                study_id = df.iloc[idx][study_col]
                obs_info['study_id'] = str(study_id)
                obs_info['row_mapping'] = 'provisional'  # Flag: not guaranteed exact
                
                # Aggregate by study (with caution: mapping is approximate)
                if study_id not in study_aggregation:
                    study_aggregation[study_id] = {
                        'bad_count': 0,
                        'very_bad_count': 0,
                        'mean_k': 0.0,
                        'max_k': 0.0,
                        'n_obs_in_study': int((df[study_col] == study_id).sum()),
                        'mapping_status': 'provisional'  # Caution flag
                    }
                
                if severity == 'very_bad':
                    study_aggregation[study_id]['very_bad_count'] += 1
                else:
                    study_aggregation[study_id]['bad_count'] += 1
                
                study_aggregation[study_id]['mean_k'] += k_val / len(all_problematic_idx)
                study_aggregation[study_id]['max_k'] = max(study_aggregation[study_id]['max_k'], k_val)
            
            problematic_obs.append(obs_info)
        
        # Log study-level summary (PROVISIONAL MAPPING)
        if study_aggregation:
            logger.warning(f"  Study-level aggregation (PROVISIONAL row mapping; top contributing studies):")
            logger.warning(f"    Note: Study assignment is approximate (indices may not align exactly with df rows)")
            sorted_studies = sorted(
                study_aggregation.items(),
                key=lambda x: x[1]['very_bad_count'] + x[1]['bad_count'],
                reverse=True
            )
            
            for study_id, stats in sorted_studies[:5]:  # Top 5
                logger.warning(f"    Study {study_id}: "
                              f"{stats['very_bad_count']} very_bad + {stats['bad_count']} bad; "
                              f"max_k={stats['max_k']:.3f}; study_size={stats['n_obs_in_study']} [PROVISIONAL]")
        
        logger.info(f"  Identified {len(all_problematic_idx)} high-influence observations")
        logger.info("  Note: Index-to-study mapping is PROVISIONAL and approximate")
        
        return {
            'status': 'detected',
            'n_problematic': int(len(all_problematic_idx)),
            'n_very_bad': int(k_very_bad),
            'n_bad': int(k_bad),
            'problematic_obs': problematic_obs,
            'study_aggregation': study_aggregation if study_aggregation else None,
            'row_mapping_status': 'provisional',
            'note': 'High-influence observations via Pareto-k smoothness diagnostic (study mapping approximate)'
        }
        
    except Exception as e:
        logger.error(f"Error in influence diagnostics: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


def stratified_posterior_predictive_checks(trace, df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Issue H: Stratified PPC and residual checks by key strata (Metal, Resin Type, Aging).
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace
    df : pd.DataFrame
        Modeling data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Stratified diagnostics
    """
    logger.info("\n[ISSUE H: STRATIFIED PPC AND RESIDUAL CHECKS]")
    
    try:
        # Get fitted values
        if 'mu' not in trace.posterior.data_vars:
            logger.info("  Fitted values not available; skipping stratified analysis")
            return {'status': 'skipped', 'reason': 'No fitted values'}
        
        # Auto-detect obs var
        obs_var_name = get_data_var_safely(
            trace.observed_data,
            preferred_names=['likelihood', 'y', 'y_obs'],
            data_type="observed_data"
        )
        
        if obs_var_name is None or not hasattr(trace, 'observed_data'):
            logger.info("  Observed data not available; skipping stratified analysis")
            return {'status': 'skipped', 'reason': 'No observed data'}
        
        observed = trace.observed_data[obs_var_name].values
        posterior_mean = trace.posterior['mu'].mean(dim=('chain', 'draw')).values
        residuals = observed - posterior_mean
        
        # Identify strata columns with sufficient diversity
        # Primary model predictors: Metal, ReT, AgS, Temp (continuous)
        # Stratifying by categorical predictors to assess fit heterogeneity
        strata_candidates = []
        for col in ['Metal', 'ReT', 'AgS', 'Aged condition']:  # AgS variants for compatibility
            if col in df.columns:
                n_unique = df[col].nunique()
                if n_unique >= 2:  # At least 2 groups
                    strata_candidates.append((col, n_unique))
        
        if not strata_candidates:
            logger.info("  No suitable strata columns found")
            return {'status': 'skipped', 'reason': 'No strata columns'}
        
        strata_results = {}
        logger.info(f"  Found {len(strata_candidates)} potential strata: {[s[0] for s in strata_candidates]}")
        
        # Sample size requirement for stratified analysis
        min_stratum_size = max(5, len(observed) // 10)
        
        # Prioritize primary model categorical predictors: Metal, ReT, AgS
        primary_strata_order = ['Metal', 'ReT', 'AgS']
        reordered_strata = sorted(strata_candidates, key=lambda x: (x[0] not in primary_strata_order, primary_strata_order.index(x[0]) if x[0] in primary_strata_order else 999))
        
        for stratum_col, n_groups in reordered_strata:  # Analyze all strata, prioritizing primary model predictors
            logger.info(f"\n  Stratifying by {stratum_col} ({n_groups} groups)...")
            
            stratified_metrics = {}
            sufficient_strata = 0
            
            for group_val in df[stratum_col].unique():
                if pd.isna(group_val):
                    continue
                
                # Get mask for this stratum (approximate mapping)
                mask = (df[stratum_col] == group_val).values[:len(observed)]
                
                if mask.sum() < min_stratum_size:
                    logger.debug(f"    {group_val}: n={mask.sum()} (too small, skipping)")
                    continue
                
                sufficient_strata += 1
                
                # Calculate metrics for this stratum
                obs_stratum = observed[mask]
                pred_stratum = posterior_mean[mask]
                res_stratum = residuals[mask]
                
                mae = mean_absolute_error(obs_stratum, pred_stratum)
                rmse = np.sqrt(mean_squared_error(obs_stratum, pred_stratum))
                ss_res = np.sum((obs_stratum - pred_stratum) ** 2)
                ss_tot = np.sum((obs_stratum - np.mean(obs_stratum)) ** 2)
                r2 = (1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
                
                stratified_metrics[str(group_val)] = {
                    'n': int(mask.sum()),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'residual_mean': float(np.mean(res_stratum)),
                    'residual_std': float(np.std(res_stratum))
                }
                
                logger.info(f"    {group_val:15s}: n={mask.sum():3d}, MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
            
            if sufficient_strata > 0:
                # Store results
                strata_results[stratum_col] = {
                    'metrics': stratified_metrics,
                    'n_sufficient_strata': sufficient_strata
                }
            else:
                logger.info(f"    No strata with n >= {min_stratum_size}")
        
        # Check overall heteroskedasticity (residual spread vs fitted) (Issue H)
        # Note: residuals/fitted are based on trace (n_obs observations), not full df (n_df observations)
        logger.info(f"\n  Checking for heteroskedasticity (based on {len(residuals)} trace observations)...")
        
        if len(residuals) >= 3:
            try:
                # Divide fitted values into thirds based on trace subset
                sorted_fitted_indices = np.argsort(posterior_mean)
                third_size = len(posterior_mean) // 3
                thirds_idx = [
                    sorted_fitted_indices[:third_size],
                    sorted_fitted_indices[third_size:2*third_size],
                    sorted_fitted_indices[2*third_size:]
                ]
                
                spreads = [np.std(residuals[idx]) if len(idx) > 0 else 0 for idx in thirds_idx]
                spread_ratio = max(spreads) / (min(spreads) + 1e-10)  # Add small constant to avoid division by zero
                
                if spread_ratio > 2.0:
                    logger.warning(f"    Residual spread increases with fitted values (ratio={spread_ratio:.2f})")
                    logger.warning("    Consider: (1) heteroskedastic sigma model, (2) Student-t likelihood, (3) variable transformation")
                    strata_results['heteroskedasticity'] = {
                        'detected': True,
                        'spread_ratio': float(spread_ratio),
                        'recommendation': 'Student-t likelihood or heteroskedastic sigma'
                    }
                else:
                    logger.info(f"    Residual spread is homogeneous (ratio={spread_ratio:.2f})")
                    strata_results['heteroskedasticity'] = {'detected': False, 'spread_ratio': float(spread_ratio)}
            except Exception as e:
                logger.debug(f"    Heteroskedasticity check failed: {e}")
                strata_results['heteroskedasticity'] = {'error': str(e)}
        else:
            logger.info(f"    Insufficient observations ({len(residuals)}) for heteroskedasticity check")
        
        logger.info(f"  Stratified analysis complete")
        
        return {'status': 'completed', 'results': strata_results}
        
    except Exception as e:
        logger.error(f"Error in stratified PPC: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '06_model_diagnostics')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 06: MODEL DIAGNOSTICS", logger=logger)
        
        # Load models
        models_dir = config.get_path('models')
        
        # Load primary model trace with fallback
        logger.info("Loading primary model trace...")
        primary_trace = load_trace_with_fallback(models_dir / "primary_model_trace.nc", logger)
        
        if primary_trace is None:
            logger.error("Could not load primary model trace. Exiting.")
            return 1
        
        # Defensive check: warn if outdated alternative_model files exist
        outdated_paths = [
            models_dir / "alternative_model_trace.nc",
            models_dir / "alternative_model_summary.json"
        ]
        for outdated_path in outdated_paths:
            if outdated_path.exists():
                logger.warning(
                    f"Outdated file detected: {outdated_path.name}. "
                    "This has been renamed to 'main_mechanistic_model_*'. "
                    "Please rerun Script 05 to generate updated files."
                )
        
        # Load data
        data_path = config.get_path('processed_data') / "03_enriched_data.csv"
        df = load_dataframe(data_path)
        logger.info(f"Loaded {len(df)} observations")
        
        diagnostics_results = {}
        
        # Step 1: Convergence diagnostics
        diagnostics_results['convergence'] = check_convergence(primary_trace, config, logger)
        
        # Step 2: Advanced Bayesian diagnostics
        diagnostics_results['advanced_diagnostics'] = advanced_bayesian_diagnostics(primary_trace, config, logger)
        
        # Step 3: Posterior predictive checks
        diagnostics_results['posterior_predictive'] = posterior_predictive_checks(
            primary_trace, df, config, logger)
        
        # Step 4: Multicollinearity
        diagnostics_results['multicollinearity'] = assess_multicollinearity(df, config, logger)
        
        # Step 5: LOSO validation framework
        diagnostics_results['loso_validation'] = leave_one_study_out_validation(df, config, logger)
        
        # Step 6 (Issue G): Influence diagnostics with Pareto-k
        diagnostics_results['influence_diagnostics'] = influence_diagnostics_with_pareto_k(
            primary_trace, df, config, logger)
        
        # Step 7 (Issue H): Stratified PPC
        diagnostics_results['stratified_ppc'] = stratified_posterior_predictive_checks(
            primary_trace, df, config, logger)
        
        # Save results
        results_dir = config.get_path('results')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "06_diagnostics_results.json"
        save_json(diagnostics_results, results_path)
        logger.info(f"\n[OK] Saved diagnostics results: {results_path}")
        
        logger.info("\n" + "="*60)
        logger.info("MODEL DIAGNOSTICS COMPLETED [SUCCESS]")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in model diagnostics: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
