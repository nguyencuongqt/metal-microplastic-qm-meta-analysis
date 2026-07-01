"""
Script 05: Hierarchical Bayesian Modeling
=========================================

Fit hierarchical Bayesian mixed-effects models for meta-analysis.

Models:
1. Primary (core dataset):
    log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)
    (categorical predictors use reference coding for identifiability)
2. Main mechanistic model:
    log(Qm) ~ Hydration_Energy + pH + AgS + (1|Study)
3. Supplementary SA sensitivity model (optional):
    log(Qm) ~ Hydration_Energy + pH + SA + AgS + (1|Study)

Uses PyMC for Bayesian inference with MCMC sampling.

Author: Manuscript authors
Date: March 2026
"""

import os
# Configure PyTensor via environment BEFORE any imports that use it
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import warnings
warnings.filterwarnings('ignore')

# Import pandas API utilities
from pandas.api import types as pd_types

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, save_model, save_json, print_section_header)


def _log_model_row_loss(df: pd.DataFrame, required_cols: list, model_name: str, logger) -> pd.DataFrame:
    """Log transparent row-loss reasons and return complete-case subset."""
    logger.info(f"\n{model_name} row-loss audit:")
    logger.info(f"  Starting rows: {len(df)}")

    for col in required_cols:
        missing_n = int(df[col].isna().sum())
        logger.info(f"  Missing in {col}: {missing_n}")

    df_complete = df[required_cols].dropna().copy()
    dropped_n = int(len(df) - len(df_complete))
    logger.info(f"  Final complete-case rows: {len(df_complete)}")
    logger.info(f"  Rows dropped total: {dropped_n}")

    return df_complete


def _encode_categorical_with_reference(df: pd.DataFrame, col_name: str, logger) -> tuple:
    """
    Encode categorical values with deterministic ordering and reference level.

    Returns
    -------
    tuple
        (codes, levels, reference_level)
    """
    values = df[col_name]
    categories = sorted(values.dropna().astype(str).unique().tolist())
    if len(categories) < 2:
        raise ValueError(f"{col_name} needs at least 2 levels for reference coding; found {categories}")

    reference_level = categories[0]
    code_map = {lvl: idx for idx, lvl in enumerate(categories)}
    codes = values.astype(str).map(code_map).astype(int).values

    logger.info(
        f"Reference coding for {col_name}: reference='{reference_level}', "
        f"non-reference levels={categories[1:]}"
    )
    return codes, categories, reference_level


def _build_reference_effect(name: str, codes: np.ndarray, levels: list, beta_mu: float, beta_sigma: float):
    """Create reference-coded categorical effects (K-1 coefficients + global intercept)."""
    n_nonref = len(levels) - 1
    beta_nonref = pm.Normal(name, mu=beta_mu, sigma=beta_sigma, shape=n_nonref)
    beta_full = pt.concatenate([pt.zeros(1), beta_nonref])
    return beta_full[codes]


def _extract_sampler_diagnostics(trace) -> dict:
    """Extract sampler diagnostics, including explicit max_treedepth hits."""
    diagnostics = {
        'divergences': 0,
        'max_treedepth_hits': 0,
        'max_observed_tree_depth': None,
        'has_max_treedepth_stat': False
    }

    if not hasattr(trace, 'sample_stats'):
        return diagnostics

    if 'diverging' in trace.sample_stats:
        diagnostics['divergences'] = int(trace.sample_stats['diverging'].sum().values)

    if 'tree_depth' in trace.sample_stats:
        diagnostics['max_observed_tree_depth'] = int(trace.sample_stats['tree_depth'].max().values)

    if 'reached_max_treedepth' in trace.sample_stats:
        diagnostics['has_max_treedepth_stat'] = True
        diagnostics['max_treedepth_hits'] = int(trace.sample_stats['reached_max_treedepth'].sum().values)

    return diagnostics


def prepare_modeling_data(df: pd.DataFrame, logger) -> tuple:
    """
    Prepare data for modeling by handling missing values and encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (df_model, categorical_mappings, numeric_columns)
    """
    logger.info("Preparing data for modeling...")
    
    df_model = df.copy()
    
    # Remove rows with missing log_Qm
    initial_n = len(df_model)
    df_model = df_model.dropna(subset=['log_Qm'])
    removed_n = initial_n - len(df_model)
    
    if removed_n > 0:
        logger.info(f"Removed {removed_n} rows with missing log_Qm")
    
    # Identify and encode categorical variables
    categorical_mappings = {}
    
    # Exclude metadata and technical columns from encoding
    exclude_cols = {'quality_flags', 'quality_category', 'Study_ID', 'Study_ID_numeric', 
                   'outlier_flag', 'log_Qm', 'Qm', 'quality_score'}
    
    for col in df_model.columns:
        if df_model[col].dtype == 'object' and col not in exclude_cols:
            # Create numeric encoding
            unique_vals = df_model[col].dropna().unique()
            if len(unique_vals) > 0 and len(unique_vals) < 50:  # Reasonable number of categories
                categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
                df_model[f'{col}_encoded'] = df_model[col].map(categorical_mappings[col])
                logger.info(f"Encoded {col}: {len(unique_vals)} categories")
    
    # Re-encode Study_ID_numeric to be local indices (0-based consecutive)
    if 'Study_ID_numeric' in df_model.columns:
        unique_studies = sorted(df_model['Study_ID_numeric'].dropna().unique())
        study_mapping = {study_id: idx for idx, study_id in enumerate(unique_studies)}
        df_model['Study_ID_local'] = df_model['Study_ID_numeric'].map(study_mapping)
        categorical_mappings['Study_ID_numeric'] = study_mapping
        logger.info(f"Re-encoded Study_ID_numeric: {len(unique_studies)} unique studies")
    
    # Identify numeric columns for modeling
    numeric_columns = df_model.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"\nFinal modeling dataset: {len(df_model)} rows, {len(df_model.columns)} columns")
    logger.info(f"Numeric columns: {len(numeric_columns)}")
    
    return df_model, categorical_mappings, numeric_columns


def build_primary_model(df: pd.DataFrame, config: ProjectConfig, logger) -> tuple:
    """
    Build primary hierarchical model on broader core dataset.
    
    Core model: log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)
    
    pH and SA are intentionally excluded from the core model when complete-case
    filtering would cause severe sample loss; they are handled in the mechanistic subset model.
    
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
    tuple
        (model, trace, model_spec)
    """
    logger.info("\n" + "="*60)
    logger.info("BUILDING PRIMARY MODEL")
    logger.info("="*60)
    
    # Identify required core predictors with raw columns preferred for explicit reference levels.
    required_predictors = {
        'metal': [col for col in df.columns if col == 'Metal_raw'] or 
                 [col for col in df.columns if col == 'Metal_raw_encoded'],
        'polymer': [col for col in df.columns if col == 'ReT'] or
                   [col for col in df.columns if col == 'ReT_encoded'],
        'aging': [col for col in df.columns if col == 'AgS'] or 
                [col for col in df.columns if col == 'AgS_encoded'],
        'Temp': [col for col in df.columns if col == 'Temp'],
        'Study': [col for col in df.columns if col == 'Study_ID_local']
    }
    
    # Get actual column names
    predictors = {}
    for key, candidates in required_predictors.items():
        if candidates:
            predictors[key] = candidates[0]
            logger.info(f"Using {key}: {predictors[key]}")
        else:
            logger.warning(f"Column not found for {key}")
            predictors[key] = None
    
    # Enforce required core terms so ReT/polymer cannot silently drop out.
    core_keys = ['metal', 'polymer', 'aging', 'Temp', 'Study']
    missing_core_terms = [k for k in core_keys if predictors.get(k) is None]
    if missing_core_terms:
        raise ValueError(
            "Primary core model missing required predictors: "
            + ", ".join(missing_core_terms)
        )

    core_required_cols = ['log_Qm'] + [predictors[k] for k in core_keys]

    # Prepare core modeling data (broader dataset, no mandatory pH/SA complete-case restriction).
    modeling_cols = core_required_cols
    df_clean = _log_model_row_loss(df, modeling_cols, "Primary model", logger)

    # Quantify complete-case loss for main mechanistic model (Hydration_Energy + pH + Aging + Study).
    subset_check_cols = ['log_Qm', 'Hydration_Energy', 'pH', predictors['aging'], predictors['Study']]
    has_mech_inputs = all(c in df.columns for c in subset_check_cols)

    n_core = int(len(df[core_required_cols].dropna())) if len(core_required_cols) > 0 else 0
    n_mech_subset = int(len(df[subset_check_cols].dropna())) if has_mech_inputs else 0

    logger.info("Primary core predictors used: " + ", ".join([f"{k}={predictors[k]}" for k in core_keys]))
    logger.info(f"Primary core model sample size (n): {len(df_clean)}")
    logger.info(f"Main mechanistic subset sample size if HE+pH+AgS required (n): {n_mech_subset}")
    logger.info(f"Core vs mechanistic subset availability: {n_core} vs {n_mech_subset}")

    if n_core > 0 and n_mech_subset / n_core < 0.5:
        logger.warning(
            "Issue detected: forcing mechanistic terms into the primary model causes severe complete-case loss "
            f"({n_core} -> {n_mech_subset})."
        )
        logger.info(
            "Fix applied: primary model remains core-only; mechanistic terms are handled in mechanistic model."
        )
    
    if len(df_clean) < 10:
        logger.error("Insufficient data for modeling!")
        raise ValueError("Not enough complete cases for modeling")
    
    # Standardize continuous predictors for core model.
    continuous_vars = [predictors['Temp']]
    continuous_vars = [v for v in continuous_vars if v is not None]
    
    df_scaled = df_clean.copy()
    scaling_params = {}
    
    for var in continuous_vars:
        mean_val = df_clean[var].mean()
        std_val = df_clean[var].std()
        if std_val > 0:
            df_scaled[var] = (df_clean[var] - mean_val) / std_val
        else:
            df_scaled[var] = 0
        scaling_params[var] = {'mean': mean_val, 'std': std_val}
        logger.info(f"Scaled {var}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Explicitly build reference coding metadata (no full K-coefficient parameterization).
    metal_codes, metal_levels, metal_reference = _encode_categorical_with_reference(df_scaled, predictors['metal'], logger)
    polymer_codes, polymer_levels, polymer_reference = _encode_categorical_with_reference(df_scaled, predictors['polymer'], logger)
    aging_codes, aging_levels, aging_reference = _encode_categorical_with_reference(df_scaled, predictors['aging'], logger)
    
    # Get Bayesian configuration
    n_chains = config.get('modeling.bayesian.chains', 4)
    n_draws = config.get('modeling.bayesian.draws', 2000)
    n_tune = config.get('modeling.bayesian.tune', 1000)
    target_accept = config.get('modeling.bayesian.target_accept', 0.95)
    
    logger.info(f"\nMCMC Configuration:")
    logger.info(f"  Chains: {n_chains}")
    logger.info(f"  Draws: {n_draws}")
    logger.info(f"  Tune: {n_tune}")
    logger.info(f"  Target accept: {target_accept}")
    
    # Build PyMC model
    logger.info("\nBuilding PyMC model...")
    
    with pm.Model() as model:
        # Data
        y_obs = df_scaled['log_Qm'].values
        
        # Priors from config
        intercept_mu = config.get('modeling.bayesian.priors.intercept.mu', 0)
        intercept_sigma_cfg = config.get('modeling.bayesian.priors.intercept.sigma', 5)
        beta_mu = config.get('modeling.bayesian.priors.fixed_effects.mu', 0)
        beta_sigma_cfg = config.get('modeling.bayesian.priors.fixed_effects.sigma', 2)
        intercept_sigma = min(float(intercept_sigma_cfg), 5.0)
        beta_sigma = min(float(beta_sigma_cfg), 2.0)

        logger.info(
            f"Primary priors used: intercept ~ Normal({intercept_mu}, {intercept_sigma}), "
            f"effects ~ Normal({beta_mu}, {beta_sigma})"
        )
        
        # Intercept
        intercept = pm.Normal('intercept', mu=intercept_mu, sigma=intercept_sigma)
        
        # Fixed effects for categorical variables with explicit reference coding.
        metal_effect = _build_reference_effect('beta_metal', metal_codes, metal_levels, beta_mu, beta_sigma)
        polymer_effect = _build_reference_effect('beta_polymer', polymer_codes, polymer_levels, beta_mu, beta_sigma)
        aging_effect = _build_reference_effect('beta_aging', aging_codes, aging_levels, beta_mu, beta_sigma)
        
        # Fixed effects for continuous variables
        beta_Temp = pm.Normal('beta_Temp', mu=beta_mu, sigma=beta_sigma) if predictors['Temp'] else 0

        # Study-level random effects
        if predictors['Study'] is not None:
            n_studies = int(df_scaled[predictors['Study']].max()) + 1  # 0-based max
            study_idx = df_scaled[predictors['Study']].astype(int).values  # Already 0-based
            
            sigma_study = pm.HalfNormal('sigma_study', 
                                       sigma=config.get('modeling.bayesian.priors.random_effects_sd.sigma', 2))
            study_effect = pm.Normal('study_effect', mu=0, sigma=sigma_study, shape=n_studies)
            study_contribution = study_effect[study_idx]
        else:
            study_contribution = 0
        
        # Linear predictor
        mu = intercept + metal_effect + polymer_effect + aging_effect
        
        if predictors['Temp']:
            mu += beta_Temp * df_scaled[predictors['Temp']].values
        
        mu += study_contribution
        
        # Save fitted values as a Deterministic (for diagnostics)
        fitted_values = pm.Deterministic("mu", mu)
        
        # Likelihood
        sigma_residual = pm.HalfNormal('sigma_residual', 
                                      sigma=config.get('modeling.bayesian.priors.residual_sd.sigma', 1))
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_residual, observed=y_obs)
    
    logger.info(f"Model built with {len(model.free_RVs)} free random variables")
    
    # Sample from posterior
    logger.info("\nSampling from posterior...")
    try:
        with model:
            trace = pm.sample(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=config.get('random_seed', 42)
            )
    except Exception as e:
        logger.error(f"Failed to sample from model: {str(e)}")
        raise
    
    logger.info("Sampling complete!")

    primary_sampler_diag = _extract_sampler_diagnostics(trace)
    logger.info(
        "Primary sampler depth diagnostics: "
        f"max_observed_tree_depth={primary_sampler_diag['max_observed_tree_depth']}, "
        f"max_treedepth_hits={primary_sampler_diag['max_treedepth_hits']}"
    )
    if primary_sampler_diag['has_max_treedepth_stat']:
        if primary_sampler_diag['max_treedepth_hits'] > 0:
            logger.warning(
                "max_treedepth warnings persist in the primary model. "
                "Consider increasing tune/target_accept/max_treedepth only after this audit checkpoint."
            )
        else:
            logger.info("No max_treedepth warnings in the primary model.")
    
    # Generate posterior predictive samples
    logger.info("Generating posterior predictive samples...")
    with model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=config.get('random_seed', 42))
        pm.compute_log_likelihood(trace)
    logger.info("Posterior predictive sampling complete!")
    
    # Model specification for saving
    model_spec = {
        'name': 'primary_model',
        'formula': 'log(Qm) ~ Metal + ReT + AgS + Temp + (1|Study)',
        'predictors': predictors,
        'predictors_used': {k: predictors[k] for k in core_keys},
        'reference_levels': {
            'metal': metal_reference,
            'polymer': polymer_reference,
            'aging': aging_reference
        },
        'scaling_params': scaling_params,
        'n_observations': len(df_clean),
        'core_model_sample_size': len(df_clean),
        'mechanistic_subset_sample_size': n_mech_subset,
        'sampler_depth_diagnostics': primary_sampler_diag,
        'data_partition': {
            'core_dataset_n': n_core,
            'mechanistic_subset_n': n_mech_subset,
            'strategy': 'core_primary_plus_mechanistic_subset'
        },
        'mcmc_config': {
            'chains': n_chains,
            'draws': n_draws,
            'tune': n_tune,
            'target_accept': target_accept
        }
    }
    
    return model, trace, model_spec


def build_main_mechanistic_model(df: pd.DataFrame, config: ProjectConfig, logger) -> tuple:
    """
    Build main mechanistic subset model.
    
    Main mechanistic model:
    log(Qm) ~ Hydration_Energy + pH + AgS + (1|Study)
    
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
    tuple
        (model, trace, model_spec)
    """
    logger.info("\n" + "="*60)
    logger.info("BUILDING MAIN MECHANISTIC MODEL")
    logger.info("="*60)
    
    # Required outcome + required mechanistic covariates.
    required_cols = ['log_Qm']
    predictor_candidates = {
        'Hydration_Energy': [col for col in df.columns if col == 'Hydration_Energy'],
        'pH': [col for col in df.columns if col == 'pH'],
        'Aging': [col for col in df.columns if col == 'AgS'],
        'Study': [col for col in df.columns if col == 'Study_ID_local']
    }
    
    predictors = {}
    for key, candidates in predictor_candidates.items():
        if candidates:
            predictors[key] = candidates[0]
            logger.info(f"Found {key}: {predictors[key]}")
        else:
            predictors[key] = None

    missing_mechanistic_terms = [k for k in ['Hydration_Energy', 'pH', 'Aging', 'Study'] if predictors.get(k) is None]
    if missing_mechanistic_terms:
        raise ValueError(
            "Mechanistic subset model missing required predictors: "
            + ", ".join(missing_mechanistic_terms)
        )

    logger.info("Descriptor terms used in main mechanistic model: ['Hydration_Energy']")
    logger.info(
        "Mechanistic predictors used: "
        + ", ".join([
            f"Hydration_Energy={predictors['Hydration_Energy']}",
            f"pH={predictors['pH']}",
            f"Aging={predictors['Aging']}",
            f"Study={predictors['Study']}"
        ])
    )

    # Prepare mechanistic subset data (explicitly requires HE + pH, no SA in main model).
    modeling_cols = required_cols + [predictors['Hydration_Energy'], predictors['pH'], predictors['Aging'], predictors['Study']]
    df_clean = _log_model_row_loss(df, modeling_cols, "Main mechanistic model", logger)
    
    logger.info(f"Mechanistic subset model sample size (n): {len(df_clean)}")
    
    if len(df_clean) < 10:
        logger.warning("Insufficient data for alternative model. Skipping.")
        return None, None, None
    
    # Standardize continuous predictors
    continuous_vars = [predictors['Hydration_Energy'], predictors['pH']]
    
    df_scaled = df_clean.copy()
    scaling_params = {}
    
    for var in continuous_vars:
        mean_val = df_clean[var].mean()
        std_val = df_clean[var].std()
        if std_val > 0:
            df_scaled[var] = (df_clean[var] - mean_val) / std_val
        else:
            df_scaled[var] = 0
        scaling_params[var] = {'mean': mean_val, 'std': std_val}
    
    # Get MCMC config
    n_chains = config.get('modeling.bayesian.chains', 4)
    n_draws = config.get('modeling.bayesian.draws', 2000)
    n_tune = config.get('modeling.bayesian.tune', 1000)
    target_accept = config.get('modeling.bayesian.target_accept', 0.95)
    
    # Build model
    logger.info("\nBuilding PyMC model...")
    
    with pm.Model() as model:
        # Data
        y_obs = df_scaled['log_Qm'].values
        
        # Priors
        beta_mu = config.get('modeling.bayesian.priors.fixed_effects.mu', 0)
        beta_sigma_cfg = config.get('modeling.bayesian.priors.fixed_effects.sigma', 2)
        beta_sigma = min(float(beta_sigma_cfg), 2.0)
        intercept_mu = config.get('modeling.bayesian.priors.intercept.mu', 0)
        intercept_sigma_cfg = config.get('modeling.bayesian.priors.intercept.sigma', 5)
        intercept_sigma = min(float(intercept_sigma_cfg), 5.0)

        logger.info(
            f"Main mechanistic priors used: intercept ~ Normal({intercept_mu}, {intercept_sigma}), "
            f"effects ~ Normal({beta_mu}, {beta_sigma})"
        )
        
        # Intercept
        intercept = pm.Normal('intercept', mu=intercept_mu, sigma=intercept_sigma)
        
        # Main mechanistic fixed effects
        mu = intercept

        beta_hydration_energy = pm.Normal('beta_hydration_energy', mu=beta_mu, sigma=beta_sigma)
        mu += beta_hydration_energy * df_scaled[predictors['Hydration_Energy']].values

        beta_pH = pm.Normal('beta_pH', mu=beta_mu, sigma=beta_sigma)
        mu += beta_pH * df_scaled[predictors['pH']].values

        aging_col = predictors['Aging']
        aging_vals = df_scaled[aging_col].values

        aging_codes, aging_levels, aging_reference = _encode_categorical_with_reference(df_scaled, aging_col, logger)
        beta_aging_effect = _build_reference_effect('beta_aging', aging_codes, aging_levels, beta_mu, beta_sigma)
        mu += beta_aging_effect
        
        # Random effects
        n_studies = int(df_scaled[predictors['Study']].max()) + 1  # 0-based max
        study_idx = df_scaled[predictors['Study']].astype(int).values  # Already 0-based

        sigma_study = pm.HalfNormal('sigma_study', sigma=2)
        study_effect = pm.Normal('study_effect', mu=0, sigma=sigma_study, shape=n_studies)
        mu += study_effect[study_idx]
        
        # Save fitted values as a Deterministic (for diagnostics)
        fitted_values = pm.Deterministic("mu", mu)
        
        # Likelihood
        sigma_residual = pm.HalfNormal('sigma_residual', sigma=1)
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_residual, observed=y_obs)
    
    # Sample
    logger.info("\nSampling from posterior...")
    with model:
        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=config.get('random_seed', 42)
        )
    
    logger.info("Sampling complete!")

    mech_sampler_diag = _extract_sampler_diagnostics(trace)
    logger.info(
        "Main mechanistic sampler depth diagnostics: "
        f"max_observed_tree_depth={mech_sampler_diag['max_observed_tree_depth']}, "
        f"max_treedepth_hits={mech_sampler_diag['max_treedepth_hits']}"
    )
    if mech_sampler_diag['has_max_treedepth_stat']:
        if mech_sampler_diag['max_treedepth_hits'] > 0:
            logger.warning(
                "max_treedepth warnings persist in the main mechanistic model. "
                "Consider increasing tune/target_accept/max_treedepth only after this audit checkpoint."
            )
        else:
            logger.info("No max_treedepth warnings in the main mechanistic model.")
    
    # Generate posterior predictive samples
    logger.info("Generating posterior predictive samples...")
    with model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=config.get('random_seed', 42))
        pm.compute_log_likelihood(trace)
    logger.info("Posterior predictive sampling complete!")
    
    model_spec = {
        'name': 'main_mechanistic_model',
        'formula': 'log(Qm) ~ Hydration_Energy + pH + AgS + (1|Study)',
        'predictors': predictors,
        'predictors_used': {
            'Hydration_Energy': predictors['Hydration_Energy'],
            'pH': predictors['pH'],
            'Aging': predictors['Aging'],
            'Study': predictors['Study']
        },
        'reference_levels': {
            'aging': aging_reference
        },
        'scaling_params': scaling_params,
        'n_observations': len(df_clean),
        'mechanistic_subset_sample_size': len(df_clean),
        'sampler_depth_diagnostics': mech_sampler_diag,
        'mcmc_config': {
            'chains': n_chains,
            'draws': n_draws,
            'tune': n_tune,
            'target_accept': target_accept
        }
    }
    
    return model, trace, model_spec


def build_sa_sensitivity_model(df: pd.DataFrame, config: ProjectConfig, logger) -> tuple:
    """Optional supplementary sensitivity model: add SA to main mechanistic terms."""
    logger.info("\n" + "="*60)
    logger.info("BUILDING SUPPLEMENTARY SA SENSITIVITY MODEL")
    logger.info("="*60)

    needed = ['log_Qm', 'Hydration_Energy', 'pH', 'SA', 'AgS', 'Study_ID_local']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(f"Skipping SA sensitivity model; missing columns: {missing}")
        return None, None, None

    df_clean = _log_model_row_loss(df, needed, "Supplementary SA sensitivity model", logger)
    if len(df_clean) < 10:
        logger.warning("Insufficient data for SA sensitivity model. Skipping.")
        return None, None, None

    df_scaled = df_clean.copy()
    scaling_params = {}
    for var in ['Hydration_Energy', 'pH', 'SA']:
        mean_val = df_clean[var].mean()
        std_val = df_clean[var].std()
        df_scaled[var] = (df_clean[var] - mean_val) / std_val if std_val > 0 else 0
        scaling_params[var] = {'mean': mean_val, 'std': std_val}

    n_chains = config.get('modeling.bayesian.chains', 4)
    n_draws = config.get('modeling.bayesian.draws', 2000)
    n_tune = config.get('modeling.bayesian.tune', 1000)
    target_accept = config.get('modeling.bayesian.target_accept', 0.95)

    with pm.Model() as model:
        y_obs = df_scaled['log_Qm'].values
        beta_mu = config.get('modeling.bayesian.priors.fixed_effects.mu', 0)
        beta_sigma = min(float(config.get('modeling.bayesian.priors.fixed_effects.sigma', 2)), 2.0)
        intercept_mu = config.get('modeling.bayesian.priors.intercept.mu', 0)
        intercept_sigma = min(float(config.get('modeling.bayesian.priors.intercept.sigma', 5)), 5.0)

        intercept = pm.Normal('intercept', mu=intercept_mu, sigma=intercept_sigma)
        mu = intercept
        mu += pm.Normal('beta_hydration_energy', mu=beta_mu, sigma=beta_sigma) * df_scaled['Hydration_Energy'].values
        mu += pm.Normal('beta_pH', mu=beta_mu, sigma=beta_sigma) * df_scaled['pH'].values
        mu += pm.Normal('beta_SA', mu=beta_mu, sigma=beta_sigma) * df_scaled['SA'].values

        aging_codes, aging_levels, aging_reference = _encode_categorical_with_reference(df_scaled, 'AgS', logger)
        mu += _build_reference_effect('beta_aging', aging_codes, aging_levels, beta_mu, beta_sigma)

        n_studies = int(df_scaled['Study_ID_local'].max()) + 1
        study_idx = df_scaled['Study_ID_local'].astype(int).values
        sigma_study = pm.HalfNormal('sigma_study', sigma=2)
        study_effect = pm.Normal('study_effect', mu=0, sigma=sigma_study, shape=n_studies)
        mu += study_effect[study_idx]

        pm.Deterministic('mu', mu)
        sigma_residual = pm.HalfNormal('sigma_residual', sigma=1)
        pm.Normal('likelihood', mu=mu, sigma=sigma_residual, observed=y_obs)

    with model:
        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=config.get('random_seed', 42)
        )

    sa_sampler_diag = _extract_sampler_diagnostics(trace)
    logger.info(
        "SA sensitivity sampler depth diagnostics: "
        f"max_observed_tree_depth={sa_sampler_diag['max_observed_tree_depth']}, "
        f"max_treedepth_hits={sa_sampler_diag['max_treedepth_hits']}"
    )

    with model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=config.get('random_seed', 42))
        pm.compute_log_likelihood(trace)

    model_spec = {
        'name': 'supplementary_sa_sensitivity_model',
        'formula': 'log(Qm) ~ Hydration_Energy + pH + SA + AgS + (1|Study)',
        'reference_levels': {'aging': aging_reference},
        'scaling_params': scaling_params,
        'n_observations': len(df_clean),
        'sampler_depth_diagnostics': sa_sampler_diag,
        'mcmc_config': {
            'chains': n_chains,
            'draws': n_draws,
            'tune': n_tune,
            'target_accept': target_accept
        }
    }
    return model, trace, model_spec


def summarize_results(trace, model_spec: dict, logger) -> dict:
    """
    Summarize model results.
    
    Parameters
    ----------
    trace : arviz.InferenceData
        MCMC trace
    model_spec : dict
        Model specification
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Model summary results
    """
    logger.info("\\nSummarizing model results...")
    
    # Posterior summary
    summary = az.summary(trace, hdi_prob=0.95)
    logger.info(f"\nPosterior Summary:\n{summary}")
    
    # Convergence diagnostics
    rhat = az.rhat(trace)
    ess_bulk = az.ess(trace, method='bulk')
    ess_tail = az.ess(trace, method='tail')
    sampler_diag = _extract_sampler_diagnostics(trace)
    divergences = sampler_diag['divergences']
    
    logger.info(f"\nConvergence Diagnostics:")
    logger.info(f"  R-hat range: [{rhat.to_array().min().values:.4f}, {rhat.to_array().max().values:.4f}]")
    logger.info(f"  ESS (bulk) min: {ess_bulk.to_array().min().values:.0f}")
    logger.info(f"  ESS (tail) min: {ess_tail.to_array().min().values:.0f}")
    logger.info(f"  Divergences: {divergences}")
    logger.info(f"  Max observed tree depth: {sampler_diag['max_observed_tree_depth']}")
    logger.info(f"  Max treedepth hits: {sampler_diag['max_treedepth_hits']}")
    if sampler_diag['has_max_treedepth_stat']:
        if sampler_diag['max_treedepth_hits'] > 0:
            logger.warning("max_treedepth warnings persist.")
        else:
            logger.info("No max_treedepth warnings detected.")
    
    results = {
        'posterior_summary': summary.to_dict(),
        'convergence': {
            'rhat_min': float(rhat.to_array().min().values),
            'rhat_max': float(rhat.to_array().max().values),
            'ess_bulk_min': float(ess_bulk.to_array().min().values),
            'ess_tail_min': float(ess_tail.to_array().min().values),
            'divergences': divergences,
            'max_observed_tree_depth': sampler_diag['max_observed_tree_depth'],
            'max_treedepth_hits': sampler_diag['max_treedepth_hits']
        }
    }
    
    return results


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '05_model_fitting')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 05: HIERARCHICAL BAYESIAN MODELING", logger=logger)
        
        # Load data
        logger.info("Loading enriched data...")
        data_path = config.get_path('processed_data') / "03_enriched_data.csv"
        df = load_dataframe(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Prepare data
        df_model, categorical_mappings, numeric_columns = prepare_modeling_data(df, logger)
        
        # Build and fit primary model
        try:
            primary_model, primary_trace, primary_spec = build_primary_model(df_model, config, logger)
            primary_results = summarize_results(primary_trace, primary_spec, logger)
            
            # Save primary model
            models_dir = config.get_path('models')
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to save as NetCDF, with fallback to JSON
            try:
                primary_trace.to_netcdf(str(models_dir / "primary_model_trace.nc"))
                logger.info(f"Saved primary model trace as NetCDF")
            except Exception as e:
                logger.warning(f"Failed to save as NetCDF: {e}. Trying JSON format...")
                try:
                    primary_trace.to_json(str(models_dir / "primary_model_trace.json"))
                    logger.info(f"Saved primary model trace as JSON")
                except Exception as e2:
                    logger.error(f"Failed to save trace in both formats: {e2}")
            
            save_json({**primary_spec, **primary_results}, 
                     models_dir / "primary_model_summary.json")
            logger.info(f"\nSaved primary model to {models_dir}")
            
        except Exception as e:
            logger.error(f"Error fitting primary model: {e}", exc_info=True)
            primary_results = None
        
        # Build and fit main mechanistic model
        try:
            mech_model, mech_trace, mech_spec = build_main_mechanistic_model(df_model, config, logger)
            
            if mech_trace is not None:
                mech_results = summarize_results(mech_trace, mech_spec, logger)
                
                # Save main mechanistic model
                try:
                    mech_trace.to_netcdf(str(models_dir / "main_mechanistic_model_trace.nc"))
                    logger.info(f"Saved main mechanistic model trace as NetCDF")
                except Exception as e:
                    logger.warning(f"Failed to save as NetCDF: {e}. Trying JSON format...")
                    try:
                        mech_trace.to_json(str(models_dir / "main_mechanistic_model_trace.json"))
                        logger.info(f"Saved main mechanistic model trace as JSON")
                    except Exception as e2:
                        logger.error(f"Failed to save trace in both formats: {e2}")
                
                save_json({**mech_spec, **mech_results}, 
                         models_dir / "main_mechanistic_model_summary.json")
                logger.info(f"\nSaved main mechanistic model to {models_dir}")
            else:
                mech_results = None
                
        except Exception as e:
            logger.error(f"Error fitting main mechanistic model: {e}", exc_info=True)
            mech_results = None

        # Optional supplementary SA sensitivity model (not the main mechanistic model)
        run_sa_sensitivity = bool(config.get('modeling.supplementary_sa_sensitivity.enabled', True))
        if run_sa_sensitivity:
            try:
                sa_model, sa_trace, sa_spec = build_sa_sensitivity_model(df_model, config, logger)
                if sa_trace is not None:
                    sa_results = summarize_results(sa_trace, sa_spec, logger)
                    try:
                        sa_trace.to_netcdf(str(models_dir / "supplementary_sa_sensitivity_model_trace.nc"))
                        logger.info("Saved SA sensitivity model trace as NetCDF")
                    except Exception as e:
                        logger.warning(f"Failed to save SA sensitivity trace as NetCDF: {e}. Trying JSON format...")
                        try:
                            sa_trace.to_json(str(models_dir / "supplementary_sa_sensitivity_model_trace.json"))
                            logger.info("Saved SA sensitivity model trace as JSON")
                        except Exception as e2:
                            logger.error(f"Failed to save SA sensitivity trace in both formats: {e2}")

                    save_json({**sa_spec, **sa_results}, models_dir / "supplementary_sa_sensitivity_model_summary.json")
                    logger.info(f"\nSaved supplementary SA sensitivity model to {models_dir}")
            except Exception as e:
                logger.error(f"Error fitting supplementary SA sensitivity model: {e}", exc_info=True)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL FITTING COMPLETED")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in model fitting: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
