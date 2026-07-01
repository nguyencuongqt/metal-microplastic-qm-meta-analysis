"""
Script 04: Exploratory Data Analysis
====================================

Comprehensive EDA including:
- Univariate distributions
- Bivariate relationships
- Study-level variation
- Missing data patterns
- Preliminary correlation analysis

Author: Manuscript authors
Date: March 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas import CategoricalDtype

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, save_json, print_section_header)


def set_plot_style(config: ProjectConfig):
    """Set matplotlib style based on config."""
    style = config.get('figures.global.style', 'seaborn-v0_8-paper')
    font_family = config.get('figures.global.font_family', 'Arial')
    font_size = config.get('figures.global.font_size', 10)
    
    plt.style.use(style)
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['figure.dpi'] = 100  # Screen DPI


def analyze_response_distribution(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Analyze distribution of response variable (log_Qm).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Distribution statistics
    """
    logger.info("Analyzing response variable distribution...")
    
    if 'log_Qm' not in df.columns:
        logger.error("log_Qm column not found!")
        raise ValueError("log_Qm column missing")
    
    data = df['log_Qm'].dropna()
    
    # Descriptive statistics
    stats_dict = {
        'n': len(data),
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
    }
    
    # Normality tests
    shapiro_stat, shapiro_pval = stats.shapiro(data)
    ks_stat, ks_pval = stats.kstest(data, 'norm', 
                                    args=(data.mean(), data.std()))
    
    stats_dict['normality_tests'] = {
        'shapiro_wilk': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_pval)},
        'kolmogorov_smirnov': {'statistic': float(ks_stat), 'p_value': float(ks_pval)}
    }
    
    logger.info(f"\nlog_Qm Distribution Statistics:")
    logger.info(f"  N:        {stats_dict['n']}")
    logger.info(f"  Mean:     {stats_dict['mean']:.3f}")
    logger.info(f"  Median:   {stats_dict['median']:.3f}")
    logger.info(f"  Std:      {stats_dict['std']:.3f}")
    logger.info(f"  Range:    [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
    logger.info(f"  Skewness: {stats_dict['skewness']:.3f}")
    logger.info(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")
    logger.info(f"\nNormality Tests:")
    logger.info(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_pval:.4f}")
    logger.info(f"  K-S Test:     D={ks_stat:.4f}, p={ks_pval:.4f}")
    
    # Create distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('log(Qm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of log(Qm)')
    axes[0].axvline(data.mean(), color='red', linestyle='--', label='Mean')
    axes[0].axvline(data.median(), color='blue', linestyle='--', label='Median')
    axes[0].legend()
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    
    # Box plot
    axes[2].boxplot(data)
    axes[2].set_ylabel('log(Qm)')
    axes[2].set_title('Box Plot')
    
    plt.tight_layout()
    fig_path = config.get_path('figures') / 'EDA_01_response_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure: {fig_path}")
    
    return stats_dict


def analyze_categorical_factors(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Analyze categorical factors (Metal, Polymer, Aging).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Categorical analysis results
    """
    logger.info("\nAnalyzing categorical factors...")
    
    results = {
        'tests_used_per_factor': {},
        'min_group_size': {},
        'primary_metal_factor': None,
        'supporting_descriptive': {
            'metal_representations': {},
            'technical_provenance_factors': {}
        }
    }

    excluded_prefixes = ('metal_descriptor_missing_',)
    excluded_exact = {'species_requires_speciation', 'metal_descriptor_missing_any'}
    technical_exact = {'Study_ID', 'quality_flags', 'quality_category'}

    def is_technical_flag(col_name: str) -> bool:
        return (
            col_name.startswith(excluded_prefixes)
            or col_name in excluded_exact
            or col_name in technical_exact
        )

    def numeric_like_object(col_name: str, threshold: float = 0.90) -> bool:
        series = df[col_name]
        if is_numeric_dtype(series):
            return True
        if not (is_string_dtype(series) or isinstance(series.dtype, CategoricalDtype) or series.dtype == 'object'):
            return False

        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        cleaned = non_null.astype(str).str.replace('\xa0', ' ', regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors='coerce')
        success_rate = converted.notna().mean()
        return success_rate >= threshold

    # Priority 1: Config whitelist if provided
    cfg_vars = config.get('eda.categorical_vars', [])
    categorical_vars = []
    if isinstance(cfg_vars, list) and len(cfg_vars) > 0:
        categorical_vars = [col for col in cfg_vars if col in df.columns and not is_technical_flag(col)]
        logger.info("Using config whitelist for categorical vars: {}".format(categorical_vars))

    # Priority 2: fallback dtype-based detection + keyword prioritization
    if len(categorical_vars) == 0:
        keyword_candidate_cols = []
        for keyword in ['polymer', 'plastic', 'aging', 'ageing', 'weathering', 'metal']:
            keyword_candidate_cols.extend([
                col for col in df.columns
                if keyword in col.lower() and not is_technical_flag(col)
            ])

        object_like_cols = [
            col for col in df.columns
            if (is_string_dtype(df[col]) or isinstance(df[col].dtype, CategoricalDtype) or df[col].dtype == 'object')
            and not is_technical_flag(col)
            and not numeric_like_object(col, threshold=0.90)
        ]

        excluded_numeric_like = [
            col for col in df.columns
            if (is_string_dtype(df[col]) or isinstance(df[col].dtype, CategoricalDtype) or df[col].dtype == 'object')
            and not is_technical_flag(col)
            and numeric_like_object(col, threshold=0.90)
        ]
        if excluded_numeric_like:
            logger.info("Excluded numeric-like object columns from categorical analysis: {}".format(excluded_numeric_like))

        priority_cols = []
        for keyword in ['polymer', 'plastic', 'aging', 'ageing', 'weathering', 'metal']:
            priority_cols.extend([col for col in keyword_candidate_cols if keyword in col.lower()])

        ordered = []
        for col in priority_cols + object_like_cols:
            if col not in ordered:
                ordered.append(col)

        categorical_vars = ordered
        logger.info("Using dtype-based categorical vars (prioritized Polymer/Aging): {}".format(categorical_vars))

    # Keep one primary metal representation for inferential tests.
    metal_priority = ['Metal', 'Metal_Species', 'Metal_raw']
    metal_candidates = [c for c in metal_priority if c in categorical_vars]
    primary_metal = metal_candidates[0] if len(metal_candidates) > 0 else None
    supporting_metal = [c for c in metal_candidates if c != primary_metal]
    results['primary_metal_factor'] = primary_metal

    if len(metal_candidates) > 1:
        logger.warning(
            "Issue detected: multiple metal representations found in main categorical inference: {}".format(
                metal_candidates
            )
        )
        logger.info(
            "Fix applied: kept '{}' as primary metal factor; moved {} to supporting descriptive output.".format(
                primary_metal, supporting_metal
            )
        )

    # Move technical/provenance factors out of inferential categorical tests.
    technical_vars = [c for c in technical_exact if c in df.columns]
    if technical_vars:
        logger.warning(
            "Issue detected: technical/provenance variables detected and could be misinterpreted as scientific factors: {}".format(
                technical_vars
            )
        )
        logger.info(
            "Fix applied: reserved technical/provenance variables for descriptive-only summaries (excluded from inferential tests)."
        )

    main_categorical_vars = [
        c for c in categorical_vars
        if c not in supporting_metal and c not in technical_vars
    ]

    # Build descriptive-only summaries for supporting metal representations and technical factors.
    supporting_only_vars = supporting_metal + technical_vars
    for var in supporting_only_vars:
        if var not in df.columns:
            continue
        value_counts = df[var].value_counts(dropna=False)
        total = len(df[var])
        safe_total = total if total > 0 else 1
        summary = {
            'n_categories': int(df[var].nunique(dropna=False)),
            'counts': value_counts.to_dict(),
            'proportions': (value_counts / safe_total).to_dict(),
            'summary_type': 'descriptive_only'
        }

        if var in supporting_metal:
            results['supporting_descriptive']['metal_representations'][var] = summary
        else:
            results['supporting_descriptive']['technical_provenance_factors'][var] = summary

        logger.info("\n{} (descriptive/supporting only):".format(var))
        logger.info("  Categories: {}".format(summary['n_categories']))
    
    # Analyze each categorical variable
    for var in main_categorical_vars:
        if var not in df.columns:
            continue
        
        # Value counts
        value_counts = df[var].value_counts()
        total = len(df[var].dropna())
        
        results[var] = {
            'n_categories': len(value_counts),
            'counts': value_counts.to_dict(),
            'proportions': (value_counts / total).to_dict()
        }
        
        logger.info(f"\n{var}:")
        logger.info(f"  Categories: {len(value_counts)}")
        for category, count in value_counts.items():
            logger.info(f"    {category}: {count} ({count/total*100:.1f}%)")
        
        # If log_Qm available, analyze by category
        if 'log_Qm' in df.columns:
            category_means = df.groupby(var)['log_Qm'].agg(['mean', 'std', 'count'])
            results[var]['log_Qm_by_category'] = category_means.to_dict()

            groups = [df[df[var] == cat]['log_Qm'].dropna() for cat in value_counts.index]
            group_sizes = [len(g) for g in groups]
            min_group = min(group_sizes) if len(group_sizes) > 0 else 0
            n_total = int(sum(group_sizes))
            n_groups = int(len(groups))

            results[var]['min_group_size'] = int(min_group)
            results['min_group_size'][var] = int(min_group)

            if n_groups < 2 or n_total < 10:
                results['tests_used_per_factor'][var] = 'descriptive_only_sparse'
                results[var]['descriptive_only'] = {
                    'reason': 'too_sparse_for_inference',
                    'n_groups': n_groups,
                    'n_total': n_total,
                    'min_group_size': int(min_group)
                }
                logger.info("  Test: descriptive only (too sparse; n_groups={}, n_total={}, min_group={})".format(
                    n_groups, n_total, min_group))
            elif min_group < 5:
                h_stat, p_val = stats.kruskal(*groups)
                epsilon_sq = (h_stat - n_groups + 1) / (n_total - n_groups) if (n_total - n_groups) > 0 else np.nan
                epsilon_sq = max(0.0, float(epsilon_sq)) if not np.isnan(epsilon_sq) else np.nan

                results['tests_used_per_factor'][var] = 'kruskal_wallis'
                results[var]['kruskal_wallis'] = {
                    'h_statistic': float(h_stat),
                    'p_value': float(p_val),
                    'epsilon_squared': float(epsilon_sq) if not np.isnan(epsilon_sq) else None,
                    'min_group_size': int(min_group)
                }
                logger.info("  Kruskal-Wallis: H={:.2f}, p={:.4f}, epsilon^2={:.3f}".format(
                    h_stat, p_val, epsilon_sq if not np.isnan(epsilon_sq) else 0.0))
            else:
                f_stat, p_val = stats.f_oneway(*groups)
                eta_sq = (f_stat * (n_groups - 1)) / ((f_stat * (n_groups - 1)) + (n_total - n_groups)) if n_total > n_groups else np.nan
                eta_sq = max(0.0, float(eta_sq)) if not np.isnan(eta_sq) else np.nan

                results['tests_used_per_factor'][var] = 'anova'
                results[var]['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_val),
                    'eta_squared': float(eta_sq) if not np.isnan(eta_sq) else None,
                    'min_group_size': int(min_group)
                }
                logger.info("  ANOVA: F={:.2f}, p={:.4f}, eta^2={:.3f}".format(
                    f_stat, p_val, eta_sq if not np.isnan(eta_sq) else 0.0))
    
    # Create categorical plots
    n_vars = len(main_categorical_vars)
    if n_vars > 0:
        fig, axes = plt.subplots(1, min(n_vars, 3), figsize=(5*min(n_vars, 3), 4))
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(main_categorical_vars[:3]):
            if 'log_Qm' in df.columns:
                df.boxplot(column='log_Qm', by=var, ax=axes[i])
                axes[i].set_title(f'log(Qm) by {var}')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('log(Qm)')
        
        plt.tight_layout()
        fig_path = config.get_path('figures') / 'EDA_02_categorical_factors.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"\nSaved figure: {fig_path}")
    
    return results


def analyze_continuous_predictors(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Analyze continuous predictors (pH, SA, Temp, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Continuous variable analysis
    """
    logger.info("\nAnalyzing continuous predictors...")
    
    results = {
        'constant_features_skipped': [],
        'binary_indicator_analysis': {}
    }
    
    def numeric_like_object(series: pd.Series, threshold: float = 0.90) -> bool:
        if is_numeric_dtype(series):
            return True
        if not (is_string_dtype(series) or isinstance(series.dtype, CategoricalDtype) or series.dtype == 'object'):
            return False
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        cleaned = non_null.astype(str).str.replace('\xa0', ' ', regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors='coerce')
        return converted.notna().mean() >= threshold

    def get_numeric_series(series: pd.Series) -> pd.Series:
        if is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        cleaned = series.astype(str).str.replace('\xa0', ' ', regex=False).str.strip()
        return pd.to_numeric(cleaned, errors='coerce')

    # Identify continuous variables (include numeric-like object columns)
    continuous_vars = []
    for keyword in ['ph', 'temp', 'sa', 'surface', 'area', 'time', 'concentration', 'aic', 'rpm']:
        matches = [
            col for col in df.columns
            if keyword in col.lower() and numeric_like_object(df[col], threshold=0.90)
        ]
        continuous_vars.extend(matches)
    
    # Add descriptor columns
    descriptor_cols = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 
                      'Electronegativity', 'Charge_Density']
    continuous_vars.extend([col for col in descriptor_cols if col in df.columns and numeric_like_object(df[col], threshold=0.90)])

    # Explicitly include key numeric predictors if detection misses them.
    explicit_numeric_predictors = [
        col for col in ['AIC', 'rpm']
        if col in df.columns and numeric_like_object(df[col], threshold=0.90)
    ]
    missing_explicit = [col for col in explicit_numeric_predictors if col not in continuous_vars]
    if missing_explicit:
        logger.warning(
            "Issue detected: important numeric predictors were missed by keyword-based continuous detection: {}".format(
                missing_explicit
            )
        )
        continuous_vars.extend(missing_explicit)
        logger.info("Fix applied: explicitly added {} to continuous predictor analysis.".format(missing_explicit))

    included_explicit = [col for col in explicit_numeric_predictors if col in continuous_vars]
    if included_explicit:
        logger.info(
            "Policy check: ensured key numeric predictors are included in continuous analysis: {}".format(
                included_explicit
            )
        )
    
    continuous_vars = list(set(continuous_vars))

    # Move provenance imputation flags out of ordinary continuous predictors.
    imputed_in_continuous = [col for col in continuous_vars if col.lower().startswith('imputed_')]
    if imputed_in_continuous:
        logger.warning(
            "Issue detected: provenance imputation flags were included as ordinary continuous predictors: {}".format(
                imputed_in_continuous
            )
        )
        continuous_vars = [col for col in continuous_vars if col not in imputed_in_continuous]
        logger.info(
            "Fix applied: moved {} to binary-indicator/provenance analysis group.".format(
                imputed_in_continuous
            )
        )

    logger.info(f"Found {len(continuous_vars)} continuous variables")

    # Identify binary missing-indicator columns for separate analysis
    binary_indicator_vars = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue
        non_na = df[col].dropna()
        if len(non_na) == 0:
            continue
        unique_vals = set(non_na.unique().tolist())
        if unique_vals.issubset({0, 1}):
            is_missing_indicator = ('missing' in col.lower())
            is_imputed_indicator = col.lower().startswith('imputed_')
            is_tech_indicator = col.startswith('metal_descriptor_missing_') or col in [
                'species_requires_speciation', 'metal_descriptor_missing_any'
            ]
            if is_missing_indicator or is_imputed_indicator or is_tech_indicator:
                binary_indicator_vars.append(col)

    if binary_indicator_vars:
        logger.info("Binary indicator columns excluded from main correlation matrix: {}".format(binary_indicator_vars))
    
    # Analyze each
    for var in continuous_vars:
        numeric_series = get_numeric_series(df[var])
        data = numeric_series.dropna()
        
        if len(data) == 0:
            continue
        
        results[var] = {
            'n': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'missing': int(numeric_series.isna().sum()),
            'missing_pct': float(numeric_series.isna().sum() / len(df) * 100)
        }
        
        # Correlation with log_Qm (Spearman, skip constant features)
        if 'log_Qm' in df.columns:
            valid_data = pd.DataFrame({var: numeric_series, 'log_Qm': df['log_Qm']}).dropna()
            n_unique = valid_data[var].nunique()
            if var in binary_indicator_vars:
                logger.info("    Skipping Spearman for {}: handled in binary indicator analysis".format(var))
            elif n_unique < 2:
                msg = "{} (nunique<2)".format(var)
                results['constant_features_skipped'].append(msg)
                logger.info("    Skipping correlation for {}: constant feature (nunique<2)".format(var))
            elif len(valid_data) > 1 and valid_data['log_Qm'].nunique() > 1:
                corr, pval = stats.spearmanr(valid_data[var], valid_data['log_Qm'])
                results[var]['spearman_with_log_Qm'] = {
                    'rho': float(corr),
                    'p_value': float(pval),
                    'n': int(len(valid_data))
                }
        
        logger.info(f"\n  {var}:")
        logger.info(f"    N:      {results[var]['n']}")
        logger.info(f"    Mean:   {results[var]['mean']:.2f}")
        logger.info(f"    Std:    {results[var]['std']:.2f}")
        logger.info(f"    Range:  [{results[var]['min']:.2f}, {results[var]['max']:.2f}]")
        logger.info(f"    Missing: {results[var]['missing']} ({results[var]['missing_pct']:.1f}%)")
        if 'spearman_with_log_Qm' in results[var]:
            logger.info("    Spearman with log(Qm): rho={:.3f}, p={:.4f} (n={})".format(
                results[var]['spearman_with_log_Qm']['rho'],
                results[var]['spearman_with_log_Qm']['p_value'],
                results[var]['spearman_with_log_Qm']['n']))
    
    # Create robust Spearman correlation matrix plot
    if 'log_Qm' in df.columns and len(continuous_vars) > 0:
        excluded_set = set(binary_indicator_vars)
        corr_candidates = [v for v in continuous_vars if v in df.columns and v not in excluded_set]

        valid_corr_vars = []
        corr_data = {}
        for var in corr_candidates:
            numeric_series = get_numeric_series(df[var])
            n_unique = numeric_series.dropna().nunique()
            if n_unique < 2:
                msg = "{} (nunique<2)".format(var)
                if msg not in results['constant_features_skipped']:
                    results['constant_features_skipped'].append(msg)
                logger.info("Skipping {} from correlation matrix: constant feature (nunique<2)".format(var))
                continue
            valid_corr_vars.append(var)
            corr_data[var] = numeric_series

        corr_data['log_Qm'] = pd.to_numeric(df['log_Qm'], errors='coerce')
        plot_vars = valid_corr_vars[:10] + ['log_Qm']
        plot_vars = [v for v in plot_vars if v in corr_data]

        if len(plot_vars) >= 2:
            corr_df = pd.DataFrame({v: corr_data[v] for v in plot_vars})
            corr_matrix = corr_df.corr(method='spearman')

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, linewidths=1)
            plt.title('Spearman Correlation Matrix')
            plt.tight_layout()

            fig_path = config.get_path('figures') / 'EDA_03_correlation_matrix.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"\nSaved figure: {fig_path}")
        else:
            logger.warning("Not enough valid continuous variables to draw correlation matrix")

    # Separate analysis for binary indicators vs log_Qm
    if 'log_Qm' in df.columns and len(binary_indicator_vars) > 0:
        logger.info("\nAnalyzing binary indicators vs log(Qm) separately...")

        n_plot = len(binary_indicator_vars)
        fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4))
        if n_plot == 1:
            axes = [axes]

        for idx, var in enumerate(binary_indicator_vars):
            valid = df[[var, 'log_Qm']].dropna()
            groups = {k: g['log_Qm'].values for k, g in valid.groupby(var)}

            n0 = len(groups.get(0, []))
            n1 = len(groups.get(1, []))

            # Plot
            sns.boxplot(data=valid, x=var, y='log_Qm', ax=axes[idx])
            axes[idx].set_title("log(Qm) by {}".format(var))
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel('log(Qm)')

            analysis = {'n0': int(n0), 'n1': int(n1), 'test': 'descriptive_only'}

            if n0 >= 2 and n1 >= 2:
                r_pb, p_pb = stats.pointbiserialr(valid[var], valid['log_Qm'])
                analysis = {
                    'n0': int(n0),
                    'n1': int(n1),
                    'test': 'point_biserial',
                    'r_pb': float(r_pb),
                    'p_value': float(p_pb)
                }
                logger.info("  {}: point-biserial r={:.3f}, p={:.4f} (n0={}, n1={})".format(
                    var, r_pb, p_pb, n0, n1))
            elif n0 >= 1 and n1 >= 1:
                t_stat, p_t = stats.ttest_ind(groups.get(0, []), groups.get(1, []), equal_var=False)
                analysis = {
                    'n0': int(n0),
                    'n1': int(n1),
                    'test': 'welch_ttest',
                    't_statistic': float(t_stat),
                    'p_value': float(p_t)
                }
                logger.info("  {}: Welch t-test t={:.3f}, p={:.4f} (n0={}, n1={})".format(
                    var, t_stat, p_t, n0, n1))
            else:
                logger.info("  {}: descriptive only (insufficient class sizes)".format(var))

            results['binary_indicator_analysis'][var] = analysis

        plt.tight_layout()
        fig_path = config.get_path('figures') / 'EDA_03b_binary_indicator_logQm.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved figure: {}".format(fig_path))

    # Keep unique list for JSON output
    results['constant_features_skipped'] = sorted(list(set(results['constant_features_skipped'])))
    
    return results


def analyze_study_variation(df: pd.DataFrame, config: ProjectConfig, logger) -> dict:
    """
    Analyze between-study variation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Study variation analysis
    """
    logger.info("\nAnalyzing study-level variation...")
    
    results = {}
    
    # Find study ID column
    study_col = None
    for col in df.columns:
        if 'study' in col.lower() and 'id' in col.lower():
            study_col = col
            break
    
    if not study_col or study_col not in df.columns:
        logger.warning("Study ID column not found. Skipping study variation analysis.")
        return results
    
    logger.info(f"Using {study_col} for study identification")
    
    # Study-level summary
    study_summary = df.groupby(study_col).agg({
        'log_Qm': ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    
    results['n_studies'] = int(df[study_col].nunique())
    results['total_observations'] = int(len(df))
    results['obs_per_study_mean'] = float(len(df) / results['n_studies'])
    results['obs_per_study_std'] = float(df.groupby(study_col).size().std())
    
    logger.info(f"  Number of studies: {results['n_studies']}")
    logger.info(f"  Total observations: {results['total_observations']}")
    logger.info(f"  Obs per study (mean +/- std): {results['obs_per_study_mean']:.1f} +/- "
                f"{results['obs_per_study_std']:.1f}")
    
    # Between-study variance
    study_means = df.groupby(study_col)['log_Qm'].mean()
    results['between_study_variance'] = float(study_means.var())
    results['overall_variance'] = float(df['log_Qm'].var())
    results['ICC_estimate'] = float(results['between_study_variance'] / results['overall_variance'])
    
    logger.info(f"  Between-study variance: {results['between_study_variance']:.3f}")
    logger.info(f"  Overall variance: {results['overall_variance']:.3f}")
    logger.info(f"  ICC (estimated): {results['ICC_estimate']:.3f}")
    
    # Plot study means
    plt.figure(figsize=(10, 6))
    study_means_sorted = study_means.sort_values()
    plt.barh(range(len(study_means_sorted)), study_means_sorted.values)
    plt.xlabel('Mean log(Qm)')
    plt.ylabel('Study')
    plt.title('Study-Level Mean log(Qm)')
    plt.axvline(df['log_Qm'].mean(), color='red', linestyle='--', label='Overall Mean')
    plt.legend()
    plt.tight_layout()
    
    fig_path = config.get_path('figures') / 'EDA_04_study_variation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"\nSaved figure: {fig_path}")
    
    return results


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '04_exploratory_analysis')
        set_random_seed(config=config)
        set_plot_style(config)
        
        print_section_header("SCRIPT 04: EXPLORATORY DATA ANALYSIS", logger=logger)
        
        # Load enriched data
        logger.info("Loading enriched data...")
        data_path = config.get_path('processed_data') / "03_enriched_data.csv"
        df = load_dataframe(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Prepare output dictionary
        eda_results = {}
        
        # Step 1: Response distribution
        eda_results['response_distribution'] = analyze_response_distribution(df, config, logger)
        
        # Step 2: Categorical factors
        eda_results['categorical_factors'] = analyze_categorical_factors(df, config, logger)
        
        # Step 3: Continuous predictors
        eda_results['continuous_predictors'] = analyze_continuous_predictors(df, config, logger)

        # Add required top-level fields for downstream reporting
        eda_results['constant_features_skipped'] = eda_results['continuous_predictors'].get(
            'constant_features_skipped', []
        )
        eda_results['tests_used_per_factor'] = eda_results['categorical_factors'].get(
            'tests_used_per_factor', {}
        )
        eda_results['min_group_size'] = eda_results['categorical_factors'].get(
            'min_group_size', {}
        )
        
        # Step 4: Study variation
        eda_results['study_variation'] = analyze_study_variation(df, config, logger)
        
        # Save results
        logger.info("\nSaving EDA results...")
        results_path = config.get_path('results') / "04_eda_results.json"
        save_json(eda_results, results_path)
        logger.info(f"Saved to: {results_path}")
        
        logger.info("\n" + "="*60)
        logger.info("EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in exploratory analysis: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
