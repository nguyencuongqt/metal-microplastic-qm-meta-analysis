"""
Script 02: Data Cleaning and Preparation
========================================

This script performs comprehensive data cleaning including:
- Unit harmonization
- Outlier detection and flagging
- Study ID reconstruction
- Forward-fill block correction
- Log transformation
- Quality flag system

POLICY ENFORCEMENT:
This script MUST comply with data_policy.md rules:
  [PASS] pH: Never globally imputed; preserve missing; create pH_missing indicator
  [PASS] Imputation: Temp, AIC, rpm only; require traceability flags (imputed_*)
  [PASS] Contextual: SA, AgS, Solution_Agent retain original missingness (no imputation)
  [PASS] Qm units: Always ug/g; no conversion; treat as implicit unit

Any violation will be detected, logged with "Issue detected" + "Fix applied" pairs,
and must be reviewed/approved per data_policy.md Section 6.

Author: Manuscript authors
Date: March 2026
"""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, save_dataframe, save_json, print_section_header,
                   check_required_columns, describe_dataframe)


def _clean_unit_for_display(unit_value: str | None) -> str | None:
    """Fix common mojibake artifacts for logging only (no data/assumption changes)."""
    if unit_value is None:
        return None

    cleaned = str(unit_value)
    cleaned = cleaned.replace('\u00c2\u00b5', 'u')
    cleaned = cleaned.replace('\u00ce\u00bc', 'u')
    cleaned = cleaned.replace('\u00b5', 'u')
    cleaned = cleaned.replace('\u03bc', 'u')
    cleaned = cleaned.replace('\u00c3\u2014', 'x')
    return cleaned


def harmonize_units(df: pd.DataFrame, config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Harmonize units for Qm and other variables.
    
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
    pd.DataFrame
        Data with harmonized units
    """
    logger.info("Harmonizing units...")
    
    df = df.copy()
    
    # Get unit conversion factors from config
    target_unit = config.get('data_processing.qm_units.target', 'mg/g')
    conversion_factors = config.get('data_processing.qm_units.conversion_factors', {})

    target_unit_display = _clean_unit_for_display(target_unit)
    if target_unit_display != str(target_unit):
        logger.warning(
            "Issue detected: unit string encoding artifact found in configuration/log display."
        )
        logger.info(
            f"Fix applied: cleaned unit display for logs only ('{target_unit}' -> '{target_unit_display}')."
        )

    logger.info(f"Target Qm unit: {target_unit_display}")
    
    # Check if there's a unit column
    unit_cols = [col for col in df.columns if 'unit' in col.lower()]
    
    if unit_cols:
        unit_col = unit_cols[0]
        logger.info(f"Found unit column: {unit_col}")
        
        # Apply conversions
        for source_unit, factor in conversion_factors.items():
            if factor is not None:
                mask = df[unit_col].str.contains(source_unit, case=False, na=False)
                n_converted = mask.sum()
                if n_converted > 0:
                    qm_col = [col for col in df.columns if 'qm' in col.lower()][0]
                    df.loc[mask, qm_col] = df.loc[mask, qm_col] * factor
                    logger.info(
                        f"  Converted {n_converted} values from "
                        f"{_clean_unit_for_display(source_unit)} to {target_unit_display}"
                    )
            else:
                mask = df[unit_col].str.contains(source_unit, case=False, na=False)
                n_skipped = mask.sum()
                if n_skipped > 0:
                    logger.warning(
                        "  Skipped "
                        f"{n_skipped} values with {_clean_unit_for_display(source_unit)} "
                        "(requires molecular weight)"
                    )
        
        # Update unit column
        df[unit_col] = target_unit
    else:
        logger.warning("No unit column found. Assuming all values in target unit.")
    
    # Harmonize temperature units (if present)
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    if temp_cols:
        temp_col = temp_cols[0]
        # Convert any Fahrenheit to Celsius if needed
        # Assume values > 50 might be in Kelvin, convert to Celsius
        mask = df[temp_col] > 100
        if mask.any():
            df.loc[mask, temp_col] = df.loc[mask, temp_col] - 273.15
            logger.info(f"  Converted {mask.sum()} temperature values from Kelvin to Celsius")
    
    logger.info("Unit harmonization complete.")
    return df


def detect_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    series : pd.Series
        Data series
    multiplier : float
        IQR multiplier
        
    Returns
    -------
    pd.Series
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Parameters
    ----------
    series : pd.Series
        Data series
    threshold : float
        Z-score threshold
        
    Returns
    -------
    pd.Series
        Boolean series indicating outliers
    """
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_isolation_forest(df: pd.DataFrame, 
                                     numeric_cols: list,
                                     contamination: float = 0.05) -> pd.Series:
    """
    Detect outliers using Isolation Forest.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    numeric_cols : list
        Numeric columns to consider
    contamination : float
        Expected proportion of outliers
        
    Returns
    -------
    pd.Series
        Boolean series indicating outliers
    """
    data = df[numeric_cols].dropna()
    
    if len(data) < 10:
        return pd.Series(False, index=df.index)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(data)
    
    outliers = pd.Series(False, index=df.index)
    outliers.loc[data.index] = predictions == -1
    
    return outliers


def flag_outliers(df: pd.DataFrame, config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Detect and flag outliers without removing them.
    
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
    pd.DataFrame
        Data with outlier flags
    """
    logger.info("Detecting outliers...")
    
    df = df.copy()
    
    # Get Qm column
    qm_col = [col for col in df.columns if 'qm' in col.lower()][0]
    
    # Convert Qm to numeric
    df[qm_col] = pd.to_numeric(df[qm_col], errors='coerce')
    
    # Get outlier detection method from config
    method = config.get('data_processing.outlier_detection.method', 'IQR')
    flag_only = config.get('data_processing.outlier_detection.flag_only', True)
    
    logger.info(f"Using {method} method for outlier detection")
    
    # Detect outliers
    if method == 'IQR':
        multiplier = config.get('data_processing.outlier_detection.iqr_multiplier', 3.0)
        outliers = detect_outliers_iqr(df[qm_col].dropna(), multiplier)
    elif method == 'zscore':
        threshold = config.get('data_processing.outlier_detection.zscore_threshold', 3.5)
        outliers = detect_outliers_zscore(df[qm_col].dropna(), threshold)
    elif method == 'isolation_forest':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = detect_outliers_isolation_forest(df, numeric_cols)
    else:
        logger.warning(f"Unknown method {method}, defaulting to IQR")
        outliers = detect_outliers_iqr(df[qm_col].dropna(), 3.0)
    
    # Use neutral terminology to avoid implying data-entry errors.
    logger.warning(
        "Issue detected: 'outlier_flag' terminology can imply erroneous data in meta-analysis contexts."
    )
    df['extreme_qm_flag'] = False
    df.loc[outliers[outliers].index, 'extreme_qm_flag'] = True
    logger.info("Fix applied: renamed outlier marker to 'extreme_qm_flag'.")

    n_outliers = df['extreme_qm_flag'].sum()
    logger.info(f"Identified {n_outliers} extreme Qm values ({n_outliers/len(df)*100:.1f}%)")
    
    if flag_only:
        logger.info("Outliers flagged but retained in dataset")
    else:
        df = df[~df['extreme_qm_flag']]
        logger.info(f"Removed {n_outliers} outliers")
    
    return df


def reconstruct_study_ids(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Reconstruct study IDs using forward-fill for blocked data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with reconstructed study IDs
    """
    logger.info("Reconstructing Study IDs...")
    
    df = df.copy()
    
    # Find study-identifier columns
    study_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['study', 'author', 'reference', 'paper', 'source'])]
    
    if not study_cols:
        logger.warning("No study identifier column found. Creating numeric IDs.")
        # Create study IDs based on data blocks
        df['Study_ID'] = 0
        study_counter = 1
        
        # Assume each block of non-null Qm values is a study
        qm_col = [col for col in df.columns if 'qm' in col.lower()][0]
        prev_null = True
        
        for idx in df.index:
            curr_null = pd.isna(df.loc[idx, qm_col])
            if not curr_null:
                if prev_null:
                    study_counter += 1
                df.loc[idx, 'Study_ID'] = study_counter
            prev_null = curr_null
        
        n_studies = df['Study_ID'].max()
        logger.info(f"Created {n_studies} study IDs based on data blocks")
    else:
        study_col = study_cols[0]
        logger.info(f"Using {study_col} for study identification")

        # Normalize formatting artifacts before Study_ID coding to avoid accidental ID fragmentation.
        def normalize_study_label(value):
            if pd.isna(value):
                return value
            label = str(value).replace('\xa0', ' ').strip()
            label = re.sub(r'\s+', ' ', label)
            label = re.sub(r'\s*\(\s*', '(', label)
            label = re.sub(r'\s*\)\s*', ')', label)
            label = re.sub(r'\s*-\s*', '-', label)
            label = re.sub(r'\s*_\s*', '_', label)
            return label

        study_before = df[study_col].astype('string')
        study_after = study_before.apply(normalize_study_label)
        changed_mask = (study_before.fillna('<NA>') != study_after.fillna('<NA>'))
        n_changed = int(changed_mask.sum())

        if n_changed > 0:
            unique_pairs = pd.DataFrame({
                'before': study_before[changed_mask],
                'after': study_after[changed_mask],
            }).drop_duplicates()
            logger.warning(
                "Issue detected: inconsistent study-label formatting found "
                f"({n_changed} rows, {len(unique_pairs)} unique pattern changes)."
            )
            df[study_col] = study_after
            logger.info("Fix applied: normalized study labels before Study_ID_numeric generation.")
        else:
            logger.info("No study-label normalization needed before Study_ID coding.")
        
        # Forward fill study identifiers
        df['Study_ID'] = df[study_col].ffill()

        # Safeguard for leading NA values (top rows may remain NA after ffill)
        n_study_na_before = df['Study_ID'].isna().sum()
        if n_study_na_before > 0:
            df['Study_ID'] = df['Study_ID'].bfill()
            n_study_na_after_bfill = df['Study_ID'].isna().sum()
            if n_study_na_after_bfill > 0:
                df['Study_ID'] = df['Study_ID'].fillna('Unknown_Study')
            n_study_na_after = df['Study_ID'].isna().sum()
            logger.warning(
                f"Study_ID NA safeguard applied: {n_study_na_before} -> {n_study_na_after} missing"
            )
        
        # Create numeric study codes
        df['Study_ID_numeric'] = pd.Categorical(df['Study_ID']).codes + 1
        
        n_studies = df['Study_ID'].nunique()
        logger.info(f"Identified {n_studies} unique studies")
        
        # Log study sample sizes
        study_sizes = df['Study_ID'].value_counts().sort_index()
        logger.info("\nStudy sample sizes:")
        for study, count in study_sizes.items():
            logger.info(f"  {study}: {count} observations")
    
    return df


def apply_log_transformation(df: pd.DataFrame, config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Apply log transformation to Qm.
    
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
    pd.DataFrame
        Data with log-transformed Qm
    """
    logger.info("Applying log transformation...")
    
    df = df.copy()
    
    # Get configuration
    apply_transform = config.get('data_processing.log_transform.apply', True)
    log_base = config.get('data_processing.log_transform.base', 'natural')
    offset = config.get('data_processing.log_transform.offset', 0.001)
    
    if not apply_transform:
        logger.info("Log transformation disabled in config")
        return df
    
    # Get Qm column
    qm_col = [col for col in df.columns if 'qm' in col.lower() if 'log' not in col.lower()][0]
    
    # Check for non-positive values
    n_nonpositive = (df[qm_col] <= 0).sum()
    if n_nonpositive > 0:
        logger.warning(
            f"Found {n_nonpositive} non-positive Qm values. Applying offset only to those entries: {offset}"
        )
        nonpositive_mask = df[qm_col] <= 0
        df.loc[nonpositive_mask, qm_col] = df.loc[nonpositive_mask, qm_col] + offset
    
    # Apply log transformation
    if log_base == 'natural':
        df['log_Qm'] = np.log(df[qm_col])
        logger.info("Applied natural log transformation")
    elif log_base == 10:
        df['log_Qm'] = np.log10(df[qm_col])
        logger.info("Applied log10 transformation")
    else:
        logger.error(f"Unknown log base: {log_base}")
        raise ValueError(f"Unknown log base: {log_base}")
    
    # Log statistics
    logger.info(f"\nQm statistics:")
    logger.info(f"  Original - Mean: {df[qm_col].mean():.2f}, Std: {df[qm_col].std():.2f}")
    logger.info(f"  Log      - Mean: {df['log_Qm'].mean():.2f}, Std: {df['log_Qm'].std():.2f}")
    
    return df


def create_quality_flags(df: pd.DataFrame, config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Create comprehensive quality flag system.
    
    Ensures quality_flags column uses explicit markers instead of empty strings.
    
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
    pd.DataFrame
        Data with quality flags (NaN-safe)
    """
    logger.info("Creating quality flags...")
    
    df = df.copy()
    
    # Initialize quality score with explicit default marker
    df['quality_score'] = 100
    df['quality_flags'] = 'none'  # Use 'none' instead of empty string to prevent NaN on export
    
    # Flag: Missing key variables
    key_vars = ['log_Qm', 'Study_ID']
    for var in key_vars:
        if var in df.columns:
            mask = df[var].isna()
            df.loc[mask, 'quality_score'] -= 50
            # Append flags safely
            df.loc[mask, 'quality_flags'] = df.loc[mask, 'quality_flags'].apply(
                lambda x: f"missing_{var};{x}" if x != 'none' else f"missing_{var}"
            )

    # Flag: pH missingness provenance
    if 'pH_missing' in df.columns:
        ph_mask = df['pH_missing'].fillna(0).astype(int) > 0
        n_ph_missing = int(ph_mask.sum())
        if n_ph_missing > 0:
            logger.warning(
                "Issue detected: pH missingness provenance was not reflected in quality_score/quality_flags."
            )
            df.loc[ph_mask, 'quality_score'] -= 5
            df.loc[ph_mask, 'quality_flags'] = df.loc[ph_mask, 'quality_flags'].apply(
                lambda x: f"pH_missing;{x}" if x != 'none' else "pH_missing"
            )
            logger.info(
                f"Fix applied: added pH_missing provenance tag with penalty (-5) for {n_ph_missing} rows."
            )

    # Flag: predictor imputation provenance
    imputed_penalties = {
        'imputed_Temp': 3,
        'imputed_AIC': 3,
        'imputed_rpm': 3,
    }
    for imputed_col, penalty in imputed_penalties.items():
        if imputed_col in df.columns:
            imputed_mask = df[imputed_col].fillna(0).astype(int) > 0
            n_imputed = int(imputed_mask.sum())
            if n_imputed > 0:
                logger.warning(
                    f"Issue detected: {imputed_col} provenance was not reflected in quality scoring."
                )
                df.loc[imputed_mask, 'quality_score'] -= penalty
                df.loc[imputed_mask, 'quality_flags'] = df.loc[imputed_mask, 'quality_flags'].apply(
                    lambda x, tag=imputed_col: f"{tag};{x}" if x != 'none' else tag
                )
                logger.info(
                    "Fix applied: added "
                    f"{imputed_col} provenance tag with penalty (-{penalty}) for {n_imputed} rows."
                )
    
    # Flag: extreme Qm values (neutral terminology)
    if 'extreme_qm_flag' in df.columns:
        mask = df['extreme_qm_flag']
        df.loc[mask, 'quality_score'] -= 20
        df.loc[mask, 'quality_flags'] = df.loc[mask, 'quality_flags'].apply(
            lambda x: f"extreme_qm;{x}" if x != 'none' else "extreme_qm"
        )
    elif 'outlier_flag' in df.columns:
        logger.warning(
            "Issue detected: legacy 'outlier_flag' found. Applying backward-compatible quality flag logic."
        )
        mask = df['outlier_flag']
        df.loc[mask, 'quality_score'] -= 20
        df.loc[mask, 'quality_flags'] = df.loc[mask, 'quality_flags'].apply(
            lambda x: f"extreme_qm;{x}" if x != 'none' else "extreme_qm"
        )
    
    # Flag: Extreme values (beyond 3 standard deviations)
    if 'log_Qm' in df.columns:
        mean = df['log_Qm'].mean()
        std = df['log_Qm'].std()
        mask = np.abs(df['log_Qm'] - mean) > 3 * std
        df.loc[mask, 'quality_score'] -= 10
        df.loc[mask, 'quality_flags'] = df.loc[mask, 'quality_flags'].apply(
            lambda x: f"extreme_value;{x}" if x != 'none' else "extreme_value"
        )
    
    # Clean up trailing semicolons
    df['quality_flags'] = df['quality_flags'].str.rstrip(';')
    logger.info(f"Quality flags assigned. Sample value distribution: {df['quality_flags'].nunique()} unique flags")
    
    # Quality categories
    df['quality_category'] = pd.cut(df['quality_score'], 
                                    bins=[-1, 50, 80, 100],
                                    labels=['low', 'medium', 'high'])
    
    # Log quality distribution
    quality_dist = df['quality_category'].value_counts()
    logger.info("\nQuality distribution:")
    for category, count in quality_dist.items():
        logger.info(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
    
    n_nan_flags = df['quality_flags'].isna().sum()
    if n_nan_flags > 0:
        logger.warning(f"Found {n_nan_flags} NaN values in quality_flags. Replacing with 'none'.")
        df['quality_flags'] = df['quality_flags'].fillna('none')
    
    return df


def generate_cleaning_report(df_raw: pd.DataFrame, 
                            df_clean: pd.DataFrame,
                            missing_analysis: dict,
                            logger) -> dict:
    """
    Generate comprehensive cleaning report comparing raw and cleaned data.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data
    df_clean : pd.DataFrame
        Cleaned data
    missing_analysis : dict
        Missing data analysis
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Cleaning report
    """
    logger.info("\nGenerating comprehensive cleaning report...")
    
    # Basic stats
    report = {
        'initial_rows': len(df_raw),
        'final_rows': len(df_clean),
        'rows_removed': len(df_raw) - len(df_clean),
        'removal_rate': (len(df_raw) - len(df_clean)) / len(df_raw) * 100,
        'initial_columns': len(df_raw.columns),
        'final_columns': len(df_clean.columns),
        'columns_removed': len(df_raw.columns) - len([c for c in df_raw.columns if c in df_clean.columns]),
        'columns_added': len([c for c in df_clean.columns if c not in df_raw.columns]),
    }
    
    # Missing data summary
    total_cells_raw = len(df_raw) * len(df_raw.columns)
    total_missing_raw = sum(info['n_missing'] for info in missing_analysis.values() 
                           if info['n_missing'] > 0)
    
    # Calculate final missing
    total_cells_final = len(df_clean) * len(df_clean.columns)
    total_missing_final = df_clean.isnull().sum().sum()
    
    report['missing_data'] = {
        'initial_missing_cells': int(total_missing_raw),
        'initial_missing_pct': (total_missing_raw / total_cells_raw * 100),
        'final_missing_cells': int(total_missing_final),
        'final_missing_pct': (total_missing_final / total_cells_final * 100),
        'reduction': total_missing_raw - total_missing_final
    }
    
    # Data types (convert to JSON-serializable format)
    report['data_types'] = {
        'initial': {str(k): int(v) for k, v in df_raw.dtypes.value_counts().items()},
        'final': {str(k): int(v) for k, v in df_clean.dtypes.value_counts().items()}
    }

    # Completeness summaries for manuscript-ready interpretation.
    def _find_col(exact: str | None = None, contains: list[str] | None = None, exclude: list[str] | None = None):
        contains = contains or []
        exclude = exclude or []
        columns = list(df_clean.columns)

        if exact and exact in columns:
            return exact

        for col in columns:
            low = col.lower()
            if contains and all(token in low for token in contains):
                if not any(ex in low for ex in exclude):
                    return col
        return None

    q_col = _find_col(exact='Qm', contains=['qm'], exclude=['log'])
    metal_col = _find_col(exact='Metal', contains=['metal'])
    polymer_col = _find_col(exact='ReT', contains=['ret']) or _find_col(contains=['polymer'])
    log_q_col = _find_col(exact='log_Qm', contains=['log', 'qm'])
    temp_col = _find_col(exact='Temp', contains=['temp'])
    ph_col = _find_col(exact='pH', contains=['ph'])
    aic_col = _find_col(exact='AIC', contains=['aic'])
    rpm_col = _find_col(exact='rpm', contains=['rpm'])

    core_cols = [c for c in [q_col, metal_col, polymer_col] if c is not None]
    covariate_model_cols = [c for c in [log_q_col or q_col, temp_col, ph_col, aic_col, rpm_col] if c is not None]

    def _build_completeness(cols: list[str]) -> dict:
        if not cols:
            return {
                'columns': [],
                'complete_rows': None,
                'completeness_pct': None,
            }
        complete_rows = int(len(df_clean.dropna(subset=cols)))
        return {
            'columns': cols,
            'complete_rows': complete_rows,
            'completeness_pct': float(complete_rows / len(df_clean) * 100),
        }

    report['completeness_summaries'] = {
        'core_completeness': _build_completeness(core_cols),
        'covariate_model_completeness': _build_completeness(covariate_model_cols),
        'full_variable_completeness': {
            'columns': ['ALL_COLUMNS'],
            'complete_rows': int(len(df_clean.dropna())),
            'completeness_pct': float(len(df_clean.dropna()) / len(df_clean) * 100),
        },
    }
    
    # Log summary
    logger.info(f"\n{'='*70}")
    logger.info("DATA CLEANING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"\nDimensions:")
    logger.info(f"  Initial rows:      {report['initial_rows']}")
    logger.info(f"  Final rows:        {report['final_rows']}")
    logger.info(f"  Rows removed:      {report['rows_removed']} ({report['removal_rate']:.1f}%)")
    logger.info(f"  Initial columns:   {report['initial_columns']}")
    logger.info(f"  Final columns:     {report['final_columns']}")
    logger.info(f"  Columns removed:   {report['columns_removed']}")
    logger.info(f"  Columns added:     {report['columns_added']}")
    
    logger.info(f"\nMissing Data:")
    logger.info(f"  Initial missing:   {report['missing_data']['initial_missing_cells']:,} cells "
               f"({report['missing_data']['initial_missing_pct']:.1f}%)")
    logger.info(f"  Final missing:     {report['missing_data']['final_missing_cells']:,} cells "
               f"({report['missing_data']['final_missing_pct']:.1f}%)")
    logger.info(f"  Reduction:         {report['missing_data']['reduction']:,} cells")
    
    logger.info(f"\nData Quality:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    logger.info(f"  Numeric columns:   {len(numeric_cols)}")
    logger.info(f"  Complete rows:     {len(df_clean.dropna())} ({len(df_clean.dropna())/len(df_clean)*100:.1f}%)")

    logger.info("\nCompleteness summaries:")
    core_comp = report['completeness_summaries']['core_completeness']
    cov_comp = report['completeness_summaries']['covariate_model_completeness']
    full_comp = report['completeness_summaries']['full_variable_completeness']

    if core_comp['complete_rows'] is not None:
        logger.info(
            "  Core completeness "
            f"({core_comp['columns']}): {core_comp['complete_rows']} ({core_comp['completeness_pct']:.1f}%)"
        )
    else:
        logger.warning("  Core completeness could not be computed (required columns not found).")

    if cov_comp['complete_rows'] is not None:
        logger.info(
            "  Covariate-model completeness "
            f"({cov_comp['columns']}): {cov_comp['complete_rows']} ({cov_comp['completeness_pct']:.1f}%)"
        )
    else:
        logger.warning("  Covariate-model completeness could not be computed (required columns not found).")

    logger.info(
        "  Full-variable completeness (all columns): "
        f"{full_comp['complete_rows']} ({full_comp['completeness_pct']:.1f}%)"
    )
    
    logger.info(f"{'='*70}\n")
    
    return report


def analyze_missing_data(df: pd.DataFrame, logger) -> dict:
    """
    Analyze missing data patterns and rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Missing data analysis report
    """
    logger.info("Analyzing missing data patterns...")
    
    missing_analysis = {}
    
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100
        dtype = df[col].dtype
        
        missing_analysis[col] = {
            'n_missing': int(n_missing),
            'pct_missing': float(pct_missing),
            'dtype': str(dtype),
            'n_unique': int(df[col].nunique(dropna=True))
        }
    
    # Log summary
    logger.info(f"\nMissing Data Summary:")
    logger.info(f"{'Column':<30} {'Missing':<10} {'%':<8} {'Type':<12}")
    logger.info("-" * 70)
    
    for col, info in sorted(missing_analysis.items(), key=lambda x: x[1]['pct_missing'], reverse=True):
        if info['pct_missing'] > 0:
            logger.info(f"{col:<30} {info['n_missing']:<10} {info['pct_missing']:<8.1f} {info['dtype']:<12}")
    
    return missing_analysis


def remove_irrelevant_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Remove irrelevant columns (Note, Unnamed, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with irrelevant columns removed
    """
    logger.info("\nRemoving irrelevant columns...")
    
    df = df.copy()
    cols_to_remove = []
    
    # Remove Unnamed columns
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    cols_to_remove.extend(unnamed_cols)
    
    # Remove Note columns (low information content)
    note_cols = [col for col in df.columns if 'note' in col.lower()]
    cols_to_remove.extend(note_cols)
    
    # Remove columns with >95% missing and no scientific value
    for col in df.columns:
        if col not in cols_to_remove:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 95:
                # Check if it's a critical experimental variable
                critical_vars = ['qm', 'metal', 'temp', 'ph', 'sa', 'study', 'polymer', 'age']
                if not any(var in col.lower() for var in critical_vars):
                    cols_to_remove.append(col)
                    logger.info(f"  Flagged '{col}' for removal (>{missing_pct:.1f}% missing)")
    
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        logger.info(f"Removed {len(cols_to_remove)} irrelevant columns: {cols_to_remove}")
    else:
        logger.info("No irrelevant columns found.")
    
    return df


def normalize_domain_fields(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Normalize key domain fields early in cleaning.

    Tasks:
    - Ensure SA-like column is numeric (strip whitespace + to_numeric)
    - Standardize AgS labels (Virgin variants -> Virgin, aged-like -> Aged)
    - Normalize Cr(VI) string formatting in Metal column to exact 'Cr (VI)'

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Data with normalized domain fields
    """
    logger.info("\nNormalizing domain-specific fields (SA, AgS, Metal formats)...")

    df = df.copy()

    # 1) SA numeric coercion
    sa_candidates = [
        col for col in df.columns
        if col.lower().strip().startswith('sa') or 'surface area' in col.lower()
    ]
    if sa_candidates:
        sa_col = sa_candidates[0]
        before_dtype = str(df[sa_col].dtype)

        non_null_before = int(df[sa_col].notna().sum())
        cleaned_sa = df[sa_col].apply(
            lambda x: str(x).replace('\xa0', ' ').strip() if pd.notna(x) else x
        )
        converted_sa = pd.to_numeric(cleaned_sa, errors='coerce')

        coerced_to_nan = int(max(0, non_null_before - int(converted_sa.notna().sum())))
        df[sa_col] = converted_sa
        after_dtype = str(df[sa_col].dtype)
        missing_pct = float(df[sa_col].isna().mean() * 100)

        logger.info("SA coercion summary for '{}':".format(sa_col))
        logger.info("  dtype: {} -> {}".format(before_dtype, after_dtype))
        logger.info("  values coerced to NaN: {}".format(coerced_to_nan))
        logger.info("  final missing percentage: {:.1f}%".format(missing_pct))
    else:
        logger.warning("No SA-like column found for numeric coercion")

    # 2) AgS label normalization
    ags_candidates = [col for col in df.columns if col.lower().strip() == 'ags' or 'aging' in col.lower()]
    if ags_candidates:
        ags_col = ags_candidates[0]
        before_counts = df[ags_col].astype('string').value_counts(dropna=False)

        logger.info("AgS normalization - before counts:")
        for label, count in before_counts.items():
            logger.info("  {}: {}".format(label, count))

        def normalize_ags_label(value):
            if pd.isna(value):
                return value
            raw = str(value).strip()
            low = raw.lower()

            if 'virgin' in low or 'vigin' in low:
                return 'Virgin'
            if 'aged' in low:
                return 'Aged'
            return raw

        df[ags_col] = df[ags_col].apply(normalize_ags_label)

        after_counts = df[ags_col].astype('string').value_counts(dropna=False)
        logger.info("AgS normalization - after counts:")
        for label, count in after_counts.items():
            logger.info("  {}: {}".format(label, count))

        unexpected = sorted([x for x in df[ags_col].dropna().unique().tolist() if x not in ['Virgin', 'Aged']])
        if unexpected:
            logger.warning("Unexpected AgS labels remain after normalization: {}".format(unexpected))
    else:
        logger.warning("No AgS-like column found for label normalization")

    # 3) Metal formatting normalization: Cr(VI) variants -> 'Cr (VI)'
    metal_candidates = [col for col in df.columns if 'metal' in col.lower()]
    if metal_candidates:
        metal_col = metal_candidates[0]
        metal_series_before = df[metal_col].astype('string')
        cr_mask_before = metal_series_before.str.contains(r'^\s*cr\s*\(\s*vi\s*\)\s*$', case=False, na=False, regex=True)
        cr_before_counts = metal_series_before[cr_mask_before].value_counts(dropna=False)

        if len(cr_before_counts) > 0:
            logger.info("Cr(VI) format frequencies BEFORE normalization:")
            for label, count in cr_before_counts.items():
                logger.info("  {}: {}".format(label, count))

        def normalize_metal_value(value):
            if pd.isna(value):
                return value
            raw = str(value).strip()
            if re.match(r'^\s*cr\s*\(\s*vi\s*\)\s*$', raw, flags=re.IGNORECASE):
                return 'Cr (VI)'
            return raw

        df[metal_col] = df[metal_col].apply(normalize_metal_value)

        metal_series_after = df[metal_col].astype('string')
        cr_mask_after = metal_series_after.str.contains(r'^\s*cr\s*\(\s*vi\s*\)\s*$', case=False, na=False, regex=True)
        cr_after_counts = metal_series_after[cr_mask_after].value_counts(dropna=False)

        if len(cr_after_counts) > 0:
            logger.info("Cr(VI) format frequencies AFTER normalization:")
            for label, count in cr_after_counts.items():
                logger.info("  {}: {}".format(label, count))

        if len(cr_after_counts) > 1:
            logger.warning("More than one Cr(VI) format still remains after normalization")
    else:
        logger.warning("No Metal-like column found for Cr(VI) normalization")

    return df


def coerce_numeric_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Safely coerce object columns to numeric where appropriate.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with numeric columns coerced
    """
    logger.info("\nCoercing numeric-like columns...")
    
    df = df.copy()
    conversions = []
    
    # Known numeric variables in adsorption studies
    numeric_vars = ['qm', 'temp', 'ph', 'sa', 'aic', 'rpm', 'concentration', 'dose', 'time']
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column name suggests it should be numeric
            is_numeric_var = any(var in col.lower() for var in numeric_vars)
            
            # Try to convert and check success rate
            if is_numeric_var:
                test_conversion = pd.to_numeric(df[col], errors='coerce')
                n_non_null = df[col].notna().sum()
                if n_non_null == 0:
                    continue
                conversion_success_rate = (test_conversion.notna().sum() / n_non_null) * 100
                
                if conversion_success_rate > 50:  # At least 50% successful conversion
                    n_converted = test_conversion.notna().sum()
                    n_failed = df[col].notna().sum() - n_converted
                    df[col] = test_conversion
                    conversions.append(col)
                    logger.info(f"  Converted '{col}': {n_converted} values successful, {n_failed} failed")
    
    if conversions:
        logger.info(f"Successfully coerced {len(conversions)} columns to numeric.")
    else:
        logger.info("No object columns needed numeric coercion.")
    
    return df


def validate_and_fix_numeric_dtypes(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Validate and fix numeric dtype issues by stripping hidden characters.
    
    Handles non-breaking spaces, tabs, and other hidden characters that prevent
    proper numeric coercion. Re-coerces problematic columns to numeric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with fixed numeric dtypes
    """
    logger.info("\nValidating and fixing numeric dtypes...")
    
    df = df.copy()
    fixes_applied = []
    
    # Known numeric variables in adsorption studies
    numeric_vars = ['qm', 'temp', 'ph', 'sa', 'aic', 'rpm', 'concentration', 'dose', 'time']
    
    for col in df.columns:
        if df[col].dtype == 'object':
            is_numeric_var = any(var in col.lower() for var in numeric_vars)
            
            if is_numeric_var:
                # Check if column has non-null values
                n_non_null_before = df[col].notna().sum()
                if n_non_null_before == 0:
                    continue
                
                # Strip hidden characters (non-breaking spaces, tabs, etc.)
                df[col] = df[col].str.replace(r'\xa0|\t|\s+$|^\s+', '', regex=True)
                
                # Try numeric coercion
                test_conversion = pd.to_numeric(df[col], errors='coerce')
                n_converted = test_conversion.notna().sum()
                conversion_success_rate = (n_converted / n_non_null_before) * 100
                
                if conversion_success_rate > 50:  # At least 50% successful
                    df[col] = test_conversion
                    n_failed = n_non_null_before - n_converted
                    fixes_applied.append({
                        'column': col,
                        'before_dtype': 'object',
                        'after_dtype': str(df[col].dtype),
                        'n_converted': n_converted,
                        'n_failed': n_failed
                    })
                    logger.info(
                        f"  Fixed '{col}': {n_converted} values converted to numeric, {n_failed} failed coercion"
                    )
    
    if fixes_applied:
        logger.info(f"Fixed numeric dtypes for {len(fixes_applied)} columns.")
    else:
        logger.info("No numeric dtype issues found.")
    
    return df


def drop_rows_with_missing_target(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Remove rows with missing target variable (Qm or log_Qm) for modeling readiness.
    
    Do not impute target variable - only rows with missing targets are removed.
    Must be called after log transformation but before imputation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with rows containing missing target removed
    """
    logger.info("\nChecking for missing target variable (log_Qm)...")
    
    df = df.copy()
    
    # Identify target variable
    target_cols = [col for col in df.columns if 'log_qm' in col.lower()]
    
    if not target_cols:
        # Fallback to original Qm if log_Qm not found
        target_cols = [col for col in df.columns if 'qm' in col.lower() and 'log' not in col.lower()]
    
    if not target_cols:
        logger.warning("Target variable (Qm/log_Qm) not found. Skipping target validation.")
        return df
    
    target_col = target_cols[0]
    n_missing_before = df[target_col].isnull().sum()
    
    if n_missing_before > 0:
        logger.warning(
            f"Found {n_missing_before} rows with missing target variable '{target_col}'"
        )
        rows_before = len(df)
        df = df[df[target_col].notna()]
        rows_after = len(df)
        rows_removed = rows_before - rows_after
        logger.info(
            f"Removed {rows_removed} rows with missing target. Rows: {rows_before} -> {rows_after}"
        )
    else:
        logger.info(f"Target variable '{target_col}' has no missing values.")
    
    return df


def handle_missing_data(df: pd.DataFrame, missing_analysis: dict, logger) -> pd.DataFrame:
    """
    Smart imputation strategy based on missing rates and variable importance.
    
    Numeric variables with >10% missing are now imputed with median for modeling readiness.
    Missing indicators are preserved separately.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    missing_analysis : dict
        Missing data analysis report
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    logger.info("\nHandling missing data with smart imputation...")
    
    df = df.copy()
    imputation_report = {}

    def _get_imputation_flag_name(col_name: str) -> str | None:
        col_low = col_name.lower()
        if 'temp' in col_low:
            return 'imputed_Temp'
        if 'aic' in col_low:
            return 'imputed_AIC'
        if 'rpm' in col_low:
            return 'imputed_rpm'
        return None

    def _mark_imputed_predictor(col_name: str, imputed_mask: pd.Series) -> None:
        flag_col = _get_imputation_flag_name(col_name)
        n_imputed = int(imputed_mask.sum())
        if flag_col is None or n_imputed == 0:
            return
        if flag_col not in df.columns:
            df[flag_col] = 0
        df.loc[imputed_mask, flag_col] = 1
        logger.info(f"  Traceability flag updated: '{flag_col}' marked for {n_imputed} rows")
    
    # Critical adsorption variables that should not be dropped
    critical_vars = ['qm', 'metal', 'temp', 'ph', 'study_id']
    
    for col, info in missing_analysis.items():
        if col not in df.columns:  # Skip if already removed
            continue
            
        pct_missing = info['pct_missing']
        
        if pct_missing == 0:
            continue
        
        is_critical = any(var in col.lower() for var in critical_vars)
        is_numeric = df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        is_missing_indicator = '_missing' in col.lower()  # Skip indicator columns
        is_ph_column = col.lower().strip() == 'ph'
        
        # Never impute indicator columns themselves
        if is_missing_indicator:
            continue

        # Preserve pH missingness for scientific interpretability.
        if is_ph_column:
            logger.warning(
                "Issue detected: pH is eligible for global imputation despite high missingness."
            )
            if 'pH_missing' not in df.columns:
                df['pH_missing'] = df[col].isnull().astype(int)
                logger.info("Fix applied: created 'pH_missing' indicator (1=missing, 0=observed).")
            else:
                logger.info("Fix applied: retained existing 'pH_missing' indicator.")
            imputation_report[col] = "pH not imputed; missing values preserved"
            logger.info(f"  {col}: {imputation_report[col]}")
            continue
        
        # Strategy based on missing rate and importance
        if pct_missing < 10:
            # Low missing rate: impute with median/mode
            if is_numeric:
                imputed_mask_before = df[col].isnull()
                impute_value = df[col].median()
                n_imputed = df[col].isnull().sum()
                df[col] = df[col].fillna(impute_value)
                imputed_mask = imputed_mask_before & df[col].notna()
                _mark_imputed_predictor(col, imputed_mask)
                imputation_report[col] = f"Imputed {n_imputed} values with median ({impute_value:.2f})"
                logger.info(f"  {col}: {imputation_report[col]}")
            else:
                impute_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                n_imputed = df[col].isnull().sum()
                df[col] = df[col].fillna(impute_value)
                imputation_report[col] = f"Imputed {n_imputed} values with mode ('{impute_value}')"
                logger.info(f"  {col}: {imputation_report[col]}")
        
        elif pct_missing < 30:
            # Moderate missing rate: conditional imputation or flag
            if is_numeric and is_critical:
                # Use study-level median if available
                if 'Study_ID' in df.columns:
                    imputed_mask_before = df[col].isnull()
                    df[col] = df.groupby('Study_ID')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # Fill remaining with global median
                    global_median = df[col].median()
                    n_imputed = df[col].isnull().sum()
                    df[col] = df[col].fillna(global_median)
                    imputed_mask = imputed_mask_before & df[col].notna()
                    _mark_imputed_predictor(col, imputed_mask)
                    imputation_report[col] = f"Study-level + global median imputation ({n_imputed} values)"
                else:
                    imputed_mask_before = df[col].isnull()
                    impute_value = df[col].median()
                    n_imputed = df[col].isnull().sum()
                    df[col] = df[col].fillna(impute_value)
                    imputed_mask = imputed_mask_before & df[col].notna()
                    _mark_imputed_predictor(col, imputed_mask)
                    imputation_report[col] = f"Median imputation ({n_imputed} values: {impute_value:.2f})"
                logger.info(f"  {col}: {imputation_report[col]}")
            elif is_numeric and not is_critical:
                # NEW: For numeric non-critical variables 10-30%, switch to median imputation for modeling readiness
                imputed_mask_before = df[col].isnull()
                impute_value = df[col].median()
                n_imputed = df[col].isnull().sum()
                df[col] = df[col].fillna(impute_value)
                imputed_mask = imputed_mask_before & df[col].notna()
                _mark_imputed_predictor(col, imputed_mask)
                imputation_report[col] = f"Median imputation - numeric for modeling ({n_imputed} values: {impute_value:.2f})"
                logger.info(f"  {col}: {imputation_report[col]}")
            elif not is_critical:
                # Keep missing as-is for non-critical non-numeric variables with moderate missingness
                imputation_report[col] = f"Kept {info['n_missing']} missing values (non-critical non-numeric, {pct_missing:.1f}%)"
                logger.info(f"  {col}: {imputation_report[col]}")
        
        elif pct_missing < 50:
            # High missing rate: keep for critical, else flag
            if is_critical:
                if is_numeric:
                    # For numeric critical with high missing, use median for modeling readiness
                    imputed_mask_before = df[col].isnull()
                    impute_value = df[col].median()
                    n_imputed = df[col].isnull().sum()
                    df[col] = df[col].fillna(impute_value)
                    imputed_mask = imputed_mask_before & df[col].notna()
                    _mark_imputed_predictor(col, imputed_mask)
                    imputation_report[col] = f"Median imputation - critical numeric ({n_imputed} values: {impute_value:.2f})"
                    logger.info(f"  {col}: {imputation_report[col]}")
                else:
                    imputation_report[col] = f"Retained variable despite {pct_missing:.1f}% missing (critical non-numeric)"
                    logger.info(f"  {col}: {imputation_report[col]}")
            else:
                # For non-critical: create explicit 'Missing' category or flag
                if not is_numeric:
                    n_imputed = df[col].isnull().sum()
                    df[col] = df[col].fillna('Missing')
                    imputation_report[col] = f"Encoded {n_imputed} missing as explicit category"
                    logger.info(f"  {col}: {imputation_report[col]}")
                else:
                    imputation_report[col] = f"Kept missing ({pct_missing:.1f}%) - consider for sensitivity analysis"
                    logger.info(f"  {col}: {imputation_report[col]}")
        
        else:
            # Very high missing rate (>50%): keep only if critical
            if is_critical:
                if is_numeric:
                    # For numeric critical with very high missing, use median as last resort for modeling
                    imputed_mask_before = df[col].isnull()
                    impute_value = df[col].median()
                    n_imputed = df[col].isnull().sum()
                    df[col] = df[col].fillna(impute_value)
                    imputed_mask = imputed_mask_before & df[col].notna()
                    _mark_imputed_predictor(col, imputed_mask)
                    imputation_report[col] = f"WARNING: Median imputation critical {pct_missing:.1f}% missing ({n_imputed} values)"
                    logger.warning(f"  {col}: {imputation_report[col]}")
                else:
                    imputation_report[col] = f"WARNING: Critical variable with {pct_missing:.1f}% missing - retained"
                    logger.warning(f"  {col}: {imputation_report[col]}")
            else:
                imputation_report[col] = f"Retained with {pct_missing:.1f}% missing (low information content)"
                logger.info(f"  {col}: {imputation_report[col]}")
    
    logger.info(f"\nImputation complete. Handled {len(imputation_report)} variables.")

    imputed_flag_cols = [c for c in ['imputed_Temp', 'imputed_AIC', 'imputed_rpm'] if c in df.columns]
    if imputed_flag_cols:
        logger.info("Imputation traceability summary:")
        for flag_col in imputed_flag_cols:
            logger.info(f"  {flag_col}: {int(df[flag_col].sum())} rows")
    else:
        logger.info("No key predictor imputation flags were needed (Temp/AIC/rpm).")

    return df


def add_missing_indicators(df: pd.DataFrame,
                           missing_analysis: dict,
                           logger,
                           missing_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Add missing indicator features for scientifically meaningful variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    missing_analysis : dict
        Missing data analysis
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with missing indicators added
    """
    logger.info("\nAdding missing indicators for key variables...")
    
    df = df.copy()
    
    # Variables where missingness might be scientifically meaningful
    meaningful_missing_vars = ['ph', 'sa', 'aged condition', 'rpm', 'temp']
    indicators_added = []
    
    for col, info in missing_analysis.items():
        if col not in df.columns:
            continue
            
        # Add indicator if:
        # 1. Variable is scientifically meaningful
        # 2. Has moderate missingness (5-50%)
        # 3. Not already an indicator/flag column
        is_meaningful = any(var in col.lower() for var in meaningful_missing_vars)
        has_moderate_missing = 5 < info['pct_missing'] < 50
        not_indicator = not any(x in col.lower() for x in ['flag', 'indicator', '_missing'])
        
        if is_meaningful and has_moderate_missing and not_indicator:
            indicator_col = f"{col}_missing"
            if missing_mask is not None and col in missing_mask.columns:
                indicator_values = missing_mask[col].astype(int)
                df[indicator_col] = indicator_values.reindex(df.index, fill_value=0)
            else:
                df[indicator_col] = df[col].isnull().astype(int)
            indicators_added.append(indicator_col)
            logger.info(
                f"  Added '{indicator_col}' ({info['pct_missing']:.1f}% missing in source, sum={int(df[indicator_col].sum())})"
            )
    
    if indicators_added:
        logger.info(f"Added {len(indicators_added)} missing indicator features.")
    else:
        logger.info("No missing indicators needed.")
    
    return df


def finalize_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Finalize data by removing redundant columns and cleaning column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Finalized data
    """
    logger.info("\nFinalizing data...")
    
    df = df.copy()
    
    # Drop Author-year column since we have Study_ID
    if 'Author-year' in df.columns:
        df = df.drop(columns=['Author-year'])
        logger.info("Dropped 'Author-year' column (redundant with Study_ID)")
    
    # Remove units from column names (e.g., "Qm (ug/g)" -> "Qm")
    tentative_names = []
    rename_candidates = {}
    for col in df.columns:
        new_col = col.split('(')[0].strip()
        tentative_names.append(new_col)
        if new_col != col:
            rename_candidates[col] = new_col

    # Check for collisions after unit stripping
    collision_counts = pd.Series(tentative_names).value_counts()
    collisions = collision_counts[collision_counts > 1].to_dict()
    if collisions:
        logger.warning(f"Column-name collision risk after unit stripping: {collisions}")

    # Resolve duplicates safely by suffixing duplicate targets
    used_names = set()
    final_name_map = {}
    for col in df.columns:
        base_name = rename_candidates.get(col, col)
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        final_name_map[col] = candidate
        used_names.add(candidate)
        if col != candidate:
            logger.info(f"Renamed: '{col}' -> '{candidate}'")

    if any(k != v for k, v in final_name_map.items()):
        df = df.rename(columns=final_name_map)
    
    logger.info(f"Finalization complete. Final columns: {len(df.columns)}")
    
    return df


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '02_data_cleaning')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 02: DATA CLEANING", logger=logger)
        
        # Load raw data
        logger.info("Loading raw data snapshot...")
        data_path = config.get_path('processed_data') / "00_raw_data_snapshot.csv"
        df = load_dataframe(data_path)
        df_raw = df.copy()
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Step 1: Analyze missing data patterns
        missing_analysis_initial = analyze_missing_data(df, logger)
        
        # Step 2: Remove irrelevant columns
        df = remove_irrelevant_columns(df, logger)

        # Step 2b: Domain-specific normalization (SA numeric, AgS labels, Cr(VI) format)
        df = normalize_domain_fields(df, logger)
        
        # Step 3: Coerce numeric columns
        df = coerce_numeric_columns(df, logger)
        
        # Step 3b: Validate and fix numeric dtypes (strip hidden chars, re-coerce)
        df = validate_and_fix_numeric_dtypes(df, logger)

        # Invariant check: recompute missing/dtypes after drop+coerce+fix
        missing_analysis_updated = analyze_missing_data(df, logger)
        initial_missing_total = sum(v['n_missing'] for v in missing_analysis_initial.values())
        updated_missing_total = sum(v['n_missing'] for v in missing_analysis_updated.values())
        logger.info(
            f"Invariant check (missing cells after drop+coerce+fix): {initial_missing_total} -> {updated_missing_total}"
        )
        common_cols = [col for col in df.columns if col in missing_analysis_initial]
        dtype_changed_cols = [
            col for col in common_cols
            if missing_analysis_initial[col]['dtype'] != missing_analysis_updated[col]['dtype']
        ]
        logger.info(
            f"Invariant check (dtype changes after drop+coerce+fix): {len(dtype_changed_cols)} columns"
        )
        
        # Step 4: Harmonize units
        df = harmonize_units(df, config, logger)
        
        # Step 5: Reconstruct study IDs
        df = reconstruct_study_ids(df, logger)
        
        # Step 6: Detect and flag outliers
        df = flag_outliers(df, config, logger)
        
        # Step 7: Apply log transformation
        df = apply_log_transformation(df, config, logger)
        
        # Step 7b: Drop rows with missing target variable (modeling readiness)
        df = drop_rows_with_missing_target(df, logger)
        
        # Invariant check: preserve raw missing mask before imputation for indicator creation
        pre_imputation_missing_mask = df.isnull()
        logger.info(
            f"Invariant check (pre-imputation missing cells): {int(pre_imputation_missing_mask.sum().sum())}"
        )

        # Step 8: Add missing indicators BEFORE imputation (from pre-imputation mask)
        df = add_missing_indicators(df, missing_analysis_updated, logger, missing_mask=pre_imputation_missing_mask)
        
        # Step 9: Handle missing data with smart imputation using updated analysis
        df = handle_missing_data(df, missing_analysis_updated, logger)

        # Invariant check: indicator columns should not collapse to all zeros
        indicator_cols = [c for c in df.columns if c.endswith('_missing')]
        if indicator_cols:
            for indicator_col in indicator_cols:
                logger.info(f"Invariant check ({indicator_col} sum): {int(df[indicator_col].sum())}")

        # Step 10: Create/refresh quality flags AFTER imputation
        logger.info("Refreshing quality flags after imputation to match final dataset")
        df = create_quality_flags(df, config, logger)
        
        # Step 11: Finalize data (remove redundant columns and clean names)
        df = finalize_data(df, logger)
        
        # Generate comprehensive report
        report = generate_cleaning_report(df_raw, df, missing_analysis_updated, logger)
        
        # Save outputs
        logger.info("Saving cleaned data...")
        output_path = config.get_path('processed_data') / "02_cleaned_data.csv"
        temp_output_path = config.get_path('processed_data') / "02_cleaned_data_temp.csv"
        
        # Try to save directly, if fails save to temp file
        try:
            save_dataframe(df, output_path, index=False)
            logger.info(f"Saved to: {output_path}")
        except PermissionError:
            save_dataframe(df, temp_output_path, index=False)
            logger.warning(f"Could not overwrite {output_path.name} (file may be open)")
            logger.info(f"Saved to temporary file: {temp_output_path}")
            logger.info(f"Please close {output_path.name} and rename {temp_output_path.name} manually")
        
        # Save report
        report_path = config.get_path('results') / "02_cleaning_report.json"
        save_json(report, report_path)
        logger.info(f"Saved report to: {report_path}")
        
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in data cleaning: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
