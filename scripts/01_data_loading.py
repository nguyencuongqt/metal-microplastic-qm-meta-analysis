"""
Script 01: Data Loading
=======================

This script loads raw data from Excel file and performs initial inspection.

Tasks:
- Load raw Qm data
- Inspect data structure
- Identify columns and data types
- Document initial data quality
- Save raw data snapshot

Author: Manuscript authors
Date: March 2026
"""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   save_dataframe, describe_dataframe, print_section_header)


def load_raw_data(config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Load raw data from Excel file.
    
    Parameters
    ----------
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Raw data
    """
    logger.info("Loading raw data...")
    
    # Get file path from config
    data_path = config.get_path('raw_data')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    
    # Load Excel file
    # Try to read all sheets and combine if necessary
    try:
        # First, check what sheets exist
        excel_file = pd.ExcelFile(data_path)
        sheet_names = excel_file.sheet_names
        logger.info(f"Found {len(sheet_names)} sheet(s): {sheet_names}")
        
        # Load the first sheet (or a specific sheet if configured)
        target_sheet = sheet_names[0]  # Modify if needed
        df = pd.read_excel(data_path, sheet_name=target_sheet)
        
        logger.info(f"Loaded sheet: {target_sheet}")
        logger.info(f"Initial shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise
    
    return df


def inspect_data_structure(df: pd.DataFrame, logger) -> dict:
    """
    Inspect and document data structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Data structure summary
    """
    logger.info("\nInspecting data structure...")
    
    inspection = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_counts': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Log column information
    logger.info(f"\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns")
    logger.info("\nColumns found:")
    for i, col in enumerate(df.columns, 1):
        missing_pct = inspection['missing_percentages'][col]
        dtype = inspection['dtypes'][col]
        logger.info(f"  {i:2d}. {col:30s} | {str(dtype):12s} | {missing_pct:5.1f}% missing")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    inspection['numeric_columns'] = numeric_cols
    inspection['categorical_columns'] = categorical_cols
    
    return inspection


def identify_key_variables(df: pd.DataFrame, logger) -> dict:
    """
    Identify key variables expected in meta-analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Mapping of expected variables to actual columns
    """
    logger.info("\nIdentifying key variables...")
    
    # Expected variable patterns (case-insensitive)
    variable_patterns = {
        'Qm': ['qm', 'q_max', 'qmax', 'adsorption', 'capacity'],
        'Metal': ['metal', 'heavy metal', 'element'],
        'Polymer': ['polymer', 'plastic', 'microplastic', 'material', 'resin', 'ret'],
        'pH': ['ph', 'ph value'],
        'Temperature': ['temp', 'temperature'],
        'SA': ['sa', 'surface area', 'surface_area', 'area'],
        'Aging': ['aging', 'age', 'weathering', 'treatment'],
        'Study': ['study', 'reference', 'author', 'paper', 'source'],
        'DOI': ['doi', 'doi number'],
        'Year': ['year', 'publication year'],
    }
    
    variable_mapping = {}
    
    columns_lower = {col.lower().strip(): col for col in df.columns}
    
    for var_name, patterns in variable_patterns.items():
        found = False
        for pattern in patterns:
            for col_lower, col_original in columns_lower.items():
                if pattern in col_lower:
                    variable_mapping[var_name] = col_original
                    logger.info(f"  {var_name:15s} -> {col_original}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            logger.warning(f"  {var_name:15s} -> NOT FOUND")
            variable_mapping[var_name] = None

    # Post-check for known false positive: Polymer incorrectly mapped to temperature columns.
    polymer_col = variable_mapping.get('Polymer')
    if polymer_col and 'temp' in polymer_col.lower():
        logger.warning(
            "Issue detected: Polymer variable mapped to a temperature-like column "
            f"('{polymer_col}')."
        )

        polymer_candidates = [
            col for col in df.columns
            if any(token in col.lower() for token in ['ret', 'resin', 'polymer', 'plastic'])
            and 'temp' not in col.lower()
        ]

        if polymer_candidates:
            variable_mapping['Polymer'] = polymer_candidates[0]
            logger.info(
                "Fix applied: Re-mapped Polymer variable to "
                f"'{variable_mapping['Polymer']}'."
            )
        else:
            variable_mapping['Polymer'] = None
            logger.warning(
                "Fix attempt: No suitable Polymer candidate column found; mapping set to None."
            )

    logger.info("Final key variable mapping:")
    for key, value in variable_mapping.items():
        logger.info(f"  {key:15s} -> {value}")
    
    return variable_mapping


def audit_and_fix_qm_numeric(df: pd.DataFrame,
                             inspection: dict,
                             variable_mapping: dict,
                             logger) -> dict:
    """
    Audit Qm numeric integrity and coerce to numeric if currently object-like.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    inspection : dict
        Data structure inspection (updated in-place when fix is applied)
    variable_mapping : dict
        Variable mapping results
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Qm audit summary
    """
    qm_col = variable_mapping.get('Qm')
    audit = {
        'qm_column': qm_col,
        'dtype_before': None,
        'dtype_after': None,
        'non_empty_values': 0,
        'numeric_values': 0,
        'malformed_count': 0,
        'malformed_examples': [],
        'fix_applied': False,
    }

    if not qm_col or qm_col not in df.columns:
        logger.warning("Qm audit skipped: Qm column is not available.")
        return audit

    logger.info("\nAuditing Qm numeric integrity...")

    raw_series = df[qm_col]
    audit['dtype_before'] = str(raw_series.dtype)

    non_empty_mask = raw_series.notna() & (raw_series.astype(str).str.strip() != '')
    normalized = raw_series.astype(str).str.replace(',', '', regex=False).str.strip()
    parsed = pd.to_numeric(normalized, errors='coerce')

    malformed_mask = non_empty_mask & parsed.isna()
    malformed_examples = raw_series[malformed_mask].astype(str).unique().tolist()[:10]

    audit['non_empty_values'] = int(non_empty_mask.sum())
    audit['numeric_values'] = int((non_empty_mask & parsed.notna()).sum())
    audit['malformed_count'] = int(malformed_mask.sum())
    audit['malformed_examples'] = malformed_examples

    if not pd.api.types.is_numeric_dtype(raw_series.dtype):
        logger.warning(
            "Issue detected: Qm column is object/non-numeric "
            f"(dtype={audit['dtype_before']})."
        )
        df[qm_col] = parsed
        audit['fix_applied'] = True
        logger.info(
            "Fix applied: coerced Qm column to numeric with errors='coerce'. "
            f"Numeric parsed values: {audit['numeric_values']} / {audit['non_empty_values']}."
        )
    else:
        logger.info(f"No fix needed: Qm column already numeric (dtype={audit['dtype_before']}).")

    if audit['malformed_count'] > 0:
        logger.warning(
            "Issue detected: Non-numeric Qm values remain after parsing. "
            f"Count={audit['malformed_count']}, examples={audit['malformed_examples']}"
        )
    else:
        logger.info("No malformed Qm values detected among non-empty entries.")

    audit['dtype_after'] = str(df[qm_col].dtype)

    # Keep inspection metadata aligned with the fixed dataframe.
    if audit['fix_applied']:
        inspection['dtypes'][qm_col] = df[qm_col].dtype
        inspection['numeric_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
        inspection['categorical_columns'] = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

    return audit


def check_qm_unit_consistency(df: pd.DataFrame,
                              variable_mapping: dict,
                              logger) -> dict:
    """
    Check whether Qm unit labeling appears consistent from available headers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    variable_mapping : dict
        Variable mapping results
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Unit consistency summary
    """
    logger.info("\nChecking Qm unit consistency...")

    unit_tokens = [
        'mg/g',
        'ug/g',
        '\\u00b5g/g',
        '\\u03bcg/g',
        '\\u00c2\\u00b5g/g',
        '\\u00ce\\u00bcg/g',
    ]

    def extract_unit(label: str) -> str | None:
        normalized = label.lower()
        normalized = normalized.replace('\u00c2\u00b5', 'u')
        normalized = normalized.replace('\u00ce\u00bc', 'u')
        normalized = normalized.replace('\u00b5', 'u')
        normalized = normalized.replace('\u03bc', 'u')
        match = re.search(r'(mg/g|ug/g)', normalized)
        return match.group(1) if match else None

    qm_col = variable_mapping.get('Qm')
    qm_columns = [col for col in df.columns if 'qm' in col.lower()]
    units_found = sorted({extract_unit(col) for col in qm_columns if extract_unit(col) is not None})

    unit_report = {
        'qm_column': qm_col,
        'qm_related_columns': qm_columns,
        'units_found_in_headers': units_found,
        'unit_consistency_flag': 'unknown',
    }

    if len(units_found) == 0:
        logger.warning(
            "Issue detected: Qm unit label is unclear in column headers; no explicit mg/g or ug/g token found."
        )
        unit_report['unit_consistency_flag'] = 'unclear'
    elif len(units_found) == 1:
        logger.info(f"No issue detected: Qm unit label appears consistent ({units_found[0]}).")
        unit_report['unit_consistency_flag'] = 'consistent'
    else:
        logger.warning(
            "Issue detected: Multiple Qm units found in headers "
            f"({units_found}); unit harmonization may be needed in downstream processing."
        )
        unit_report['unit_consistency_flag'] = 'mixed'

    # Keep a reference of recognized unit tokens for reproducibility/debugging.
    unit_report['recognized_tokens'] = unit_tokens
    return unit_report


def generate_data_quality_report(df: pd.DataFrame, 
                                 inspection: dict,
                                 variable_mapping: dict,
                                 qm_unit_report: dict,
                                 logger) -> dict:
    """
    Generate comprehensive data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    inspection : dict
        Data structure inspection results
    variable_mapping : dict
        Variable mapping results
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Data quality summary
    """
    logger.info("\nGenerating data quality report...")
    
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'complete_rows': len(df.dropna()),
        'completeness_rate': len(df.dropna()) / len(df) * 100,
        'variables_found': sum(1 for v in variable_mapping.values() if v is not None),
        'variables_missing': sum(1 for v in variable_mapping.values() if v is None),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("DATA QUALITY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total observations: {quality_report['total_rows']}")
    logger.info(
        "Complete cases (strict, all columns): "
        f"{quality_report['complete_rows']} ({quality_report['completeness_rate']:.1f}%)"
    )
    logger.info(f"Variables identified: {quality_report['variables_found']} / "
                f"{len(variable_mapping)}")

    # For literature meta-analysis tables, strict all-column completeness can be misleading.
    core_required_columns = [
        variable_mapping.get('Qm'),
        variable_mapping.get('Metal'),
        variable_mapping.get('Polymer'),
    ]
    core_required_columns = [col for col in core_required_columns if col is not None and col in df.columns]

    if core_required_columns:
        core_complete_rows = len(df.dropna(subset=core_required_columns))
        core_completeness_rate = core_complete_rows / len(df) * 100
        quality_report['core_required_columns'] = core_required_columns
        quality_report['core_complete_rows'] = core_complete_rows
        quality_report['core_completeness_rate'] = core_completeness_rate

        if quality_report['complete_rows'] == 0 and core_complete_rows > 0:
            logger.warning(
                "Issue detected: strict complete-case metric is 0 but core analysis fields are populated."
            )
            logger.info(
                "Fix applied: added core completeness metric for manuscript-ready interpretation "
                f"using columns {core_required_columns}."
            )

        logger.info(
            "Core complete cases (Qm/Metal/Polymer): "
            f"{core_complete_rows} ({core_completeness_rate:.1f}%)"
        )
    else:
        logger.warning(
            "Core completeness metric not computed because one or more required columns "
            "(Qm, Metal, Polymer) were not identified."
        )

    quality_report['qm_unit_report'] = qm_unit_report
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    quality_report['duplicates'] = duplicates
    logger.info(f"Duplicate rows: {duplicates}")
    
    # Summary statistics for numeric columns if Qm identified
    if variable_mapping.get('Qm'):
        qm_col = variable_mapping['Qm']
        if qm_col in df.columns:
            qm_data = pd.to_numeric(df[qm_col], errors='coerce').dropna()
            quality_report['qm_stats'] = {
                'count': len(qm_data),
                'mean': float(qm_data.mean()),
                'std': float(qm_data.std()),
                'min': float(qm_data.min()),
                'max': float(qm_data.max()),
                'median': float(qm_data.median()),
            }
            
            logger.info(f"\nQm ({qm_col}) Statistics:")
            logger.info(f"  Count:  {quality_report['qm_stats']['count']}")
            logger.info(f"  Mean:   {quality_report['qm_stats']['mean']:.2f}")
            logger.info(f"  Std:    {quality_report['qm_stats']['std']:.2f}")
            logger.info(f"  Min:    {quality_report['qm_stats']['min']:.2f}")
            logger.info(f"  Max:    {quality_report['qm_stats']['max']:.2f}")
            logger.info(f"  Median: {quality_report['qm_stats']['median']:.2f}")
    
    logger.info(f"{'='*60}\n")
    
    return quality_report


def save_outputs(df: pd.DataFrame, 
                inspection: dict,
                variable_mapping: dict,
                quality_report: dict,
                qm_numeric_audit: dict,
                config: ProjectConfig,
                logger) -> None:
    """
    Save outputs from data loading step.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    inspection : dict
        Data structure inspection
    variable_mapping : dict
        Variable mapping
    quality_report : dict
        Data quality report
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
    """
    logger.info("Saving outputs...")
    
    # Save raw data snapshot (CSV for version control friendly format)
    output_dir = config.get_path('processed_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_snapshot_path = output_dir / "00_raw_data_snapshot.csv"
    save_dataframe(df, raw_snapshot_path, index=False)
    logger.info(f"Saved raw data snapshot: {raw_snapshot_path}")
    
    # Save inspection results
    from utils import save_json
    
    results_dir = config.get_path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all metadata
    metadata = {
        'inspection': inspection,
        'variable_mapping': variable_mapping,
        'qm_numeric_audit': qm_numeric_audit,
        'quality_report': quality_report,
    }
    
    # Convert non-JSON serializable objects
    metadata['inspection']['dtypes'] = {k: str(v) for k, v in metadata['inspection']['dtypes'].items()}
    
    metadata_path = results_dir / "01_data_loading_metadata.json"
    save_json(metadata, metadata_path)
    logger.info(f"Saved metadata: {metadata_path}")


def main():
    """Main execution function."""
    
    # Initialize configuration and logging
    try:
        config = ProjectConfig()
        logger = setup_logging(config, '01_data_loading')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 01: DATA LOADING", logger=logger)
        
        # Step 1: Load raw data
        df = load_raw_data(config, logger)
        
        # Step 2: Inspect data structure
        inspection = inspect_data_structure(df, logger)
        
        # Step 3: Identify key variables
        variable_mapping = identify_key_variables(df, logger)

        # Step 3b: Audit/fix Qm numeric values and dtype
        qm_numeric_audit = audit_and_fix_qm_numeric(df, inspection, variable_mapping, logger)

        # Step 3c: Check unit consistency labels for Qm
        qm_unit_report = check_qm_unit_consistency(df, variable_mapping, logger)
        
        # Step 4: Generate quality report
        quality_report = generate_data_quality_report(
            df, inspection, variable_mapping, qm_unit_report, logger
        )
        
        # Step 5: Save outputs
        save_outputs(
            df, inspection, variable_mapping, quality_report, qm_numeric_audit, config, logger
        )
        
        logger.info("\n" + "="*60)
        logger.info("DATA LOADING COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in data loading: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
