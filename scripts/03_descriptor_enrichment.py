"""
Script 03: Descriptor Enrichment
=================================

This script enriches the dataset with metal physicochemical properties
from the configuration file, enabling mechanistic interpretation.

Tasks:
- Add hydration energy
- Add ionic radius
- Add valence
- Add electronegativity
- Add hardness classification (HSAB theory)
- Validate enrichment

Author: Manuscript authors
Date: March 2026
"""

import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import (ProjectConfig, setup_logging, set_random_seed, 
                   load_dataframe, save_dataframe, save_json, print_section_header)


def identify_metal_column(df: pd.DataFrame, logger) -> str:
    """
    Identify the metal column in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    str
        Metal column name
    """
    metal_keywords = ['metal', 'element', 'heavy_metal']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in metal_keywords):
            logger.info(f"Identified metal column: {col}")
            return col
    
    logger.error("No metal column found!")
    raise ValueError("Metal column not found in dataset")


def standardize_metal_names(df: pd.DataFrame, metal_col: str, logger) -> pd.DataFrame:
    """
    Standardize metal names to match configuration using robust regex cleaning.
    
    Preserves oxidation states in Metal_raw column for later species extraction.
    Metal column gets base element symbol only (for config.metal_descriptors lookup).
    
    Handles variations like:
    - "Cu2+", "Cu(ii)", "Cu (ii)", "Copper"
    - "Pb(ii)", "Pb (ii)", "Pb2+"
    - "Cr(vi)", "Cr (vi)", "Cr(VI)"
    - "As(iii)", "As (iii)"
    - "Zn2+", "Zn(ii)"
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    metal_col : str
        Name of metal column
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with standardized metal names + Metal_raw column
    """
    logger.info("Standardizing metal names while preserving oxidation states...")
    
    df = df.copy()
    
    # Common element names to symbol mappings
    name_to_symbol = {
        'copper': 'Cu',
        'lead': 'Pb',
        'cadmium': 'Cd',
        'zinc': 'Zn',
        'chromium': 'Cr',
        'nickel': 'Ni',
        'arsenic': 'As',
        'mercury': 'Hg',
        'iron': 'Fe',
        'tin': 'Sn',
        'silver': 'Ag',
        'cobalt': 'Co',
        'manganese': 'Mn',
        'aluminum': 'Al',
    }
    
    # IMPORTANT: Save raw metal value BEFORE any cleaning
    df['Metal_raw'] = df[metal_col].copy()
    
    def extract_element_symbol(name):
        """Extract element symbol only (base metal, no oxidation state)."""
        if pd.isna(name):
            return np.nan
        
        name = str(name).strip()
        
        # Step 1: Try to match element symbol at the beginning (1-2 letters, first uppercase)
        match = re.match(r'^([A-Z][a-z]?)', name)
        if match:
            return match.group(1)
        
        # Step 2: Try name_to_symbol lookup
        name_lower = name.lower()
        for full_name, symbol in name_to_symbol.items():
            if name_lower.startswith(full_name):
                return symbol
        
        # Step 3: Fallback - return first 1-2 chars capitalized
        if len(name) > 0:
            result = name[0].upper()
            if len(name) > 1 and name[1].islower():
                result += name[1]
            return result
        
        return np.nan
    
    # Apply extraction to get base element
    df[metal_col] = df[metal_col].apply(extract_element_symbol)
    
    # Log standardization results before enrichment
    metal_counts = df[metal_col].value_counts().sort_values(ascending=False)
    logger.info("Standardized metals (frequency table):")
    logger.info("{:<15} {:<10} {}".format("Metal", "Count", "Percentage"))
    logger.info("-" * 40)
    for metal, count in metal_counts.items():
        pct = count / len(df) * 100
        logger.info("{:<15} {:<10} {:.1f}%".format(str(metal), count, pct))
    
    n_unique = df[metal_col].nunique()
    logger.info("Total unique metals (base element): {}".format(n_unique))
    
    return df


def extract_metal_species(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Extract metal species (with oxidation states) from Metal_raw column.
    
    Creates Metal_Species column with values like Cu_II, As_III, Cr_VI, etc.
    If oxidation state not available, sets to Unknown.
    
    Supports formats:
    - Cr(VI), Cr (VI)
    - CrVI, Cr VI
    - Cr6+, Cr +6
    - As(III), As III
    - As3+
    """
    logger.info("Extracting metal species (with oxidation states from Metal_raw)...")
    
    df = df.copy()
    
    def extract_species_from_raw(metal_raw):
        """Extract species from raw metal string."""
        if pd.isna(metal_raw):
            return np.nan
        
        raw_str = str(metal_raw).strip()
        
        # Step 1: Extract base metal symbol (first 1-2 letters, first uppercase)
        metal_match = re.match(r'^([A-Z][a-z]?)', raw_str)
        if not metal_match:
            return "{}_Unknown".format(raw_str)
        
        metal_symbol = metal_match.group(1)
        
        # Step 2: Try to extract oxidation state
        
        # Format 1: Roman numerals in parentheses: Cr(VI), Cr (VI)
        roman_match = re.search(r'\s*\(([IVX]+)\)', raw_str)
        if roman_match:
            roman = roman_match.group(1).upper()
            return "{}_{}" .format(metal_symbol, roman)
        
        # Format 2: Roman numerals without parens: CrVI, Cr VI
        roman_match2 = re.search(r'\s+([IVX]{1,3})(?:\s|$)', raw_str)
        if roman_match2:
            roman = roman_match2.group(1).upper()
            return "{}_{}" .format(metal_symbol, roman)
        
        # Format 3: +n format: Cr6+, Cr +6
        charge_match = re.search(r'\s*\+\s*(\d+)', raw_str)
        if charge_match:
            charge = int(charge_match.group(1))
            charge_to_roman = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7:'VII', 8:'VIII'}
            roman = charge_to_roman.get(charge, str(charge))
            return "{}_{}" .format(metal_symbol, roman)
        
        # Format 4: Simple number: Cr6, Cr 6
        num_match = re.search(r'[Cr]?\s*(\d+)$', raw_str)
        if num_match:
            charge = int(num_match.group(1))
            charge_to_roman = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7:'VII', 8:'VIII'}
            roman = charge_to_roman.get(charge, str(charge))
            return "{}_{}" .format(metal_symbol, roman)
        
        # No oxidation state found
        return "{}_Unknown".format(metal_symbol)
    
    df['Metal_Species'] = df['Metal_raw'].apply(extract_species_from_raw)
    
    species_counts = df['Metal_Species'].value_counts()
    logger.info("Metal species extracted ({} unique):".format(len(species_counts)))
    for species, count in species_counts.items():
        logger.info("  {}: {:4d} rows".format(species, count))
    
    return df


def apply_species_rules(df: pd.DataFrame, config: ProjectConfig, logger) -> pd.DataFrame:
    """
    Apply species-specific descriptor enrichment rules.
    
    For species in config.species_handling.require_speciation:
    - Set all descriptors to NA
    - Set species_requires_speciation = 1
    - Set metal_descriptor_missing_speciation = 1
    
    Creates indicator columns:
    - metal_descriptor_missing_unmatched: 1 if metal not in config (set elsewhere)
    - metal_descriptor_missing_speciation: 1 if species requires special treatment
    """
    logger.info("Applying species-specific enrichment rules...")
    
    df = df.copy()
    
    # Initialize indicator columns
    if 'metal_descriptor_missing_unmatched' not in df.columns:
        df['metal_descriptor_missing_unmatched'] = 0
    if 'metal_descriptor_missing_speciation' not in df.columns:
        df['metal_descriptor_missing_speciation'] = 0
    if 'species_requires_speciation' not in df.columns:
        df['species_requires_speciation'] = 0
    
    # Get list of species that require speciation from config
    species_requiring_speciation = config.get(
        'data_processing.species_handling.require_speciation', 
        []
    )
    
    descriptor_cols = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 'Electronegativity', 'Hardness']
    
    # Apply rules for each species requiring speciation
    n_speciation_total = 0
    for species in species_requiring_speciation:
        mask = df['Metal_Species'] == species
        n_species = mask.sum()
        
        if n_species > 0:
            # Set all descriptors to NA for this species
            for col in descriptor_cols:
                if col in df.columns:
                    df.loc[mask, col] = np.nan
            
            # Set speciation flags and indicator
            df.loc[mask, 'species_requires_speciation'] = 1
            df.loc[mask, 'metal_descriptor_missing_speciation'] = 1
            
            logger.info("  {}: {} rows - descriptors set to NA (speciation required)".format(species, n_species))
            n_speciation_total += n_species
    
    logger.info("Species rule summary:")
    logger.info("  Rows flagged for speciation: {}".format(n_speciation_total))
    
    return df


def enrich_with_descriptors(df: pd.DataFrame, 
                           metal_col: str,
                           config: ProjectConfig, 
                           logger) -> tuple:
    """
    Enrich dataset with metal physicochemical descriptors using vectorized merge.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with standardized metal names
    metal_col : str
        Name of metal column
    config : ProjectConfig
        Project configuration
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (enriched_df, unmatched_metals_set) where unmatched_metals_set is for downstream handling
        
    Raises
    ------
    ValueError
        If enrichment coverage falls below configured threshold (after policy applied)
    """
    logger.info("Enriching with metal descriptors (vectorized merge)...")
    
    df = df.copy()
    
    # Get metal descriptors from config
    metal_descriptors = config.get('metal_descriptors', {})
    
    if not metal_descriptors:
        logger.error("No metal descriptors found in configuration!")
        raise ValueError("Metal descriptors missing from config")
    
    logger.info(f"Loaded {len(metal_descriptors)} metal descriptors from configuration")
    
    # Analyze metal matching BEFORE merge
    unique_metals_in_data = df[metal_col].dropna().unique()
    config_metals = set(metal_descriptors.keys())
    data_metals = set(unique_metals_in_data)
    
    matched_metals = data_metals & config_metals
    unmatched_metals = data_metals - config_metals
    
    logger.info(f"\nMetal matching analysis (before policy):")
    logger.info(f"  Unique metals in data: {len(data_metals)}")
    logger.info(f"  Metals available in config: {len(config_metals)}")
    logger.info(f"  Successfully matched: {len(matched_metals)}")
    
    # Log unmatched metals with frequencies
    if unmatched_metals:
        metal_counts_in_data = df[metal_col].value_counts()
        logger.warning(f"\nUnmatched metals found (with frequencies):")
        for metal in sorted(unmatched_metals):
            count = metal_counts_in_data.get(metal, 0)
            pct = count / len(df) * 100
            logger.warning(f"  {metal}: {count} rows ({pct:.1f}%)")
        logger.warning(f"  Available metals in config: {sorted(config_metals)}")
    
    # Build descriptors DataFrame from config
    descriptors_list = []
    for metal_key, props in metal_descriptors.items():
        row = {'metal_key': metal_key}
        row.update(props)
        descriptors_list.append(row)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # Perform vectorized merge on metal column
    df = df.merge(descriptors_df, left_on=metal_col, right_on='metal_key', how='left')
    
    # Rename descriptor columns to standardized names
    column_mapping = {
        'hydration_energy': 'Hydration_Energy',
        'ionic_radius': 'Ionic_Radius',
        'valence': 'Valence',
        'electronegativity': 'Electronegativity',
        'hardness': 'Hardness'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
            df = df.drop(columns=[old_col])
    
    # Clean up temporary merge key
    if 'metal_key' in df.columns:
        df = df.drop(columns=['metal_key'])
    
    # Log descriptor statistics (numeric only) - before checking coverage
    descriptor_numeric = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 'Electronegativity']
    logger.info("\nDescriptor statistics (after merge):")
    for desc in descriptor_numeric:
        if desc in df.columns:
            data = df[desc].dropna()
            if len(data) > 0:
                logger.info(f"  {desc:25s}: Mean={data.mean():8.2f}, "
                          f"Std={data.std():8.2f}, "
                          f"Range=[{data.min():8.2f}, {data.max():8.2f}]")
    
    return df, unmatched_metals


def handle_unmatched_metals(df: pd.DataFrame,
                            metal_col: str,
                            unmatched_metals: set,
                            config: ProjectConfig,
                            logger) -> pd.DataFrame:
    """
    Handle metals that are not in config['metal_descriptors'] based on policy.
    Sets metal_descriptor_missing_unmatched indicator for tracking.
    """
    if not unmatched_metals:
        return df
    
    df = df.copy()
    
    # Get policy from config (default: "error")
    policy = config.get('data_processing.unmatched_metals_policy', 'error')
    logger.info("Applying unmatched metals policy: '{}'".format(policy))
    
    rows_before = len(df)
    
    if policy == 'error':
        error_msg = (
            "Unmatched metals found in data: {}. "
            "Set 'data_processing.unmatched_metals_policy' to 'drop_rows' or 'keep_with_na' "
            "to handle this.".format(sorted(unmatched_metals))
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    elif policy == 'drop_rows':
        mask = df[metal_col].isin(unmatched_metals)
        n_drop = mask.sum()
        df = df[~mask]
        logger.info("Dropped {} rows with unmatched metals {}".format(n_drop, sorted(unmatched_metals)))
        logger.info("Rows: {} -> {}".format(rows_before, len(df)))
    
    elif policy == 'keep_with_na':
        # Keep rows but add indicator for unmatched metals
        mask = df[metal_col].isin(unmatched_metals)
        n_unmatched = mask.sum()
        
        # Initialize indicator if needed
        if 'metal_descriptor_missing_unmatched' not in df.columns:
            df['metal_descriptor_missing_unmatched'] = 0
        
        df.loc[mask, 'metal_descriptor_missing_unmatched'] = 1
        logger.info("Kept {} rows with unmatched metals (added indicator)".format(n_unmatched))
    
    else:
        logger.warning("Unknown policy '{}'. Defaulting to 'error'.".format(policy))
        raise ValueError("Unknown unmatched_metals_policy: {}".format(policy))
    
    return df


def ensure_derived_features_finite(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Check and fix derived features to ensure no infinite values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with derived features
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with infinite values replaced by NaN
    """
    df = df.copy()
    
    derived_cols = ['Charge_Density', 'Hydration_Energy_per_Charge', 'Softness_Parameter']
    
    for col in derived_cols:
        if col in df.columns:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                logger.warning(f"Found {n_inf} infinite values in '{col}'. Replacing with NaN.")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    return df


def validate_enrichment(df: pd.DataFrame, logger, cation_enriched_mask: pd.Series = None) -> dict:
    """
    Validate descriptor enrichment.
    
    Parameters
    ----------
    df : pd.DataFrame
        Enriched data
    logger : logging.Logger
        Logger instance
    cation_enriched_mask : pd.Series, optional
        Boolean mask for cation-enriched rows (rows with ALL descriptors).
        If provided, correlation matrix will be computed only on this subset.
        
    Returns
    -------
    dict
        Validation report
    """
    logger.info("\nValidating enrichment...")
    
    descriptor_cols = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 
                      'Electronegativity', 'Hardness']
    
    validation = {
        'total_rows': len(df),
        'descriptor_completeness': {},
        'descriptor_coverage': {},
        'modeling_cautions': []
    }
    
    for col in descriptor_cols:
        if col in df.columns:
            n_present = df[col].notna().sum()
            completeness = n_present / len(df) * 100
            validation['descriptor_completeness'][col] = completeness
            validation['descriptor_coverage'][col] = n_present
            
            logger.info("  {0:25s}: {1:4d}/{2} ({3:.1f}%)".format(
                col, n_present, len(df), completeness))
    
    # Check correlations between descriptors
    # Only compute on cation-enriched subset to avoid bias from NA rows
    numeric_descriptors = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 'Electronegativity']
    
    # Determine subset for correlation calculation
    if cation_enriched_mask is not None and cation_enriched_mask.sum() > 0:
        df_corr = df[cation_enriched_mask].copy()
        logger.info("\nDescriptor correlations (computed on {} cation-enriched rows):".format(
            cation_enriched_mask.sum()))
    else:
        df_corr = df.copy()
        logger.info("\nDescriptor correlations:")

    # Explicit usability check for Valence in the enriched subset.
    valence_summary = {
        'subset_rows': int(len(df_corr)),
        'n_valid': 0,
        'unique_values': 0,
        'std': None,
        'mean': None,
        'cv': None,
        'near_zero_variance': None,
        'non_informative_for_modeling': None,
    }
    valence_near_zero = False

    if 'Valence' in df_corr.columns:
        valence_data = df_corr['Valence'].dropna()
        n_valid = int(len(valence_data))
        n_unique = int(valence_data.nunique()) if n_valid > 0 else 0
        val_std = float(valence_data.std()) if n_valid > 1 else 0.0
        val_mean = float(valence_data.mean()) if n_valid > 0 else None
        val_cv = None
        if val_mean is not None and val_mean != 0:
            val_cv = float(abs(val_std / val_mean))

        # Near-zero variance rule for downstream modeling usability.
        valence_near_zero = (
            (n_valid > 0 and n_unique <= 1) or
            (val_std <= 1e-8) or
            (val_cv is not None and val_cv <= 0.01)
        )

        valence_summary.update({
            'n_valid': n_valid,
            'unique_values': n_unique,
            'std': val_std,
            'mean': val_mean,
            'cv': val_cv,
            'near_zero_variance': bool(valence_near_zero),
            'non_informative_for_modeling': bool(valence_near_zero),
        })

        if valence_near_zero:
            caution_msg = (
                "Valence shows near-zero variance in enriched subset; "
                "treat Valence as non-informative for downstream modeling."
            )
            logger.warning(caution_msg)
            validation['modeling_cautions'].append({
                'type': 'non_informative_predictor',
                'predictor': 'Valence',
                'message': caution_msg,
            })

    validation['valence_usability'] = valence_summary

    # If Valence is near-constant, derived features using Valence may add limited new information.
    derived_usability = {
        'valence_near_zero_variance': bool(valence_near_zero),
        'affected_features': [],
        'warning': None,
    }
    derived_cols = ['Charge_Density', 'Hydration_Energy_per_Charge', 'Softness_Parameter']
    available_derived = [c for c in derived_cols if c in df_corr.columns]

    if valence_near_zero and available_derived:
        derived_usability['affected_features'] = available_derived
        derived_warning = (
            "Derived features may add limited new information because Valence is "
            "constant/nearly constant in enriched rows."
        )
        derived_usability['warning'] = derived_warning
        logger.warning(derived_warning)
        logger.warning(
            "Modeling caution: review {} for redundancy before treating them as independent predictors.".format(
                available_derived
            )
        )
        validation['modeling_cautions'].append({
            'type': 'derived_feature_redundancy',
            'predictors': available_derived,
            'message': derived_warning,
        })

    validation['derived_feature_usability'] = derived_usability
    
    # Filter descriptors: remove columns with std==0 or n_valid<10
    valid_descriptors = []
    removed_descriptors = []
    
    for col in numeric_descriptors:
        if col not in df_corr.columns:
            removed_descriptors.append((col, 'column not found'))
            continue
        
        col_data = df_corr[col].dropna()
        n_valid = len(col_data)
        
        if n_valid < 10:
            removed_descriptors.append((col, 'n_valid={} < 10'.format(n_valid)))
            continue
        
        col_std = col_data.std()
        if col_std == 0 or np.isnan(col_std):
            removed_descriptors.append((col, 'std={:.4f} (no variance)'.format(
                col_std if not np.isnan(col_std) else 0)))
            continue
        
        valid_descriptors.append(col)
    
    # Log removed descriptors
    if removed_descriptors:
        logger.info("  Descriptors excluded from correlation (insufficient data):")
        for col, reason in removed_descriptors:
            logger.info("    - {}: {}".format(col, reason))
    
    # Compute correlation matrix on valid descriptors
    if len(valid_descriptors) >= 2:
        corr_matrix = df_corr[valid_descriptors].corr()
        logger.info("\n" + corr_matrix.to_string())
        validation['correlations'] = corr_matrix.to_dict()

        # Explicit caution for very high Hydration_Energy-Ionic_Radius collinearity.
        if 'Hydration_Energy' in corr_matrix.index and 'Ionic_Radius' in corr_matrix.columns:
            he_ir_corr = float(corr_matrix.loc['Hydration_Energy', 'Ionic_Radius'])
            if abs(he_ir_corr) > 0.95:
                caution_msg = (
                    "Hydration_Energy and Ionic_Radius are highly collinear (|r|={:.3f}). "
                    "Do not treat them as independent predictors in downstream models."
                ).format(abs(he_ir_corr))
                logger.warning(caution_msg)
                validation['modeling_cautions'].append({
                    'type': 'high_collinearity',
                    'predictor_pair': ['Hydration_Energy', 'Ionic_Radius'],
                    'correlation_abs': float(abs(he_ir_corr)),
                    'message': caution_msg,
                })
        
        # Flag potential multicollinearity
        high_corr = []
        for i in range(len(valid_descriptors)):
            for j in range(i+1, len(valid_descriptors)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr.append({
                        'var1': valid_descriptors[i],
                        'var2': valid_descriptors[j],
                        'correlation': float(corr)
                    })
                    logger.warning("  High correlation: {} vs {} = {:.2f}".format(
                        valid_descriptors[i], valid_descriptors[j], corr))
        
        validation['high_correlations'] = high_corr
    else:
        logger.warning("  Cannot compute correlation: fewer than 2 valid descriptors")
        validation['correlations'] = {}
        validation['high_correlations'] = []
    
    return validation


def create_derived_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Create derived features from metal descriptors.
    Safely handles NaN and infinite values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Data with derived features
    """
    logger.info("\nCreating derived features...")
    
    df = df.copy()
    features_created = []
    
    # Charge density (valence / ionic radius)
    if 'Valence' in df.columns and 'Ionic_Radius' in df.columns:
        # Only compute where both values are present
        mask = df['Valence'].notna() & df['Ionic_Radius'].notna()
        df['Charge_Density'] = np.nan
        df.loc[mask, 'Charge_Density'] = df.loc[mask, 'Valence'] / df.loc[mask, 'Ionic_Radius']
        features_created.append('Charge_Density')
        logger.info(f"  Created: Charge_Density = Valence / Ionic_Radius (n={mask.sum()} non-NaN)")
    
    # Hydration energy per charge
    if 'Hydration_Energy' in df.columns and 'Valence' in df.columns:
        mask = df['Hydration_Energy'].notna() & df['Valence'].notna() & (df['Valence'] != 0)
        df['Hydration_Energy_per_Charge'] = np.nan
        df.loc[mask, 'Hydration_Energy_per_Charge'] = df.loc[mask, 'Hydration_Energy'] / df.loc[mask, 'Valence']
        features_created.append('Hydration_Energy_per_Charge')
        logger.info(f"  Created: Hydration_Energy_per_Charge (n={mask.sum()} non-NaN)")
    
    # Softness parameter (inverse of electronegativity * valence)
    if 'Electronegativity' in df.columns and 'Valence' in df.columns:
        mask = df['Electronegativity'].notna() & df['Valence'].notna() & (df['Electronegativity'] * df['Valence'] != 0)
        df['Softness_Parameter'] = np.nan
        df.loc[mask, 'Softness_Parameter'] = 1.0 / (df.loc[mask, 'Electronegativity'] * df.loc[mask, 'Valence'])
        features_created.append('Softness_Parameter')
        logger.info(f"  Created: Softness_Parameter (n={mask.sum()} non-NaN)")
    
    logger.info(f"Created {len(features_created)} derived features")
    return df


def main():
    """Main execution function."""
    
    try:
        # Initialize
        config = ProjectConfig()
        logger = setup_logging(config, '03_descriptor_enrichment')
        set_random_seed(config=config)
        
        print_section_header("SCRIPT 03: DESCRIPTOR ENRICHMENT", logger=logger)
        
        # Load cleaned data
        logger.info("Loading cleaned data...")
        data_path = config.get_path('processed_data') / "02_cleaned_data.csv"
        df = load_dataframe(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        rows_initial = len(df)
        
        # Step 1: Identify metal column
        metal_col = identify_metal_column(df, logger)
        
        # Step 2: Standardize metal names (creates Metal_raw for species extraction)
        df = standardize_metal_names(df, metal_col, logger)
        
        # Step 2b: Extract metal species from Metal_raw (NOT from clean Metal column)
        df = extract_metal_species(df, logger)
        
        # Step 3: Enrich with descriptors and get unmatched metals
        df, unmatched_metals = enrich_with_descriptors(df, metal_col, config, logger)
        
        # Step 3b: Handle unmatched metals based on policy
        if unmatched_metals:
            df = handle_unmatched_metals(df, metal_col, unmatched_metals, config, logger)
            rows_after_policy = len(df)
            logger.info("Rows after applying unmatched metals policy: {} -> {}".format(rows_initial, rows_after_policy))
        
        # Step 3c: Apply species-specific enrichment rules (NA for Cr_VI, As_III, etc.)
        df = apply_species_rules(df, config, logger)
        
        # Step 3d: Create combined indicator for any missing descriptor
        descriptor_cols_check = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 'Electronegativity', 'Hardness']
        has_missing_descriptors = df[[c for c in descriptor_cols_check if c in df.columns]].isna().any(axis=1)
        
        if 'metal_descriptor_missing_any' not in df.columns:
            df['metal_descriptor_missing_any'] = 0
        
        # metal_descriptor_missing_any = 1 if unmatched OR speciation OR has NA
        df['metal_descriptor_missing_any'] = (
            (df['metal_descriptor_missing_unmatched'] == 1) |
            (df['metal_descriptor_missing_speciation'] == 1) |
            (has_missing_descriptors)
        ).astype(int)
        
        n_descriptor_missing_total = (df['metal_descriptor_missing_any'] == 1).sum()
        logger.info("Descriptor missing summary:")
        logger.info("  Unmatched metals: {}".format((df['metal_descriptor_missing_unmatched'] == 1).sum()))
        logger.info("  Speciation-flagged: {}".format((df['metal_descriptor_missing_speciation'] == 1).sum()))
        logger.info("  Any missing descriptor: {}".format(n_descriptor_missing_total))
        
        # Step 4: Create derived features with safety checks
        df = create_derived_features(df, logger)
        
        # Step 4b: Ensure no infinite values in derived features
        df = ensure_derived_features_finite(df, logger)
        
        # Step 5: Calculate and report final enrichment coverage
        descriptor_all = ['Hydration_Energy', 'Ionic_Radius', 'Valence', 'Electronegativity', 'Hardness']
        n_total = len(df)
        
        # Three distinct groups:
        # 1. Cation-enriched: has ALL descriptors (intentionally enriched)
        cation_enriched_mask = df[descriptor_all].notna().all(axis=1)
        n_cation_enriched = cation_enriched_mask.sum()
        pct_cation_enriched = n_cation_enriched / n_total * 100 if n_total > 0 else 0
        
        # 2. Speciation-flagged: species_requires_speciation==1 (descriptors intentionally NA)
        speciation_flagged_mask = (df['species_requires_speciation'] == 1)
        n_speciation_flagged = speciation_flagged_mask.sum()
        pct_speciation_flagged = n_speciation_flagged / n_total * 100 if n_total > 0 else 0
        
        # 3. Unmatched: metal_descriptor_missing_unmatched==1 (metal not in config)
        unmatched_mask = (df['metal_descriptor_missing_unmatched'] == 1)
        n_unmatched = unmatched_mask.sum()
        pct_unmatched = n_unmatched / n_total * 100 if n_total > 0 else 0
        
        # Rows with partial/missing descriptors (excluding intentional speciation flags)
        # = rows that are NOT cation_enriched AND NOT speciation_flagged
        partial_missing_mask = (~cation_enriched_mask) & (~speciation_flagged_mask)
        n_partial_missing = partial_missing_mask.sum()
        pct_partial_missing = n_partial_missing / n_total * 100 if n_total > 0 else 0
        
        # Final coverage = cation-enriched rows (rows we successfully enriched)
        final_coverage = pct_cation_enriched
        
        logger.info("Final enrichment coverage:")
        logger.info("  Total rows after processing: {}".format(n_total))
        logger.info("")
        logger.info("  [1] Cation-enriched rows (ALL descriptors present): {} ({:.1f}%)".format(
            n_cation_enriched, pct_cation_enriched))
        logger.info("  [2] Speciation-flagged rows (descriptors intentionally NA): {} ({:.1f}%)".format(
            n_speciation_flagged, pct_speciation_flagged))
        logger.info("  [3] Unmatched metal rows (metal not in config): {} ({:.1f}%)".format(
            n_unmatched, pct_unmatched))
        logger.info("  [4] Partial/missing descriptors (other): {} ({:.1f}%)".format(
            n_partial_missing, pct_partial_missing))
        logger.info("")
        logger.info("  Total enrichment coverage (group 1): {:.1f}%".format(final_coverage))
        
        # Check coverage threshold
        min_coverage_threshold = config.get('data_processing.enrichment_min_coverage', 50)
        if final_coverage < min_coverage_threshold:
            error_msg = (
                "Final enrichment coverage ({:.1f}%) is below minimum threshold "
                "({})%. Consider adjusting unmatched_metals_policy "
                "or metal_descriptors in config.".format(final_coverage, min_coverage_threshold)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 6: Validate enrichment
        validation = validate_enrichment(df, logger, cation_enriched_mask)
        validation['final_coverage_percent'] = final_coverage
        validation['rows_processed'] = len(df)
        validation['n_cation_enriched'] = n_cation_enriched
        validation['n_speciation_flagged'] = n_speciation_flagged
        validation['n_unmatched'] = n_unmatched
        validation['n_partial_missing'] = n_partial_missing
        
        # Save outputs
        logger.info("\nSaving enriched data...")
        output_path = config.get_path('processed_data') / "03_enriched_data.csv"
        save_dataframe(df, output_path, index=False)
        logger.info(f"Saved to: {output_path}")
        
        # Save validation report
        report_path = config.get_path('results') / "03_enrichment_validation.json"
        save_json(validation, report_path)
        logger.info(f"Saved validation to: {report_path}")
        
        logger.info("\n" + "="*60)
        logger.info("DESCRIPTOR ENRICHMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Final dataset: {len(df)} rows x {len(df.columns)} columns")
        logger.info(f"Enrichment coverage: {final_coverage:.1f}%")
        logger.info("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in descriptor enrichment: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
