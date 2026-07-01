#!/usr/bin/env python
"""Validate the cleaned dataset against the project data-handling policy."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def validate_compliance() -> int:
    """Validate cleaned dataset against data policy rules."""

    dataset_path = Path("data_processed/02_cleaned_data.csv")

    if not dataset_path.exists():
        print(f"{FAIL} Dataset not found at {dataset_path}")
        return 1

    df = pd.read_csv(dataset_path)

    print("=" * 70)
    print("DATA POLICY COMPLIANCE VALIDATION REPORT")
    print("=" * 70)

    all_pass = True

    print("\n1. pH HANDLING COMPLIANCE:")
    if "pH" in df.columns:
        print(f"   {PASS} pH column exists")
    else:
        print(f"   {FAIL} pH column missing")
        all_pass = False

    if "pH_missing" in df.columns:
        ph_missing_count = int(df["pH_missing"].sum())
        ph_na_count = int(df["pH"].isna().sum())
        print(f"   {PASS} pH_missing indicator exists: {ph_missing_count} rows")

        if ph_missing_count == ph_na_count:
            print(f"   {PASS} pH_missing indicator matches NA count ({ph_na_count})")
        else:
            print(
                f"   {FAIL} Mismatch: pH_missing ({ph_missing_count}) vs pH NAs ({ph_na_count})"
            )
            all_pass = False
    else:
        print(f"   {FAIL} pH_missing indicator missing")
        all_pass = False

    print("\n2. LIMITED IMPUTATION COMPLIANCE (Temp, AIC, rpm):")
    for flag in ["imputed_Temp", "imputed_AIC", "imputed_rpm"]:
        if flag in df.columns:
            count = int(df[flag].sum())
            print(f"   {PASS} {flag}: {count} rows marked as imputed")
        else:
            print(f"   {FAIL} {flag} column missing")
            all_pass = False

    print("\n3. CONTEXTUAL VARIABLES (NO IMPUTATION):")
    for var in ["SA", "AgS"]:
        if var in df.columns:
            na_count = int(df[var].isna().sum())
            print(f"   {PASS} {var}: {na_count} NA values preserved")
        else:
            print(f"   {INFO} {var}: column not present (optional)")

    print("\n4. QM UNIT COMPLIANCE (ug/g):")
    if "Qm" in df.columns:
        print(f"   {PASS} Qm dtype: {df['Qm'].dtype}")

        if pd.api.types.is_numeric_dtype(df["Qm"]):
            print(f"   {PASS} Qm is numeric")
        else:
            print(f"   {FAIL} Qm is not numeric ({df['Qm'].dtype})")
            all_pass = False

        qm_text_vals = int(df["Qm"].astype(str).str.contains(r"[a-zA-Z]", na=False).sum())
        if qm_text_vals == 0:
            print(f"   {PASS} No text values in Qm column")
        else:
            print(f"   {FAIL} {qm_text_vals} Qm values contain text")
            all_pass = False

        if "log_Qm" in df.columns:
            log_qm_check = np.allclose(
                df["log_Qm"], np.log(df["Qm"]), rtol=1e-5, equal_nan=True
            )
            if log_qm_check:
                print(f"   {PASS} log_Qm = ln(Qm)")
            else:
                print(f"   {FAIL} log_Qm does not equal ln(Qm)")
                all_pass = False
        else:
            print(f"   {FAIL} log_Qm column missing")
            all_pass = False
    else:
        print(f"   {FAIL} Qm column missing")
        all_pass = False

    print("\n5. QUALITY SCORING COMPLIANCE:")
    if "quality_score" in df.columns and "quality_flags" in df.columns:
        print(f"   {PASS} quality_score column exists")
        print(f"   {PASS} quality_flags column exists")

        if "pH_missing" in df["quality_flags"].values:
            print(f"   {PASS} pH_missing tag present in quality_flags")

        if any("imputed_" in str(flag) for flag in df["quality_flags"].unique()):
            print(f"   {PASS} Imputation tags present in quality_flags")

        qual_dist = df["quality_category"].value_counts().to_dict()
        print(f"   {PASS} Quality distribution: {qual_dist}")
    else:
        print(f"   {FAIL} quality_score or quality_flags missing")
        all_pass = False

    print("\n" + "=" * 70)
    print("COMPLIANCE CHECK SUMMARY")
    print("=" * 70)

    if all_pass:
        print(f"{PASS} ALL POLICY RULES PASSED")
        print(f"{PASS} Dataset is ready for downstream analysis")
        return 0

    print(f"{FAIL} SOME POLICY RULES FAILED")
    print(f"{FAIL} Review violations above and apply fixes")
    return 1


if __name__ == "__main__":
    sys.exit(validate_compliance())
