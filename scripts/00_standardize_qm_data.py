"""
Standardize the raw Qm extraction workbook and attach source metadata.

This script uses the curated 50-source reference table, optionally cross-checked
against the local Zotero metadata export, to add DOI and bibliographic fields to
each raw Qm observation.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_raw" / "Qm data_draft.xlsx"
FALLBACK_INPUT = ROOT / "data_raw" / "Qm data.xlsx"
DEFAULT_OUTPUT_XLSX = ROOT / "data_raw" / "Qm data_standardized.xlsx"
DEFAULT_OUTPUT_CSV = ROOT / "data_raw" / "Qm_data_standardized.csv"
RAW_DATA_PATH = ROOT / "data_raw" / "Qm data.xlsx"
REFS_PATH = ROOT / "manuscript" / "references" / "QM_50_SOURCES.csv"
ZOTERO_EXPORT = ROOT / "tmp" / "zotero_qm_extracted_sources_metadata.csv"
AUDIT_PATH = ROOT / "results" / "data_qc" / "qm_data_standardization_audit.csv"
LEGACY_NOTE_AUDIT_PATH = ROOT / "results" / "data_qc" / "qm_data_legacy_unnamed14_notes.csv"
EXCLUDED_METADATA_COLUMNS = ["Aged condition", "Solution- added agent", "Note"]


def normalize_label(value: object) -> str:
    """Normalize source labels without changing their intended identity."""
    if pd.isna(value):
        return ""
    label = str(value).replace("\xa0", " ").strip()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s*\(\s*", "(", label)
    label = re.sub(r"\s*\)\s*", ")", label)
    label = re.sub(r"\s*-\s*", "-", label)
    label = re.sub(r"\s*_\s*", "_", label)
    return label


def normalize_doi(value: object) -> str:
    """Normalize DOI strings for set comparisons."""
    if pd.isna(value):
        return ""
    doi = str(value).strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.lower()


def parse_numeric_series(series: pd.Series, column_name: str) -> pd.Series:
    """Parse numeric values after removing common spreadsheet text artifacts."""
    cleaned = (
        series.astype("string")
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    bad_mask = cleaned.notna() & cleaned.ne("") & parsed.isna()
    if bad_mask.any():
        examples = cleaned.loc[bad_mask].head(10).tolist()
        raise ValueError(f"Could not parse numeric values in {column_name}: {examples}")
    return parsed


def select_input_path() -> Path:
    """Prefer the fuller draft workbook when available."""
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT
    return FALLBACK_INPUT


def load_reference_table(path: Path) -> pd.DataFrame:
    refs = pd.read_csv(path)
    required = {
        "study_label_in_SI",
        "ris_index",
        "year",
        "first_author",
        "title",
        "journal",
        "doi",
        "all_authors",
    }
    missing = required - set(refs.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")

    refs = refs.copy()
    refs["label_norm"] = refs["study_label_in_SI"].map(normalize_label)
    refs["doi_norm"] = refs["doi"].map(normalize_doi)
    refs["Source_ID"] = [f"QMS{i:03d}" for i in range(1, len(refs) + 1)]

    if refs["label_norm"].duplicated().any():
        dupes = refs.loc[refs["label_norm"].duplicated(keep=False), "study_label_in_SI"]
        raise ValueError(f"Duplicate source labels in reference table: {dupes.tolist()}")
    if refs["doi_norm"].eq("").any():
        missing_labels = refs.loc[refs["doi_norm"].eq(""), "study_label_in_SI"].tolist()
        raise ValueError(f"Missing DOI in reference table: {missing_labels}")
    if refs["doi_norm"].duplicated().any():
        dupes = refs.loc[refs["doi_norm"].duplicated(keep=False), "doi"]
        raise ValueError(f"Duplicate DOI in reference table: {dupes.tolist()}")

    return refs


def check_zotero_export(refs: pd.DataFrame, path: Path) -> dict[str, object]:
    """Compare DOI coverage against the Zotero metadata export, when present."""
    if not path.exists():
        return {"zotero_export_present": False}

    zotero = pd.read_csv(path)
    if "doi" not in zotero.columns:
        raise ValueError(f"Zotero export has no DOI column: {path}")

    zotero_dois = set(zotero["doi"].map(normalize_doi))
    zotero_dois.discard("")
    ref_dois = set(refs["doi_norm"])
    ref_dois.discard("")

    missing_from_zotero = sorted(ref_dois - zotero_dois)
    extra_in_zotero = sorted(zotero_dois - ref_dois)

    if missing_from_zotero or extra_in_zotero:
        raise ValueError(
            "Zotero DOI set does not match QM_50_SOURCES.csv. "
            f"Missing from Zotero: {missing_from_zotero}; extra in Zotero: {extra_in_zotero}"
        )

    return {
        "zotero_export_present": True,
        "zotero_items": int(len(zotero)),
        "zotero_doi_match": True,
    }


def standardize_dataframe(raw: pd.DataFrame, refs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    df = df.dropna(axis=1, how="all")
    df = df.drop(columns=[col for col in EXCLUDED_METADATA_COLUMNS if col in df.columns])

    if "Author-year" not in df.columns:
        raise ValueError("Raw Qm workbook must contain an 'Author-year' column.")

    q_cols = [col for col in df.columns if "qm" in str(col).lower()]
    if len(q_cols) != 1:
        raise ValueError(f"Expected exactly one Qm column, found: {q_cols}")
    q_col = q_cols[0]
    if q_col != "Qm (ug/g)":
        df = df.rename(columns={q_col: "Qm (ug/g)"})

    legacy_note_audit = pd.DataFrame()
    if "Unnamed: 14" in df.columns:
        legacy_note_audit = df.loc[df["Unnamed: 14"].notna(), ["Author-year", "Unnamed: 14"]].copy()
        legacy_note_audit.insert(0, "Raw_row_number", legacy_note_audit.index + 2)
        legacy_note_audit = legacy_note_audit.rename(columns={"Unnamed: 14": "legacy_unaligned_note"})
        df = df.drop(columns=["Unnamed: 14"])

    df["label_norm"] = df["Author-year"].map(normalize_label)
    ref_lookup = refs.set_index("label_norm")

    data_labels = set(df["label_norm"])
    ref_labels = set(refs["label_norm"])
    if data_labels != ref_labels:
        raise ValueError(
            "Raw workbook labels do not match the 50-source reference table. "
            f"Data-only labels: {sorted(data_labels - ref_labels)}; "
            f"Reference-only labels: {sorted(ref_labels - data_labels)}"
        )

    meta = df["label_norm"].map(ref_lookup["Source_ID"]).to_frame("Source_ID")
    meta["Author-year"] = df["label_norm"].map(ref_lookup["study_label_in_SI"])
    meta["DOI"] = df["label_norm"].map(ref_lookup["doi"])
    meta["Publication_year"] = df["label_norm"].map(ref_lookup["year"])
    meta["First_author"] = df["label_norm"].map(ref_lookup["first_author"])
    meta["Reference_title"] = df["label_norm"].map(ref_lookup["title"])
    meta["Journal"] = df["label_norm"].map(ref_lookup["journal"])
    meta["All_authors"] = df["label_norm"].map(ref_lookup["all_authors"])
    meta["RIS_index"] = df["label_norm"].map(ref_lookup["ris_index"])

    df = df.drop(columns=["Author-year", "label_norm"])
    standardized = pd.concat([meta, df], axis=1)
    standardized.insert(0, "Row_ID", [f"QMR{i:03d}" for i in range(1, len(standardized) + 1)])

    standardized["Qm (ug/g)"] = parse_numeric_series(standardized["Qm (ug/g)"], "Qm (ug/g)")

    audit = (
        standardized.groupby(["Source_ID", "Author-year", "DOI"], dropna=False)
        .agg(
            n_rows=("Row_ID", "size"),
            metals=("Metal", lambda x: "; ".join(sorted(map(str, pd.unique(x))))),
            polymers=("ReT", lambda x: "; ".join(sorted(map(str, pd.unique(x))))),
            min_qm_ug_g=("Qm (ug/g)", "min"),
            max_qm_ug_g=("Qm (ug/g)", "max"),
        )
        .reset_index()
    )

    return standardized, audit, legacy_note_audit


def write_workbook(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Qm_data_standardized")
        ws = writer.book["Qm_data_standardized"]
        ws.freeze_panes = "A2"
        for column_cells in ws.columns:
            header = str(column_cells[0].value)
            width = min(max(len(header) + 2, 12), 45)
            ws.column_dimensions[column_cells[0].column_letter].width = width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=select_input_path())
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--replace-raw", action="store_true", help="Also replace data_raw/Qm data.xlsx after backup.")
    args = parser.parse_args()

    refs = load_reference_table(REFS_PATH)
    zotero_check = check_zotero_export(refs, ZOTERO_EXPORT)
    raw = pd.read_excel(args.input)
    standardized, audit, legacy_note_audit = standardize_dataframe(raw, refs)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_workbook(standardized, args.output_xlsx)
    standardized.to_csv(args.output_csv, index=False, encoding="utf-8")
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8")

    if not legacy_note_audit.empty:
        legacy_note_audit.to_csv(LEGACY_NOTE_AUDIT_PATH, index=False, encoding="utf-8")

    if args.replace_raw:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = ROOT / "tmp" / f"Qm data_pre_standardization_backup_{timestamp}.xlsx"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if RAW_DATA_PATH.exists():
            shutil.copy2(RAW_DATA_PATH, backup_path)
        shutil.copy2(args.output_xlsx, RAW_DATA_PATH)
    else:
        backup_path = None

    print("Qm data standardization complete.")
    print(f"Input: {args.input}")
    print(f"Rows: {len(standardized)}")
    print(f"Sources: {standardized['Source_ID'].nunique()}")
    print(f"Missing DOI: {int(standardized['DOI'].isna().sum())}")
    print(f"Output XLSX: {args.output_xlsx}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Audit: {AUDIT_PATH}")
    print(f"Legacy note audit: {LEGACY_NOTE_AUDIT_PATH if not legacy_note_audit.empty else 'none'}")
    print(f"Zotero check: {zotero_check}")
    if backup_path is not None:
        print(f"Raw workbook replaced: {RAW_DATA_PATH}")
        print(f"Backup: {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
