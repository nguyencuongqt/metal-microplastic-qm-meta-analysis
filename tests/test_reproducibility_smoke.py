from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_core_files_exist():
    required = [
        "config/config.yaml",
        "data_raw/Qm data.xlsx",
        "data_processed/00_raw_data_snapshot.csv",
        "data_processed/02_cleaned_data.csv",
        "data_processed/03_enriched_data.csv",
        "scripts/01_data_loading.py",
        "scripts/02_data_cleaning.py",
        "scripts/03_descriptor_enrichment.py",
        "run_pipeline.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing


def test_config_loads_and_uses_relative_paths():
    with open(ROOT / "config/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["random_seed"] == 42
    assert config["paths"]["raw_data"] == "data_raw/Qm data.xlsx"
    for value in config["paths"].values():
        assert not Path(value).is_absolute()


def test_processed_data_policy_invariants():
    clean = pd.read_csv(ROOT / "data_processed/02_cleaned_data.csv")
    enriched = pd.read_csv(ROOT / "data_processed/03_enriched_data.csv")

    assert len(clean) == 316
    assert len(enriched) == len(clean)
    assert clean.duplicated().sum() == 0
    assert enriched.duplicated().sum() == 0
    assert clean["Qm"].notna().all()
    assert enriched["Qm"].notna().all()
    assert np.allclose(clean["log_Qm"], np.log(clean["Qm"]), rtol=1e-5)
    assert int(clean["pH_missing"].sum()) == int(clean["pH"].isna().sum())


def test_model_summaries_present_without_required_trace_files():
    for path in [
        "models/primary_model_summary.json",
        "models/main_mechanistic_model_summary.json",
        "models/supplementary_sa_sensitivity_model_summary.json",
    ]:
        assert (ROOT / path).exists()
