from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "13b_race_channel_year_heterogeneity_memorysafe.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "Keep this file beside 13b_race_channel_year_heterogeneity_memorysafe.py"
    )

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "21d_county_year_fe_robustness"
OUTPUT_XLSX = OUTPUT_DIR / "21D_COUNTY_YEAR_FE_ROBUSTNESS.xlsx"

EXPECTED_N = 9_200_321
HEADLINE = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


def add_county_year(reg):
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)
    year_idx = race_year % 10
    if not np.array_equal(year_idx, channel_year % 10):
        raise RuntimeError("Race-year and channel-year encodings disagree.")
    reg["county_year_fe"] = (
        reg["county_fe"].to_numpy(np.int64) * 10 + year_idx.astype(np.int64)
    ).astype(np.int64)
    return reg


def pooled_data(reg):
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)
    reg["race_fe"] = (race_year // 10).astype(np.int8)
    reg["channel_fe"] = (channel_year // 10).astype(np.int8)
    annual_cols = [c for c in reg.columns if "_x_NonDirect_x_Y" in c]
    reg.drop(
        columns=annual_cols + ["race_year_fe", "channel_year_fe"],
        inplace=True,
        errors="ignore",
    )
    gc.collect()
    return reg


def pooled_formula():
    explicit = HEADLINE + [
        "income_asinh", "log_loan_amount", "log_property_value"
    ]
    fes = [
        "lender_year_fe", "county_year_fe", "race_fe", "channel_fe",
        "dti_fe", "age_fe", "sex_fe", "coapp_fe", "loan_type_fe",
    ]
    return "approval ~ " + " + ".join(explicit) + " | " + " + ".join(fes)


def annual_formula():
    explicit = HEADLINE.copy()
    for y in range(2019, 2025):
        for race in ["Black", "Hispanic", "Asian"]:
            explicit.append(f"{race}_x_NonDirect_x_Y{y}")
    explicit += ["income_asinh", "log_loan_amount", "log_property_value"]

    fes = [
        "lender_year_fe", "county_year_fe", "race_year_fe", "channel_year_fe",
        "dti_fe", "age_fe", "sex_fe", "coapp_fe", "loan_type_fe",
    ]
    return "approval ~ " + " + ".join(explicit) + " | " + " + ".join(fes)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No V1 Parquets found:\n{DATA_DIR}")

    schema = base.detect_schema(parts)
    reg, audit, maps = base.build_data(parts, schema)
    if audit["common_sample"] != EXPECTED_N:
        raise RuntimeError(
            f"Expected preferred sample {EXPECTED_N:,}; got {audit['common_sample']:,}"
        )

    reg = add_county_year(reg)

    info = pd.DataFrame(
        [{
            "n": len(reg),
            "lenders": reg["lender_cluster"].nunique(),
            "lender_year_fe": reg["lender_year_fe"].nunique(),
            "county_fe": reg["county_fe"].nunique(),
            "county_year_fe": reg["county_year_fe"].nunique(),
        }]
    )

    # Pooled model
    pdat = pooled_data(reg.copy())
    fit_pool = base.feols_memorysafe(pooled_formula(), pdat)
    pooled = base.tidy_fit(fit_pool)
    pooled = pooled[pooled["term"].isin(HEADLINE)].copy()

    # Annual H3
    fit_h3 = base.feols_memorysafe(annual_formula(), reg)
    names, beta, V = base.get_coef_vcov(fit_h3)
    n_clusters = int(reg["lender_cluster"].nunique())
    annual = base.build_annual_effects(names, beta, V, n_clusters - 1)
    joint = base.build_joint_tests(names, beta, V, n_clusters)

    pooled.to_csv(OUTPUT_DIR / "county_year_pooled_m5.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "county_year_annual_h3.csv", index=False)
    joint.to_csv(OUTPUT_DIR / "county_year_joint_h3.csv", index=False)
    info.to_csv(OUTPUT_DIR / "county_year_model_info.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        pooled.to_excel(writer, "Pooled M5", index=False)
        annual.to_excel(writer, "Annual H3", index=False)
        joint.to_excel(writer, "H3 joint", index=False)
        info.to_excel(writer, "Model info", index=False)

    print("\nPooled Race×Channel with county×year FE")
    print(
        pooled[["term", "estimate_pp", "std_error_pp", "p_value"]]
        .to_string(index=False)
    )
    print("\nAnnual H3 joint tests with county×year FE")
    print(joint[["test", "p_value"]].to_string(index=False))
    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
