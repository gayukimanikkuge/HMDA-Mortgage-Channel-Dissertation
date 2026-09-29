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
        "\nKeep this script in the SAME folder as:\n"
        "  13b_race_channel_year_heterogeneity_memorysafe.py\n"
    )

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "coapplicant_control_sensitivity"
OUTPUT_XLSX = OUTPUT_DIR / "COAPP_CONTROL_SENSITIVITY.xlsx"

EXPECTED_COMMON_SAMPLE_N = 9_200_321

HEADLINE = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


def pooled_formula(include_coapp=True):
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
        "income_asinh",
        "log_loan_amount",
        "log_property_value",
    ]

    fixed_effects = [
        "lender_year_fe",
        "county_fe",
        "race_fe",
        "channel_fe",
        "dti_fe",
        "age_fe",
        "sex_fe",
        "loan_type_fe",
    ]

    if include_coapp:
        fixed_effects.insert(-1, "coapp_fe")

    return (
        "approval ~ "
        + " + ".join(explicit)
        + " | "
        + " + ".join(fixed_effects)
    )


def prepare_pooled_data(reg):
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    race_code = (race_year // 10).astype(np.int8)
    nd = (channel_year // 10).astype(np.int8)

    reg["race_fe"] = race_code
    reg["channel_fe"] = nd

    reg["Black_x_NonDirect"] = (
        (race_code == 1).astype(np.int8) * nd
    ).astype(np.int8)

    reg["Hispanic_x_NonDirect"] = (
        (race_code == 2).astype(np.int8) * nd
    ).astype(np.int8)

    reg["Asian_x_NonDirect"] = (
        (race_code == 3).astype(np.int8) * nd
    ).astype(np.int8)

    annual_cols = [
        c for c in reg.columns
        if "_x_NonDirect_x_Y" in c
    ]

    reg.drop(
        columns=annual_cols + [
            "race_year_fe",
            "channel_year_fe",
        ],
        inplace=True,
        errors="ignore",
    )

    gc.collect()
    return reg


def extract(fit, model):
    t = base.tidy_fit(fit).copy()
    t["model"] = model

    out = t.loc[
        t["term"].isin(HEADLINE)
    ].copy()

    keep = [
        c for c in [
            "model", "term",
            "estimate", "std_error", "p_value",
            "ci_low", "ci_high",
            "estimate_pp", "std_error_pp",
            "ci_low_pp", "ci_high_pp",
        ]
        if c in out.columns
    ]

    return out[keep]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquets found:\n{DATA_DIR}"
        )

    print("=" * 100)
    print("CO-APPLICANT CONTROL SENSITIVITY")
    print("=" * 100)

    schema = base.detect_schema(parts)

    print("\nBuilding exact preferred dual50 complete-case sample...")
    reg, audit, maps = base.build_data(parts, schema)

    if audit["common_sample"] != EXPECTED_COMMON_SAMPLE_N:
        raise RuntimeError(
            f"Expected {EXPECTED_COMMON_SAMPLE_N:,} observations, "
            f"got {audit['common_sample']:,}."
        )

    reg = prepare_pooled_data(reg)

    formula_with = pooled_formula(include_coapp=True)
    formula_without = pooled_formula(include_coapp=False)

    print("\nA. Preferred M5 WITH co-applicant control")
    print(formula_with)
    fit_with = base.feols_memorysafe(
        formula_with,
        reg,
    )

    print("\nB. Preferred M5 WITHOUT co-applicant control")
    print(formula_without)
    fit_without = base.feols_memorysafe(
        formula_without,
        reg,
    )

    results = pd.concat([
        extract(fit_with, "Preferred M5 with coapp FE"),
        extract(fit_without, "M5 without coapp FE"),
    ], ignore_index=True)

    # Add coefficient movement table.
    piv = results.pivot(
        index="term",
        columns="model",
        values="estimate_pp",
    ).reset_index()

    if {
        "Preferred M5 with coapp FE",
        "M5 without coapp FE",
    }.issubset(piv.columns):
        piv["change_without_minus_with_pp"] = (
            piv["M5 without coapp FE"]
            - piv["Preferred M5 with coapp FE"]
        )

    info = pd.DataFrame([{
        "preferred_complete_case_n_before_singleton_removal": audit["common_sample"],
        "purpose": (
            "Test whether headline Race × Channel estimates depend on "
            "co-applicant status coding."
        ),
        "interpretation_rule": (
            "If coefficients and inference are materially unchanged when "
            "coapp_fe is omitted, main results are not sensitive to the "
            "co-applicant control."
        ),
    }])

    results.to_csv(
        OUTPUT_DIR / "coapp_control_sensitivity.csv",
        index=False,
    )
    piv.to_csv(
        OUTPUT_DIR / "coapp_control_coefficient_changes.csv",
        index=False,
    )
    info.to_csv(
        OUTPUT_DIR / "coapp_control_model_info.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:
        results.to_excel(
            writer,
            sheet_name="Results",
            index=False,
        )
        piv.to_excel(
            writer,
            sheet_name="Coefficient changes",
            index=False,
        )
        info.to_excel(
            writer,
            sheet_name="Model info",
            index=False,
        )

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(results.to_string(index=False))

    print("\nCoefficient movement (percentage points):")
    print(piv.to_string(index=False))

    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
