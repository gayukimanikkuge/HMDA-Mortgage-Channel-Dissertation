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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "21a_bad_control_sensitivity"
OUTPUT_XLSX = OUTPUT_DIR / "21A_BAD_CONTROL_SENSITIVITY.xlsx"

EXPECTED_N = 9_200_321
HEADLINE = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


def prepare_pooled(reg):
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    year_a = race_year % 10
    year_b = channel_year % 10
    if not np.array_equal(year_a, year_b):
        raise RuntimeError("Race-year and channel-year encodings disagree.")

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


def formula(include_loan_property=True):
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
        "income_asinh",
    ]
    if include_loan_property:
        explicit += ["log_loan_amount", "log_property_value"]

    fes = [
        "lender_year_fe",
        "county_fe",
        "race_fe",
        "channel_fe",
        "dti_fe",
        "age_fe",
        "sex_fe",
        "coapp_fe",
        "loan_type_fe",
    ]
    return "approval ~ " + " + ".join(explicit) + " | " + " + ".join(fes)


def extract(fit, model):
    t = base.tidy_fit(fit).copy()
    out = t[t["term"].isin(HEADLINE)].copy()
    out["model"] = model
    keep = [
        "model", "term", "estimate", "std_error", "p_value",
        "ci_low", "ci_high", "estimate_pp", "std_error_pp",
        "ci_low_pp", "ci_high_pp",
    ]
    return out[[c for c in keep if c in out.columns]]


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

    reg = prepare_pooled(reg)

    f_full = formula(True)
    f_drop = formula(False)

    print("\nA. Preferred M5")
    print(f_full)
    fit_full = base.feols_memorysafe(f_full, reg)

    print("\nB. Preferred M5 WITHOUT loan amount/property value")
    print(f_drop)
    fit_drop = base.feols_memorysafe(f_drop, reg)

    results = pd.concat(
        [
            extract(fit_full, "Preferred M5"),
            extract(fit_drop, "No loan amount / property value"),
        ],
        ignore_index=True,
    )

    wide = results.pivot(
        index="term", columns="model", values=["estimate_pp", "std_error_pp", "p_value"]
    )

    comparison = []
    for term in HEADLINE:
        a = results[(results.model == "Preferred M5") & (results.term == term)].iloc[0]
        b = results[
            (results.model == "No loan amount / property value")
            & (results.term == term)
        ].iloc[0]
        comparison.append(
            {
                "term": term,
                "preferred_estimate_pp": a["estimate_pp"],
                "reduced_estimate_pp": b["estimate_pp"],
                "change_pp": b["estimate_pp"] - a["estimate_pp"],
                "preferred_p": a["p_value"],
                "reduced_p": b["p_value"],
            }
        )
    comparison = pd.DataFrame(comparison)

    info = pd.DataFrame(
        [{
            "preferred_complete_case_n": len(reg),
            "lender_clusters": reg["lender_cluster"].nunique(),
            "lender_year_fe_groups": reg["lender_year_fe"].nunique(),
            "county_fe_groups": reg["county_fe"].nunique(),
            "preferred_formula": f_full,
            "reduced_formula": f_drop,
            "purpose": (
                "Tests whether pooled Race×Channel estimates depend on conditioning "
                "on loan amount and property value."
            ),
        }]
    )

    results.to_csv(OUTPUT_DIR / "bad_control_results.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "bad_control_comparison.csv", index=False)
    info.to_csv(OUTPUT_DIR / "bad_control_model_info.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        results.to_excel(writer, "Results", index=False)
        comparison.to_excel(writer, "Comparison", index=False)
        info.to_excel(writer, "Model info", index=False)

    print("\n" + "=" * 100)
    print("BAD-CONTROL SENSITIVITY — HEADLINE COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
