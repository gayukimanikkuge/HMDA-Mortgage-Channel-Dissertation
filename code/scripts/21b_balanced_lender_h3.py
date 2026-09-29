from __future__ import annotations

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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "21b_balanced_lender_h3"
OUTPUT_XLSX = OUTPUT_DIR / "21B_BALANCED_LENDER_H3.xlsx"

EXPECTED_N = 9_200_321
YEARS = list(range(2018, 2025))


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

    year_idx = (reg["race_year_fe"].to_numpy(np.int16) % 10).astype(np.int8)
    year_idx_2 = (reg["channel_year_fe"].to_numpy(np.int16) % 10).astype(np.int8)
    if not np.array_equal(year_idx, year_idx_2):
        raise RuntimeError("Race-year and channel-year encodings disagree.")

    reg["year"] = (2018 + year_idx).astype(np.int16)

    lender_year_presence = (
        reg[["lender_cluster", "year"]]
        .drop_duplicates()
        .groupby("lender_cluster")["year"]
        .nunique()
    )

    balanced_lenders = lender_year_presence[lender_year_presence.eq(7)].index
    balanced = reg[reg["lender_cluster"].isin(balanced_lenders)].copy()

    year_counts = (
        balanced.groupby("year")
        .agg(
            observations=("approval", "size"),
            lenders=("lender_cluster", "nunique"),
            lender_years=("lender_year_fe", "nunique"),
        )
        .reset_index()
    )

    if set(year_counts["year"]) != set(YEARS):
        raise RuntimeError("Balanced sample does not contain all seven years.")

    print("\nBalanced lenders:", len(balanced_lenders))
    print("Balanced sample observations:", f"{len(balanced):,}")
    print(year_counts.to_string(index=False))

    formula = base.build_formula()
    fit = base.feols_memorysafe(formula, balanced)

    names, beta, V = base.get_coef_vcov(fit)
    n_clusters = int(balanced["lender_cluster"].nunique())
    annual = base.build_annual_effects(names, beta, V, n_clusters - 1)
    joint = base.build_joint_tests(names, beta, V, n_clusters)
    coefs = base.tidy_fit(fit)

    audit_out = pd.DataFrame(
        [{
            "full_preferred_n": len(reg),
            "full_preferred_lenders": reg["lender_cluster"].nunique(),
            "balanced_lenders": len(balanced_lenders),
            "balanced_n": len(balanced),
            "balanced_share_percent": 100 * len(balanced) / len(reg),
            "criterion": "Lender appears in preferred dual50 complete-case sample in all 7 years",
        }]
    )

    lender_list = pd.DataFrame({"lender_cluster": balanced_lenders})

    annual.to_csv(OUTPUT_DIR / "balanced_annual_race_channel_effects.csv", index=False)
    joint.to_csv(OUTPUT_DIR / "balanced_joint_h3_tests.csv", index=False)
    year_counts.to_csv(OUTPUT_DIR / "balanced_sample_by_year.csv", index=False)
    audit_out.to_csv(OUTPUT_DIR / "balanced_sample_audit.csv", index=False)
    lender_list.to_csv(OUTPUT_DIR / "balanced_lender_list.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "balanced_model_coefficients.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        annual.to_excel(writer, "Annual effects", index=False)
        joint.to_excel(writer, "Joint H3 tests", index=False)
        year_counts.to_excel(writer, "Sample by year", index=False)
        audit_out.to_excel(writer, "Audit", index=False)
        lender_list.to_excel(writer, "Balanced lenders", index=False)
        coefs.to_excel(writer, "Coefficients", index=False)

    print("\n" + "=" * 100)
    print("BALANCED-LENDER ANNUAL EFFECTS")
    print("=" * 100)
    print(
        annual[
            ["race_group", "year", "estimate_pp", "std_error_pp", "p_value",
             "ci_low_pp", "ci_high_pp"]
        ].to_string(index=False)
    )
    print("\nJOINT H3 TESTS")
    print(joint[["test", "f_statistic", "p_value"]].to_string(index=False))
    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
