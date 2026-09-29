from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# LOAD THE VERIFIED MEMORY-SAFE BASELINE/H3 BUILDER

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "13b_race_channel_year_heterogeneity_memorysafe.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "\nThis script reuses the verified memory-safe data builder.\n"
        "Please keep this file in the same folder as:\n\n"
        "  13b_race_channel_year_heterogeneity_memorysafe.py\n\n"
        f"Expected location:\n  {BASE_SCRIPT}\n"
    )

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# CONFIGURATION

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "race_channel_period_heterogeneity"
OUTPUT_XLSX = OUTPUT_DIR / "RACE_CHANNEL_PERIOD_HETEROGENEITY.xlsx"

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_PRIMARY_RACE_N = 9_395_768
EXPECTED_COMMON_SAMPLE_N = 9_200_321

RACES = ["Black", "Hispanic", "Asian"]

PERIODS = {
    0: "2018–2019 Pre-COVID",
    1: "2020–2021 COVID / low-rate",
    2: "2022–2024 Higher-rate / tightening",
}


# PERIOD CONSTRUCTION

def construct_period_variables(reg):
    """
    The verified H3 builder creates:
        race_year_fe    = race_code*10 + year_index
        channel_year_fe = channel_code*10 + year_index

    where year_index = 0,...,6 for 2018,...,2024.

    We recover race/channel/year identifiers from those compact codes and
    construct the three-period lower-order FEs and Race×Channel×Period terms.
    """

    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    year_index = race_year % 10
    race_code = race_year // 10
    channel_code = channel_year // 10

    # Internal consistency check.
    if not np.array_equal(year_index, channel_year % 10):
        raise RuntimeError("Race-year and channel-year encodings disagree.")

    period = np.full(len(reg), -1, dtype=np.int8)
    period[year_index <= 1] = 0
    period[(year_index >= 2) & (year_index <= 3)] = 1
    period[year_index >= 4] = 2

    if np.any(period < 0):
        raise RuntimeError("Unexpected period code.")

    reg["period"] = period
    reg["race_code"] = race_code.astype(np.int8)
    reg["channel_code"] = channel_code.astype(np.int8)

    reg["race_period_fe"] = (
        race_code.astype(np.int16) * 10 + period.astype(np.int16)
    ).astype(np.int16)

    reg["channel_period_fe"] = (
        channel_code.astype(np.int16) * 10 + period.astype(np.int16)
    ).astype(np.int16)

    # Build period-specific triple terms by combining disjoint year dummies.
    for race in RACES:
        reg[f"{race}_x_NonDirect_x_COVID"] = (
            reg[f"{race}_x_NonDirect_x_Y2020"].to_numpy(np.int8)
            + reg[f"{race}_x_NonDirect_x_Y2021"].to_numpy(np.int8)
        ).astype(np.int8)

        reg[f"{race}_x_NonDirect_x_Late"] = (
            reg[f"{race}_x_NonDirect_x_Y2022"].to_numpy(np.int8)
            + reg[f"{race}_x_NonDirect_x_Y2023"].to_numpy(np.int8)
            + reg[f"{race}_x_NonDirect_x_Y2024"].to_numpy(np.int8)
        ).astype(np.int8)

    # Drop annual triple terms and annual lower-order FEs to release RAM.
    annual_cols = [
        f"{race}_x_NonDirect_x_Y{year}"
        for race in RACES
        for year in range(2019, 2025)
    ]

    reg.drop(
        columns=annual_cols + ["race_year_fe", "channel_year_fe"],
        inplace=True,
    )

    gc.collect()
    return reg


# MODEL

def build_formula():
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",

        "Black_x_NonDirect_x_COVID",
        "Hispanic_x_NonDirect_x_COVID",
        "Asian_x_NonDirect_x_COVID",

        "Black_x_NonDirect_x_Late",
        "Hispanic_x_NonDirect_x_Late",
        "Asian_x_NonDirect_x_Late",

        "income_asinh",
        "log_loan_amount",
        "log_property_value",
    ]

    fixed_effects = [
        "lender_year_fe",
        "county_fe",
        "race_period_fe",
        "channel_period_fe",
        "dti_fe",
        "age_fe",
        "sex_fe",
        "coapp_fe",
        "loan_type_fe",
    ]

    return (
        "approval ~ "
        + " + ".join(explicit)
        + " | "
        + " + ".join(fixed_effects)
    )


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# PERIOD-SPECIFIC EFFECTS

def build_period_effects(names, beta, V, df_t):
    rows = []

    specs = [
        (0, PERIODS[0], None),
        (1, PERIODS[1], "COVID"),
        (2, PERIODS[2], "Late"),
    ]

    for race in RACES:
        base_term = f"{race}_x_NonDirect"

        for pcode, plabel, suffix in specs:
            weights = {base_term: 1.0}

            if suffix is not None:
                weights[f"{race}_x_NonDirect_x_{suffix}"] = 1.0

            est, se, tval, pval, lo, hi = base.linear_combo(
                names, beta, V, weights, df_t
            )

            rows.append({
                "race_group": race,
                "period_code": pcode,
                "period": plabel,
                "estimate": est,
                "std_error": se,
                "t_value": tval,
                "p_value": pval,
                "ci_low": lo,
                "ci_high": hi,
                "estimate_pp": est * 100,
                "std_error_pp": se * 100,
                "ci_low_pp": lo * 100,
                "ci_high_pp": hi * 100,
                "significance": stars(pval),
                "interpretation": (
                    "Race-vs-White gap narrower in non-direct"
                    if est > 0
                    else "Race-vs-White gap wider in non-direct"
                    if est < 0
                    else "No point-estimate difference"
                ),
            })

    return pd.DataFrame(rows)


# PAIRWISE CHANGES

def build_change_tests(names, beta, V, df_t):
    rows = []

    for race in RACES:
        covid = f"{race}_x_NonDirect_x_COVID"
        late = f"{race}_x_NonDirect_x_Late"

        comparisons = [
            (
                "2018–19 → 2020–21",
                {covid: 1.0},
            ),
            (
                "2018–19 → 2022–24",
                {late: 1.0},
            ),
            (
                "2020–21 → 2022–24",
                {late: 1.0, covid: -1.0},
            ),
        ]

        for label, weights in comparisons:
            est, se, tval, pval, lo, hi = base.linear_combo(
                names, beta, V, weights, df_t
            )

            rows.append({
                "race_group": race,
                "comparison": label,
                "change": est,
                "std_error": se,
                "t_value": tval,
                "p_value": pval,
                "ci_low": lo,
                "ci_high": hi,
                "change_pp": est * 100,
                "std_error_pp": se * 100,
                "ci_low_pp": lo * 100,
                "ci_high_pp": hi * 100,
                "significance": stars(pval),
            })

    return pd.DataFrame(rows)


# JOINT TESTS

def build_joint_tests(names, beta, V, n_clusters):
    rows = []
    all_terms = []

    for race in RACES:
        terms = [
            f"{race}_x_NonDirect_x_COVID",
            f"{race}_x_NonDirect_x_Late",
        ]
        all_terms.extend(terms)

        result = base.joint_zero_test(
            names, beta, V, terms, n_clusters - 1
        )

        rows.append({
            "test": f"{race}: Race × Channel constant across 3 periods",
            "null_hypothesis": (
                f"{race} Race×NonDirect effect is equal across "
                "2018–19, 2020–21 and 2022–24"
            ),
            **result,
            "significance": stars(result["p_value"]),
        })

    overall = base.joint_zero_test(
        names, beta, V, all_terms, n_clusters - 1
    )

    rows.append({
        "test": "OVERALL: Race × Channel constant across 3 periods",
        "null_hypothesis": (
            "All six period-change coefficients are jointly zero"
        ),
        **overall,
        "significance": stars(overall["p_value"]),
    })

    return pd.DataFrame(rows)


# CELL COUNTS

def build_cell_counts(reg):
    race_labels = {
        0: "White",
        1: "Black",
        2: "Hispanic",
        3: "Asian",
    }

    rows = []

    for pcode, plabel in PERIODS.items():
        for rcode, rlabel in race_labels.items():
            for ccode, clabel in [(0, "Direct"), (1, "Non-direct")]:
                m = (
                    reg["period"].eq(pcode)
                    & reg["race_code"].eq(rcode)
                    & reg["channel_code"].eq(ccode)
                )

                rows.append({
                    "period_code": pcode,
                    "period": plabel,
                    "race_group": rlabel,
                    "channel": clabel,
                    "n": int(m.sum()),
                    "raw_approval_percent": 100 * reg.loc[m, "approval"].mean(),
                })

    return pd.DataFrame(rows)


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))

    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquets found in:\n{DATA_DIR}"
        )

    print("=" * 104)
    print("RACE × CHANNEL × PERIOD HETEROGENEITY — PREFERRED MEMORY-SAFE LPM")
    print("=" * 104)
    print(f"Parts: {len(parts)}")
    print(f"PyFixest: {base.package_version('pyfixest')}")
    print("Reference: NH White / Direct / 2018–2019")
    print("SE: CRV1 clustered by lender")
    print("\nPeriods:")
    for code_, label in PERIODS.items():
        print(f"  {code_}: {label}")
    print(
        "\nThese are broad market environments, not causal treatment periods.\n"
    )

    schema = base.detect_schema(parts)

    print("Building the exact verified baseline-M5 sample...")
    reg, audit, category_maps = base.build_data(parts, schema)

    sample_audit = pd.DataFrame([
        {
            "stage": "Eligible V1 rows scanned",
            "n": audit["eligible"],
            "expected_n": EXPECTED_ELIGIBLE_N,
            "pass": audit["eligible"] == EXPECTED_ELIGIBLE_N,
        },
        {
            "stage": "dual50 all races",
            "n": audit["dual50"],
            "expected_n": EXPECTED_DUAL50_N,
            "pass": audit["dual50"] == EXPECTED_DUAL50_N,
        },
        {
            "stage": "dual50 + primary four races",
            "n": audit["primary_race"],
            "expected_n": EXPECTED_PRIMARY_RACE_N,
            "pass": audit["primary_race"] == EXPECTED_PRIMARY_RACE_N,
        },
        {
            "stage": "Common complete-case sample before FE singleton removal",
            "n": audit["common_sample"],
            "expected_n": EXPECTED_COMMON_SAMPLE_N,
            "pass": audit["common_sample"] == EXPECTED_COMMON_SAMPLE_N,
        },
    ])

    print("\nSample replication:")
    print(sample_audit.to_string(index=False))

    if not sample_audit["pass"].all():
        raise RuntimeError(
            "Period analysis does not reproduce the verified M5 sample."
        )

    print("\nConstructing three-period parameterisation...")
    reg = construct_period_variables(reg)

    mem_gb = reg.memory_usage(deep=True).sum() / (1024 ** 3)

    print(
        f"Compact dataframe after dropping annual interaction columns: "
        f"{len(reg):,} rows | {mem_gb:.2f} GB"
    )

    for pcode, plabel in PERIODS.items():
        n = int(reg["period"].eq(pcode).sum())
        print(f"  {plabel:40s}: {n:,}")

    cell_counts = build_cell_counts(reg)

    formula = build_formula()

    print("\nEstimating Race × Channel × Period model...")
    fit = base.feols_memorysafe(formula, reg)

    names, beta, V = base.get_coef_vcov(fit)

    n_clusters = int(reg["lender_cluster"].nunique())
    df_t = n_clusters - 1

    effects = build_period_effects(
        names, beta, V, df_t
    )

    changes = build_change_tests(
        names, beta, V, df_t
    )

    joint = build_joint_tests(
        names, beta, V, n_clusters
    )

    coefs = base.tidy_fit(fit)
    if "p_value" in coefs.columns:
        coefs["significance"] = coefs["p_value"].apply(stars)

    model_info = pd.DataFrame([{
        "observations_before_singleton_removal": len(reg),
        "lender_clusters": n_clusters,
        "lender_year_fe_groups": reg["lender_year_fe"].nunique(),
        "county_fe_groups": reg["county_fe"].nunique(),
        "race_period_fe_groups": reg["race_period_fe"].nunique(),
        "channel_period_fe_groups": reg["channel_period_fe"].nunique(),
        "dataframe_gb_before_fit": mem_gb,
        "cluster_df": df_t,
        "reference_period": PERIODS[0],
        "period_2": PERIODS[1],
        "period_3": PERIODS[2],
        "interpretation_note": (
            "Period coefficients capture heterogeneity across broad market "
            "environments and are not causal effects of COVID or monetary tightening."
        ),
        "formula": formula,
    }])

    schema_used = pd.DataFrame([
        {"role": k, "detected_column": v if v else "NOT FOUND"}
        for k, v in schema.items()
    ])

    category_rows = []
    for var, mapping in category_maps.items():
        for label, code_ in sorted(mapping.items(), key=lambda z: z[1]):
            category_rows.append({
                "variable": var,
                "code_used": code_,
                "cleaned_label": label,
            })
    category_levels = pd.DataFrame(category_rows)

    tables = {
        "period_race_channel_effects": effects,
        "period_change_tests": changes,
        "joint_period_tests": joint,
        "period_cell_counts": cell_counts,
        "model_coefficients": coefs,
        "sample_audit": sample_audit,
        "model_info": model_info,
        "schema_used": schema_used,
        "category_levels": category_levels,
    }

    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = f"Workbook not written ({exc}); CSVs were written."

    print("\n" + "=" * 104)
    print("PERIOD-SPECIFIC RACE × NON-DIRECT DIFFERENTIALS")
    print("=" * 104)
    print(
        effects[
            [
                "race_group", "period", "estimate_pp",
                "std_error_pp", "p_value",
                "ci_low_pp", "ci_high_pp", "significance",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 104)
    print("PAIRWISE PERIOD CHANGES")
    print("=" * 104)
    print(
        changes[
            [
                "race_group", "comparison", "change_pp",
                "std_error_pp", "p_value",
                "ci_low_pp", "ci_high_pp", "significance",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 104)
    print("JOINT TESTS")
    print("=" * 104)
    print(
        joint[
            [
                "test", "f_statistic", "df_num",
                "df_denom", "p_value", "significance",
            ]
        ].to_string(index=False)
    )

    overall_p = joint.loc[
        joint["test"].str.startswith("OVERALL"),
        "p_value",
    ].iloc[0]

    print("\nOverall period-heterogeneity conclusion:")
    if overall_p < 0.05:
        print(
            "  Reject constancy at the 5% level: Race × Channel differs "
            "across the three broad periods somewhere across the three groups."
        )
    else:
        print(
            "  Do not reject constancy at the 5% level: the three-period "
            "model does not provide strong joint evidence of heterogeneity."
        )

    print(
        "\nIMPORTANT: do not describe the 2022–2024 difference as a causal "
        "effect of monetary tightening."
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. period_race_channel_effects.csv")
    print("  2. period_change_tests.csv")
    print("  3. joint_period_tests.csv")
    print("  4. period_cell_counts.csv")
    print("  5. sample_audit.csv")
    print("  6. model_info.csv")


if __name__ == "__main__":
    main()
