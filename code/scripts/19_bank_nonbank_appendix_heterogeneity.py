from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# VERIFIED BASE HELPERS

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


# CONFIG

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "institution_type_heterogeneity"
OUTPUT_XLSX = OUTPUT_DIR / "INSTITUTION_TYPE_HETEROGENEITY.xlsx"

EXPECTED_COMMON_SAMPLE_N = 9_200_321

DEPOSITORY_CU_TYPES = {
    10, 11, 12, 13, 14,
    20, 21, 22, 23,
    30, 31, 32, 33,
}

MORTGAGE_COMPANY_TYPES = {40, 41}

RACES = ["Black", "Hispanic", "Asian"]


# HELPERS

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


def find_avery_file():
    candidates = [
        Path.home() / "Downloads" / "hmda-2018-present.xlsx",
        PROJECT_DIR / "hmda-2018-present.xlsx",
        PROJECT_DIR / "raw_data" / "hmda-2018-present.xlsx",
        PROJECT_DIR / "lender_files" / "hmda-2018-present.xlsx",
        Path.home() / "Desktop" / "hmda-2018-present.xlsx",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Limited project-folder search only; do not scan the whole computer.
    for root in [
        Path.home() / "Downloads",
        PROJECT_DIR,
    ]:
        if root.exists():
            hits = list(root.rglob("hmda-2018-present.xlsx"))
            if hits:
                return hits[0]

    raise FileNotFoundError(
        "\nPhiladelphia Fed Avery lender file not found.\n\n"
        "Download the current '2018–2025 HMDA Lender File' from the "
        "Federal Reserve Bank of Philadelphia and save it as:\n\n"
        f"  {Path.home() / 'Downloads' / 'hmda-2018-present.xlsx'}\n\n"
        "Then run this script again."
    )


def load_avery(path):
    print(f"\nReading Avery lender file:\n  {path}")

    a = pd.read_excel(path)

    # Case-insensitive schema matching.
    lookup = {str(c).strip().upper(): c for c in a.columns}

    required = ["YEAR", "LEI", "TYPE"]
    missing = [x for x in required if x not in lookup]

    if missing:
        raise KeyError(
            "Avery file is missing required field(s): "
            + ", ".join(missing)
        )

    a = a[
        [
            lookup["YEAR"],
            lookup["LEI"],
            lookup["TYPE"],
        ]
    ].copy()

    a.columns = ["year", "lei", "type"]

    a["year"] = pd.to_numeric(
        a["year"],
        errors="coerce",
    ).astype("Int16")

    a["lei"] = (
        a["lei"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    a["type"] = pd.to_numeric(
        a["type"],
        errors="coerce",
    ).astype("Int16")

    a = a.loc[
        a["year"].between(2018, 2024, inclusive="both")
        & a["lei"].notna()
        & a["type"].notna()
    ].copy()

    # Ensure one unambiguous TYPE per filer-year.
    conflicts = (
        a.groupby(["year", "lei"])["type"]
        .nunique()
        .reset_index(name="n_types")
    )

    bad = conflicts.loc[conflicts["n_types"] > 1]

    if len(bad):
        raise RuntimeError(
            f"Avery file contains {len(bad):,} YEAR+LEI combinations "
            "with conflicting TYPE codes."
        )

    a = a.drop_duplicates(
        ["year", "lei"],
        keep="last",
    )

    def group_type(t):
        t = int(t)
        if t in DEPOSITORY_CU_TYPES:
            return 0
        if t in MORTGAGE_COMPANY_TYPES:
            return 1
        return np.nan

    a["mortgage_company"] = (
        a["type"]
        .map(group_type)
    )

    return a


def prepare_institution_data(reg, maps, avery):

    if len(reg) != EXPECTED_COMMON_SAMPLE_N:
        raise RuntimeError(
            f"Expected {EXPECTED_COMMON_SAMPLE_N:,} preferred observations "
            f"but got {len(reg):,}."
        )

    # Recover year from the verified compact lender-year encoding.
    year = (
        reg["lender_year_fe"].to_numpy(np.int32) % 10
        + 2018
    ).astype(np.int16)

    lender_code = reg["lender_cluster"].to_numpy(np.int32)

    inv_lender = {
        int(code): str(label).strip().upper()
        for label, code in maps["lender"].items()
    }

    ly = pd.DataFrame({
        "lender_code": lender_code,
        "year": year,
        "lender_year_fe": reg["lender_year_fe"].to_numpy(np.int32),
    }).drop_duplicates(
        ["lender_code", "year", "lender_year_fe"]
    )

    ly["lei"] = ly["lender_code"].map(inv_lender)

    if ly["lei"].isna().any():
        raise RuntimeError(
            "At least one compact lender code could not be mapped back to LEI."
        )

    ly = ly.merge(
        avery[
            ["year", "lei", "type", "mortgage_company"]
        ],
        how="left",
        on=["year", "lei"],
        validate="one_to_one",
    )

    matched_ly = int(
        ly["mortgage_company"].notna().sum()
    )

    total_ly = len(ly)

    print(
        f"\nAvery match at lender-year level: "
        f"{matched_ly:,}/{total_ly:,} "
        f"({100*matched_ly/total_ly:.2f}%)"
    )

    ly_map = (
        ly.set_index("lender_year_fe")["mortgage_company"]
        .to_dict()
    )

    inst = (
        reg["lender_year_fe"]
        .map(ly_map)
    )

    matched_obs = int(inst.notna().sum())
    coverage = 100 * matched_obs / len(reg)

    print(
        f"Avery match at observation level: "
        f"{matched_obs:,}/{len(reg):,} "
        f"({coverage:.3f}%)"
    )

    if coverage < 95:
        raise RuntimeError(
            "Institution-type coverage is below 95%. "
            "Do not estimate until the LEI/year merge is investigated."
        )

    keep = inst.notna().to_numpy()

    reg = reg.loc[keep].copy()
    inst = inst.loc[keep].astype(np.int8)

    # Recover race and channel codes from the verified H3 encodings.
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    race_code = (race_year // 10).astype(np.int8)
    nd = (channel_year // 10).astype(np.int8)
    instv = inst.to_numpy(np.int8)

    # Lower-order Race×Institution and Channel×Institution terms.
    reg["race_institution_fe"] = (
        race_code.astype(np.int16) * 10
        + instv.astype(np.int16)
    ).astype(np.int16)

    reg["channel_institution_fe"] = (
        nd.astype(np.int16) * 10
        + instv.astype(np.int16)
    ).astype(np.int16)

    # Explicit triple differences.
    for race, code in [
        ("Black", 1),
        ("Hispanic", 2),
        ("Asian", 3),
    ]:
        reg[
            f"{race}_x_NonDirect_x_MortgageCompany"
        ] = (
            (race_code == code)
            & (nd == 1)
            & (instv == 1)
        ).astype(np.int8)

    reg["mortgage_company"] = instv

    # Annual H3-specific columns are not used here.
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
    )

    gc.collect()

    return reg, ly, matched_obs


def build_formula():
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
        "Black_x_NonDirect_x_MortgageCompany",
        "Hispanic_x_NonDirect_x_MortgageCompany",
        "Asian_x_NonDirect_x_MortgageCompany",
        "income_asinh",
        "log_loan_amount",
        "log_property_value",
    ]

    fixed_effects = [
        "lender_year_fe",
        "county_fe",
        "race_institution_fe",
        "channel_institution_fe",
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


def build_effects(names, beta, V, df_t):
    rows = []

    for race in RACES:
        base_term = f"{race}_x_NonDirect"
        triple = (
            f"{race}_x_NonDirect_x_MortgageCompany"
        )

        for group, label in [
            (0, "Depository / credit union"),
            (1, "Mortgage-company filer"),
        ]:
            weights = {base_term: 1.0}

            if group == 1:
                weights[triple] = 1.0

            est, se, tval, pval, lo, hi = base.linear_combo(
                names,
                beta,
                V,
                weights,
                df_t,
            )

            rows.append({
                "race_group": race,
                "institution_group": label,
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
            })

    return pd.DataFrame(rows)


def build_difference_tests(names, beta, V, df_t):
    rows = []

    triple_terms = []

    for race in RACES:
        term = (
            f"{race}_x_NonDirect_x_MortgageCompany"
        )

        triple_terms.append(term)

        est, se, tval, pval, lo, hi = base.linear_combo(
            names,
            beta,
            V,
            {term: 1.0},
            df_t,
        )

        rows.append({
            "test": (
                f"{race}: mortgage-company vs depository/CU "
                "difference in Race × Channel"
            ),
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
        })

    joint = base.joint_zero_test(
        names,
        beta,
        V,
        triple_terms,
        int(df_t),
    )

    rows.append({
        "test": (
            "OVERALL: all Race × Channel × MortgageCompany "
            "differences jointly zero"
        ),
        "estimate": np.nan,
        "std_error": np.nan,
        "t_value": np.nan,
        "p_value": joint["p_value"],
        "ci_low": np.nan,
        "ci_high": np.nan,
        "estimate_pp": np.nan,
        "std_error_pp": np.nan,
        "ci_low_pp": np.nan,
        "ci_high_pp": np.nan,
        "significance": stars(joint["p_value"]),
        "f_statistic": joint["f_statistic"],
        "df_num": joint["df_num"],
        "df_denom": joint["df_denom"],
    })

    return pd.DataFrame(rows)


# MAIN

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    avery_path = find_avery_file()
    avery = load_avery(avery_path)

    parts = sorted(
        DATA_DIR.glob("*.parquet")
    )

    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquet files found:\n{DATA_DIR}"
        )

    print("=" * 108)
    print("SUPPLEMENTARY INSTITUTION-TYPE HETEROGENEITY")
    print("=" * 108)

    schema = base.detect_schema(parts)

    print("\nBuilding exact preferred M5 sample...")
    reg, audit, maps = base.build_data(
        parts,
        schema,
    )

    reg, ly_match, matched_obs = prepare_institution_data(
        reg,
        maps,
        avery,
    )

    counts = (
        reg.groupby("mortgage_company")
        .size()
        .rename("n")
        .reset_index()
    )

    counts["institution_group"] = counts[
        "mortgage_company"
    ].map({
        0: "Depository / credit union",
        1: "Mortgage-company filer",
    })

    print("\nObservation counts by institution group:")
    print(counts.to_string(index=False))

    formula = build_formula()

    print("\nEstimating:")
    print(formula)

    fit = base.feols_memorysafe(
        formula,
        reg,
    )

    names, beta, V = base.get_coef_vcov(fit)

    n_clusters = int(
        reg["lender_cluster"].nunique()
    )

    df_t = n_clusters - 1

    effects = build_effects(
        names,
        beta,
        V,
        df_t,
    )

    tests = build_difference_tests(
        names,
        beta,
        V,
        df_t,
    )

    model_info = pd.DataFrame([{
        "preferred_sample_n": EXPECTED_COMMON_SAMPLE_N,
        "matched_institution_type_observations": matched_obs,
        "institution_type_analysis_n": len(reg),
        "institution_type_coverage_percent": (
            100 * matched_obs / EXPECTED_COMMON_SAMPLE_N
        ),
        "lender_clusters": n_clusters,
        "lender_years_in_analysis": int(
            reg["lender_year_fe"].nunique()
        ),
        "avery_file": str(avery_path),
        "reference_institution_group": "Depository / credit union",
        "formula": formula,
        "interpretation_note": (
            "Supplementary heterogeneity only; not causal and not H5."
        ),
    }])

    sample_audit = pd.DataFrame([
        {
            "stage": "Preferred complete-case sample",
            "n": EXPECTED_COMMON_SAMPLE_N,
        },
        {
            "stage": "Matched to Avery institution type",
            "n": matched_obs,
        },
        {
            "stage": "Final institution-type heterogeneity sample",
            "n": len(reg),
        },
    ])

    tables = {
        "institution_type_effects": effects,
        "institution_type_difference_tests": tests,
        "institution_type_sample_audit": sample_audit,
        "institution_type_model_info": model_info,
        "institution_type_counts": counts,
        "lender_year_match_audit": ly_match,
    }

    for name, df in tables.items():
        df.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )

    try:
        with pd.ExcelWriter(
            OUTPUT_XLSX,
            engine="openpyxl",
        ) as writer:
            for name, df in tables.items():
                df.to_excel(
                    writer,
                    sheet_name=name[:31],
                    index=False,
                )
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = (
            f"Workbook not written ({exc}); CSV files were written."
        )

    print("\n" + "=" * 108)
    print("INSTITUTION-TYPE HETEROGENEITY COMPLETE")
    print("=" * 108)

    print("\nInstitution-specific Race × Channel effects:")
    print(
        effects[
            [
                "race_group",
                "institution_group",
                "estimate_pp",
                "std_error_pp",
                "p_value",
                "significance",
            ]
        ].to_string(index=False)
    )

    print("\nDifference tests:")
    print(
        tests[
            [
                "test",
                "estimate_pp",
                "p_value",
                "significance",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. institution_type_effects.csv")
    print("  2. institution_type_difference_tests.csv")
    print("  3. institution_type_sample_audit.csv")
    print("  4. institution_type_model_info.csv")


if __name__ == "__main__":
    main()
