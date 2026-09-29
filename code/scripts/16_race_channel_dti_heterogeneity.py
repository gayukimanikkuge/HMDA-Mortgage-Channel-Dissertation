from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# LOAD VERIFIED MEMORY-SAFE BUILDER

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "13b_race_channel_year_heterogeneity_memorysafe.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "\nKeep this script in the SAME folder as:\n\n"
        "  13b_race_channel_year_heterogeneity_memorysafe.py\n\n"
        f"Expected:\n  {BASE_SCRIPT}\n"
    )

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# CONFIGURATION

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "race_channel_dti_heterogeneity"
OUTPUT_XLSX = OUTPUT_DIR / "RACE_CHANNEL_DTI_HETEROGENEITY.xlsx"

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_PRIMARY_RACE_N = 9_395_768
EXPECTED_COMMON_SAMPLE_N = 9_200_321

RACES = ["Black", "Hispanic", "Asian"]

DTI_ORDER = [
    "<20",
    "20-<30",
    "30-<36",
    "36-<40",
    "40-<43",
    "43-<50",
    "50-60",
    ">60",
]

MISSING_DTI_LABEL = "Missing / Not relied upon"


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


def normalize_dti_label(label):
    x = str(label).strip()
    replacements = {
        "43 - <50": "43-<50",
        "20 - <30": "20-<30",
        "30 - <36": "30-<36",
        "36 - <40": "36-<40",
        "40 - <43": "40-<43",
        "50 - 60": "50-60",
        "Missing/Not relied upon": MISSING_DTI_LABEL,
        "Missing / Not Relied Upon": MISSING_DTI_LABEL,
    }
    return replacements.get(x, x)


# CONSTRUCT DTI HETEROGENEITY VARIABLES

def prepare_dti_data(reg, category_maps):


    if "dti" not in category_maps:
        raise KeyError("DTI category map was not returned by the verified builder.")

    dti_map = {
        normalize_dti_label(label): int(code)
        for label, code in category_maps["dti"].items()
    }

    print("\nDTI labels detected in final V1:")
    for label, code in sorted(dti_map.items(), key=lambda z: z[1]):
        print(f"  code {code}: {label}")

    expected = set(DTI_ORDER + [MISSING_DTI_LABEL])
    unexpected = set(dti_map) - expected
    missing_expected = set(DTI_ORDER) - set(dti_map)

    if unexpected:
        raise RuntimeError(
            "Unexpected DTI label(s) detected. Review before estimation:\n"
            + "\n".join(sorted(unexpected))
        )

    if missing_expected:
        raise RuntimeError(
            "Expected reported DTI bin(s) missing:\n"
            + "\n".join(sorted(missing_expected))
        )

    # Recover race and channel from the verified compact lower-order encodings.
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    year_index = race_year % 10
    if not np.array_equal(year_index, channel_year % 10):
        raise RuntimeError("Race-year and channel-year encodings disagree.")

    race_code = (race_year // 10).astype(np.int8)
    channel_code = (channel_year // 10).astype(np.int8)

    # Invert internal DTI code -> clean label.
    inverse_dti = {code: label for label, code in dti_map.items()}

    internal_dti = reg["dti_fe"].to_numpy(np.int16)
    labels = np.array(
        [inverse_dti.get(int(code), "__UNKNOWN__") for code in internal_dti],
        dtype=object,
    )

    if np.any(labels == "__UNKNOWN__"):
        raise RuntimeError("At least one internal DTI code could not be mapped.")

    # Exclude missing/not relied upon DTI only for this mechanism analysis
    keep = labels != MISSING_DTI_LABEL

    before_n = len(reg)
    excluded_n = int((~keep).sum())

    reg = reg.loc[keep].copy()
    labels = labels[keep]
    race_code = race_code[keep]
    channel_code = channel_code[keep]

    print(
        f"\nDTI mechanism sample: {len(reg):,} / {before_n:,} "
        f"preferred observations retained"
    )
    print(
        f"Excluded because DTI was missing/not relied upon: {excluded_n:,} "
        f"({100*excluded_n/before_n:.3f}%)"
    )

    # Recode into ordered analytical DTI index 0,...,7 independent of the arbitrary factor-code order in the source builder.
    order_lookup = {label: i for i, label in enumerate(DTI_ORDER)}

    dti_order = np.array(
        [order_lookup[label] for label in labels],
        dtype=np.int8,
    )

    reg["race_code"] = race_code
    reg["channel_code"] = channel_code
    reg["dti_order"] = dti_order

    # Lower-order interactions required for a correct three-way specification.
    reg["race_dti_fe"] = (
        race_code.astype(np.int16) * 10
        + dti_order.astype(np.int16)
    ).astype(np.int16)

    reg["channel_dti_fe"] = (
        channel_code.astype(np.int16) * 10
        + dti_order.astype(np.int16)
    ).astype(np.int16)

    # Base Race×NonDirect terms already exist and represent the reference DTI bin once lower-order Race×DTI and Channel×DTI terms are absorbed.
    
    # Create seven changes relative to <20.
    for race, rcode in [("Black", 1), ("Hispanic", 2), ("Asian", 3)]:
        for j, label in enumerate(DTI_ORDER[1:], start=1):
            safe = (
                label.replace("<", "lt")
                     .replace(">", "gt")
                     .replace("-", "_")
            )
            col = f"{race}_x_NonDirect_x_DTI_{safe}"

            reg[col] = (
                (race_code == rcode)
                & (channel_code == 1)
                & (dti_order == j)
            ).astype(np.int8)

    # Drop annual H3-specific columns and the original standalone dti_fe.
    annual_cols = [
        f"{race}_x_NonDirect_x_Y{year}"
        for race in RACES
        for year in range(2019, 2025)
        if f"{race}_x_NonDirect_x_Y{year}" in reg.columns
    ]

    drop_cols = annual_cols + [
        c for c in ["race_year_fe", "channel_year_fe", "dti_fe"]
        if c in reg.columns
    ]

    reg.drop(columns=drop_cols, inplace=True)
    gc.collect()

    return reg, excluded_n


# MODEL

def build_formula():
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
    ]

    for race in RACES:
        for label in DTI_ORDER[1:]:
            safe = (
                label.replace("<", "lt")
                     .replace(">", "gt")
                     .replace("-", "_")
            )
            explicit.append(
                f"{race}_x_NonDirect_x_DTI_{safe}"
            )

    explicit += [
        "income_asinh",
        "log_loan_amount",
        "log_property_value",
    ]

    fixed_effects = [
        "lender_year_fe",
        "county_fe",
        "race_dti_fe",
        "channel_dti_fe",
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


# ABSOLUTE DTI-SPECIFIC EFFECTS

def dti_term_name(race, label):
    safe = (
        label.replace("<", "lt")
             .replace(">", "gt")
             .replace("-", "_")
    )
    return f"{race}_x_NonDirect_x_DTI_{safe}"


def build_dti_effects(names, beta, V, df_t):
    rows = []

    for race in RACES:
        base_term = f"{race}_x_NonDirect"

        for idx, label in enumerate(DTI_ORDER):
            weights = {base_term: 1.0}

            if idx > 0:
                weights[dti_term_name(race, label)] = 1.0

            est, se, tval, pval, lo, hi = base.linear_combo(
                names, beta, V, weights, df_t
            )

            rows.append({
                "race_group": race,
                "dti_order": idx,
                "dti_category": label,
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


# CHANGE TERMS RELATIVE TO <20

def build_change_terms(names, beta, V, df_t):
    rows = []

    for race in RACES:
        for label in DTI_ORDER[1:]:
            term = dti_term_name(race, label)

            est, se, tval, pval, lo, hi = base.linear_combo(
                names, beta, V, {term: 1.0}, df_t
            )

            rows.append({
                "race_group": race,
                "comparison": f"{label} vs <20",
                "dti_category": label,
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
            dti_term_name(race, label)
            for label in DTI_ORDER[1:]
        ]

        all_terms.extend(terms)

        result = base.joint_zero_test(
            names, beta, V, terms, n_clusters - 1
        )

        rows.append({
            "test": f"{race}: Race × Channel constant across DTI bins",
            "null_hypothesis": (
                f"{race} Race×NonDirect differential is equal across "
                "all eight reported DTI categories"
            ),
            **result,
            "significance": stars(result["p_value"]),
        })

    overall = base.joint_zero_test(
        names, beta, V, all_terms, n_clusters - 1
    )

    rows.append({
        "test": "OVERALL: Race × Channel constant across DTI bins",
        "null_hypothesis": (
            "All 21 Race×NonDirect×DTI change coefficients are jointly zero"
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

    for dti_idx, dti_label in enumerate(DTI_ORDER):
        for rcode, rlabel in race_labels.items():
            for ccode, clabel in [(0, "Direct"), (1, "Non-direct")]:
                m = (
                    reg["dti_order"].eq(dti_idx)
                    & reg["race_code"].eq(rcode)
                    & reg["channel_code"].eq(ccode)
                )

                n = int(m.sum())

                rows.append({
                    "dti_order": dti_idx,
                    "dti_category": dti_label,
                    "race_group": rlabel,
                    "channel": clabel,
                    "n": n,
                    "raw_approval_percent": (
                        100 * reg.loc[m, "approval"].mean()
                        if n > 0
                        else np.nan
                    ),
                })

    return pd.DataFrame(rows)


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))

    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquet files found:\n{DATA_DIR}"
        )

    print("=" * 108)
    print("RACE × CHANNEL × DTI HETEROGENEITY — PREFERRED MEMORY-SAFE LPM")
    print("=" * 108)
    print(f"Parts: {len(parts)}")
    print(f"PyFixest: {base.package_version('pyfixest')}")
    print("Reference race: NH White")
    print("Reference channel: Direct")
    print("Reference DTI category: <20")
    print("SE: CRV1 clustered by lender")
    print(
        "\nDTI categories use HMDA's reported bins. Missing/not-relied-upon "
        "DTI is excluded only from this heterogeneity analysis.\n"
    )

    schema = base.detect_schema(parts)

    print("Building exact verified baseline-M5 sample...")
    reg, audit, category_maps = base.build_data(parts, schema)

    baseline_audit = pd.DataFrame([
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
            "stage": "Preferred complete-case sample before DTI restriction",
            "n": audit["common_sample"],
            "expected_n": EXPECTED_COMMON_SAMPLE_N,
            "pass": audit["common_sample"] == EXPECTED_COMMON_SAMPLE_N,
        },
    ])

    print("\nBaseline sample replication:")
    print(baseline_audit.to_string(index=False))

    if not baseline_audit["pass"].all():
        raise RuntimeError(
            "DTI analysis does not reproduce the verified M5 sample."
        )

    reg, missing_dti_excluded = prepare_dti_data(
        reg, category_maps
    )

    dti_sample_n = len(reg)

    sample_audit = pd.concat([
        baseline_audit,
        pd.DataFrame([{
            "stage": "DTI heterogeneity sample: reported DTI bins only",
            "n": dti_sample_n,
            "expected_n": np.nan,
            "pass": True,
        }])
    ], ignore_index=True)

    mem_gb = reg.memory_usage(deep=True).sum() / (1024 ** 3)

    print(
        f"\nCompact DTI dataframe: {len(reg):,} observations | "
        f"{mem_gb:.2f} GB"
    )

    print("\nSample by DTI category:")
    for idx, label in enumerate(DTI_ORDER):
        n = int(reg["dti_order"].eq(idx).sum())
        print(f"  {label:8s}: {n:,}")

    cells = build_cell_counts(reg)
    formula = build_formula()

    print("\nEstimating Race × Channel × DTI model...")
    fit = base.feols_memorysafe(formula, reg)

    names, beta, V = base.get_coef_vcov(fit)

    n_clusters = int(reg["lender_cluster"].nunique())
    df_t = n_clusters - 1

    effects = build_dti_effects(
        names, beta, V, df_t
    )

    changes = build_change_terms(
        names, beta, V, df_t
    )

    joint = build_joint_tests(
        names, beta, V, n_clusters
    )

    coefs = base.tidy_fit(fit)
    if "p_value" in coefs.columns:
        coefs["significance"] = coefs["p_value"].apply(stars)

    model_info = pd.DataFrame([{
        "preferred_sample_before_dti_restriction": EXPECTED_COMMON_SAMPLE_N,
        "missing_not_relied_upon_dti_excluded": missing_dti_excluded,
        "dti_heterogeneity_sample": dti_sample_n,
        "percent_preferred_sample_retained": 100*dti_sample_n/EXPECTED_COMMON_SAMPLE_N,
        "lender_clusters": n_clusters,
        "lender_year_fe_groups": reg["lender_year_fe"].nunique(),
        "county_fe_groups": reg["county_fe"].nunique(),
        "race_dti_fe_groups": reg["race_dti_fe"].nunique(),
        "channel_dti_fe_groups": reg["channel_dti_fe"].nunique(),
        "dataframe_gb_before_fit": mem_gb,
        "cluster_df": df_t,
        "reference_dti_category": "<20",
        "dti_categories": " | ".join(DTI_ORDER),
        "interpretation_note": (
            "DTI heterogeneity is conditional association, not a causal mechanism. "
            "DTI may reflect borrower selection and underwriting processes."
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
        "dti_race_channel_effects": effects,
        "dti_change_terms": changes,
        "joint_dti_tests": joint,
        "dti_cell_counts": cells,
        "dti_sample_audit": sample_audit,
        "model_coefficients": coefs,
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
        excel_msg = f"Workbook not written ({exc}); CSV outputs were written."

    print("\n" + "=" * 108)
    print("DTI-SPECIFIC ADJUSTED RACE × NON-DIRECT DIFFERENTIALS")
    print("=" * 108)
    print(
        effects[
            [
                "race_group", "dti_category", "estimate_pp",
                "std_error_pp", "p_value", "ci_low_pp",
                "ci_high_pp", "significance",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 108)
    print("JOINT TESTS OF DTI HETEROGENEITY")
    print("=" * 108)
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

    print("\nOverall DTI-heterogeneity conclusion:")
    if overall_p < 0.05:
        print(
            "  Reject constancy at 5%: the adjusted Race × Channel "
            "relationship differs across reported DTI categories somewhere "
            "across the three race/ethnicity groups."
        )
    else:
        print(
            "  Do not reject constancy at 5%: there is not strong joint "
            "evidence that Race × Channel varies across reported DTI bins."
        )

    print(
        "\nDo NOT interpret any DTI pattern as proving a causal affordability "
        "mechanism. This analysis tests conditional heterogeneity only."
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. dti_race_channel_effects.csv")
    print("  2. joint_dti_tests.csv")
    print("  3. dti_change_terms.csv")
    print("  4. dti_cell_counts.csv")
    print("  5. dti_sample_audit.csv")
    print("  6. model_info.csv")


if __name__ == "__main__":
    main()
