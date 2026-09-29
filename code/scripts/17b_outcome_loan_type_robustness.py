from __future__ import annotations

import gc
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# VERIFIED HELPERS

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "13b_race_channel_year_heterogeneity_memorysafe.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "\nKeep this script in the SAME folder as:\n"
        "  13b_race_channel_year_heterogeneity_memorysafe.py\n\n"
        f"Expected:\n  {BASE_SCRIPT}\n"
    )

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# CONFIG

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "outcome_loan_type_robustness"
OUTPUT_XLSX = OUTPUT_DIR / "OUTCOME_LOAN_TYPE_ROBUSTNESS.xlsx"

YEARS = list(range(2018, 2025))
EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_PRIMARY_RACE_N = 9_395_768
EXPECTED_MAIN_COMPLETE_N = 9_200_321

ACTION_CANDIDATES = [
    "action_taken",
    "action_taken_clean",
    "action",
    "action_code",
]

HEADLINE_TERMS = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


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


def first_existing(columns, candidates):
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def normalize_loan_type_label(s):
    """
    Convert either cleaned labels or raw HMDA loan-type codes to four readable
    programme categories.
    """
    raw = base.clean_string(s)
    n = pd.to_numeric(raw, errors="coerce")

    out = pd.Series(pd.NA, index=s.index, dtype="string")

    # Raw HMDA:
    # 1 Conventional, 2 FHA, 3 VA, 4 RHS/FSA
    out.loc[n.eq(1)] = "Conventional"
    out.loc[n.eq(2)] = "FHA"
    out.loc[n.eq(3)] = "VA"
    out.loc[n.eq(4)] = "USDA/RHS/FSA"

    x = raw.str.lower().str.replace(r"\s+", " ", regex=True)

    out.loc[x.str.contains("conventional", na=False)] = "Conventional"
    out.loc[x.eq("fha") | x.str.contains("federal housing", na=False)] = "FHA"
    out.loc[x.eq("va") | x.str.contains("veteran", na=False)] = "VA"
    out.loc[
        x.str.contains("rhs", na=False)
        | x.str.contains("fsa", na=False)
        | x.str.contains("usda", na=False)
        | x.str.contains("rural", na=False)
    ] = "USDA/RHS/FSA"

    return out


def build_data(parts, schema, action_col):
    """
    Build one compact preferred dual50 complete-case dataset carrying:
    - main approval outcome
    - raw action code
    - conventional indicator
    - all pooled M5 regressors / fixed effects

    Then each robustness model is just a cheap in-memory subset.
    """

    maps = {
        "dti": {},
        "age": {},
        "sex": {},
        "coapp": {},
        "loan_type": {},
        "lender": {},
        "county": {},
    }

    chunks = []
    audit = Counter()
    loan_type_counts = Counter()
    action_counts = Counter()

    readcols = [
        schema["dual50"], schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], action_col,
        schema["income"], schema["dti"], schema["age"], schema["sex"],
        schema["coapp"], schema["loan_amount"], schema["property_value"],
        schema["loan_type"], schema["county"],
    ]

    if schema["log_loan_amount"] is not None:
        readcols.append(schema["log_loan_amount"])
    if schema["log_property_value"] is not None:
        readcols.append(schema["log_property_value"])

    readcols = list(dict.fromkeys(readcols))

    for pno, part in enumerate(parts, 1):
        d = pd.read_parquet(part, columns=readcols)

        audit["eligible"] += len(d)

        m50 = base.truthy(d[schema["dual50"]])
        audit["dual50"] += int(m50.sum())

        d = d.loc[m50].copy()
        if d.empty:
            continue

        race = base.normalize_race(d[schema["race"]])
        m_primary = race.notna()

        audit["primary_race"] += int(m_primary.sum())

        d = d.loc[m_primary].copy()
        race = race.loc[m_primary]

        if d.empty:
            continue

        approval = base.normalize_approval(d[schema["approval"]])
        action = pd.to_numeric(d[action_col], errors="coerce")
        nd = base.normalize_channel(d[schema["channel"]])
        year = base.numeric(d[schema["year"]])
        income = base.numeric(d[schema["income"]]).astype("float64")
        lender_raw = base.clean_string(d[schema["lender"]])
        county_raw = base.normalize_county(d[schema["county"]])
        loan_type_label = normalize_loan_type_label(d[schema["loan_type"]])

        log_loan = (
            base.numeric(d[schema["log_loan_amount"]]).astype("float64")
            if schema["log_loan_amount"] is not None
            else base.safe_log_positive(d[schema["loan_amount"]])
        )

        log_prop = (
            base.numeric(d[schema["log_property_value"]]).astype("float64")
            if schema["log_property_value"] is not None
            else base.safe_log_positive(d[schema["property_value"]])
        )

        valid = (
            approval.isin([0, 1])
            & action.isin([1, 2, 3])
            & nd.isin([0, 1])
            & year.isin(YEARS)
            & lender_raw.notna()
            & income.notna() & np.isfinite(income)
            & log_loan.notna() & np.isfinite(log_loan)
            & log_prop.notna() & np.isfinite(log_prop)
            & loan_type_label.notna()
        )

        county_ok = (
            county_raw.str.fullmatch(r"\d{5}", na=False)
            & ~county_raw.isin({"00000", "88888", "99999"})
        )
        valid &= county_ok

        audit["main_complete"] += int(valid.sum())

        if not valid.any():
            continue

        d = d.loc[valid].copy()
        race = race.loc[valid]
        approval = approval.loc[valid]
        action = action.loc[valid].astype(np.int8)
        nd = nd.loc[valid]
        year = year.loc[valid].astype(np.int16)
        income = income.loc[valid]
        log_loan = log_loan.loc[valid]
        log_prop = log_prop.loc[valid]
        lender_raw = lender_raw.loc[valid]
        county_raw = county_raw.loc[valid]
        loan_type_label = loan_type_label.loc[valid]

        for val, n in action.value_counts().items():
            action_counts[int(val)] += int(n)

        for label, n in loan_type_label.value_counts().items():
            loan_type_counts[str(label)] += int(n)

        lender = base.encode_incremental(
            lender_raw, maps["lender"], np.int16
        )
        county = base.encode_incremental(
            county_raw, maps["county"], np.int32
        )
        dti = base.encode_incremental(
            d[schema["dti"]], maps["dti"], np.int16
        )
        age = base.encode_incremental(
            d[schema["age"]], maps["age"], np.int16
        )
        sex = base.encode_incremental(
            d[schema["sex"]], maps["sex"], np.int16
        )
        coapp = base.encode_incremental(
            d[schema["coapp"]], maps["coapp"], np.int16
        )
        loan_type = base.encode_incremental(
            loan_type_label, maps["loan_type"], np.int8
        )

        yr = year.to_numpy(np.int16)
        ndv = nd.to_numpy(np.int8)

        race_code = np.zeros(len(race), dtype=np.int8)
        race_code[race.eq("Black").to_numpy()] = 1
        race_code[race.eq("Hispanic").to_numpy()] = 2
        race_code[race.eq("Asian").to_numpy()] = 3

        black = (race_code == 1).astype(np.int8)
        hispanic = (race_code == 2).astype(np.int8)
        asian = (race_code == 3).astype(np.int8)

        action_arr = action.to_numpy(np.int8)
        originated_vs_denied = np.full(len(action_arr), -1, dtype=np.int8)
        originated_vs_denied[action_arr == 1] = 1
        originated_vs_denied[action_arr == 3] = 0

        conventional = (
            loan_type_label.eq("Conventional")
            .to_numpy(np.int8)
        )

        x = pd.DataFrame({
            "approval": approval.to_numpy(np.int8),
            "originated_vs_denied": originated_vs_denied,
            "action_taken": action_arr,
            "conventional": conventional,

            "Black_x_NonDirect": (black * ndv).astype(np.int8),
            "Hispanic_x_NonDirect": (hispanic * ndv).astype(np.int8),
            "Asian_x_NonDirect": (asian * ndv).astype(np.int8),

            "income_asinh": np.arcsinh(
                income.to_numpy(np.float64)
            ).astype(np.float32),
            "log_loan_amount": log_loan.to_numpy(np.float32),
            "log_property_value": log_prop.to_numpy(np.float32),

            "lender_cluster": lender,
            "lender_year_fe": (
                lender.astype(np.int32) * 10 + (yr - 2018)
            ).astype(np.int32),
            "county_fe": county,

            "race_fe": race_code,
            "channel_fe": ndv,
            "dti_fe": dti,
            "age_fe": age,
            "sex_fe": sex,
            "coapp_fe": coapp,
            "loan_type_fe": loan_type,
        })

        chunks.append(x)

        if pno % 50 == 0 or pno == len(parts):
            print(
                f"  parts {pno:>3}/{len(parts)} | "
                f"eligible={audit['eligible']:,} | "
                f"dual50={audit['dual50']:,} | "
                f"four-race={audit['primary_race']:,} | "
                f"complete={audit['main_complete']:,}"
            )

        del d, x
        gc.collect()

    if not chunks:
        raise RuntimeError("No observations survived.")

    print("\nCombining compact robustness dataset...")
    reg = pd.concat(chunks, ignore_index=True, copy=False)

    del chunks
    gc.collect()

    return reg, audit, maps, action_counts, loan_type_counts


def formula(outcome, include_loan_type_fe=True):
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
        "coapp_fe",
    ]

    if include_loan_type_fe:
        fixed_effects.append("loan_type_fe")

    return (
        f"{outcome} ~ "
        + " + ".join(explicit)
        + " | "
        + " + ".join(fixed_effects)
    )


def extract(fit, model_name):
    t = base.tidy_fit(fit)
    t["model"] = model_name

    out = t.loc[t["term"].isin(HEADLINE_TERMS)].copy()

    if "p_value" in out.columns:
        out["significance"] = out["p_value"].apply(stars)

    keep = [
        c for c in [
            "model", "term",
            "estimate", "std_error", "p_value",
            "ci_low", "ci_high",
            "estimate_pp", "std_error_pp",
            "ci_low_pp", "ci_high_pp",
            "significance",
        ]
        if c in out.columns
    ]

    return out[keep]


def run_model(reg, model_name, outcome, mask, include_loan_type_fe):
    sub = reg.loc[mask].copy()

    fml = formula(
        outcome,
        include_loan_type_fe=include_loan_type_fe,
    )

    print(
        f"\n>>> {model_name}\n"
        f"    N before FE singleton removal: {len(sub):,}\n"
        f"    lenders: {sub['lender_cluster'].nunique():,}"
    )

    fit = base.feols_memorysafe(fml, sub)
    result = extract(fit, model_name)

    info = {
        "model": model_name,
        "outcome": outcome,
        "observations_before_singleton_removal": len(sub),
        "lender_clusters": int(sub["lender_cluster"].nunique()),
        "lender_year_fe_groups": int(sub["lender_year_fe"].nunique()),
        "county_fe_groups": int(sub["county_fe"].nunique()),
        "conventional_only": bool(sub["conventional"].eq(1).all()),
        "action2_excluded": bool(
            outcome == "originated_vs_denied"
        ),
        "formula": fml,
    }

    print(
        result[
            [
                "term", "estimate_pp", "std_error_pp",
                "p_value", "significance",
            ]
        ].to_string(index=False)
    )

    del fit, sub
    gc.collect()

    return result, info


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquets found:\n{DATA_DIR}"
        )

    schema_cols = list(
        pq.ParquetFile(parts[0]).schema_arrow.names
    )

    schema = base.detect_schema(parts)

    action_col = first_existing(
        schema_cols,
        ACTION_CANDIDATES,
    )

    print("=" * 106)
    print("OUTCOME + LOAN-TYPE ROBUSTNESS")
    print("=" * 106)
    print(f"Parts: {len(parts)}")
    print(f"PyFixest: {base.package_version('pyfixest')}")
    print(f"Detected action field: {action_col}")
    print(f"Detected loan-type field: {schema['loan_type']}")
    print("SE: CRV1 clustered by lender\n")

    if action_col is None:
        print("\nAvailable V1 columns:")
        for c in schema_cols:
            print(f"  {c}")

        raise KeyError(
            "\nNo action-taken field was found in final V1. "
            "The originated-vs-denied robustness cannot be reconstructed "
            "from the binary approval indicator because action 1 and action 2 "
            "are both coded approved. Do not guess. If this occurs, send me "
            "the printed V1 column list and we will use the raw/base source safely."
        )

    print("Building exact preferred dual50 complete-case sample...")
    reg, audit, maps, action_counts, loan_type_counts = build_data(
        parts,
        schema,
        action_col,
    )

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
            "stage": "dual50 + four headline groups",
            "n": audit["primary_race"],
            "expected_n": EXPECTED_PRIMARY_RACE_N,
            "pass": audit["primary_race"] == EXPECTED_PRIMARY_RACE_N,
        },
        {
            "stage": "Main complete-case sample",
            "n": audit["main_complete"],
            "expected_n": EXPECTED_MAIN_COMPLETE_N,
            "pass": audit["main_complete"] == EXPECTED_MAIN_COMPLETE_N,
        },
    ])

    print("\nSample audit:")
    print(sample_audit.to_string(index=False))

    if not sample_audit["pass"].all():
        raise RuntimeError(
            "Main sample did not reproduce baseline M5 exactly. "
            "Do not interpret robustness results."
        )

    action_table = pd.DataFrame([
        {
            "action_taken": action,
            "label": {
                1: "Originated",
                2: "Approved, not accepted",
                3: "Denied",
            }.get(action, "Other"),
            "n": n,
        }
        for action, n in sorted(action_counts.items())
    ])

    loan_type_table = pd.DataFrame([
        {
            "loan_type": label,
            "n": n,
            "percent": 100*n/len(reg),
        }
        for label, n in sorted(
            loan_type_counts.items(),
            key=lambda z: -z[1],
        )
    ])

    print("\nAction distribution in preferred complete-case sample:")
    print(action_table.to_string(index=False))

    print("\nLoan-type distribution:")
    print(loan_type_table.to_string(index=False))

    results = []
    info = []

    # Reference model, recomputed to ensure identical parameterisation.
    r, i = run_model(
        reg,
        "Reference_Main_Approval_AllLoanTypes",
        "approval",
        pd.Series(True, index=reg.index),
        include_loan_type_fe=True,
    )
    results.append(r)
    info.append(i)

    # C. Alternative outcome.
    m_action13 = reg["originated_vs_denied"].isin([0, 1])
    r, i = run_model(
        reg,
        "Robustness_Originated_vs_Denied",
        "originated_vs_denied",
        m_action13,
        include_loan_type_fe=True,
    )
    results.append(r)
    info.append(i)

    # D. Conventional-only.
    m_conv = reg["conventional"].eq(1)
    r, i = run_model(
        reg,
        "Robustness_Conventional_Only",
        "approval",
        m_conv,
        include_loan_type_fe=False,
    )
    results.append(r)
    info.append(i)

    # Intersection check.
    r, i = run_model(
        reg,
        "Robustness_Conventional_Originated_vs_Denied",
        "originated_vs_denied",
        m_conv & m_action13,
        include_loan_type_fe=False,
    )
    results.append(r)
    info.append(i)

    results = pd.concat(results, ignore_index=True)
    model_info = pd.DataFrame(info)

    category_rows = []
    for var, mapping in maps.items():
        for label, code_ in sorted(
            mapping.items(),
            key=lambda z: z[1],
        ):
            category_rows.append({
                "variable": var,
                "code_used": code_,
                "cleaned_label": label,
            })
    category_levels = pd.DataFrame(category_rows)

    schema_used = pd.DataFrame([
        {
            "role": k,
            "detected_column": v if v else "NOT FOUND",
        }
        for k, v in schema.items()
    ])
    schema_used = pd.concat([
        schema_used,
        pd.DataFrame([{
            "role": "action_taken",
            "detected_column": action_col,
        }])
    ], ignore_index=True)

    tables = {
        "robustness_headline_results": results,
        "robustness_model_info": model_info,
        "outcome_sample_audit": sample_audit,
        "action_counts": action_table,
        "loan_type_counts": loan_type_table,
        "schema_used": schema_used,
        "category_levels": category_levels,
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
            f"Workbook not written ({exc}); CSVs were written."
        )

    print("\n" + "=" * 106)
    print("ROBUSTNESS BLOCK C+D COMPLETE")
    print("=" * 106)

    print(
        results[
            [
                "model", "term", "estimate_pp",
                "std_error_pp", "p_value", "significance",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. robustness_headline_results.csv")
    print("  2. robustness_model_info.csv")
    print("  3. outcome_sample_audit.csv")
    print("  4. action_counts.csv")
    print("  5. loan_type_counts.csv")


if __name__ == "__main__":
    main()
