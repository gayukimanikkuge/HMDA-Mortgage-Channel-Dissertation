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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "cltv_robustness"
OUTPUT_XLSX = OUTPUT_DIR / "CLTV_ROBUSTNESS.xlsx"

YEARS = list(range(2019, 2025))
EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203

CLTV_CANDIDATES = [
    "cltv_clean",
    "combined_loan_to_value_ratio_clean",
    "combined_loan_to_value_ratio",
    "cltv",
    "combined_ltv",
    "combined_loan_to_value",
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
    s = set(columns)
    for c in candidates:
        if c in s:
            return c
    return None


def clean_numeric(s):
    x = base.clean_string(s)
    x = x.replace({
        "Exempt": pd.NA,
        "exempt": pd.NA,
        "EXEMPT": pd.NA,
    })
    return pd.to_numeric(x, errors="coerce")


def build_data(parts, schema, cltv_col):
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

    readcols = [
        schema["dual50"], schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], schema["income"], schema["dti"],
        schema["age"], schema["sex"], schema["coapp"], schema["loan_amount"],
        schema["property_value"], schema["loan_type"], schema["county"],
        cltv_col,
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
        primary = race.notna()

        d = d.loc[primary].copy()
        race = race.loc[primary]

        if d.empty:
            continue

        approval = base.normalize_approval(d[schema["approval"]])
        nd = base.normalize_channel(d[schema["channel"]])
        year = base.numeric(d[schema["year"]])
        income = base.numeric(d[schema["income"]]).astype("float64")
        lender_raw = base.clean_string(d[schema["lender"]])
        county_raw = base.normalize_county(d[schema["county"]])
        cltv = clean_numeric(d[cltv_col]).astype("float64")

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

        valid_county = (
            county_raw.str.fullmatch(r"\d{5}", na=False)
            & ~county_raw.isin({"00000", "88888", "99999"})
        )

        # Same core complete-case rules as M5, but 2019–2024 only.
        core = (
            approval.isin([0, 1])
            & nd.isin([0, 1])
            & year.isin(YEARS)
            & lender_raw.notna()
            & income.notna() & np.isfinite(income)
            & log_loan.notna() & np.isfinite(log_loan)
            & log_prop.notna() & np.isfinite(log_prop)
            & valid_county
        )

        audit["2019_2024_core"] += int(core.sum())

        if not core.any():
            continue

        d = d.loc[core].copy()
        race = race.loc[core]
        approval = approval.loc[core]
        nd = nd.loc[core]
        year = year.loc[core].astype(np.int16)
        income = income.loc[core]
        log_loan = log_loan.loc[core]
        log_prop = log_prop.loc[core]
        lender_raw = lender_raw.loc[core]
        county_raw = county_raw.loc[core]
        cltv = cltv.loc[core]

        usable_cltv = cltv.notna() & np.isfinite(cltv)
        audit["cltv_complete"] += int(usable_cltv.sum())

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
            d[schema["loan_type"]], maps["loan_type"], np.int16
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

        cltv_arr = cltv.to_numpy(np.float64)
        cltv_asinh = np.full(len(cltv_arr), np.nan, dtype=np.float32)
        good = np.isfinite(cltv_arr)
        cltv_asinh[good] = np.arcsinh(cltv_arr[good]).astype(np.float32)

        x = pd.DataFrame({
            "approval": approval.to_numpy(np.int8),

            "Black_x_NonDirect": (black * ndv).astype(np.int8),
            "Hispanic_x_NonDirect": (hispanic * ndv).astype(np.int8),
            "Asian_x_NonDirect": (asian * ndv).astype(np.int8),

            "income_asinh": np.arcsinh(
                income.to_numpy(np.float64)
            ).astype(np.float32),
            "log_loan_amount": log_loan.to_numpy(np.float32),
            "log_property_value": log_prop.to_numpy(np.float32),
            "cltv_asinh": cltv_asinh,
            "cltv_complete": usable_cltv.to_numpy(np.int8),

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
                f"2019–24 core={audit['2019_2024_core']:,} | "
                f"CLTV complete={audit['cltv_complete']:,}"
            )

        del d, x
        gc.collect()

    if not chunks:
        raise RuntimeError("No observations survived.")

    print("\nCombining compact 2019–2024 dataframe...")
    reg = pd.concat(chunks, ignore_index=True, copy=False)

    del chunks
    gc.collect()

    return reg, audit, maps


def formula(add_cltv=False):
    explicit = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
        "income_asinh",
        "log_loan_amount",
        "log_property_value",
    ]

    if add_cltv:
        explicit.append("cltv_asinh")

    fixed_effects = [
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

    return (
        "approval ~ "
        + " + ".join(explicit)
        + " | "
        + " + ".join(fixed_effects)
    )


def extract(fit, model):
    t = base.tidy_fit(fit)
    t["model"] = model
    out = t.loc[t["term"].isin(HEADLINE_TERMS)].copy()

    if "p_value" in out.columns:
        out["significance"] = out["p_value"].apply(stars)

    keep = [
        c for c in [
            "model", "term", "estimate", "std_error", "p_value",
            "ci_low", "ci_high", "estimate_pp", "std_error_pp",
            "ci_low_pp", "ci_high_pp", "significance",
        ]
        if c in out.columns
    ]
    return out[keep]


def run_model(reg, model_name, mask, add_cltv):
    sub = reg.loc[mask].copy()
    fml = formula(add_cltv=add_cltv)

    print(
        f"\n>>> {model_name}\n"
        f"    N before singleton removal: {len(sub):,}\n"
        f"    lender clusters: {sub['lender_cluster'].nunique():,}"
    )

    fit = base.feols_memorysafe(fml, sub)
    result = extract(fit, model_name)

    info = {
        "model": model_name,
        "observations_before_singleton_removal": len(sub),
        "lender_clusters": int(sub["lender_cluster"].nunique()),
        "lender_year_fe_groups": int(sub["lender_year_fe"].nunique()),
        "county_fe_groups": int(sub["county_fe"].nunique()),
        "cltv_complete_sample": bool(sub["cltv_complete"].eq(1).all()),
        "cltv_control_included": bool(add_cltv),
        "formula": fml,
    }

    print(
        result[
            ["term", "estimate_pp", "std_error_pp", "p_value", "significance"]
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
            f"No final V1 Parquet files found:\n{DATA_DIR}"
        )

    schema_cols = list(
        pq.ParquetFile(parts[0]).schema_arrow.names
    )
    schema = base.detect_schema(parts)

    cltv_col = first_existing(
        schema_cols,
        CLTV_CANDIDATES,
    )

    print("=" * 104)
    print("CLTV ROBUSTNESS — 2019–2024")
    print("=" * 104)
    print(f"Parts: {len(parts)}")
    print(f"Detected CLTV field: {cltv_col}")

    if cltv_col is None:
        candidates = [
            c for c in schema_cols
            if (
                "cltv" in c.lower()
                or (
                    "loan" in c.lower()
                    and "value" in c.lower()
                )
            )
        ]

        print("\nNo known CLTV field was found in final V1.")
        print("Potentially relevant V1 columns:")
        if candidates:
            for c in candidates:
                print(f"  {c}")
        else:
            print("  [none]")

        print(
            "\nSTOPPING SAFELY. The existing robustness package is already "
            "methodologically sufficient; do not rebuild raw HMDA solely for "
            "this optional CLTV test unless you explicitly decide the extra "
            "data work is worthwhile."
        )
        return

    print(
        "\nBuilding preferred 2019–2024 sample. "
        "CLTV is NOT required for Model A and IS required for Models B/C.\n"
    )

    reg, audit, maps = build_data(
        parts,
        schema,
        cltv_col,
    )

    coverage = (
        100 * audit["cltv_complete"] / audit["2019_2024_core"]
        if audit["2019_2024_core"] else np.nan
    )

    sample_audit = pd.DataFrame([
        {
            "stage": "Eligible V1 rows scanned",
            "n": audit["eligible"],
            "share_percent": 100.0,
        },
        {
            "stage": "dual50 all races (all years)",
            "n": audit["dual50"],
            "share_percent": (
                100*audit["dual50"]/audit["eligible"]
                if audit["eligible"] else np.nan
            ),
        },
        {
            "stage": "Preferred four-race complete-case sample, 2019–2024",
            "n": audit["2019_2024_core"],
            "share_percent": 100.0,
        },
        {
            "stage": "Same 2019–2024 sample with usable CLTV",
            "n": audit["cltv_complete"],
            "share_percent": coverage,
        },
    ])

    print("\nCLTV sample audit:")
    print(sample_audit.to_string(index=False))

    results = []
    info = []

    all_mask = pd.Series(True, index=reg.index)
    cltv_mask = reg["cltv_complete"].eq(1)

    r, i = run_model(
        reg,
        "A_2019_2024_No_CLTV_FullPreferred",
        all_mask,
        add_cltv=False,
    )
    results.append(r); info.append(i)

    r, i = run_model(
        reg,
        "B_CLTVComplete_No_CLTV",
        cltv_mask,
        add_cltv=False,
    )
    results.append(r); info.append(i)

    r, i = run_model(
        reg,
        "C_CLTVComplete_With_CLTV",
        cltv_mask,
        add_cltv=True,
    )
    results.append(r); info.append(i)

    results = pd.concat(results, ignore_index=True)
    model_info = pd.DataFrame(info)

    schema_used = pd.DataFrame([
        {"role": k, "detected_column": v if v else "NOT FOUND"}
        for k, v in schema.items()
    ])
    schema_used = pd.concat([
        schema_used,
        pd.DataFrame([{
            "role": "cltv",
            "detected_column": cltv_col,
        }])
    ], ignore_index=True)

    tables = {
        "cltv_robustness_results": results,
        "cltv_model_info": model_info,
        "cltv_sample_audit": sample_audit,
        "schema_used": schema_used,
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
        excel_msg = f"Workbook not written ({exc}); CSVs were written."

    print("\n" + "=" * 104)
    print("CLTV ROBUSTNESS COMPLETE")
    print("=" * 104)

    print(
        results[
            ["model", "term", "estimate_pp", "std_error_pp", "p_value", "significance"]
        ].to_string(index=False)
    )

    print(
        "\nInterpret B → C on the SAME sample. "
        "If the focal interactions remain qualitatively similar, the pooled "
        "Race × Channel conclusion is robust to adding observable CLTV."
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. cltv_robustness_results.csv")
    print("  2. cltv_model_info.csv")
    print("  3. cltv_sample_audit.csv")


if __name__ == "__main__":
    main()
