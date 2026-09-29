from __future__ import annotations

import gc
import inspect
import math
import re
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "pyarrow is required. It is normally already installed with pandas/Parquet support."
    ) from exc

try:
    import pyfixest as pf
except ImportError as exc:
    raise ImportError(
        "\nPyFixest is required for the high-dimensional fixed-effect regressions.\n"
        "In the SAME Spyder console, run once:\n\n"
        "    %pip install pyfixest\n\n"
        "Then rerun this script."
    ) from exc

# 1. CONFIGURATION

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "baseline_race_channel_regressions"
OUTPUT_XLSX = OUTPUT_DIR / "BASELINE_RACE_CHANNEL_REGRESSIONS.xlsx"

MAX_PARTS = None

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203

YEARS = set(range(2018, 2025))

PREFERRED_COLUMNS = {
    "dual50": ["dual50"],
    "year": ["year", "activity_year"],
    "lender": ["lei_clean", "lei", "lender_lei"],
    "race": ["race_group", "race_ethnicity", "race_category", "race_ethnicity_group"],
    "channel": ["channel", "submission_channel", "submission_of_application", "non_direct"],
    "approval": ["approval", "approved", "approval_binary", "approved_binary"],
    "income": ["income_thousands", "income_clean", "income", "income_numeric"],
    "dti": ["dti_category", "dti_band", "debt_to_income_category"],
    "age": ["age_category", "applicant_age_category"],
    "sex": ["sex_category", "applicant_sex_category"],
    "coapp": ["coapplicant_present", "co_applicant_present", "coapp_present"],
    "loan_amount": ["loan_amount_clean", "loan_amount"],
    "property_value": ["property_value_clean", "property_value"],
    "loan_type": ["loan_type_clean", "loan_type", "loan_type_category"],
    "county": ["county_clean", "county_code", "county"],
    "log_loan_amount": ["log_loan_amount"],
    "log_property_value": ["log_property_value"],
}

HEADLINE_TERMS = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


# 2. BASIC HELPERS

def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def clean_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.replace({
        "": pd.NA,
        "nan": pd.NA,
        "NaN": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "NULL": pd.NA,
    })


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().any():
        return n.eq(1)
    x = clean_string(s).str.lower()
    return x.isin({"true", "t", "yes", "y", "1"})


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_race(s: pd.Series) -> pd.Series:
    x = clean_string(s).str.lower()
    x = x.str.replace(r"\s+", " ", regex=True)

    mapping = {
        "white": "White",
        "non-hispanic white": "White",
        "non hispanic white": "White",
        "nh white": "White",

        "black": "Black",
        "black or african american": "Black",
        "non-hispanic black": "Black",
        "non hispanic black": "Black",
        "nh black": "Black",

        "hispanic": "Hispanic",
        "hispanic/latino": "Hispanic",
        "hispanic / latino": "Hispanic",
        "hispanic or latino": "Hispanic",
        "latino": "Hispanic",

        "asian": "Asian",
        "non-hispanic asian": "Asian",
        "non hispanic asian": "Asian",
        "nh asian": "Asian",
    }
    return x.map(mapping).astype("string")


def normalize_channel(s: pd.Series) -> pd.Series:
    """Return 0 Direct, 1 Non-direct, NA otherwise."""
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")

    # Raw HMDA submission_of_application encoding
    out.loc[n.eq(1)] = 0
    out.loc[n.eq(2)] = 1

    x = clean_string(s).str.lower()
    x = x.str.replace(r"\s+", " ", regex=True)

    out.loc[x.isin({"direct", "submitted directly"})] = 0
    out.loc[x.isin({
        "non-direct", "non direct", "nondirect", "not direct",
        "not submitted directly", "intermediated"
    })] = 1

    return out


def normalize_approval(s: pd.Series) -> pd.Series:
    """Return 0/1 approval indicator or NA."""
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    valid_binary = n.isin([0, 1])
    out.loc[valid_binary] = n.loc[valid_binary]

    x = clean_string(s).str.lower()
    out.loc[x.isin({"approved", "approval", "yes", "true"})] = 1.0
    out.loc[x.isin({"denied", "no", "false"})] = 0.0
    return out


def valid_key(s: pd.Series) -> pd.Series:
    x = clean_string(s)
    return x.notna()


def normalize_county(s: pd.Series) -> pd.Series:
    x = clean_string(s)
    x = x.str.replace(r"\.0$", "", regex=True)
    numeric_mask = x.str.fullmatch(r"\d{1,5}", na=False)
    x.loc[numeric_mask] = x.loc[numeric_mask].str.zfill(5)
    return x


def safe_log_positive(s: pd.Series) -> pd.Series:
    x = numeric(s).astype("float64")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    good = x.notna() & np.isfinite(x) & x.gt(0)
    out.loc[good] = np.log(x.loc[good])
    return out


def category_labels(s: pd.Series) -> pd.Series:
    x = clean_string(s)
    return x.fillna("Missing / Not reported")


def encode_incremental(
    s: pd.Series,
    mapping: dict[str, int],
    dtype=np.int16,
) -> np.ndarray:
    labels = category_labels(s)
    local_codes, uniques = pd.factorize(labels, sort=False)

    global_for_local = np.empty(len(uniques), dtype=np.int32)
    for j, label in enumerate(uniques.astype(str)):
        if label not in mapping:
            mapping[label] = len(mapping)
        global_for_local[j] = mapping[label]

    result = global_for_local[local_codes]
    return result.astype(dtype, copy=False)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


# 3. SCHEMA DETECTION

def detect_schema(parts: list[Path]) -> tuple[list[str], dict[str, str | None]]:
    cols = list(pq.ParquetFile(parts[0]).schema_arrow.names)

    schema = {
        role: first_existing(cols, candidates)
        for role, candidates in PREFERRED_COLUMNS.items()
    }

    required_roles = [
        "dual50", "year", "lender", "race", "channel", "approval",
        "income", "dti", "age", "sex", "coapp",
        "loan_amount", "property_value", "loan_type", "county",
    ]

    missing = [r for r in required_roles if schema[r] is None]
    if missing:
        raise KeyError(
            "Required cleaned regression fields were not found in the final V1 schema:\n"
            + "\n".join(f"  - {r}" for r in missing)
            + "\n\nNo substitute has been guessed. Inspect the V1 schema before proceeding."
        )

    return cols, schema


# 4. BUILD COMPACT COMMON ESTIMATION SAMPLE

def build_estimation_data(parts: list[Path], schema: dict[str, str | None]):
    category_maps: dict[str, dict[str, int]] = {
        "dti": {},
        "age": {},
        "sex": {},
        "coapp": {},
        "loan_type": {},
        "lender": {},
        "county": {},
    }

    chunks: list[pd.DataFrame] = []
    race_source_counts = Counter()
    race_normalized_counts = Counter()

    audit = Counter()
    preflight_rows = []

    # Read only columns genuinely required.
    readcols = [
        schema["dual50"], schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], schema["income"], schema["dti"],
        schema["age"], schema["sex"], schema["coapp"], schema["loan_amount"],
        schema["property_value"], schema["loan_type"], schema["county"],
    ]

    # Reuse precomputed logs if they exist.
    if schema["log_loan_amount"] is not None:
        readcols.append(schema["log_loan_amount"])
    if schema["log_property_value"] is not None:
        readcols.append(schema["log_property_value"])

    readcols = list(dict.fromkeys(readcols))

    for pno, part in enumerate(parts, start=1):
        d = pd.read_parquet(part, columns=readcols)
        audit["eligible_scanned"] += len(d)

        # Preferred support restriction.
        m_dual = truthy(d[schema["dual50"]])
        audit["dual50_all_races"] += int(m_dual.sum())
        d = d.loc[m_dual].copy()
        if d.empty:
            continue

        # Race normalization: exact headline labels only.
        raw_race = category_labels(d[schema["race"]])
        for label, n in raw_race.value_counts().items():
            race_source_counts[str(label)] += int(n)

        race = normalize_race(d[schema["race"]])
        m_race = race.notna()
        audit["dual50_primary_race"] += int(m_race.sum())
        d = d.loc[m_race].copy()
        race = race.loc[m_race]

        if d.empty:
            continue

        for label, n in race.value_counts().items():
            race_normalized_counts[str(label)] += int(n)

        # Outcome and channel.
        approval = normalize_approval(d[schema["approval"]])
        nondirect = normalize_channel(d[schema["channel"]])

        # Core numeric / key variables.
        year = numeric(d[schema["year"]])
        income = numeric(d[schema["income"]]).astype("float64")
        loan_amt = numeric(d[schema["loan_amount"]]).astype("float64")
        prop_val = numeric(d[schema["property_value"]]).astype("float64")
        lender_raw = clean_string(d[schema["lender"]])
        county_raw = normalize_county(d[schema["county"]])

        # Use existing safe log fields if available; otherwise reconstruct.
        if schema["log_loan_amount"] is not None:
            log_loan = numeric(d[schema["log_loan_amount"]]).astype("float64")
        else:
            log_loan = safe_log_positive(d[schema["loan_amount"]])

        if schema["log_property_value"] is not None:
            log_prop = numeric(d[schema["log_property_value"]]).astype("float64")
        else:
            log_prop = safe_log_positive(d[schema["property_value"]])

        # Common estimation sample across M1–M5.
        valid_approval = approval.isin([0, 1])
        valid_channel = nondirect.isin([0, 1])
        valid_year = year.isin(YEARS)
        valid_lender = lender_raw.notna()

        valid_income = income.notna() & np.isfinite(income)
        valid_log_loan = log_loan.notna() & np.isfinite(log_loan)
        valid_log_prop = log_prop.notna() & np.isfinite(log_prop)

        county_regex = county_raw.str.fullmatch(r"\d{5}", na=False)
        county_special = county_raw.isin({"00000", "88888", "99999"})
        valid_county = county_regex & ~county_special

        audit["missing_or_invalid_approval"] += int((~valid_approval).sum())
        audit["missing_or_invalid_channel"] += int((~valid_channel).sum())
        audit["missing_or_invalid_year"] += int((~valid_year).sum())
        audit["missing_lender"] += int((~valid_lender).sum())
        audit["missing_income"] += int((~valid_income).sum())
        audit["missing_or_nonpositive_loan_amount"] += int((~valid_log_loan).sum())
        audit["missing_or_nonpositive_property_value"] += int((~valid_log_prop).sum())
        audit["missing_or_invalid_county"] += int((~valid_county).sum())

        keep = (
            valid_approval
            & valid_channel
            & valid_year
            & valid_lender
            & valid_income
            & valid_log_loan
            & valid_log_prop
            & valid_county
        )

        audit["common_complete_case"] += int(keep.sum())
        if not keep.any():
            continue

        d = d.loc[keep].copy()
        race = race.loc[keep]
        approval = approval.loc[keep]
        nondirect = nondirect.loc[keep]
        year = year.loc[keep].astype(np.int16)
        income = income.loc[keep]
        log_loan = log_loan.loc[keep]
        log_prop = log_prop.loc[keep]
        lender_raw = lender_raw.loc[keep]
        county_raw = county_raw.loc[keep]

        # Stable compact categorical encodings across parts.
        lender_code = encode_incremental(
            lender_raw, category_maps["lender"], dtype=np.int16
        )
        county_code = encode_incremental(
            county_raw, category_maps["county"], dtype=np.int32
        )
        dti_code = encode_incremental(
            d[schema["dti"]], category_maps["dti"], dtype=np.int16
        )
        age_code = encode_incremental(
            d[schema["age"]], category_maps["age"], dtype=np.int16
        )
        sex_code = encode_incremental(
            d[schema["sex"]], category_maps["sex"], dtype=np.int16
        )
        coapp_code = encode_incremental(
            d[schema["coapp"]], category_maps["coapp"], dtype=np.int16
        )
        loan_type_code = encode_incremental(
            d[schema["loan_type"]], category_maps["loan_type"], dtype=np.int16
        )

        #stable key dummiies
        black = race.eq("Black").to_numpy(dtype=np.int8)
        hispanic = race.eq("Hispanic").to_numpy(dtype=np.int8)
        asian = race.eq("Asian").to_numpy(dtype=np.int8)
        nd = nondirect.to_numpy(dtype=np.int8)

        out = pd.DataFrame({
            "approval": approval.to_numpy(dtype=np.int8),

            "Black": black,
            "Hispanic": hispanic,
            "Asian": asian,
            "NonDirect": nd,

            "Black_x_NonDirect": (black * nd).astype(np.int8),
            "Hispanic_x_NonDirect": (hispanic * nd).astype(np.int8),
            "Asian_x_NonDirect": (asian * nd).astype(np.int8),

            # income_thousands is intentionally transformed on its stored scale.
            "income_asinh": np.arcsinh(
                income.to_numpy(dtype=np.float64)
            ).astype(np.float32),

            "dti_code": dti_code,
            "age_code": age_code,
            "sex_code": sex_code,
            "coapp_code": coapp_code,

            "log_loan_amount": log_loan.to_numpy(dtype=np.float32),
            "log_property_value": log_prop.to_numpy(dtype=np.float32),
            "loan_type_code": loan_type_code,

            "lender_cluster": lender_code,
            "county_fe": county_code,
            "year": year.to_numpy(dtype=np.int16),
        })

        chunks.append(out)

        if pno % 50 == 0 or pno == len(parts):
            print(
                f"  parts {pno:>3}/{len(parts)} | "
                f"eligible scanned={audit['eligible_scanned']:,} | "
                f"dual50={audit['dual50_all_races']:,} | "
                f"primary-race={audit['dual50_primary_race']:,} | "
                f"common sample={audit['common_complete_case']:,}"
            )

        del d, out
        gc.collect()

    if not chunks:
        raise RuntimeError("No observations survived the preferred sample filters.")

    print("\nCombining compact regression chunks...")
    reg = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()

    # Unique lender x year FE without creating millions of long strings.
    # lender_code is stable across chunks; 10 is safely larger than the 7-year span.
    reg["lender_year_fe"] = (
        reg["lender_cluster"].astype(np.int32) * 10
        + (reg["year"].astype(np.int32) - 2018)
    ).astype(np.int32)

    # Final preflight statistics.
    preflight_rows.extend([
        {
            "check": "Eligible rows scanned",
            "value": int(audit["eligible_scanned"]),
            "status": (
                "PASS"
                if MAX_PARTS is not None or audit["eligible_scanned"] == EXPECTED_ELIGIBLE_N
                else "CHECK"
            ),
            "note": f"Audited full V1 expectation = {EXPECTED_ELIGIBLE_N:,}",
        },
        {
            "check": "dual50 rows before race restriction",
            "value": int(audit["dual50_all_races"]),
            "status": (
                "PASS"
                if MAX_PARTS is not None or audit["dual50_all_races"] == EXPECTED_DUAL50_N
                else "CHECK"
            ),
            "note": f"Representativeness audit expectation = {EXPECTED_DUAL50_N:,}",
        },
        {
            "check": "Common complete-case estimation sample",
            "value": int(len(reg)),
            "status": "PASS" if len(reg) > 1_000_000 or MAX_PARTS is not None else "CHECK",
            "note": "Same base estimation sample is used for the regression ladder.",
        },
        {
            "check": "Unique lender clusters",
            "value": int(reg["lender_cluster"].nunique()),
            "status": "PASS" if reg["lender_cluster"].nunique() >= 50 else "CHECK",
            "note": "CRV1 standard errors are clustered at lender level.",
        },
        {
            "check": "Unique lender-year fixed effects",
            "value": int(reg["lender_year_fe"].nunique()),
            "status": "PASS",
            "note": "Preferred design identifies within lender-year.",
        },
        {
            "check": "Unique county fixed effects",
            "value": int(reg["county_fe"].nunique()),
            "status": "PASS",
            "note": "County strings had to be valid 5-digit non-special codes.",
        },
        {
            "check": "Outcome mean",
            "value": float(reg["approval"].mean()),
            "status": "PASS" if 0 < reg["approval"].mean() < 1 else "CHECK",
            "note": "Approval indicator should contain both approvals and denials.",
        },
        {
            "check": "Non-direct share",
            "value": float(reg["NonDirect"].mean()),
            "status": "PASS" if 0 < reg["NonDirect"].mean() < 1 else "CHECK",
            "note": "Both channels must be represented.",
        },
    ])

    sensible_caps = {
        "dti": 50,
        "age": 30,
        "sex": 20,
        "coapp": 20,
        "loan_type": 20,
    }
    for key, cap in sensible_caps.items():
        levels = len(category_maps[key])
        preflight_rows.append({
            "check": f"{key} category levels",
            "value": levels,
            "status": "PASS" if levels <= cap else "CHECK",
            "note": f"Expected a cleaned categorical control; warning cap={cap}.",
        })

    return (
        reg,
        audit,
        category_maps,
        race_source_counts,
        race_normalized_counts,
        pd.DataFrame(preflight_rows),
    )


# 5. REGRESSION HELPERS

def feols_low_memory(formula: str, data: pd.DataFrame):
    """
    Call PyFixest with explicit lender clustering and use low-memory options
    only when the installed PyFixest version supports them.
    """
    sig = inspect.signature(pf.feols)
    params = sig.parameters

    kwargs = {
        "fml": formula,
        "data": data,
        "vcov": {"CRV1": "lender_cluster"},
    }

    optional = {
        "copy_data": False,
        "store_data": False,
        "lean": True,
        "fixef_rm": "singleton",
    }
    for key, value_ in optional.items():
        if key in params:
            kwargs[key] = value_

    return pf.feols(**kwargs)


def tidy_result(fit, model_name: str) -> pd.DataFrame:
    t = fit.tidy().copy()

    # PyFixest returns coefficient names in the index.
    if "Coefficient" not in t.columns:
        t = t.reset_index()
        first = t.columns[0]
        t = t.rename(columns={first: "term"})
    else:
        t = t.rename(columns={"Coefficient": "term"})

    if "term" not in t.columns:
        t.insert(0, "term", t.index.astype(str))

    rename = {
        "Estimate": "estimate",
        "Std. Error": "std_error",
        "t value": "t_value",
        "Pr(>|t|)": "p_value",
        "2.5%": "ci_low",
        "97.5%": "ci_high",
    }
    t = t.rename(columns=rename)
    t.insert(0, "model", model_name)

    # Add percentage-point versions for easy LPM interpretation.
    for c in ["estimate", "std_error", "ci_low", "ci_high"]:
        if c in t.columns:
            t[f"{c}_pp"] = pd.to_numeric(t[c], errors="coerce") * 100.0

    return t.reset_index(drop=True)


def get_first_attr(obj, names, default=np.nan):
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            try:
                if callable(val):
                    val = val()
            except TypeError:
                pass
            if np.isscalar(val):
                return val
    return default


def model_stats(fit, model_name: str, formula: str) -> dict:
    return {
        "model": model_name,
        "formula": formula,
        "observations": get_first_attr(fit, ["_N", "nobs", "_nobs"]),
        "r_squared": get_first_attr(fit, ["_r2", "r2", "_r2_regular"]),
        "within_r_squared": get_first_attr(
            fit, ["_r2_within", "r2_within", "_wr2"]
        ),
        "cluster": "lender (CRV1)",
        "pyfixest_version": package_version("pyfixest"),
    }


# 6. MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No final V1 Parquet files found in:\n{DATA_DIR}")

    if MAX_PARTS is not None:
        parts = parts[:MAX_PARTS]

    print("=" * 100)
    print("BASELINE RACE × CHANNEL REGRESSIONS — HMDA 2018–2024")
    print("=" * 100)
    print(f"Data folder: {DATA_DIR}")
    print(f"Parquet parts to read: {len(parts)}")
    print(f"PyFixest version: {package_version('pyfixest')}")
    print("Preferred support sample: dual50")
    print("Primary groups: White (ref), Black, Hispanic/Latino, Asian")
    print("Inference: lender-clustered CRV1 standard errors")
    print()

    schema_cols, schema = detect_schema(parts)

    print("Detected V1 regression schema:")
    for role, col in schema.items():
        print(f"  {role:20s}: {col}")

    schema_used = pd.DataFrame([
        {"role": k, "detected_column": v if v is not None else "NOT FOUND"}
        for k, v in schema.items()
    ])

    print("\nBuilding compact common estimation sample...")
    (
        reg,
        audit,
        category_maps,
        race_source_counts,
        race_normalized_counts,
        preflight,
    ) = build_estimation_data(parts, schema)

    # Sample / category auditing
    sample_audit_rows = [
        {"stage": "Eligible V1 rows scanned", "n": int(audit["eligible_scanned"])},
        {"stage": "dual50, all race groups", "n": int(audit["dual50_all_races"])},
        {"stage": "dual50 + four headline race groups", "n": int(audit["dual50_primary_race"])},
        {"stage": "Common complete-case regression sample", "n": int(audit["common_complete_case"])},
    ]
    for key in [
        "missing_or_invalid_approval",
        "missing_or_invalid_channel",
        "missing_or_invalid_year",
        "missing_lender",
        "missing_income",
        "missing_or_nonpositive_loan_amount",
        "missing_or_nonpositive_property_value",
        "missing_or_invalid_county",
    ]:
        sample_audit_rows.append({"stage": key, "n": int(audit[key])})

    sample_audit = pd.DataFrame(sample_audit_rows)

    # Race labels seen brefore headline filtering: helps verify no accidental substring assignment of AIAN/NHPI/multiracial into the headline groups.
    race_labels = pd.DataFrame([
        {"source_race_label": k, "dual50_n": v}
        for k, v in race_source_counts.items()
    ]).sort_values("dual50_n", ascending=False)

    normalized_race = pd.DataFrame([
        {"headline_group": k, "n": v}
        for k, v in race_normalized_counts.items()
    ]).sort_values("n", ascending=False)

    category_rows = []
    for var, mapping in category_maps.items():
        for label, code in sorted(mapping.items(), key=lambda kv: kv[1]):
            category_rows.append({
                "variable": var,
                "code_used_in_regression": code,
                "original_cleaned_label": label,
            })
    category_levels = pd.DataFrame(category_rows)

    # Preflight display.
    print("\n" + "=" * 100)
    print("PREFLIGHT CHECKS")
    print("=" * 100)
    print(preflight.to_string(index=False))

    checks = preflight["status"].astype(str).eq("CHECK")
    if checks.any():
        print(
            "\nNOTE: One or more preflight items are marked CHECK. "
            "The script will continue because these are diagnostics rather than "
            "automatic evidence of a coding error. Review them with the exported output."
        )

    print("\nHeadline estimation-sample race counts:")
    print(normalized_race.to_string(index=False))

    print(
        f"\nCompact regression dataframe: {len(reg):,} observations, "
        f"{reg.memory_usage(deep=True).sum() / (1024**3):.2f} GB in memory."
    )

    # Model ladder
    core = (
        "approval ~ "
        "Black + Hispanic + Asian + NonDirect + "
        "Black_x_NonDirect + Hispanic_x_NonDirect + Asian_x_NonDirect"
    )

    borrower_controls = (
        " + income_asinh"
        " + C(dti_code)"
        " + C(age_code)"
        " + C(sex_code)"
        " + C(coapp_code)"
    )

    loan_controls = (
        " + log_loan_amount"
        " + log_property_value"
        " + C(loan_type_code)"
    )

    models = [
        ("M1_Core_Race_x_Channel", core),
        ("M2_Borrower_Controls", core + borrower_controls),
        ("M3_Borrower_and_Loan_Controls", core + borrower_controls + loan_controls),
        (
            "M4_LenderYear_FE",
            core + borrower_controls + loan_controls + " | lender_year_fe",
        ),
        (
            "M5_Preferred_LenderYear_County_FE",
            core + borrower_controls + loan_controls + " | lender_year_fe + county_fe",
        ),
    ]

    all_tidy = []
    summaries = []

    print("\n" + "=" * 100)
    print("ESTIMATING REGRESSION LADDER")
    print("=" * 100)

    for model_name, formula in models:
        print(f"\n>>> {model_name}")
        print(formula)

        fit = feols_low_memory(formula, reg)

        t = tidy_result(fit, model_name)
        all_tidy.append(t)
        summaries.append(model_stats(fit, model_name, formula))

        h = t.loc[t["term"].isin(HEADLINE_TERMS)].copy()
        if not h.empty:
            cols = [
                c for c in [
                    "term", "estimate_pp", "std_error_pp",
                    "p_value", "ci_low_pp", "ci_high_pp"
                ]
                if c in h.columns
            ]
            print("\nHeadline Race × NonDirect interactions (percentage points):")
            print(h[cols].to_string(index=False))

        # Release the model so that the next specification can be estimated
        del fit, t
        gc.collect()

    regression_results = pd.concat(all_tidy, ignore_index=True)
    del all_tidy
    gc.collect()

    model_summary = pd.DataFrame(summaries)

    headline = regression_results.loc[
        regression_results["term"].isin(HEADLINE_TERMS)
    ].copy()

    def interaction_interpretation(row):
        est = pd.to_numeric(row.get("estimate", np.nan), errors="coerce")
        if pd.isna(est):
            return ""
        group = str(row["term"]).split("_x_")[0]
        if est < 0:
            direction = "wider (more negative) in non-direct"
        elif est > 0:
            direction = "narrower (less negative) in non-direct"
        else:
            direction = "no difference across channels"
        return (
            f"For {group} relative to White, the adjusted approval gap is "
            f"{direction}. This is an association, not a causal channel effect."
        )

    if not headline.empty:
        headline["directional_interpretation"] = headline.apply(
            interaction_interpretation, axis=1
        )

    # Writing outputs
    tables = {
        "regression_results_long": regression_results,
        "headline_interactions": headline,
        "model_summary": model_summary,
        "sample_audit": sample_audit,
        "schema_used": schema_used,
        "category_levels": category_levels,
        "preflight_checks": preflight,
        "source_race_labels": race_labels,
        "headline_race_counts": normalized_race,
    }

    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    excel_msg = ""
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = (
            f"Excel workbook was not written ({exc}); all CSV outputs were written."
        )

    print("\n" + "=" * 100)
    print("BASELINE REGRESSION RUN COMPLETE")
    print("=" * 100)

    preferred = headline.loc[
        headline["model"].eq("M5_Preferred_LenderYear_County_FE")
    ].copy()

    if not preferred.empty:
        show = [
            c for c in [
                "term", "estimate_pp", "std_error_pp",
                "p_value", "ci_low_pp", "ci_high_pp"
            ]
            if c in preferred.columns
        ]
        print("\nPREFERRED MODEL — HEADLINE INTERACTIONS")
        print(preferred[show].to_string(index=False))

        print("\nInterpretation key:")
        print("  Negative interaction -> racial/ethnic approval gap is wider in non-direct.")
        print("  Positive interaction -> racial/ethnic approval gap is narrower in non-direct.")
        print("  Always interpret as an adjusted association, NOT a causal channel effect.")

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")
    print("\nPLEASE SEND BACK THESE FIRST:")
    print("  1. headline_interactions.csv")
    print("  2. model_summary.csv")
    print("  3. sample_audit.csv")
    print("  4. preflight_checks.csv")
    print("  5. source_race_labels.csv")
    print("  6. category_levels.csv")
    print("\nOnce those are checked, the next step is interpretation of the baseline result —")
    print("not adding Year/DTI triple interactions until we understand the core finding.")


if __name__ == "__main__":
    main()
