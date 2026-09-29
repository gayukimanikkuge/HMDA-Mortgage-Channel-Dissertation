#!/usr/bin/env python3
from __future__ import annotations

import gc
import inspect
import math
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("pyarrow is required.") from exc

try:
    import pyfixest as pf
except ImportError as exc:
    raise ImportError(
        "\nPyFixest is required. In the SAME Spyder environment run once:\n\n"
        "    %pip install -U pyfixest\n"
    ) from exc

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("scipy is required and should normally be installed with PyFixest.") from exc

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "race_channel_year_heterogeneity"
OUTPUT_XLSX = OUTPUT_DIR / "RACE_CHANNEL_YEAR_HETEROGENEITY.xlsx"

MAX_PARTS = None
EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_COMMON_SAMPLE_N = 9_200_321

STUDY_YEARS = list(range(2018, 2025))
BASE_YEAR = 2018
POST_BASE_YEARS = list(range(2019, 2025))
RACES = ["Black", "Hispanic", "Asian"]

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

def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"

def first_existing(columns, candidates):
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None

def clean_string(s):
    out = s.astype("string").str.strip()
    return out.replace({
        "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
        "<NA>": pd.NA, "NA": pd.NA, "N/A": pd.NA, "NULL": pd.NA,
    })

def truthy(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().any():
        return n.eq(1)
    x = clean_string(s).str.lower()
    return x.isin({"true", "t", "yes", "y", "1"})

def numeric(s):
    return pd.to_numeric(s, errors="coerce")

def category_labels(s):
    return clean_string(s).fillna("Missing / Not reported")

def normalize_race(s):
    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)
    mapping = {
        "white": "White", "non-hispanic white": "White",
        "non hispanic white": "White", "nh white": "White",
        "black": "Black", "black or african american": "Black",
        "non-hispanic black": "Black", "non hispanic black": "Black",
        "nh black": "Black",
        "hispanic": "Hispanic", "hispanic/latino": "Hispanic",
        "hispanic / latino": "Hispanic", "hispanic or latino": "Hispanic",
        "latino": "Hispanic",
        "asian": "Asian", "non-hispanic asian": "Asian",
        "non hispanic asian": "Asian", "nh asian": "Asian",
    }
    return x.map(mapping).astype("string")

def normalize_channel(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")
    out.loc[n.eq(1)] = 0
    out.loc[n.eq(2)] = 1
    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)
    out.loc[x.isin({"direct", "submitted directly"})] = 0
    out.loc[x.isin({
        "non-direct", "non direct", "nondirect", "not direct",
        "not submitted directly", "intermediated"
    })] = 1
    return out

def normalize_approval(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    valid_binary = n.isin([0, 1])
    out.loc[valid_binary] = n.loc[valid_binary]
    x = clean_string(s).str.lower()
    out.loc[x.isin({"approved", "approval", "yes", "true"})] = 1.0
    out.loc[x.isin({"denied", "no", "false"})] = 0.0
    return out

def normalize_county(s):
    x = clean_string(s)
    x = x.str.replace(r"\.0$", "", regex=True)
    m = x.str.fullmatch(r"\d{1,5}", na=False)
    x.loc[m] = x.loc[m].str.zfill(5)
    return x

def safe_log_positive(s):
    x = numeric(s).astype("float64")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    good = x.notna() & np.isfinite(x) & x.gt(0)
    out.loc[good] = np.log(x.loc[good])
    return out

def encode_incremental(s, mapping, dtype):
    labels = category_labels(s)
    local_codes, uniques = pd.factorize(labels, sort=False)
    global_for_local = np.empty(len(uniques), dtype=np.int32)
    for j, label in enumerate(uniques.astype(str)):
        if label not in mapping:
            mapping[label] = len(mapping)
        global_for_local[j] = mapping[label]
    return global_for_local[local_codes].astype(dtype, copy=False)

def detect_schema(parts):
    cols = list(pq.ParquetFile(parts[0]).schema_arrow.names)
    schema = {role: first_existing(cols, candidates) for role, candidates in PREFERRED_COLUMNS.items()}
    required = [
        "dual50", "year", "lender", "race", "channel", "approval",
        "income", "dti", "age", "sex", "coapp", "loan_amount",
        "property_value", "loan_type", "county",
    ]
    missing = [r for r in required if schema[r] is None]
    if missing:
        raise KeyError("Missing required final-V1 fields:\n" + "\n".join(f"  - {x}" for x in missing))
    return schema

def build_data(parts, schema):
    category_maps = {
        "dti": {}, "age": {}, "sex": {}, "coapp": {},
        "loan_type": {}, "lender": {}, "county": {},
    }
    chunks = []
    audit = Counter()

    readcols = [
        schema["dual50"], schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], schema["income"], schema["dti"],
        schema["age"], schema["sex"], schema["coapp"], schema["loan_amount"],
        schema["property_value"], schema["loan_type"], schema["county"],
    ]
    if schema["log_loan_amount"] is not None:
        readcols.append(schema["log_loan_amount"])
    if schema["log_property_value"] is not None:
        readcols.append(schema["log_property_value"])
    readcols = list(dict.fromkeys(readcols))

    for pno, part in enumerate(parts, 1):
        d = pd.read_parquet(part, columns=readcols)
        audit["eligible"] += len(d)

        m50 = truthy(d[schema["dual50"]])
        audit["dual50"] += int(m50.sum())
        d = d.loc[m50].copy()
        if d.empty:
            continue

        race = normalize_race(d[schema["race"]])
        mrace = race.notna()
        audit["dual50_primary_race"] += int(mrace.sum())
        d = d.loc[mrace].copy()
        race = race.loc[mrace]
        if d.empty:
            continue

        approval = normalize_approval(d[schema["approval"]])
        nondirect = normalize_channel(d[schema["channel"]])
        year = numeric(d[schema["year"]])
        income = numeric(d[schema["income"]]).astype("float64")
        lender_raw = clean_string(d[schema["lender"]])
        county_raw = normalize_county(d[schema["county"]])

        log_loan = (
            numeric(d[schema["log_loan_amount"]]).astype("float64")
            if schema["log_loan_amount"] is not None
            else safe_log_positive(d[schema["loan_amount"]])
        )
        log_prop = (
            numeric(d[schema["log_property_value"]]).astype("float64")
            if schema["log_property_value"] is not None
            else safe_log_positive(d[schema["property_value"]])
        )

        valid_approval = approval.isin([0, 1])
        valid_channel = nondirect.isin([0, 1])
        valid_year = year.isin(STUDY_YEARS)
        valid_lender = lender_raw.notna()
        valid_income = income.notna() & np.isfinite(income)
        valid_log_loan = log_loan.notna() & np.isfinite(log_loan)
        valid_log_prop = log_prop.notna() & np.isfinite(log_prop)
        county_regex = county_raw.str.fullmatch(r"\d{5}", na=False)
        county_special = county_raw.isin({"00000", "88888", "99999"})
        valid_county = county_regex & ~county_special

        keep = (
            valid_approval & valid_channel & valid_year & valid_lender
            & valid_income & valid_log_loan & valid_log_prop & valid_county
        )
        audit["common_sample"] += int(keep.sum())
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

        lender_code = encode_incremental(lender_raw, category_maps["lender"], np.int16)
        county_code = encode_incremental(county_raw, category_maps["county"], np.int32)
        dti_code = encode_incremental(d[schema["dti"]], category_maps["dti"], np.int16)
        age_code = encode_incremental(d[schema["age"]], category_maps["age"], np.int16)
        sex_code = encode_incremental(d[schema["sex"]], category_maps["sex"], np.int16)
        coapp_code = encode_incremental(d[schema["coapp"]], category_maps["coapp"], np.int16)
        loan_type_code = encode_incremental(d[schema["loan_type"]], category_maps["loan_type"], np.int16)

        black = race.eq("Black").to_numpy(np.int8)
        hispanic = race.eq("Hispanic").to_numpy(np.int8)
        asian = race.eq("Asian").to_numpy(np.int8)
        nd = nondirect.to_numpy(np.int8)
        yr = year.to_numpy(np.int16)

        x = pd.DataFrame({
            "approval": approval.to_numpy(np.int8),
            "Black": black, "Hispanic": hispanic, "Asian": asian, "NonDirect": nd,
            "Black_x_NonDirect": (black * nd).astype(np.int8),
            "Hispanic_x_NonDirect": (hispanic * nd).astype(np.int8),
            "Asian_x_NonDirect": (asian * nd).astype(np.int8),
            "income_asinh": np.arcsinh(income.to_numpy(np.float64)).astype(np.float32),
            "dti_code": dti_code, "age_code": age_code, "sex_code": sex_code,
            "coapp_code": coapp_code,
            "log_loan_amount": log_loan.to_numpy(np.float32),
            "log_property_value": log_prop.to_numpy(np.float32),
            "loan_type_code": loan_type_code,
            "lender_cluster": lender_code, "county_fe": county_code, "year": yr,
        })

        race_arrays = {"Black": black, "Hispanic": hispanic, "Asian": asian}
        for y in POST_BASE_YEARS:
            ydummy = (yr == y).astype(np.int8)
            x[f"NonDirect_x_Y{y}"] = (nd * ydummy).astype(np.int8)
            for r, arr in race_arrays.items():
                x[f"{r}_x_Y{y}"] = (arr * ydummy).astype(np.int8)
                x[f"{r}_x_NonDirect_x_Y{y}"] = (arr * nd * ydummy).astype(np.int8)

        chunks.append(x)

        if pno % 50 == 0 or pno == len(parts):
            print(
                f"  parts {pno:>3}/{len(parts)} | eligible={audit['eligible']:,} | "
                f"dual50={audit['dual50']:,} | primary race={audit['dual50_primary_race']:,} | "
                f"common sample={audit['common_sample']:,}"
            )
        del d, x
        gc.collect()

    if not chunks:
        raise RuntimeError("No observations survived.")

    reg = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()

    reg["lender_year_fe"] = (
        reg["lender_cluster"].astype(np.int32) * 10
        + (reg["year"].astype(np.int32) - BASE_YEAR)
    ).astype(np.int32)
    return reg, audit, category_maps

def feols_low_memory(formula, data):
    sig = inspect.signature(pf.feols)
    params = sig.parameters
    kwargs = {"fml": formula, "data": data, "vcov": {"CRV1": "lender_cluster"}}
    for key, value_ in {
        "copy_data": False, "store_data": False, "lean": True, "fixef_rm": "singleton"
    }.items():
        if key in params:
            kwargs[key] = value_
    return pf.feols(**kwargs)

def build_formula():
    terms = [
        "Black", "Hispanic", "Asian", "NonDirect",
        "Black_x_NonDirect", "Hispanic_x_NonDirect", "Asian_x_NonDirect",
    ]
    for y in POST_BASE_YEARS:
        terms.append(f"NonDirect_x_Y{y}")
        for r in RACES:
            terms.append(f"{r}_x_Y{y}")
            terms.append(f"{r}_x_NonDirect_x_Y{y}")
    terms += [
        "income_asinh", "C(dti_code)", "C(age_code)", "C(sex_code)",
        "C(coapp_code)", "log_loan_amount", "log_property_value",
        "C(loan_type_code)",
    ]
    return "approval ~ " + " + ".join(terms) + " | lender_year_fe + county_fe"

def get_coef_vcov(fit):
    coef = fit.coef()
    if not isinstance(coef, pd.Series):
        coef = pd.Series(coef)
    names = list(coef.index)
    beta = coef.to_numpy(dtype=float)
    V = np.asarray(fit._vcov, dtype=float)
    if V.shape != (len(names), len(names)):
        raise RuntimeError("Could not align covariance matrix with coefficient vector.")
    return names, beta, V

def linear_combo(names, beta, V, weights, df_t):
    index = {name: i for i, name in enumerate(names)}
    missing = [k for k in weights if k not in index]
    if missing:
        raise KeyError("Required coefficient(s) missing:\n" + "\n".join(missing))
    a = np.zeros(len(names), dtype=float)
    for k, w in weights.items():
        a[index[k]] = float(w)
    est = float(a @ beta)
    var = float(a @ V @ a)
    se = math.sqrt(max(var, 0.0))
    tval = est / se if se > 0 else np.nan
    pval = 2 * stats.t.sf(abs(tval), df=df_t) if np.isfinite(tval) else np.nan
    crit = stats.t.ppf(0.975, df=df_t)
    lo, hi = est - crit * se, est + crit * se
    return {
        "estimate": est, "std_error": se, "t_value": tval,
        "p_value": pval, "ci_low": lo, "ci_high": hi,
    }

def annual_channel_effects(names, beta, V, df_t):
    rows = []
    for race in RACES:
        base = f"{race}_x_NonDirect"
        for year in STUDY_YEARS:
            weights = {base: 1.0}
            if year != BASE_YEAR:
                weights[f"{race}_x_NonDirect_x_Y{year}"] = 1.0
            r = linear_combo(names, beta, V, weights, df_t)
            rows.append({
                "race_group": race, "year": year, **r,
                "estimate_pp": 100 * r["estimate"],
                "std_error_pp": 100 * r["std_error"],
                "ci_low_pp": 100 * r["ci_low"],
                "ci_high_pp": 100 * r["ci_high"],
                "interpretation": (
                    "Gap narrower in non-direct" if r["estimate"] > 0
                    else "Gap wider in non-direct" if r["estimate"] < 0
                    else "No point-estimate difference"
                ),
            })
    return pd.DataFrame(rows)

def annual_racial_gaps_by_channel(names, beta, V, df_t):
    rows = []
    for race in RACES:
        for year in STUDY_YEARS:
            w_direct = {race: 1.0}
            if year != BASE_YEAR:
                w_direct[f"{race}_x_Y{year}"] = 1.0
            rd = linear_combo(names, beta, V, w_direct, df_t)

            w_nd = dict(w_direct)
            w_nd[f"{race}_x_NonDirect"] = 1.0
            if year != BASE_YEAR:
                w_nd[f"{race}_x_NonDirect_x_Y{year}"] = 1.0
            rn = linear_combo(names, beta, V, w_nd, df_t)

            for channel, result in [("Direct", rd), ("Non-direct", rn)]:
                rows.append({
                    "race_group": race, "year": year, "channel": channel, **result,
                    "gap_vs_white_pp": 100 * result["estimate"],
                    "std_error_pp": 100 * result["std_error"],
                    "ci_low_pp": 100 * result["ci_low"],
                    "ci_high_pp": 100 * result["ci_high"],
                })
    return pd.DataFrame(rows)

def joint_zero_test(names, beta, V, restriction_terms, df_denom):
    index = {name: i for i, name in enumerate(names)}
    missing = [t for t in restriction_terms if t not in index]
    if missing:
        raise KeyError("Joint-test coefficient(s) missing:\n" + "\n".join(missing))
    q = len(restriction_terms)
    R = np.zeros((q, len(names)), dtype=float)
    for j, term in enumerate(restriction_terms):
        R[j, index[term]] = 1.0
    rb = R @ beta
    RVRT = R @ V @ R.T
    wald = float(rb.T @ np.linalg.pinv(RVRT) @ rb)
    fstat = wald / q
    pval = float(stats.f.sf(fstat, q, df_denom))
    return {
        "restrictions": q, "wald_chi_square_equivalent": wald,
        "f_statistic": fstat, "df_num": q, "df_denom": df_denom,
        "p_value": pval,
    }

def run_joint_tests(names, beta, V, n_clusters):
    rows, all_terms = [], []
    for race in RACES:
        terms = [f"{race}_x_NonDirect_x_Y{y}" for y in POST_BASE_YEARS]
        all_terms.extend(terms)
        res = joint_zero_test(names, beta, V, terms, n_clusters - 1)
        rows.append({
            "test": f"{race}: Race × Channel constant across 2018–2024",
            "null_hypothesis": f"All {race} × NonDirect × Year changes relative to 2018 = 0",
            **res,
        })
    overall = joint_zero_test(names, beta, V, all_terms, n_clusters - 1)
    rows.append({
        "test": "OVERALL H3: Race × Channel constant across 2018–2024",
        "null_hypothesis": "All 18 race-specific Race × NonDirect × Year changes = 0",
        **overall,
    })
    return pd.DataFrame(rows)

def tidy_fit(fit):
    t = fit.tidy().copy().reset_index()
    first = t.columns[0]
    if first != "term":
        t = t.rename(columns={first: "term"})
    t = t.rename(columns={
        "Estimate": "estimate", "Std. Error": "std_error", "t value": "t_value",
        "Pr(>|t|)": "p_value", "2.5%": "ci_low", "97.5%": "ci_high",
    })
    for c in ["estimate", "std_error", "ci_low", "ci_high"]:
        if c in t.columns:
            t[f"{c}_pp"] = pd.to_numeric(t[c], errors="coerce") * 100
    return t

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No V1 Parquets found in:\n{DATA_DIR}")
    if MAX_PARTS is not None:
        parts = parts[:MAX_PARTS]

    print("=" * 102)
    print("RACE × CHANNEL × YEAR HETEROGENEITY — PREFERRED LPM")
    print("=" * 102)
    print(f"Parts: {len(parts)}")
    print(f"PyFixest: {package_version('pyfixest')}")
    print("Reference year: 2018 | race: NH White | channel: Direct")
    print("FE: lender × year + county | SE: CRV1 clustered by lender\n")

    schema = detect_schema(parts)
    schema_used = pd.DataFrame([
        {"role": k, "detected_column": v if v else "NOT FOUND"}
        for k, v in schema.items()
    ])

    print("Building exact preferred estimation sample...")
    reg, audit, category_maps = build_data(parts, schema)

    sample_audit = pd.DataFrame([
        {"stage": "Eligible V1 rows scanned", "n": audit["eligible"],
         "expected_n": EXPECTED_ELIGIBLE_N,
         "pass": True if MAX_PARTS is not None else audit["eligible"] == EXPECTED_ELIGIBLE_N},
        {"stage": "dual50 all races", "n": audit["dual50"],
         "expected_n": EXPECTED_DUAL50_N,
         "pass": True if MAX_PARTS is not None else audit["dual50"] == EXPECTED_DUAL50_N},
        {"stage": "dual50 + primary four race groups", "n": audit["dual50_primary_race"],
         "expected_n": np.nan, "pass": True},
        {"stage": "Common complete-case sample before FE singleton removal",
         "n": audit["common_sample"], "expected_n": EXPECTED_COMMON_SAMPLE_N,
         "pass": True if MAX_PARTS is not None else audit["common_sample"] == EXPECTED_COMMON_SAMPLE_N},
    ])

    print("\nSample replication:")
    print(sample_audit.to_string(index=False))
    if MAX_PARTS is None and not sample_audit["pass"].all():
        raise RuntimeError(
            "\nThe year-heterogeneity sample does not exactly reproduce the baseline sample."
        )

    print(
        f"\nRegression dataframe: {len(reg):,} rows | "
        f"{reg.memory_usage(deep=True).sum() / (1024**3):.2f} GB"
    )
    print(
        f"Lender clusters: {reg['lender_cluster'].nunique():,} | "
        f"Lender-years: {reg['lender_year_fe'].nunique():,} | "
        f"Counties: {reg['county_fe'].nunique():,}"
    )

    formula = build_formula()
    print("\nEstimating saturated Race × Channel × Year model...")
    fit = feols_low_memory(formula, reg)

    names, beta, V = get_coef_vcov(fit)
    n_clusters = int(reg["lender_cluster"].nunique())
    df_t = n_clusters - 1

    annual_effects = annual_channel_effects(names, beta, V, df_t)
    annual_gaps = annual_racial_gaps_by_channel(names, beta, V, df_t)
    joint_tests = run_joint_tests(names, beta, V, n_clusters)
    model_coefs = tidy_fit(fit)

    category_rows = []
    for var, mapping in category_maps.items():
        for label, code_ in sorted(mapping.items(), key=lambda z: z[1]):
            category_rows.append({
                "variable": var, "code_used_in_regression": code_,
                "original_cleaned_label": label,
            })
    category_levels = pd.DataFrame(category_rows)

    model_info = pd.DataFrame([{
        "observations_before_fe_singleton_removal": len(reg),
        "lender_clusters": n_clusters,
        "lender_year_fe_groups": reg["lender_year_fe"].nunique(),
        "county_fe_groups": reg["county_fe"].nunique(),
        "cluster_df_for_linear_combinations": df_t,
        "pyfixest_version": package_version("pyfixest"),
        "formula": formula,
    }])

    tables = {
        "annual_race_channel_effects": annual_effects,
        "annual_racial_gaps_by_channel": annual_gaps,
        "joint_h3_tests": joint_tests,
        "year_model_coefficients": model_coefs,
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

    print("\n" + "=" * 102)
    print("PRIMARY H3 RESULT: ANNUAL RACE × NON-DIRECT DIFFERENTIALS")
    print("=" * 102)
    print(
        annual_effects[
            ["race_group", "year", "estimate_pp", "std_error_pp",
             "p_value", "ci_low_pp", "ci_high_pp"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 102)
    print("JOINT TESTS OF TEMPORAL HETEROGENEITY")
    print("=" * 102)
    print(
        joint_tests[
            ["test", "f_statistic", "df_num", "df_denom", "p_value"]
        ].to_string(index=False)
    )

    overall_p = joint_tests.loc[
        joint_tests["test"].str.startswith("OVERALL H3"), "p_value"
    ].iloc[0]
    print("\nInterpretation:")
    if overall_p < 0.05:
        print(
            "  OVERALL H3 TEST REJECTS constancy at 5%: the Race × Channel "
            "relationship varies over time somewhere across the three groups."
        )
    else:
        print(
            "  OVERALL H3 TEST DOES NOT REJECT constancy at 5%: there is not "
            "strong joint evidence that Race × Channel varies over time."
        )

    print("\nDo not cherry-pick individual years; interpret them with the joint tests.")
    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")
    print("\nPLEASE SEND BACK THESE FIRST:")
    print("  1. annual_race_channel_effects.csv")
    print("  2. joint_h3_tests.csv")
    print("  3. annual_racial_gaps_by_channel.csv")
    print("  4. sample_audit.csv")
    print("  5. model_info.csv")
    print("\nThen we decide whether the 3-period model is substantively justified.")

if __name__ == "__main__":
    main()
