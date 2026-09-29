from __future__ import annotations

import gc
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# LOAD VERIFIED HELPERS

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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "common_support_robustness"
OUTPUT_XLSX = OUTPUT_DIR / "COMMON_SUPPORT_ROBUSTNESS.xlsx"

YEARS = list(range(2018, 2025))
RACES = ["Black", "Hispanic", "Asian"]

SUPPORT_FLAGS = ["dual10", "dual25", "dual50", "dual100"]
RACE_COMMON_FLAGS = {
    "Black": "bw_common10",
    "Hispanic": "hw_common10",
    "Asian": "aw_common10",
}

EXPECTED_SUPPORT_N = {
    "dual10": 12_054_579,
    "dual25": 11_656_368,
    "dual50": 11_268_203,
    "dual100": 10_708_093,
}

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_COMMON_N = 9_200_321


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


def detect_columns(parts):
    cols = list(pq.ParquetFile(parts[0]).schema_arrow.names)

    # Reuse verified schema logic for regression variables.
    schema = base.detect_schema(parts)

    missing_support = [
        col for col in SUPPORT_FLAGS + list(RACE_COMMON_FLAGS.values())
        if col not in cols
    ]

    if missing_support:
        raise KeyError(
            "Required common-support flag(s) missing from final V1:\n"
            + "\n".join(f"  - {x}" for x in missing_support)
        )

    return cols, schema


def build_compact_dual10_data(parts, schema):
    maps = {
        "dti": {},
        "age": {},
        "sex": {},
        "coapp": {},
        "loan_type": {},
        "lender": {},
        "county": {},
    }

    audit = Counter()
    chunks = []

    readcols = [
        "dual10", "dual25", "dual50", "dual100",
        "bw_common10", "hw_common10", "aw_common10",
        schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], schema["income"],
        schema["dti"], schema["age"], schema["sex"], schema["coapp"],
        schema["loan_amount"], schema["property_value"],
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

        # All-race support counts before any fruther restrictions.
        flag_arrays = {}
        for flag in SUPPORT_FLAGS:
            m = base.truthy(d[flag])
            flag_arrays[flag] = m
            audit[f"{flag}_all_races"] += int(m.sum())

        # Build once from broadest threshold.
        m10 = flag_arrays["dual10"]
        d = d.loc[m10].copy()
        if d.empty:
            continue

        # Recreate support flags on the retained rows.
        support = {
            flag: flag_arrays[flag].loc[d.index].to_numpy(bool)
            for flag in SUPPORT_FLAGS
        }
        common_flags = {
            race: base.truthy(d[col]).to_numpy(bool)
            for race, col in RACE_COMMON_FLAGS.items()
        }

        race = base.normalize_race(d[schema["race"]])
        primary = race.notna()

        audit["dual10_primary_race"] += int(primary.sum())

        d = d.loc[primary].copy()
        race = race.loc[primary]

        support = {k: v[primary.to_numpy()] for k, v in support.items()}
        common_flags = {
            k: v[primary.to_numpy()] for k, v in common_flags.items()
        }

        if d.empty:
            continue

        approval = base.normalize_approval(d[schema["approval"]])
        nd = base.normalize_channel(d[schema["channel"]])
        year = base.numeric(d[schema["year"]])
        income = base.numeric(d[schema["income"]]).astype("float64")
        lender_raw = base.clean_string(d[schema["lender"]])
        county_raw = base.normalize_county(d[schema["county"]])

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
            & nd.isin([0, 1])
            & year.isin(YEARS)
            & lender_raw.notna()
            & income.notna() & np.isfinite(income)
            & log_loan.notna() & np.isfinite(log_loan)
            & log_prop.notna() & np.isfinite(log_prop)
        )

        county_ok = (
            county_raw.str.fullmatch(r"\d{5}", na=False)
            & ~county_raw.isin({"00000", "88888", "99999"})
        )
        valid &= county_ok

        audit["dual10_common_sample"] += int(valid.sum())

        if not valid.any():
            continue

        v = valid.to_numpy()

        d = d.loc[valid].copy()
        race = race.loc[valid]
        approval = approval.loc[valid]
        nd = nd.loc[valid]
        year = year.loc[valid].astype(np.int16)
        income = income.loc[valid]
        log_loan = log_loan.loc[valid]
        log_prop = log_prop.loc[valid]
        lender_raw = lender_raw.loc[valid]
        county_raw = county_raw.loc[valid]

        support = {k: arr[v] for k, arr in support.items()}
        common_flags = {k: arr[v] for k, arr in common_flags.items()}

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

            "lender_cluster": lender,
            "lender_year_fe": (
                lender.astype(np.int32) * 10 + (yr - 2018)
            ).astype(np.int32),
            "county_fe": county,

            "race_fe": race_code.astype(np.int8),
            "channel_fe": ndv.astype(np.int8),
            "dti_fe": dti,
            "age_fe": age,
            "sex_fe": sex,
            "coapp_fe": coapp,
            "loan_type_fe": loan_type,

            "race_code": race_code,
        })

        for flag in SUPPORT_FLAGS:
            x[flag] = support[flag].astype(np.int8)

        for race_name in RACES:
            x[RACE_COMMON_FLAGS[race_name]] = (
                common_flags[race_name].astype(np.int8)
            )

        chunks.append(x)

        if pno % 50 == 0 or pno == len(parts):
            print(
                f"  parts {pno:>3}/{len(parts)} | "
                f"eligible={audit['eligible']:,} | "
                f"dual10 all={audit['dual10_all_races']:,} | "
                f"dual10 complete={audit['dual10_common_sample']:,}"
            )

        del d, x
        gc.collect()

    if not chunks:
        raise RuntimeError("No observations survived the dual10 build.")

    print("\nCombining compact dual10 dataset...")
    reg = pd.concat(chunks, ignore_index=True, copy=False)

    del chunks
    gc.collect()

    return reg, audit, maps


def pooled_formula(races=("Black", "Hispanic", "Asian")):
    explicit = [
        f"{race}_x_NonDirect" for race in races
    ] + [
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
        "loan_type_fe",
    ]

    return (
        "approval ~ "
        + " + ".join(explicit)
        + " | "
        + " + ".join(fixed_effects)
    )


def extract_headline(fit, model_name, terms):
    t = base.tidy_fit(fit)
    t["model"] = model_name

    out = t.loc[t["term"].isin(terms)].copy()

    if "p_value" in out.columns:
        out["significance"] = out["p_value"].apply(stars)

    keep = [
        c for c in [
            "model", "term",
            "estimate", "std_error", "p_value", "ci_low", "ci_high",
            "estimate_pp", "std_error_pp", "ci_low_pp", "ci_high_pp",
            "significance",
        ]
        if c in out.columns
    ]

    return out[keep]


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquets found:\n{DATA_DIR}"
        )

    print("=" * 108)
    print("COMMON-SUPPORT ROBUSTNESS — ALTERNATIVE THRESHOLDS + RACE-SPECIFIC COMMON10")
    print("=" * 108)
    print(f"Parts: {len(parts)}")
    print(f"PyFixest: {base.package_version('pyfixest')}")
    print("SE: CRV1 clustered by lender")
    print("Main specification: pooled preferred M5, memory-safe parameterisation\n")

    _, schema = detect_columns(parts)

    schema_used = pd.DataFrame([
        {"role": k, "detected_column": v if v else "NOT FOUND"}
        for k, v in schema.items()
    ])

    print("Building the broad dual10 complete-case dataset once...")
    reg, audit, maps = build_compact_dual10_data(parts, schema)

    # Audit

    audit_rows = [{
        "check": "Eligible V1 rows scanned",
        "observed_n": audit["eligible"],
        "expected_n": EXPECTED_ELIGIBLE_N,
        "pass": audit["eligible"] == EXPECTED_ELIGIBLE_N,
    }]

    for flag in SUPPORT_FLAGS:
        observed = audit[f"{flag}_all_races"]
        expected = EXPECTED_SUPPORT_N[flag]
        audit_rows.append({
            "check": f"{flag} all-race support count",
            "observed_n": observed,
            "expected_n": expected,
            "pass": observed == expected,
        })

    # Exact dual50 complete-case N in the compact dataset.
    dual50_common = int(reg["dual50"].eq(1).sum())
    audit_rows.append({
        "check": "dual50 four-race complete-case sample",
        "observed_n": dual50_common,
        "expected_n": EXPECTED_DUAL50_COMMON_N,
        "pass": dual50_common == EXPECTED_DUAL50_COMMON_N,
    })

    sample_audit = pd.DataFrame(audit_rows)

    print("\nSample audit:")
    print(sample_audit.to_string(index=False))

    if not sample_audit["pass"].all():
        raise RuntimeError(
            "Support robustness sample audit failed. "
            "Do not interpret estimates until resolved."
        )

    mem_gb = reg.memory_usage(deep=True).sum() / (1024 ** 3)
    print(
        f"\nCompact dual10 dataframe: {len(reg):,} observations | "
        f"{mem_gb:.2f} GB"
    )

    # A. Alternative thresholds

    threshold_results = []
    threshold_info = []

    formula = pooled_formula()
    headline_terms = [
        "Black_x_NonDirect",
        "Hispanic_x_NonDirect",
        "Asian_x_NonDirect",
    ]

    print("\n" + "=" * 108)
    print("A. ALTERNATIVE DUAL-CHANNEL THRESHOLDS")
    print("=" * 108)

    for flag in SUPPORT_FLAGS:
        sub = reg.loc[reg[flag].eq(1)].copy()

        print(
            f"\n>>> {flag}: N before FE singleton removal = {len(sub):,}"
        )

        fit = base.feols_memorysafe(formula, sub)

        h = extract_headline(
            fit,
            flag,
            headline_terms,
        )
        threshold_results.append(h)

        threshold_info.append({
            "model": flag,
            "observations_before_singleton_removal": len(sub),
            "lender_clusters": int(sub["lender_cluster"].nunique()),
            "lender_year_fe_groups": int(sub["lender_year_fe"].nunique()),
            "county_fe_groups": int(sub["county_fe"].nunique()),
            "formula": formula,
        })

        print(
            h[
                [
                    "term", "estimate_pp", "std_error_pp",
                    "p_value", "significance"
                ]
            ].to_string(index=False)
        )

        del fit, h, sub
        gc.collect()

    threshold_results = pd.concat(
        threshold_results, ignore_index=True
    )
    threshold_info = pd.DataFrame(threshold_info)

    # B. Race-specific common10

    pair_results = []
    pair_info = []

    print("\n" + "=" * 108)
    print("B. RACE-SPECIFIC COMMON10 PAIRWISE SAMPLES")
    print("=" * 108)

    race_codes = {
        "Black": 1,
        "Hispanic": 2,
        "Asian": 3,
    }

    for race_name, race_code in race_codes.items():
        flag = RACE_COMMON_FLAGS[race_name]

        mask = (
            reg[flag].eq(1)
            & reg["race_code"].isin([0, race_code])
        )

        sub = reg.loc[mask].copy()

        # Recode race FE to a compact 0/1 for the pair.
        sub["race_fe"] = (
            sub["race_code"].eq(race_code)
        ).astype(np.int8)

        target_term = f"{race_name}_x_NonDirect"
        pair_formula = pooled_formula(races=(race_name,))

        print(
            f"\n>>> {race_name} vs White on {flag}: "
            f"N before FE singleton removal = {len(sub):,}"
        )

        fit = base.feols_memorysafe(
            pair_formula,
            sub,
        )

        h = extract_headline(
            fit,
            f"{race_name}_White_{flag}",
            [target_term],
        )

        pair_results.append(h)

        pair_info.append({
            "comparison": f"{race_name} vs White",
            "support_flag": flag,
            "observations_before_singleton_removal": len(sub),
            "target_race_n": int(
                sub["race_code"].eq(race_code).sum()
            ),
            "white_n": int(
                sub["race_code"].eq(0).sum()
            ),
            "lender_clusters": int(sub["lender_cluster"].nunique()),
            "lender_year_fe_groups": int(sub["lender_year_fe"].nunique()),
            "county_fe_groups": int(sub["county_fe"].nunique()),
            "formula": pair_formula,
        })

        print(
            h[
                [
                    "term", "estimate_pp", "std_error_pp",
                    "p_value", "significance"
                ]
            ].to_string(index=False)
        )

        del fit, h, sub
        gc.collect()

    pair_results = pd.concat(pair_results, ignore_index=True)
    pair_info = pd.DataFrame(pair_info)

    # Category map

    category_rows = []
    for var, mapping in maps.items():
        for label, code_ in sorted(mapping.items(), key=lambda z: z[1]):
            category_rows.append({
                "variable": var,
                "code_used": code_,
                "cleaned_label": label,
            })
    category_levels = pd.DataFrame(category_rows)

    # Save

    tables = {
        "threshold_robustness_results": threshold_results,
        "threshold_model_info": threshold_info,
        "race_specific_common10_results": pair_results,
        "race_specific_model_info": pair_info,
        "sample_audit": sample_audit,
        "schema_used": schema_used,
        "category_levels": category_levels,
    }

    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(
                    writer,
                    sheet_name=name[:31],
                    index=False,
                )
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = f"Workbook not written ({exc}); CSVs were written."

    print("\n" + "=" * 108)
    print("ROBUSTNESS BLOCK A+B COMPLETE")
    print("=" * 108)

    print("\nThreshold results:")
    print(
        threshold_results[
            [
                "model", "term", "estimate_pp",
                "std_error_pp", "p_value", "significance",
            ]
        ].to_string(index=False)
    )

    print("\nRace-specific common10 results:")
    print(
        pair_results[
            [
                "model", "term", "estimate_pp",
                "std_error_pp", "p_value", "significance",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK FIRST:")
    print("  1. threshold_robustness_results.csv")
    print("  2. threshold_model_info.csv")
    print("  3. race_specific_common10_results.csv")
    print("  4. race_specific_model_info.csv")
    print("  5. sample_audit.csv")
    print(
        "\nAfter this, the next robustness block is:"
        "\n  - originated (1) vs denied (3)"
        "\n  - conventional-only mortgages"
    )


if __name__ == "__main__":
    main()
