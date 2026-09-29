from pathlib import Path
from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("pyarrow is required to inspect the Parquet schema.") from exc


PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "pre_regression_integrity_checks"
OUTPUT_XLSX = OUTPUT_DIR / "DUAL50_REPRESENTATIVENESS_AUDIT.xlsx"

EXPECTED_ELIGIBLE_N = 27_692_586

PRIMARY_RACE_ORDER = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic/Latino",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
    "Non-Hispanic NHPI",
    "Non-Hispanic Multiracial",
    "Unknown/Not reported",
]


def pick_column(columns, exact_candidates, contains_all=None, excludes=None):
    """Return the first sensible matching column, or None."""
    colset = set(columns)
    for c in exact_candidates:
        if c in colset:
            return c

    contains_all = contains_all or []
    excludes = excludes or []
    for c in columns:
        lc = c.lower()
        if all(token.lower() in lc for token in contains_all) and not any(
            token.lower() in lc for token in excludes
        ):
            return c
    return None


def truthy(series):
    """Robustly interpret boolean / 0-1 / string indicator columns."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    s = series.astype("string").str.strip().str.lower()
    return s.isin({"1", "true", "t", "yes", "y"})


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_category(series):
    s = series.astype("string").str.strip()
    return s.fillna("Missing").replace({"": "Missing", "<NA>": "Missing", "nan": "Missing"})


def add_category_counts(counter, sample_name, variable, series):
    vc = clean_category(series).value_counts(dropna=False)
    for category, n in vc.items():
        counter[(sample_name, variable, str(category))] += int(n)


def counter_to_composition(counter, variable, category_order=None):
    rows = []
    for (sample, var, category), n in counter.items():
        if var == variable:
            rows.append({"sample": sample, "category": category, "n": n})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["sample", "category", "n", "share"])

    totals = out.groupby("sample")["n"].transform("sum")
    out["share"] = out["n"] / totals

    if category_order:
        order_map = {x: i for i, x in enumerate(category_order)}
        out["__order"] = out["category"].map(order_map).fillna(9999)
        out = out.sort_values(["__order", "sample", "category"]).drop(columns="__order")
    else:
        out = out.sort_values(["category", "sample"])
    return out.reset_index(drop=True)


def canonical_channel(series, source_col):
    """Produce Direct / Non-direct labels from common encodings."""
    if source_col == "submission_of_application":
        x = numeric(series)
        return x.map({1: "Direct", 2: "Non-direct"}).fillna("Other/Missing")

    if "non_direct" in source_col.lower() or "nondirect" in source_col.lower():
        m = truthy(series)
        out = pd.Series("Direct", index=series.index, dtype="string")
        out.loc[m] = "Non-direct"
        out.loc[series.isna()] = "Other/Missing"
        return out

    s = clean_category(series)
    low = s.str.lower()
    out = s.copy()
    out.loc[low.isin({"1", "direct"})] = "Direct"
    out.loc[low.isin({"2", "non-direct", "non direct", "nondirect", "intermediated"})] = "Non-direct"
    return out


def derive_approval(df, approval_col, action_col):
    if approval_col is not None:
        s = df[approval_col]
        if pd.api.types.is_bool_dtype(s):
            return s.astype("float")
        sn = numeric(s)
        # If already 0/1, use it.
        vals = set(sn.dropna().unique().tolist())
        if vals.issubset({0, 1}):
            return sn.astype("float")
        st = s.astype("string").str.strip().str.lower()
        mapped = st.map({
            "approved": 1.0, "approval": 1.0, "yes": 1.0, "true": 1.0,
            "denied": 0.0, "no": 0.0, "false": 0.0,
        })
        if mapped.notna().any():
            return mapped

    if action_col is not None:
        a = numeric(df[action_col])
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[a.isin([1, 2])] = 1.0
        out.loc[a.eq(3)] = 0.0
        return out

    return pd.Series(np.nan, index=df.index, dtype="float64")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found in: {DATA_DIR}")

    schema_cols = list(pq.ParquetFile(parts[0]).schema_arrow.names)

    if "dual50" not in schema_cols:
        raise KeyError(
            "The final V1 schema does not contain `dual50`. "
            "Do not guess a replacement; inspect the support merge first."
        )

    year_col = pick_column(schema_cols, ["activity_year", "year"], contains_all=["year"], excludes=["lender"])
    lender_col = pick_column(schema_cols, ["lei", "lender_lei"], contains_all=["lei"])
    race_col = pick_column(
        schema_cols,
        ["race_ethnicity", "race_group", "race_category", "race_ethnicity_group", "primary_race_group"],
        contains_all=["race"],
        excludes=["applicant_race", "co_applicant", "common", "bw_", "hw_", "aw_"],
    )
    channel_col = pick_column(
        schema_cols,
        ["channel", "channel_label", "submission_channel", "submission_of_application", "non_direct"],
        contains_all=["channel"],
        excludes=["dual", "common", "share"],
    )
    if channel_col is None and "submission_of_application" in schema_cols:
        channel_col = "submission_of_application"
    if channel_col is None and "non_direct" in schema_cols:
        channel_col = "non_direct"

    approval_col = pick_column(
        schema_cols,
        ["approval", "approved", "approval_binary", "approved_binary"],
        contains_all=["approval"],
        excludes=["rate", "raw"],
    )
    action_col = "action_taken" if "action_taken" in schema_cols else None

    dti_col = pick_column(
        schema_cols,
        ["dti_category", "dti_band", "debt_to_income_category"],
        contains_all=["dti"],
        excludes=["common"],
    )
    loan_type_col = "loan_type" if "loan_type" in schema_cols else pick_column(
        schema_cols, ["loan_type_clean", "loan_type_category"], contains_all=["loan", "type"]
    )

    income_col = pick_column(
        schema_cols,
        ["income_clean", "income", "income_numeric", "income_value"],
        contains_all=["income"],
        excludes=["asinh", "missing", "coverage"],
    )
    loan_amount_col = pick_column(
        schema_cols,
        ["loan_amount_clean", "loan_amount"],
        contains_all=["loan", "amount"],
        excludes=["log"],
    )
    property_value_col = pick_column(
        schema_cols,
        ["property_value_clean", "property_value"],
        contains_all=["property", "value"],
        excludes=["log"],
    )

    selected = {
        "dual50": "dual50",
        "year": year_col,
        "lender": lender_col,
        "race": race_col,
        "channel": channel_col,
        "approval": approval_col,
        "action_taken_fallback": action_col,
        "dti_category": dti_col,
        "loan_type": loan_type_col,
        "income_numeric": income_col,
        "loan_amount_numeric": loan_amount_col,
        "property_value_numeric": property_value_col,
    }

    print("=" * 96)
    print("DUAL50 REPRESENTATIVENESS / SAMPLE-SELECTION AUDIT")
    print("=" * 96)
    print(f"Parquet parts: {len(parts)}")
    print("\nDetected variables:")
    for k, v in selected.items():
        print(f"  {k:24s}: {v}")

    usecols = [v for v in selected.values() if v is not None]
    usecols = list(dict.fromkeys(usecols))

    category_counts = Counter()
    n_total = defaultdict(int)
    approval_sum = defaultdict(float)
    approval_n = defaultdict(int)
    numeric_sum = Counter()
    numeric_n = Counter()
    lender_sets = {"Full eligible": set(), "dual50": set()}
    lender_year_counts = {"Full eligible": Counter(), "dual50": Counter()}

    numeric_specs = {
        "income": income_col,
        "loan_amount": loan_amount_col,
        "property_value": property_value_col,
    }

    for i, part in enumerate(parts, start=1):
        d = pd.read_parquet(part, columns=usecols)
        m50 = truthy(d["dual50"])

        samples = {
            "Full eligible": d,
            "dual50": d.loc[m50],
        }

        for sample_name, x in samples.items():
            n_total[sample_name] += len(x)
            if x.empty:
                continue

            # Approval
            ap = derive_approval(x, approval_col, action_col)
            valid_ap = ap.notna()
            approval_sum[sample_name] += float(ap.loc[valid_ap].sum())
            approval_n[sample_name] += int(valid_ap.sum())

            # Categorical composition
            if year_col:
                add_category_counts(category_counts, sample_name, "year", x[year_col])
            if race_col:
                add_category_counts(category_counts, sample_name, "race", x[race_col])
            if dti_col:
                add_category_counts(category_counts, sample_name, "dti", x[dti_col])
            if loan_type_col:
                add_category_counts(category_counts, sample_name, "loan_type", x[loan_type_col])
            if channel_col:
                add_category_counts(
                    category_counts, sample_name, "channel",
                    canonical_channel(x[channel_col], channel_col)
                )

            # Numeric means / coverage
            for metric, col in numeric_specs.items():
                if col is None:
                    continue
                vals = numeric(x[col])
                good = vals.notna() & np.isfinite(vals)
                numeric_sum[(sample_name, metric)] += float(vals.loc[good].sum())
                numeric_n[(sample_name, metric)] += int(good.sum())

            # Lenders and lender-year sizes
            if lender_col:
                lenders = clean_category(x[lender_col])
                lender_sets[sample_name].update(lenders[lenders.ne("Missing")].unique().tolist())

                if year_col:
                    yrs = clean_category(x[year_col])
                    tmp = pd.DataFrame({"lei": lenders, "year": yrs})
                    tmp = tmp.loc[tmp["lei"].ne("Missing") & tmp["year"].ne("Missing")]
                    vc = tmp.value_counts(["year", "lei"])
                    for key, n in vc.items():
                        lender_year_counts[sample_name][tuple(key)] += int(n)

        if i % 50 == 0 or i == len(parts):
            print(
                f"  parts {i:>3}/{len(parts)} | "
                f"full={n_total['Full eligible']:,} | dual50={n_total['dual50']:,}"
            )

    full_n = n_total["Full eligible"]
    dual_n = n_total["dual50"]

    overall_rows = []
    for sample in ["Full eligible", "dual50"]:
        ly_sizes = np.array(list(lender_year_counts[sample].values()), dtype=float)
        overall_rows.append({
            "sample": sample,
            "applications": n_total[sample],
            "share_of_full_eligible": n_total[sample] / full_n if full_n else np.nan,
            "approval_rate": approval_sum[sample] / approval_n[sample] if approval_n[sample] else np.nan,
            "approval_nonmissing_n": approval_n[sample],
            "unique_lenders": len(lender_sets[sample]) if lender_col else np.nan,
            "lender_years": len(lender_year_counts[sample]) if lender_col and year_col else np.nan,
            "median_applications_per_lender_year": float(np.median(ly_sizes)) if ly_sizes.size else np.nan,
            "mean_applications_per_lender_year": float(np.mean(ly_sizes)) if ly_sizes.size else np.nan,
            "p25_applications_per_lender_year": float(np.quantile(ly_sizes, 0.25)) if ly_sizes.size else np.nan,
            "p75_applications_per_lender_year": float(np.quantile(ly_sizes, 0.75)) if ly_sizes.size else np.nan,
        })
    overall = pd.DataFrame(overall_rows)

    year_comp = counter_to_composition(category_counts, "year")
    channel_comp = counter_to_composition(category_counts, "channel", ["Direct", "Non-direct", "Other/Missing"])
    race_comp = counter_to_composition(category_counts, "race", PRIMARY_RACE_ORDER)
    dti_comp = counter_to_composition(category_counts, "dti")
    loan_type_comp = counter_to_composition(category_counts, "loan_type")

    def add_difference(df):
        if df.empty:
            return df
        p = df.pivot_table(index="category", columns="sample", values=["n", "share"], aggfunc="sum")
        p.columns = ["_".join(map(str, c)).replace(" ", "_").lower() for c in p.columns]
        p = p.reset_index()
        full_share_col = "share_full_eligible"
        dual_share_col = "share_dual50"
        if full_share_col in p.columns and dual_share_col in p.columns:
            p["dual50_minus_full_share_pp"] = 100 * (p[dual_share_col] - p[full_share_col])
        return p

    numeric_rows = []
    for sample in ["Full eligible", "dual50"]:
        for metric, col in numeric_specs.items():
            if col is None:
                continue
            n = numeric_n[(sample, metric)]
            numeric_rows.append({
                "sample": sample,
                "metric": metric,
                "source_column": col,
                "nonmissing_n": n,
                "coverage": n / n_total[sample] if n_total[sample] else np.nan,
                "mean": numeric_sum[(sample, metric)] / n if n else np.nan,
            })
    numeric_means = pd.DataFrame(numeric_rows)

    ly_rows = []
    for sample in ["Full eligible", "dual50"]:
        vals = np.array(list(lender_year_counts[sample].values()), dtype=float)
        if vals.size:
            for label, value in [
                ("p10", np.quantile(vals, 0.10)),
                ("p25", np.quantile(vals, 0.25)),
                ("median", np.median(vals)),
                ("p75", np.quantile(vals, 0.75)),
                ("p90", np.quantile(vals, 0.90)),
                ("mean", np.mean(vals)),
            ]:
                ly_rows.append({"sample": sample, "statistic": label, "applications_per_lender_year": value})
    ly_size = pd.DataFrame(ly_rows)

    schema_used = pd.DataFrame(
        [{"role": k, "detected_column": v if v is not None else "NOT FOUND"} for k, v in selected.items()]
    )

    tables = {
        "overall_comparison": overall,
        "year_composition": add_difference(year_comp),
        "channel_composition": add_difference(channel_comp),
        "race_composition": add_difference(race_comp),
        "dti_composition": add_difference(dti_comp),
        "loan_type_composition": add_difference(loan_type_comp),
        "lender_year_size_comparison": ly_size,
        "numeric_means": numeric_means,
        "schema_used": schema_used,
    }

    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = f"Excel workbook not written ({exc}); CSV files were still written."

    print("\n" + "=" * 96)
    print("AUDIT COMPLETE")
    print("=" * 96)
    print(f"Full eligible applications: {full_n:,}")
    print(f"dual50 applications:        {dual_n:,}")
    print(f"dual50 share of full:       {dual_n / full_n:.2%}" if full_n else "")
    if full_n != EXPECTED_ELIGIBLE_N:
        print(
            f"\nWARNING: full eligible N differs from the audited V1 expectation "
            f"({EXPECTED_ELIGIBLE_N:,}). Investigate before regressions."
        )
    else:
        print("\nPASS: full eligible N exactly matches the audited V1 total.")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Workbook: {excel_msg}")
    print("\nSEND BACK:")
    print("  1. overall_comparison.csv")
    print("  2. year_composition.csv")
    print("  3. channel_composition.csv")
    print("  4. race_composition.csv")
    print("  5. dti_composition.csv")
    print("  6. loan_type_composition.csv")
    print("  7. lender_year_size_comparison.csv")
    print("  8. schema_used.csv")


if __name__ == "__main__":
    main()
