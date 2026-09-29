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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "21f_institution_type_subtype_audit"
OUTPUT_XLSX = OUTPUT_DIR / "21F_INSTITUTION_TYPE_SUBTYPE_AUDIT.xlsx"

EXPECTED_N = 9_200_321

TYPE_LABELS = {
    10: "Commercial bank",
    11: "Commercial bank subsidiary",
    12: "Subsidiary of bank/financial holding company",
    13: "Failed commercial bank, no successor",
    14: "U.S. branch of foreign bank",
    20: "Noncommercial bank depository / thrift",
    21: "Noncommercial bank depository subsidiary",
    22: "Subsidiary of thrift holding company",
    23: "Failed thrift, no successor",
    30: "Credit union",
    31: "Subsidiary of credit union",
    32: "Credit union service company",
    33: "Failed credit union, no successor",
    40: "Independent mortgage bank",
    41: "Mortgage bank affiliated with depository / related holding company",
}


def broad_group(t):
    if 10 <= t <= 14:
        return "Commercial-bank family"
    if 20 <= t <= 23:
        return "Thrift / noncommercial-depository family"
    if 30 <= t <= 33:
        return "Credit-union family"
    if 40 <= t <= 41:
        return "Mortgage-company family"
    return "Other / unexpected"


def find_avery():
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
    for root in [Path.home() / "Downloads", PROJECT_DIR]:
        if root.exists():
            hits = list(root.rglob("hmda-2018-present.xlsx"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "Philadelphia Fed hmda-2018-present.xlsx not found."
    )


def load_avery(path):
    a = pd.read_excel(path)
    lookup = {str(c).strip().upper(): c for c in a.columns}
    for req in ["YEAR", "LEI", "TYPE"]:
        if req not in lookup:
            raise KeyError(f"Avery file missing {req}")

    a = a[[lookup["YEAR"], lookup["LEI"], lookup["TYPE"]]].copy()
    a.columns = ["year", "lei", "type"]
    a["year"] = pd.to_numeric(a["year"], errors="coerce").astype("Int16")
    a["lei"] = a["lei"].astype("string").str.strip().str.upper()
    a["type"] = pd.to_numeric(a["type"], errors="coerce").astype("Int16")

    a = a[
        a["year"].between(2018, 2024, inclusive="both")
        & a["lei"].notna()
        & a["type"].notna()
    ].copy()

    conflicts = (
        a.groupby(["year", "lei"])["type"]
        .nunique()
        .reset_index(name="n_type")
    )
    bad = conflicts[conflicts["n_type"] > 1]
    if len(bad):
        raise RuntimeError(
            f"Avery file has {len(bad)} lender-years with multiple TYPE values."
        )

    return a.drop_duplicates(["year", "lei"])


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

    # maps["lender"] is LEI label -> numeric lender_cluster code.
    code_to_lei = {int(code): str(lei).upper() for lei, code in maps["lender"].items()}

    year_idx = (reg["race_year_fe"].to_numpy(np.int16) % 10).astype(np.int8)
    preferred = pd.DataFrame(
        {
            "lender_cluster": reg["lender_cluster"].to_numpy(),
            "year": (2018 + year_idx).astype(np.int16),
        }
    )
    preferred["lei"] = preferred["lender_cluster"].map(code_to_lei)

    if preferred["lei"].isna().any():
        raise RuntimeError("At least one lender code could not be mapped back to LEI.")

    avery_path = find_avery()
    avery = load_avery(avery_path)

    merged = preferred.merge(
        avery,
        on=["year", "lei"],
        how="left",
        validate="many_to_one",
    )

    match_rate = 100 * merged["type"].notna().mean()
    if match_rate < 99.9:
        raise RuntimeError(f"Preferred-sample Avery match rate only {match_rate:.3f}%")

    merged["type"] = merged["type"].astype(int)
    merged["type_label"] = merged["type"].map(TYPE_LABELS).fillna("Unexpected TYPE")
    merged["broad_subtype"] = merged["type"].map(broad_group)

    by_type = (
        merged.groupby(["type", "type_label"])
        .agg(
            applications=("type", "size"),
            lenders=("lender_cluster", "nunique"),
        )
        .reset_index()
    )
    ly_type = (
        merged[["year", "lender_cluster", "type", "type_label", "broad_subtype"]]
        .drop_duplicates()
        .groupby(["type", "type_label"])
        .size()
        .reset_index(name="lender_years")
    )
    by_type = by_type.merge(ly_type, on=["type", "type_label"], how="left")
    by_type["application_share_percent"] = 100 * by_type["applications"] / len(merged)

    by_broad = (
        merged.groupby("broad_subtype")
        .agg(
            applications=("type", "size"),
            lenders=("lender_cluster", "nunique"),
        )
        .reset_index()
    )
    ly_broad = (
        merged[["year", "lender_cluster", "broad_subtype"]]
        .drop_duplicates()
        .groupby("broad_subtype")
        .size()
        .reset_index(name="lender_years")
    )
    by_broad = by_broad.merge(ly_broad, on="broad_subtype", how="left")
    by_broad["application_share_percent"] = (
        100 * by_broad["applications"] / len(merged)
    )

    by_year_broad = (
        merged.groupby(["year", "broad_subtype"])
        .agg(
            applications=("type", "size"),
            lenders=("lender_cluster", "nunique"),
        )
        .reset_index()
    )
    by_year_broad["share_within_year_percent"] = (
        by_year_broad["applications"]
        / by_year_broad.groupby("year")["applications"].transform("sum")
        * 100
    )

    audit_out = pd.DataFrame(
        [{
            "preferred_n": len(merged),
            "matched_n": int(merged["type"].notna().sum()),
            "match_rate_percent": match_rate,
            "avery_path": str(avery_path),
        }]
    )

    by_type.to_csv(OUTPUT_DIR / "preferred_sample_by_exact_TYPE.csv", index=False)
    by_broad.to_csv(OUTPUT_DIR / "preferred_sample_by_broad_subtype.csv", index=False)
    by_year_broad.to_csv(
        OUTPUT_DIR / "preferred_sample_subtype_by_year.csv", index=False
    )
    audit_out.to_csv(OUTPUT_DIR / "subtype_match_audit.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        audit_out.to_excel(writer, "Audit", index=False)
        by_type.to_excel(writer, "Exact TYPE", index=False)
        by_broad.to_excel(writer, "Broad subtype", index=False)
        by_year_broad.to_excel(writer, "Subtype by year", index=False)

    print("\nExact TYPE composition")
    print(by_type.to_string(index=False))
    print("\nBroad subtype composition")
    print(by_broad.to_string(index=False))
    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
