from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "13b_race_channel_year_heterogeneity_memorysafe.py"
DTI_SCRIPT = HERE / "16_race_channel_dti_heterogeneity.py"

for p in [BASE_SCRIPT, DTI_SCRIPT]:
    if not p.exists():
        raise FileNotFoundError(f"Required companion script not found:\n{p}")

spec = importlib.util.spec_from_file_location("h3base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

spec2 = importlib.util.spec_from_file_location("dtibase", DTI_SCRIPT)
dti = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(dti)

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "21c_income_tail_sensitivity"
OUTPUT_XLSX = OUTPUT_DIR / "21C_INCOME_TAIL_SENSITIVITY.xlsx"

EXPECTED_N = 9_200_321
HEADLINE = [
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]


def prepare_pooled(reg):
    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)
    if not np.array_equal(race_year % 10, channel_year % 10):
        raise RuntimeError("Race-year and channel-year encodings disagree.")

    reg["race_fe"] = (race_year // 10).astype(np.int8)
    reg["channel_fe"] = (channel_year // 10).astype(np.int8)

    annual_cols = [c for c in reg.columns if "_x_NonDirect_x_Y" in c]
    reg.drop(
        columns=annual_cols + ["race_year_fe", "channel_year_fe"],
        inplace=True,
        errors="ignore",
    )
    gc.collect()
    return reg


def pooled_formula():
    explicit = HEADLINE + [
        "income_asinh", "log_loan_amount", "log_property_value"
    ]
    fes = [
        "lender_year_fe", "county_fe", "race_fe", "channel_fe",
        "dti_fe", "age_fe", "sex_fe", "coapp_fe", "loan_type_fe",
    ]
    return "approval ~ " + " + ".join(explicit) + " | " + " + ".join(fes)


def extract_headline(fit, label):
    t = base.tidy_fit(fit).copy()
    out = t[t["term"].isin(HEADLINE)].copy()
    out["model"] = label
    return out


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

    q = float(reg["income_asinh"].quantile(0.999))
    keep = reg["income_asinh"].le(q)
    removed_n = int((~keep).sum())
    trimmed = reg.loc[keep].copy()

    # Back-transform only for an interpretable audit threshold.
    raw_income_threshold = float(np.sinh(np.float64(q)))

    audit_out = pd.DataFrame(
        [{
            "preferred_n": len(reg),
            "income_asinh_99_9_percentile": q,
            "approx_raw_income_threshold_thousands": raw_income_threshold,
            "removed_n": removed_n,
            "removed_percent": 100 * removed_n / len(reg),
            "retained_n": len(trimmed),
        }]
    )

    print("\nIncome-tail audit:")
    print(audit_out.to_string(index=False))

    # 1. Pooled M5 after top-tail exclusion
    pooled_data = prepare_pooled(trimmed.copy())
    fit_pool = base.feols_memorysafe(pooled_formula(), pooled_data)
    pooled = extract_headline(fit_pool, "Top 0.1% income excluded")

    # 2. Annual H3 after top-tail exclusion
    fit_h3 = base.feols_memorysafe(base.build_formula(), trimmed)
    names, beta, V = base.get_coef_vcov(fit_h3)
    n_clusters = int(trimmed["lender_cluster"].nunique())
    annual = base.build_annual_effects(names, beta, V, n_clusters - 1)
    annual_joint = base.build_joint_tests(names, beta, V, n_clusters)

    # 3. DTI H4 after top-tail exclusion
    dti_reg, dti_excluded = dti.prepare_dti_data(trimmed.copy(), maps)
    fit_dti = base.feols_memorysafe(dti.build_formula(), dti_reg)
    n2, b2, v2 = base.get_coef_vcov(fit_dti)
    n_clusters_dti = int(dti_reg["lender_cluster"].nunique())
    dti_effects = dti.build_dti_effects(n2, b2, v2, n_clusters_dti - 1)
    dti_joint = dti.build_joint_tests(n2, b2, v2, n_clusters_dti)

    pooled.to_csv(OUTPUT_DIR / "income_trim_pooled_m5.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "income_trim_annual_h3.csv", index=False)
    annual_joint.to_csv(OUTPUT_DIR / "income_trim_joint_h3.csv", index=False)
    dti_effects.to_csv(OUTPUT_DIR / "income_trim_dti_h4.csv", index=False)
    dti_joint.to_csv(OUTPUT_DIR / "income_trim_joint_h4.csv", index=False)
    audit_out.to_csv(OUTPUT_DIR / "income_trim_audit.csv", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        audit_out.to_excel(writer, "Audit", index=False)
        pooled.to_excel(writer, "Pooled M5", index=False)
        annual.to_excel(writer, "Annual H3", index=False)
        annual_joint.to_excel(writer, "H3 joint", index=False)
        dti_effects.to_excel(writer, "DTI H4", index=False)
        dti_joint.to_excel(writer, "H4 joint", index=False)

    print("\n" + "=" * 100)
    print("INCOME-TAIL SENSITIVITY — POOLED")
    print("=" * 100)
    print(
        pooled[["term", "estimate_pp", "std_error_pp", "p_value"]]
        .to_string(index=False)
    )
    print("\nH3 joint tests:")
    print(annual_joint[["test", "p_value"]].to_string(index=False))
    print("\nH4 joint tests:")
    print(dti_joint[["test", "p_value"]].to_string(index=False))
    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
