from __future__ import annotations

import gc
import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import pyfixest as pf


# LOAD VERIFIED SAMPLE BUILDER

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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "logit_probability_contrast_robustness"
OUTPUT_XLSX = OUTPUT_DIR / "LOGIT_PROBABILITY_CONTRAST_ROBUSTNESS.xlsx"

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_PRIMARY_RACE_N = 9_395_768
EXPECTED_COMMON_SAMPLE_N = 9_200_321

RACES = ["Black", "Hispanic", "Asian"]

FOCAL_TERMS = [
    "Black",
    "Hispanic",
    "Asian",
    "NonDirect",
    "Black_x_NonDirect",
    "Hispanic_x_NonDirect",
    "Asian_x_NonDirect",
]

CONTINUOUS_TERMS = [
    "income_asinh",
    "log_loan_amount",
    "log_property_value",
]

CHUNK_SIZE = 500_000


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


def logistic(z):
    """
    Numerically stable logistic CDF.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)

    return out


def get_coef_vcov(fit):
    coef = fit.coef()

    if not isinstance(coef, pd.Series):
        coef = pd.Series(coef)

    names = list(coef.index)
    beta = coef.to_numpy(dtype=np.float64)
    V = np.asarray(fit._vcov, dtype=np.float64)

    if V.shape != (len(names), len(names)):
        raise RuntimeError(
            "Coefficient vector and covariance matrix do not align."
        )

    return names, beta, V


def tidy_fit(fit):
    t = fit.tidy().copy().reset_index()

    first = t.columns[0]
    if first != "term":
        t = t.rename(columns={first: "term"})

    t = t.rename(columns={
        "Estimate": "estimate",
        "Std. Error": "std_error",
        "t value": "t_value",
        "z value": "z_value",
        "Pr(>|t|)": "p_value",
        "Pr(>|z|)": "p_value",
        "2.5%": "ci_low",
        "97.5%": "ci_high",
    })

    if "p_value" in t.columns:
        t["significance"] = t["p_value"].apply(stars)

    return t


# BUILD EXACT M5 DATA WITH EXPLICIT RACE / CHANNEL TERMS

def prepare_logit_data(reg):
    """
    The verified H3 builder stores race×year and channel×year FE codes to make
    the annual model memory-safe. Recover the underlying race and channel codes,
    then convert the compact dataframe into the pooled M5 parameterisation.
    """

    race_year = reg["race_year_fe"].to_numpy(np.int16)
    channel_year = reg["channel_year_fe"].to_numpy(np.int16)

    year_from_race = race_year % 10
    year_from_channel = channel_year % 10

    if not np.array_equal(year_from_race, year_from_channel):
        raise RuntimeError(
            "Race-year and channel-year encodings disagree."
        )

    race_code = (race_year // 10).astype(np.int8)
    nd = (channel_year // 10).astype(np.int8)

    reg["Black"] = (race_code == 1).astype(np.int8)
    reg["Hispanic"] = (race_code == 2).astype(np.int8)
    reg["Asian"] = (race_code == 3).astype(np.int8)
    reg["NonDirect"] = nd

    reg["Black_x_NonDirect"] = (
        reg["Black"].to_numpy(np.int8) * nd
    ).astype(np.int8)

    reg["Hispanic_x_NonDirect"] = (
        reg["Hispanic"].to_numpy(np.int8) * nd
    ).astype(np.int8)

    reg["Asian_x_NonDirect"] = (
        reg["Asian"].to_numpy(np.int8) * nd
    ).astype(np.int8)

    annual_cols = [
        c for c in reg.columns
        if "_x_NonDirect_x_Y" in c
    ]

    drop_cols = annual_cols + [
        c for c in ["race_year_fe", "channel_year_fe"]
        if c in reg.columns
    ]

    reg.drop(columns=drop_cols, inplace=True)
    gc.collect()

    return reg


def build_formula():
    explicit = FOCAL_TERMS + CONTINUOUS_TERMS

    fixed_effects = [
        "lender_year_fe",
        "county_fe",
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


# FIXED-EFFECTS LOGIT

def fit_logit(formula, reg):
    sig = inspect.signature(pf.feglm)
    params = sig.parameters

    kwargs = {
        "fml": formula,
        "data": reg,
        "family": "logit",
        "vcov": {"CRV1": "lender_cluster"},
    }

    optional = {
        "copy_data": False,
        "store_data": True,   # needed for fitted link predictions
        "lean": False,        # needed for post-estimation
        "fixef_rm": "singleton",
        "iwls_tol": 1e-8,
        "iwls_maxiter": 50,
        "accelerate": True,
    }

    for key, value in optional.items():
        if key in params:
            kwargs[key] = value

    print("\nLogit estimation arguments:")
    for key, value in kwargs.items():
        if key != "data":
            print(f"  {key}: {value}")

    return pf.feglm(**kwargs)


# PROBABILITY-SCALE RACE x CHANNEL CONTRASTS

def average_probability_contrast(
    race,
    est_data,
    eta_obs,
    names,
    beta,
    V,
    n_clusters,
):


    idx = {name: i for i, name in enumerate(names)}

    race_term = race
    interaction_term = f"{race}_x_NonDirect"

    required = [
        race_term,
        "NonDirect",
        interaction_term,
        *CONTINUOUS_TERMS,
    ]

    missing = [term for term in required if term not in idx]

    if missing:
        raise KeyError(
            "Required logit coefficient(s) missing:\n"
            + "\n".join(f"  - {x}" for x in missing)
            + "\n\nThis can occur if a term is perfectly collinear or separated."
        )

    # Remove the observed focal Race/Channel contribution from the fitted linear predictor. The remainder contains all controls + HDFE.
    focal_contribution = np.zeros(len(est_data), dtype=np.float64)

    for term in FOCAL_TERMS:
        if term not in idx:
            continue

        if term not in est_data.columns:
            raise KeyError(
                f"Structural term {term!r} is absent from estimation data."
            )

        focal_contribution += (
            beta[idx[term]]
            * est_data[term].to_numpy(np.float64, copy=False)
        )

    base_eta = eta_obs - focal_contribution

    b_r = beta[idx[race_term]]
    b_nd = beta[idx["NonDirect"]]
    b_int = beta[idx[interaction_term]]

    N = len(est_data)
    if len(base_eta) != N:
        raise RuntimeError(
            "Fitted linear predictor length does not match estimation data."
        )

    contrast_sum = 0.0
    grad_sum = np.zeros(len(names), dtype=np.float64)

    for start in range(0, N, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N)
        b = base_eta[start:stop]

        # White / Direct
        eta_00 = b

        # White / Non-direct
        eta_01 = b + b_nd

        # Target race / Direct
        eta_10 = b + b_r

        # Target race / Non-direct
        eta_11 = b + b_r + b_nd + b_int

        p00 = logistic(eta_00)
        p01 = logistic(eta_01)
        p10 = logistic(eta_10)
        p11 = logistic(eta_11)

        contrast = p11 - p10 - p01 + p00
        contrast_sum += float(np.sum(contrast))

        # Logistic derivative p(1-p).
        d00 = p00 * (1.0 - p00)
        d01 = p01 * (1.0 - p01)
        d10 = p10 * (1.0 - p10)
        d11 = p11 * (1.0 - p11)

        # Focal coefficients.
        grad_sum[idx[race_term]] += float(
            np.sum(d11 - d10)
        )

        grad_sum[idx["NonDirect"]] += float(
            np.sum(d11 - d01)
        )

        grad_sum[idx[interaction_term]] += float(
            np.sum(d11)
        )

        # Terms belonging to the other racial groups are zero under all four target-race counterfactual scenarios, so their derivatives are zero.

        common_weight = d11 - d10 - d01 + d00

        for term in CONTINUOUS_TERMS:
            x = est_data[term].to_numpy(
                np.float64,
                copy=False,
            )[start:stop]

            grad_sum[idx[term]] += float(
                np.sum(common_weight * x)
            )

        if "Intercept" in idx:
            grad_sum[idx["Intercept"]] += float(
                np.sum(common_weight)
            )

        del (
            p00, p01, p10, p11,
            d00, d01, d10, d11,
            contrast, common_weight,
        )

    estimate = contrast_sum / N
    gradient = grad_sum / N

    variance = float(
        gradient @ V @ gradient
    )

    se = math.sqrt(max(variance, 0.0))

    df = n_clusters - 1

    tval = (
        estimate / se
        if se > 0
        else np.nan
    )

    pval = (
        2 * stats.t.sf(abs(tval), df=df)
        if np.isfinite(tval)
        else np.nan
    )

    crit = stats.t.ppf(0.975, df=df)
    lo = estimate - crit * se
    hi = estimate + crit * se

    return {
        "race_group": race,
        "estimand": (
            "(Race-White approval gap in Non-direct) "
            "- (Race-White approval gap in Direct)"
        ),
        "estimate": estimate,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "ci_low": lo,
        "ci_high": hi,
        "estimate_pp": 100 * estimate,
        "std_error_pp": 100 * se,
        "ci_low_pp": 100 * lo,
        "ci_high_pp": 100 * hi,
        "significance": stars(pval),
        "interpretation": (
            "Race-vs-White gap narrower in non-direct"
            if estimate > 0
            else "Race-vs-White gap wider in non-direct"
            if estimate < 0
            else "No point-estimate channel difference"
        ),
    }


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))

    if not parts:
        raise FileNotFoundError(
            f"No final V1 Parquet parts found:\n{DATA_DIR}"
        )

    print("=" * 108)
    print("FIXED-EFFECTS LOGIT ROBUSTNESS — PROBABILITY-SCALE RACE × CHANNEL CONTRASTS")
    print("=" * 108)
    print(f"Parquet parts: {len(parts)}")
    print(f"PyFixest: {base.package_version('pyfixest')}")
    print("Family: logit")
    print("Inference: CRV1 clustered by lender")
    print("Reference race: NH White")
    print("Reference channel: Direct")

    schema = base.detect_schema(parts)

    print("\nBuilding the exact verified preferred M5 sample...")
    reg, audit, category_maps = base.build_data(
        parts,
        schema,
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
            "stage": "dual50 + four headline race groups",
            "n": audit["primary_race"],
            "expected_n": EXPECTED_PRIMARY_RACE_N,
            "pass": audit["primary_race"] == EXPECTED_PRIMARY_RACE_N,
        },
        {
            "stage": "Preferred M5 complete-case sample before FE separation/singletons",
            "n": audit["common_sample"],
            "expected_n": EXPECTED_COMMON_SAMPLE_N,
            "pass": audit["common_sample"] == EXPECTED_COMMON_SAMPLE_N,
        },
    ])

    print("\nSample replication audit:")
    print(sample_audit.to_string(index=False))

    if not sample_audit["pass"].all():
        raise RuntimeError(
            "Logit robustness does not reproduce the verified M5 sample. "
            "Do not estimate until this is resolved."
        )

    reg = prepare_logit_data(reg)

    mem_gb = (
        reg.memory_usage(deep=True).sum()
        / (1024 ** 3)
    )

    print(
        f"\nCompact pooled-logit dataframe: "
        f"{len(reg):,} rows | {mem_gb:.2f} GB"
    )

    formula = build_formula()

    print("\nFormula:")
    print(formula)

    print(
        "\nEstimating fixed-effects logit. "
        "This is computationally heavier than the LPM because the model "
        "uses iterative weighted least squares."
    )

    fit = fit_logit(formula, reg)

    # Retrieve the actual estimation data after any GLM separation/singleton handling. PyFixest stores it because store_data=True.
    est_data = getattr(fit, "_data", None)

    if not isinstance(est_data, pd.DataFrame):
        fit_n = int(getattr(fit, "_N", len(reg)))
        if fit_n != len(reg):
            raise RuntimeError(
                "PyFixest dropped observations but the post-estimation "
                "estimation dataframe is unavailable. Please send model_info "
                "and the console output rather than guessing."
            )
        est_data = reg

    fit_n = int(getattr(fit, "_N", len(est_data)))

    if fit_n != len(est_data):
        raise RuntimeError(
            f"fit._N={fit_n:,} but stored estimation data has "
            f"{len(est_data):,} rows."
        )

    n_clusters = int(
        est_data["lender_cluster"].nunique()
    )

    print(
        f"\nActual logit estimation observations: {fit_n:,}"
    )
    print(
        f"Lender clusters in logit estimation sample: {n_clusters:,}"
    )

    names, beta, V = get_coef_vcov(fit)

    print("\nEstimated structural coefficients:")
    print(names)

    # Fitted linear predictor for actual estimation observations.
    eta_obs = np.asarray(
        fit.predict(type="link"),
        dtype=np.float64,
    ).reshape(-1)

    if len(eta_obs) != fit_n:
        raise RuntimeError(
            "Logit fitted link predictions do not align with fit._N."
        )

    rows = []

    print("\nComputing average probability-scale Race × Channel contrasts...")

    for race in RACES:
        result = average_probability_contrast(
            race=race,
            est_data=est_data,
            eta_obs=eta_obs,
            names=names,
            beta=beta,
            V=V,
            n_clusters=n_clusters,
        )
        rows.append(result)

        print(
            f"  {race:9s}: "
            f"{result['estimate_pp']:+.4f} pp "
            f"(SE {result['std_error_pp']:.4f}), "
            f"p={result['p_value']:.6g}"
        )

    contrasts = pd.DataFrame(rows)

    tidy = tidy_fit(fit)
    link_terms = tidy.loc[
        tidy["term"].isin(FOCAL_TERMS)
    ].copy()

    # Adding link-scale odds ratios as a diagnostic
    if "estimate" in link_terms.columns:
        link_terms["odds_ratio"] = np.exp(
            link_terms["estimate"]
        )

    model_info = pd.DataFrame([{
        "preferred_sample_before_logit_fe_processing": EXPECTED_COMMON_SAMPLE_N,
        "logit_estimation_observations": fit_n,
        "observations_removed_by_logit_fe_separation_or_singletons": (
            EXPECTED_COMMON_SAMPLE_N - fit_n
        ),
        "percent_preferred_sample_retained": (
            100 * fit_n / EXPECTED_COMMON_SAMPLE_N
        ),
        "lender_clusters": n_clusters,
        "family": "logit",
        "cluster_vcov": "CRV1 by lender",
        "probability_estimand": (
            "Average [Race-White gap in Non-direct - Race-White gap in Direct]"
        ),
        "delta_method_note": (
            "SE uses lender-clustered covariance of structural coefficients; "
            "high-dimensional fixed effects treated as nuisance parameters."
        ),
        "formula": formula,
        "dataframe_gb_before_fit": mem_gb,
        "pyfixest_version": base.package_version("pyfixest"),
        "converged_attribute": str(
            getattr(fit, "_converged", getattr(fit, "converged", "not exposed"))
        ),
        "iterations_attribute": str(
            getattr(
                fit,
                "_iterations",
                getattr(fit, "_n_iter", getattr(fit, "iterations", "not exposed"))
            )
        ),
    }])

    # Extend audit with actual estimation N.
    sample_audit = pd.concat([
        sample_audit,
        pd.DataFrame([{
            "stage": "Actual fixed-effects logit estimation sample",
            "n": fit_n,
            "expected_n": np.nan,
            "pass": True,
        }]),
    ], ignore_index=True)

    tables = {
        "logit_probability_contrasts": contrasts,
        "logit_link_coefficients": link_terms,
        "logit_model_info": model_info,
        "logit_sample_audit": sample_audit,
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
            f"Workbook not written ({exc}); CSV outputs were written."
        )

    print("\n" + "=" * 108)
    print("LOGIT ROBUSTNESS COMPLETE")
    print("=" * 108)

    print("\nProbability-scale contrasts:")
    print(
        contrasts[
            [
                "race_group",
                "estimate_pp",
                "std_error_pp",
                "p_value",
                "ci_low_pp",
                "ci_high_pp",
                "significance",
            ]
        ].to_string(index=False)
    )

    print(
        "\nInterpretation rule: these probability-scale contrasts are the "
        "nonlinear counterparts of the pooled LPM Race × NonDirect "
        "interaction. A value near zero means the adjusted racial gap is "
        "similar across direct and non-direct channels on average."
    )

    print(
        "\nDo NOT interpret the raw link-scale logit interaction coefficient "
        "as a percentage-point effect."
    )

    print(f"\nOutput folder:\n  {OUTPUT_DIR}")
    print(f"\nWorkbook:\n  {excel_msg}")

    print("\nSEND BACK:")
    print("  1. logit_probability_contrasts.csv")
    print("  2. logit_link_coefficients.csv")
    print("  3. logit_model_info.csv")
    print("  4. logit_sample_audit.csv")


if __name__ == "__main__":
    main()
