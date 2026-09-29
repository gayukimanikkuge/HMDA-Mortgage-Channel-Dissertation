from __future__ import annotations

import gc
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# CONFIGURATION

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "descriptive_tables"

MAX_PARTS = None

EXPECTED_ELIGIBLE_N = 27_692_586
EXPECTED_DUAL50_N = 11_268_203
EXPECTED_PRIMARY_RACE_N = 9_395_768
EXPECTED_COMMON_SAMPLE_N = 9_200_321

STUDY_YEARS = list(range(2018, 2025))
HEADLINE_RACES = ["White", "Black", "Hispanic", "Asian"]

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


# HELPERS

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


def category_labels(s):
    return clean_string(s).fillna("Missing / Not reported")


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def truthy(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    n = pd.to_numeric(s, errors="coerce")
    if n.notna().any():
        return n.eq(1)

    x = clean_string(s).str.lower()
    return x.isin({"true", "t", "yes", "y", "1"})


def normalize_race(s):
    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)

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


def normalize_channel(s):
    """
    Output:
        0 = Direct
        1 = Non-direct
    """
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")

    out.loc[n.eq(1)] = 0
    out.loc[n.eq(2)] = 1

    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)
    out.loc[x.isin({"direct", "submitted directly"})] = 0
    out.loc[x.isin({
        "non-direct", "non direct", "nondirect",
        "not direct", "not submitted directly", "intermediated",
    })] = 1

    return out


def normalize_approval(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    m = n.isin([0, 1])
    out.loc[m] = n.loc[m]

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


def detect_schema(parts):
    columns = list(pq.ParquetFile(parts[0]).schema_arrow.names)

    schema = {
        role: first_existing(columns, candidates)
        for role, candidates in PREFERRED_COLUMNS.items()
    }

    required = [
        "dual50", "year", "lender", "race", "channel", "approval",
        "income", "dti", "age", "sex", "coapp", "loan_amount",
        "property_value", "loan_type", "county",
    ]

    missing = [role for role in required if schema[role] is None]
    if missing:
        raise KeyError(
            "Required final-V1 field(s) missing:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return schema


class RunningStats:
    """Streaming N, mean and sample SD for finite numeric values."""

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, values):
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            return

        self.n += int(len(x))
        self.sum += float(x.sum(dtype=np.float64))
        self.sumsq += float(np.square(x, dtype=np.float64).sum(dtype=np.float64))

    @property
    def mean(self):
        return self.sum / self.n if self.n else np.nan

    @property
    def sd(self):
        if self.n <= 1:
            return np.nan

        numerator = self.sumsq - (self.sum * self.sum / self.n)
        numerator = max(numerator, 0.0)
        return math.sqrt(numerator / (self.n - 1))


def add_counter(counter, series):
    vc = category_labels(series).value_counts(dropna=False)
    for label, count in vc.items():
        counter[str(label)] += int(count)


def percentage(num, den):
    return 100.0 * num / den if den else np.nan


# AGGREGATION

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = sorted(DATA_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found in:\n{DATA_DIR}")

    if MAX_PARTS is not None:
        parts = parts[:MAX_PARTS]

    schema = detect_schema(parts)

    print("=" * 100)
    print("DISSERTATION DESCRIPTIVE TABLES — EXACT PREFERRED SAMPLE")
    print("=" * 100)
    print(f"Parts: {len(parts)}")
    print(f"Output: {OUTPUT_DIR}\n")

    print("Detected schema:")
    for role, col in schema.items():
        print(f"  {role:20s}: {col}")

    readcols = [
        schema["dual50"], schema["year"], schema["lender"], schema["race"],
        schema["channel"], schema["approval"], schema["income"], schema["dti"],
        schema["age"], schema["sex"], schema["coapp"], schema["loan_amount"],
        schema["property_value"], schema["loan_type"], schema["county"],
    ]
    readcols = list(dict.fromkeys(readcols))

    # Sample flow
    audit = Counter()

    # Preferred sample counts/stats: Total, Direct, Non-direct.
    groups = ["Total", "Direct", "Non-direct"]
    n_group = Counter()
    approval_sum = Counter()
    coapp_sum = Counter()

    continuous = {
        g: {
            "income": RunningStats(),
            "loan_amount": RunningStats(),
            "property_value": RunningStats(),
        }
        for g in groups
    }

    preferred_categories = {
        g: {
            "race": Counter(),
            "dti": Counter(),
            "age": Counter(),
            "sex": Counter(),
            "coapp": Counter(),
            "loan_type": Counter(),
        }
        for g in groups
    }

    # Table 3: Race x channel.
    race_channel_n = Counter()
    race_channel_approval = Counter()

    # Figure / yearly raw summaries.
    eligible_year_n = Counter()
    eligible_year_nd = Counter()
    preferred_year_n = Counter()
    preferred_year_nd = Counter()

    race_channel_year_n = Counter()
    race_channel_year_approval = Counter()

    # Representativeness: full eligible vs dual50.
    repr_n = Counter()
    repr_approval = Counter()
    repr_nd = Counter()

    repr_categories = {
        "Full eligible": {
            "race": Counter(),
            "dti": Counter(),
            "loan_type": Counter(),
        },
        "dual50": {
            "race": Counter(),
            "dti": Counter(),
            "loan_type": Counter(),
        },
    }

    for pno, part in enumerate(parts, 1):
        d = pd.read_parquet(part, columns=readcols)
        audit["eligible"] += len(d)

        # Normalize once.
        race = normalize_race(d[schema["race"]])
        channel = normalize_channel(d[schema["channel"]])
        approval = normalize_approval(d[schema["approval"]])
        year = numeric(d[schema["year"]])

        # Full eligible descriptives / representativeness
        repr_n["Full eligible"] += len(d)

        valid_approval_all = approval.isin([0, 1])
        repr_approval["Full eligible"] += float(
            approval.loc[valid_approval_all].sum()
        )
        repr_approval[("Full eligible", "den")] += int(valid_approval_all.sum())

        valid_channel_all = channel.isin([0, 1])
        repr_nd["Full eligible"] += int(channel.loc[valid_channel_all].eq(1).sum())
        repr_nd[("Full eligible", "den")] += int(valid_channel_all.sum())

        # Race categories for representativeness should preserve all original
        # cleaned V1 race labels, not only headline normalized groups.
        add_counter(
            repr_categories["Full eligible"]["race"],
            d[schema["race"]],
        )
        add_counter(
            repr_categories["Full eligible"]["dti"],
            d[schema["dti"]],
        )
        add_counter(
            repr_categories["Full eligible"]["loan_type"],
            d[schema["loan_type"]],
        )

        # Eligible non-direct share by year.
        valid_year_channel = year.isin(STUDY_YEARS) & valid_channel_all
        for y in STUDY_YEARS:
            my = valid_year_channel & year.eq(y)
            eligible_year_n[y] += int(my.sum())
            eligible_year_nd[y] += int(channel.loc[my].eq(1).sum())

        # dual50
        m50 = truthy(d[schema["dual50"]])
        audit["dual50"] += int(m50.sum())

        d50 = d.loc[m50]
        race50 = race.loc[m50]
        channel50 = channel.loc[m50]
        approval50 = approval.loc[m50]

        repr_n["dual50"] += len(d50)

        va50 = approval50.isin([0, 1])
        repr_approval["dual50"] += float(approval50.loc[va50].sum())
        repr_approval[("dual50", "den")] += int(va50.sum())

        vc50 = channel50.isin([0, 1])
        repr_nd["dual50"] += int(channel50.loc[vc50].eq(1).sum())
        repr_nd[("dual50", "den")] += int(vc50.sum())

        add_counter(
            repr_categories["dual50"]["race"],
            d50[schema["race"]],
        )
        add_counter(
            repr_categories["dual50"]["dti"],
            d50[schema["dti"]],
        )
        add_counter(
            repr_categories["dual50"]["loan_type"],
            d50[schema["loan_type"]],
        )

        # Four headline race groups
        m_primary = m50 & race.notna()
        audit["primary_race"] += int(m_primary.sum())

        # Preferred complete case sample
        income = numeric(d[schema["income"]])
        loan_amount = numeric(d[schema["loan_amount"]])
        property_value = numeric(d[schema["property_value"]])
        lender = clean_string(d[schema["lender"]])
        county = normalize_county(d[schema["county"]])

        valid_county = (
            county.str.fullmatch(r"\d{5}", na=False)
            & ~county.isin({"00000", "88888", "99999"})
        )

        common = (
            m_primary
            & approval.isin([0, 1])
            & channel.isin([0, 1])
            & year.isin(STUDY_YEARS)
            & lender.notna()
            & income.notna() & np.isfinite(income)
            & loan_amount.notna() & np.isfinite(loan_amount) & loan_amount.gt(0)
            & property_value.notna() & np.isfinite(property_value) & property_value.gt(0)
            & valid_county
        )

        audit["common_sample"] += int(common.sum())

        if common.any():
            p = d.loc[common]
            prace = race.loc[common]
            pchannel = channel.loc[common]
            papproval = approval.loc[common]
            pyear = year.loc[common].astype(int)
            pincome = income.loc[common]
            ploan = loan_amount.loc[common]
            pprop = property_value.loc[common]

            # coapp numeric/binary for summary rate; categorical values also preserved in category table.
            coapp_num = pd.to_numeric(p[schema["coapp"]], errors="coerce")
            coapp_true = coapp_num.eq(1)
            if not coapp_num.notna().any():
                cstr = category_labels(p[schema["coapp"]]).str.lower()
                coapp_true = cstr.isin({"1", "yes", "true", "present"})

            masks = {
                "Total": pd.Series(True, index=p.index),
                "Direct": pchannel.eq(0),
                "Non-direct": pchannel.eq(1),
            }

            for g, mg in masks.items():
                idx = mg[mg].index
                n = int(mg.sum())
                n_group[g] += n

                approval_sum[g] += float(papproval.loc[idx].sum())
                coapp_sum[g] += int(coapp_true.loc[idx].sum())

                continuous[g]["income"].add(pincome.loc[idx])
                continuous[g]["loan_amount"].add(ploan.loc[idx])
                continuous[g]["property_value"].add(pprop.loc[idx])

                add_counter(
                    preferred_categories[g]["race"],
                    prace.loc[idx],
                )
                add_counter(
                    preferred_categories[g]["dti"],
                    p.loc[idx, schema["dti"]],
                )
                add_counter(
                    preferred_categories[g]["age"],
                    p.loc[idx, schema["age"]],
                )
                add_counter(
                    preferred_categories[g]["sex"],
                    p.loc[idx, schema["sex"]],
                )
                add_counter(
                    preferred_categories[g]["coapp"],
                    p.loc[idx, schema["coapp"]],
                )
                add_counter(
                    preferred_categories[g]["loan_type"],
                    p.loc[idx, schema["loan_type"]],
                )

            # Race x channel raw approval
            for r in HEADLINE_RACES:
                for ch_code, ch_label in [(0, "Direct"), (1, "Non-direct")]:
                    m = prace.eq(r) & pchannel.eq(ch_code)
                    key = (r, ch_label)
                    race_channel_n[key] += int(m.sum())
                    race_channel_approval[key] += float(
                        papproval.loc[m[m].index].sum()
                    )

            # Preferred channel share by year and raw race x channel x year.
            for y in STUDY_YEARS:
                my = pyear.eq(y)
                preferred_year_n[y] += int(my.sum())
                preferred_year_nd[y] += int(
                    pchannel.loc[my[my].index].eq(1).sum()
                )

                for r in HEADLINE_RACES:
                    for ch_code, ch_label in [(0, "Direct"), (1, "Non-direct")]:
                        m = my & prace.eq(r) & pchannel.eq(ch_code)
                        key = (r, ch_label, y)
                        race_channel_year_n[key] += int(m.sum())
                        race_channel_year_approval[key] += float(
                            papproval.loc[m[m].index].sum()
                        )

        if pno % 50 == 0 or pno == len(parts):
            print(
                f"  parts {pno:>3}/{len(parts)} | "
                f"eligible={audit['eligible']:,} | "
                f"dual50={audit['dual50']:,} | "
                f"primary={audit['primary_race']:,} | "
                f"preferred={audit['common_sample']:,}"
            )

        del d, d50
        gc.collect()

    # AUDIT / TABLE 1

    sample_audit = pd.DataFrame([
        {
            "stage": "Eligible V1 mortgage sample",
            "n": audit["eligible"],
            "expected_n": EXPECTED_ELIGIBLE_N,
            "pass": (
                True if MAX_PARTS is not None
                else audit["eligible"] == EXPECTED_ELIGIBLE_N
            ),
        },
        {
            "stage": "Preferred support restriction: dual50",
            "n": audit["dual50"],
            "expected_n": EXPECTED_DUAL50_N,
            "pass": (
                True if MAX_PARTS is not None
                else audit["dual50"] == EXPECTED_DUAL50_N
            ),
        },
        {
            "stage": "dual50 + four headline race/ethnicity groups",
            "n": audit["primary_race"],
            "expected_n": EXPECTED_PRIMARY_RACE_N,
            "pass": (
                True if MAX_PARTS is not None
                else audit["primary_race"] == EXPECTED_PRIMARY_RACE_N
            ),
        },
        {
            "stage": "Preferred complete-case regression sample",
            "n": audit["common_sample"],
            "expected_n": EXPECTED_COMMON_SAMPLE_N,
            "pass": (
                True if MAX_PARTS is not None
                else audit["common_sample"] == EXPECTED_COMMON_SAMPLE_N
            ),
        },
    ])

    if MAX_PARTS is None and not sample_audit["pass"].all():
        raise RuntimeError(
            "Sample flow does not reproduce the audited regression sample. "
            "Stop before using the tables."
        )

    table1 = sample_audit[["stage", "n"]].copy()
    table1["percent_of_eligible"] = 100 * table1["n"] / audit["eligible"]
    table1["percent_of_previous_stage"] = (
        100 * table1["n"] / table1["n"].shift(1)
    )
    table1.loc[0, "percent_of_previous_stage"] = 100.0

    # TABLE 2 - PREFERRED SAMPLE SUMMARY STATISTICS

    rows = []

    def add_row(section, variable, values, unit=""):
        rows.append({
            "section": section,
            "variable": variable,
            "unit": unit,
            **values,
        })

    add_row(
        "Sample",
        "Observations",
        {g: n_group[g] for g in groups},
        "N",
    )

    add_row(
        "Outcome",
        "Approval rate",
        {
            g: percentage(approval_sum[g], n_group[g])
            for g in groups
        },
        "%",
    )

    add_row(
        "Borrower characteristics",
        "Applicant income — mean",
        {
            g: continuous[g]["income"].mean
            for g in groups
        },
        "$000s (HMDA reported scale)",
    )

    add_row(
        "Borrower characteristics",
        "Applicant income — SD",
        {
            g: continuous[g]["income"].sd
            for g in groups
        },
        "$000s (HMDA reported scale)",
    )

    add_row(
        "Borrower characteristics",
        "Co-applicant present",
        {
            g: percentage(coapp_sum[g], n_group[g])
            for g in groups
        },
        "%",
    )

    add_row(
        "Loan characteristics",
        "Loan amount - mean",
        {
            g: continuous[g]["loan_amount"].mean
            for g in groups
        },
        "HMDA reported units",
    )

    add_row(
        "Loan characteristics",
        "Loan amount — SD",
        {
            g: continuous[g]["loan_amount"].sd
            for g in groups
        },
        "HMDA reported units",
    )

    add_row(
        "Loan characteristics",
        "Property value - mean",
        {
            g: continuous[g]["property_value"].mean
            for g in groups
        },
        "HMDA reported units",
    )

    add_row(
        "Loan characteristics",
        "Property value - SD",
        {
            g: continuous[g]["property_value"].sd
            for g in groups
        },
        "HMDA reported units",
    )

    # Race shares
    for r in HEADLINE_RACES:
        add_row(
            "Race/ethnicity composition",
            f"{r} share",
            {
                g: percentage(
                    preferred_categories[g]["race"][r],
                    n_group[g],
                )
                for g in groups
            },
            "%",
        )

    # Including all observed loan-type and DTI categories in Table 2.
    for var, label in [("loan_type", "Loan type"), ("dti", "DTI")]:
        labels = sorted(
            set().union(*[
                set(preferred_categories[g][var].keys())
                for g in groups
            ])
        )
        for cat in labels:
            add_row(
                label,
                str(cat),
                {
                    g: percentage(
                        preferred_categories[g][var][cat],
                        n_group[g],
                    )
                    for g in groups
                },
                "%",
            )

    table2 = pd.DataFrame(rows)

    # Full categorical shares for appendix
    appendix_cat_rows = []
    for var, title in [
        ("race", "Race/ethnicity"),
        ("dti", "DTI"),
        ("age", "Age"),
        ("sex", "Sex"),
        ("coapp", "Co-applicant"),
        ("loan_type", "Loan type"),
    ]:
        labels = sorted(
            set().union(*[
                set(preferred_categories[g][var].keys())
                for g in groups
            ])
        )

        for cat in labels:
            row = {
                "variable": title,
                "category": cat,
            }
            for g in groups:
                row[f"{g}_n"] = preferred_categories[g][var][cat]
                row[f"{g}_percent"] = percentage(
                    preferred_categories[g][var][cat],
                    n_group[g],
                )
            appendix_cat_rows.append(row)

    appendix_categories = pd.DataFrame(appendix_cat_rows)

    # TABLE 3 RAW RACE X CHANNEL APPROVAL

    white_direct = (
        100 * race_channel_approval[("White", "Direct")]
        / race_channel_n[("White", "Direct")]
    )
    white_nd = (
        100 * race_channel_approval[("White", "Non-direct")]
        / race_channel_n[("White", "Non-direct")]
    )

    table3_rows = []

    for r in HEADLINE_RACES:
        dn = race_channel_n[(r, "Direct")]
        nn = race_channel_n[(r, "Non-direct")]

        da = percentage(race_channel_approval[(r, "Direct")], dn)
        na = percentage(race_channel_approval[(r, "Non-direct")], nn)

        direct_gap = da - white_direct
        nd_gap = na - white_nd
        gap_diff = nd_gap - direct_gap

        table3_rows.append({
            "race_ethnicity": r,
            "direct_n": dn,
            "direct_approval_percent": da,
            "nondirect_n": nn,
            "nondirect_approval_percent": na,
            "direct_gap_vs_white_pp": direct_gap,
            "nondirect_gap_vs_white_pp": nd_gap,
            "raw_race_x_channel_differential_pp": gap_diff,
        })

    table3 = pd.DataFrame(table3_rows)

    # APPENDIX - FULL ELIGIBLE VS DUAL50 REPRESENTATIVENESS

    repr_rows = []

    for sample in ["Full eligible", "dual50"]:
        repr_rows.append({
            "section": "Sample",
            "variable": "Applications",
            "category": "",
            "sample": sample,
            "value": repr_n[sample],
            "unit": "N",
        })

        repr_rows.append({
            "section": "Outcome",
            "variable": "Approval rate",
            "category": "",
            "sample": sample,
            "value": percentage(
                repr_approval[sample],
                repr_approval[(sample, "den")],
            ),
            "unit": "%",
        })

        repr_rows.append({
            "section": "Channel",
            "variable": "Non-direct share",
            "category": "",
            "sample": sample,
            "value": percentage(
                repr_nd[sample],
                repr_nd[(sample, "den")],
            ),
            "unit": "%",
        })

    for var, title in [
        ("race", "Race/ethnicity"),
        ("dti", "DTI"),
        ("loan_type", "Loan type"),
    ]:
        labels = sorted(
            set(repr_categories["Full eligible"][var].keys())
            | set(repr_categories["dual50"][var].keys())
        )

        for cat in labels:
            for sample in ["Full eligible", "dual50"]:
                total = repr_n[sample]
                repr_rows.append({
                    "section": title,
                    "variable": title,
                    "category": cat,
                    "sample": sample,
                    "value": percentage(
                        repr_categories[sample][var][cat],
                        total,
                    ),
                    "unit": "%",
                })

    appendix_repr = pd.DataFrame(repr_rows)

    # FIGURE DATA

    figure1_rows = []

    for y in STUDY_YEARS:
        figure1_rows.append({
            "year": y,
            "sample": "Full eligible",
            "n": eligible_year_n[y],
            "nondirect_n": eligible_year_nd[y],
            "nondirect_share_percent": percentage(
                eligible_year_nd[y],
                eligible_year_n[y],
            ),
        })

        figure1_rows.append({
            "year": y,
            "sample": "Preferred regression sample",
            "n": preferred_year_n[y],
            "nondirect_n": preferred_year_nd[y],
            "nondirect_share_percent": percentage(
                preferred_year_nd[y],
                preferred_year_n[y],
            ),
        })

    figure1 = pd.DataFrame(figure1_rows)

    figure2_rows = []
    for y in STUDY_YEARS:
        for r in HEADLINE_RACES:
            for ch in ["Direct", "Non-direct"]:
                n = race_channel_year_n[(r, ch, y)]
                appr = percentage(
                    race_channel_year_approval[(r, ch, y)],
                    n,
                )
                figure2_rows.append({
                    "year": y,
                    "race_ethnicity": r,
                    "channel": ch,
                    "n": n,
                    "raw_approval_percent": appr,
                })

    figure2 = pd.DataFrame(figure2_rows)

    schema_used = pd.DataFrame([
        {
            "role": role,
            "detected_column": col if col is not None else "NOT FOUND",
        }
        for role, col in schema.items()
    ])

    # SAVE

    outputs = {
        "table1_sample_flow.csv": table1,
        "table2_preferred_summary_statistics.csv": table2,
        "table3_race_channel_raw_approval.csv": table3,
        "appendix_full_vs_dual50_representativeness.csv": appendix_repr,
        "appendix_categorical_shares_preferred.csv": appendix_categories,
        "figure1_channel_share_by_year.csv": figure1,
        "figure2_raw_race_channel_by_year.csv": figure2,
        "sample_audit.csv": sample_audit,
        "schema_used.csv": schema_used,
    }

    for filename, df in outputs.items():
        df.to_csv(OUTPUT_DIR / filename, index=False)

    print("\n" + "=" * 100)
    print("SAMPLE AUDIT")
    print("=" * 100)
    print(sample_audit.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 1 — SAMPLE FLOW")
    print("=" * 100)
    print(table1.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 3 — RAW RACE × CHANNEL APPROVAL")
    print("=" * 100)
    print(table3.to_string(index=False))

    print(f"\nOutputs written to:\n  {OUTPUT_DIR}")

    print("\nSEND BACK FIRST:")
    print("  1. table1_sample_flow.csv")
    print("  2. table2_preferred_summary_statistics.csv")
    print("  3. table3_race_channel_raw_approval.csv")
    print("  4. appendix_full_vs_dual50_representativeness.csv")
    print("  5. sample_audit.csv")
    print("\nThen we will format the final dissertation tables and decide what stays in the main text vs appendix.")


if __name__ == "__main__":
    main()
