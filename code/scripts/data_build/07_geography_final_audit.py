"""
Created on Sun Aug 16 18:47:14 2026

@author: gayukimanikkuge
"""

from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq


# PATHS

PROJECT_DIR = Path(
    "/Users/gayukimanikkuge/Desktop/HMDA_Dissertation"
)

DATA_DIR = (
    PROJECT_DIR
    / "final_data"
    / "channel_analysis_v1_parts"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_XLSX = (
    OUTPUT_DIR
    / "GEOGRAPHY_FINAL_AUDIT.xlsx"
)


STATE_FIPS = {

    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",

    # Island areas / territories
    "AS": "60",
    "FM": "64",
    "GU": "66",
    "MH": "68",
    "MP": "69",
    "PW": "70",
    "PR": "72",
    "VI": "78",
}


VALID_PREFIXES = set(
    STATE_FIPS.values()
)


# HELPERS

def clean_for_audit(series):
    """
    Minimal normalisation only.

    We deliberately do NOT zero-pad or otherwise repair geography here,
    because this is an audit of the already-cleaned Dataset V1.
    """

    s = (
        series
        .astype("string")
        .str.strip()
    )

    s = s.mask(
        s.isin(
            [
                "",
                "nan",
                "NaN",
                "None",
                "<NA>",
            ]
        )
    )

    return s


def pct(n, d):

    if d == 0:
        return 0.0

    return 100 * n / d


# FIND PARQUET FILES

parts = sorted(
    DATA_DIR.glob("*.parquet")
)

if not parts:

    raise FileNotFoundError(
        f"No Parquet files found in:\n{DATA_DIR}"
    )


print(
    "Parquet files found:",
    len(parts)
)


# CHECK AVAILABLE COLUMNS

schema = pq.read_schema(
    parts[0]
)

available_columns = set(
    schema.names
)

print(
    "\nGeography columns available:"
)

for col in [
    "year",
    "state_clean",
    "county_clean",
]:

    print(
        f"{col}:",
        col in available_columns
    )


required = [
    "state_clean",
    "county_clean",
]

missing_required = [
    c
    for c in required
    if c not in available_columns
]

if missing_required:

    raise RuntimeError(
        "Required columns missing: "
        + str(missing_required)
    )


columns_to_read = [
    "state_clean",
    "county_clean",
]

HAS_YEAR = (
    "year"
    in available_columns
)

if HAS_YEAR:

    columns_to_read.append(
        "year"
    )


# GLOBAL COUNTERS

total_rows = 0

state_missing = 0
county_missing = 0

both_missing = 0

state_present_county_missing = 0
state_missing_county_present = 0

both_present = 0


# State checks
invalid_state_rows = 0

state_counts = Counter()
invalid_state_values = Counter()


# County format checks
county_nondigit_rows = 0
county_wrong_length_rows = 0
county_malformed_rows = 0

county_counts = Counter()
malformed_county_values = Counter()


# County prefix checks
invalid_prefix_rows = 0
prefix_mismatch_rows = 0

invalid_prefix_values = Counter()
prefix_mismatch_pairs = Counter()


# Suspicious-value checks
county_00000 = 0
county_99999 = 0
county_88888 = 0

county_last3_000 = 0
county_last3_999 = 0

county_dot_zero = 0


# Year counters
year_stats = defaultdict(
    lambda: Counter()
)


# READ DATASET

for i, part in enumerate(
    parts,
    start=1
):

    d = pd.read_parquet(
        part,
        columns=columns_to_read
    )

    n = len(d)

    total_rows += n


    # NORMALISE FOR AUDIT

    state = clean_for_audit(
        d["state_clean"]
    )

    county = clean_for_audit(
        d["county_clean"]
    )


    # BASIC MISSINGNESS

    s_missing = state.isna()
    c_missing = county.isna()

    s_present = ~s_missing
    c_present = ~c_missing

    both_present_mask = (
        s_present
        &
        c_present
    )

    both_missing_mask = (
        s_missing
        &
        c_missing
    )

    state_missing += int(
        s_missing.sum()
    )

    county_missing += int(
        c_missing.sum()
    )

    both_missing += int(
        both_missing_mask.sum()
    )

    state_present_county_missing += int(
        (
            s_present
            &
            c_missing
        ).sum()
    )

    state_missing_county_present += int(
        (
            s_missing
            &
            c_present
        ).sum()
    )

    both_present += int(
        both_present_mask.sum()
    )


    # STATE VALIDITY

    state_counts.update(
        state.dropna().tolist()
    )

    state_valid = state.isin(
        STATE_FIPS.keys()
    )

    invalid_state = (
        s_present
        &
        ~state_valid
    )

    invalid_state_rows += int(
        invalid_state.sum()
    )

    invalid_state_values.update(
        state[
            invalid_state
        ].dropna().tolist()
    )


    # COUNTY FORMAT

    county_digit = county.str.fullmatch(
        r"\d+",
        na=False
    )

    county_5digit = county.str.fullmatch(
        r"\d{5}",
        na=False
    )

    county_wrong_length = (
        c_present
        &
        county_digit
        &
        ~county_5digit
    )

    county_nondigit = (
        c_present
        &
        ~county_digit
    )

    county_malformed = (
        c_present
        &
        ~county_5digit
    )

    county_wrong_length_rows += int(
        county_wrong_length.sum()
    )

    county_nondigit_rows += int(
        county_nondigit.sum()
    )

    county_malformed_rows += int(
        county_malformed.sum()
    )

    county_counts.update(
        county.dropna().tolist()
    )

    malformed_county_values.update(
        county[
            county_malformed
        ].dropna().tolist()
    )


    # CHECK WHETHER ".0" SURVIVED CLEANING

    county_dot_zero_mask = (
        c_present
        &
        county.str.endswith(
            ".0",
            na=False
        )
    )

    county_dot_zero += int(
        county_dot_zero_mask.sum()
    )


    # COUNTY PREFIX
    # Full county GEOID:
    #   first 2 digits = state FIPS
    #   final 3 digits = county FIPS

    valid_county_for_prefix = (
        c_present
        &
        county_5digit
    )

    observed_prefix = county.str.slice(
        0,
        2
    )

    prefix_known = observed_prefix.isin(
        VALID_PREFIXES
    )

    invalid_prefix = (
        valid_county_for_prefix
        &
        ~prefix_known
    )

    invalid_prefix_rows += int(
        invalid_prefix.sum()
    )

    invalid_prefix_values.update(
        county[
            invalid_prefix
        ].dropna().tolist()
    )


    # STATE VS COUNTY PREFIX

    expected_prefix = state.map(
        STATE_FIPS
    )

    comparison_eligible = (
        both_present_mask
        &
        state_valid
        &
        county_5digit
    )

    mismatch = (
        comparison_eligible
        &
        observed_prefix.ne(
            expected_prefix
        )
    )

    prefix_mismatch_rows += int(
        mismatch.sum()
    )

    if mismatch.any():

        pairs = zip(
            state[mismatch],
            county[mismatch]
        )

        prefix_mismatch_pairs.update(
            pairs
        )


    # SUSPICIOUS COUNTY VALUES
    # These are flagged for inspection.
    # They are NOT silently changed or deleted.

    county_00000 += int(
        county.eq(
            "00000"
        ).sum()
    )

    county_99999 += int(
        county.eq(
            "99999"
        ).sum()
    )

    county_88888 += int(
        county.eq(
            "88888"
        ).sum()
    )

    county_last3_000 += int(
        (
            county_5digit
            &
            county.str.endswith(
                "000",
                na=False
            )
        ).sum()
    )

    county_last3_999 += int(
        (
            county_5digit
            &
            county.str.endswith(
                "999",
                na=False
            )
        ).sum()
    )


    # YEAR-BY-YEAR

    if HAS_YEAR:

        year = (
            pd.to_numeric(
                d["year"],
                errors="coerce"
            )
            .astype("Int64")
        )

        for y in year.dropna().unique():

            ym = year.eq(y)

            year_stats[int(y)][
                "rows"
            ] += int(
                ym.sum()
            )

            year_stats[int(y)][
                "state_missing"
            ] += int(
                (
                    ym
                    &
                    s_missing
                ).sum()
            )

            year_stats[int(y)][
                "county_missing"
            ] += int(
                (
                    ym
                    &
                    c_missing
                ).sum()
            )

            year_stats[int(y)][
                "malformed_county"
            ] += int(
                (
                    ym
                    &
                    county_malformed
                ).sum()
            )

            year_stats[int(y)][
                "invalid_state"
            ] += int(
                (
                    ym
                    &
                    invalid_state
                ).sum()
            )

            year_stats[int(y)][
                "invalid_prefix"
            ] += int(
                (
                    ym
                    &
                    invalid_prefix
                ).sum()
            )

            year_stats[int(y)][
                "prefix_mismatch"
            ] += int(
                (
                    ym
                    &
                    mismatch
                ).sum()
            )


    # PROGRESS

    if (
        i % 50 == 0
        or
        i == len(parts)
    ):

        print(
            f"Processed {i}/{len(parts)} parts"
            f" | rows = {total_rows:,}"
        )


# SUMMARY

print("\n")
print("=" * 80)
print("FINAL GEOGRAPHY AUDIT")
print("=" * 80)

print(
    "\nTotal observations:",
    f"{total_rows:,}"
)


# MISSINGNESS

print("\n")
print("=" * 80)
print("GEOGRAPHY COVERAGE")
print("=" * 80)

print(
    "State missing:",
    f"{state_missing:,}",
    f"({pct(state_missing, total_rows):.4f}%)"
)

print(
    "County missing:",
    f"{county_missing:,}",
    f"({pct(county_missing, total_rows):.4f}%)"
)

print(
    "Both state + county present:",
    f"{both_present:,}",
    f"({pct(both_present, total_rows):.4f}%)"
)

print(
    "Both missing:",
    f"{both_missing:,}"
)

print(
    "State present / county missing:",
    f"{state_present_county_missing:,}"
)

print(
    "State missing / county present:",
    f"{state_missing_county_present:,}"
)


# STATE

print("\n")
print("=" * 80)
print("STATE VALIDITY")
print("=" * 80)

print(
    "Unique non-missing state values:",
    len(state_counts)
)

print(
    "Invalid/unrecognised state rows:",
    f"{invalid_state_rows:,}"
)

if invalid_state_values:

    print(
        "\nINVALID STATE VALUES:"
    )

    for value, count in (
        invalid_state_values
        .most_common(30)
    ):

        print(
            repr(value),
            f"{count:,}"
        )


# COUNTY FORMAT

print("\n")
print("=" * 80)
print("COUNTY FORMAT")
print("=" * 80)

print(
    "Unique non-missing county values:",
    len(county_counts)
)

print(
    "Malformed non-missing county rows:",
    f"{county_malformed_rows:,}"
)

print(
    "  Numeric but not exactly 5 digits:",
    f"{county_wrong_length_rows:,}"
)

print(
    "  Contains non-numeric characters:",
    f"{county_nondigit_rows:,}"
)

print(
    'County values ending in ".0":',
    f"{county_dot_zero:,}"
)

if malformed_county_values:

    print(
        "\nTOP MALFORMED COUNTY VALUES:"
    )

    for value, count in (
        malformed_county_values
        .most_common(30)
    ):

        print(
            repr(value),
            f"{count:,}"
        )


# PREFIX CHECK

print("\n")
print("=" * 80)
print("COUNTY FIPS PREFIX CHECK")
print("=" * 80)

print(
    "Five-digit counties with unrecognised state prefix:",
    f"{invalid_prefix_rows:,}"
)

print(
    "State abbreviation / county prefix mismatches:",
    f"{prefix_mismatch_rows:,}"
)

if prefix_mismatch_pairs:

    print(
        "\nTOP STATE / COUNTY MISMATCHES:"
    )

    for (
        state_value,
        county_value
    ), count in (
        prefix_mismatch_pairs
        .most_common(30)
    ):

        print(
            repr(state_value),
            repr(county_value),
            f"{count:,}"
        )


# SUSPICIOUS VALUES

print("\n")
print("=" * 80)
print("SUSPICIOUS COUNTY VALUES")
print("=" * 80)

print(
    "county = 00000:",
    f"{county_00000:,}"
)

print(
    "county = 99999:",
    f"{county_99999:,}"
)

print(
    "county = 88888:",
    f"{county_88888:,}"
)

print(
    "County code ending 000:",
    f"{county_last3_000:,}"
)

print(
    "County code ending 999:",
    f"{county_last3_999:,}"
)


# STATE DISTRIBUTION

print("\n")
print("=" * 80)
print("STATE COUNTS")
print("=" * 80)

for state_value, count in sorted(
    state_counts.items()
):

    print(
        repr(state_value),
        f"{count:,}"
    )


# YEAR TABLE

year_df = pd.DataFrame()

if HAS_YEAR:

    rows = []

    for year_value in sorted(
        year_stats
    ):

        s = year_stats[
            year_value
        ]

        rows.append(
            {
                "year":
                    year_value,

                "rows":
                    s["rows"],

                "state_missing":
                    s["state_missing"],

                "state_missing_pct":
                    pct(
                        s["state_missing"],
                        s["rows"]
                    ),

                "county_missing":
                    s["county_missing"],

                "county_missing_pct":
                    pct(
                        s["county_missing"],
                        s["rows"]
                    ),

                "malformed_county":
                    s["malformed_county"],

                "invalid_state":
                    s["invalid_state"],

                "invalid_prefix":
                    s["invalid_prefix"],

                "prefix_mismatch":
                    s["prefix_mismatch"],
            }
        )

    year_df = pd.DataFrame(
        rows
    )

    print("\n")
    print("=" * 80)
    print("YEAR-BY-YEAR GEOGRAPHY AUDIT")
    print("=" * 80)

    print(
        year_df.to_string(
            index=False
        )
    )


# BUILD EXCEL OUTPUT

summary_df = pd.DataFrame(
    [
        ["Total observations", total_rows],
        ["State missing", state_missing],
        ["County missing", county_missing],
        ["Both present", both_present],
        ["Both missing", both_missing],

        [
            "State present / county missing",
            state_present_county_missing
        ],

        [
            "State missing / county present",
            state_missing_county_present
        ],

        ["Invalid state rows", invalid_state_rows],

        [
            "Malformed county rows",
            county_malformed_rows
        ],

        [
            "County numeric but wrong length",
            county_wrong_length_rows
        ],

        [
            "County non-numeric",
            county_nondigit_rows
        ],

        [
            "County .0 survived",
            county_dot_zero
        ],

        [
            "Invalid county prefix",
            invalid_prefix_rows
        ],

        [
            "State/county prefix mismatch",
            prefix_mismatch_rows
        ],

        ["County 00000", county_00000],
        ["County 99999", county_99999],
        ["County 88888", county_88888],

        [
            "County ending 000",
            county_last3_000
        ],

        [
            "County ending 999",
            county_last3_999
        ],
    ],
    columns=[
        "check",
        "count"
    ]
)


state_df = pd.DataFrame(
    sorted(
        state_counts.items()
    ),
    columns=[
        "state_clean",
        "count"
    ]
)


county_df = pd.DataFrame(
    county_counts.most_common(),
    columns=[
        "county_clean",
        "count"
    ]
)


invalid_state_df = pd.DataFrame(
    invalid_state_values.most_common(),
    columns=[
        "invalid_state",
        "count"
    ]
)


malformed_df = pd.DataFrame(
    malformed_county_values.most_common(),
    columns=[
        "malformed_county",
        "count"
    ]
)


mismatch_df = pd.DataFrame(
    [
        {
            "state_clean": state_value,
            "county_clean": county_value,
            "count": count,
        }

        for (
            state_value,
            county_value
        ), count in (
            prefix_mismatch_pairs
            .most_common()
        )
    ]
)


with pd.ExcelWriter(
    OUT_XLSX,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    if not year_df.empty:

        year_df.to_excel(
            writer,
            sheet_name="By Year",
            index=False
        )

    state_df.to_excel(
        writer,
        sheet_name="States",
        index=False
    )

    county_df.to_excel(
        writer,
        sheet_name="Counties",
        index=False
    )

    invalid_state_df.to_excel(
        writer,
        sheet_name="Invalid States",
        index=False
    )

    malformed_df.to_excel(
        writer,
        sheet_name="Malformed Counties",
        index=False
    )

    mismatch_df.to_excel(
        writer,
        sheet_name="Prefix Mismatches",
        index=False
    )


# PASS / REVIEW DECISION

critical_errors = (
    invalid_state_rows
    +
    county_malformed_rows
    +
    invalid_prefix_rows
    +
    prefix_mismatch_rows
)


print("\n")
print("=" * 80)
print("AUDIT DECISION")
print("=" * 80)

if critical_errors == 0:

    print(
        "PASS: No structural geography coding errors detected."
    )

else:

    print(
        "REVIEW REQUIRED:",
        f"{critical_errors:,}",
        "potential structural geography issues detected."
    )


print(
    "\nExcel audit written to:"
)

print(
    OUT_XLSX
)

print("\nDONE.")