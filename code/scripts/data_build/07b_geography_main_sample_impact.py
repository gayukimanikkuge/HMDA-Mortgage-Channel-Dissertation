#!/usr/bin/env python3
#coding: utf-8
"""
Created on Sun Aug 16 18:57:36 2026

@author: gayukimanikkuge
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


# PATH

DATA_DIR = Path(
    "/Users/gayukimanikkuge/Desktop/HMDA_Dissertation/"
    "final_data/channel_analysis_v1_parts"
)


# STATE -> FIPS

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

    "AS": "60",
    "FM": "64",
    "GU": "66",
    "MH": "68",
    "MP": "69",
    "PW": "70",
    "PR": "72",
    "VI": "78",
}


# FILES

parts = sorted(
    DATA_DIR.glob("*.parquet")
)

if not parts:

    raise FileNotFoundError(
        f"No parquet files found in {DATA_DIR}"
    )


schema = pq.read_schema(
    parts[0]
)

print(
    "main_analysis_sample available:",
    "main_analysis_sample" in schema.names
)

if "main_analysis_sample" not in schema.names:

    raise RuntimeError(
        "main_analysis_sample not found."
    )


# COUNTERS

total = 0
main_total = 0

all_invalid_geo = 0
main_invalid_geo = 0

main_malformed = 0
main_prefix_mismatch = 0
main_missing_county = 0

by_year = Counter()
invalid_by_year = Counter()

bad_pairs = Counter()


# PROCESS

for i, part in enumerate(
    parts,
    start=1
):

    d = pd.read_parquet(
        part,
        columns=[
            "year",
            "state_clean",
            "county_clean",
            "main_analysis_sample",
        ]
    )

    total += len(d)


    # Normalise

    state = (
        d["state_clean"]
        .astype("string")
        .str.strip()
    )

    county = (
        d["county_clean"]
        .astype("string")
        .str.strip()
    )

    state = state.mask(
        state.isin(
            [
                "",
                "nan",
                "NaN",
                "None",
                "<NA>",
            ]
        )
    )

    county = county.mask(
        county.isin(
            [
                "",
                "nan",
                "NaN",
                "None",
                "<NA>",
            ]
        )
    )


    # Main sample

    main = (
        pd.to_numeric(
            d["main_analysis_sample"],
            errors="coerce"
        )
        .fillna(0)
        .eq(1)
    )

    main_total += int(
        main.sum()
    )


    # Geography validity

    state_valid = state.isin(
        STATE_FIPS.keys()
    )

    county_5digit = county.str.fullmatch(
        r"\d{5}",
        na=False
    )

    expected_prefix = state.map(
        STATE_FIPS
    )

    observed_prefix = county.str.slice(
        0,
        2
    )

    prefix_matches = observed_prefix.eq(
        expected_prefix
    )


    # VALID COUNTY FOR FIXED EFFECTS
    # Requirements:
    #   state known and valid
    #   county exactly five digits
    #   state/county FIPS agree
    # No attempt is made to "repair" mismatches.

    valid_geo = (
        state.notna()
        &
        county.notna()
        &
        state_valid
        &
        county_5digit
        &
        prefix_matches
        &
        ~county.isin(
            [
                "00000",
                "88888",
                "99999",
            ]
        )
    )


    invalid_geo = ~valid_geo


    # Counts

    all_invalid_geo += int(
        invalid_geo.sum()
    )

    main_invalid = (
        main
        &
        invalid_geo
    )

    main_invalid_geo += int(
        main_invalid.sum()
    )


    malformed = (
        county.notna()
        &
        ~county_5digit
    )

    main_malformed += int(
        (
            main
            &
            malformed
        ).sum()
    )


    mismatch = (
        state.notna()
        &
        county.notna()
        &
        state_valid
        &
        county_5digit
        &
        ~prefix_matches
    )

    main_prefix_mismatch += int(
        (
            main
            &
            mismatch
        ).sum()
    )


    main_missing_county += int(
        (
            main
            &
            county.isna()
        ).sum()
    )


    # By year

    year = pd.to_numeric(
        d["year"],
        errors="coerce"
    )

    for y in sorted(
        year.dropna().unique()
    ):

        ym = year.eq(y)

        by_year[int(y)] += int(
            (
                main
                &
                ym
            ).sum()
        )

        invalid_by_year[int(y)] += int(
            (
                main_invalid
                &
                ym
            ).sum()
        )


    # Bad state/county pairs in main sample

    bad_pair_mask = (
        main
        &
        mismatch
    )

    if bad_pair_mask.any():

        bad_pairs.update(
            zip(
                state[
                    bad_pair_mask
                ].tolist(),

                county[
                    bad_pair_mask
                ].tolist(),
            )
        )


    if (
        i % 50 == 0
        or
        i == len(parts)
    ):

        print(
            f"Processed {i}/{len(parts)}"
        )


# RESULTS

print("\n")
print("=" * 80)
print("GEOGRAPHY IMPACT ON MAIN ANALYSIS SAMPLE")
print("=" * 80)


print(
    "\nDataset observations:",
    f"{total:,}"
)

print(
    "Main analysis sample:",
    f"{main_total:,}"
)


print("\n")
print("=" * 80)
print("MAIN-SAMPLE GEOGRAPHY")
print("=" * 80)


print(
    "Main observations with missing county:",
    f"{main_missing_county:,}",
    f"({100 * main_missing_county / main_total:.5f}%)"
)

print(
    "Main observations with malformed county:",
    f"{main_malformed:,}"
)

print(
    "Main state/county prefix mismatches:",
    f"{main_prefix_mismatch:,}"
)

print(
    "Main observations NOT valid for county FE:",
    f"{main_invalid_geo:,}",
    f"({100 * main_invalid_geo / main_total:.5f}%)"
)

print(
    "Main observations VALID for county FE:",
    f"{main_total - main_invalid_geo:,}",
    f"({100 * (main_total - main_invalid_geo) / main_total:.5f}%)"
)


print("\n")
print("=" * 80)
print("BY YEAR — MAIN SAMPLE")
print("=" * 80)


for year in sorted(
    by_year
):

    n = by_year[
        year
    ]

    bad = invalid_by_year[
        year
    ]

    print(
        year,
        "| main:",
        f"{n:,}",
        "| invalid/missing geo:",
        f"{bad:,}",
        "| share:",
        f"{100 * bad / n:.5f}%"
    )


print("\n")
print("=" * 80)
print("TOP INVALID STATE / COUNTY PAIRS IN MAIN SAMPLE")
print("=" * 80)


if not bad_pairs:

    print(
        "None."
    )

else:

    for (
        state_value,
        county_value
    ), count in (
        bad_pairs
        .most_common(30)
    ):

        print(
            repr(state_value),
            repr(county_value),
            f"{count:,}"
        )


print("\nDONE.")