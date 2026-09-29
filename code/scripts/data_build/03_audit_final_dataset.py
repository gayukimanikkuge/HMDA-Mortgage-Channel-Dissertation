# 03_audit_final_dataset.py

# FINAL DATASET V1 AUDIT

from pathlib import Path
import pandas as pd
import numpy as np


# 1. PATHS

BASE = (
    Path.home()
    / "Desktop"
    / "HMDA_Dissertation"
)

DATA_DIR = (
    BASE
    / "final_data"
    / "channel_analysis_v1_parts"
)

OUTPUT_DIR = (
    BASE
    / "outputs"
    / "final_dataset_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 2. FILES

files = sorted(
    DATA_DIR.glob(
        "*.parquet"
    )
)

print(
    f"Parquet files found: {len(files):,}"
)

if len(files) == 0:
    raise ValueError(
        "No final Parquet files found."
    )


# 3. STORAGE

year_counts = {}

channel_counts = {}

race_counts = {}

main_race_counts = {}

race_channel = {}

support_year = {}

unique_lenders = set()

main_unique_lenders = set()


# Support counters
support_flags = [

    "dual_channel_ly",
    "dual10",
    "dual25",
    "dual50",

    "bw_common_any",
    "bw_common5",
    "bw_common10",

    "hw_common_any",
    "hw_common5",
    "hw_common10",

    "aw_common_any",
    "aw_common5",
    "aw_common10"
]


support_ly_sets = {
    flag: set()
    for flag in support_flags
}


support_app_counts = {
    flag: 0
    for flag in support_flags
}


# Missingness controls
controls = [

    "income_thousands",
    "dti_category",
    "cltv",
    "age_category",
    "sex_category",
    "coapplicant_present",
    "loan_amount_clean",
    "property_value_clean",
    "county_clean"
]


control_counts = {

    c: {
        "total": 0,
        "usable": 0
    }

    for c in controls
}


# Diagnostics
missing_support_rows = 0

total_rows = 0
main_rows = 0


# 4. PROCESS PARTS

for i, file in enumerate(
    files,
    start=1
):

    d = pd.read_parquet(
        file
    )

    total_rows += len(d)

    main_mask = (
        d[
            "main_analysis_sample"
        ].eq(1)
    )

    main_rows += int(
        main_mask.sum()
    )


    # UNIQUE LENDERS

    unique_lenders.update(
        d[
            "lei_clean"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    main_unique_lenders.update(
        d.loc[
            main_mask,
            "lei_clean"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    # YEAR

    for year, n in (
        d[
            "year"
        ]
        .value_counts()
        .items()
    ):

        year_counts[
            int(year)
        ] = (
            year_counts.get(
                int(year),
                0
            )
            +
            int(n)
        )


    # CHANNEL

    temp = (
        d.groupby(
            [
                "year",
                "channel"
            ]
        )
        .size()
    )

    for (
        year,
        channel
    ), n in temp.items():

        key = (
            int(year),
            str(channel)
        )

        channel_counts[
            key
        ] = (
            channel_counts.get(
                key,
                0
            )
            +
            int(n)
        )


    # RACE

    temp = (
        d.groupby(
            [
                "year",
                "race_group"
            ],
            dropna=False
        )
        .size()
    )

    for (
        year,
        race
    ), n in temp.items():

        key = (
            int(year),
            str(race)
        )

        race_counts[
            key
        ] = (
            race_counts.get(
                key,
                0
            )
            +
            int(n)
        )


    # MAIN SAMPLE RACE

    temp = (
        d.loc[
            main_mask
        ]
        .groupby(
            [
                "year",
                "race_group"
            ]
        )
        .size()
    )

    for (
        year,
        race
    ), n in temp.items():

        key = (
            int(year),
            str(race)
        )

        main_race_counts[
            key
        ] = (
            main_race_counts.get(
                key,
                0
            )
            +
            int(n)
        )


    # MAIN SAMPLE RACE x CHANNEL x APPROVAL

    temp = (

        d.loc[
            main_mask
        ]

        .groupby(
            [
                "year",
                "race_group",
                "channel"
            ]
        )[
            "approval"
        ]

        .agg(
            ["count", "sum"]
        )
    )


    for (
        year,
        race,
        channel
    ), row in temp.iterrows():

        key = (
            int(year),
            str(race),
            str(channel)
        )

        if key not in race_channel:

            race_channel[
                key
            ] = {
                "n": 0,
                "approved": 0
            }

        race_channel[
            key
        ][
            "n"
        ] += int(
            row["count"]
        )

        race_channel[
            key
        ][
            "approved"
        ] += int(
            row["sum"]
        )


    # SUPPORT

    if (
        "dual_channel_ly"
        in d.columns
    ):

        missing_support_rows += int(
            d[
                "dual_channel_ly"
            ]
            .isna()
            .sum()
        )


    for flag in support_flags:

        if flag not in d.columns:
            continue

        flag_mask = (
            d[
                flag
            ].eq(1)
        )

        support_app_counts[
            flag
        ] += int(
            flag_mask.sum()
        )

        support_ly_sets[
            flag
        ].update(
            d.loc[
                flag_mask,
                "lender_year_id"
            ]
            .dropna()
            .astype(str)
            .unique()
        )


    # CONTROL COMPLETENESS

    for c in controls:

        if c not in d.columns:
            continue

        control_counts[
            c
        ][
            "total"
        ] += len(d)


        if c == "dti_category":

            usable = (
                d[c]
                .notna()
                &
                ~d[c].eq(
                    "Missing / Not relied upon"
                )
            )

        elif c == "age_category":

            usable = (
                d[c]
                .notna()
                &
                ~d[c].eq(
                    "Unknown / Not reported"
                )
            )

        elif c == "sex_category":

            usable = (
                d[c]
                .notna()
                &
                ~d[c].eq(
                    "Unknown / Not reported"
                )
            )

        else:

            usable = (
                d[c]
                .notna()
            )


        control_counts[
            c
        ][
            "usable"
        ] += int(
            usable.sum()
        )


    if (
        i % 50 == 0
        or
        i == len(files)
    ):

        print(
            f"Audited "
            f"{i:,}/{len(files):,} parts"
        )


# 5. BUILD OUTPUT TABLES

# OVERALL SUMMARY

overall_df = pd.DataFrame({

    "metric": [

        "Eligible observations",

        "Main analysis observations",

        "Main sample share",

        "Unique eligible lenders",

        "Unique main-sample lenders",

        "Missing support-flag rows"
    ],

    "value": [

        total_rows,

        main_rows,

        (
            main_rows
            /
            total_rows
        ),

        len(
            unique_lenders
        ),

        len(
            main_unique_lenders
        ),

        missing_support_rows
    ]
})


# YEAR

year_df = pd.DataFrame(

    [
        {
            "year":
                year,

            "n":
                n
        }

        for year, n
        in sorted(
            year_counts.items()
        )
    ]
)


# CHANNEL

channel_df = pd.DataFrame(

    [
        {
            "year":
                year,

            "channel":
                channel,

            "n":
                n
        }

        for (
            year,
            channel
        ), n
        in sorted(
            channel_counts.items()
        )
    ]
)


channel_df[
    "year_total"
] = (

    channel_df

    .groupby(
        "year"
    )[
        "n"
    ]

    .transform(
        "sum"
    )
)


channel_df[
    "share"
] = (

    channel_df[
        "n"
    ]

    /
    channel_df[
        "year_total"
    ]
)


# RACE

race_df = pd.DataFrame(

    [
        {
            "year":
                year,

            "race_group":
                race,

            "n":
                n
        }

        for (
            year,
            race
        ), n
        in sorted(
            race_counts.items()
        )
    ]
)


race_df[
    "year_total"
] = (

    race_df

    .groupby(
        "year"
    )[
        "n"
    ]

    .transform(
        "sum"
    )
)


race_df[
    "share"
] = (

    race_df[
        "n"
    ]

    /
    race_df[
        "year_total"
    ]
)


# MAIN RACE

main_race_df = pd.DataFrame(

    [
        {
            "year":
                year,

            "race_group":
                race,

            "n":
                n
        }

        for (
            year,
            race
        ), n
        in sorted(
            main_race_counts.items()
        )
    ]
)


# RACE x CHANNEL APPROVAL

race_channel_rows = []


for (
    year,
    race,
    channel
), values in sorted(
    race_channel.items()
):

    n = (
        values[
            "n"
        ]
    )

    approved = (
        values[
            "approved"
        ]
    )

    race_channel_rows.append({

        "year":
            year,

        "race_group":
            race,

        "channel":
            channel,

        "n":
            n,

        "approved":
            approved,

        "denied":
            n - approved,

        "approval_rate":
            (
                approved
                /
                n

                if n > 0

                else np.nan
            )
    })


race_channel_df = pd.DataFrame(
    race_channel_rows
)


# SUPPORT SUMMARY

support_rows = []


for flag in support_flags:

    support_rows.append({

        "support_definition":
            flag,

        "lender_years":
            len(
                support_ly_sets[
                    flag
                ]
            ),

        "applications":
            support_app_counts[
                flag
            ],

        "application_share":
            (
                support_app_counts[
                    flag
                ]
                /
                total_rows

                if total_rows > 0

                else np.nan
            )
    })


support_df = pd.DataFrame(
    support_rows
)


# CONTROL COVERAGE

control_rows = []


for variable, values in (
    control_counts.items()
):

    total = (
        values[
            "total"
        ]
    )

    usable = (
        values[
            "usable"
        ]
    )

    control_rows.append({

        "variable":
            variable,

        "total":
            total,

        "usable":
            usable,

        "usable_share":
            (
                usable
                /
                total

                if total > 0

                else np.nan
            )
    })


control_df = pd.DataFrame(
    control_rows
)


# 6. SAVE

overall_df.to_csv(
    OUTPUT_DIR
    / "01_overall.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    / "02_year.csv",
    index=False
)

channel_df.to_csv(
    OUTPUT_DIR
    / "03_channel.csv",
    index=False
)

race_df.to_csv(
    OUTPUT_DIR
    / "04_race.csv",
    index=False
)

main_race_df.to_csv(
    OUTPUT_DIR
    / "05_main_race.csv",
    index=False
)

race_channel_df.to_csv(
    OUTPUT_DIR
    / "06_race_channel_approval.csv",
    index=False
)

support_df.to_csv(
    OUTPUT_DIR
    / "07_support.csv",
    index=False
)

control_df.to_csv(
    OUTPUT_DIR
    / "08_control_coverage.csv",
    index=False
)


excel_path = (
    OUTPUT_DIR
    / "FINAL_DATASET_V1_AUDIT.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    overall_df.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    year_df.to_excel(
        writer,
        sheet_name="Year",
        index=False
    )

    channel_df.to_excel(
        writer,
        sheet_name="Channel",
        index=False
    )

    race_df.to_excel(
        writer,
        sheet_name="Race",
        index=False
    )

    main_race_df.to_excel(
        writer,
        sheet_name="Main Race",
        index=False
    )

    race_channel_df.to_excel(
        writer,
        sheet_name="Race Channel Approval",
        index=False
    )

    support_df.to_excel(
        writer,
        sheet_name="Support",
        index=False
    )

    control_df.to_excel(
        writer,
        sheet_name="Controls",
        index=False
    )


# 7. PRINT KEY RESULTS

print("\n")
print("=" * 80)
print("FINAL DATASET V1 AUDIT COMPLETE")
print("=" * 80)


print("\nOVERALL\n")

print(
    overall_df.to_string(
        index=False
    )
)


print("\nSUPPORT\n")

print(
    support_df.to_string(
        index=False
    )
)


print("\nCONTROL COVERAGE\n")

print(
    control_df.to_string(
        index=False
    )
)


print("\nAudit workbook:")

print(
    excel_path
)

print("\nDONE.")