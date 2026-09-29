#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:52:40 2026

@author: gayukimanikkuge
"""

from pathlib import Path
from collections import Counter
import pandas as pd
import pyarrow.parquet as pq


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

files = sorted(
    DATA_DIR.glob("*.parquet")
)

print("Files:", len(files))


# CHECKING WHICH VARIABLES ACTUALLY EXIST

schema = pq.read_schema(
    files[0]
)

available = set(
    schema.names
)

wanted = [

    "applicant_sex",
    "co_applicant_sex",

    "applicant_age",
    "co_applicant_age",

    "debt_to_income_ratio",

    "state_code",
    "county_code",

    "sex_category",
    "coapplicant_present",
    "age_category",
    "dti_category",

    "state_clean",
    "county_clean",

    "main_analysis_sample"
]


cols = [
    c
    for c in wanted
    if c in available
]


print("\nColumns available:")

for c in cols:
    print("  ", c)


# COUNTERS

app_sex = Counter()
coapp_sex = Counter()

app_age = Counter()
coapp_age = Counter()

dti = Counter()

sex_mapping = Counter()
age_mapping = Counter()
dti_mapping = Counter()

coapp_mapping = Counter()


# Important diagnostics

sex5_wrong = 0
sex_present_wrong = 0

age9999_wrong = 0

age8888_rows = 0
age8888_present = 0

missing_sex_rows = 0
missing_sex_age8888 = 0
missing_sex_age8888_main = 0

total = 0


# PROCESS

for i, file in enumerate(
    files,
    start=1
):

    d = pd.read_parquet(
        file,
        columns=cols
    )

    total += len(d)


    # RAW VALUE COUNTS

    if "applicant_sex" in d:

        app_sex.update(
            d[
                "applicant_sex"
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )


    if "co_applicant_sex" in d:

        coapp_sex.update(
            d[
                "co_applicant_sex"
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )


    if "applicant_age" in d:

        app_age.update(
            d[
                "applicant_age"
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )


    if "co_applicant_age" in d:

        coapp_age.update(
            d[
                "co_applicant_age"
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )


    if "debt_to_income_ratio" in d:

        dti.update(
            d[
                "debt_to_income_ratio"
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )


    # SEX CODING CROSS-CHECK

    if (
        "applicant_sex" in d
        and
        "sex_category" in d
    ):

        x = (
            d[
                [
                    "applicant_sex",
                    "sex_category"
                ]
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
        )

        sex_mapping.update(
            x.to_dict()
        )


    # AGE CODING CROSS-CHECK

    if (
        "applicant_age" in d
        and
        "age_category" in d
    ):

        x = (
            d[
                [
                    "applicant_age",
                    "age_category"
                ]
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
        )

        age_mapping.update(
            x.to_dict()
        )


    # DTI CODING CROSS-CHECK

    if (
        "debt_to_income_ratio" in d
        and
        "dti_category" in d
    ):

        x = (
            d[
                [
                    "debt_to_income_ratio",
                    "dti_category"
                ]
            ]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
        )

        dti_mapping.update(
            x.to_dict()
        )


    # CO-APPLICANT CHECK

    if (
        "co_applicant_sex" in d
        and
        "coapplicant_present" in d
    ):

        s = (
            d[
                "co_applicant_sex"
            ]
            .astype("string")
            .str.strip()
        )

        present = (
            pd.to_numeric(
                d[
                    "coapplicant_present"
                ],
                errors="coerce"
            )
        )


        # Sex code 5 MUST mean no co-applicant

        sex5_wrong += int(
            (
                s.eq("5")
                &
                present.ne(0)
            ).sum()
        )


        # Sex codes indicating an actual co-app field should normally be coded present

        sex_present_wrong += int(
            (
                s.isin(
                    [
                        "1",
                        "2",
                        "3",
                        "4",
                        "6"
                    ]
                )
                &
                present.ne(1)
            ).sum()
        )


        missing_sex = (
            s.isna()
        )

        missing_sex_rows += int(
            missing_sex.sum()
        )


        if "co_applicant_age" in d:

            a = (
                d[
                    "co_applicant_age"
                ]
                .astype("string")
                .str.strip()
            )


            # 9999 means NO co-applicant

            age9999_wrong += int(
                (
                    a.eq("9999")
                    &
                    present.ne(0)
                ).sum()
            )


            # 8888 is NOT APPLICABLE, not proof that a co-applicant exists

            age8888 = (
                a.eq("8888")
            )

            age8888_rows += int(
                age8888.sum()
            )

            age8888_present += int(
                (
                    age8888
                    &
                    present.eq(1)
                ).sum()
            )


            ambiguous = (
                missing_sex
                &
                age8888
            )

            missing_sex_age8888 += int(
                ambiguous.sum()
            )


            if "main_analysis_sample" in d:

                main = (
                    pd.to_numeric(
                        d[
                            "main_analysis_sample"
                        ],
                        errors="coerce"
                    )
                    .eq(1)
                )

                missing_sex_age8888_main += int(
                    (
                        ambiguous
                        &
                        main
                    ).sum()
                )


    if (
        i % 50 == 0
        or
        i == len(files)
    ):

        print(
            f"Checked {i}/{len(files)}"
        )


# PRINT RESULTS

print("\n" + "=" * 70)
print("HMDA CODE VERIFICATION")
print("=" * 70)

print(
    "\nTotal observations:",
    f"{total:,}"
)


print("\nAPPLICANT SEX RAW VALUES")

for k, v in app_sex.most_common():
    print(k, f"{v:,}")


print("\nCO-APPLICANT SEX RAW VALUES")

for k, v in coapp_sex.most_common():
    print(k, f"{v:,}")


print("\nAPPLICANT AGE RAW VALUES")

for k, v in app_age.most_common():
    print(k, f"{v:,}")


print("\nCO-APPLICANT AGE RAW VALUES")

for k, v in coapp_age.most_common():
    print(k, f"{v:,}")


print("\nDTI RAW VALUES")

for k, v in dti.most_common():
    print(k, f"{v:,}")


print("\n" + "=" * 70)
print("CO-APPLICANT DIAGNOSTICS")
print("=" * 70)

print(
    "Sex code 5 but present != 0:",
    f"{sex5_wrong:,}"
)

print(
    "Sex 1/2/3/4/6 but present != 1:",
    f"{sex_present_wrong:,}"
)

print(
    "Age 9999 but present != 0:",
    f"{age9999_wrong:,}"
)

print(
    "All age-8888 rows:",
    f"{age8888_rows:,}"
)

print(
    "Age 8888 currently coded present:",
    f"{age8888_present:,}"
)

print(
    "Missing co-app sex rows:",
    f"{missing_sex_rows:,}"
)

print(
    "Missing sex + age 8888:",
    f"{missing_sex_age8888:,}"
)

print(
    "Missing sex + age 8888 "
    "inside MAIN SAMPLE:",
    f"{missing_sex_age8888_main:,}"
)


print("\n" + "=" * 70)
print("APPLICANT SEX -> CLEAN CATEGORY")
print("=" * 70)

for key, n in sorted(
    sex_mapping.items(),
    key=lambda x: str(x[0])
):
    print(
        key,
        f"{n:,}"
    )


print("\n" + "=" * 70)
print("APPLICANT AGE -> CLEAN CATEGORY")
print("=" * 70)

for key, n in sorted(
    age_mapping.items(),
    key=lambda x: str(x[0])
):
    print(
        key,
        f"{n:,}"
    )


print("\n" + "=" * 70)
print("DTI -> CLEAN CATEGORY")
print("=" * 70)

for key, n in sorted(
    dti_mapping.items(),
    key=lambda x: str(x[0])
):
    print(
        key,
        f"{n:,}"
    )


print("\nDONE.")