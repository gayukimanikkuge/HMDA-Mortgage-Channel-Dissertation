"""
Created on Sun Aug 16 18:07:50 2026

@author: gayukimanikkuge
"""

from pathlib import Path
from collections import Counter
import zipfile
import pandas as pd


# PATHS

RAW_DIR = Path(
    "/Users/gayukimanikkuge/Desktop/HMDA_Dissertation/"
    "preapproval_national_data"
)

YEARS = range(2018, 2025)

CHUNKSIZE = 500_000


# VARIABLES TO AUDIT

WANTED = [
    "co_applicant_age",
    "co_applicant_sex",
    "debt_to_income_ratio",
    "submission_of_application",
    "state_code",
    "county_code",
]


# COUNTERS

total_rows = 0

coapp_age_counts = Counter()
coapp_sex_counts = Counter()
dti_counts = Counter()
submission_counts = Counter()
state_counts = Counter()
county_counts = Counter()

# Critical diagnostics
coapp_age_8888 = 0
coapp_age_9999 = 0

# Cases where sex did NOT already determine co-applicant status(the age fallback in classify_coapplicant() could matter)
fallback_8888 = 0
fallback_9999 = 0
fallback_other_age = 0

# Cross-code consistency diagnostics
sex5_age9999 = 0
sex5_age_not9999 = 0
age9999_sex_not5 = 0
age8888_sex5 = 0

# DTI boundary diagnostic
dti_numeric_60 = 0
dti_numeric_60_raw = Counter()


# HELPERS

def normalize_raw(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def detect_separator(zip_path, member_name):

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            first_line = f.readline().decode(
                "utf-8-sig",
                errors="replace"
            )

    candidates = {
        ",": first_line.count(","),
        "|": first_line.count("|"),
        "\t": first_line.count("\t"),
    }

    sep = max(
        candidates,
        key=candidates.get
    )

    print(
        "Detected separator:",
        repr(sep),
        "| counts:",
        candidates
    )

    return sep


def find_data_member(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:

        candidates = [
            info
            for info in z.infolist()
            if (
                not info.is_dir()
                and info.filename.lower().endswith(
                    (".csv", ".txt")
                )
            )
        ]

        if not candidates:
            raise RuntimeError(
                f"No CSV/TXT file found inside {zip_path}"
            )

        member = max(
            candidates,
            key=lambda x: x.file_size
        )

    return member.filename


# MAIN AUDIT

for year in YEARS:

    print("\n")
    print("=" * 80)
    print(f"READING {year}")
    print("=" * 80)

    zip_path = RAW_DIR / f"hmda_{year}.zip"

    if not zip_path.exists():
        print(
            "FILE NOT FOUND:",
            zip_path
        )
        continue

    # Find actual dataset inside ZIP

    member_name = find_data_member(
        zip_path
    )

    print(
        "ZIP member:",
        member_name
    )

    # Detect delimiter

    sep = detect_separator(
        zip_path,
        member_name
    )

    # Read header

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            member_name
        ) as f:

            header = pd.read_csv(
                f,
                sep=sep,
                nrows=0,
                dtype=str
            )

    # Clean possible BOM / whitespace
    header.columns = (
        pd.Index(header.columns)
        .astype(str)
        .str.strip()
        .str.replace(
            "\ufeff",
            "",
            regex=False
        )
    )

    print(
        "Number of columns:",
        len(header.columns)
    )

    found = [
        c
        for c in WANTED
        if c in header.columns
    ]

    missing = [
        c
        for c in WANTED
        if c not in header.columns
    ]

    print(
        "Columns found:",
        found
    )

    if missing:
        print(
            "MISSING:",
            missing
        )

    if not found:

        print("\nFIRST 25 ACTUAL COLUMN NAMES:")

        for c in header.columns[:25]:
            print(
                repr(c)
            )

        continue

    # Re-open ZIP and stream required columns only

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            member_name
        ) as f:

            reader = pd.read_csv(
                f,
                sep=sep,
                usecols=found,
                dtype=str,
                chunksize=CHUNKSIZE,
                low_memory=False
            )

            for chunk_number, chunk in enumerate(
                reader,
                start=1
            ):

                total_rows += len(chunk)

                # CO-APPLICANT SEX

                if "co_applicant_sex" in chunk.columns:

                    sex = normalize_raw(
                        chunk[
                            "co_applicant_sex"
                        ]
                    )

                    coapp_sex_counts.update(
                        sex.dropna().tolist()
                    )

                else:

                    sex = pd.Series(
                        pd.NA,
                        index=chunk.index,
                        dtype="string"
                    )

                # CO-APPLICANT AGE

                if "co_applicant_age" in chunk.columns:

                    age = normalize_raw(
                        chunk[
                            "co_applicant_age"
                        ]
                    )

                    coapp_age_counts.update(
                        age.dropna().tolist()
                    )

                    coapp_age_8888 += int(
                        age.eq("8888").sum()
                    )

                    coapp_age_9999 += int(
                        age.eq("9999").sum()
                    )

                    valid_sex_codes = [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                    ]

                    sex_unresolved = (
                        sex.isna()
                        |
                        ~sex.isin(
                            valid_sex_codes
                        )
                    )

                    fallback_8888 += int(
                        (
                            sex_unresolved
                            &
                            age.eq("8888")
                        ).sum()
                    )

                    fallback_9999 += int(
                        (
                            sex_unresolved
                            &
                            age.eq("9999")
                        ).sum()
                    )

                    age_numeric = pd.to_numeric(
                        age,
                        errors="coerce"
                    )

                    fallback_other_age += int(
                        (
                            sex_unresolved
                            &
                            age_numeric.between(
                                1,
                                120
                            )
                        ).sum()
                    )

                    # SEX x AGE CONSISTENCY

                    sex5_age9999 += int(
                        (
                            sex.eq("5")
                            &
                            age.eq("9999")
                        ).sum()
                    )

                    sex5_age_not9999 += int(
                        (
                            sex.eq("5")
                            &
                            ~age.eq("9999")
                            &
                            age.notna()
                        ).sum()
                    )

                    age9999_sex_not5 += int(
                        (
                            age.eq("9999")
                            &
                            ~sex.eq("5")
                            &
                            sex.notna()
                        ).sum()
                    )

                    age8888_sex5 += int(
                        (
                            age.eq("8888")
                            &
                            sex.eq("5")
                        ).sum()
                    )

                # DTI

                if "debt_to_income_ratio" in chunk.columns:

                    dti_raw = normalize_raw(
                        chunk[
                            "debt_to_income_ratio"
                        ]
                    )

                    dti_counts.update(
                        dti_raw.dropna().tolist()
                    )

                    dti_numeric = pd.to_numeric(
                        dti_raw,
                        errors="coerce"
                    )

                    is_60 = dti_numeric.eq(60)

                    dti_numeric_60 += int(
                        is_60.sum()
                    )

                    if is_60.any():

                        dti_numeric_60_raw.update(
                            dti_raw[
                                is_60
                            ].dropna().tolist()
                        )

                # SUBMISSION OF APPLICATION

                if (
                    "submission_of_application"
                    in chunk.columns
                ):

                    x = normalize_raw(
                        chunk[
                            "submission_of_application"
                        ]
                    )

                    submission_counts.update(
                        x.dropna().tolist()
                    )

                # STATE

                if "state_code" in chunk.columns:

                    x = normalize_raw(
                        chunk[
                            "state_code"
                        ]
                    )

                    state_counts.update(
                        x.dropna().tolist()
                    )

                # COUNTY

                if "county_code" in chunk.columns:

                    x = normalize_raw(
                        chunk[
                            "county_code"
                        ]
                    )

                    county_counts.update(
                        x.dropna().tolist()
                    )

                if chunk_number % 20 == 0:

                    print(
                        "  chunks processed:",
                        chunk_number,
                        "| cumulative raw rows:",
                        f"{total_rows:,}"
                    )


# RESULTS

print("\n\n")
print("=" * 80)
print("RAW HMDA CODE AUDIT")
print("=" * 80)

print(
    "\nTotal raw records scanned:",
    f"{total_rows:,}"
)


# CO-APPLICANT

print("\n")
print("=" * 80)
print("CRITICAL CO-APPLICANT CHECK")
print("=" * 80)

print(
    "Co-applicant age = 8888:",
    f"{coapp_age_8888:,}"
)

print(
    "Co-applicant age = 9999:",
    f"{coapp_age_9999:,}"
)

print("\nAGE FALLBACK ACTUALLY USED:")

print(
    "Sex unresolved + age 8888:",
    f"{fallback_8888:,}"
)

print(
    "Sex unresolved + age 9999:",
    f"{fallback_9999:,}"
)

print(
    "Sex unresolved + numeric age 1-120:",
    f"{fallback_other_age:,}"
)

print("\nSEX / AGE CONSISTENCY:")

print(
    "Sex 5 AND age 9999:",
    f"{sex5_age9999:,}"
)

print(
    "Sex 5 BUT age not 9999:",
    f"{sex5_age_not9999:,}"
)

print(
    "Age 9999 BUT sex not 5:",
    f"{age9999_sex_not5:,}"
)

print(
    "Age 8888 AND sex 5:",
    f"{age8888_sex5:,}"
)


# DTI

print("\n")
print("=" * 80)
print("CRITICAL DTI CHECK")
print("=" * 80)

print(
    "Raw DTI parsed as exact numeric 60:",
    f"{dti_numeric_60:,}"
)

print(
    "\nRaw representations that parsed to 60:"
)

for value, count in (
    dti_numeric_60_raw
    .most_common()
):

    print(
        repr(value),
        f"{count:,}"
    )

print("\nTOP DTI RAW VALUES")

for value, count in (
    dti_counts
    .most_common(40)
):

    print(
        repr(value),
        f"{count:,}"
    )


# OTHER RAW CODES

print("\n")
print("=" * 80)
print("CO-APPLICANT SEX")
print("=" * 80)

for value, count in sorted(
    coapp_sex_counts.items()
):

    print(
        repr(value),
        f"{count:,}"
    )


print("\n")
print("=" * 80)
print("CO-APPLICANT AGE — TOP VALUES")
print("=" * 80)

for value, count in (
    coapp_age_counts
    .most_common(30)
):

    print(
        repr(value),
        f"{count:,}"
    )


print("\n")
print("=" * 80)
print("SUBMISSION OF APPLICATION")
print("=" * 80)

for value, count in sorted(
    submission_counts.items()
):

    print(
        repr(value),
        f"{count:,}"
    )


print("\n")
print("=" * 80)
print("STATE CODE — ALL VALUES")
print("=" * 80)

for value, count in sorted(
    state_counts.items()
):

    print(
        repr(value),
        f"{count:,}"
    )


print("\n")
print("=" * 80)
print("COUNTY CODE — TOP VALUES")
print("=" * 80)

for value, count in (
    county_counts
    .most_common(40)
):

    print(
        repr(value),
        f"{count:,}"
    )


print("\nDONE.")