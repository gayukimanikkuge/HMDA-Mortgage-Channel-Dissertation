from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import zipfile
import shutil
import re
import warnings

warnings.filterwarnings("ignore")


# 1. REQUIRE PYARROW

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise SystemExit(
        "\nPyArrow is required for the final Parquet dataset.\n"
        "Install it using:\n\n"
        "    pip install pyarrow\n"
    )


# 2. PROJECT PATHS

BASE_DIR = (
    Path.home()
    / "Desktop"
    / "HMDA_Dissertation"
)

RAW_DIR = (
    BASE_DIR
    / "preapproval_national_data"
)

FINAL_DIR = (
    BASE_DIR
    / "final_data"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "data_construction"
)


# First-pass filtered data
BASE_PARTS_DIR = (
    FINAL_DIR
    / "channel_analysis_base_parts"
)

# Final data with lender-year support flags attached
FINAL_PARTS_DIR = (
    FINAL_DIR
    / "channel_analysis_v1_parts"
)


FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 3. RERUN SETTING
OVERWRITE_OUTPUTS = False


for folder in [
    BASE_PARTS_DIR,
    FINAL_PARTS_DIR
]:

    if folder.exists():

        if OVERWRITE_OUTPUTS:

            shutil.rmtree(
                folder
            )

        else:

            raise SystemExit(
                f"\nOUTPUT FOLDER ALREADY EXISTS:\n"
                f"{folder}\n\n"
                f"If this is an intentional rerun, set:\n"
                f"OVERWRITE_OUTPUTS = True\n"
            )


BASE_PARTS_DIR.mkdir(
    parents=True
)

FINAL_PARTS_DIR.mkdir(
    parents=True
)


# 4. YEARS / RAW FILES

STUDY_YEARS = list(
    range(2018, 2025)
)

YEAR_FILES = {

    year:
        RAW_DIR
        / f"hmda_{year}.zip"

    for year
    in STUDY_YEARS
}


# Conservative chunk size
CHUNK_SIZE = 250_000


# 5. VARIABLES TO READ

RACE_COLS = [
    f"applicant_race_{i}"
    for i in range(1, 6)
]

ETHNICITY_COLS = [
    f"applicant_ethnicity_{i}"
    for i in range(1, 6)
]


WANTED_COLS = [

    # IDs
    "activity_year",
    "lei",

    # Outcome
    "action_taken",

    # Channel
    "submission_of_application",
    "initially_payable_to_institution",

    # Loan
    "loan_type",
    "loan_purpose",
    "loan_amount",
    "lien_status",
    "reverse_mortgage",
    "open_end_line_of_credit",
    "business_or_commercial_purpose",

    # Property
    "occupancy_type",
    "construction_method",
    "total_units",
    "property_value",

    # Borrower
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "applicant_age",
    "applicant_sex",

    # Co-applicant
    "co_applicant_age",
    "co_applicant_sex",

    # Geography
    "state_code",
    "county_code",
    "census_tract",

] + RACE_COLS + ETHNICITY_COLS


# Variables essential to define the main sample
REQUIRED_COLS = [

    "lei",
    "action_taken",
    "loan_purpose",

    "submission_of_application",

    "open_end_line_of_credit",
    "reverse_mortgage",
    "business_or_commercial_purpose",

    "occupancy_type",
    "construction_method",
    "total_units",
    "lien_status",

    "applicant_race_1",
    "applicant_ethnicity_1"
]


# 6. ZIP HELPERS

def find_data_file_inside_zip(zip_path):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        candidates = []

        for info in z.infolist():

            if info.is_dir():
                continue

            lower_name = (
                info.filename.lower()
            )

            if (
                lower_name.endswith(".csv")
                or
                lower_name.endswith(".txt")
            ):

                candidates.append(
                    info
                )

        if not candidates:

            raise ValueError(
                f"No CSV/TXT found inside {zip_path}"
            )

        chosen = max(
            candidates,
            key=lambda x: x.file_size
        )

        return chosen.filename


def detect_separator(
    zip_path,
    inner_file
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            inner_file
        ) as f:

            first_line = (
                f.readline()
                .decode(
                    "utf-8-sig",
                    errors="replace"
                )
            )

    candidates = {

        ",":
            first_line.count(","),

        "|":
            first_line.count("|"),

        "\t":
            first_line.count("\t")
    }

    return max(
        candidates,
        key=candidates.get
    )


def get_columns(
    zip_path,
    inner_file,
    separator
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            inner_file
        ) as f:

            header = pd.read_csv(
                f,
                sep=separator,
                nrows=0,
                encoding="utf-8-sig"
            )

    return header.columns.tolist()


# 7. BASIC CLEANING HELPERS

MISSING_STRINGS = {

    "",
    "NA",
    "N/A",
    "NULL",
    "NONE",
    "NAN",
    "EXEMPT"
}


def clean_string(series):

    s = (
        series
        .astype("string")
        .str.strip()
    )

    upper = (
        s.str.upper()
    )

    s = s.mask(
        upper.isin(
            MISSING_STRINGS
        )
    )

    return s


def numeric(series):

    return pd.to_numeric(
        clean_string(series),
        errors="coerce"
    )
def safe_log_positive(series):
    """
    Natural log for strictly positive numeric values.
    Missing, zero, and negative values become NaN.
    Avoids pandas pd.NA / np.where ambiguity.
    """

    x = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Force ordinary float dtype so missing values are np.nan, not pandas
    x = x.astype("float64")

    result = pd.Series(
        np.nan,
        index=x.index,
        dtype="float64"
    )

    valid = (
        x.notna()
        &
        x.gt(0)
    )

    result.loc[valid] = np.log(
        x.loc[valid]
    )

    return result

def clean_lei(series):

    return (
        clean_string(series)
        .str.upper()
    )


# 8. GEOGRAPHY

def clean_geo_code(
    series,
    width=None
):

    s = clean_string(
        series
    )

    # Remove accidental ".0"
    s = (
        s
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )

    if width is not None:

        valid_numeric = (
            s.str.fullmatch(
                r"\d+",
                na=False
            )
        )

        s.loc[
            valid_numeric
        ] = (
            s.loc[
                valid_numeric
            ]
            .str.zfill(
                width
            )
        )

    return s


# 9. RACE / ETHNICITY

# Using all 5 applicant race fields and all 5 ethnicity fields.


HISPANIC_CODES = {
    1, 11, 12, 13, 14
}

ASIAN_CODES = {
    2, 21, 22, 23,
    24, 25, 26, 27
}

NHPI_CODES = {
    4, 41, 42, 43, 44
}


def classify_race_ethnicity(df):

    # Ensure every one of the five fields exists

    race_data = {}

    for col in RACE_COLS:

        if col in df.columns:

            race_data[col] = (
                numeric(
                    df[col]
                )
            )

        else:

            race_data[col] = pd.Series(
                np.nan,
                index=df.index
            )


    ethnicity_data = {}

    for col in ETHNICITY_COLS:

        if col in df.columns:

            ethnicity_data[col] = (
                numeric(
                    df[col]
                )
            )

        else:

            ethnicity_data[col] = pd.Series(
                np.nan,
                index=df.index
            )


    race = pd.DataFrame(
        race_data,
        index=df.index
    )

    eth = pd.DataFrame(
        ethnicity_data,
        index=df.index
    )


    # Ethnicity

    is_hispanic = (
        eth.isin(
            HISPANIC_CODES
        )
        .any(axis=1)
    )

    is_non_hispanic = (
        eth.eq(2)
        .any(axis=1)
    )

    multiple_ethnicity_flags = (
        is_hispanic
        &
        is_non_hispanic
    )


    # Race families

    has_aian = (
        race.eq(1)
        .any(axis=1)
    )

    has_asian = (
        race.isin(
            ASIAN_CODES
        )
        .any(axis=1)
    )

    has_black = (
        race.eq(3)
        .any(axis=1)
    )

    has_nhpi = (
        race.isin(
            NHPI_CODES
        )
        .any(axis=1)
    )

    has_white = (
        race.eq(5)
        .any(axis=1)
    )


    race_family_count = (

        has_aian.astype(int)
        +
        has_asian.astype(int)
        +
        has_black.astype(int)
        +
        has_nhpi.astype(int)
        +
        has_white.astype(int)
    )


    # Mutually exclusive final category

    result = pd.Series(
        "Unknown / Not reported",
        index=df.index,
        dtype="string"
    )


    # Hispanic takes precedence
    result.loc[
        is_hispanic
    ] = "Hispanic / Latino"


    nh = (
        ~is_hispanic
        &
        is_non_hispanic
    )


    result.loc[
        nh
        &
        race_family_count.eq(1)
        &
        has_white
    ] = "Non-Hispanic White"


    result.loc[
        nh
        &
        race_family_count.eq(1)
        &
        has_black
    ] = "Non-Hispanic Black"


    result.loc[
        nh
        &
        race_family_count.eq(1)
        &
        has_asian
    ] = "Non-Hispanic Asian"


    result.loc[
        nh
        &
        race_family_count.eq(1)
        &
        has_aian
    ] = "Non-Hispanic AIAN"


    result.loc[
        nh
        &
        race_family_count.eq(1)
        &
        has_nhpi
    ] = "Non-Hispanic NHPI"


    result.loc[
        nh
        &
        race_family_count.gt(1)
    ] = "Non-Hispanic Multiracial"


    audit = pd.DataFrame({

        "race_group":
            result,

        "race_family_count":
            race_family_count,

        "hispanic_flag":
            is_hispanic.astype(int),

        "non_hispanic_flag":
            is_non_hispanic.astype(int),

        "multiple_ethnicity_flag":
            multiple_ethnicity_flags.astype(int)
    })


    return audit


# 10. PRIMARY RACE GROUP FLAG

PRIMARY_RACE_GROUPS = {

    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic / Latino",
    "Non-Hispanic Asian"
}


# 11. APPLICANT SEX

def classify_sex(series):

    code = numeric(
        series
    )

    result = pd.Series(
        "Unknown / Not reported",
        index=series.index,
        dtype="string"
    )

    result.loc[
        code.eq(1)
    ] = "Male"

    result.loc[
        code.eq(2)
    ] = "Female"

    result.loc[
        code.eq(6)
    ] = "Both male and female"

    return result


# 12. CO-APPLICANT

def classify_coapplicant(df):

    result = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int8"
    )


    # Best source: co-applicant sex
    if (
        "co_applicant_sex"
        in df.columns
    ):

        sex = numeric(
            df[
                "co_applicant_sex"
            ]
        )

        # HMDA code 5 = no co-applicant
        result.loc[
            sex.eq(5)
        ] = 0

        result.loc[
            sex.isin(
                [1, 2, 3, 4, 6]
            )
        ] = 1


    # Age fallback
    if (
        "co_applicant_age"
        in df.columns
    ):

        age = numeric(
            df[
                "co_applicant_age"
            ]
        )

        result.loc[
            result.isna()
            &
            age.eq(9999)
        ] = 0

        result.loc[
            result.isna()
            &
            (
                age.between(
                    1,
                    120
                )
                |
                age.eq(8888)
            )
        ] = 1


    return result


# 13. AGE CATEGORY

def classify_age(series):

    raw = (
        clean_string(series)
        .str.upper()
    )

    result = pd.Series(
        "Unknown / Not reported",
        index=series.index,
        dtype="string"
    )


    # Public modified-LAR category strings

    category_map = {

        "<25":
            "<25",

        "25-34":
            "25-34",

        "35-44":
            "35-44",

        "45-54":
            "45-54",

        "55-64":
            "55-64",

        "65-74":
            "65-74",

        ">74":
            "75+",

        "75+":
            "75+"
    }


    for raw_value, clean_value in (
        category_map.items()
    ):

        result.loc[
            raw.eq(
                raw_value
            )
        ] = clean_value


    # Handling numeric age if present

    age_num = pd.to_numeric(
        raw,
        errors="coerce"
    )


    valid_age = (
        age_num.between(
            18,
            120
        )
    )


    result.loc[
        valid_age
        &
        age_num.lt(25)
    ] = "<25"


    result.loc[
        valid_age
        &
        age_num.between(
            25,
            34
        )
    ] = "25-34"


    result.loc[
        valid_age
        &
        age_num.between(
            35,
            44
        )
    ] = "35-44"


    result.loc[
        valid_age
        &
        age_num.between(
            45,
            54
        )
    ] = "45-54"


    result.loc[
        valid_age
        &
        age_num.between(
            55,
            64
        )
    ] = "55-64"


    result.loc[
        valid_age
        &
        age_num.between(
            65,
            74
        )
    ] = "65-74"


    result.loc[
        valid_age
        &
        age_num.ge(75)
    ] = "75+"


    return result


# 14. DTI CATEGORY

# Works with:
# - exact numeric DTI
# - modified-LAR ranges such as 20%-<30%
# - 30%-<36%
# - 50%-60%
# - >60%

def classify_dti(series):

    raw = (
        clean_string(series)
        .str.upper()
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    result = pd.Series(
        "Missing / Not relied upon",
        index=series.index,
        dtype="string"
    )


    # Existing range forms

    result.loc[
        raw.str.contains(
            r"20.*30",
            regex=True,
            na=False
        )
    ] = "20-<30"


    result.loc[
        raw.str.contains(
            r"30.*36",
            regex=True,
            na=False
        )
    ] = "30-<36"


    result.loc[
        raw.str.contains(
            r"50.*60",
            regex=True,
            na=False
        )
    ] = "50-60"


    result.loc[
        raw.str.contains(
            r">60|60\+",
            regex=True,
            na=False
        )
    ] = ">60"


    result.loc[
        raw.str.contains(
            r"<20",
            regex=True,
            na=False
        )
    ] = "<20"


    # Exact numbers

    numeric_text = (
        raw
        .str.replace(
            "%",
            "",
            regex=False
        )
    )


    dti = pd.to_numeric(
        numeric_text,
        errors="coerce"
    )


    result.loc[
        dti.lt(20)
    ] = "<20"


    result.loc[
        dti.ge(20)
        &
        dti.lt(30)
    ] = "20-<30"


    result.loc[
        dti.ge(30)
        &
        dti.lt(36)
    ] = "30-<36"


    result.loc[
        dti.ge(36)
        &
        dti.lt(40)
    ] = "36-<40"


    result.loc[
        dti.ge(40)
        &
        dti.lt(43)
    ] = "40-<43"


    result.loc[
        dti.ge(43)
        &
        dti.lt(50)
    ] = "43-<50"


    result.loc[
        dti.ge(50)
        &
        dti.le(60)
    ] = "50-60"


    result.loc[
        dti.gt(60)
    ] = ">60"


    return result


# 14. LOAN TYPE

def classify_loan_type(series):

    code = numeric(
        series
    )

    result = pd.Series(
        "Unknown",
        index=series.index,
        dtype="string"
    )


    result.loc[
        code.eq(1)
    ] = "Conventional"

    result.loc[
        code.eq(2)
    ] = "FHA"

    result.loc[
        code.eq(3)
    ] = "VA"

    result.loc[
        code.eq(4)
    ] = "USDA/RHS/FSA"


    return result


# 15. CHANNEL / PAYABLE STATUS

def classify_payable(series):

    code = numeric(
        series
    )

    result = pd.Series(
        "Unavailable",
        index=series.index,
        dtype="string"
    )

    result.loc[
        code.eq(1)
    ] = "Initially payable"

    result.loc[
        code.eq(2)
    ] = "Not initially payable"

    result.loc[
        code.eq(3)
    ] = "Not applicable"

    result.loc[
        code.eq(1111)
    ] = "Exempt"

    return result


# 16. SAMPLE FLOW STAGES

FLOW_STAGES = [

    "01 National HMDA rows",

    "02 Home purchase",

    "03 Action 1/2/3",

    "04 Closed-end",

    "05 Non-reverse",

    "06 Non-business purpose",

    "07 Principal residence",

    "08 Site-built",

    "09 1-4 units",

    "10 First lien",

    "11 Direct/non-direct channel",

    "12 Substantive race/ethnicity known",

    "13 Primary four race groups"
]


sample_flow = defaultdict(
    lambda:
        defaultdict(int)
)


# 17. OTHER AUDIT STORAGE

race_audit_counts = (
    defaultdict(int)
)

year_final_counts = (
    defaultdict(int)
)

channel_year_counts = (
    defaultdict(int)
)


# lender-year channel counts
ly_channel_counts = defaultdict(
    lambda: {
        "direct_n": 0,
        "non_direct_n": 0
    }
)


# lender-year × race × channel counts
ly_race_channel_counts = (
    defaultdict(int)
)


# Control completeness
CONTROL_NAMES = [

    "income",
    "dti",
    "cltv",
    "loan_amount",
    "property_value",
    "age",
    "sex",
    "coapplicant",
    "county"
]


control_counts = defaultdict(
    lambda: {
        "total": 0,
        "usable": 0
    }
)


# 18. PROCESS RAW HMDA

global_part_number = 0


for year in STUDY_YEARS:

    zip_path = (
        YEAR_FILES[year]
    )


    print("\n")
    print("=" * 90)
    print(
        f"FINAL DATA CONSTRUCTION — {year}"
    )
    print("=" * 90)


    if not zip_path.exists():

        raise FileNotFoundError(
            f"\nMissing ZIP:\n{zip_path}"
        )


    inner_file = (
        find_data_file_inside_zip(
            zip_path
        )
    )


    separator = (
        detect_separator(
            zip_path,
            inner_file
        )
    )


    available_cols = (
        get_columns(
            zip_path,
            inner_file,
            separator
        )
    )


    missing_required = [

        col
        for col in REQUIRED_COLS

        if col
        not in available_cols
    ]


    if missing_required:

        raise ValueError(

            f"\n{year}: required columns missing:\n"
            f"{missing_required}\n"
        )


    usecols = [

        col
        for col in WANTED_COLS

        if col
        in available_cols
    ]


    print(
        f"ZIP: {zip_path.name}"
    )

    print(
        f"Internal file: {inner_file}"
    )

    print(
        f"Delimiter: {repr(separator)}"
    )

    print(
        f"Using {len(usecols)} columns."
    )


    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            inner_file
        ) as stream:


            reader = pd.read_csv(

                stream,

                sep=separator,

                usecols=usecols,

                dtype="string",

                chunksize=CHUNK_SIZE,

                low_memory=False,

                encoding="utf-8-sig",

                encoding_errors="replace"
            )


            for chunk_no, df in enumerate(
                reader,
                start=1
            ):


                # CORE NUMERIC CODES

                action = numeric(
                    df[
                        "action_taken"
                    ]
                )

                purpose = numeric(
                    df[
                        "loan_purpose"
                    ]
                )

                open_end = numeric(
                    df[
                        "open_end_line_of_credit"
                    ]
                )

                reverse = numeric(
                    df[
                        "reverse_mortgage"
                    ]
                )

                business = numeric(
                    df[
                        "business_or_commercial_purpose"
                    ]
                )

                occupancy = numeric(
                    df[
                        "occupancy_type"
                    ]
                )

                construction = numeric(
                    df[
                        "construction_method"
                    ]
                )

                lien = numeric(
                    df[
                        "lien_status"
                    ]
                )

                submission = numeric(
                    df[
                        "submission_of_application"
                    ]
                )


                # SAMPLE FLOW

                m01 = pd.Series(
                    True,
                    index=df.index
                )

                m02 = (
                    m01
                    &
                    purpose.eq(1)
                )

                m03 = (
                    m02
                    &
                    action.isin(
                        [1, 2, 3]
                    )
                )

                m04 = (
                    m03
                    &
                    open_end.eq(2)
                )

                m05 = (
                    m04
                    &
                    reverse.eq(2)
                )

                m06 = (
                    m05
                    &
                    business.eq(2)
                )

                m07 = (
                    m06
                    &
                    occupancy.eq(1)
                )

                m08 = (
                    m07
                    &
                    construction.eq(1)
                )


                units = (
                    clean_string(
                        df[
                            "total_units"
                        ]
                    )
                )


                m09 = (
                    m08
                    &
                    units.isin(
                        [
                            "1",
                            "2",
                            "3",
                            "4"
                        ]
                    )
                )


                m10 = (
                    m09
                    &
                    lien.eq(1)
                )


                m11 = (
                    m10
                    &
                    submission.isin(
                        [1, 2]
                    )
                )


                stage_masks = [

                    m01,
                    m02,
                    m03,
                    m04,
                    m05,
                    m06,
                    m07,
                    m08,
                    m09,
                    m10,
                    m11
                ]


                for stage, mask in zip(
                    FLOW_STAGES[:11],
                    stage_masks
                ):

                    sample_flow[
                        year
                    ][
                        stage
                    ] += int(
                        mask.sum()
                    )


                # NOTHING LEFT?

                if not m11.any():

                    continue


                # FILTER TO FINAL ELIGIBLE MORTGAGE SAMPLE
                # BEFORE RACE RESTRICTION

                d = (
                    df.loc[
                        m11
                    ]
                    .copy()
                )


                d_action = (
                    action.loc[
                        m11
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                d_submission = (
                    submission.loc[
                        m11
                    ]
                    .reset_index(
                        drop=True
                    )
                )


                d.reset_index(
                    drop=True,
                    inplace=True
                )


                # IDs

                d[
                    "year"
                ] = int(year)


                d[
                    "lei_clean"
                ] = clean_lei(
                    d["lei"]
                )


                d[
                    "lender_year_id"
                ] = (

                    d[
                        "lei_clean"
                    ]
                    .fillna(
                        "MISSING_LEI"
                    )

                    +

                    "_"

                    +

                    str(year)
                )


                # OUTCOME

                d[
                    "approval"
                ] = (
                    d_action
                    .isin(
                        [1, 2]
                    )
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "originated"
                ] = (
                    d_action
                    .eq(1)
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "action_code"
                ] = (
                    d_action
                    .astype(
                        "Int8"
                    )
                )


                # CHANNEL

                d[
                    "non_direct"
                ] = (
                    d_submission
                    .eq(2)
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "channel"
                ] = np.where(

                    d[
                        "non_direct"
                    ]
                    .eq(1),

                    "Non-direct",

                    "Direct"
                )


                if (
                    "initially_payable_to_institution"
                    in d.columns
                ):

                    d[
                        "payable_status"
                    ] = classify_payable(

                        d[
                            "initially_payable_to_institution"
                        ]
                    )

                else:

                    d[
                        "payable_status"
                    ] = "Unavailable"


                # RACE / ETHNICITY

                race_info = (
                    classify_race_ethnicity(
                        d
                    )
                )


                d[
                    "race_group"
                ] = (
                    race_info[
                        "race_group"
                    ]
                )


                d[
                    "race_family_count"
                ] = (
                    race_info[
                        "race_family_count"
                    ]
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "multiple_ethnicity_flag"
                ] = (
                    race_info[
                        "multiple_ethnicity_flag"
                    ]
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "primary_race_eligible"
                ] = (

                    d[
                        "race_group"
                    ]

                    .isin(
                        PRIMARY_RACE_GROUPS
                    )

                    .astype(
                        "Int8"
                    )
                )


                # Dummy variables
                d[
                    "black"
                ] = (
                    d[
                        "race_group"
                    ]
                    .eq(
                        "Non-Hispanic Black"
                    )
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "hispanic"
                ] = (
                    d[
                        "race_group"
                    ]
                    .eq(
                        "Hispanic / Latino"
                    )
                    .astype(
                        "Int8"
                    )
                )


                d[
                    "asian"
                ] = (
                    d[
                        "race_group"
                    ]
                    .eq(
                        "Non-Hispanic Asian"
                    )
                    .astype(
                        "Int8"
                    )
                )


                # Sample flow after race coding
                substantive_race = (
                    ~d[
                        "race_group"
                    ]
                    .eq(
                        "Unknown / Not reported"
                    )
                )


                sample_flow[
                    year
                ][
                    "12 Substantive race/ethnicity known"
                ] += int(
                    substantive_race.sum()
                )


                sample_flow[
                    year
                ][
                    "13 Primary four race groups"
                ] += int(
                    d[
                        "primary_race_eligible"
                    ]
                    .sum()
                )


                # BORROWER CONTROLS

                if (
                    "income"
                    in d.columns
                ):

                    d[
                        "income_thousands"
                    ] = numeric(
                        d[
                            "income"
                        ]
                    )

                    d[
                        "asinh_income"
                    ] = np.arcsinh(

                        d[
                            "income_thousands"
                        ]
                    )

                else:

                    d[
                        "income_thousands"
                    ] = np.nan

                    d[
                        "asinh_income"
                    ] = np.nan


                # DTI
                if (
                    "debt_to_income_ratio"
                    in d.columns
                ):

                    d[
                        "dti_raw"
                    ] = clean_string(

                        d[
                            "debt_to_income_ratio"
                        ]
                    )

                    d[
                        "dti_category"
                    ] = classify_dti(

                        d[
                            "debt_to_income_ratio"
                        ]
                    )

                else:

                    d[
                        "dti_raw"
                    ] = pd.NA

                    d[
                        "dti_category"
                    ] = (
                        "Missing / Not relied upon"
                    )


                # CLTV
                if (
                    "combined_loan_to_value_ratio"
                    in d.columns
                ):

                    d[
                        "cltv"
                    ] = numeric(

                        d[
                            "combined_loan_to_value_ratio"
                        ]
                    )

                else:

                    d[
                        "cltv"
                    ] = np.nan


                # Age
                if (
                    "applicant_age"
                    in d.columns
                ):

                    d[
                        "age_category"
                    ] = classify_age(

                        d[
                            "applicant_age"
                        ]
                    )

                else:

                    d[
                        "age_category"
                    ] = (
                        "Unknown / Not reported"
                    )


                # Sex
                if (
                    "applicant_sex"
                    in d.columns
                ):

                    d[
                        "sex_category"
                    ] = classify_sex(

                        d[
                            "applicant_sex"
                        ]
                    )

                else:

                    d[
                        "sex_category"
                    ] = (
                        "Unknown / Not reported"
                    )


                # Co-applicant
                d[
                    "coapplicant_present"
                ] = classify_coapplicant(
                    d
                )


                # LOAN CONTROLS

                if (
                    "loan_type"
                    in d.columns
                ):

                    d[
                        "loan_type_clean"
                    ] = classify_loan_type(

                        d[
                            "loan_type"
                        ]
                    )

                else:

                    d[
                        "loan_type_clean"
                    ] = "Unknown"


                if (
                    "loan_amount"
                    in d.columns
                ):

                    d[
                        "loan_amount_clean"
                    ] = numeric(

                        d[
                            "loan_amount"
                        ]
                    )

                    d[
                        "log_loan_amount"
                    ] = safe_log_positive(
                        d[
                            "loan_amount_clean"
                         ]
                    )

                else:

                    d[
                        "loan_amount_clean"
                    ] = np.nan

                    d[
                        "log_loan_amount"
                    ] = np.nan


                if (
                    "property_value"
                    in d.columns
                ):

                    d[
                        "property_value_clean"
                    ] = numeric(

                        d[
                            "property_value"
                        ]
                    )

                    d[
                        "log_property_value"
                    ] = safe_log_positive(
                        d[
                        "property_value_clean"
                        ]
                     )

                else:

                    d[
                        "property_value_clean"
                    ] = np.nan

                    d[
                        "log_property_value"
                    ] = np.nan


                # GEOGRAPHY

                if (
                    "state_code"
                    in d.columns
                ):

                    d[
                        "state_clean"
                    ] = clean_geo_code(

                        d[
                            "state_code"
                        ],

                        width=2
                    )

                else:

                    d[
                        "state_clean"
                    ] = pd.NA


                if (
                    "county_code"
                    in d.columns
                ):

                    d[
                        "county_clean"
                    ] = clean_geo_code(

                        d[
                            "county_code"
                        ],

                        width=5
                    )

                else:

                    d[
                        "county_clean"
                    ] = pd.NA


                if (
                    "census_tract"
                    in d.columns
                ):

                    d[
                        "tract_clean"
                    ] = clean_geo_code(

                        d[
                            "census_tract"
                        ],

                        width=11
                    )

                else:

                    d[
                        "tract_clean"
                    ] = pd.NA


                # RACE AUDIT

                race_vc = (
                    d[
                        "race_group"
                    ]
                    .value_counts(
                        dropna=False
                    )
                )


                for race_name, n in (
                    race_vc.items()
                ):

                    race_audit_counts[
                        (
                            year,
                            str(race_name)
                        )
                    ] += int(n)


                # YEAR / CHANNEL COUNTS

                year_final_counts[
                    year
                ] += len(d)


                for channel, n in (

                    d[
                        "channel"
                    ]
                    .value_counts()
                    .items()
                ):

                    channel_year_counts[
                        (
                            year,
                            channel
                        )
                    ] += int(n)


                # LENDER x YEAR CHANNEL SUPPORT COUNTS

                ly_temp = (

                    d
                    .groupby(
                        [
                            "lei_clean",
                            "non_direct"
                        ],
                        dropna=False
                    )
                    .size()
                )


                for (
                    lei,
                    nondirect
                ), n in (
                    ly_temp.items()
                ):

                    if pd.isna(lei):
                        continue

                    key = (
                        str(lei),
                        int(year)
                    )

                    if int(nondirect) == 0:

                        ly_channel_counts[
                            key
                        ][
                            "direct_n"
                        ] += int(n)

                    else:

                        ly_channel_counts[
                            key
                        ][
                            "non_direct_n"
                        ] += int(n)


                # RACE-SPECIFIC SUPPORT COUNTS

                support_races = (
                    d.loc[
                        d[
                            "race_group"
                        ].isin(
                            PRIMARY_RACE_GROUPS
                        ),
                        [
                            "lei_clean",
                            "race_group",
                            "non_direct"
                        ]
                    ]
                    .dropna(
                        subset=[
                            "lei_clean"
                        ]
                    )
                )


                race_cell_counts = (

                    support_races

                    .groupby(
                        [
                            "lei_clean",
                            "race_group",
                            "non_direct"
                        ]
                    )

                    .size()
                )


                for (
                    lei,
                    race_name,
                    nondirect
                ), n in (
                    race_cell_counts.items()
                ):

                    ly_race_channel_counts[
                        (
                            str(lei),
                            int(year),
                            str(race_name),
                            int(nondirect)
                        )
                    ] += int(n)


                # CONTROL COMPLETENESS AUDIT

                control_masks = {

                    "income":
                        d[
                            "income_thousands"
                        ].notna(),

                    "dti":
                        ~d[
                            "dti_category"
                        ].eq(
                            "Missing / Not relied upon"
                        ),

                    "cltv":
                        d[
                            "cltv"
                        ].notna(),

                    "loan_amount":
                        d[
                            "loan_amount_clean"
                        ].notna(),

                    "property_value":
                        d[
                            "property_value_clean"
                        ].notna(),

                    "age":
                        ~d[
                            "age_category"
                        ].eq(
                            "Unknown / Not reported"
                        ),

                    "sex":
                        ~d[
                            "sex_category"
                        ].eq(
                            "Unknown / Not reported"
                        ),

                    "coapplicant":
                        d[
                            "coapplicant_present"
                        ].notna(),

                    "county":
                        d[
                            "county_clean"
                        ].notna()
                }


                for variable, valid in (
                    control_masks.items()
                ):

                    key = (
                        year,
                        variable
                    )

                    control_counts[
                        key
                    ][
                        "total"
                    ] += len(d)

                    control_counts[
                        key
                    ][
                        "usable"
                    ] += int(
                        valid.sum()
                    )


                # KEEPING RAW DEMOGRAPHIC CODES FOR REPRODUCIBILITY AAND DROPPING OTHER RAW VARIABLES

                final_columns = [

                    "year",
                    "lei_clean",
                    "lender_year_id",

                    "action_code",
                    "approval",
                    "originated",

                    "non_direct",
                    "channel",
                    "payable_status",

                    "race_group",
                    "primary_race_eligible",

                    "black",
                    "hispanic",
                    "asian",

                    "race_family_count",
                    "multiple_ethnicity_flag",

                    "income_thousands",
                    "asinh_income",

                    "dti_raw",
                    "dti_category",

                    "cltv",

                    "age_category",
                    "sex_category",
                    "coapplicant_present",

                    "loan_type_clean",

                    "loan_amount_clean",
                    "log_loan_amount",

                    "property_value_clean",
                    "log_property_value",

                    "state_clean",
                    "county_clean",
                    "tract_clean"

                ]


                # Preserve all 5 raw race/ethnicity codes
                for col in (
                    RACE_COLS
                    +
                    ETHNICITY_COLS
                ):

                    if col in d.columns:

                        final_columns.append(
                            col
                        )


                out = (
                    d[
                        final_columns
                    ]
                    .copy()
                )


                global_part_number += 1


                part_path = (

                    BASE_PARTS_DIR
                    /
                    (
                        f"year={year}_"
                        f"part={global_part_number:05d}.parquet"
                    )
                )


                out.to_parquet(

                    part_path,

                    index=False,

                    engine="pyarrow",

                    compression="snappy"
                )


                # PROGRESS

                if (
                    chunk_no % 10 == 0
                ):

                    print(

                        f"Chunk {chunk_no:>4} | "

                        f"Raw rows processed: "
                        f"{sample_flow[year]['01 National HMDA rows']:,} | "

                        f"Eligible rows written: "
                        f"{year_final_counts[year]:,}"
                    )


    print(
        f"\n{year} eligible channel sample: "
        f"{year_final_counts[year]:,}"
    )


# 19. BUILD LENDER x YEAR SUPPORT TABLE

support_rows = []


for (
    lei,
    year
), values in (
    ly_channel_counts.items()
):

    direct_n = (
        values[
            "direct_n"
        ]
    )

    non_direct_n = (
        values[
            "non_direct_n"
        ]
    )


    total_n = (
        direct_n
        +
        non_direct_n
    )


    row = {

        "lei_clean":
            lei,

        "year":
            year,

        "lender_year_id":
            f"{lei}_{year}",

        "direct_n":
            direct_n,

        "non_direct_n":
            non_direct_n,

        "total_n":
            total_n,

        "non_direct_share":
            (
                non_direct_n
                /
                total_n

                if total_n > 0

                else np.nan
            ),

        "dual_channel_ly":
            int(
                direct_n > 0
                and
                non_direct_n > 0
            ),

        "dual10":
            int(
                direct_n >= 10
                and
                non_direct_n >= 10
            ),

        "dual25":
            int(
                direct_n >= 25
                and
                non_direct_n >= 25
            ),

        "dual50":
            int(
                direct_n >= 50
                and
                non_direct_n >= 50
            ),

        "dual100":
            int(
                direct_n >= 100
                and
                non_direct_n >= 100
            )
    }


    support_rows.append(
        row
    )


support_df = pd.DataFrame(
    support_rows
)


# 20. ADD RACE-SPECIFIC CELL COUNTS

RACE_SHORT = {

    "Non-Hispanic White":
        "white",

    "Non-Hispanic Black":
        "black",

    "Hispanic / Latino":
        "hispanic",

    "Non-Hispanic Asian":
        "asian"
}


# Initialise cell columns
for short_name in (
    RACE_SHORT.values()
):

    support_df[
        f"{short_name}_direct"
    ] = 0

    support_df[
        f"{short_name}_non_direct"
    ] = 0


# Convert support table to index for fast updates
support_df.set_index(
    [
        "lei_clean",
        "year"
    ],
    inplace=True
)


for (
    lei,
    year,
    race_name,
    nondirect
), n in (
    ly_race_channel_counts.items()
):

    key = (
        lei,
        year
    )

    if (
        key
        not in support_df.index
    ):

        continue


    short_name = (
        RACE_SHORT[
            race_name
        ]
    )


    channel_name = (

        "non_direct"
        if nondirect == 1

        else "direct"
    )


    col = (
        f"{short_name}_{channel_name}"
    )


    support_df.loc[
        key,
        col
    ] += int(n)


support_df.reset_index(
    inplace=True
)


# 21. RACE-SPECIFIC COMMON SUPPORT FLAGS

def add_common_support_flags(
    df,
    minority_short,
    prefix
):

    relevant = [

        "white_direct",
        "white_non_direct",

        f"{minority_short}_direct",
        f"{minority_short}_non_direct"
    ]


    minimum_cell = (
        df[
            relevant
        ]
        .min(axis=1)
    )


    df[
        f"{prefix}_common_any"
    ] = (
        minimum_cell
        .gt(0)
        .astype("Int8")
    )


    df[
        f"{prefix}_common5"
    ] = (
        minimum_cell
        .ge(5)
        .astype("Int8")
    )


    df[
        f"{prefix}_common10"
    ] = (
        minimum_cell
        .ge(10)
        .astype("Int8")
    )


    df[
        f"{prefix}_common25"
    ] = (
        minimum_cell
        .ge(25)
        .astype("Int8")
    )


    return df


support_df = add_common_support_flags(

    support_df,

    minority_short="black",

    prefix="bw"
)


support_df = add_common_support_flags(

    support_df,

    minority_short="hispanic",

    prefix="hw"
)


support_df = add_common_support_flags(

    support_df,

    minority_short="asian",

    prefix="aw"
)


# 22. SAVE SUPPORT TABLE

support_path = (

    OUTPUT_DIR
    /
    "lender_year_channel_support.csv"
)


support_df.to_csv(

    support_path,

    index=False
)


# 23. SUPPORT SUMMARY

support_summary_rows = []


general_flags = [

    "dual_channel_ly",
    "dual10",
    "dual25",
    "dual50",
    "dual100",

    "bw_common_any",
    "bw_common5",
    "bw_common10",
    "bw_common25",

    "hw_common_any",
    "hw_common5",
    "hw_common10",
    "hw_common25",

    "aw_common_any",
    "aw_common5",
    "aw_common10",
    "aw_common25"
]


total_eligible_apps = (
    support_df[
        "total_n"
    ].sum()
)


for flag in general_flags:

    selected = (

        support_df[
            flag
        ].eq(1)
    )


    apps = (

        support_df.loc[
            selected,
            "total_n"
        ]
        .sum()
    )


    support_summary_rows.append({

        "support_definition":
            flag,

        "lender_years":
            int(
                selected.sum()
            ),

        "applications_in_those_lender_years":
            int(apps),

        "share_of_eligible_applications":
            (
                apps
                /
                total_eligible_apps

                if total_eligible_apps > 0

                else np.nan
            )
    })


support_summary_df = pd.DataFrame(
    support_summary_rows
)


support_summary_df.to_csv(

    OUTPUT_DIR
    /
    "dual_channel_support_summary.csv",

    index=False
)


# 24. YEAR-SPECIFIC SUPPORT

year_support_rows = []


for year in STUDY_YEARS:

    y = support_df.loc[
        support_df[
            "year"
        ].eq(year)
    ]


    total_apps = (
        y[
            "total_n"
        ].sum()
    )


    for flag in [

        "dual_channel_ly",
        "dual10",
        "dual25",

        "bw_common_any",
        "hw_common_any",
        "aw_common_any"
    ]:

        selected = (
            y[
                flag
            ].eq(1)
        )


        apps = (

            y.loc[
                selected,
                "total_n"
            ]
            .sum()
        )


        year_support_rows.append({

            "year":
                year,

            "support_definition":
                flag,

            "lender_years":
                int(
                    selected.sum()
                ),

            "applications":
                int(apps),

            "application_share":
                (
                    apps
                    /
                    total_apps

                    if total_apps > 0

                    else np.nan
                )
        })


year_support_df = pd.DataFrame(
    year_support_rows
)


year_support_df.to_csv(

    OUTPUT_DIR
    /
    "lender_year_support_by_year.csv",

    index=False
)


# 25. SECOND PASS:

print("\n")
print("=" * 90)
print("SECOND PASS — ATTACHING LENDER-YEAR SUPPORT FLAGS")
print("=" * 90)


merge_columns = [

    "lender_year_id",

    "direct_n",
    "non_direct_n",
    "total_n",
    "non_direct_share",

    "dual_channel_ly",
    "dual10",
    "dual25",
    "dual50",
    "dual100",

    "bw_common_any",
    "bw_common5",
    "bw_common10",
    "bw_common25",

    "hw_common_any",
    "hw_common5",
    "hw_common10",
    "hw_common25",

    "aw_common_any",
    "aw_common5",
    "aw_common10",
    "aw_common25"
]


support_merge = (

    support_df[
        merge_columns
    ]
    .copy()
)


base_files = sorted(

    BASE_PARTS_DIR.glob(
        "*.parquet"
    )
)


main_analysis_n = 0
full_analysis_n = 0


for i, part_file in enumerate(
    base_files,
    start=1
):

    d = pd.read_parquet(
        part_file
    )


    d = d.merge(

        support_merge,

        on="lender_year_id",

        how="left",

        validate="many_to_one"
    )


    # Headline dissertation sample indicator
    # Main:
    # - Primary four race groups
    # - lender-year uses both channels

    d[
        "main_analysis_sample"
    ] = (

        d[
            "primary_race_eligible"
        ].eq(1)

        &

        d[
            "dual_channel_ly"
        ].eq(1)
    ).astype(
        "Int8"
    )


    full_analysis_n += len(d)


    main_analysis_n += int(

        d[
            "main_analysis_sample"
        ].sum()
    )


    final_path = (

        FINAL_PARTS_DIR
        /
        part_file.name
    )


    d.to_parquet(

        final_path,

        index=False,

        engine="pyarrow",

        compression="snappy"
    )


    if i % 20 == 0:

        print(

            f"Enriched parts: "
            f"{i:,}/{len(base_files):,}"
        )


# 26. SAMPLE FLOW DATAFRAME

flow_rows = []


for year in STUDY_YEARS:

    previous = None


    for stage in FLOW_STAGES:

        n = (
            sample_flow[
                year
            ][
                stage
            ]
        )


        if previous is None:

            retained_from_previous = 1.0

        else:

            retained_from_previous = (

                n / previous

                if previous > 0

                else np.nan
            )


        flow_rows.append({

            "year":
                year,

            "stage":
                stage,

            "n":
                n,

            "retained_from_previous":
                retained_from_previous
        })


        previous = n


sample_flow_df = pd.DataFrame(
    flow_rows
)


sample_flow_df.to_csv(

    OUTPUT_DIR
    /
    "sample_flow.csv",

    index=False
)


# 27. RACE CODING AUDIT

race_rows = []


for (
    year,
    race_name
), n in (
    race_audit_counts.items()
):

    race_rows.append({

        "year":
            year,

        "race_group":
            race_name,

        "n":
            n
    })


race_audit_df = pd.DataFrame(
    race_rows
)


if not race_audit_df.empty:

    race_audit_df[
        "year_total"
    ] = (

        race_audit_df

        .groupby(
            "year"
        )[
            "n"
        ]

        .transform(
            "sum"
        )
    )


    race_audit_df[
        "share"
    ] = (

        race_audit_df[
            "n"
        ]

        /

        race_audit_df[
            "year_total"
        ]
    )


race_audit_df.to_csv(

    OUTPUT_DIR
    /
    "race_coding_audit.csv",

    index=False
)


unknown_race_df = (

    race_audit_df.loc[

        race_audit_df[
            "race_group"
        ].eq(
            "Unknown / Not reported"
        )

    ]
    .copy()
)


unknown_race_df.to_csv(

    OUTPUT_DIR
    /
    "unknown_race_audit.csv",

    index=False
)


# 28. CONTROL MISSINGNESS AUDIT

control_rows = []


for (
    year,
    variable
), values in (
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

        "year":
            year,

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


control_df.to_csv(

    OUTPUT_DIR
    /
    "control_missingness.csv",

    index=False
)


# 29. YEAR SAMPLE COUNTS

year_rows = []


for year in STUDY_YEARS:

    direct = (
        channel_year_counts[
            (
                year,
                "Direct"
            )
        ]
    )

    non_direct = (
        channel_year_counts[
            (
                year,
                "Non-direct"
            )
        ]
    )

    total = (
        direct
        +
        non_direct
    )


    year_rows.append({

        "year":
            year,

        "eligible_channel_sample":
            total,

        "direct_n":
            direct,

        "non_direct_n":
            non_direct,

        "non_direct_share":
            (
                non_direct
                /
                total

                if total > 0

                else np.nan
            )
    })


year_counts_df = pd.DataFrame(
    year_rows
)


year_counts_df.to_csv(

    OUTPUT_DIR
    /
    "year_sample_counts.csv",

    index=False
)


# 30. OVERALL CONSTRUCTION SUMMARY

overall_summary_df = pd.DataFrame({

    "metric": [

        "Study years",

        "Eligible direct/non-direct mortgage observations",

        "Primary headline analysis observations",

        "Unique lender-years",

        "Dual-channel lender-years",

        "Dual10 lender-years",

        "Dual25 lender-years",

        "Unique lenders in eligible sample"
    ],

    "value": [

        len(STUDY_YEARS),

        int(
            full_analysis_n
        ),

        int(
            main_analysis_n
        ),

        int(
            len(
                support_df
            )
        ),

        int(
            support_df[
                "dual_channel_ly"
            ].sum()
        ),

        int(
            support_df[
                "dual10"
            ].sum()
        ),

        int(
            support_df[
                "dual25"
            ].sum()
        ),

        int(
            support_df[
                "lei_clean"
            ].nunique()
        )
    ]
})


overall_summary_df.to_csv(

    OUTPUT_DIR
    /
    "construction_summary.csv",

    index=False
)


# 31. EXCEL AUDIT WORKBOOK

excel_path = (

    OUTPUT_DIR
    /
    "FINAL_DATA_CONSTRUCTION_AUDIT.xlsx"
)


with pd.ExcelWriter(

    excel_path,

    engine="openpyxl"

) as writer:


    overall_summary_df.to_excel(

        writer,

        sheet_name="Overall",

        index=False
    )


    sample_flow_df.to_excel(

        writer,

        sheet_name="Sample Flow",

        index=False
    )


    year_counts_df.to_excel(

        writer,

        sheet_name="Year Counts",

        index=False
    )


    race_audit_df.to_excel(

        writer,

        sheet_name="Race Coding",

        index=False
    )


    unknown_race_df.to_excel(

        writer,

        sheet_name="Unknown Race",

        index=False
    )


    support_summary_df.to_excel(

        writer,

        sheet_name="Support Summary",

        index=False
    )


    year_support_df.to_excel(

        writer,

        sheet_name="Support by Year",

        index=False
    )


    support_df.to_excel(

        writer,

        sheet_name="Lender Year Support",

        index=False
    )


    control_df.to_excel(

        writer,

        sheet_name="Control Coverage",

        index=False
    )


# 32. ENDING

print("\n\n")
print("=" * 90)
print("FINAL CHANNEL ANALYSIS DATASET COMPLETE")
print("=" * 90)


print("\nConstruction summary:\n")

print(
    overall_summary_df.to_string(
        index=False
    )
)


print("\nFinal analysis dataset directory:")

print(
    FINAL_PARTS_DIR
)


print("\nAudit workbook:")

print(
    excel_path
)


print("\nSupport table:")

print(
    support_path
)


print("\nIMPORTANT:")
print(
    "Do NOT alter the raw HMDA ZIP files."
)

print(
    "All regressions will now use the final Parquet dataset."
)

print("\nDONE.")