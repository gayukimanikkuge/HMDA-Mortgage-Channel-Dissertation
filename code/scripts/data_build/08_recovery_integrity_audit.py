"""
Created on Sun Aug 16 19:03:47 2026

@author: gayukimanikkuge
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# PATHS

PROJECT_DIR = Path(
    "/Users/gayukimanikkuge/Desktop/HMDA_Dissertation"
)

BASE_DIR = (
    PROJECT_DIR
    / "final_data"
    / "channel_analysis_base_parts"
)

FINAL_DIR = (
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

OUT_CSV = (
    OUTPUT_DIR
    / "RECOVERY_INTEGRITY_AUDIT.csv"
)


# EXPECTED VALUES FROM PREVIOUS AUDITS

EXPECTED_PARTS = 500

EXPECTED_TOTAL_ROWS = 27_692_586

EXPECTED_MAIN_SAMPLE = 11_071_156


# HELPERS

def parquet_rows(path):
    """
    Read row count from Parquet metadata without loading the full file.
    """

    pf = pq.ParquetFile(
        path
    )

    return pf.metadata.num_rows


def schema_dict(path):
    """
    Return {column_name: pyarrow_type_string}.
    """

    schema = pq.read_schema(
        path
    )

    return {
        field.name: str(field.type)
        for field in schema
    }


def dataframe_fingerprints(df):
    row_hashes = (
        pd.util.hash_pandas_object(
            df,
            index=False
        )
        .to_numpy(
            dtype=np.uint64,
            copy=False
        )
    )

    ordered_sha = hashlib.sha256(
        row_hashes.tobytes()
    ).hexdigest()

    sorted_hashes = np.sort(
        row_hashes
    )

    unordered_sha = hashlib.sha256(
        sorted_hashes.tobytes()
    ).hexdigest()

    return (
        ordered_sha,
        unordered_sha
    )


# LOCATE FILES

base_parts = sorted(
    BASE_DIR.glob("*.parquet")
)

final_parts = sorted(
    FINAL_DIR.glob("*.parquet")
)


print("\n")
print("=" * 80)
print("RECOVERY INTEGRITY AUDIT")
print("=" * 80)


print(
    "\nBase directory:"
)

print(
    BASE_DIR
)

print(
    "\nFinal V1 directory:"
)

print(
    FINAL_DIR
)


print(
    "\nBase Parquet parts:",
    len(base_parts)
)

print(
    "Final Parquet parts:",
    len(final_parts)
)


if not base_parts:

    raise FileNotFoundError(
        f"No base Parquet files found in:\n{BASE_DIR}"
    )

if not final_parts:

    raise FileNotFoundError(
        f"No final Parquet files found in:\n{FINAL_DIR}"
    )


# FILE-NAME ALIGNMENT

base_names = {
    p.name
    for p in base_parts
}

final_names = {
    p.name
    for p in final_parts
}

same_names = (
    base_names
    ==
    final_names
)


print(
    "\nExact same Parquet filenames:",
    same_names
)


if same_names:

    # Best case:
    # Pair exact same filenames.

    pairs = [
        (
            BASE_DIR / name,
            FINAL_DIR / name
        )
        for name in sorted(
            base_names
        )
    ]

    pairing_method = (
        "exact filename"
    )

else:

    missing_from_final = sorted(
        base_names
        -
        final_names
    )

    extra_in_final = sorted(
        final_names
        -
        base_names
    )

    print(
        "\nFiles present in BASE but not FINAL:",
        len(missing_from_final)
    )

    for x in missing_from_final[:20]:

        print(
            "  ",
            x
        )

    print(
        "\nFiles present in FINAL but not BASE:",
        len(extra_in_final)
    )

    for x in extra_in_final[:20]:

        print(
            "  ",
            x
        )

    if (
        len(base_parts)
        !=
        len(final_parts)
    ):

        raise RuntimeError(
            "Cannot safely pair parts because file counts differ."
        )


    pairs = list(
        zip(
            base_parts,
            final_parts
        )
    )

    pairing_method = (
        "sorted position"
    )


print(
    "Pairing method:",
    pairing_method
)


# SCHEMA CHECK

base_schema = schema_dict(
    base_parts[0]
)

final_schema = schema_dict(
    final_parts[0]
)


base_columns = list(
    base_schema.keys()
)

final_columns = list(
    final_schema.keys()
)


print("\n")
print("=" * 80)
print("SCHEMA CHECK")
print("=" * 80)


print(
    "\nBase columns:",
    len(base_columns)
)

print(
    "Final columns:",
    len(final_columns)
)


missing_base_columns = [
    col
    for col in base_columns
    if col not in final_schema
]


added_final_columns = [
    col
    for col in final_columns
    if col not in base_schema
]


print(
    "\nBase columns missing from Final V1:",
    len(missing_base_columns)
)


if missing_base_columns:

    for col in missing_base_columns:

        print(
            "  MISSING:",
            col
        )


print(
    "\nColumns intentionally/newly present in Final V1:",
    len(added_final_columns)
)


for col in added_final_columns:

    print(
        "  ADDED:",
        col
    )


# SHARED COLUMN TYPE CHECK

schema_type_mismatches = []

for col in base_columns:

    if col not in final_schema:
        continue

    if (
        base_schema[col]
        !=
        final_schema[col]
    ):

        schema_type_mismatches.append(
            (
                col,
                base_schema[col],
                final_schema[col]
            )
        )


print(
    "\nShared-column schema/type mismatches:",
    len(schema_type_mismatches)
)


for (
    col,
    base_type,
    final_type
) in schema_type_mismatches:

    print(
        f"  {col}: "
        f"BASE={base_type} | FINAL={final_type}"
    )


# STOP IF BASE COLUMNS DISAPPEARED

if missing_base_columns:

    raise RuntimeError(
        "Final V1 is missing one or more base columns. "
        "Integrity audit cannot pass."
    )


# PART-BY-PART FULL INTEGRITY CHECK

print("\n")
print("=" * 80)
print("PART-BY-PART ROW + CONTENT CHECK")
print("=" * 80)


audit_rows = []

base_total_rows = 0
final_total_rows = 0

row_count_mismatch_parts = 0

ordered_hash_mismatch_parts = 0
unordered_hash_mismatch_parts = 0

read_errors = 0

main_sample_total = 0


for i, (
    base_path,
    final_path
) in enumerate(
    pairs,
    start=1
):

    result = {
        "part_number": i,
        "base_file": base_path.name,
        "final_file": final_path.name,
        "base_rows": pd.NA,
        "final_rows": pd.NA,
        "row_count_match": False,
        "ordered_content_match": False,
        "unordered_content_match": False,
        "status": "NOT CHECKED",
    }

    try:

        # METADATA ROW COUNTS

        b_rows = parquet_rows(
            base_path
        )

        f_rows = parquet_rows(
            final_path
        )

        result[
            "base_rows"
        ] = b_rows

        result[
            "final_rows"
        ] = f_rows


        base_total_rows += b_rows
        final_total_rows += f_rows


        row_match = (
            b_rows
            ==
            f_rows
        )

        result[
            "row_count_match"
        ] = row_match


        if not row_match:

            row_count_mismatch_parts += 1

        base_df = pd.read_parquet(
            base_path,
            columns=base_columns
        )

        final_df = pd.read_parquet(
            final_path,
            columns=base_columns
        )


        # Explicitly align column order.
        base_df = base_df[
            base_columns
        ]

        final_df = final_df[
            base_columns
        ]

        # FINGERPRINT COMPLETE SHARED CONTENT

        (
            base_ordered,
            base_unordered
        ) = dataframe_fingerprints(
            base_df
        )

        (
            final_ordered,
            final_unordered
        ) = dataframe_fingerprints(
            final_df
        )


        ordered_match = (
            base_ordered
            ==
            final_ordered
        )

        unordered_match = (
            base_unordered
            ==
            final_unordered
        )


        result[
            "ordered_content_match"
        ] = ordered_match

        result[
            "unordered_content_match"
        ] = unordered_match


        if not ordered_match:

            ordered_hash_mismatch_parts += 1


        if not unordered_match:

            unordered_hash_mismatch_parts += 1


        # MAIN SAMPLE COUNT FROM FINAL

        if (
            "main_analysis_sample"
            in final_columns
        ):

            main_flag = pd.read_parquet(
                final_path,
                columns=[
                    "main_analysis_sample"
                ]
            )[
                "main_analysis_sample"
            ]

            main_sample_total += int(
                pd.to_numeric(
                    main_flag,
                    errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .sum()
            )


        # PART STATUS

        if (
            row_match
            and
            ordered_match
            and
            unordered_match
        ):

            result[
                "status"
            ] = "EXACT MATCH"


        elif (
            row_match
            and
            unordered_match
        ):

            result[
                "status"
            ] = (
                "SAME CONTENT - ROW ORDER DIFFERENT"
            )


        else:

            result[
                "status"
            ] = "CONTENT MISMATCH"


        # Release memory explicitly.
        del base_df
        del final_df


    except Exception as exc:

        read_errors += 1

        result[
            "status"
        ] = (
            f"READ/CHECK ERROR: {repr(exc)}"
        )


    audit_rows.append(
        result
    )


    if (
        i % 25 == 0
        or
        i == len(pairs)
    ):

        print(
            f"Processed {i}/{len(pairs)} parts"
            f" | BASE rows: {base_total_rows:,}"
            f" | FINAL rows: {final_total_rows:,}"
        )


# SAVE PART-BY-PART AUDIT

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUT_CSV,
    index=False
)


# RESULT SUMMARY

print("\n")
print("=" * 80)
print("INTEGRITY RESULTS")
print("=" * 80)


print(
    "\nBase parts:",
    len(base_parts)
)

print(
    "Final parts:",
    len(final_parts)
)


print(
    "\nBase total rows:",
    f"{base_total_rows:,}"
)

print(
    "Final total rows:",
    f"{final_total_rows:,}"
)


print(
    "\nExpected total rows:",
    f"{EXPECTED_TOTAL_ROWS:,}"
)


print(
    "\nParts with row-count mismatch:",
    f"{row_count_mismatch_parts:,}"
)

print(
    "Parts with ordered-content mismatch:",
    f"{ordered_hash_mismatch_parts:,}"
)

print(
    "Parts with actual content/multiset mismatch:",
    f"{unordered_hash_mismatch_parts:,}"
)

print(
    "Parts with read/check errors:",
    f"{read_errors:,}"
)


# STATUS BREAKDOWN

print("\n")
print("=" * 80)
print("PART STATUS BREAKDOWN")
print("=" * 80)


status_counts = (
    audit_df[
        "status"
    ]
    .value_counts(
        dropna=False
    )
)


for status, count in status_counts.items():

    print(
        status,
        ":",
        f"{count:,}"
    )


# MAIN SAMPLE CHECK

print("\n")
print("=" * 80)
print("MAIN SAMPLE CHECK")
print("=" * 80)


if (
    "main_analysis_sample"
    in final_columns
):

    print(
        "Final main_analysis_sample:",
        f"{main_sample_total:,}"
    )

    print(
        "Expected main_analysis_sample:",
        f"{EXPECTED_MAIN_SAMPLE:,}"
    )

    print(
        "Main sample count matches:",
        (
            main_sample_total
            ==
            EXPECTED_MAIN_SAMPLE
        )
    )

else:

    print(
        "main_analysis_sample not present in Final V1."
    )


# OVERALL PASS CONDITIONS

file_count_pass = (
    len(base_parts)
    ==
    EXPECTED_PARTS
    and
    len(final_parts)
    ==
    EXPECTED_PARTS
)


total_rows_pass = (
    base_total_rows
    ==
    final_total_rows
    ==
    EXPECTED_TOTAL_ROWS
)


part_rows_pass = (
    row_count_mismatch_parts
    ==
    0
)


content_pass = (
    unordered_hash_mismatch_parts
    ==
    0
)


read_pass = (
    read_errors
    ==
    0
)


columns_pass = (
    len(missing_base_columns)
    ==
    0
)


schema_pass = (
    len(schema_type_mismatches)
    ==
    0
)


if (
    "main_analysis_sample"
    in final_columns
):

    main_sample_pass = (
        main_sample_total
        ==
        EXPECTED_MAIN_SAMPLE
    )

else:

    main_sample_pass = False


overall_pass = all(
    [
        file_count_pass,
        total_rows_pass,
        part_rows_pass,
        content_pass,
        read_pass,
        columns_pass,
        schema_pass,
        main_sample_pass,
    ]
)


# FINAL DECISION

print("\n")
print("=" * 80)
print("FINAL RECOVERY DECISION")
print("=" * 80)


print(
    "\n500 + 500 files:",
    file_count_pass
)

print(
    "Base/final total rows identical:",
    total_rows_pass
)

print(
    "Every corresponding part has same row count:",
    part_rows_pass
)

print(
    "Every BASE column retained:",
    columns_pass
)

print(
    "Shared schemas preserved:",
    schema_pass
)

print(
    "Every part has identical row CONTENT:",
    content_pass
)

print(
    "All Parquet files readable:",
    read_pass
)

print(
    "Main sample count reproduced:",
    main_sample_pass
)


if overall_pass:

    print("\n")
    print(
        "PASS: RECOVERY INTEGRITY VERIFIED."
    )

    print(
        "The recovered Final Dataset V1 contains the same "
        "base observations and base-variable content as the "
        "completed first-pass dataset."
    )

    if ordered_hash_mismatch_parts == 0:

        print(
            "Row order is also identical in every paired part."
        )

    else:

        print(
            "Some row ordering changed, but row content is identical."
        )

    print(
        "There is no evidence that the disk-space failure caused "
        "data loss, duplication, truncation, or alteration."
    )


else:

    print("\n")
    print(
        "REVIEW REQUIRED: one or more recovery-integrity "
        "conditions did not pass."
    )

    print(
        "DO NOT rebuild anything yet."
    )

    print(
        "Inspect the detailed results above and the audit CSV first."
    )


print(
    "\nDetailed part audit written to:"
)

print(
    OUT_CSV
)

print("\nDONE.")