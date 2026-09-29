from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


YEARS = list(range(2018, 2025))
CHUNK_SIZE = 175_000
RAW_DIR: Optional[Path] = None

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "coapplicant_8888_audit"
OUTPUT_XLSX = OUTPUT_DIR / "COAPPLICANT_8888_AUDIT.xlsx"

REQUIRED_COLUMNS = [
    "loan_purpose",
    "occupancy_type",
    "construction_method",
    "open_end_line_of_credit",
    "reverse_mortgage",
    "business_or_commercial_purpose",
    "lien_status",
    "total_units",
    "action_taken",
    "submission_of_application",
    "co_applicant_sex",
    "co_applicant_age",
    "co_applicant_ethnicity_1",
    "co_applicant_race_1",
    "lei",
]

CORE_FILTERS = {
    "loan_purpose": {1},
    "occupancy_type": {1},
    "construction_method": {1},
    "open_end_line_of_credit": {2},
    "reverse_mortgage": {2},
    "business_or_commercial_purpose": {2},
    "lien_status": {1},
}

ALLOWED_UNITS = {"1", "2", "3", "4"}


def norm_col(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x).strip("_")
    return x


def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        })
    )


def int_codes(s: pd.Series) -> pd.Series:
    x = clean_text(s).str.replace(r"\.0$", "", regex=True)
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=df.index, dtype="string")


def candidate_raw_dirs() -> list[Path]:
    if RAW_DIR is not None:
        return [Path(RAW_DIR).expanduser()]

    candidates = [
        PROJECT_DIR / "raw_data",
        PROJECT_DIR / "raw",
        PROJECT_DIR / "data" / "raw",
        PROJECT_DIR / "hmda_raw",
        PROJECT_DIR,
        Path.home() / "Downloads",
    ]
    return [p for p in candidates if p.exists()]


def source_score(path: Path, year: int) -> tuple[int, int]:
    name = path.name.lower()
    score = 0
    if str(year) in name:
        score += 10
    if "lar" in name:
        score += 8
    if "public" in name:
        score += 5
    if "modified" in name:
        score += 3
    if "nation" in name:
        score += 2
    if path.suffix.lower() == ".zip":
        score += 2

    bad_terms = [
        "audit", "output", "summary", "feasibility",
        "candidate", "final_data", "parts",
    ]
    if any(term in str(path).lower() for term in bad_terms):
        score -= 20

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return score, size


def find_year_sources() -> dict[int, Path]:
    all_files: list[Path] = []

    for root in candidate_raw_dirs():
        try:
            all_files.extend(
                p for p in root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".zip", ".csv", ".txt", ".tsv"}
            )
        except Exception:
            continue

    sources = {}

    for year in YEARS:
        candidates = [
            p for p in all_files
            if str(year) in p.name
        ]

        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda p: source_score(p, year),
            reverse=True,
        )

        best = ranked[0]

        if source_score(best, year)[0] > 0:
            sources[year] = best

    return sources


def detect_sep(first_line: str) -> str:
    options = [",", "|", "\t"]
    return max(options, key=lambda x: first_line.count(x))


def choose_zip_member(zf: zipfile.ZipFile) -> str:
    members = [
        n for n in zf.namelist()
        if n.lower().endswith((".csv", ".txt", ".tsv"))
    ]

    if not members:
        raise ValueError("ZIP contains no CSV/TXT/TSV member.")

    def score(name: str):
        low = name.lower()
        s = (5 if "lar" in low else 0) + (3 if "public" in low else 0)
        try:
            size = zf.getinfo(name).file_size
        except KeyError:
            size = 0
        return s, size

    return sorted(members, key=score, reverse=True)[0]


def columns_for_header(columns) -> tuple[list[str], dict[str, str]]:
    mapping = {norm_col(c): c for c in columns}

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in mapping
    ]

    if missing:
        raise KeyError(
            "Raw HMDA source is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    originals = [mapping[c] for c in REQUIRED_COLUMNS]
    return originals, mapping


def read_chunks(path: Path) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = choose_zip_member(zf)

            with zf.open(member) as fh:
                wrapper = io.TextIOWrapper(
                    fh,
                    encoding="utf-8-sig",
                    errors="replace",
                )
                first_line = wrapper.readline()

            sep = detect_sep(first_line)

            with zf.open(member) as fh:
                header = pd.read_csv(
                    fh,
                    sep=sep,
                    nrows=0,
                    encoding="utf-8-sig",
                )

            usecols, _ = columns_for_header(header.columns)

            with zf.open(member) as fh:
                for chunk in pd.read_csv(
                    fh,
                    sep=sep,
                    usecols=usecols,
                    dtype="string",
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                    encoding="utf-8-sig",
                    on_bad_lines="warn",
                ):
                    chunk.columns = [norm_col(c) for c in chunk.columns]
                    yield chunk

        return

    if path.suffix.lower() in {".txt", ".tsv"}:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
        ) as fh:
            first_line = fh.readline()
        sep = detect_sep(first_line)
    else:
        sep = ","

    header = pd.read_csv(
        path,
        sep=sep,
        nrows=0,
        encoding="utf-8-sig",
    )

    usecols, _ = columns_for_header(header.columns)

    for chunk in pd.read_csv(
        path,
        sep=sep,
        usecols=usecols,
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8-sig",
        on_bad_lines="warn",
    ):
        chunk.columns = [norm_col(c) for c in chunk.columns]
        yield chunk


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    for col, allowed in CORE_FILTERS.items():
        mask &= int_codes(
            get_series(df, col)
        ).isin(allowed)

    units = (
        clean_text(get_series(df, "total_units"))
        .str.replace(r"\.0$", "", regex=True)
    )
    mask &= units.isin(ALLOWED_UNITS)

    action = int_codes(get_series(df, "action_taken"))
    mask &= action.isin([1, 2, 3])

    channel = int_codes(
        get_series(df, "submission_of_application")
    )
    mask &= channel.isin([1, 2])

    return mask


def audit_chunk(df: pd.DataFrame, year: int):
    sex = int_codes(get_series(df, "co_applicant_sex"))
    age = int_codes(get_series(df, "co_applicant_age"))
    eth = int_codes(get_series(df, "co_applicant_ethnicity_1"))
    race = int_codes(get_series(df, "co_applicant_race_1"))

    # Existing classifier resolves sex 5 as absent and 1/2/3/4/6 as present.
    sex_resolved = sex.isin([1, 2, 3, 4, 5, 6])
    sex_unresolved = ~sex_resolved

    ambiguous_8888 = sex_unresolved & age.eq(8888)

    # Cleaning "No co-applicant" codes.
    no_coapp_clean = eth.eq(5) | race.eq(8)

    # Explicit "not applicable" demographic evidence.
    not_applicable_demo = eth.eq(4) | race.eq(7)

    eligible = eligible_mask(df)

    disputed_eligible = eligible & ambiguous_8888
    wrong_current_present = disputed_eligible & no_coapp_clean

    explicit_na_context = (
        disputed_eligible
        & ~no_coapp_clean
        & not_applicable_demo
    )

    unresolved_after_tiebreak = (
        disputed_eligible
        & ~no_coapp_clean
        & ~not_applicable_demo
    )

    summary = {
        "year": year,
        "all_rows": len(df),
        "ambiguous_8888_all": int(ambiguous_8888.sum()),
        "eligible_rows": int(eligible.sum()),
        "ambiguous_8888_eligible": int(disputed_eligible.sum()),
        "wrong_current_present_no_coapp_code": int(wrong_current_present.sum()),
        "explicit_not_applicable_demo_context": int(explicit_na_context.sum()),
        "unresolved_after_eth_race_tiebreak": int(
            unresolved_after_tiebreak.sum()
        ),
    }

    examples = df.loc[
        disputed_eligible,
        [
            "lei",
            "action_taken",
            "submission_of_application",
            "co_applicant_sex",
            "co_applicant_age",
            "co_applicant_ethnicity_1",
            "co_applicant_race_1",
        ],
    ].copy()

    if len(examples):
        examples.insert(0, "year", year)
        examples["clean_no_coapp_code"] = (
            no_coapp_clean.loc[examples.index]
            .astype("boolean")
            .to_numpy()
        )
        examples["demo_not_applicable_context"] = (
            not_applicable_demo.loc[examples.index]
            .astype("boolean")
            .to_numpy()
        )

    return summary, examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = find_year_sources()

    print("=" * 96)
    print("TARGETED CO-APPLICANT AGE 8888 AUDIT")
    print("=" * 96)

    if len(sources) != len(YEARS):
        missing = sorted(set(YEARS) - set(sources))
        print("\nCould not auto-detect raw HMDA files for:")
        print("  " + ", ".join(map(str, missing)))
        print(
            "\nSet RAW_DIR near the top of this script to the folder containing "
            "your 2018–2024 national public LAR ZIP/CSV files, then rerun."
        )
        return

    print("\nRaw sources:")
    for year in YEARS:
        print(f"  {year}: {sources[year]}")

    by_year = []
    example_frames = []

    for year in YEARS:
        print(f"\nScanning {year}...")
        source = sources[year]

        year_counter = Counter()

        for cno, chunk in enumerate(read_chunks(source), 1):
            row, examples = audit_chunk(chunk, year)

            for k, v in row.items():
                if k != "year":
                    year_counter[k] += int(v)

            # Keep only a small sample for manual inspection.
            if len(examples):
                example_frames.append(
                    examples.head(20).copy()
                )

            if cno % 20 == 0:
                print(
                    f"  chunks={cno:>3} | "
                    f"eligible={year_counter['eligible_rows']:,} | "
                    f"ambiguous eligible={year_counter['ambiguous_8888_eligible']:,} | "
                    f"clean no-coapp contradictions="
                    f"{year_counter['wrong_current_present_no_coapp_code']:,}"
                )

        by_year.append({
            "year": year,
            **dict(year_counter),
        })

    by_year_df = pd.DataFrame(by_year)

    numeric_cols = [
        c for c in by_year_df.columns
        if c != "year"
    ]

    totals = {
        c: int(by_year_df[c].sum())
        for c in numeric_cols
    }

    eligible_n = totals.get("eligible_rows", 0)
    ambiguous_n = totals.get("ambiguous_8888_eligible", 0)
    contradiction_n = totals.get(
        "wrong_current_present_no_coapp_code",
        0,
    )

    summary_df = pd.DataFrame([
        {
            "metric": "Eligible mortgage applications scanned",
            "n": eligible_n,
            "percent_of_eligible": 100.0,
        },
        {
            "metric": (
                "Eligible rows decided by sex-unresolved + age==8888 fallback"
            ),
            "n": ambiguous_n,
            "percent_of_eligible": (
                100 * ambiguous_n / eligible_n
                if eligible_n else np.nan
            ),
        },
        {
            "metric": (
                "Potentially misclassified as co-app present: "
                "ambiguous age==8888 but ethnicity/race says No co-applicant"
            ),
            "n": contradiction_n,
            "percent_of_eligible": (
                100 * contradiction_n / eligible_n
                if eligible_n else np.nan
            ),
        },
        {
            "metric": (
                "Ambiguous 8888 with ethnicity/race Not Applicable context"
            ),
            "n": totals.get(
                "explicit_not_applicable_demo_context",
                0,
            ),
            "percent_of_eligible": (
                100
                * totals.get(
                    "explicit_not_applicable_demo_context",
                    0,
                )
                / eligible_n
                if eligible_n else np.nan
            ),
        },
        {
            "metric": (
                "Ambiguous 8888 unresolved after ethnicity/race tiebreak"
            ),
            "n": totals.get(
                "unresolved_after_eth_race_tiebreak",
                0,
            ),
            "percent_of_eligible": (
                100
                * totals.get(
                    "unresolved_after_eth_race_tiebreak",
                    0,
                )
                / eligible_n
                if eligible_n else np.nan
            ),
        },
    ])

    examples_df = (
        pd.concat(
            example_frames,
            ignore_index=True,
        )
        if example_frames
        else pd.DataFrame()
    )

    summary_df.to_csv(
        OUTPUT_DIR / "coapp_8888_summary.csv",
        index=False,
    )

    by_year_df.to_csv(
        OUTPUT_DIR / "coapp_8888_by_year.csv",
        index=False,
    )

    examples_df.to_csv(
        OUTPUT_DIR / "coapp_8888_examples.csv",
        index=False,
    )

    try:
        with pd.ExcelWriter(
            OUTPUT_XLSX,
            engine="openpyxl",
        ) as writer:
            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )
            by_year_df.to_excel(
                writer,
                sheet_name="By year",
                index=False,
            )
            examples_df.to_excel(
                writer,
                sheet_name="Examples",
                index=False,
            )
    except Exception as exc:
        print(f"\nExcel workbook not written: {exc}")

    print("\n" + "=" * 96)
    print("AUDIT COMPLETE")
    print("=" * 96)
    print(summary_df.to_string(index=False))

    print(
        "\nDECISION RULE:\n"
        "- If clean no-coapp contradictions are zero or trivially small, "
        "document the audit and do not rebuild V1.\n"
        "- If non-trivial, patch classify_coapplicant so dedicated co-app "
        "ethnicity/race no-coapp codes override the age==8888 fallback, "
        "then assess whether any preferred-sample results change."
    )

    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
