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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "coapplicant_conflict_resolution"
OUTPUT_XLSX = OUTPUT_DIR / "COAPPLICANT_CONFLICT_RESOLUTION.xlsx"

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

SEX_LABELS = {
    1: "1 Male",
    2: "2 Female",
    3: "3 Information not provided",
    4: "4 Not applicable",
    5: "5 No co-applicant",
    6: "6 Both male and female",
}


def norm_col(x: str) -> str:
    x = str(x).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", x).strip("_")


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


def candidate_raw_dirs():
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


def source_score(path: Path, year: int):
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

    if any(
        term in str(path).lower()
        for term in [
            "audit", "output", "summary", "feasibility",
            "candidate", "final_data", "parts",
        ]
    ):
        score -= 20

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return score, size


def find_year_sources():
    files = []
    for root in candidate_raw_dirs():
        try:
            files.extend(
                p for p in root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".zip", ".csv", ".txt", ".tsv"}
            )
        except Exception:
            continue

    sources = {}
    for year in YEARS:
        candidates = [p for p in files if str(year) in p.name]
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


def detect_sep(first_line: str):
    return max([",", "|", "\t"], key=lambda x: first_line.count(x))


def choose_zip_member(zf: zipfile.ZipFile):
    members = [
        n for n in zf.namelist()
        if n.lower().endswith((".csv", ".txt", ".tsv"))
    ]
    if not members:
        raise ValueError("ZIP contains no CSV/TXT/TSV member.")

    def score(name):
        low = name.lower()
        s = (5 if "lar" in low else 0) + (3 if "public" in low else 0)
        return s, zf.getinfo(name).file_size

    return sorted(members, key=score, reverse=True)[0]


def choose_columns(columns):
    mapping = {norm_col(c): c for c in columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        raise KeyError(
            "Missing raw HMDA columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )
    return [mapping[c] for c in REQUIRED_COLUMNS]


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
            usecols = choose_columns(header.columns)

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
    usecols = choose_columns(header.columns)

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


def eligible_mask(df):
    mask = pd.Series(True, index=df.index)

    for col, allowed in CORE_FILTERS.items():
        mask &= int_codes(get_series(df, col)).isin(allowed)

    units = (
        clean_text(get_series(df, "total_units"))
        .str.replace(r"\.0$", "", regex=True)
    )
    mask &= units.isin(ALLOWED_UNITS)

    mask &= int_codes(
        get_series(df, "action_taken")
    ).isin([1, 2, 3])

    mask &= int_codes(
        get_series(df, "submission_of_application")
    ).isin([1, 2])

    return mask


def sex_label(code):
    if pd.isna(code):
        return "Missing / other"
    return SEX_LABELS.get(int(code), f"Other {code}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = find_year_sources()
    if len(sources) != len(YEARS):
        print("Missing raw-year sources:", sorted(set(YEARS)-set(sources)))
        print("Set RAW_DIR at the top of the script and rerun.")
        return

    overall = Counter()
    by_year_rows = []
    sex9999_rows = []
    examples = []

    print("=" * 100)
    print("FINAL CO-APPLICANT CONFLICT RESOLUTION AUDIT")
    print("=" * 100)

    for year in YEARS:
        yc = Counter()

        for cno, df in enumerate(read_chunks(sources[year]), 1):
            elig = eligible_mask(df)

            sex = int_codes(get_series(df, "co_applicant_sex"))
            age = int_codes(get_series(df, "co_applicant_age"))
            eth = int_codes(get_series(df, "co_applicant_ethnicity_1"))
            race = int_codes(get_series(df, "co_applicant_race_1"))

            sex5 = sex.eq(5)
            age9999 = age.eq(9999)
            eth5 = eth.eq(5)
            race8 = race.eq(8)

            no_coapp_count = (
                sex5.astype("int8")
                + age9999.astype("int8")
                + eth5.astype("int8")
                + race8.astype("int8")
            )

            conflict = elig & age9999 & ~sex5
            strong2 = conflict & no_coapp_count.ge(2)
            strong3 = conflict & no_coapp_count.ge(3)

            reverse_conflict = (
                elig
                & sex5
                & age.between(1, 120)
            )

            yc["eligible"] += int(elig.sum())
            yc["age9999_sex_not5"] += int(conflict.sum())
            yc["age9999_sex_not5_eth5"] += int(
                (conflict & eth5).sum()
            )
            yc["age9999_sex_not5_race8"] += int(
                (conflict & race8).sum()
            )
            yc["age9999_sex_not5_eth5_or_race8"] += int(
                (conflict & (eth5 | race8)).sum()
            )
            yc["age9999_sex_not5_two_plus_no_coapp"] += int(
                strong2.sum()
            )
            yc["age9999_sex_not5_three_plus_no_coapp"] += int(
                strong3.sum()
            )
            yc["sex5_real_age_1_120"] += int(
                reverse_conflict.sum()
            )

            # sex-code distribution among age9999 conflicts
            vals = sex.loc[conflict].value_counts(dropna=False)
            for code, n in vals.items():
                yc[f"sex::{sex_label(code)}"] += int(n)

            if conflict.any() and len(examples) < 100:
                ex = pd.DataFrame({
                    "year": year,
                    "co_applicant_sex": sex.loc[conflict],
                    "co_applicant_age": age.loc[conflict],
                    "co_applicant_ethnicity_1": eth.loc[conflict],
                    "co_applicant_race_1": race.loc[conflict],
                    "no_coapp_indicator_count": no_coapp_count.loc[conflict],
                })
                examples.append(ex.head(20))

            if cno % 20 == 0:
                print(
                    f"{year} chunks={cno:>3} | "
                    f"eligible={yc['eligible']:,} | "
                    f"age9999/sex!=5={yc['age9999_sex_not5']:,} | "
                    f"also eth5/race8={yc['age9999_sex_not5_eth5_or_race8']:,}"
                )

        row = {"year": year, **yc}
        by_year_rows.append(row)
        overall.update(yc)

        for key, val in yc.items():
            if key.startswith("sex::"):
                sex9999_rows.append({
                    "year": year,
                    "sex_code_label": key.split("::",1)[1],
                    "n": val,
                })

    by_year = pd.DataFrame(by_year_rows)
    sex_dist = pd.DataFrame(sex9999_rows)

    conflict_n = overall["age9999_sex_not5"]

    summary = pd.DataFrame([
        {
            "metric": "Eligible applications",
            "n": overall["eligible"],
            "percent_of_conflict": np.nan,
        },
        {
            "metric": "Age 9999 + sex != 5 conflicts",
            "n": conflict_n,
            "percent_of_conflict": 100.0 if conflict_n else np.nan,
        },
        {
            "metric": "Conflicts also showing ethnicity 5 OR race 8",
            "n": overall["age9999_sex_not5_eth5_or_race8"],
            "percent_of_conflict": (
                100 * overall["age9999_sex_not5_eth5_or_race8"] / conflict_n
                if conflict_n else np.nan
            ),
        },
        {
            "metric": "Conflicts with >=2 independent no-coapp indicators",
            "n": overall["age9999_sex_not5_two_plus_no_coapp"],
            "percent_of_conflict": (
                100 * overall["age9999_sex_not5_two_plus_no_coapp"] / conflict_n
                if conflict_n else np.nan
            ),
        },
        {
            "metric": "Conflicts with >=3 independent no-coapp indicators",
            "n": overall["age9999_sex_not5_three_plus_no_coapp"],
            "percent_of_conflict": (
                100 * overall["age9999_sex_not5_three_plus_no_coapp"] / conflict_n
                if conflict_n else np.nan
            ),
        },
        {
            "metric": "Reverse conflict: sex 5 but numeric co-applicant age 1-120",
            "n": overall["sex5_real_age_1_120"],
            "percent_of_conflict": np.nan,
        },
    ])

    examples_df = (
        pd.concat(examples, ignore_index=True)
        if examples
        else pd.DataFrame()
    )

    summary.to_csv(
        OUTPUT_DIR / "coapp_conflict_summary.csv",
        index=False,
    )
    by_year.to_csv(
        OUTPUT_DIR / "coapp_conflict_by_year.csv",
        index=False,
    )
    sex_dist.to_csv(
        OUTPUT_DIR / "age9999_conflict_by_sex.csv",
        index=False,
    )
    examples_df.to_csv(
        OUTPUT_DIR / "coapp_conflict_examples.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        by_year.to_excel(writer, sheet_name="By year", index=False)
        sex_dist.to_excel(writer, sheet_name="Conflict by sex", index=False)
        examples_df.to_excel(writer, sheet_name="Examples", index=False)

    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print(summary.to_string(index=False))

    print(
        "\nDECISION GUIDE:\n"
        "- If almost all age9999/sex!=5 rows also have ethnicity5 or race8, "
        "the weight of the raw evidence says 'No co-applicant' and the sex-first "
        "classifier should be patched for those rows.\n"
        "- If the conflict is concentrated in sex=4 with ethnicity/race Not Applicable "
        "rather than No-coapp codes, investigate reporting conventions before changing anything.\n"
        "- If conflicts are extremely small in the final eligible sample, a sensitivity "
        "patch may be enough rather than rebuilding the entire empirical pipeline."
    )

    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
