from __future__ import annotations

import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


YEARS = list(range(2018, 2025))
CHUNK_SIZE = 175_000
RAW_DIR: Optional[Path] = None

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "coapplicant_code_crosstab_audit"
OUTPUT_XLSX = OUTPUT_DIR / "COAPPLICANT_CODE_CROSSTAB_AUDIT.xlsx"

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


def find_year_sources() -> dict[int, Path]:
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


def detect_sep(first_line: str) -> str:
    return max([",", "|", "\t"], key=lambda x: first_line.count(x))


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


def eligible_mask(df: pd.DataFrame) -> pd.Series:
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


def sex_bucket(sex: pd.Series) -> pd.Series:
    out = pd.Series("Missing / other", index=sex.index, dtype="string")

    for code, label in SEX_LABELS.items():
        out.loc[sex.eq(code)] = label

    return out


def add_counts(counter, prefix, sex, age, eth, race, mask):
    s = sex.loc[mask]
    a = age.loc[mask]
    e = eth.loc[mask]
    r = race.loc[mask]

    counter[f"{prefix}_n"] += int(mask.sum())

    # Headline age / sex counts.
    counter[f"{prefix}_age8888"] += int(a.eq(8888).sum())
    counter[f"{prefix}_age9999"] += int(a.eq(9999).sum())
    counter[f"{prefix}_sex4"] += int(s.eq(4).sum())
    counter[f"{prefix}_sex5"] += int(s.eq(5).sum())

    # Raw-code consistency contradictions.
    age8888 = a.eq(8888)
    age9999 = a.eq(9999)
    sex4 = s.eq(4)
    sex5 = s.eq(5)

    no_coapp_demo = e.eq(5) | r.eq(8)
    na_demo = e.eq(4) | r.eq(7)

    counter[f"{prefix}_age8888_sex4"] += int((age8888 & sex4).sum())
    counter[f"{prefix}_age8888_sex5"] += int((age8888 & sex5).sum())
    counter[f"{prefix}_age8888_eth5_or_race8"] += int(
        (age8888 & no_coapp_demo).sum()
    )
    counter[f"{prefix}_age8888_eth4_or_race7"] += int(
        (age8888 & na_demo).sum()
    )

    counter[f"{prefix}_age9999_sex5"] += int((age9999 & sex5).sum())
    counter[f"{prefix}_age9999_sex_not5"] += int((age9999 & ~sex5).sum())
    counter[f"{prefix}_age9999_no_coapp_demo"] += int(
        (age9999 & no_coapp_demo).sum()
    )

    counter[f"{prefix}_sex4_age8888"] += int((sex4 & age8888).sum())
    counter[f"{prefix}_sex4_age_not8888"] += int((sex4 & ~age8888).sum())

    # Sex distribution among age 8888.
    sb = sex_bucket(s)
    for label, n in sb.loc[age8888].value_counts(dropna=False).items():
        counter[f"{prefix}_age8888_sex::{label}"] += int(n)

    # Age distribution among sex 4: explicit buckets.
    a4 = a.loc[sex4]
    counter[f"{prefix}_sex4_age::8888"] += int(a4.eq(8888).sum())
    counter[f"{prefix}_sex4_age::9999"] += int(a4.eq(9999).sum())
    counter[f"{prefix}_sex4_age::1_120"] += int(a4.between(1, 120).sum())
    counter[f"{prefix}_sex4_age::missing_other"] += int(
        ~(a4.eq(8888) | a4.eq(9999) | a4.between(1, 120))
    ).sum()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = find_year_sources()

    if len(sources) != len(YEARS):
        missing = sorted(set(YEARS) - set(sources))
        print("Could not auto-detect raw files for:", missing)
        print(
            "\nSet RAW_DIR near the top of the script to the folder "
            "containing the 2018–2024 national public LAR files."
        )
        return

    print("=" * 100)
    print("CO-APPLICANT CODING CROSS-TAB AUDIT")
    print("=" * 100)

    for year in YEARS:
        print(f"{year}: {sources[year]}")

    by_year_rows = []
    crosstab_rows = []

    for year in YEARS:
        print(f"\nScanning {year}...")
        counter = Counter()

        for cno, df in enumerate(read_chunks(sources[year]), 1):
            sex = int_codes(get_series(df, "co_applicant_sex"))
            age = int_codes(get_series(df, "co_applicant_age"))
            eth = int_codes(get_series(df, "co_applicant_ethnicity_1"))
            race = int_codes(get_series(df, "co_applicant_race_1"))

            all_mask = pd.Series(True, index=df.index)
            elig = eligible_mask(df)

            add_counts(counter, "all", sex, age, eth, race, all_mask)
            add_counts(counter, "eligible", sex, age, eth, race, elig)

            if cno % 20 == 0:
                print(
                    f"  chunks={cno:>3} | "
                    f"eligible={counter['eligible_n']:,} | "
                    f"eligible age8888={counter['eligible_age8888']:,} | "
                    f"of which sex4={counter['eligible_age8888_sex4']:,} | "
                    f"sex5={counter['eligible_age8888_sex5']:,}"
                )

        headline = {
            "year": year,
            "all_rows": counter["all_n"],
            "eligible_rows": counter["eligible_n"],
            "all_age8888": counter["all_age8888"],
            "eligible_age8888": counter["eligible_age8888"],
            "eligible_age8888_sex4": counter["eligible_age8888_sex4"],
            "eligible_age8888_sex5": counter["eligible_age8888_sex5"],
            "eligible_age8888_eth5_or_race8": counter[
                "eligible_age8888_eth5_or_race8"
            ],
            "eligible_age8888_eth4_or_race7": counter[
                "eligible_age8888_eth4_or_race7"
            ],
            "eligible_age9999": counter["eligible_age9999"],
            "eligible_age9999_sex5": counter["eligible_age9999_sex5"],
            "eligible_age9999_sex_not5": counter[
                "eligible_age9999_sex_not5"
            ],
            "eligible_sex4": counter["eligible_sex4"],
            "eligible_sex4_age8888": counter["eligible_sex4_age8888"],
            "eligible_sex4_age_not8888": counter[
                "eligible_sex4_age_not8888"
            ],
        }
        by_year_rows.append(headline)

        for scope in ["all", "eligible"]:
            for label in list(SEX_LABELS.values()) + ["Missing / other"]:
                crosstab_rows.append({
                    "year": year,
                    "scope": scope,
                    "condition": "co_applicant_age == 8888",
                    "sex_code_label": label,
                    "n": counter[f"{scope}_age8888_sex::{label}"],
                })

    by_year = pd.DataFrame(by_year_rows)
    crosstab = pd.DataFrame(crosstab_rows)

    totals = {
        c: int(by_year[c].sum())
        for c in by_year.columns
        if c != "year"
    }

    summary = pd.DataFrame([
        {
            "metric": "Eligible applications",
            "n": totals["eligible_rows"],
        },
        {
            "metric": "Eligible co-applicant age == 8888",
            "n": totals["eligible_age8888"],
        },
        {
            "metric": "Age 8888 paired with co-applicant sex == 4 (Not applicable)",
            "n": totals["eligible_age8888_sex4"],
        },
        {
            "metric": "Age 8888 paired with co-applicant sex == 5 (No co-applicant)",
            "n": totals["eligible_age8888_sex5"],
        },
        {
            "metric": "Age 8888 paired with ethnicity 5 OR race 8 (No co-applicant)",
            "n": totals["eligible_age8888_eth5_or_race8"],
        },
        {
            "metric": "Age 8888 paired with ethnicity 4 OR race 7 (Not applicable)",
            "n": totals["eligible_age8888_eth4_or_race7"],
        },
        {
            "metric": "Eligible co-applicant age == 9999",
            "n": totals["eligible_age9999"],
        },
        {
            "metric": "Age 9999 paired with co-applicant sex == 5",
            "n": totals["eligible_age9999_sex5"],
        },
        {
            "metric": "Age 9999 paired with sex != 5",
            "n": totals["eligible_age9999_sex_not5"],
        },
        {
            "metric": "Eligible co-applicant sex == 4",
            "n": totals["eligible_sex4"],
        },
        {
            "metric": "Sex 4 paired with age == 8888",
            "n": totals["eligible_sex4_age8888"],
        },
        {
            "metric": "Sex 4 paired with age != 8888",
            "n": totals["eligible_sex4_age_not8888"],
        },
    ])

    # Consistency metrics.
    age8888_n = totals["eligible_age8888"]
    if age8888_n:
        summary["percent_of_age8888"] = np.nan
        for idx in summary.index:
            if summary.loc[idx, "metric"].startswith("Age 8888 paired"):
                summary.loc[idx, "percent_of_age8888"] = (
                    100 * summary.loc[idx, "n"] / age8888_n
                )

    by_year.to_csv(
        OUTPUT_DIR / "coapp_code_crosstab_by_year.csv",
        index=False,
    )
    crosstab.to_csv(
        OUTPUT_DIR / "age8888_by_sex_crosstab.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "coapp_code_crosstab_summary.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        by_year.to_excel(writer, sheet_name="By year", index=False)
        crosstab.to_excel(writer, sheet_name="Age8888 by sex", index=False)

    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print(summary.to_string(index=False))

    print(
        "\nInterpretation guide:\n"
        "1. If age8888 is common and virtually all of it pairs with sex=4, "
        "the first audit's zero is expected: sex already resolved those rows.\n"
        "2. If age8888 pairs materially with sex=5 or ethnicity=5/race=8, "
        "there is contradictory raw coding that needs investigation.\n"
        "3. If sex=4 almost always pairs with age8888, the original classifier "
        "is internally consistent with HMDA's Not-applicable coding.\n"
        "4. No dataset change should be made until these cross-tabs are known."
    )

    print(f"\nOutputs:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
