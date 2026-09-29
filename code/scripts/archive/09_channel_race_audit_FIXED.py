import io
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# 1. CONFIGURATION

YEARS = list(range(2018, 2025))
CHUNK_SIZE = 175_000

RAW_DIR: Optional[Path] = None

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "channel_race_feasibility"
OUTPUT_XLSX = OUTPUT_DIR / "CHANNEL_RACE_FEASIBILITY_AUDIT.xlsx"

CORE_FILTERS = {
    "loan_purpose": {1},                  # home purchase
    "occupancy_type": {1},               # principal residence
    "construction_method": {1},          # site-built
    "open_end_line_of_credit": {2},      # closed-end
    "reverse_mortgage": {2},             # not reverse mortgage
    "business_or_commercial_purpose": {2},
    "lien_status": {1},                  # first lien
}
ALLOWED_UNITS = {"1", "2", "3", "4"}
DECISION_ACTIONS = {1, 2, 3}
APPROVED_ACTIONS = {1, 2}
CHANNEL_CODES = {1: "Direct", 2: "Non-direct"}
SUPPORT_THRESHOLDS = [1, 10, 25, 50, 100]

HISPANIC_CODES = {1, 11, 12, 13, 14}
NON_HISPANIC_CODE = 2

RACE_FAMILY_BY_CODE = {
    1: "AIAN",
    2: "Asian", 21: "Asian", 22: "Asian", 23: "Asian", 24: "Asian",
    25: "Asian", 26: "Asian", 27: "Asian",
    3: "Black",
    4: "NHPI", 41: "NHPI", 42: "NHPI", 43: "NHPI", 44: "NHPI",
    5: "White",
}
RACE_FAMILY_ORDER = ["White", "Black", "Asian", "AIAN", "NHPI"]

ASIAN_SUBGROUP_BY_CODE = {
    21: "Asian Indian", 22: "Chinese", 23: "Filipino", 24: "Japanese",
    25: "Korean", 26: "Vietnamese", 27: "Other Asian",
}
NHPI_SUBGROUP_BY_CODE = {
    41: "Native Hawaiian", 42: "Guamanian or Chamorro",
    43: "Samoan", 44: "Other Pacific Islander",
}

PRIMARY_GROUP_ORDER = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic/Latino",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
    "Non-Hispanic NHPI",
    "Non-Hispanic Multiracial",
    "Unknown/Not reported",
]

REQUIRED_COLUMNS = [
    "activity_year", "lei", "action_taken", "submission_of_application",
    "loan_type", "loan_purpose", "occupancy_type", "construction_method",
    "total_units", "open_end_line_of_credit", "reverse_mortgage",
    "business_or_commercial_purpose", "lien_status",
]
for prefix in ["applicant", "co_applicant"]:
    for i in range(1, 6):
        REQUIRED_COLUMNS.append(f"{prefix}_ethnicity_{i}")
        REQUIRED_COLUMNS.append(f"{prefix}_race_{i}")

# Minimum columns that must be recognised before a source/member can be treated as HMDA LAR.
KEY_COLUMNS = {
    "activity_year", "lei", "action_taken", "submission_of_application",
    "loan_purpose", "occupancy_type", "lien_status",
}


# 2. HELPERS

def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def numeric(series: pd.Series) -> pd.Series:
    s = clean_text(series).replace({
        "": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA,
        "None": pd.NA, "Exempt": pd.NA, "exempt": pd.NA, "NULL": pd.NA,
    })
    return pd.to_numeric(s, errors="coerce")


def int_codes(series: pd.Series) -> pd.Series:
    return numeric(series).astype("Int64")


def get_series(df: pd.DataFrame, col: str, default=pd.NA) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype="string")


def period_from_year(year: int) -> str:
    if year in (2018, 2019):
        return "2018-2019 Pre-COVID"
    if year in (2020, 2021):
        return "2020-2021 COVID / low-rate"
    if year in (2022, 2023, 2024):
        return "2022-2024 Tightening / high-rate"
    return "Other"


def detect_sep_from_text(first_line: str) -> str:
    counts = {",": first_line.count(","), "|": first_line.count("|"), "\t": first_line.count("\t")}
    return max(counts, key=counts.get)


# 3. FIND RAW HMDA SOURCES

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
    if str(year) in name: score += 10
    if "combined" in name: score += 8
    if "mlar" in name or "lar" in name: score += 8
    if "public" in name: score += 5
    if "modified" in name: score += 3
    if "nation" in name: score += 2
    if path.suffix.lower() == ".zip": score += 2
    bad_terms = ["audit", "output", "summary", "feasibility", "candidate", "final_data", "parts"]
    if any(term in str(path).lower() for term in bad_terms): score -= 30
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
                if p.is_file() and p.suffix.lower() in {".zip", ".csv", ".txt", ".tsv"}
            )
        except Exception:
            continue

    sources: dict[int, Path] = {}
    for year in YEARS:
        candidates = [p for p in all_files if str(year) in p.name]
        if candidates:
            ranked = sorted(candidates, key=lambda p: source_score(p, year), reverse=True)
            if source_score(ranked[0], year)[0] > 0:
                sources[year] = ranked[0]
    return sources


def _header_info_from_zip_member(zf: zipfile.ZipFile, member: str):
    """Return (separator, header, overlap) or None if member is not parseable as tabular text."""
    try:
        with zf.open(member) as fh:
            wrapper = io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace")
            first_line = wrapper.readline()
        if not first_line.strip():
            return None
        sep = detect_sep_from_text(first_line)
        with zf.open(member) as fh:
            header = pd.read_csv(fh, sep=sep, nrows=0, encoding="utf-8-sig")
        normed = {norm_col(c) for c in header.columns}
        overlap = len(normed.intersection(REQUIRED_COLUMNS))
        key_overlap = len(normed.intersection(KEY_COLUMNS))
        return sep, header, overlap, key_overlap
    except Exception:
        return None


def choose_hmda_zip_member(zf: zipfile.ZipFile):
    members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt", ".tsv"))]
    if not members:
        raise ValueError("ZIP contains no CSV/TXT/TSV members")

    candidates = []
    diagnostics = []
    for member in members:
        info = _header_info_from_zip_member(zf, member)
        size = zf.getinfo(member).file_size
        if info is None:
            diagnostics.append((member, 0, 0, size))
            continue
        sep, header, overlap, key_overlap = info
        diagnostics.append((member, overlap, key_overlap, size))
        if key_overlap >= 4 and overlap >= 6:
            candidates.append((key_overlap, overlap, size, member, sep, header))

    if not candidates:
        diag = "\n".join(
            f"  {m}: recognised={o}, key={k}, bytes={s:,}"
            for m, o, k, s in sorted(diagnostics, key=lambda x: (x[2], x[1], x[3]), reverse=True)[:20]
        )
        raise ValueError(
            "Could not identify an HMDA LAR data member inside ZIP.\n"
            "ZIP member diagnostics:\n" + diag
        )

    # Highest key-column overlap, then total overlap, then largest file.
    _, overlap, _, member, sep, header = sorted(candidates, reverse=True)[0]
    return member, sep, header, overlap


def _plain_header_info(path: Path):
    # Try delimiter inferred from first line, then common alternatives.
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline()
    inferred = detect_sep_from_text(first_line)
    seps = []
    for sep in [inferred, ",", "|", "\t"]:
        if sep not in seps:
            seps.append(sep)

    best = None
    for sep in seps:
        try:
            header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
            normed = {norm_col(c) for c in header.columns}
            overlap = len(normed.intersection(REQUIRED_COLUMNS))
            key_overlap = len(normed.intersection(KEY_COLUMNS))
            cand = (key_overlap, overlap, sep, header)
            if best is None or cand[:2] > best[:2]:
                best = cand
        except Exception:
            continue
    return best


def read_chunks_from_source(path: Path) -> Iterable[pd.DataFrame]:
    """Yield normalised HMDA chunks without loading an annual national file into memory."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member, sep, header, overlap = choose_hmda_zip_member(zf)
            print(f"    ZIP member: {member} | recognised HMDA columns: {overlap}")
            mapping = {norm_col(c): c for c in header.columns}
            use_norm = [c for c in REQUIRED_COLUMNS if c in mapping]
            use_original = [mapping[c] for c in use_norm]
            missing_keys = sorted(KEY_COLUMNS.difference(mapping))
            if missing_keys:
                raise ValueError(f"HMDA source is missing essential columns: {missing_keys}")

            with zf.open(member) as fh:
                for chunk in pd.read_csv(
                    fh, sep=sep, usecols=use_original, dtype="string",
                    chunksize=CHUNK_SIZE, low_memory=False,
                    encoding="utf-8-sig", on_bad_lines="warn",
                ):
                    chunk.columns = [norm_col(c) for c in chunk.columns]
                    yield chunk
        return

    info = _plain_header_info(path)
    if info is None:
        raise ValueError(f"Could not parse header from {path}")
    key_overlap, overlap, sep, header = info
    mapping = {norm_col(c): c for c in header.columns}
    if key_overlap < 4 or overlap < 6:
        raise ValueError(
            f"No recognised HMDA header in {path}. Recognised {overlap} requested columns "
            f"and {key_overlap} essential columns."
        )
    use_norm = [c for c in REQUIRED_COLUMNS if c in mapping]
    use_original = [mapping[c] for c in use_norm]
    print(f"    delimiter={repr(sep)} | recognised HMDA columns: {overlap}")

    for chunk in pd.read_csv(
        path, sep=sep, usecols=use_original, dtype="string",
        chunksize=CHUNK_SIZE, low_memory=False,
        encoding="utf-8-sig", on_bad_lines="warn",
    ):
        chunk.columns = [norm_col(c) for c in chunk.columns]
        yield chunk


# 4. SAMPLE FILTERS

def apply_core_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, allowed in CORE_FILTERS.items():
        mask &= int_codes(get_series(df, col)).isin(allowed)
    units = clean_text(get_series(df, "total_units")).str.replace(r"\.0$", "", regex=True)
    mask &= units.isin(ALLOWED_UNITS)
    return df.loc[mask].copy()


def keep_channel_decisions(df: pd.DataFrame) -> pd.DataFrame:
    action = int_codes(get_series(df, "action_taken"))
    channel = int_codes(get_series(df, "submission_of_application"))
    return df.loc[action.isin(DECISION_ACTIONS) & channel.isin(CHANNEL_CODES)].copy()


# 5. DEMOGRAPHIC CLASSIFICATION

def code_matrix(df: pd.DataFrame, prefix: str, kind: str) -> pd.DataFrame:
    cols = [f"{prefix}_{kind}_{i}" for i in range(1, 6)]
    return pd.concat([int_codes(get_series(df, c)).rename(c) for c in cols], axis=1)


def classify_person(df: pd.DataFrame, prefix: str = "applicant") -> pd.DataFrame:
    """Mutually exclusive person-level race/ethnicity classification using all five HMDA fields."""
    eth = code_matrix(df, prefix, "ethnicity")
    race = code_matrix(df, prefix, "race")

    if prefix == "co_applicant":
        no_coapp = eth.eq(5).any(axis=1) | race.eq(8).any(axis=1)
    else:
        no_coapp = pd.Series(False, index=df.index)

    is_hispanic = eth.isin(HISPANIC_CODES).any(axis=1)
    is_non_hispanic = eth.eq(NON_HISPANIC_CODE).any(axis=1) & ~is_hispanic

    fam = pd.DataFrame(False, index=df.index, columns=RACE_FAMILY_ORDER)
    for code, family in RACE_FAMILY_BY_CODE.items():
        fam[family] |= race.eq(code).any(axis=1)
    family_n = fam.sum(axis=1)

    race_combo = pd.Series("", index=df.index, dtype="string")
    for family in RACE_FAMILY_ORDER:
        has = fam[family]
        empty = race_combo.eq("")
        race_combo.loc[has & ~empty] = race_combo.loc[has & ~empty] + " + " + family
        race_combo.loc[has & empty] = family
    race_combo = race_combo.replace("", pd.NA)

    broad = pd.Series("Unknown/Not reported", index=df.index, dtype="string")
    broad.loc[is_hispanic & ~no_coapp] = "Hispanic/Latino"
    nh = is_non_hispanic & ~no_coapp
    broad.loc[nh & family_n.eq(1) & fam["White"]] = "Non-Hispanic White"
    broad.loc[nh & family_n.eq(1) & fam["Black"]] = "Non-Hispanic Black"
    broad.loc[nh & family_n.eq(1) & fam["Asian"]] = "Non-Hispanic Asian"
    broad.loc[nh & family_n.eq(1) & fam["AIAN"]] = "Non-Hispanic AIAN"
    broad.loc[nh & family_n.eq(1) & fam["NHPI"]] = "Non-Hispanic NHPI"
    broad.loc[nh & family_n.gt(1)] = "Non-Hispanic Multiracial"
    if prefix == "co_applicant":
        broad.loc[no_coapp] = "No co-applicant"

    detailed_asian = pd.DataFrame(
        {name: race.eq(code).any(axis=1) for code, name in ASIAN_SUBGROUP_BY_CODE.items()},
        index=df.index,
    )
    asian_sub = pd.Series(pd.NA, index=df.index, dtype="string")
    asian_mask = broad.eq("Non-Hispanic Asian")
    n_asian = detailed_asian.sum(axis=1)
    asian_sub.loc[asian_mask & n_asian.eq(0)] = "Asian - unspecified"
    for name in ASIAN_SUBGROUP_BY_CODE.values():
        asian_sub.loc[asian_mask & n_asian.eq(1) & detailed_asian[name]] = name
    asian_sub.loc[asian_mask & n_asian.gt(1)] = "Multiple Asian subgroups"

    detailed_nhpi = pd.DataFrame(
        {name: race.eq(code).any(axis=1) for code, name in NHPI_SUBGROUP_BY_CODE.items()},
        index=df.index,
    )
    nhpi_sub = pd.Series(pd.NA, index=df.index, dtype="string")
    nhpi_mask = broad.eq("Non-Hispanic NHPI")
    n_nhpi = detailed_nhpi.sum(axis=1)
    nhpi_sub.loc[nhpi_mask & n_nhpi.eq(0)] = "NHPI - unspecified"
    for name in NHPI_SUBGROUP_BY_CODE.values():
        nhpi_sub.loc[nhpi_mask & n_nhpi.eq(1) & detailed_nhpi[name]] = name
    nhpi_sub.loc[nhpi_mask & n_nhpi.gt(1)] = "Multiple NHPI subgroups"

    return pd.DataFrame({
        "broad_group": broad,
        "race_combo": race_combo,
        "asian_subgroup": asian_sub,
        "nhpi_subgroup": nhpi_sub,
        "no_coapplicant": no_coapp,
    }, index=df.index)


# 6. AGGREGATION HELPERS

def add_value_counts(counter: Counter, frame: pd.DataFrame, cols: list[str]) -> None:
    vc = frame.value_counts(subset=cols, dropna=False)
    for key, n in vc.items():
        if not isinstance(key, tuple):
            key = (key,)
        counter[tuple(key)] += int(n)


def counter_df(counter: Counter, columns: list[str], value_name="n") -> pd.DataFrame:
    rows = []
    for key, value in counter.items():
        if not isinstance(key, tuple):
            key = (key,)
        rows.append((*key, value))
    if not rows:
        return pd.DataFrame(columns=columns + [value_name])
    return pd.DataFrame(rows, columns=columns + [value_name])


def add_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Input must have action_taken and n; collapse to totals, approvals, denials and approval rate."""
    group_cols = [c for c in df.columns if c not in {"action_taken", "n"}]
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["total_n", "approved_n", "denied_n", "approval_rate"])
    d = df.copy()
    d["approved_n_part"] = np.where(d["action_taken"].isin(APPROVED_ACTIONS), d["n"], 0)
    d["denied_n_part"] = np.where(d["action_taken"].eq(3), d["n"], 0)
    out = d.groupby(group_cols, dropna=False, as_index=False).agg(
        total_n=("n", "sum"),
        approved_n=("approved_n_part", "sum"),
        denied_n=("denied_n_part", "sum"),
    )
    out["approval_rate"] = np.where(out["total_n"] > 0, out["approved_n"] / out["total_n"], np.nan)
    return out


def ordered_race(df: pd.DataFrame, col="race_ethnicity") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.Categorical(df[col], categories=PRIMARY_GROUP_ORDER, ordered=True)
        sort_cols = [c for c in [col, "year", "period", "channel"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df[col] = df[col].astype("string")
    return df


# 7. MAIN SCAN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = find_year_sources()
    missing = [y for y in YEARS if y not in sources]
    if missing:
        raise FileNotFoundError(
            f"Could not auto-detect raw HMDA sources for years: {missing}. "
            "Set RAW_DIR near the top of the script to the folder containing the annual files."
        )

    print("=" * 92)
    print("CHANNEL RACE / ETHNICITY FEASIBILITY AUDIT — HMDA 2018–2024")
    print("=" * 92)
    print("Final study sample: home-purchase + core mortgage filters + Actions 1/2/3 + channel 1/2")
    print("Race coding: Hispanic first; otherwise explicitly non-Hispanic + all five race fields")
    print()
    print("Detected annual sources:")
    for year, path in sorted(sources.items()):
        print(f"  {year}: {path}")
    print()

    source_rows = []
    race_action = Counter()          # race, year, period, channel, action
    asian_action = Counter()         # subgroup, year, channel, action
    nhpi_action = Counter()          # subgroup, year, channel, action
    multi_action = Counter()         # race combo, year, channel, action
    pair_counter = Counter()         # applicant group, co-app group, year, channel
    ly_channel = Counter()           # year, lei, channel
    ly_race_channel = Counter()      # year, lei, race, channel

    for source_year in YEARS:
        path = sources[source_year]
        print("-" * 92)
        print(f"YEAR {source_year}: {path.name}")
        raw_n = core_n = sample_n = 0
        chunk_no = 0

        for chunk in read_chunks_from_source(path):
            chunk_no += 1
            raw_n += len(chunk)

            core = apply_core_filters(chunk)
            core_n += len(core)
            if core.empty:
                continue

            sample = keep_channel_decisions(core)
            sample_n += len(sample)
            if sample.empty:
                continue

            years = int_codes(get_series(sample, "activity_year"))
            years = years.where(years.isin(YEARS), source_year).astype("Int64")
            actions = int_codes(get_series(sample, "action_taken"))
            channels = int_codes(get_series(sample, "submission_of_application"))

            sample["__year"] = years
            sample["__period"] = sample["__year"].map(lambda x: period_from_year(int(x)))
            sample["__action"] = actions
            sample["__channel"] = channels.map(CHANNEL_CODES).astype("string")
            sample["__lei"] = clean_text(get_series(sample, "lei"))

            app = classify_person(sample, "applicant")
            co = classify_person(sample, "co_applicant")
            sample["__race"] = app["broad_group"]
            sample["__race_combo"] = app["race_combo"]
            sample["__asian"] = app["asian_subgroup"]
            sample["__nhpi"] = app["nhpi_subgroup"]
            sample["__co_race"] = co["broad_group"]

            # Core race x channel x outcome counts.
            tmp = sample[["__race", "__year", "__period", "__channel", "__action"]].rename(columns={
                "__race": "race_ethnicity", "__year": "year", "__period": "period",
                "__channel": "channel", "__action": "action_taken",
            })
            add_value_counts(race_action, tmp, ["race_ethnicity", "year", "period", "channel", "action_taken"])

            # Detailed Asian and NHPI counts.
            asian = sample.loc[sample["__race"].eq("Non-Hispanic Asian")]
            if not asian.empty:
                tmp = asian[["__asian", "__year", "__channel", "__action"]].rename(columns={
                    "__asian": "subgroup", "__year": "year", "__channel": "channel", "__action": "action_taken"
                })
                add_value_counts(asian_action, tmp, ["subgroup", "year", "channel", "action_taken"])

            nhpi = sample.loc[sample["__race"].eq("Non-Hispanic NHPI")]
            if not nhpi.empty:
                tmp = nhpi[["__nhpi", "__year", "__channel", "__action"]].rename(columns={
                    "__nhpi": "subgroup", "__year": "year", "__channel": "channel", "__action": "action_taken"
                })
                add_value_counts(nhpi_action, tmp, ["subgroup", "year", "channel", "action_taken"])

            multi = sample.loc[sample["__race"].eq("Non-Hispanic Multiracial")]
            if not multi.empty:
                tmp = multi[["__race_combo", "__year", "__channel", "__action"]].rename(columns={
                    "__race_combo": "race_combo", "__year": "year", "__channel": "channel", "__action": "action_taken"
                })
                add_value_counts(multi_action, tmp, ["race_combo", "year", "channel", "action_taken"])

            # Applicant/co-applicant pair coding audit.
            tmp = sample[["__race", "__co_race", "__year", "__channel"]].rename(columns={
                "__race": "applicant_group", "__co_race": "coapplicant_group",
                "__year": "year", "__channel": "channel",
            })
            add_value_counts(pair_counter, tmp, ["applicant_group", "coapplicant_group", "year", "channel"])

            # Lender-year channel support.
            tmp = sample[["__year", "__lei", "__channel"]].rename(columns={
                "__year": "year", "__lei": "lei", "__channel": "channel"
            })
            add_value_counts(ly_channel, tmp, ["year", "lei", "channel"])

            tmp = sample[["__year", "__lei", "__race", "__channel"]].rename(columns={
                "__year": "year", "__lei": "lei", "__race": "race_ethnicity", "__channel": "channel"
            })
            add_value_counts(ly_race_channel, tmp, ["year", "lei", "race_ethnicity", "channel"])

            if chunk_no % 10 == 0:
                print(f"  chunks {chunk_no:>3} | raw {raw_n:,} | core {core_n:,} | channel decisions {sample_n:,}")

        print(f"  DONE {source_year}: raw={raw_n:,}, core={core_n:,}, channel decisions={sample_n:,}")
        source_rows.append({
            "year": source_year, "source_file": str(path), "raw_rows": raw_n,
            "core_filter_rows": core_n, "channel_decision_rows": sample_n,
            "sample_share_of_core": sample_n / core_n if core_n else np.nan,
        })

    # Build output tables.
    source_audit = pd.DataFrame(source_rows)

    race_raw = counter_df(
        race_action,
        ["race_ethnicity", "year", "period", "channel", "action_taken"]
    )

    race_overall = add_outcome_columns(
        race_raw.groupby(["race_ethnicity", "action_taken"], as_index=False)["n"].sum()
    )
    race_overall = ordered_race(race_overall)

    race_by_year = add_outcome_columns(
        race_raw.groupby(["race_ethnicity", "year", "action_taken"], as_index=False)["n"].sum()
    )
    race_by_year = ordered_race(race_by_year)

    race_by_channel = add_outcome_columns(
        race_raw.groupby(["race_ethnicity", "channel", "action_taken"], as_index=False)["n"].sum()
    )
    race_by_channel = ordered_race(race_by_channel)

    race_channel_year = add_outcome_columns(
        race_raw.groupby(["race_ethnicity", "year", "channel", "action_taken"], as_index=False)["n"].sum()
    )
    race_channel_year = ordered_race(race_channel_year)

    race_channel_period = add_outcome_columns(
        race_raw.groupby(["race_ethnicity", "period", "channel", "action_taken"], as_index=False)["n"].sum()
    )
    race_channel_period = ordered_race(race_channel_period)

    aian_nhpi = race_channel_year.loc[
        race_channel_year["race_ethnicity"].isin(["Non-Hispanic AIAN", "Non-Hispanic NHPI"])
    ].copy()
    if not aian_nhpi.empty:
        mins = aian_nhpi.groupby(["race_ethnicity", "channel"], as_index=False).agg(
            min_year_total_n=("total_n", "min"),
            min_year_denied_n=("denied_n", "min"),
            max_year_total_n=("total_n", "max"),
            total_n_all_years=("total_n", "sum"),
            denied_n_all_years=("denied_n", "sum"),
        )
        aian_nhpi = aian_nhpi.merge(mins, on=["race_ethnicity", "channel"], how="left")

    asian_subgroups = add_outcome_columns(counter_df(asian_action, ["subgroup", "year", "channel", "action_taken"]))
    nhpi_subgroups = add_outcome_columns(counter_df(nhpi_action, ["subgroup", "year", "channel", "action_taken"]))
    multiracial = add_outcome_columns(counter_df(multi_action, ["race_combo", "year", "channel", "action_taken"]))
    coapp_pairs = counter_df(pair_counter, ["applicant_group", "coapplicant_group", "year", "channel"])

    # Dual-channel total support by lender-year.
    ly = counter_df(ly_channel, ["year", "lei", "channel"])
    if not ly.empty:
        piv = ly.pivot_table(index=["year", "lei"], columns="channel", values="n", aggfunc="sum", fill_value=0).reset_index()
        for c in ["Direct", "Non-direct"]:
            if c not in piv.columns:
                piv[c] = 0
        support_rows = []
        for t in SUPPORT_THRESHOLDS:
            keep = (piv["Direct"] >= t) & (piv["Non-direct"] >= t)
            support_rows.append({
                "threshold_each_channel": t,
                "lender_years": int(keep.sum()),
                "direct_apps": int(piv.loc[keep, "Direct"].sum()),
                "non_direct_apps": int(piv.loc[keep, "Non-direct"].sum()),
                "total_apps": int(piv.loc[keep, ["Direct", "Non-direct"]].to_numpy().sum()),
            })
        dual_support = pd.DataFrame(support_rows)
    else:
        piv = pd.DataFrame()
        dual_support = pd.DataFrame()

    # Race-specific common support: White + target race must each have >= threshold in BOTH channels within the same lender-year.
    lyr = counter_df(ly_race_channel, ["year", "lei", "race_ethnicity", "channel"])
    common_rows = []
    if not lyr.empty:
        rp = lyr.pivot_table(
            index=["year", "lei"], columns=["race_ethnicity", "channel"],
            values="n", aggfunc="sum", fill_value=0
        )
        targets = [
            "Non-Hispanic Black", "Hispanic/Latino", "Non-Hispanic Asian",
            "Non-Hispanic AIAN", "Non-Hispanic NHPI", "Non-Hispanic Multiracial",
        ]
        white = "Non-Hispanic White"
        for target in targets:
            for t in [1, 5, 10, 25, 50]:
                cols = [(white, "Direct"), (white, "Non-direct"), (target, "Direct"), (target, "Non-direct")]
                vals = []
                for col in cols:
                    if col in rp.columns:
                        vals.append(rp[col])
                    else:
                        vals.append(pd.Series(0, index=rp.index))
                keep = (vals[0] >= t) & (vals[1] >= t) & (vals[2] >= t) & (vals[3] >= t)
                target_apps = int((vals[2][keep] + vals[3][keep]).sum())
                white_apps = int((vals[0][keep] + vals[1][keep]).sum())
                common_rows.append({
                    "target_group": target,
                    "minimum_per_race_channel_cell": t,
                    "lender_years": int(keep.sum()),
                    "target_group_apps": target_apps,
                    "white_apps": white_apps,
                })
    race_common_support = pd.DataFrame(common_rows)

    # Write CSVs + workbook.
    tables = {
        "source_audit": source_audit,
        "race_overall": race_overall,
        "race_by_year": race_by_year,
        "race_by_channel": race_by_channel,
        "race_channel_year": race_channel_year,
        "race_channel_period": race_channel_period,
        "aian_nhpi_cell_check": aian_nhpi,
        "asian_subgroups": asian_subgroups,
        "nhpi_subgroups": nhpi_subgroups,
        "multiracial_combinations": multiracial,
        "coapplicant_pairs": coapp_pairs,
        "dual_channel_support": dual_support,
        "race_common_support": race_common_support,
    }

    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            for name, df in tables.items():
                sheet = name[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = f"Excel workbook not written ({exc}); CSVs were still written."

    print("\n" + "=" * 92)
    print("AUDIT COMPLETE")
    print("=" * 92)
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Workbook: {excel_msg}")
    print("\nFIRST FILES TO SEND BACK:")
    print("  1. source_audit.csv")
    print("  2. race_by_channel.csv")
    print("  3. race_channel_year.csv")
    print("  4. aian_nhpi_cell_check.csv")
    print("  5. race_common_support.csv")
    print("  6. multiracial_combinations.csv")
    print("  7. coapplicant_pairs.csv")
    print("\nThese are the files we need before locking the final race coding and running regressions.")


if __name__ == "__main__":
    main()
