import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
RAW_DIR = PROJECT_DIR / "preapproval_national_data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "pre_regression_integrity_checks"
OUTPUT_XLSX = OUTPUT_DIR / "MULTIRACE_RAW_RECORD_SPOTCHECK.xlsx"

YEARS = list(range(2018, 2025))
CHUNK_SIZE = 175_000
TARGET_PER_CASE = 8

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
DECISION_ACTIONS = {1, 2, 3}
CHANNEL_CODES = {1, 2}

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
ASIAN_DETAIL = set(range(21, 28))
NHPI_DETAIL = set(range(41, 45))

TARGET_CASES = [
    "Asian aggregate + detailed Asian subtype",
    "Multiple detailed Asian subtypes only",
    "NHPI aggregate + detailed NHPI subtype",
    "White + Asian family",
    "Black + White",
    "AIAN + White",
    "Asian + NHPI",
    "Hispanic + one race family",
    "Hispanic + multiple race families",
]

REQUIRED_COLUMNS = [
    "activity_year", "action_taken", "submission_of_application",
    "loan_purpose", "occupancy_type", "construction_method",
    "total_units", "open_end_line_of_credit", "reverse_mortgage",
    "business_or_commercial_purpose", "lien_status",
]
for i in range(1, 6):
    REQUIRED_COLUMNS.append(f"applicant_ethnicity_{i}")
    REQUIRED_COLUMNS.append(f"applicant_race_{i}")

KEY_COLUMNS = {
    "activity_year", "action_taken", "submission_of_application",
    "loan_purpose", "occupancy_type", "lien_status",
}


def norm_col(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def clean_text(series):
    return series.astype("string").str.strip()


def numeric(series):
    s = clean_text(series).replace({
        "": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA,
        "None": pd.NA, "Exempt": pd.NA, "exempt": pd.NA, "NULL": pd.NA,
    })
    return pd.to_numeric(s, errors="coerce")


def int_codes(series):
    return numeric(series).astype("Int64")


def get_series(df, col, default=pd.NA):
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype="string")


def detect_sep_from_text(first_line):
    counts = {",": first_line.count(","), "|": first_line.count("|"), "\t": first_line.count("\t")}
    return max(counts, key=counts.get)


def header_info_from_zip_member(zf, member):
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


def choose_hmda_zip_member(zf):
    members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt", ".tsv"))]
    candidates = []
    for member in members:
        info = header_info_from_zip_member(zf, member)
        if info is None:
            continue
        sep, header, overlap, key_overlap = info
        if key_overlap >= 4 and overlap >= 6:
            size = zf.getinfo(member).file_size
            candidates.append((key_overlap, overlap, size, member, sep, header))
    if not candidates:
        raise ValueError("Could not identify HMDA LAR member inside ZIP.")
    _, overlap, _, member, sep, header = sorted(candidates, reverse=True)[0]
    return member, sep, header, overlap


def read_chunks(path):
    with zipfile.ZipFile(path) as zf:
        member, sep, header, overlap = choose_hmda_zip_member(zf)
        print(f"    ZIP member: {member} | recognised requested columns: {overlap}")
        mapping = {norm_col(c): c for c in header.columns}
        use_norm = [c for c in REQUIRED_COLUMNS if c in mapping]
        use_original = [mapping[c] for c in use_norm]

        missing = sorted(KEY_COLUMNS.difference(mapping))
        if missing:
            raise ValueError(f"Missing essential HMDA columns: {missing}")

        with zf.open(member) as fh:
            for chunk in pd.read_csv(
                fh,
                sep=sep,
                usecols=use_original,
                dtype="string",
                chunksize=CHUNK_SIZE,
                low_memory=False,
                encoding="utf-8-sig",
                on_bad_lines="warn",
            ):
                chunk.columns = [norm_col(c) for c in chunk.columns]
                yield chunk


def apply_study_filters(df):
    mask = pd.Series(True, index=df.index)
    for col, allowed in CORE_FILTERS.items():
        mask &= int_codes(get_series(df, col)).isin(allowed)

    units = clean_text(get_series(df, "total_units")).str.replace(r"\.0$", "", regex=True)
    mask &= units.isin(ALLOWED_UNITS)

    action = int_codes(get_series(df, "action_taken"))
    channel = int_codes(get_series(df, "submission_of_application"))
    mask &= action.isin(DECISION_ACTIONS)
    mask &= channel.isin(CHANNEL_CODES)
    return df.loc[mask].copy()


def code_matrix(df, kind):
    cols = [f"applicant_{kind}_{i}" for i in range(1, 6)]
    return pd.concat([int_codes(get_series(df, c)).rename(c) for c in cols], axis=1)


def vectorised_classification(df):
    """Exact broad-group logic used by the channel race audit."""
    eth = code_matrix(df, "ethnicity")
    race = code_matrix(df, "race")

    is_hispanic = eth.isin(HISPANIC_CODES).any(axis=1)
    is_non_hispanic = eth.eq(NON_HISPANIC_CODE).any(axis=1) & ~is_hispanic

    fam = pd.DataFrame(False, index=df.index, columns=RACE_FAMILY_ORDER)
    for code, family in RACE_FAMILY_BY_CODE.items():
        fam[family] |= race.eq(code).any(axis=1)
    family_n = fam.sum(axis=1)

    broad = pd.Series("Unknown/Not reported", index=df.index, dtype="string")
    broad.loc[is_hispanic] = "Hispanic/Latino"
    nh = is_non_hispanic
    broad.loc[nh & family_n.eq(1) & fam["White"]] = "Non-Hispanic White"
    broad.loc[nh & family_n.eq(1) & fam["Black"]] = "Non-Hispanic Black"
    broad.loc[nh & family_n.eq(1) & fam["Asian"]] = "Non-Hispanic Asian"
    broad.loc[nh & family_n.eq(1) & fam["AIAN"]] = "Non-Hispanic AIAN"
    broad.loc[nh & family_n.eq(1) & fam["NHPI"]] = "Non-Hispanic NHPI"
    broad.loc[nh & family_n.gt(1)] = "Non-Hispanic Multiracial"
    return broad


def row_codes(row, kind):
    vals = []
    for i in range(1, 6):
        value = row.get(f"applicant_{kind}_{i}", pd.NA)
        try:
            if pd.isna(value):
                continue
            val = int(float(str(value).strip()))
            vals.append(val)
        except Exception:
            continue
    return vals


def independent_expected(eth_codes, race_codes):
    """
    Independent row-level expected classification.

    This intentionally does not call the vectorised classifier.
    """
    if any(c in HISPANIC_CODES for c in eth_codes):
        return "Hispanic/Latino"

    if NON_HISPANIC_CODE not in eth_codes:
        return "Unknown/Not reported"

    families = []
    for code in race_codes:
        family = RACE_FAMILY_BY_CODE.get(code)
        if family is not None and family not in families:
            families.append(family)

    if len(families) == 0:
        return "Unknown/Not reported"
    if len(families) > 1:
        return "Non-Hispanic Multiracial"

    mapping = {
        "White": "Non-Hispanic White",
        "Black": "Non-Hispanic Black",
        "Asian": "Non-Hispanic Asian",
        "AIAN": "Non-Hispanic AIAN",
        "NHPI": "Non-Hispanic NHPI",
    }
    return mapping[families[0]]


def identify_edge_cases(eth_codes, race_codes):
    """A record may satisfy more than one targeted edge case."""
    cases = []
    race_set = set(race_codes)
    families = {RACE_FAMILY_BY_CODE[c] for c in race_codes if c in RACE_FAMILY_BY_CODE}
    is_hispanic = any(c in HISPANIC_CODES for c in eth_codes)
    is_non_hispanic = NON_HISPANIC_CODE in eth_codes and not is_hispanic

    if is_non_hispanic:
        if 2 in race_set and bool(race_set & ASIAN_DETAIL) and families == {"Asian"}:
            cases.append("Asian aggregate + detailed Asian subtype")

        if len(race_set & ASIAN_DETAIL) >= 2 and families == {"Asian"}:
            cases.append("Multiple detailed Asian subtypes only")

        if 4 in race_set and bool(race_set & NHPI_DETAIL) and families == {"NHPI"}:
            cases.append("NHPI aggregate + detailed NHPI subtype")

        if "White" in families and "Asian" in families:
            cases.append("White + Asian family")

        if "Black" in families and "White" in families:
            cases.append("Black + White")

        if "AIAN" in families and "White" in families:
            cases.append("AIAN + White")

        if "Asian" in families and "NHPI" in families:
            cases.append("Asian + NHPI")

    if is_hispanic:
        if len(families) == 1:
            cases.append("Hispanic + one race family")
        elif len(families) >= 2:
            cases.append("Hispanic + multiple race families")

    return cases


def fmt_codes(codes):
    return " | ".join(str(x) for x in codes) if codes else ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_files = [str(RAW_DIR / f"hmda_{y}.zip") for y in YEARS if not (RAW_DIR / f"hmda_{y}.zip").exists()]
    if missing_files:
        raise FileNotFoundError(
            "Expected annual HMDA ZIP(s) not found:\n" + "\n".join(missing_files)
        )

    collected = Counter()
    examples = []
    scanned_rows = 0
    eligible_rows = 0

    print("=" * 96)
    print("MULTIRACE RAW-RECORD SPOT CHECK")
    print("=" * 96)
    print(f"Target examples per edge case: {TARGET_PER_CASE}")
    print(f"Raw folder: {RAW_DIR}\n")

    done = False

    for year in YEARS:
        if done:
            break

        path = RAW_DIR / f"hmda_{year}.zip"
        print("-" * 96)
        print(f"YEAR {year}: {path.name}")

        for chunk_no, chunk in enumerate(read_chunks(path), start=1):
            scanned_rows += len(chunk)
            sample = apply_study_filters(chunk)
            eligible_rows += len(sample)
            if sample.empty:
                continue

            script_class = vectorised_classification(sample)

            # Restrict row-wise work to records with >1 entered race code OR Hispanic cases.
            race_mat = code_matrix(sample, "race")
            eth_mat = code_matrix(sample, "ethnicity")
            potentially_relevant = (
                race_mat.notna().sum(axis=1).ge(2)
                | eth_mat.isin(HISPANIC_CODES).any(axis=1)
            )

            candidate_idx = sample.index[potentially_relevant]
            for idx in candidate_idx:
                row = sample.loc[idx]
                eth_codes = row_codes(row, "ethnicity")
                race_codes = row_codes(row, "race")
                cases = identify_edge_cases(eth_codes, race_codes)
                if not cases:
                    continue

                expected = independent_expected(eth_codes, race_codes)
                actual = str(script_class.loc[idx])

                for case in cases:
                    if collected[case] >= TARGET_PER_CASE:
                        continue

                    examples.append({
                        "edge_case": case,
                        "year": int(float(str(row.get("activity_year", year)))) if pd.notna(row.get("activity_year", pd.NA)) else year,
                        "action_taken": row.get("action_taken", pd.NA),
                        "submission_of_application": row.get("submission_of_application", pd.NA),
                        "applicant_ethnicity_codes": fmt_codes(eth_codes),
                        "applicant_race_codes": fmt_codes(race_codes),
                        "aggregate_race_families": " + ".join(
                            sorted({RACE_FAMILY_BY_CODE[c] for c in race_codes if c in RACE_FAMILY_BY_CODE})
                        ),
                        "independent_expected": expected,
                        "script_classification": actual,
                        "PASS_expected_equals_script": expected == actual,
                    })
                    collected[case] += 1

                if all(collected[c] >= TARGET_PER_CASE for c in TARGET_CASES):
                    done = True
                    break

            if chunk_no % 10 == 0 or done:
                filled = sum(min(collected[c], TARGET_PER_CASE) for c in TARGET_CASES)
                target_total = TARGET_PER_CASE * len(TARGET_CASES)
                print(
                    f"  chunks {chunk_no:>3} | raw scanned {scanned_rows:,} | "
                    f"eligible {eligible_rows:,} | examples {filled}/{target_total}"
                )

            if done:
                break

    examples_df = pd.DataFrame(examples)

    summary_rows = []
    for case in TARGET_CASES:
        d = examples_df.loc[examples_df["edge_case"].eq(case)] if not examples_df.empty else pd.DataFrame()
        summary_rows.append({
            "edge_case": case,
            "target_n": TARGET_PER_CASE,
            "examples_found": len(d),
            "all_pass": bool(d["PASS_expected_equals_script"].all()) if len(d) else False,
            "status": (
                "PASS"
                if len(d) >= TARGET_PER_CASE and bool(d["PASS_expected_equals_script"].all())
                else "CHECK"
            ),
        })
    summary = pd.DataFrame(summary_rows)

    if not examples_df.empty:
        examples_df.to_csv(OUTPUT_DIR / "multirace_raw_record_examples.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "edge_case", "year", "action_taken", "submission_of_application",
            "applicant_ethnicity_codes", "applicant_race_codes",
            "aggregate_race_families", "independent_expected",
            "script_classification", "PASS_expected_equals_script"
        ]).to_csv(OUTPUT_DIR / "multirace_raw_record_examples.csv", index=False)

    summary.to_csv(OUTPUT_DIR / "multirace_spotcheck_summary.csv", index=False)

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            examples_df.to_excel(writer, sheet_name="raw_examples", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
        excel_msg = str(OUTPUT_XLSX)
    except Exception as exc:
        excel_msg = f"Excel workbook not written ({exc}); CSV files were still written."

    print("\n" + "=" * 96)
    print("SPOT CHECK COMPLETE")
    print("=" * 96)
    print(summary.to_string(index=False))

    failed = summary.loc[summary["status"].ne("PASS")]
    if failed.empty:
        print("\nPASS: every targeted edge case was found and independently matched the script classification.")
    else:
        print(
            "\nCHECK REQUIRED: at least one edge case was not found enough times "
            "or produced a classification mismatch. Do not change coding automatically; "
            "inspect the exported examples first."
        )

    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Workbook: {excel_msg}")
    print("\nSEND BACK:")
    print("  1. multirace_spotcheck_summary.csv")
    print("  2. multirace_raw_record_examples.csv")


if __name__ == "__main__":
    main()
