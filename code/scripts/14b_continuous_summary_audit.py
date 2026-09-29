from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path.home() / "Desktop" / "HMDA_Dissertation"
DATA_DIR = PROJECT_DIR / "final_data" / "channel_analysis_v1_parts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "descriptive_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 9_200_321
YEARS = list(range(2018, 2025))

PREFERRED_COLUMNS = {
    "dual50": ["dual50"],
    "year": ["year", "activity_year"],
    "lender": ["lei_clean", "lei", "lender_lei"],
    "race": ["race_group", "race_ethnicity", "race_category", "race_ethnicity_group"],
    "channel": ["channel", "submission_channel", "submission_of_application", "non_direct"],
    "approval": ["approval", "approved", "approval_binary", "approved_binary"],
    "income": ["income_thousands", "income_clean", "income", "income_numeric"],
    "loan_amount": ["loan_amount_clean", "loan_amount"],
    "property_value": ["property_value_clean", "property_value"],
    "county": ["county_clean", "county_code", "county"],
}

def first_existing(columns, candidates):
    s = set(columns)
    for c in candidates:
        if c in s:
            return c
    return None

def clean_string(s):
    out = s.astype("string").str.strip()
    return out.replace({
        "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
        "<NA>": pd.NA, "NA": pd.NA, "N/A": pd.NA, "NULL": pd.NA,
    })

def truthy(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().any():
        return n.eq(1)
    return clean_string(s).str.lower().isin({"true","t","yes","y","1"})

def normalize_race(s):
    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)
    mapping = {
        "white":"White", "non-hispanic white":"White",
        "non hispanic white":"White", "nh white":"White",
        "black":"Black", "black or african american":"Black",
        "non-hispanic black":"Black", "non hispanic black":"Black",
        "nh black":"Black",
        "hispanic":"Hispanic", "hispanic/latino":"Hispanic",
        "hispanic / latino":"Hispanic", "hispanic or latino":"Hispanic",
        "latino":"Hispanic",
        "asian":"Asian", "non-hispanic asian":"Asian",
        "non hispanic asian":"Asian", "nh asian":"Asian",
    }
    return x.map(mapping).astype("string")

def normalize_channel(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")
    out.loc[n.eq(1)] = 0
    out.loc[n.eq(2)] = 1
    x = clean_string(s).str.lower().str.replace(r"\s+", " ", regex=True)
    out.loc[x.isin({"direct","submitted directly"})] = 0
    out.loc[x.isin({
        "non-direct","non direct","nondirect","not direct",
        "not submitted directly","intermediated"
    })] = 1
    return out

def normalize_approval(s):
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    m = n.isin([0,1])
    out.loc[m] = n.loc[m]
    x = clean_string(s).str.lower()
    out.loc[x.isin({"approved","approval","yes","true"})] = 1
    out.loc[x.isin({"denied","no","false"})] = 0
    return out

def normalize_county(s):
    x = clean_string(s).str.replace(r"\.0$", "", regex=True)
    m = x.str.fullmatch(r"\d{1,5}", na=False)
    x.loc[m] = x.loc[m].str.zfill(5)
    return x

parts = sorted(DATA_DIR.glob("*.parquet"))
if not parts:
    raise FileNotFoundError(DATA_DIR)

cols = list(pq.ParquetFile(parts[0]).schema_arrow.names)
schema = {k:first_existing(cols,v) for k,v in PREFERRED_COLUMNS.items()}
missing = [k for k,v in schema.items() if v is None]
if missing:
    raise KeyError(f"Missing fields: {missing}")

arrays = {
    "Total": {"income": [], "loan_amount": [], "property_value": []},
    "Direct": {"income": [], "loan_amount": [], "property_value": []},
    "Non-direct": {"income": [], "loan_amount": [], "property_value": []},
}

n_total = 0

readcols = list(dict.fromkeys(schema.values()))

for i, part in enumerate(parts, 1):
    d = pd.read_parquet(part, columns=readcols)

    race = normalize_race(d[schema["race"]])
    ch = normalize_channel(d[schema["channel"]])
    approval = normalize_approval(d[schema["approval"]])
    year = pd.to_numeric(d[schema["year"]], errors="coerce")
    income = pd.to_numeric(d[schema["income"]], errors="coerce")
    loan = pd.to_numeric(d[schema["loan_amount"]], errors="coerce")
    prop = pd.to_numeric(d[schema["property_value"]], errors="coerce")
    lender = clean_string(d[schema["lender"]])
    county = normalize_county(d[schema["county"]])

    valid_county = (
        county.str.fullmatch(r"\d{5}", na=False)
        & ~county.isin({"00000","88888","99999"})
    )

    keep = (
        truthy(d[schema["dual50"]])
        & race.notna()
        & ch.isin([0,1])
        & approval.isin([0,1])
        & year.isin(YEARS)
        & lender.notna()
        & income.notna() & np.isfinite(income)
        & loan.notna() & np.isfinite(loan) & loan.gt(0)
        & prop.notna() & np.isfinite(prop) & prop.gt(0)
        & valid_county
    )

    n_total += int(keep.sum())

    for label, mask in {
        "Total": keep,
        "Direct": keep & ch.eq(0),
        "Non-direct": keep & ch.eq(1),
    }.items():
        arrays[label]["income"].append(
            income.loc[mask].to_numpy(dtype=np.float64, copy=True)
        )
        arrays[label]["loan_amount"].append(
            loan.loc[mask].to_numpy(dtype=np.float64, copy=True)
        )
        arrays[label]["property_value"].append(
            prop.loc[mask].to_numpy(dtype=np.float64, copy=True)
        )

    if i % 50 == 0 or i == len(parts):
        print(f"parts {i:>3}/{len(parts)} | preferred N={n_total:,}")

    del d
    gc.collect()

if n_total != EXPECTED_N:
    raise RuntimeError(
        f"Preferred sample mismatch: {n_total:,} != {EXPECTED_N:,}"
    )

rows = []
qprobs = [0.01, 0.25, 0.50, 0.75, 0.99, 0.995, 0.999]

for group in ["Total","Direct","Non-direct"]:
    for var in ["income","loan_amount","property_value"]:
        x = np.concatenate(arrays[group][var])
        q = np.quantile(x, qprobs)

        row = {
            "group": group,
            "variable": var,
            "n": len(x),
            "mean": np.mean(x),
            "sd": np.std(x, ddof=1),
            "min": np.min(x),
            "p1": q[0],
            "p25": q[1],
            "median": q[2],
            "p75": q[3],
            "p99": q[4],
            "p99_5": q[5],
            "p99_9": q[6],
            "max": np.max(x),
        }
        rows.append(row)

        del x
        gc.collect()

summary = pd.DataFrame(rows)
summary.to_csv(
    OUTPUT_DIR / "continuous_distribution_audit.csv",
    index=False
)

# Income upper-tail thresholds are in $000s.
tail_rows = []
for group in ["Total","Direct","Non-direct"]:
    x = np.concatenate(arrays[group]["income"])
    for threshold, label in [
        (1000, "Income > $1m"),
        (5000, "Income > $5m"),
        (10000, "Income > $10m"),
        (100000, "Income > $100m"),
    ]:
        n = int(np.sum(x > threshold))
        tail_rows.append({
            "group": group,
            "threshold": label,
            "n": n,
            "percent": 100*n/len(x),
        })
    del x
    gc.collect()

tails = pd.DataFrame(tail_rows)
tails.to_csv(
    OUTPUT_DIR / "income_upper_tail_audit.csv",
    index=False
)

print("\nCONTINUOUS DISTRIBUTION AUDIT")
print(summary.to_string(index=False))

print("\nINCOME UPPER TAIL")
print(tails.to_string(index=False))

print("\nSEND BACK:")
print("  continuous_distribution_audit.csv")
print("  income_upper_tail_audit.csv")