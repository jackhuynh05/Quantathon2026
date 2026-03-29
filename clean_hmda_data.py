# clean_hmda.py
# ── Cleans raw HMDA data down to what Phase 4 needs ───────────────────────────
import pandas as pd
import numpy as np

RAW_PATH   = 'hmda_raw.csv'
CLEAN_PATH = 'hmda_clean.csv'

print("Loading raw HMDA data...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"  Raw shape: {df.shape}")

# ── Step 1: Keep only originated loans ────────────────────────────────────────
# action_taken == 1 means the loan was actually originated
# Everything else is applications that were denied, withdrawn, incomplete etc.
df = df[df['action_taken'] == 1].copy()
print(f"  After filtering to originated loans: {df.shape[0]:,} rows")

# ── Step 2: Keep only owner-occupied 1-4 family properties ───────────────────
# occupancy_type == 1 is principal residence (owner-occupied)
# We exclude investment properties and second homes
df = df[df['occupancy_type'] == 1].copy()
print(f"  After filtering to owner-occupied: {df.shape[0]:,} rows")

# ── Step 3: Keep only relevant loan purposes ──────────────────────────────────
# loan_purpose codes:
#   1  = Home purchase
#   2  = Home improvement
#   31 = Refinancing
#   32 = Cash-out refinancing
#   4  = Other purpose (includes HELOCs when open-end = 1)
#
# For Phase 4 we want:
#   - Refinancing (31, 32) → refi channel
#   - Open-end lines of credit → HELOC channel (loan_purpose=4, open_end=1)
#   - Home purchase (1) → sale channel calibration
df = df[df['loan_purpose'].isin([1, 2, 31, 32, 4])].copy()
print(f"  After filtering loan purposes: {df.shape[0]:,} rows")

# ── Step 4: Tag loan type ─────────────────────────────────────────────────────
# open-end_line_of_credit == 1 means HELOC
df['loan_category'] = 'other'
df.loc[df['loan_purpose'] == 1,  'loan_category'] = 'purchase'
df.loc[df['loan_purpose'].isin([31, 32]), 'loan_category'] = 'refinance'
df.loc[
    (df['loan_purpose'] == 4) &
    (df['open-end_line_of_credit'] == 1),
    'loan_category'
] = 'heloc'

print(f"\n  Loan category breakdown:")
print(df['loan_category'].value_counts().to_string())

# ── Step 5: Keep only the columns we actually need ────────────────────────────
keep_cols = [
    'activity_year',
    'census_tract',
    'loan_purpose',
    'loan_category',
    'open-end_line_of_credit',
    'loan_amount',
    'interest_rate',
    'loan_term',
    'property_value',
    'occupancy_type',
    'derived_dwelling_category',
    'total_units',
    'tract_owner_occupied_units',
    'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units',
    'tract_to_msa_income_percentage',
    'applicant_age',
    'applicant_age_above_62',
]

# Only keep columns that exist in this file
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()
print(f"\n  Columns kept: {len(keep_cols)}")

# ── Step 6: Clean numeric columns ─────────────────────────────────────────────
# HMDA uses 'NA', 'Exempt', and other strings for missing values
numeric_cols = [
    'loan_amount', 'interest_rate', 'loan_term',
    'property_value', 'total_units',
    'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units', 'tract_to_msa_income_percentage',
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Step 7: Filter to Columbus-area census tracts ─────────────────────────────
# Columbus MSA census tracts start with 39049 (Franklin County)
# or nearby counties: 39041 (Delaware), 39045 (Fairfield),
# 39089 (Licking), 39097 (Madison), 39127 (Pickaway), 39159 (Union)
columbus_county_fips = [
    '39049',  # Franklin (Columbus proper)
    '39041',  # Delaware
    '39045',  # Fairfield
    '39089',  # Licking
    '39097',  # Madison
    '39127',  # Pickaway
    '39159',  # Union
]

# census_tract is an 11-digit FIPS code: state(2) + county(3) + tract(6)
# Convert to string and check the first 5 digits
df['census_tract'] = df['census_tract'].astype(str).str.zfill(11)
df['county_fips']  = df['census_tract'].str[:5]
df = df[df['county_fips'].isin(columbus_county_fips)].copy()
print(f"\n  After filtering to Columbus MSA: {df.shape[0]:,} rows")

# ── Step 8: Summary statistics for Phase 4 calibration ───────────────────────
print("\n=== Phase 4 Calibration Statistics ===\n")

# Annual rates by loan category
print("Loan counts by category:")
print(df['loan_category'].value_counts().to_string())

# Total owner-occupied housing stock (for rate denominator)
# Use tract_owner_occupied_units — take mean across unique tracts to avoid double-counting
unique_tracts = df.drop_duplicates('census_tract')[
    ['census_tract', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes']
]
total_owner_occ = unique_tracts['tract_owner_occupied_units'].sum()
total_1to4      = unique_tracts['tract_one_to_four_family_homes'].sum()
print(f"\nEstimated owner-occupied units in Columbus MSA: {total_owner_occ:,.0f}")
print(f"Estimated 1-4 family homes in Columbus MSA:     {total_1to4:,.0f}")

# Annual rates
n_refi  = (df['loan_category'] == 'refinance').sum()
n_heloc = (df['loan_category'] == 'heloc').sum()
n_purch = (df['loan_category'] == 'purchase').sum()

if total_owner_occ > 0:
    refi_rate  = n_refi  / total_owner_occ
    heloc_rate = n_heloc / total_owner_occ
    purch_rate = n_purch / total_owner_occ
    print(f"\nAnnual rates (as % of owner-occupied stock):")
    print(f"  Refinancing rate:  {refi_rate:.4f} ({refi_rate:.2%})")
    print(f"  HELOC rate:        {heloc_rate:.4f} ({heloc_rate:.2%})")
    print(f"  Purchase rate:     {purch_rate:.4f} ({purch_rate:.2%})")

# Interest rate distribution for refi loans
refi_rates = df.loc[df['loan_category'] == 'refinance', 'interest_rate'].dropna()
if len(refi_rates) > 0:
    print(f"\nRefinance interest rate distribution:")
    print(f"  Mean:    {refi_rates.mean():.2f}%")
    print(f"  Median:  {refi_rates.median():.2f}%")
    print(f"  Std:     {refi_rates.std():.2f}%")
    print(f"  25th pct: {refi_rates.quantile(0.25):.2f}%")
    print(f"  75th pct: {refi_rates.quantile(0.75):.2f}%")

# Median age of housing stock (useful for Weibull calibration)
med_age = df['tract_median_age_of_housing_units'].dropna()
if len(med_age) > 0:
    print(f"\nMedian housing age in Columbus MSA:")
    print(f"  Mean across tracts: {med_age.mean():.1f} years")
    print(f"  Median:             {med_age.median():.1f} years")

# ── Step 9: Save clean file ───────────────────────────────────────────────────
df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved clean file: {CLEAN_PATH}")
print(f"Final shape: {df.shape}")
print(f"\nColumn list:")
print(df.columns.tolist())
