import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('cross_table.csv')
df['Count'] = df['Count'].astype(str).str.replace(',', '').astype(int)

HAZARDOUS_CUSTOMER = {'Galvanized'}
UNKNOWN_CUSTOMER   = 'Unknown - Material Unknown'

# ── Step 1a: Split into known vs unknown customer-side ────────────────────────
known   = df[df['Customer Material'] != UNKNOWN_CUSTOMER].copy()
unknown = df[df['Customer Material'] == UNKNOWN_CUSTOMER].copy()

print("=== Raw Totals ===")
print(f"Total lines (all rows):            {df['Count'].sum():>10,}")
print(f"Lines with known customer material: {known['Count'].sum():>10,}")
print(f"Lines with unknown customer side:   {unknown['Count'].sum():>10,}")

# ── Step 1b: Compute P(hazardous | utility material) from known rows ──────────
known['is_hazardous'] = known['Customer Material'].isin(HAZARDOUS_CUSTOMER)

# For each utility material: hazardous count and total known count
hazard_rates = (
    known
    .groupby('Utility Material')
    .apply(lambda g: pd.Series({
        'known_total':     g['Count'].sum(),
        'known_hazardous': g.loc[g['is_hazardous'], 'Count'].sum(),
    }))
    .assign(hazard_rate=lambda x: x['known_hazardous'] / x['known_total'])
)

print("\n=== Hazard Rates by Utility Material (from known customer rows) ===")
print(hazard_rates.to_string())

# ── Step 1c: Apply rates to unknown-customer rows ─────────────────────────────
unknown = unknown.merge(
    hazard_rates[['hazard_rate']],
    left_on='Utility Material',
    right_index=True,
    how='left'
)

# If a utility material has NO known-customer rows at all, fall back to
# the overall base rate across all known lines
base_rate = (
    known.loc[known['is_hazardous'], 'Count'].sum() /
    known['Count'].sum()
)
print(f"\nOverall base hazard rate (fallback): {base_rate:.4f}")

unknown['hazard_rate'] = unknown['hazard_rate'].fillna(base_rate)
unknown['estimated_hazardous'] = unknown['Count'] * unknown['hazard_rate']

print("\n=== Unknown Rows with Estimated Hazardous Count ===")
print(unknown[['Utility Material','Count','hazard_rate','estimated_hazardous']].to_string())

# ── Step 1d: Known hazardous count (customer side is directly Galvanized) ─────
known_hazardous_count = known.loc[known['is_hazardous'], 'Count'].sum()
estimated_hazardous_from_unknowns = unknown['estimated_hazardous'].sum()

N_total = known_hazardous_count + estimated_hazardous_from_unknowns

print("\n=== UNIVERSE ESTIMATE ===")
print(f"Known hazardous lines (customer = Galvanized):    {known_hazardous_count:>10,.0f}")
print(f"Estimated hazardous from unknown customer lines:  {estimated_hazardous_from_unknowns:>10,.0f}")
print(f"──────────────────────────────────────────────────────────")
print(f"N_total (program-eligible universe):              {N_total:>10,.0f}")
print(f"\nBase rate assumption used for any fallback:        {base_rate:.2%}")

# ── Step 2: Participation Rate ─────────────────────────────────────────────────

# Pilot funnel data
pilot_interested        = 127
pilot_apps_sent         = 90
pilot_applied           = 82
pilot_dropped           = 18
pilot_rejected          = 2
pilot_completed         = 42
pilot_in_progress       = 31

# Base case: completed / applied
# Rationale: under a federal mandate, engagement (applying) is the realistic
# top of funnel. Post-application attrition is the true behavioral friction.
participation_rate = pilot_completed / pilot_applied

# Participating population
N_participating = N_total * participation_rate

print("=== Step 2: Participation Rate ===\n")
print("Pilot Funnel:")
print(f"  Interested:              {pilot_interested}")
print(f"  Applications sent:       {pilot_apps_sent}")
print(f"  Applied:                 {pilot_applied}")
print(f"  Dropped:                 {pilot_dropped}")
print(f"  Rejected:                {pilot_rejected}")
print(f"  Completed loans:         {pilot_completed}")
print(f"  In-progress:             {pilot_in_progress}")

print(f"\nBase case participation rate (completed / applied): "
      f"{participation_rate:.4f} ({participation_rate:.2%})")
print(f"\nEligible universe (N_total):                {N_total:>10,.0f}")
print(f"Participating population (N_participating): {N_participating:>10,.0f}")

# ── Sensitivity table ─────────────────────────────────────────────────────────
print("\n=== Sensitivity: Participation Rate Under Alternative Definitions ===")

alternatives = {
    "Completed / Interested          (floor)": 
        (pilot_completed, pilot_interested),
    "Completed / Applied          (base case)": 
        (pilot_completed, pilot_applied),
    "(Completed + In-Progress) / Applied (optimistic)": 
        (pilot_completed + pilot_in_progress, pilot_applied),
}

for label, (num, denom) in alternatives.items():
    rate = num / denom
    pop  = N_total * rate
    print(f"  {label}:  {rate:.2%}  →  {pop:,.0f} properties")

    # ── Step 3: Replacement Schedule ──────────────────────────────────────────────

# Annual replacement percentages (of N_participating)
# 2026-2028: conservative ramp-up (Option A)
# 2029-2037: 10% per year per federal mandate
replacement_schedule = {
    2026: 0.02,
    2027: 0.04,
    2028: 0.07,
    2029: 0.10,
    2030: 0.10,
    2031: 0.10,
    2032: 0.10,
    2033: 0.10,
    2034: 0.10,
    2035: 0.10,
    2036: 0.10,
    2037: 0.10,
}

# Build the schedule table
# Cap cumulative loans at N_participating to avoid overshooting
schedule_rows = []
cumulative    = 0

for year, pct in replacement_schedule.items():
    raw_loans       = N_participating * pct
    # In the final year, only issue as many loans as remain
    loans_issued    = min(raw_loans, N_participating - cumulative)
    cumulative     += loans_issued
    cumulative_pct  = cumulative / N_participating

    schedule_rows.append({
        'Year':            year,
        'Annual %':        pct,
        'Loans Issued':    round(loans_issued),
        'Cumulative Loans': round(cumulative),
        'Cumulative %':    cumulative_pct,
    })

schedule_df = pd.DataFrame(schedule_rows)

print("=== Step 3: Replacement Schedule ===\n")
print(schedule_df.to_string(index=False, 
      formatters={
          'Annual %':       '{:.0%}'.format,
          'Loans Issued':   '{:,.0f}'.format,
          'Cumulative Loans': '{:,.0f}'.format,
          'Cumulative %':   '{:.1%}'.format,
      }))

print(f"\nTotal loans issued by 2037: {schedule_df['Loans Issued'].sum():,.0f}")
print(f"N_participating:            {N_participating:,.0f}")
print(f"Difference (rounding):      "
      f"{N_participating - schedule_df['Loans Issued'].sum():,.0f}")

# ── Step 4: Annual Loan Outflows ──────────────────────────────────────────────

AVG_LOAN_BASE   = 7582       # pilot average accepted quote ($)
LOAN_CAP        = 10_000     # program maximum ($)
COST_ESCALATION = 0.03       # annual construction cost inflation
BASE_YEAR       = 2026       # escalation reference year

# Add outflow columns to schedule_df
schedule_df['Avg Loan Amount']  = schedule_df['Year'].apply(
    lambda t: AVG_LOAN_BASE * (1 + COST_ESCALATION) ** (t - BASE_YEAR)
)

schedule_df['Avg Loan Amount'] = schedule_df['Avg Loan Amount'].clip(upper=LOAN_CAP)

schedule_df['Annual Outflow ($)'] = (
    schedule_df['Loans Issued'] * schedule_df['Avg Loan Amount']
)

schedule_df['Cumulative Outflow ($)'] = (
    schedule_df['Annual Outflow ($)'].cumsum()
)

print("=== Step 4: Annual Loan Outflows ===\n")
print(schedule_df[[
    'Year',
    'Loans Issued',
    'Avg Loan Amount',
    'Annual Outflow ($)',
    'Cumulative Outflow ($)',
]].to_string(index=False,
    formatters={
        'Loans Issued':           '{:,.0f}'.format,
        'Avg Loan Amount':        '${:,.2f}'.format,
        'Annual Outflow ($)':     '${:,.0f}'.format,
        'Cumulative Outflow ($)': '${:,.0f}'.format,
    }
))

total_outflow = schedule_df['Annual Outflow ($)'].sum()
avg_outflow   = schedule_df['Annual Outflow ($)'].mean()
peak_outflow  = schedule_df['Annual Outflow ($)'].max()
peak_year     = schedule_df.loc[
    schedule_df['Annual Outflow ($)'].idxmax(), 'Year'
]

print(f"\n--- Summary ---")
print(f"Total program outflow (2026-2037):  ${total_outflow:>15,.0f}")
print(f"Average annual outflow:             ${avg_outflow:>15,.0f}")
print(f"Peak annual outflow:                ${peak_outflow:>15,.0f} (in {peak_year})")
print(f"Avg loan amount in 2026:            ${AVG_LOAN_BASE:>15,.0f}")
print(f"Avg loan amount in 2037:            "
      f"${schedule_df.loc[schedule_df['Year'] == 2037, 'Avg Loan Amount'].values[0]:>15,.2f}")
print(f"\nNote: {LOAN_CAP:,} cap applies per loan. "
      f"Avg loan is below cap in 2026 but may approach it by "
      f"{BASE_YEAR + round((LOAN_CAP/AVG_LOAN_BASE - 1)/COST_ESCALATION):.0f}.")

# ── Cap breach check ──────────────────────────────────────────────────────────
cap_breach_year = None
for _, row in schedule_df.iterrows():
    if row['Avg Loan Amount'] >= LOAN_CAP:
        cap_breach_year = row['Year']
        break

if cap_breach_year:
    print(f"WARNING: Average loan amount exceeds ${LOAN_CAP:,} cap "
          f"in {cap_breach_year}.")
else:
    print(f"Average loan amount stays below ${LOAN_CAP:,} cap "
          f"through 2037 for all years.")
    
    # ── Step 5: Repayment Dynamics ─────────────────────────────────────────────────

REPAYMENT_LAG_BASE      = 12   # base case: avg years until repayment
REPAYMENT_LAG_OPTIMISTIC = 7   # optimistic: faster housing turnover
PROJECTION_END          = 2055 # extend well beyond 2037 to show repayment story

def build_repayment_schedule(schedule_df, lag, end_year):
    """
    For each loan cohort issued in year t, all loans repay in year t + lag.
    Returns a DataFrame indexed by year with repayment inflow columns.
    """
    # Full year range: program start through projection end
    years = list(range(2026, end_year + 1))
    repayment_df = pd.DataFrame({'Year': years})
    repayment_df = repayment_df.set_index('Year')
    repayment_df['Repayment Inflow ($)'] = 0.0

    for _, row in schedule_df.iterrows():
        issue_year   = row['Year']
        loans        = row['Loans Issued']
        avg_loan     = row['Avg Loan Amount']
        repay_year   = issue_year + lag
        repay_amount = loans * avg_loan  # repaid at original loan face value

        if repay_year <= end_year:
            repayment_df.loc[repay_year, 'Repayment Inflow ($)'] += repay_amount

    repayment_df = repayment_df.reset_index()
    repayment_df['Cumulative Inflow ($)'] = (
        repayment_df['Repayment Inflow ($)'].cumsum()
    )
    return repayment_df

# Build both scenarios
repay_base       = build_repayment_schedule(
    schedule_df, REPAYMENT_LAG_BASE, PROJECTION_END
)
repay_optimistic = build_repayment_schedule(
    schedule_df, REPAYMENT_LAG_OPTIMISTIC, PROJECTION_END
)

# ── Print base case ────────────────────────────────────────────────────────────
print("=== Step 5: Repayment Schedule (Base Case: L = 12 years) ===\n")

# Only show years with non-zero inflows or within program window
display_base = repay_base[
    (repay_base['Year'] <= 2037) |
    (repay_base['Repayment Inflow ($)'] > 0)
].copy()

print(display_base.to_string(index=False,
    formatters={
        'Repayment Inflow ($)':  '${:,.0f}'.format,
        'Cumulative Inflow ($)': '${:,.0f}'.format,
    }
))

total_repaid_base = repay_base['Repayment Inflow ($)'].sum()
repaid_by_2037_base = repay_base.loc[
    repay_base['Year'] <= 2037, 'Repayment Inflow ($)'
].sum()

print(f"\n--- Base Case Summary (L = {REPAYMENT_LAG_BASE} years) ---")
print(f"Total repayments received by 2037:       ${repaid_by_2037_base:>15,.0f}")
print(f"Total repayments received by {PROJECTION_END}:       "
      f"${total_repaid_base:>15,.0f}")
print(f"Total program outflow:                   "
      f"${total_outflow:>15,.0f}")
print(f"Net program cost (outflow - inflows):    "
      f"${total_outflow - total_repaid_base:>15,.0f}")

# ── Print optimistic case ──────────────────────────────────────────────────────
print(f"\n=== Step 5: Repayment Schedule (Optimistic: L = {REPAYMENT_LAG_OPTIMISTIC} years) ===\n")

display_optimistic = repay_optimistic[
    (repay_optimistic['Year'] <= 2037) |
    (repay_optimistic['Repayment Inflow ($)'] > 0)
].copy()

print(display_optimistic.to_string(index=False,
    formatters={
        'Repayment Inflow ($)':  '${:,.0f}'.format,
        'Cumulative Inflow ($)': '${:,.0f}'.format,
    }
))

total_repaid_opt = repay_optimistic['Repayment Inflow ($)'].sum()
repaid_by_2037_opt = repay_optimistic.loc[
    repay_optimistic['Year'] <= 2037, 'Repayment Inflow ($)'
].sum()

print(f"\n--- Optimistic Case Summary (L = {REPAYMENT_LAG_OPTIMISTIC} years) ---")
print(f"Total repayments received by 2037:       ${repaid_by_2037_opt:>15,.0f}")
print(f"Total repayments received by {PROJECTION_END}:       "
      f"${total_repaid_opt:>15,.0f}")
print(f"Total program outflow:                   "
      f"${total_outflow:>15,.0f}")
print(f"Net program cost (outflow - inflows):    "
      f"${total_outflow - total_repaid_opt:>15,.0f}")

# ── Key insight: what fraction of outflows are recovered within program window ─
print(f"\n=== Key Policy Insight ===")
print(f"Base case:       {repaid_by_2037_base/total_outflow:.1%} of outflows "
      f"recovered within program window (by 2037)")
print(f"Optimistic case: {repaid_by_2037_opt/total_outflow:.1%} of outflows "
      f"recovered within program window (by 2037)")
print(f"\nConclusion: The program behaves like a grant in the near term regardless")
print(f"of repayment lag. Near-grant-level capital is required upfront.")

# ── Step 6: Cash Flow Model and Funding Gap ────────────────────────────────────

PROJECTION_END = 2055

def build_cash_flow(schedule_df, repay_df, label):
    """
    Computes the cumulative cash balance over time.
    Starting capital is set equal to the peak funding gap (trough of balance).
    """
    years = list(range(2026, PROJECTION_END + 1))

    # Build outflow series — zero after 2037
    outflow_by_year = dict(zip(
        schedule_df['Year'],
        schedule_df['Annual Outflow ($)']
    ))

    # Build inflow series from repayment schedule
    inflow_by_year = dict(zip(
        repay_df['Year'],
        repay_df['Repayment Inflow ($)']
    ))

    rows = []
    cumulative_outflow = 0
    cumulative_inflow  = 0

    for year in years:
        outflow = outflow_by_year.get(year, 0)
        inflow  = inflow_by_year.get(year, 0)

        cumulative_outflow += outflow
        cumulative_inflow  += inflow

        # Balance before adding starting capital
        # Balance(t) = Starting Capital - cumulative outflows + cumulative inflows
        # We solve for starting capital later, so track net position first
        net_position = cumulative_inflow - cumulative_outflow

        rows.append({
            'Year':                year,
            'Annual Outflow ($)':  outflow,
            'Annual Inflow ($)':   inflow,
            'Net Position ($)':    net_position,
            'Cumulative Outflow ($)': cumulative_outflow,
            'Cumulative Inflow ($)':  cumulative_inflow,
        })

    cf_df = pd.DataFrame(rows)

    # Peak funding gap = absolute value of the minimum net position
    # This is the starting capital required to keep balance >= 0 at all times
    peak_funding_gap   = abs(cf_df['Net Position ($)'].min())
    trough_year        = cf_df.loc[
        cf_df['Net Position ($)'].idxmin(), 'Year'
    ]

    # Now compute actual cash balance with starting capital = peak funding gap
    cf_df['Cash Balance ($)'] = cf_df['Net Position ($)'] + peak_funding_gap

    print(f"\n=== Step 6: Cash Flow Model ({label}) ===\n")
    print(cf_df[[
        'Year',
        'Annual Outflow ($)',
        'Annual Inflow ($)',
        'Cash Balance ($)',
    ]].to_string(index=False,
        formatters={
            'Annual Outflow ($)': '${:,.0f}'.format,
            'Annual Inflow ($)':  '${:,.0f}'.format,
            'Cash Balance ($)':   '${:,.0f}'.format,
        }
    ))

    print(f"\n--- Funding Gap Summary ({label}) ---")
    print(f"Required starting capital:   ${peak_funding_gap:>15,.0f}")
    print(f"Trough year:                 {trough_year}")
    print(f"Trough balance (after cap):  ${cf_df['Cash Balance ($)'].min():>15,.0f}")
    print(f"Final balance in {PROJECTION_END}:        "
          f"${cf_df.loc[cf_df['Year']==PROJECTION_END, 'Cash Balance ($)'].values[0]:>15,.0f}")

    return cf_df, peak_funding_gap, trough_year

# ── Run both scenarios ─────────────────────────────────────────────────────────
cf_base, gap_base, trough_base = build_cash_flow(
    schedule_df, repay_base, 'Base Case: L=12 years'
)

cf_opt, gap_opt, trough_opt = build_cash_flow(
    schedule_df, repay_optimistic, 'Optimistic: L=7 years'
)

# ── Comparison summary ─────────────────────────────────────────────────────────
print("\n=== Step 6: Scenario Comparison ===")
print(f"{'Metric':<40} {'Base (L=12)':>15} {'Optimistic (L=7)':>18}")
print("-" * 75)
print(f"{'Required starting capital':<40} "
      f"${gap_base:>14,.0f} ${gap_opt:>17,.0f}")
print(f"{'Trough year':<40} "
      f"{trough_base:>15} {trough_opt:>18}")
print(f"{'Total program outflow':<40} "
      f"${total_outflow:>14,.0f} ${total_outflow:>17,.0f}")
print(f"{'Grant equivalent cost':<40} "
      f"${total_outflow:>14,.0f} ${total_outflow:>17,.0f}")
print(f"{'Loan program net cost (long run)':<40} "
      f"{'$0':>15} {'$0':>18}")

      # ── Step 7: Grant vs Loan Comparison ──────────────────────────────────────────

DISCOUNT_RATE = 0.035   # municipal borrowing rate

# ── 7a: Grant equivalent ──────────────────────────────────────────────────────
# Grant equivalent = total outflows (no repayment expected)
grant_equivalent = total_outflow

print("=== Step 7: Grant vs Loan Comparison ===\n")
print(f"--- 7a: Grant Equivalent ---")
print(f"Total loans issued:              {schedule_df['Loans Issued'].sum():>10,.0f}")
print(f"Grant equivalent cost:           ${grant_equivalent:>14,.0f}")
print(f"  (= total program outflow, no repayment assumed)")

# ── 7b: Implicit financing subsidy per loan ───────────────────────────────────
# For each cohort, compute the subsidy as:
# loan_amount * discount_rate * repayment_lag
# This is the simple approximation: PV of foregone interest
print(f"\n--- 7b: Implicit Financing Subsidy ---")
print(f"Municipal discount rate:         {DISCOUNT_RATE:.1%}")

def compute_subsidy(schedule_df, lag, discount_rate):
    """
    Computes the aggregate implicit financing subsidy across all loan cohorts.
    Subsidy per loan = loan_amount * discount_rate * lag
    This approximates the PV of interest foregone over the deferral period.
    """
    total_subsidy      = 0
    total_loans        = 0
    subsidy_rows       = []

    for _, row in schedule_df.iterrows():
        loans      = row['Loans Issued']
        avg_loan   = row['Avg Loan Amount']
        subsidy_per_loan = avg_loan * discount_rate * lag
        cohort_subsidy   = loans * subsidy_per_loan
        total_subsidy   += cohort_subsidy
        total_loans     += loans

        subsidy_rows.append({
            'Year':              row['Year'],
            'Loans':             loans,
            'Avg Loan ($)':      avg_loan,
            'Subsidy/Loan ($)':  subsidy_per_loan,
            'Cohort Subsidy ($)': cohort_subsidy,
        })

    subsidy_df = pd.DataFrame(subsidy_rows)
    avg_subsidy_per_loan = total_subsidy / total_loans
    subsidy_pct_of_face  = avg_subsidy_per_loan / (
        schedule_df['Avg Loan Amount'].mean()
    )

    return subsidy_df, total_subsidy, avg_subsidy_per_loan, subsidy_pct_of_face

# Base case subsidy (L=12)
sub_df_base, total_sub_base, avg_sub_base, pct_base = compute_subsidy(
    schedule_df, REPAYMENT_LAG_BASE, DISCOUNT_RATE
)

# Optimistic case subsidy (L=7)
sub_df_opt, total_sub_opt, avg_sub_opt, pct_opt = compute_subsidy(
    schedule_df, REPAYMENT_LAG_OPTIMISTIC, DISCOUNT_RATE
)

print(f"\nBase Case (L = {REPAYMENT_LAG_BASE} years):")
print(sub_df_base.to_string(index=False,
    formatters={
        'Loans':              '{:,.0f}'.format,
        'Avg Loan ($)':       '${:,.2f}'.format,
        'Subsidy/Loan ($)':   '${:,.2f}'.format,
        'Cohort Subsidy ($)': '${:,.0f}'.format,
    }
))
print(f"\n  Total implicit subsidy:          ${total_sub_base:>14,.0f}")
print(f"  Avg subsidy per loan:            ${avg_sub_base:>14,.2f}")
print(f"  Subsidy as % of avg face value:  {pct_base:>14.1%}")

print(f"\nOptimistic Case (L = {REPAYMENT_LAG_OPTIMISTIC} years):")
print(f"  Total implicit subsidy:          ${total_sub_opt:>14,.0f}")
print(f"  Avg subsidy per loan:            ${avg_sub_opt:>14,.2f}")
print(f"  Subsidy as % of avg face value:  {pct_opt:>14.1%}")

# ── 7c: Full comparison table ─────────────────────────────────────────────────
print(f"\n--- 7c: Full Model Comparison ---")
print(f"\n{'Metric':<45} {'Grant':>12} {'Loan (L=12)':>14} {'Loan (L=7)':>14}")
print("-" * 87)
print(f"{'Required starting capital':<45} "
      f"${grant_equivalent:>11,.0f} "
      f"${gap_base:>13,.0f} "
      f"${gap_opt:>13,.0f}")
print(f"{'Long-run net cost to municipality':<45} "
      f"${grant_equivalent:>11,.0f} "
      f"{'$0':>14} "
      f"{'$0':>14}")
print(f"{'Implicit financing subsidy':<45} "
      f"{'N/A':>12} "
      f"${total_sub_base:>13,.0f} "
      f"${total_sub_opt:>13,.0f}")
print(f"{'True economic cost (net + subsidy)':<45} "
      f"${grant_equivalent:>11,.0f} "
      f"${total_sub_base:>13,.0f} "
      f"${total_sub_opt:>13,.0f}")
print(f"{'Trough year':<45} "
      f"{'2037':>12} "
      f"{'2037':>14} "
      f"{'2036':>14}")
print(f"{'Admin complexity':<45} "
      f"{'Low':>12} "
      f"{'High':>14} "
      f"{'High':>14}")

# ── Step 8: Sensitivity Analysis ──────────────────────────────────────────────

DISCOUNT_RATE = 0.035

# Participation rate scenarios
participation_scenarios = {
    'Floor (33.07%)':      pilot_completed / pilot_interested,
    'Low (42.00%)':        0.42,
    'Base (51.22%)':       pilot_completed / pilot_applied,
    'High (70.00%)':       0.70,
    'Optimistic (89.02%)': (pilot_completed + pilot_in_progress) / pilot_applied,
}

# Repayment lag scenarios
lag_scenarios = {
    'L = 7 years':  7,
    'L = 12 years': 12,
    'L = 20 years': 20,
}

def run_full_model(participation_rate, lag, discount_rate=DISCOUNT_RATE):
    """
    Runs the full Phase 1 model for a given participation rate and lag.
    Returns required starting capital, total outflow, and implicit subsidy.
    """
    # ── Participating population ───────────────────────────────────────────────
    n_part = N_total * participation_rate

    # ── Replacement schedule ───────────────────────────────────────────────────
    replacement_pcts = {
        2026: 0.02, 2027: 0.04, 2028: 0.07,
        2029: 0.10, 2030: 0.10, 2031: 0.10,
        2032: 0.10, 2033: 0.10, 2034: 0.10,
        2035: 0.10, 2036: 0.10, 2037: 0.10,
    }

    sched_rows = []
    cumulative = 0
    for year, pct in replacement_pcts.items():
        loans_issued = min(n_part * pct, n_part - cumulative)
        cumulative  += loans_issued
        avg_loan     = min(
            AVG_LOAN_BASE * (1 + COST_ESCALATION) ** (year - BASE_YEAR),
            LOAN_CAP
        )
        annual_outflow = loans_issued * avg_loan
        sched_rows.append({
            'Year':               year,
            'Loans Issued':       loans_issued,
            'Avg Loan Amount':    avg_loan,
            'Annual Outflow ($)': annual_outflow,
        })

    sched = pd.DataFrame(sched_rows)
    total_out = sched['Annual Outflow ($)'].sum()

    # ── Repayment schedule ─────────────────────────────────────────────────────
    years = list(range(2026, PROJECTION_END + 1))
    inflow_by_year = {y: 0.0 for y in years}
    for _, row in sched.iterrows():
        repay_year = int(row['Year']) + lag
        if repay_year <= PROJECTION_END:
            inflow_by_year[repay_year] += row['Annual Outflow ($)']

    # ── Cash flow and funding gap ──────────────────────────────────────────────
    cum_out = 0
    cum_in  = 0
    net_positions = []
    for year in years:
        outflow = sched.loc[sched['Year'] == year, 'Annual Outflow ($)'].sum() \
                  if year <= 2037 else 0
        inflow  = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        net_positions.append(cum_in - cum_out)

    peak_funding_gap = abs(min(net_positions))

    # ── Implicit subsidy ───────────────────────────────────────────────────────
    implicit_subsidy = total_out * discount_rate * lag

    return peak_funding_gap, total_out, implicit_subsidy

# ── Build sensitivity tables ───────────────────────────────────────────────────

# Table 1: Required starting capital
print("=== Step 8: Sensitivity Analysis ===\n")
print("--- Table 1: Required Starting Capital ($) ---\n")

# Header
header = f"{'Scenario':<25}" + "".join(
    f"{lag_label:>20}" for lag_label in lag_scenarios.keys()
)
print(header)
print("-" * (25 + 20 * len(lag_scenarios)))

capital_results = {}
for part_label, part_rate in participation_scenarios.items():
    row_str = f"{part_label:<25}"
    capital_results[part_label] = {}
    for lag_label, lag in lag_scenarios.items():
        gap, out, sub = run_full_model(part_rate, lag)
        capital_results[part_label][lag_label] = (gap, out, sub)
        row_str += f"  ${gap:>15,.0f}"
    print(row_str)

# Table 2: Total program outflow
print("\n--- Table 2: Total Program Outflow ($) ---\n")
print(header)
print("-" * (25 + 20 * len(lag_scenarios)))
for part_label, part_rate in participation_scenarios.items():
    row_str = f"{part_label:<25}"
    for lag_label in lag_scenarios.keys():
        gap, out, sub = capital_results[part_label][lag_label]
        row_str += f"  ${out:>15,.0f}"
    print(row_str)

# Table 3: Implicit financing subsidy
print("\n--- Table 3: Implicit Financing Subsidy ($) ---\n")
print(header)
print("-" * (25 + 20 * len(lag_scenarios)))
for part_label, part_rate in participation_scenarios.items():
    row_str = f"{part_label:<25}"
    for lag_label in lag_scenarios.keys():
        gap, out, sub = capital_results[part_label][lag_label]
        row_str += f"  ${sub:>15,.0f}"
    print(row_str)

# Table 4: True economic cost
print("\n--- Table 4: True Economic Cost (Net Cost + Subsidy) ($) ---\n")
print(header)
print("-" * (25 + 20 * len(lag_scenarios)))
for part_label, part_rate in participation_scenarios.items():
    row_str = f"{part_label:<25}"
    for lag_label in lag_scenarios.keys():
        gap, out, sub = capital_results[part_label][lag_label]
        # Net cost = 0 (full repayment assumed), so true cost = subsidy
        row_str += f"  ${sub:>15,.0f}"
    print(row_str)

# ── Key statistics ─────────────────────────────────────────────────────────────
all_gaps = [
    capital_results[p][l][0]
    for p in participation_scenarios
    for l in lag_scenarios
]

print(f"\n--- Key Statistics ---")
print(f"Minimum required capital (floor scenario, L=7):    "
      f"${min(all_gaps):>15,.0f}")
print(f"Maximum required capital (optimistic, L=20):       "
      f"${max(all_gaps):>15,.0f}")
print(f"Base case required capital (51.22%, L=12):         "
      f"${capital_results['Base (51.22%)']['L = 12 years'][0]:>15,.0f}")
print(f"Range (max - min):                                 "
      f"${max(all_gaps) - min(all_gaps):>15,.0f}")
print(f"\nBase case as % of maximum:  "
      f"{capital_results['Base (51.22%)']['L = 12 years'][0]/max(all_gaps):.1%}")
print(f"Base case as % of minimum:  "
      f"{capital_results['Base (51.22%)']['L = 12 years'][0]/min(all_gaps):.1%}")
