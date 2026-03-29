# phase1.py
# ── LEAP Program: Phase 1 Deterministic Model ─────────────────────────────────
import pandas as pd
from config import (
    CROSS_TABLE_PATH,
    EXCEL_PATH,
    AVG_LOAN_BASE,
    LOAN_CAP,
    COST_ESCALATION,
    BASE_YEAR,
    DISCOUNT_RATE,
    PROGRAM_START,
    PROGRAM_END,
    PROJECTION_END,
    REPAYMENT_LAG_BASE,
    REPAYMENT_LAG_OPTIMISTIC,
    PILOT_INTERESTED,
    PILOT_APPS_SENT,
    PILOT_APPLIED,
    PILOT_DROPPED,
    PILOT_REJECTED,
    PILOT_COMPLETED,
    PILOT_IN_PROGRESS,
    REPLACEMENT_SCHEDULE,
    PARTICIPATION_SCENARIOS,
    LAG_SCENARIOS,
)

VERBOSE = __name__ == '__main__'

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Universe Estimation
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(CROSS_TABLE_PATH)
df['Count'] = df['Count'].astype(str).str.replace(',', '').astype(int)

HAZARDOUS_CUSTOMER = {'Galvanized'}
UNKNOWN_CUSTOMER   = 'Unknown - Material Unknown'

# Split into known vs unknown customer-side
known   = df[df['Customer Material'] != UNKNOWN_CUSTOMER].copy()
unknown = df[df['Customer Material'] == UNKNOWN_CUSTOMER].copy()

if VERBOSE:
    print("=== Raw Totals ===")
    print(f"Total lines (all rows):            {df['Count'].sum():>10,}")
    print(f"Lines with known customer material: {known['Count'].sum():>10,}")
    print(f"Lines with unknown customer side:   {unknown['Count'].sum():>10,}")

# Compute P(hazardous | utility material) from known rows
known['is_hazardous'] = known['Customer Material'].isin(HAZARDOUS_CUSTOMER)

hazard_rates = (
    known
    .groupby('Utility Material')
    .apply(lambda g: pd.Series({
        'known_total':     g['Count'].sum(),
        'known_hazardous': g.loc[g['is_hazardous'], 'Count'].sum(),
    }), include_groups=False)
    .assign(hazard_rate=lambda x: x['known_hazardous'] / x['known_total'])
)

if VERBOSE:
    print("\n=== Hazard Rates by Utility Material (from known customer rows) ===")
    print(hazard_rates.to_string())

# Apply rates to unknown-customer rows
unknown = unknown.merge(
    hazard_rates[['hazard_rate']],
    left_on='Utility Material',
    right_index=True,
    how='left'
)

base_rate = (
    known.loc[known['is_hazardous'], 'Count'].sum() /
    known['Count'].sum()
)

if VERBOSE:
    print(f"\nOverall base hazard rate (fallback): {base_rate:.4f}")

unknown['hazard_rate']         = unknown['hazard_rate'].fillna(base_rate)
unknown['estimated_hazardous'] = unknown['Count'] * unknown['hazard_rate']

if VERBOSE:
    print("\n=== Unknown Rows with Estimated Hazardous Count ===")
    print(unknown[['Utility Material', 'Count', 'hazard_rate', 'estimated_hazardous']].to_string())

# Universe estimate
known_hazardous_count             = known.loc[known['is_hazardous'], 'Count'].sum()
estimated_hazardous_from_unknowns = unknown['estimated_hazardous'].sum()
N_total                           = known_hazardous_count + estimated_hazardous_from_unknowns

if VERBOSE:
    print("\n=== UNIVERSE ESTIMATE ===")
    print(f"Known hazardous lines (customer = Galvanized):    {known_hazardous_count:>10,.0f}")
    print(f"Estimated hazardous from unknown customer lines:  {estimated_hazardous_from_unknowns:>10,.0f}")
    print(f"──────────────────────────────────────────────────────────")
    print(f"N_total (program-eligible universe):              {N_total:>10,.0f}")
    print(f"\nBase rate assumption used for any fallback:        {base_rate:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Participation Rate
# ══════════════════════════════════════════════════════════════════════════════

participation_rate = PILOT_COMPLETED / PILOT_APPLIED
N_participating    = N_total * participation_rate

if VERBOSE:
    print("\n=== Step 2: Participation Rate ===\n")
    print("Pilot Funnel:")
    print(f"  Interested:              {PILOT_INTERESTED}")
    print(f"  Applications sent:       {PILOT_APPS_SENT}")
    print(f"  Applied:                 {PILOT_APPLIED}")
    print(f"  Dropped:                 {PILOT_DROPPED}")
    print(f"  Rejected:                {PILOT_REJECTED}")
    print(f"  Completed loans:         {PILOT_COMPLETED}")
    print(f"  In-progress:             {PILOT_IN_PROGRESS}")
    print(f"\nBase case participation rate (completed / applied): "
          f"{participation_rate:.4f} ({participation_rate:.2%})")
    print(f"\nEligible universe (N_total):                {N_total:>10,.0f}")
    print(f"Participating population (N_participating): {N_participating:>10,.0f}")
    print("\n=== Sensitivity: Participation Rate Under Alternative Definitions ===")
    alternatives = {
        "Completed / Interested          (floor)":
            (PILOT_COMPLETED, PILOT_INTERESTED),
        "Completed / Applied          (base case)":
            (PILOT_COMPLETED, PILOT_APPLIED),
        "(Completed + In-Progress) / Applied (optimistic)":
            (PILOT_COMPLETED + PILOT_IN_PROGRESS, PILOT_APPLIED),
    }
    for label, (num, denom) in alternatives.items():
        rate = num / denom
        pop  = N_total * rate
        print(f"  {label}:  {rate:.2%}  →  {pop:,.0f} properties")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Replacement Schedule
# ══════════════════════════════════════════════════════════════════════════════

schedule_rows = []
cumulative    = 0

for year, pct in REPLACEMENT_SCHEDULE.items():
    loans_issued   = min(N_participating * pct, N_participating - cumulative)
    cumulative    += loans_issued
    cumulative_pct = cumulative / N_participating
    schedule_rows.append({
        'Year':             year,
        'Annual %':         pct,
        'Loans Issued':     round(loans_issued),
        'Cumulative Loans': round(cumulative),
        'Cumulative %':     cumulative_pct,
    })

schedule_df = pd.DataFrame(schedule_rows)

if VERBOSE:
    print("\n=== Step 3: Replacement Schedule ===\n")
    print(schedule_df.to_string(index=False,
          formatters={
              'Annual %':         '{:.0%}'.format,
              'Loans Issued':     '{:,.0f}'.format,
              'Cumulative Loans': '{:,.0f}'.format,
              'Cumulative %':     '{:.1%}'.format,
          }))
    print(f"\nTotal loans issued by 2037: {schedule_df['Loans Issued'].sum():,.0f}")
    print(f"N_participating:            {N_participating:,.0f}")
    print(f"Difference (rounding):      "
          f"{N_participating - schedule_df['Loans Issued'].sum():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Annual Loan Outflows
# ══════════════════════════════════════════════════════════════════════════════

schedule_df['Avg Loan Amount'] = schedule_df['Year'].apply(
    lambda t: AVG_LOAN_BASE * (1 + COST_ESCALATION) ** (t - BASE_YEAR)
).clip(upper=LOAN_CAP)

schedule_df['Annual Outflow ($)']     = (
    schedule_df['Loans Issued'] * schedule_df['Avg Loan Amount']
)
schedule_df['Cumulative Outflow ($)'] = schedule_df['Annual Outflow ($)'].cumsum()

total_outflow = schedule_df['Annual Outflow ($)'].sum()
avg_outflow   = schedule_df['Annual Outflow ($)'].mean()
peak_outflow  = schedule_df['Annual Outflow ($)'].max()
peak_year     = schedule_df.loc[schedule_df['Annual Outflow ($)'].idxmax(), 'Year']

if VERBOSE:
    print("\n=== Step 4: Annual Loan Outflows ===\n")
    print(schedule_df[[
        'Year', 'Loans Issued', 'Avg Loan Amount',
        'Annual Outflow ($)', 'Cumulative Outflow ($)',
    ]].to_string(index=False,
        formatters={
            'Loans Issued':           '{:,.0f}'.format,
            'Avg Loan Amount':        '${:,.2f}'.format,
            'Annual Outflow ($)':     '${:,.0f}'.format,
            'Cumulative Outflow ($)': '${:,.0f}'.format,
        }
    ))
    print(f"\n--- Summary ---")
    print(f"Total program outflow (2026-2037):  ${total_outflow:>15,.0f}")
    print(f"Average annual outflow:             ${avg_outflow:>15,.0f}")
    print(f"Peak annual outflow:                ${peak_outflow:>15,.0f} (in {peak_year})")
    print(f"Avg loan amount in 2026:            ${AVG_LOAN_BASE:>15,.0f}")
    print(f"Avg loan amount in 2037:            "
          f"${schedule_df.loc[schedule_df['Year'] == 2037, 'Avg Loan Amount'].values[0]:>15,.2f}")

    cap_breach_year = schedule_df.loc[
        schedule_df['Avg Loan Amount'] >= LOAN_CAP, 'Year'
    ].min()
    if pd.notna(cap_breach_year):
        print(f"\nNote: ${LOAN_CAP:,} cap becomes binding in {int(cap_breach_year)}. "
              f"Homeowners bear out-of-pocket costs above the cap from that year onward.")
    else:
        print(f"\nAverage loan amount stays below ${LOAN_CAP:,} cap through 2037.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Repayment Dynamics
# ══════════════════════════════════════════════════════════════════════════════

def build_repayment_schedule(schedule_df, lag, end_year=PROJECTION_END):
    """
    For each loan cohort issued in year t, all loans repay in year t + lag.
    Returns a DataFrame with Year, Repayment Inflow ($), Cumulative Inflow ($).
    """
    years        = list(range(PROGRAM_START, end_year + 1))
    repayment_df = pd.DataFrame({'Year': years}).set_index('Year')
    repayment_df['Repayment Inflow ($)'] = 0.0

    for _, row in schedule_df.iterrows():
        repay_year   = int(row['Year']) + lag
        repay_amount = row['Loans Issued'] * row['Avg Loan Amount']
        if repay_year <= end_year:
            repayment_df.loc[repay_year, 'Repayment Inflow ($)'] += repay_amount

    repayment_df = repayment_df.reset_index()
    repayment_df['Cumulative Inflow ($)'] = (
        repayment_df['Repayment Inflow ($)'].cumsum()
    )
    return repayment_df

repay_base       = build_repayment_schedule(schedule_df, REPAYMENT_LAG_BASE)
repay_optimistic = build_repayment_schedule(schedule_df, REPAYMENT_LAG_OPTIMISTIC)

total_repaid_base   = repay_base['Repayment Inflow ($)'].sum()
repaid_by_2037_base = repay_base.loc[
    repay_base['Year'] <= PROGRAM_END, 'Repayment Inflow ($)'
].sum()

total_repaid_opt   = repay_optimistic['Repayment Inflow ($)'].sum()
repaid_by_2037_opt = repay_optimistic.loc[
    repay_optimistic['Year'] <= PROGRAM_END, 'Repayment Inflow ($)'
].sum()

if VERBOSE:
    for repay_df, lag, label in [
        (repay_base,       REPAYMENT_LAG_BASE,       f'Base Case: L = {REPAYMENT_LAG_BASE} years'),
        (repay_optimistic, REPAYMENT_LAG_OPTIMISTIC, f'Optimistic: L = {REPAYMENT_LAG_OPTIMISTIC} years'),
    ]:
        print(f"\n=== Step 5: Repayment Schedule ({label}) ===\n")
        display = repay_df[
            (repay_df['Year'] <= PROGRAM_END) |
            (repay_df['Repayment Inflow ($)'] > 0)
        ].copy()
        print(display.to_string(index=False,
            formatters={
                'Repayment Inflow ($)':  '${:,.0f}'.format,
                'Cumulative Inflow ($)': '${:,.0f}'.format,
            }
        ))

    print(f"\n=== Key Policy Insight ===")
    print(f"Base case:       {repaid_by_2037_base/total_outflow:.1%} of outflows "
          f"recovered within program window (by {PROGRAM_END})")
    print(f"Optimistic case: {repaid_by_2037_opt/total_outflow:.1%} of outflows "
          f"recovered within program window (by {PROGRAM_END})")
    print(f"\nConclusion: The program behaves like a grant in the near term regardless")
    print(f"of repayment lag. Near-grant-level capital is required upfront.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Cash Flow Model and Funding Gap
# ══════════════════════════════════════════════════════════════════════════════

def build_cash_flow(schedule_df, repay_df, label=''):
    """
    Computes cumulative cash balance. Starting capital = peak funding gap.
    Returns (cash_flow_df, peak_funding_gap, trough_year).
    """
    years           = list(range(PROGRAM_START, PROJECTION_END + 1))
    outflow_by_year = dict(zip(schedule_df['Year'], schedule_df['Annual Outflow ($)']))
    inflow_by_year  = dict(zip(repay_df['Year'],    repay_df['Repayment Inflow ($)']))

    rows = []
    cum_out = cum_in = 0
    for year in years:
        outflow  = outflow_by_year.get(year, 0)
        inflow   = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        rows.append({
            'Year':               year,
            'Annual Outflow ($)': outflow,
            'Annual Inflow ($)':  inflow,
            'Net Position ($)':   cum_in - cum_out,
        })

    cf_df            = pd.DataFrame(rows)
    peak_funding_gap = abs(cf_df['Net Position ($)'].min())
    trough_year      = cf_df.loc[cf_df['Net Position ($)'].idxmin(), 'Year']
    cf_df['Cash Balance ($)'] = cf_df['Net Position ($)'] + peak_funding_gap

    if VERBOSE:
        print(f"\n=== Step 6: Cash Flow Model ({label}) ===\n")
        print(cf_df[[
            'Year', 'Annual Outflow ($)', 'Annual Inflow ($)', 'Cash Balance ($)',
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
              f"${cf_df.loc[cf_df['Year'] == PROJECTION_END, 'Cash Balance ($)'].values[0]:>15,.0f}")

    return cf_df, peak_funding_gap, trough_year

cf_base, gap_base, trough_base = build_cash_flow(
    schedule_df, repay_base, 'Base Case: L=12 years'
)
cf_opt, gap_opt, trough_opt = build_cash_flow(
    schedule_df, repay_optimistic, 'Optimistic: L=7 years'
)

if VERBOSE:
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Grant vs Loan Comparison
# ══════════════════════════════════════════════════════════════════════════════

grant_equivalent = total_outflow

def compute_subsidy(schedule_df, lag, discount_rate=DISCOUNT_RATE):
    """
    Computes aggregate implicit financing subsidy.
    Subsidy per loan = loan_amount * discount_rate * lag.
    Returns (total_subsidy, avg_subsidy_per_loan, subsidy_pct_of_face).
    """
    subsidy_rows = []
    for _, row in schedule_df.iterrows():
        loans            = row['Loans Issued']
        avg_loan         = row['Avg Loan Amount']
        subsidy_per_loan = avg_loan * discount_rate * lag
        subsidy_rows.append({
            'Year':               row['Year'],
            'Loans':              loans,
            'Avg Loan ($)':       avg_loan,
            'Subsidy/Loan ($)':   subsidy_per_loan,
            'Cohort Subsidy ($)': loans * subsidy_per_loan,
        })

    subsidy_df           = pd.DataFrame(subsidy_rows)
    total_subsidy        = subsidy_df['Cohort Subsidy ($)'].sum()
    total_loans          = subsidy_df['Loans'].sum()
    avg_subsidy_per_loan = total_subsidy / total_loans
    subsidy_pct_of_face  = avg_subsidy_per_loan / schedule_df['Avg Loan Amount'].mean()
    return subsidy_df, total_subsidy, avg_subsidy_per_loan, subsidy_pct_of_face

sub_df_base, total_sub_base, avg_sub_base, pct_base = compute_subsidy(
    schedule_df, REPAYMENT_LAG_BASE
)
sub_df_opt, total_sub_opt, avg_sub_opt, pct_opt = compute_subsidy(
    schedule_df, REPAYMENT_LAG_OPTIMISTIC
)

if VERBOSE:
    print("\n=== Step 7: Grant vs Loan Comparison ===\n")
    print(f"--- 7a: Grant Equivalent ---")
    print(f"Total loans issued:              {schedule_df['Loans Issued'].sum():>10,.0f}")
    print(f"Grant equivalent cost:           ${grant_equivalent:>14,.0f}")
    print(f"  (= total program outflow, no repayment assumed)")
    print(f"\n--- 7b: Implicit Financing Subsidy ---")
    print(f"Municipal discount rate:         {DISCOUNT_RATE:.1%}")
    for sub_df, total_sub, avg_sub, pct, lag, label in [
        (sub_df_base, total_sub_base, avg_sub_base, pct_base,
         REPAYMENT_LAG_BASE,       f'Base Case (L = {REPAYMENT_LAG_BASE} years)'),
        (sub_df_opt,  total_sub_opt,  avg_sub_opt,  pct_opt,
         REPAYMENT_LAG_OPTIMISTIC, f'Optimistic Case (L = {REPAYMENT_LAG_OPTIMISTIC} years)'),
    ]:
        print(f"\n{label}:")
        print(sub_df.to_string(index=False,
            formatters={
                'Loans':              '{:,.0f}'.format,
                'Avg Loan ($)':       '${:,.2f}'.format,
                'Subsidy/Loan ($)':   '${:,.2f}'.format,
                'Cohort Subsidy ($)': '${:,.0f}'.format,
            }
        ))
        print(f"\n  Total implicit subsidy:          ${total_sub:>14,.0f}")
        print(f"  Avg subsidy per loan:            ${avg_sub:>14,.2f}")
        print(f"  Subsidy as % of avg face value:  {pct:>14.1%}")

    print(f"\n--- 7c: Full Model Comparison ---")
    print(f"\n{'Metric':<45} {'Grant':>12} {'Loan (L=12)':>14} {'Loan (L=7)':>14}")
    print("-" * 87)
    print(f"{'Required starting capital':<45} "
          f"${grant_equivalent:>11,.0f} ${gap_base:>13,.0f} ${gap_opt:>13,.0f}")
    print(f"{'Long-run net cost to municipality':<45} "
          f"${grant_equivalent:>11,.0f} {'$0':>14} {'$0':>14}")
    print(f"{'Implicit financing subsidy':<45} "
          f"{'N/A':>12} ${total_sub_base:>13,.0f} ${total_sub_opt:>13,.0f}")
    print(f"{'True economic cost (net + subsidy)':<45} "
          f"${grant_equivalent:>11,.0f} ${total_sub_base:>13,.0f} ${total_sub_opt:>13,.0f}")
    print(f"{'Trough year':<45} "
          f"{'2037':>12} {'2037':>14} {'2036':>14}")
    print(f"{'Admin complexity':<45} "
          f"{'Low':>12} {'High':>14} {'High':>14}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_full_model(part_rate, lag, discount_rate=DISCOUNT_RATE):
    """
    Runs the full Phase 1 model for a given participation rate and lag.
    Returns (peak_funding_gap, total_outflow, implicit_subsidy).
    """
    n_part     = N_total * part_rate
    sched_rows = []
    cumulative = 0

    for year, pct in REPLACEMENT_SCHEDULE.items():
        loans_issued   = min(n_part * pct, n_part - cumulative)
        cumulative    += loans_issued
        avg_loan       = min(
            AVG_LOAN_BASE * (1 + COST_ESCALATION) ** (year - BASE_YEAR),
            LOAN_CAP
        )
        sched_rows.append({
            'Year':               year,
            'Loans Issued':       loans_issued,
            'Avg Loan Amount':    avg_loan,
            'Annual Outflow ($)': loans_issued * avg_loan,
        })

    sched     = pd.DataFrame(sched_rows)
    total_out = sched['Annual Outflow ($)'].sum()

    years          = list(range(PROGRAM_START, PROJECTION_END + 1))
    inflow_by_year = {y: 0.0 for y in years}
    for _, row in sched.iterrows():
        repay_year = int(row['Year']) + lag
        if repay_year <= PROJECTION_END:
            inflow_by_year[repay_year] += row['Annual Outflow ($)']

    cum_out = cum_in = 0
    net_positions   = []
    for year in years:
        outflow  = sched.loc[sched['Year'] == year, 'Annual Outflow ($)'].sum() \
                   if year <= PROGRAM_END else 0
        inflow   = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        net_positions.append(cum_in - cum_out)

    return abs(min(net_positions)), total_out, total_out * discount_rate * lag

capital_results = {
    p: {l: run_full_model(r, LAG_SCENARIOS[l])
        for l in LAG_SCENARIOS}
    for p, r in PARTICIPATION_SCENARIOS.items()
}

if VERBOSE:
    header = f"{'Scenario':<25}" + "".join(
        f"{lag_label:>20}" for lag_label in LAG_SCENARIOS.keys()
    )
    divider = "-" * (25 + 20 * len(LAG_SCENARIOS))

    for table_label, value_idx, table_name in [
        ('Required Starting Capital ($)',          0, 'Table 1'),
        ('Total Program Outflow ($)',              1, 'Table 2'),
        ('Implicit Financing Subsidy ($)',         2, 'Table 3'),
        ('True Economic Cost (Net + Subsidy) ($)', 2, 'Table 4'),
    ]:
        print(f"\n=== Step 8: Sensitivity Analysis ===")
        print(f"\n--- {table_name}: {table_label} ---\n")
        print(header)
        print(divider)
        for part_label in PARTICIPATION_SCENARIOS:
            row_str = f"{part_label:<25}"
            for lag_label in LAG_SCENARIOS:
                val = capital_results[part_label][lag_label][value_idx]
                row_str += f"  ${val:>15,.0f}"
            print(row_str)

    all_gaps = [
        capital_results[p][l][0]
        for p in PARTICIPATION_SCENARIOS
        for l in LAG_SCENARIOS
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
