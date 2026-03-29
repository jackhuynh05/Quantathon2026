# phase2.py
# ── LEAP Program: Phase 2 Monte Carlo Simulation ──────────────────────────────

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import *
from phase1 import (
    N_total,
    N_participating,
    schedule_df,
    total_outflow,
    gap_base,
    gap_opt,
    build_cash_flow,
    build_repayment_schedule,
)

N_SIMULATIONS = 10_000
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Load and Fit Cost Distribution from Real Quote Data
# ══════════════════════════════════════════════════════════════════════════════

print("=== Module 1: Cost Distribution (Real Quote Data) ===\n")

quotes_raw = pd.read_excel(
    EXCEL_PATH,
    sheet_name='LEAP Quotes - No Names',
    header=None
)

# Extract chosen contractor quotes (column 5, rows 8-71)
chosen_quotes = pd.to_numeric(
    quotes_raw.iloc[7:71, 5], errors='coerce'
).dropna().values

# Extract contract amounts (column 6, rows 8-71)
contract_amounts = pd.to_numeric(
    quotes_raw.iloc[7:71, 6], errors='coerce'
).dropna().values

print(f"Chosen quotes loaded:     {len(chosen_quotes)} observations")
print(f"Contract amounts loaded:  {len(contract_amounts)} observations")
print(f"\nChosen quote summary:")
print(f"  Mean:    ${np.mean(chosen_quotes):>10,.2f}")
print(f"  Median:  ${np.median(chosen_quotes):>10,.2f}")
print(f"  Std dev: ${np.std(chosen_quotes):>10,.2f}")
print(f"  Min:     ${np.min(chosen_quotes):>10,.2f}")
print(f"  Max:     ${np.max(chosen_quotes):>10,.2f}")
print(f"  % above $10k cap: {(chosen_quotes > LOAN_CAP).mean():.1%}")

# Fit log-normal to chosen quotes
# Log-normal: log(X) ~ Normal(mu, sigma)
log_quotes = np.log(chosen_quotes)
mu_cost    = log_quotes.mean()
sigma_cost = log_quotes.std()

print(f"\nLog-normal fit to chosen quotes:")
print(f"  mu (log scale):    {mu_cost:.4f}")
print(f"  sigma (log scale): {sigma_cost:.4f}")
print(f"  Implied mean:      ${np.exp(mu_cost + 0.5 * sigma_cost**2):>10,.2f}")
print(f"  Implied median:    ${np.exp(mu_cost):>10,.2f}")

# ── KS test: how well does log-normal fit? ────────────────────────────────────
ks_stat, ks_pval = stats.kstest(
    chosen_quotes,
    'lognorm',
    args=(sigma_cost, 0, np.exp(mu_cost))
)
print(f"\nKolmogorov-Smirnov goodness-of-fit test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value:      {ks_pval:.4f}")
print(f"  Fit quality:  {'Good (p > 0.05)' if ks_pval > 0.05 else 'Marginal (p <= 0.05)'}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Load Participation Data — Proactive and Leak Loan Channels
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 2: Participation Channels ===\n")

part_raw = pd.read_excel(
    EXCEL_PATH,
    sheet_name='LEAP Participation',
    header=None
)

# Extract values from known row positions
pilot_proactive = int(part_raw.iloc[28, 3])   # proactive loans
pilot_leak      = int(part_raw.iloc[31, 3])   # leak-triggered loans
pilot_applied   = PILOT_APPLIED
pilot_completed = PILOT_COMPLETED

# ── Voluntary participation channel ───────────────────────────────────────────
# Base rate: completed / applied (same as Phase 1)
voluntary_rate = pilot_completed / pilot_applied

# ── Leak (breakage) channel ───────────────────────────────────────────────────
# Annual breakage rate = leak loans in pilot / total eligible universe
annual_breakage_rate = pilot_leak / N_total

# ── Proactive channel ─────────────────────────────────────────────────────────
# Proactive loans are utility-initiated — fixed annual add-on
annual_proactive_rate = pilot_proactive / N_total

print(f"Pilot loan breakdown:")
print(f"  Voluntary completed:  {pilot_completed}")
print(f"  Proactive loans:      {pilot_proactive}")
print(f"  Leak loans:           {pilot_leak}")
print(f"\nDerived rates (per eligible property per year):")
print(f"  Voluntary rate (completed/applied): {voluntary_rate:.4f} ({voluntary_rate:.2%})")
print(f"  Annual breakage rate:               {annual_breakage_rate:.4f} ({annual_breakage_rate:.2%})")
print(f"  Annual proactive rate:              {annual_proactive_rate:.4f} ({annual_proactive_rate:.2%})")

# ── Insurance data cross-check on breakage rate ───────────────────────────────
insurance_raw = pd.read_excel(
    EXCEL_PATH,
    sheet_name='Insurance',
    header=None
)


# Row 30 = WSL repairs (index 29), Row 20 = WSL policies (index 19)
# Column 7 = most recent snapshot March 2026 (index 6)
wsl_repairs_5months            = float(insurance_raw.iloc[30, 6])   # row 30 = WSL repairs
wsl_policies                   = float(insurance_raw.iloc[20, 6])   # row 20 = WSL policies
repair_rate_5months            = wsl_repairs_5months / wsl_policies
annual_breakage_rate_insurance = repair_rate_5months * (12 / 5)

print(f"\n--- Insurance Data Cross-Check ---")
print(f"  WSL repairs over 5 months:          {wsl_repairs_5months:.0f}")
print(f"  Active WSL policies:                {wsl_policies:.0f}")
print(f"  Repair rate (5 months):             {repair_rate_5months:.2%}")
print(f"  Insurance-implied annual rate:      {annual_breakage_rate_insurance:.2%}")
print(f"  Pilot-implied annual rate:          {annual_breakage_rate:.2%}")
print(f"  Ratio (insurance / pilot):          "
      f"{annual_breakage_rate_insurance / annual_breakage_rate:.1f}x")

if annual_breakage_rate_insurance > annual_breakage_rate * 2:
    print(f"\n  WARNING: Insurance data implies a breakage rate "
          f"{annual_breakage_rate_insurance / annual_breakage_rate:.1f}x "
          f"higher than the pilot estimate.")
    print(f"  The insurance-implied rate will be used as an adverse scenario.")

# ── Adjusted breakage rate (insurance rate corrected for selection bias) ───────
# WSL insurance adoption rate among eligible properties

insurance_adoption_rate  = wsl_policies / N_total
selection_bias_factor    = 2.5
annual_breakage_rate_adj = annual_breakage_rate_insurance / selection_bias_factor

print(f"\n--- Adjusted Breakage Rate (Selection Bias Correction) ---")
print(f"  WSL insurance adoption rate:        {insurance_adoption_rate:.2%}")
print(f"  Selection bias correction factor:   {selection_bias_factor:.1f}x")
print(f"  Adjusted annual breakage rate:      {annual_breakage_rate_adj:.2%}")
print(f"  (Insurance rate {0.0344} / {selection_bias_factor:.1f})")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Calibrate Stochastic Distributions
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 3: Stochastic Distribution Calibration ===\n")

# ── Distribution 1: Universe size N_total ─────────────────────────────────────
# The 10,062 estimated hazardous unknowns carry uncertainty.
# Model as Normal centered on N_total with sigma = 10% of estimated unknowns.
N_total_mean  = N_total
N_total_sigma = 0.10 * 10_062   # ~10% uncertainty on the estimated component

print(f"Universe size distribution:")
print(f"  N_total ~ Normal({N_total_mean:,.0f}, {N_total_sigma:,.0f})")
print(f"  95% CI: [{N_total_mean - 1.96*N_total_sigma:,.0f}, "
      f"{N_total_mean + 1.96*N_total_sigma:,.0f}]")

# ── Distribution 2: Participation rate ────────────────────────────────────────
# Beta distribution calibrated to pilot data.
# Alpha = completed, Beta = applied - completed
beta_alpha = pilot_completed                  # 42 successes
beta_beta  = pilot_applied - pilot_completed  # 40 failures

beta_mean = beta_alpha / (beta_alpha + beta_beta)
beta_var  = (beta_alpha * beta_beta) / (
    (beta_alpha + beta_beta)**2 * (beta_alpha + beta_beta + 1)
)

print(f"\nParticipation rate distribution:")
print(f"  rho ~ Beta({beta_alpha}, {beta_beta})")
print(f"  Mean:    {beta_mean:.4f} ({beta_mean:.2%})")
print(f"  Std dev: {np.sqrt(beta_var):.4f}")
print(f"  95% CI: [{stats.beta.ppf(0.025, beta_alpha, beta_beta):.2%}, "
      f"{stats.beta.ppf(0.975, beta_alpha, beta_beta):.2%}]")

# ── Distribution 3: Repayment lag ─────────────────────────────────────────────
# Log-normal calibrated to mean=12, std=5 years.
lag_mean  = 12.0
lag_std   = 5.0
lag_var   = lag_std**2
sigma_lag = np.sqrt(np.log(1 + lag_var / lag_mean**2))
mu_lag    = np.log(lag_mean) - 0.5 * sigma_lag**2

print(f"\nRepayment lag distribution:")
print(f"  L ~ LogNormal(mu={mu_lag:.4f}, sigma={sigma_lag:.4f})")
print(f"  Mean:     {lag_mean:.1f} years")
print(f"  Std dev:  {lag_std:.1f} years")
print(f"  Median:   {np.exp(mu_lag):.1f} years")
print(f"  95th pct: {stats.lognorm.ppf(0.95, sigma_lag, scale=np.exp(mu_lag)):.1f} years")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Monte Carlo Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n=== Module 4: Monte Carlo Simulation ({N_SIMULATIONS:,} runs) ===\n")

def simulate_single_run(
    mu_cost, sigma_cost,
    N_total_mean, N_total_sigma,
    beta_alpha, beta_beta,
    mu_lag, sigma_lag,
    annual_breakage_rate, annual_proactive_rate,
):
    """
    Runs one Monte Carlo simulation of the LEAP program.
    Returns (peak_funding_gap, total_outflow) for this simulation.
    """
    # ── Draw stochastic inputs ─────────────────────────────────────────────────
    n_total_s = max(
        int(np.random.normal(N_total_mean, N_total_sigma)),
        1000
    )
    rho_s = np.random.beta(beta_alpha, beta_beta)

    # ── Three participation channels ───────────────────────────────────────────
    n_voluntary = int(n_total_s * rho_s)
    n_breakage  = int(n_total_s * annual_breakage_rate * (PROGRAM_END - PROGRAM_START + 1))
    n_proactive = int(n_total_s * annual_proactive_rate * (PROGRAM_END - PROGRAM_START + 1))
    n_part_s    = n_voluntary + n_breakage + n_proactive

    # ── Replacement schedule ───────────────────────────────────────────────────
    sched_rows = []
    cumulative = 0

    for year, pct in REPLACEMENT_SCHEDULE.items():
        loans_issued = min(n_part_s * pct, n_part_s - cumulative)
        cumulative  += loans_issued

        # Draw average cost for this cohort from log-normal distribution
        if loans_issued > 0:
            cost_sample  = np.random.lognormal(
                mu_cost, sigma_cost, size=max(int(loans_issued), 1)
            )
            avg_loan_raw = np.mean(cost_sample)
        else:
            avg_loan_raw = np.exp(mu_cost + 0.5 * sigma_cost**2)

        # Apply cost escalation and cap
        escalation_factor = (1 + COST_ESCALATION) ** (year - BASE_YEAR)
        avg_loan_s        = min(avg_loan_raw * escalation_factor, LOAN_CAP)

        sched_rows.append({
            'Year':               year,
            'Loans Issued':       loans_issued,
            'Avg Loan Amount':    avg_loan_s,
            'Annual Outflow ($)': loans_issued * avg_loan_s,
        })

    sched     = pd.DataFrame(sched_rows)
    total_out = sched['Annual Outflow ($)'].sum()

    # ── Repayment schedule with stochastic lags ────────────────────────────────
    years          = list(range(PROGRAM_START, PROJECTION_END + 1))
    inflow_by_year = {y: 0.0 for y in years}

    for _, row in sched.iterrows():
        if row['Loans Issued'] > 0:
            lag_s      = max(1, int(np.random.lognormal(mu_lag, sigma_lag)))
            repay_year = int(row['Year']) + lag_s
            if repay_year <= PROJECTION_END:
                inflow_by_year[repay_year] += row['Annual Outflow ($)']

    # ── Cash flow and peak funding gap ─────────────────────────────────────────
    cum_out       = cum_in = 0
    net_positions = []

    for year in years:
        outflow  = sched.loc[
            sched['Year'] == year, 'Annual Outflow ($)'
        ].sum() if year <= PROGRAM_END else 0
        inflow   = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        net_positions.append(cum_in - cum_out)

    return abs(min(net_positions)), total_out

# ── Run primary simulation (pilot breakage rate) ───────────────────────────────
print("Running primary simulation (pilot breakage rate)...")
results_gap     = np.zeros(N_SIMULATIONS)
results_outflow = np.zeros(N_SIMULATIONS)

for s in range(N_SIMULATIONS):
    gap_s, out_s       = simulate_single_run(
        mu_cost, sigma_cost,
        N_total_mean, N_total_sigma,
        beta_alpha, beta_beta,
        mu_lag, sigma_lag,
        annual_breakage_rate,
        annual_proactive_rate,
    )
    results_gap[s]     = gap_s
    results_outflow[s] = out_s

    if (s + 1) % 1000 == 0:
        print(f"  Completed {s+1:>6,} / {N_SIMULATIONS:,} simulations...")

print(f"\nPrimary simulation complete.")

# ── Run adverse simulation (insurance-implied breakage rate) ───────────────────
print(f"\nRunning adverse simulation "
      f"(insurance-implied breakage rate: {annual_breakage_rate_insurance:.2%})...")

results_gap_insurance     = np.zeros(N_SIMULATIONS)
results_outflow_insurance = np.zeros(N_SIMULATIONS)

for s in range(N_SIMULATIONS):
    gap_s, out_s = simulate_single_run(
        mu_cost, sigma_cost,
        N_total_mean, N_total_sigma,
        beta_alpha, beta_beta,
        mu_lag, sigma_lag,
        annual_breakage_rate_insurance,
        annual_proactive_rate,
    )
    results_gap_insurance[s]     = gap_s
    results_outflow_insurance[s] = out_s

    if (s + 1) % 1000 == 0:
        print(f"  Completed {s+1:>6,} / {N_SIMULATIONS:,} simulations...")

# ── Run adjusted simulation (selection-bias-corrected breakage rate) ───────────
print(f"\nRunning adjusted simulation "
      f"(selection-bias-corrected breakage rate: {annual_breakage_rate_adj:.2%})...")

results_gap_adj     = np.zeros(N_SIMULATIONS)
results_outflow_adj = np.zeros(N_SIMULATIONS)

for s in range(N_SIMULATIONS):
    gap_s, out_s = simulate_single_run(
        mu_cost, sigma_cost,
        N_total_mean, N_total_sigma,
        beta_alpha, beta_beta,
        mu_lag, sigma_lag,
        annual_breakage_rate_adj,
        annual_proactive_rate,
    )
    results_gap_adj[s]     = gap_s
    results_outflow_adj[s] = out_s

    if (s + 1) % 1000 == 0:
        print(f"  Completed {s+1:>6,} / {N_SIMULATIONS:,} simulations...")


print(f"\nAdverse simulation complete.")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Results and Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 5: Simulation Results ===\n")

# ── Primary simulation statistics ─────────────────────────────────────────────
pct_5    = np.percentile(results_gap, 5)
pct_25   = np.percentile(results_gap, 25)
pct_50   = np.percentile(results_gap, 50)
pct_75   = np.percentile(results_gap, 75)
pct_95   = np.percentile(results_gap, 95)
pct_99   = np.percentile(results_gap, 99)
gap_mean = np.mean(results_gap)
gap_std  = np.std(results_gap)

print(f"--- Required Starting Capital: Primary Simulation ---\n")
print(f"  Mean:             ${gap_mean:>15,.0f}")
print(f"  Std deviation:    ${gap_std:>15,.0f}")
print(f"  5th percentile:   ${pct_5:>15,.0f}")
print(f"  25th percentile:  ${pct_25:>15,.0f}")
print(f"  Median (50th):    ${pct_50:>15,.0f}")
print(f"  75th percentile:  ${pct_75:>15,.0f}")
print(f"  95th percentile:  ${pct_95:>15,.0f}  ← recommended funding level")
print(f"  99th percentile:  ${pct_99:>15,.0f}")

print(f"\n--- Phase 1 vs Phase 2 Comparison ---\n")
print(f"  Phase 1 base case (deterministic, L=12):  ${gap_base:>15,.0f}")
print(f"  Phase 1 optimistic (deterministic, L=7):  ${gap_opt:>15,.0f}")
print(f"  Phase 2 mean (stochastic):                ${gap_mean:>15,.0f}")
print(f"  Phase 2 median (stochastic):              ${pct_50:>15,.0f}")
print(f"  Phase 2 95th percentile (stochastic):     ${pct_95:>15,.0f}")
print(f"\n  Buffer above Phase 1 base (95th pct):     "
      f"${pct_95 - gap_base:>15,.0f} "
      f"({(pct_95 - gap_base) / gap_base:.1%})")
print(f"  Probability program needs > Phase 1 base: "
      f"{(results_gap > gap_base).mean():.1%}")

print(f"\n--- Total Program Outflow: Primary Simulation ---\n")
print(f"  Mean outflow:           ${np.mean(results_outflow):>15,.0f}")
print(f"  Std deviation:          ${np.std(results_outflow):>15,.0f}")
print(f"  5th percentile:         ${np.percentile(results_outflow, 5):>15,.0f}")
print(f"  95th percentile:        ${np.percentile(results_outflow, 95):>15,.0f}")
print(f"  Phase 1 total outflow:  ${total_outflow:>15,.0f}")

# ── Breakage rate sensitivity comparison ──────────────────────────────────────
print(f"\n--- Breakage Rate Sensitivity: Three Scenarios ---\n")
print(f"{'Metric':<45} {'Primary':>14} {'Adjusted':>14} {'Adverse':>14}")
print("-" * 89)
print(f"{'Annual breakage rate':<45} "
      f"{annual_breakage_rate:.2%}{'':>9} "
      f"{annual_breakage_rate_adj:.2%}{'':>9} "
      f"{annual_breakage_rate_insurance:.2%}")
print(f"{'Mean required capital':<45} "
      f"${np.mean(results_gap):>13,.0f} "
      f"${np.mean(results_gap_adj):>13,.0f} "
      f"${np.mean(results_gap_insurance):>13,.0f}")
print(f"{'Median required capital':<45} "
      f"${np.percentile(results_gap, 50):>13,.0f} "
      f"${np.percentile(results_gap_adj, 50):>13,.0f} "
      f"${np.percentile(results_gap_insurance, 50):>13,.0f}")
print(f"{'95th percentile capital':<45} "
      f"${np.percentile(results_gap, 95):>13,.0f} "
      f"${np.percentile(results_gap_adj, 95):>13,.0f} "
      f"${np.percentile(results_gap_insurance, 95):>13,.0f}")
print(f"{'99th percentile capital':<45} "
      f"${np.percentile(results_gap, 99):>13,.0f} "
      f"${np.percentile(results_gap_adj, 99):>13,.0f} "
      f"${np.percentile(results_gap_insurance, 99):>13,.0f}")
print(f"{'Mean total outflow':<45} "
      f"${np.mean(results_outflow):>13,.0f} "
      f"${np.mean(results_outflow_adj):>13,.0f} "
      f"${np.mean(results_outflow_insurance):>13,.0f}")
print(f"{'95th percentile outflow':<45} "
      f"${np.percentile(results_outflow, 95):>13,.0f} "
      f"${np.percentile(results_outflow_adj, 95):>13,.0f} "
      f"${np.percentile(results_outflow_insurance, 95):>13,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 6: Generating Visualizations ===\n")

# ── Figure 1: Primary vs Adverse (pilot vs insurance) ─────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    'LEAP Program — Phase 2: Primary vs Adverse Scenarios',
    fontsize=14, fontweight='bold'
)

# Plot 1a: Distribution — primary vs adverse
ax = axes1[0, 0]
ax.hist(results_gap / 1e6, bins=80, color='steelblue',
        alpha=0.7, edgecolor='white', label='Primary (0.24%)')
ax.hist(results_gap_insurance / 1e6, bins=80, color='coral',
        alpha=0.4, edgecolor='white', label='Adverse (3.44%)')
ax.axvline(np.percentile(results_gap, 95) / 1e6,
           color='steelblue', linewidth=2, linestyle='--',
           label=f'95th pct (primary): ${np.percentile(results_gap,95)/1e6:.1f}M')
ax.axvline(np.percentile(results_gap_insurance, 95) / 1e6,
           color='darkred', linewidth=2, linestyle='--',
           label=f'95th pct (adverse): ${np.percentile(results_gap_insurance,95)/1e6:.1f}M')
ax.axvline(gap_base / 1e6, color='green', linewidth=2, linestyle='-',
           label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax.set_title('Distribution of Required Starting Capital')
ax.set_xlabel('Required Capital ($M)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# Plot 1b: CDF — primary vs adverse
ax = axes1[0, 1]
sorted_gaps     = np.sort(results_gap) / 1e6
sorted_gaps_ins = np.sort(results_gap_insurance) / 1e6
cdf             = np.arange(1, N_SIMULATIONS + 1) / N_SIMULATIONS
ax.plot(sorted_gaps,     cdf, color='steelblue', linewidth=2,
        label='Primary (0.24%)')
ax.plot(sorted_gaps_ins, cdf, color='coral',     linewidth=2,
        label='Adverse (3.44%)')
ax.axhline(0.95, color='red',    linestyle='--', linewidth=1.5,
           label='95th percentile')
ax.axhline(0.50, color='orange', linestyle='--', linewidth=1.5,
           label='50th percentile')
ax.set_title('CDF of Required Starting Capital')
ax.set_xlabel('Required Capital ($M)')
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(True, alpha=0.3)

# Plot 1c: Outflow distribution — primary vs adverse
ax = axes1[1, 0]
ax.hist(results_outflow / 1e6, bins=80, color='steelblue',
        alpha=0.7, edgecolor='white', label='Primary (0.24%)')
ax.hist(results_outflow_insurance / 1e6, bins=80, color='coral',
        alpha=0.4, edgecolor='white', label='Adverse (3.44%)')
ax.axvline(total_outflow / 1e6, color='black', linewidth=2, linestyle='-',
           label=f'Phase 1: ${total_outflow/1e6:.1f}M')
ax.set_title('Distribution of Total Program Outflow')
ax.set_xlabel('Total Outflow ($M)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# Plot 1d: Required capital by confidence level — primary vs adverse
ax = axes1[1, 1]
percentiles    = np.arange(1, 100)
pct_values     = [np.percentile(results_gap,           p) / 1e6 for p in percentiles]
pct_values_ins = [np.percentile(results_gap_insurance, p) / 1e6 for p in percentiles]
ax.plot(percentiles, pct_values,     color='steelblue', linewidth=2,
        label='Primary (0.24%)')
ax.plot(percentiles, pct_values_ins, color='coral',     linewidth=2,
        label='Adverse (3.44%)')
ax.axhline(gap_base / 1e6, color='green',  linestyle='--', linewidth=1.5,
           label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax.axhline(gap_opt / 1e6,  color='purple', linestyle='--', linewidth=1.5,
           label=f'Phase 1 optimistic: ${gap_opt/1e6:.1f}M')
ax.fill_between(percentiles, pct_values, pct_values_ins,
                alpha=0.15, color='orange',
                label='Uncertainty band')
ax.axvline(95, color='red', linestyle=':', linewidth=1.5, label='95th pct')
ax.set_title('Required Capital by Confidence Level')
ax.set_xlabel('Confidence Level (%)')
ax.set_ylabel('Required Capital ($M)')
ax.legend(fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase2_results_primary_adverse.png', dpi=150, bbox_inches='tight')
print("Saved: phase2_results_primary_adverse.png")
plt.close(fig1)

# ── Precompute shared variables for both figures ──────────────────────────────
percentiles     = np.arange(1, 100)
pct_values      = [np.percentile(results_gap,           p) / 1e6 for p in percentiles]
pct_values_adj  = [np.percentile(results_gap_adj,       p) / 1e6 for p in percentiles]
pct_values_ins  = [np.percentile(results_gap_insurance, p) / 1e6 for p in percentiles]
sorted_gaps     = np.sort(results_gap) / 1e6
sorted_gaps_adj = np.sort(results_gap_adj) / 1e6
sorted_gaps_ins = np.sort(results_gap_insurance) / 1e6
cdf             = np.arange(1, N_SIMULATIONS + 1) / N_SIMULATIONS

# ── Figure 2: Adjusted scenario only ──────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(
    f'LEAP Program — Phase 2: Adjusted Scenario '
    f'(Selection-Bias-Corrected Breakage Rate: '
    f'{annual_breakage_rate_adj:.2%})',
    fontsize=13, fontweight='bold'
)

# Plot 2a: Distribution — adjusted only
ax = axes2[0, 0]
ax.hist(results_gap_adj / 1e6, bins=80, color='mediumpurple',
        alpha=0.7, edgecolor='white')
ax.axvline(np.percentile(results_gap_adj, 95) / 1e6,
           color='red', linewidth=2, linestyle='--',
           label=f'95th pct: ${np.percentile(results_gap_adj,95)/1e6:.1f}M')
ax.axvline(np.percentile(results_gap_adj, 50) / 1e6,
           color='orange', linewidth=2, linestyle='--',
           label=f'Median: ${np.percentile(results_gap_adj,50)/1e6:.1f}M')
ax.axvline(gap_base / 1e6, color='green', linewidth=2, linestyle='-',
           label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax.set_title('Distribution of Required Starting Capital\n(Adjusted Breakage Rate)')
ax.set_xlabel('Required Capital ($M)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# Plot 2b: CDF — adjusted only with primary and adverse for reference
ax = axes2[0, 1]
sorted_gaps_adj = np.sort(results_gap_adj) / 1e6
ax.plot(sorted_gaps,     cdf, color='steelblue',    linewidth=1.5,
        linestyle=':', alpha=0.6, label='Primary (0.24%) — reference')
ax.plot(sorted_gaps_adj, cdf, color='mediumpurple', linewidth=2.5,
        label='Adjusted (1.38%)')
ax.plot(sorted_gaps_ins, cdf, color='coral',        linewidth=1.5,
        linestyle=':', alpha=0.6, label='Adverse (3.44%) — reference')
ax.axhline(0.95, color='red',    linestyle='--', linewidth=1.5,
           label='95th percentile')
ax.axhline(0.50, color='orange', linestyle='--', linewidth=1.5,
           label='50th percentile')
ax.set_title('CDF of Required Starting Capital\n(Adjusted vs Reference Scenarios)')
ax.set_xlabel('Required Capital ($M)')
ax.set_ylabel('Cumulative Probability')
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(True, alpha=0.3)

# Plot 2c: Outflow distribution — adjusted only
ax = axes2[1, 0]
ax.hist(results_outflow_adj / 1e6, bins=80, color='mediumpurple',
        alpha=0.7, edgecolor='white')
ax.axvline(np.percentile(results_outflow_adj, 95) / 1e6, color='red',
           linewidth=2, linestyle='--',
           label=f'95th pct: ${np.percentile(results_outflow_adj,95)/1e6:.1f}M')
ax.axvline(np.mean(results_outflow_adj) / 1e6, color='darkred',
           linewidth=2, linestyle='-',
           label=f'Mean: ${np.mean(results_outflow_adj)/1e6:.1f}M')
ax.axvline(total_outflow / 1e6, color='black', linewidth=2, linestyle='-',
           label=f'Phase 1: ${total_outflow/1e6:.1f}M')
ax.set_title('Distribution of Total Program Outflow\n(Adjusted Breakage Rate)')
ax.set_xlabel('Total Outflow ($M)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# Plot 2d: Required capital by confidence level — adjusted with bands
ax = axes2[1, 1]
pct_values_adj = [np.percentile(results_gap_adj, p) / 1e6 for p in percentiles]
ax.plot(percentiles, pct_values,     color='steelblue',    linewidth=1.5,
        linestyle=':', alpha=0.6, label='Primary (0.24%) — reference')
ax.plot(percentiles, pct_values_adj, color='mediumpurple', linewidth=2.5,
        label='Adjusted (1.38%)')
ax.plot(percentiles, pct_values_ins, color='coral',        linewidth=1.5,
        linestyle=':', alpha=0.6, label='Adverse (3.44%) — reference')
ax.axhline(gap_base / 1e6, color='green',  linestyle='--', linewidth=1.5,
           label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax.axhline(gap_opt / 1e6,  color='purple', linestyle='--', linewidth=1.5,
           label=f'Phase 1 optimistic: ${gap_opt/1e6:.1f}M')
ax.fill_between(percentiles, pct_values, pct_values_adj,
                alpha=0.15, color='steelblue',
                label='Primary to adjusted band')
ax.fill_between(percentiles, pct_values_adj, pct_values_ins,
                alpha=0.15, color='coral',
                label='Adjusted to adverse band')
ax.axvline(95, color='red', linestyle=':', linewidth=1.5, label='95th pct')
ax.set_title('Required Capital by Confidence Level\n(Adjusted vs Reference Scenarios)')
ax.set_xlabel('Confidence Level (%)')
ax.set_ylabel('Required Capital ($M)')
ax.legend(fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase2_results_adjusted.png', dpi=150, bbox_inches='tight')
print("Saved: phase2_results_adjusted.png")
plt.close(fig2)

print("\nPhase 2 complete.")
