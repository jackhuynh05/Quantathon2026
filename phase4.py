# phase4.py
# ── LEAP Program: Phase 4 Competing Risks Repayment Model ─────────────────────
# Replaces the single log-normal repayment lag from Phase 2 with a structured
# competing risks model calibrated to real Columbus HMDA and Census data.
# Three repayment triggers: property sale, refinancing/HELOC, permanent non-repayment.
# Interest rate sensitivity modeled via Vasicek mean-reverting process.

import pandas as pd
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import *
from phase1 import (
    N_total,
    gap_base,
    gap_opt,
    total_outflow,
    schedule_df,
)
from phase2 import (
    pct_95,
    mu_cost,
    sigma_cost,
    annual_breakage_rate,
    annual_proactive_rate,
    N_total_mean,
    N_total_sigma,
    beta_alpha,
    beta_beta,
)

N_SIMULATIONS = 3000
np.random.seed(42)

VERBOSE = __name__ == '__main__'

print("=" * 70)
print("PHASE 4: Competing Risks Repayment Model")
print("Calibrated to Columbus HMDA (2022) and Census ACS B25026")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Calibrate Sale Hazard from Census Tenure Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 1: Sale Hazard Calibration (Census B25026) ===\n")

# ── Parse Census tenure distribution ──────────────────────────────────────────
# Owner-occupied population by year moved in (rows 2-7 of B25026)
# We convert to midpoint years of tenure as of 2024 survey reference

# Tenure bucket midpoints (years owned as of 2024)
# "Moved in 2023 or later" → ~1 year
# "Moved in 2020 to 2022" → ~3 years
# "Moved in 2010 to 2019" → ~9 years
# "Moved in 2000 to 2009" → ~19 years
# "Moved in 1990 to 1999" → ~29 years
# "Moved in 1989 or earlier" → ~40 years

tenure_midpoints = np.array([1, 3, 9, 19, 29, 40])
owner_counts     = np.array([121_335, 294_678, 541_896, 251_936, 131_515, 114_908])
total_owners     = owner_counts.sum()

# Convert counts to proportions
tenure_proportions = owner_counts / total_owners

print(f"Columbus MSA owner tenure distribution:")
labels = ['<1 yr', '1-4 yr', '5-14 yr', '15-24 yr', '25-34 yr', '35+ yr']
for lbl, prop, cnt in zip(labels, tenure_proportions, owner_counts):
    print(f"  {lbl:<12} {prop:.2%}  ({cnt:>10,})")
print(f"  Total owners: {total_owners:>10,}")

# ── Fit Weibull distribution to tenure data ───────────────────────────────────
# We fit a Weibull to the observed tenure distribution using maximum likelihood
# The Weibull hazard h(t) = (k/λ)(t/λ)^(k-1) gives us the instantaneous
# probability of sale at tenure t

# Create expanded sample from grouped data for MLE fitting
tenure_sample = np.repeat(tenure_midpoints, owner_counts)

# Fit Weibull (scipy uses shape=k, scale=λ)
shape_k, loc, scale_lam = stats.weibull_min.fit(tenure_sample, floc=0)

print(f"\nWeibull fit to tenure distribution:")
print(f"  Shape k:   {shape_k:.4f}")
print(f"  Scale λ:   {scale_lam:.4f} years")
print(f"  Implied median tenure: {scale_lam * np.log(2)**(1/shape_k):.1f} years")

# ── Annual sale probability by tenure year ────────────────────────────────────
# P(sale in year t | survived to year t) = hazard rate at t
def weibull_hazard(t, k, lam):
    """Weibull hazard rate at tenure t."""
    return (k / lam) * (t / lam) ** (k - 1)

def annual_sale_prob(t, k, lam):
    """Discrete annual probability of sale given survival to year t."""
    h = weibull_hazard(max(t, 0.5), k, lam)
    return 1 - np.exp(-h)

# Verify: implied annual turnover rate should match HMDA purchase rate of 4.78%
# Average hazard over the tenure distribution
avg_hazard = np.average(
    [weibull_hazard(t, shape_k, scale_lam) for t in tenure_midpoints],
    weights=tenure_proportions
)
print(f"\nCalibration check:")
print(f"  HMDA observed annual purchase rate: 4.78%")
print(f"  Weibull implied avg hazard:         {avg_hazard:.2%}")

# ── LEAP selection effect correction ─────────────────────────────────────────
# Homeowners who just received a LEAP loan are less likely to sell immediately.
# We apply a dampening factor for the first 5 years post-loan.
SELECTION_EFFECT_YEARS  = 5      # years of reduced sale probability
SELECTION_EFFECT_FACTOR = 0.40   # LEAP borrowers sell at 40% of normal rate

print(f"\nLEAP selection effect:")
print(f"  Duration: {SELECTION_EFFECT_YEARS} years post-loan")
print(f"  Reduction factor: {SELECTION_EFFECT_FACTOR:.0%} of normal hazard")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Calibrate Refinancing/HELOC Hazard from HMDA Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 2: Refi/HELOC Hazard Calibration (HMDA 2022) ===\n")

# ── Observed rates from HMDA cleaning output ──────────────────────────────────
HMDA_REFI_RATE_OBSERVED  = 0.0233   # refi rate at ~7.4% avg rate environment
HMDA_HELOC_RATE_OBSERVED = 0.0075   # HELOC rate at ~7.4% avg rate environment
HMDA_MEAN_RATE_OBSERVED  = 7.44     # mean refi interest rate in dataset (%)

print(f"HMDA observed rates (2022, high-rate environment):")
print(f"  Annual refi rate:  {HMDA_REFI_RATE_OBSERVED:.2%}")
print(f"  Annual HELOC rate: {HMDA_HELOC_RATE_OBSERVED:.2%}")
print(f"  Mean refi rate:    {HMDA_MEAN_RATE_OBSERVED:.2f}%")

# ── Rate sensitivity model ────────────────────────────────────────────────────
# Refi activity is strongly rate-dependent. We model:
# lambda_refi(r) = lambda_base * exp(-alpha * (r - r_star))
# where r_star is the "threshold" rate below which refinancing makes sense.
#
# Calibration: at r = 7.44% (observed), lambda = 2.33%
# At r = 3.5% (2020-2021 low), national refi rate was ~8-10%
# We calibrate alpha to match these two endpoints.

R_STAR          = 4.0    # % — threshold below which refi becomes attractive
R_LOW           = 3.5    # % — low rate scenario
LAMBDA_REFI_LOW = 0.08   # 8% annual refi rate in low-rate environment (national obs)

# Solve for alpha: LAMBDA_REFI_LOW = HMDA_REFI_RATE_OBSERVED * exp(-alpha*(R_LOW - R_STAR))
# But we need to anchor differently: use observed as the base
# lambda(r) = lambda_observed * exp(-alpha * (r - r_observed))
alpha_refi = np.log(LAMBDA_REFI_LOW / HMDA_REFI_RATE_OBSERVED) / \
             (HMDA_MEAN_RATE_OBSERVED - R_LOW)

# Same for HELOC — slightly less rate sensitive than cash-out refi
alpha_heloc = alpha_refi * 0.7

print(f"\nRate sensitivity parameters:")
print(f"  Refi alpha:  {alpha_refi:.4f}")
print(f"  HELOC alpha: {alpha_heloc:.4f}")

def refi_rate(r, base=HMDA_REFI_RATE_OBSERVED, alpha=None):
    """Annual refinancing probability as function of current rate r (%)."""
    if alpha is None:
        alpha = alpha_refi
    rate = base * np.exp(-alpha * (r - HMDA_MEAN_RATE_OBSERVED))
    return np.clip(rate, 0.005, 0.20)   # floor at 0.5%, cap at 20%

def heloc_rate(r, base=HMDA_HELOC_RATE_OBSERVED, alpha=None):
    """Annual HELOC probability as function of current rate r (%)."""
    if alpha is None:
        alpha = alpha_heloc
    rate = base * np.exp(-alpha * (r - HMDA_MEAN_RATE_OBSERVED))
    return np.clip(rate, 0.002, 0.10)

# Validate rate sensitivity
print(f"\nRate sensitivity validation:")
for r_test in [3.0, 4.0, 5.0, 6.0, 7.44, 8.5]:
    print(f"  r = {r_test:.1f}%  →  refi rate: {refi_rate(r_test):.2%}  "
          f"HELOC rate: {heloc_rate(r_test):.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Interest Rate Process (Vasicek Model)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 3: Interest Rate Process (Vasicek) ===\n")

# Vasicek: dr_t = kappa*(r_bar - r_t)*dt + sigma_r*dW_t
# Parameters calibrated to Fed Funds rate history and forward expectations
VASICEK_KAPPA  = 0.30    # mean reversion speed (moderate)
VASICEK_R_BAR  = 4.50    # long-run mean rate (%)
VASICEK_SIGMA  = 0.80    # rate volatility (% per year)
VASICEK_R0     = 5.25    # starting rate 2026 (current Fed Funds approximately)
DT             = 1.0     # annual time step

# Three rate scenarios
RATE_SCENARIOS = {
    'Low rates':   {'r0': 5.25, 'r_bar': 3.50, 'kappa': 0.40, 'sigma': 0.60},
    'Base rates':  {'r0': 5.25, 'r_bar': 4.50, 'kappa': 0.30, 'sigma': 0.80},
    'High rates':  {'r0': 5.25, 'r_bar': 5.50, 'kappa': 0.20, 'sigma': 0.70},
}

def simulate_rate_path(r0, r_bar, kappa, sigma, n_years, dt=1.0):
    """
    Simulate one Vasicek interest rate path.
    Returns array of annual rates from year 1 to n_years.
    """
    rates = np.zeros(n_years)
    r_t   = r0
    for t in range(n_years):
        dW    = np.random.normal(0, np.sqrt(dt))
        dr    = kappa * (r_bar - r_t) * dt + sigma * dW
        r_t   = max(r_t + dr, 0.5)   # floor at 0.5%
        rates[t] = r_t
    return rates

# Show sample paths for each scenario
n_years = PROJECTION_END - PROGRAM_START + 1
print(f"Sample rate paths (2026-2055):")
print(f"{'Year':<8}", end='')
for s in RATE_SCENARIOS:
    print(f"{s:>15}", end='')
print()
print("-" * 53)

sample_paths = {}
for scenario, params in RATE_SCENARIOS.items():
    np.random.seed(99)
    path = simulate_rate_path(
        params['r0'], params['r_bar'],
        params['kappa'], params['sigma'], n_years
    )
    sample_paths[scenario] = path

for i, year in enumerate(range(PROGRAM_START, min(PROGRAM_START + 12, PROJECTION_END + 1))):
    print(f"{year:<8}", end='')
    for scenario in RATE_SCENARIOS:
        print(f"{sample_paths[scenario][i]:>14.2f}%", end='')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Competing Risks Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n=== Module 4: Competing Risks Monte Carlo ({N_SIMULATIONS:,} runs) ===\n")

# Non-repayment rates by property type (from Phase 3)
NON_REPAYMENT_RATE = 0.1157   # blended portfolio rate from Phase 3

def simulate_single_run_p4(rate_path, k, lam):
    """
    Fully vectorized Phase 4 simulation.
    Returns (peak_funding_gap, total_outflow, total_non_repay, trigger_counts).
    """
    # ── Draw stochastic inputs ─────────────────────────────────────────────
    n_total_s = max(int(np.random.normal(N_total_mean, N_total_sigma)), 1000)
    rho_s     = np.random.beta(beta_alpha, beta_beta)

    n_voluntary = int(n_total_s * rho_s)
    n_breakage  = int(n_total_s * annual_breakage_rate *
                      (PROGRAM_END - PROGRAM_START + 1))
    n_proactive = int(n_total_s * annual_proactive_rate *
                      (PROGRAM_END - PROGRAM_START + 1))
    n_part_s    = n_voluntary + n_breakage + n_proactive

    # ── Replacement schedule ───────────────────────────────────────────────
    sched_rows = []
    cumulative = 0
    for year, pct in REPLACEMENT_SCHEDULE.items():
        loans_issued = min(n_part_s * pct, n_part_s - cumulative)
        cumulative  += loans_issued
        n_int        = max(int(loans_issued), 1)
        avg_loan     = min(
            np.mean(np.random.lognormal(mu_cost, sigma_cost, size=n_int)) *
            (1 + COST_ESCALATION) ** (year - BASE_YEAR),
            LOAN_CAP
        )
        sched_rows.append({
            'Year':               year,
            'Loans Issued':       int(loans_issued),
            'Avg Loan Amount':    avg_loan,
            'Annual Outflow ($)': loans_issued * avg_loan,
        })

    sched     = pd.DataFrame(sched_rows)
    total_out = sched['Annual Outflow ($)'].sum()

    # ── Vectorized competing risks ─────────────────────────────────────────
    inflow_by_year  = {y: 0.0 for y in range(PROGRAM_START, PROJECTION_END + 1)}
    total_non_repay = 0.0
    trigger_counts  = {'sale': 0, 'refi': 0, 'heloc': 0, 'never': 0}

    for _, row in sched.iterrows():
        issue_year   = int(row['Year'])
        n_loans      = int(row['Loans Issued'])
        loan_amt     = row['Avg Loan Amount']

        if n_loans == 0:
            continue

        # ── Step 1: Permanent non-repayment ───────────────────────────────
        nr_mask      = np.random.uniform(0, 1, n_loans) < NON_REPAYMENT_RATE
        n_never      = nr_mask.sum()
        n_repaying   = n_loans - n_never
        total_non_repay += n_never * loan_amt
        trigger_counts['never'] += int(n_never)

        if n_repaying == 0:
            continue

        # ── Step 2: For repaying loans, find repayment year ───────────────
        # Simulate year by year vectorized across all repaying loans
        # Each loan has a state: repaid (True/False)
        repaid      = np.zeros(n_repaying, dtype=bool)
        repay_years = np.full(n_repaying, PROJECTION_END + 1)
        triggers_arr = np.full(n_repaying, 'never', dtype=object)
        tenure      = np.zeros(n_repaying)

        for year in range(issue_year, PROJECTION_END + 1):
            still_out = ~repaid
            if not still_out.any():
                break

            tenure[still_out] += 1
            rate_idx = min(year - PROGRAM_START, len(rate_path) - 1)
            r_t      = rate_path[rate_idx]

            # Sale hazard — vectorized
            t_arr    = tenure[still_out]
            h_sale   = (k / lam) * (t_arr / lam) ** (k - 1)
            h_sale   = 1 - np.exp(-h_sale)
            # Apply LEAP selection effect
            sel_mask = t_arr <= SELECTION_EFFECT_YEARS
            h_sale[sel_mask] *= SELECTION_EFFECT_FACTOR

            # Refi and HELOC hazard — rate sensitive, with 2-year seasoning
            # Loans must be outstanding at least 2 years before refi/HELOC eligible
            seasoned    = t_arr >= 2
            h_refi_vec  = np.where(seasoned, refi_rate(r_t),  0.0)
            h_heloc_vec = np.where(seasoned, heloc_rate(r_t), 0.0)

            # Draw competing events
            n_out     = still_out.sum()
            u_sale    = np.random.uniform(0, 1, n_out)
            u_refi    = np.random.uniform(0, 1, n_out)
            u_heloc   = np.random.uniform(0, 1, n_out)

            sale_fire  = u_sale  < h_sale
            refi_fire  = u_refi  < h_refi_vec
            heloc_fire = u_heloc < h_heloc_vec
            any_fire   = sale_fire | refi_fire | heloc_fire

            # Assign trigger type (priority: sale > refi > heloc if multiple)
            fired_trigger = np.full(n_out, '', dtype=object)
            fired_trigger[heloc_fire] = 'heloc'
            fired_trigger[refi_fire]  = 'refi'
            fired_trigger[sale_fire]  = 'sale'

            # Update repaid status
            out_indices          = np.where(still_out)[0]
            newly_repaid_local   = np.where(any_fire)[0]
            newly_repaid_global  = out_indices[newly_repaid_local]

            repaid[newly_repaid_global]      = True
            repay_years[newly_repaid_global] = year
            triggers_arr[newly_repaid_global] = fired_trigger[newly_repaid_local]

        # ── Step 3: Record inflows ─────────────────────────────────────────
        for i in range(n_repaying):
            ry = int(repay_years[i])
            tg = triggers_arr[i]
            trigger_counts[tg] = trigger_counts.get(tg, 0) + 1
            if ry <= PROJECTION_END:
                inflow_by_year[ry] += loan_amt

    # ── Cash flow and peak gap ─────────────────────────────────────────────
    all_years       = list(range(PROGRAM_START, PROJECTION_END + 1))
    outflow_by_year = dict(zip(sched['Year'], sched['Annual Outflow ($)']))
    cum_out = cum_in = 0
    net_positions   = []

    for year in all_years:
        outflow  = outflow_by_year.get(year, 0)
        inflow   = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        net_positions.append(cum_in - cum_out)

    return abs(min(net_positions)), total_out, total_non_repay, trigger_counts

# ── Run simulation for each rate scenario ─────────────────────────────────────
scenario_results = {}

for scenario_name, params in RATE_SCENARIOS.items():
    print(f"Running: {scenario_name}...", flush=True)
    gaps         = np.zeros(N_SIMULATIONS)
    outflows     = np.zeros(N_SIMULATIONS)
    non_repays   = np.zeros(N_SIMULATIONS)
    all_triggers = {'sale': 0, 'refi': 0, 'heloc': 0, 'never': 0}

    for s in range(N_SIMULATIONS):
        rate_path = simulate_rate_path(
            params['r0'], params['r_bar'],
            params['kappa'], params['sigma'],
            n_years
        )
        gap, out, nr, triggers = simulate_single_run_p4(
            rate_path, shape_k, scale_lam
        )
        gaps[s]       = gap
        outflows[s]   = out
        non_repays[s] = nr
        for k_t, v in triggers.items():
            all_triggers[k_t] += v

        if (s + 1) % 1000 == 0:
            print(f"  Completed {s+1:>6,} / {N_SIMULATIONS:,}...", flush=True)

    scenario_results[scenario_name] = {
        'gaps':       gaps,
        'outflows':   outflows,
        'non_repays': non_repays,
        'triggers':   all_triggers,
    }
    print(f"  Done. 95th pct: ${np.percentile(gaps, 95):,.0f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Results
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 5: Phase 4 Results ===\n")

print(f"--- Competing Risks: 95th Percentile Capital by Rate Scenario ---\n")
print(f"{'Scenario':<15} {'Mean':>14} {'Median':>14} {'95th Pct':>14} "
      f"{'99th Pct':>14}")
print("-" * 75)

for scenario_name, res in scenario_results.items():
    gaps = res['gaps']
    print(f"{scenario_name:<15} "
          f"${np.mean(gaps):>13,.0f} "
          f"${np.percentile(gaps,50):>13,.0f} "
          f"${np.percentile(gaps,95):>13,.0f} "
          f"${np.percentile(gaps,99):>13,.0f}")

print(f"\n--- Repayment Trigger Breakdown (Base Rate Scenario) ---\n")
base_triggers = scenario_results['Base rates']['triggers']
total_triggers = sum(base_triggers.values())
if total_triggers > 0:
    for trigger, count in base_triggers.items():
        pct = count / total_triggers
        print(f"  {trigger:<10} {count:>8,}  ({pct:.1%})")

print(f"\n--- Full Model Comparison ---\n")
print(f"{'Model':<45} {'95th Pct Capital':>18}")
print("-" * 65)
print(f"{'Phase 1 base (deterministic, L=12)':<45} ${gap_base:>17,.0f}")
print(f"{'Phase 1 optimistic (deterministic, L=7)':<45} ${gap_opt:>17,.0f}")
print(f"{'Phase 2 primary (pilot breakage)':<45} ${pct_95:>17,.0f}")

base_p95 = np.percentile(scenario_results['Base rates']['gaps'], 95)
low_p95  = np.percentile(scenario_results['Low rates']['gaps'],  95)
high_p95 = np.percentile(scenario_results['High rates']['gaps'], 95)

print(f"{'Phase 4 low rate scenario (95th pct)':<45} ${low_p95:>17,.0f}")
print(f"{'Phase 4 base rate scenario (95th pct)':<45} ${base_p95:>17,.0f}")
print(f"{'Phase 4 high rate scenario (95th pct)':<45} ${high_p95:>17,.0f}")

print(f"\n--- Interest Rate Impact on Funding Requirement ---\n")
print(f"  Moving from high to low rate environment saves: "
      f"${high_p95 - low_p95:>12,.0f}")
print(f"  As % of base rate requirement:                  "
      f"{(high_p95 - low_p95)/base_p95:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 6: Generating Visualizations ===\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'LEAP Program — Phase 4: Competing Risks Repayment Model\n'
    'Calibrated to Columbus HMDA (2022) and Census ACS B25026',
    fontsize=13, fontweight='bold'
)

colors = {
    'Low rates':  'steelblue',
    'Base rates': 'mediumseagreen',
    'High rates': 'coral',
}

# ── Plot 1: Distribution of required capital by rate scenario ─────────────────
ax1 = axes[0, 0]
for scenario_name, res in scenario_results.items():
    ax1.hist(res['gaps'] / 1e6, bins=60,
             color=colors[scenario_name], alpha=0.5,
             edgecolor='white', label=scenario_name)
    ax1.axvline(np.percentile(res['gaps'], 95) / 1e6,
                color=colors[scenario_name], linewidth=2, linestyle='--')
ax1.axvline(pct_95 / 1e6, color='black', linewidth=2, linestyle=':',
            label=f'Phase 2 primary: ${pct_95/1e6:.1f}M')
ax1.set_title('Required Capital by Interest Rate Scenario')
ax1.set_xlabel('Required Capital ($M)')
ax1.set_ylabel('Frequency')
ax1.legend(fontsize=7)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# ── Plot 2: Sample interest rate paths ───────────────────────────────────────
ax2 = axes[0, 1]
years = list(range(PROGRAM_START, PROGRAM_START + 30))
np.random.seed(42)
for scenario_name, params in RATE_SCENARIOS.items():
    # Plot 10 sample paths per scenario
    for _ in range(10):
        path = simulate_rate_path(
            params['r0'], params['r_bar'],
            params['kappa'], params['sigma'], 30
        )
        ax2.plot(years, path, color=colors[scenario_name], alpha=0.2, linewidth=0.8)
    # Plot mean path
    mean_path = np.mean([
        simulate_rate_path(params['r0'], params['r_bar'],
                           params['kappa'], params['sigma'], 30)
        for _ in range(100)
    ], axis=0)
    ax2.plot(years, mean_path, color=colors[scenario_name],
             linewidth=2.5, label=f'{scenario_name} (mean)')
ax2.axvline(PROGRAM_END, color='gray', linestyle=':', linewidth=1.5,
            label='Program end (2037)')
ax2.set_title('Simulated Interest Rate Paths (Vasicek Model)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Interest Rate (%)')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Repayment trigger breakdown ───────────────────────────────────────
ax3 = axes[1, 0]
scenarios_list = list(scenario_results.keys())
trigger_types  = ['sale', 'refi', 'heloc', 'never']
trigger_colors = ['steelblue', 'mediumseagreen', 'coral', 'salmon']
trigger_labels = ['Property Sale', 'Refinancing', 'HELOC', 'Non-Repayment']

bottoms = np.zeros(len(scenarios_list))
for t_type, t_color, t_label in zip(trigger_types, trigger_colors, trigger_labels):
    vals = []
    for scenario_name in scenarios_list:
        triggers = scenario_results[scenario_name]['triggers']
        total    = sum(triggers.values())
        vals.append(triggers[t_type] / total * 100 if total > 0 else 0)
    ax3.bar(scenarios_list, vals, bottom=bottoms,
            color=t_color, label=t_label, alpha=0.8)
    bottoms += np.array(vals)

ax3.set_title('Repayment Trigger Breakdown by Rate Scenario')
ax3.set_ylabel('% of Loans')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# ── Plot 4: Phase comparison — all phases at 95th percentile ─────────────────
ax4 = axes[1, 1]
phase_labels = [
    'Phase 1\nBase',
    'Phase 2\nPrimary',
    'Phase 4\nLow Rates',
    'Phase 4\nBase Rates',
    'Phase 4\nHigh Rates',
]
phase_values = [gap_base, pct_95, low_p95, base_p95, high_p95]
phase_colors = ['green', 'steelblue', 'steelblue', 'mediumseagreen', 'coral']
bars = ax4.bar(phase_labels, [v / 1e6 for v in phase_values],
               color=phase_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, phase_values):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             f'${val/1e6:.1f}M',
             ha='center', va='bottom', fontsize=8, fontweight='bold')
ax4.set_title('95th Percentile Capital: All Phases Compared')
ax4.set_ylabel('Required Capital ($M)')
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase4_results.png', dpi=150, bbox_inches='tight')
print("Saved: phase4_results.png")
plt.close()

print("\nPhase 4 complete.")
