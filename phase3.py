# phase3.py
# ── LEAP Program: Phase 3 Property-Level Micro-Simulation ─────────────────────
# Uses simulated property-level data to introduce heterogeneity across
# property types, participation probabilities, repayment lags, and
# non-repayment rates. Results are presented as structural illustrations
# given the simulated nature of the underlying dataset.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
)
from phase2 import (
    pct_95,
    mu_cost,
    sigma_cost,
    mu_lag,
    sigma_lag,
    annual_breakage_rate,
    annual_proactive_rate,
)

N_SIMULATIONS = 1_000   # property-level is slower — 1k is sufficient
np.random.seed(42)

print("=" * 70)
print("PHASE 3: Property-Level Micro-Simulation")
print("NOTE: Property dataset is simulated. Results are structural")
print("      illustrations, not empirical estimates.")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Load and Validate Property Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 1: Property Data ===\n")

prop = pd.read_csv('property_data.csv')

# Flag hazardous properties — both lead and galvanized are eligible
HAZARDOUS_MATERIALS = {'lead', 'galvanized'}
prop['is_hazardous'] = prop['line_material'].isin(HAZARDOUS_MATERIALS)

# Filter to eligible properties only
eligible = prop[prop['is_hazardous']].copy().reset_index(drop=True)

print(f"Total properties in dataset:     {len(prop):>8,}")
print(f"Hazardous (lead or galvanized):  {prop['is_hazardous'].sum():>8,}")
print(f"  - Galvanized:                  {(prop['line_material']=='galvanized').sum():>8,}")
print(f"  - Lead:                        {(prop['line_material']=='lead').sum():>8,}")
print(f"Non-hazardous:                   {(~prop['is_hazardous']).sum():>8,}")
print(f"\nEligible universe (dataset):     {len(eligible):>8,}")

print(f"\nEligible by property type:")
print(eligible['property_type'].value_counts().to_string())

print(f"\nEligible owner-occupied rate:    "
      f"{eligible['owner_occupied'].mean():.2%}")
print(f"Eligible mean assessed value:    "
      f"${eligible['assessed_value'].mean():>10,.0f}")
print(f"Eligible mean ownership length:  "
      f"{eligible['ownership_length'].mean():.1f} years")
print(f"Eligible mean lead risk score:   "
      f"{eligible['lead_risk'].mean():.3f}")
print(f"Eligible mean replacement cost:  "
      f"${eligible['estimated_replacement_cost'].mean():>10,.0f}")

# ── Scaling factor: dataset has 6,057 eligible properties ─────────────────────
# program universe is 24,416 — scale all outputs accordingly
SCALE_FACTOR = N_total / len(eligible)
print(f"\nScale factor (N_total / dataset eligible): {SCALE_FACTOR:.3f}")
print(f"All outputs will be scaled by {SCALE_FACTOR:.3f} "
      f"to match program universe of {N_total:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Participation Model — Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 2: Participation Model (Logistic Regression) ===\n")

# Generate synthetic participation labels consistent with pilot rate (51.22%)
# Participation is higher for: owner-occupied, higher lead risk, older
# properties, lower assessed value, longer ownership
np.random.seed(123)

eligible['participation_score'] = (
      0.8 * eligible['owner_occupied']
    + 0.6 * eligible['lead_risk']
    - 0.3 * (eligible['assessed_value'] / eligible['assessed_value'].max())
    + 0.4 * (eligible['ownership_length'] / eligible['ownership_length'].max())
    - 0.2 * ((eligible['year_built'] - 1900) /
              (eligible['year_built'].max() - 1900))
)

score_mean  = eligible['participation_score'].mean()
score_std   = eligible['participation_score'].std()
target_rate = PILOT_COMPLETED / PILOT_APPLIED   # 51.22%

eligible['p_participate_raw'] = stats.norm.cdf(
    (eligible['participation_score'] - score_mean) / score_std
)
eligible['p_participate'] = (
    eligible['p_participate_raw'] /
    eligible['p_participate_raw'].mean() * target_rate
).clip(0, 1)

eligible['participated'] = (
    np.random.uniform(0, 1, len(eligible)) < eligible['p_participate']
).astype(int)

actual_rate = eligible['participated'].mean()
print(f"Synthetic participation labels generated:")
print(f"  Target rate (pilot):    {target_rate:.2%}")
print(f"  Achieved rate:          {actual_rate:.2%}")

# Fit logistic regression
features = [
    'owner_occupied',
    'lead_risk',
    'assessed_value',
    'ownership_length',
    'year_built',
]

X        = eligible[features].values
y        = eligible['participated'].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_scaled, y)

eligible['p_participate_lr'] = lr_model.predict_proba(X_scaled)[:, 1]

print(f"\nLogistic regression fitted:")
print(f"  Features: {features}")
print(f"\n  Coefficients:")
for feat, coef in zip(features, lr_model.coef_[0]):
    direction = '↑' if coef > 0 else '↓'
    print(f"    {feat:<25} {coef:>8.4f}  {direction}")

print(f"\n  Mean predicted participation: "
      f"{eligible['p_participate_lr'].mean():.2%}")
print(f"  Std predicted participation:  "
      f"{eligible['p_participate_lr'].std():.4f}")

print(f"\nPredicted participation rate by property type:")
print(eligible.groupby('property_type')['p_participate_lr'].mean()
      .apply(lambda x: f"{x:.2%}").to_string())

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Repayment Survival Model
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 3: Repayment Survival Model ===\n")

repayment_params = {
    ('single_family', 1): {'mean': 14.0, 'std': 5.0},
    ('single_family', 0): {'mean': 10.0, 'std': 4.0},
    ('duplex',        1): {'mean': 12.0, 'std': 5.0},
    ('duplex',        0): {'mean':  9.0, 'std': 4.0},
    ('multi_family',  1): {'mean': 11.0, 'std': 4.5},
    ('multi_family',  0): {'mean':  8.0, 'std': 3.5},
}

def get_lag_params(prop_type, owner_occ):
    key    = (prop_type, int(owner_occ))
    params = repayment_params.get(key, {'mean': 12.0, 'std': 5.0})
    mean, std = params['mean'], params['std']
    sigma  = np.sqrt(np.log(1 + std**2 / mean**2))
    mu     = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

print("Repayment lag parameters by property type and occupancy:")
print(f"\n{'Property Type':<20} {'Owner Occ':>10} {'Mean (yr)':>10} "
      f"{'Std (yr)':>10} {'Median (yr)':>12}")
print("-" * 65)
for (pt, oc), params in repayment_params.items():
    mu, sigma = get_lag_params(pt, oc)
    print(f"  {pt:<18} {oc:>10} {params['mean']:>10.1f} "
          f"{params['std']:>10.1f} {np.exp(mu):>12.1f}")

non_repayment_rates = {
    ('single_family', 1): 0.10,
    ('single_family', 0): 0.08,
    ('duplex',        1): 0.15,
    ('duplex',        0): 0.10,
    ('multi_family',  1): 0.20,
    ('multi_family',  0): 0.12,
}

print(f"\nNon-repayment rates by property type and occupancy:")
print(f"\n{'Property Type':<20} {'Owner Occ':>10} {'Non-Repayment %':>16}")
print("-" * 48)
for (pt, oc), rate in non_repayment_rates.items():
    print(f"  {pt:<18} {oc:>10} {rate:>15.1%}")

eligible['non_repayment_rate'] = eligible.apply(
    lambda r: non_repayment_rates.get(
        (r['property_type'], int(r['owner_occupied'])), 0.12
    ), axis=1
)

blended_non_repayment = eligible['non_repayment_rate'].mean()
print(f"\nBlended non-repayment rate (portfolio):  {blended_non_repayment:.2%}")

# Pre-compute lag parameters per property for vectorized simulation
eligible['mu_lag']    = eligible.apply(
    lambda r: get_lag_params(r['property_type'], r['owner_occupied'])[0], axis=1
)
eligible['sigma_lag'] = eligible.apply(
    lambda r: get_lag_params(r['property_type'], r['owner_occupied'])[1], axis=1
)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Property-Level Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n=== Module 4: Property-Level Monte Carlo ({N_SIMULATIONS:,} runs) ===\n")

def simulate_property_level(eligible, scale_factor,
                             program_start=PROGRAM_START,
                             program_end=PROGRAM_END,
                             projection_end=PROJECTION_END):
    """
    Runs one property-level Monte Carlo simulation.
    Returns (peak_funding_gap, total_outflow, total_non_repayment).
    """
    n_props = len(eligible)

    # ── Step 1: Draw participation for each property ──────────────────────────
    u            = np.random.uniform(0, 1, n_props)
    participates = (u < eligible['p_participate_lr'].values).astype(bool)

    n_part = participates.sum()
    if n_part == 0:
        return 0.0, 0.0, 0.0

    part_props = eligible[participates].copy().reset_index(drop=True)

    # ── Step 2: Draw replacement costs ───────────────────────────────────────
    costs_raw    = np.random.lognormal(mu_cost, sigma_cost, size=n_part)
    costs_capped = np.minimum(costs_raw, LOAN_CAP)
    part_props['loan_amount'] = costs_capped

    # ── Step 3: Assign replacement year via mandate schedule ──────────────────
    years      = list(range(program_start, program_end + 1))
    year_probs = np.array([REPLACEMENT_SCHEDULE[y] for y in years])
    year_probs = year_probs / year_probs.sum()
    drawn_years = np.random.choice(years, size=n_part, p=year_probs)
    part_props['replacement_year'] = drawn_years

    # Apply cost escalation vectorized
    escalation = np.array([
        (1 + COST_ESCALATION) ** (yr - BASE_YEAR) for yr in drawn_years
    ])
    escalated = np.minimum(costs_capped * escalation, LOAN_CAP)
    part_props['loan_escalated'] = escalated
    part_props['scaled_loan']    = escalated * scale_factor

    # ── Step 4: Annual outflows ───────────────────────────────────────────────
    outflow_by_year = {y: 0.0 for y in range(program_start, projection_end + 1)}
    for yr in years:
        mask = part_props['replacement_year'] == yr
        outflow_by_year[yr] = part_props.loc[mask, 'scaled_loan'].sum()

    total_out = sum(outflow_by_year[y] for y in years)

    # ── Step 5: Repayment timing — fully vectorized ───────────────────────────
    inflow_by_year  = {y: 0.0 for y in range(program_start, projection_end + 1)}
    total_non_repay = 0.0

    # Draw non-repayment flags
    nr_draws  = np.random.uniform(0, 1, n_part)
    repays    = nr_draws >= part_props['non_repayment_rate'].values

    # Non-repayment losses
    total_non_repay = part_props.loc[~repays, 'scaled_loan'].sum()

    # Repaying properties — draw lags vectorized
    repaying = part_props[repays].copy()
    if len(repaying) > 0:
        lags = np.maximum(
            1,
            np.random.lognormal(
                repaying['mu_lag'].values,
                repaying['sigma_lag'].values
            ).astype(int)
        )
        repay_years = repaying['replacement_year'].values + lags

        for yr, amt in zip(repay_years, repaying['scaled_loan'].values):
            if yr <= projection_end:
                inflow_by_year[int(yr)] += amt

    # ── Step 6: Cash flow and peak funding gap ────────────────────────────────
    all_years     = list(range(program_start, projection_end + 1))
    cum_out = cum_in = 0
    net_positions   = []

    for year in all_years:
        outflow  = outflow_by_year.get(year, 0)
        inflow   = inflow_by_year.get(year, 0)
        cum_out += outflow
        cum_in  += inflow
        net_positions.append(cum_in - cum_out)

    peak_gap = abs(min(net_positions))
    return peak_gap, total_out, total_non_repay

# ── Run simulation ─────────────────────────────────────────────────────────────
print("Running property-level simulation...")

# Initialize arrays at N_SIMULATIONS — consistent throughout
results_gap_p3       = np.zeros(N_SIMULATIONS)
results_outflow_p3   = np.zeros(N_SIMULATIONS)
results_non_repay_p3 = np.zeros(N_SIMULATIONS)

for s in range(N_SIMULATIONS):
    gap_s, out_s, nr_s   = simulate_property_level(eligible, SCALE_FACTOR)
    results_gap_p3[s]    = gap_s
    results_outflow_p3[s] = out_s
    results_non_repay_p3[s] = nr_s

    if (s + 1) % 100 == 0:
        print(f"  Completed {s+1:>5,} / {N_SIMULATIONS:,} simulations...")

print(f"\nSimulation complete.")

# ── Diagnostic ────────────────────────────────────────────────────────────────
print(f"\nDiagnostic:")
print(f"  % of sims with gap = 0:     {(results_gap_p3 == 0).mean():.1%}")
print(f"  % of sims with outflow = 0: {(results_outflow_p3 == 0).mean():.1%}")
print(f"  Min non-zero gap:           "
      f"${results_gap_p3[results_gap_p3 > 0].min():,.0f}" 
      if (results_gap_p3 > 0).any() else "  No non-zero gaps found")
print(f"  Max gap:                    ${results_gap_p3.max():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Results and Comparison
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 5: Phase 3 Results ===\n")

p3_mean   = np.mean(results_gap_p3)
p3_median = np.percentile(results_gap_p3, 50)
p3_p95    = np.percentile(results_gap_p3, 95)
p3_p99    = np.percentile(results_gap_p3, 99)
p3_std    = np.std(results_gap_p3)
p3_p5     = np.percentile(results_gap_p3, 5)
p3_p25    = np.percentile(results_gap_p3, 25)
p3_p75    = np.percentile(results_gap_p3, 75)

nr_mean   = np.mean(results_non_repay_p3)
nr_p95    = np.percentile(results_non_repay_p3, 95)
out_mean  = np.mean(results_outflow_p3)
nr_pct    = nr_mean / out_mean if out_mean > 0 else 0

print(f"--- Required Starting Capital: Phase 3 ---\n")
print(f"  Mean:             ${p3_mean:>15,.0f}")
print(f"  Std deviation:    ${p3_std:>15,.0f}")
print(f"  5th percentile:   ${p3_p5:>15,.0f}")
print(f"  25th percentile:  ${p3_p25:>15,.0f}")
print(f"  Median (50th):    ${p3_median:>15,.0f}")
print(f"  75th percentile:  ${p3_p75:>15,.0f}")
print(f"  95th percentile:  ${p3_p95:>15,.0f}  ← recommended funding level")
print(f"  99th percentile:  ${p3_p99:>15,.0f}")

print(f"\n--- Non-Repayment Statistics ---\n")
print(f"  Mean permanent loss:          ${nr_mean:>15,.0f}")
print(f"  95th pct permanent loss:      ${nr_p95:>15,.0f}")
print(f"  As % of mean total outflow:   {nr_pct:>14.2%}")
print(f"  Blended non-repayment rate:   {blended_non_repayment:>14.2%}")

print(f"\n--- Phase 1 vs Phase 2 vs Phase 3 Comparison ---\n")
print(f"{'Model':<50} {'95th Pct Capital':>18}")
print("-" * 70)
print(f"{'Phase 1 base (deterministic, L=12)':<50} "
      f"${gap_base:>17,.0f}")
print(f"{'Phase 1 optimistic (deterministic, L=7)':<50} "
      f"${gap_opt:>17,.0f}")
print(f"{'Phase 2 primary (stochastic, pilot breakage)':<50} "
      f"${pct_95:>17,.0f}")
print(f"{'Phase 3 property-level (with non-repayment)':<50} "
      f"${p3_p95:>17,.0f}")
print(f"\n{'Phase 3 vs Phase 2 primary (95th pct)':<50} "
      f"${p3_p95 - pct_95:>+17,.0f} "
      f"({(p3_p95 - pct_95)/pct_95:+.1%})")

print(f"\n--- Total Program Outflow: Phase 3 ---\n")
print(f"  Mean outflow:           ${out_mean:>15,.0f}")
print(f"  Std deviation:          ${np.std(results_outflow_p3):>15,.0f}")
print(f"  5th percentile:         ${np.percentile(results_outflow_p3,5):>15,.0f}")
print(f"  95th percentile:        ${np.percentile(results_outflow_p3,95):>15,.0f}")
print(f"  Phase 1 total outflow:  ${total_outflow:>15,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Module 6: Generating Visualizations ===\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'LEAP Program — Phase 3: Property-Level Micro-Simulation\n'
    '(Simulated property data — structural illustration)',
    fontsize=13, fontweight='bold'
)

# ── Plot 1: Distribution of required starting capital ─────────────────────────
ax1 = axes[0, 0]
nonzero_gaps = results_gap_p3[results_gap_p3 > 0]
if len(nonzero_gaps) > 0:
    ax1.hist(nonzero_gaps / 1e6, bins=60, color='mediumseagreen',
             alpha=0.7, edgecolor='white', label='Phase 3 (property-level)')
ax1.axvline(p3_p95 / 1e6, color='red', linewidth=2, linestyle='--',
            label=f'95th pct: ${p3_p95/1e6:.1f}M')
ax1.axvline(p3_median / 1e6, color='orange', linewidth=2, linestyle='--',
            label=f'Median: ${p3_median/1e6:.1f}M')
ax1.axvline(gap_base / 1e6, color='green', linewidth=2, linestyle='-',
            label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax1.axvline(pct_95 / 1e6, color='steelblue', linewidth=2, linestyle=':',
            label=f'Phase 2 95th: ${pct_95/1e6:.1f}M')
ax1.set_title('Distribution of Required Starting Capital')
ax1.set_xlabel('Required Capital ($M)')
ax1.set_ylabel('Frequency')
ax1.legend(fontsize=7)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# ── Plot 2: CDF ───────────────────────────────────────────────────────────────
ax2 = axes[0, 1]
sorted_p3 = np.sort(results_gap_p3) / 1e6
cdf       = np.arange(1, N_SIMULATIONS + 1) / N_SIMULATIONS
ax2.plot(sorted_p3, cdf, color='mediumseagreen', linewidth=2.5,
         label='Phase 3 (property-level)')
ax2.axhline(0.95, color='red',    linestyle='--', linewidth=1.5,
            label='95th percentile')
ax2.axhline(0.50, color='orange', linestyle='--', linewidth=1.5,
            label='50th percentile')
ax2.axvline(p3_p95  / 1e6, color='red',       linestyle=':', linewidth=1)
ax2.axvline(pct_95  / 1e6, color='steelblue', linestyle=':', linewidth=1.5,
            label=f'Phase 2 95th: ${pct_95/1e6:.1f}M')
ax2.axvline(gap_base / 1e6, color='green',    linestyle=':', linewidth=1.5,
            label=f'Phase 1 base: ${gap_base/1e6:.1f}M')
ax2.set_title('CDF: Phase 3 Required Starting Capital')
ax2.set_xlabel('Required Capital ($M)')
ax2.set_ylabel('Cumulative Probability')
ax2.legend(fontsize=7)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.grid(True, alpha=0.3)

# ── Plot 3: Non-repayment distribution ───────────────────────────────────────
ax3 = axes[1, 0]
nonzero_nr = results_non_repay_p3[results_non_repay_p3 > 0]
if len(nonzero_nr) > 0:
    ax3.hist(nonzero_nr / 1e6, bins=60, color='salmon',
             alpha=0.7, edgecolor='white')
ax3.axvline(nr_mean / 1e6, color='darkred', linewidth=2, linestyle='-',
            label=f'Mean: ${nr_mean/1e6:.1f}M')
ax3.axvline(nr_p95 / 1e6, color='red', linewidth=2, linestyle='--',
            label=f'95th pct: ${nr_p95/1e6:.1f}M')
ax3.set_title('Distribution of Permanent Non-Repayment Loss')
ax3.set_xlabel('Non-Repayment Amount ($M)')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=8)
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# ── Plot 4: All phases comparison bar chart ───────────────────────────────────
ax4 = axes[1, 1]
scenarios = [
    'Phase 1\nBase',
    'Phase 1\nOptimistic',
    'Phase 2\nPrimary\n(95th pct)',
    'Phase 3\nProperty-Level\n(95th pct)',
]
values = [gap_base, gap_opt, pct_95, p3_p95]
bars   = ax4.bar(
    scenarios,
    [v / 1e6 for v in values],
    color=['green', 'purple', 'steelblue', 'mediumseagreen'],
    alpha=0.8, edgecolor='white', linewidth=1.5
)
for bar, val in zip(bars, values):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f'${val/1e6:.1f}M',
        ha='center', va='bottom', fontsize=8, fontweight='bold'
    )
ax4.set_title('Required Capital: All Phases Compared')
ax4.set_ylabel('Required Capital ($M)')
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase3_results.png', dpi=150, bbox_inches='tight')
print("Saved: phase3_results.png")
plt.close()

print("\nPhase 3 complete.")
