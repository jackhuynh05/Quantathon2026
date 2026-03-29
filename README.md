# Quantathon2026
## LEAP Program Quantitative Funding Framework

A five-phase quantitative model estimating the capital reserves required
for Columbus, Ohio's Lead Exposure Assistance Program (LEAP).

Built for the 2026 SIAM-MTI Quantathon by Jack Huynh, Maddox Roy, and
Pranshu Shrivastava.

---

## The Question

How much must Columbus set aside in 2026 to keep LEAP solvent through 2037?

**Answer:** $165M baseline · $202.5M accounting for model uncertainty · 95% CI: [$93M, $211M]

---

## Files
```
clean_hmda_data.py     # Cleaned the raw HMDA file pulled from CFPB
config.py              # Shared constants and parameters
estimate_properties.py # Point estimate for number of hazerdous service lines
phase1.py              # Deterministic baseline
phase2.py              # Monte Carlo simulation (10,000 runs)
phase3.py              # Property-level micro-simulation (1,000 runs)
phase4.py              # Competing risks + interest rate model
phase5.py              # Weighted meta-distribution and final CI
```


**Data:** Program Excel file, cross-table CSV, Census B25026, HMDA
(CFPB 2024), and SIAM-MTI simulated property dataset.

---

Each phase imports from the prior one. Phases suppress output when
imported and only print when run directly.

---

## Results Summary

| Phase | What It Does | 95th Pct |
|-------|-------------|----------|
| 1 | Deterministic baseline | $113.5M |
| 2 | Monte Carlo, three breakage scenarios | $134M–$218M |
| 3 | Property-level heterogeneity, non-repayment | $102M* |
| 4 | Competing risks, interest rate sensitivity | $90M–$102M |
| 5 | Weighted 95% CI across all scenarios | $202.5M |

\* Voluntary channel only — not directly comparable to Phase 2

---

## Acknowledgments

Property dataset by **SIAM-MTI**. Program data from the **City of
Columbus**. HMDA from **CFPB**. Tenure data from **U.S. Census Bureau**.
