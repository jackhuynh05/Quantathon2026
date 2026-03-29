# config.py
# ── Shared constants across Phase 1 and Phase 2 ───────────────────────────────

# File paths
CROSS_TABLE_PATH = 'cross_table.csv'
EXCEL_PATH       = 'CWP - LEAP Data v2 3.25.26.xlsx'

# Loan parameters
AVG_LOAN_BASE    = 7582
LOAN_CAP         = 10_000
COST_ESCALATION  = 0.03
BASE_YEAR        = 2026
DISCOUNT_RATE    = 0.035

# Program window
PROGRAM_START    = 2026
PROGRAM_END      = 2037
PROJECTION_END   = 2055

# Repayment lags
REPAYMENT_LAG_BASE       = 12
REPAYMENT_LAG_OPTIMISTIC = 7

# Pilot funnel
PILOT_INTERESTED  = 127
PILOT_APPS_SENT   = 90
PILOT_APPLIED     = 82
PILOT_DROPPED     = 18
PILOT_REJECTED    = 2
PILOT_COMPLETED   = 42
PILOT_IN_PROGRESS = 31

# Replacement schedule percentages
REPLACEMENT_SCHEDULE = {
    2026: 0.02, 2027: 0.04, 2028: 0.07,
    2029: 0.10, 2030: 0.10, 2031: 0.10,
    2032: 0.10, 2033: 0.10, 2034: 0.10,
    2035: 0.10, 2036: 0.10, 2037: 0.10,
}

# Sensitivity scenarios
PARTICIPATION_SCENARIOS = {
    'Floor (33.07%)':      PILOT_COMPLETED / PILOT_INTERESTED,
    'Low (42.00%)':        0.42,
    'Base (51.22%)':       PILOT_COMPLETED / PILOT_APPLIED,
    'High (70.00%)':       0.70,
    'Optimistic (89.02%)': (PILOT_COMPLETED + PILOT_IN_PROGRESS) / PILOT_APPLIED,
}

LAG_SCENARIOS = {
    'L = 7 years':  7,
    'L = 12 years': 12,
    'L = 20 years': 20,
}
