"""
Risk Metrics module
====================
Value-at-Risk (VaR), Expected Shortfall (ES), and Credit Valuation
Adjustment (CVA / DVA / BCVA) for option positions.
"""
from .var import (
    delta_normal_var,
    monte_carlo_var,
    historical_var,
    compute_all_var_methods,
)
from .cva import (
    CreditCurve,
    hazard_rate_from_cds,
    survival_probability,
    marginal_default_probability,
    simulate_exposure_profile,
    compute_cva,
    compute_dva,
    compute_cva_for_option,
)

__all__ = [
    # VaR
    "delta_normal_var",
    "monte_carlo_var",
    "historical_var",
    "compute_all_var_methods",
    # CVA
    "CreditCurve",
    "hazard_rate_from_cds",
    "survival_probability",
    "marginal_default_probability",
    "simulate_exposure_profile",
    "compute_cva",
    "compute_dva",
    "compute_cva_for_option",
]
