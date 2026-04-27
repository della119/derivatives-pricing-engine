"""
Risk Metrics module
====================
Value-at-Risk (VaR) and Expected Shortfall (ES/CVaR) calculations
for option positions.

Methods:
- Delta-Normal (parametric, fast)
- Monte Carlo (full revaluation)
- Historical Simulation (empirical distribution)
"""
from .var import (
    delta_normal_var,
    monte_carlo_var,
    historical_var,
    compute_all_var_methods,
)

__all__ = [
    "delta_normal_var",
    "monte_carlo_var",
    "historical_var",
    "compute_all_var_methods",
]
