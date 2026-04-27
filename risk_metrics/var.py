"""
Value-at-Risk (VaR) for Option Positions
==========================================
Three methods are implemented:

1. Delta-Normal VaR  — Parametric approximation using Greeks (Delta + Gamma).
                       Fast but only accurate for small underlying moves and
                       linear/quadratic exposure.

2. Monte Carlo VaR   — Simulate underlying price paths and fully reprice the
                       option at the horizon.  Most accurate, captures full
                       non-linearity, but slowest.

3. Historical VaR    — Use historical or user-specified return distribution
                       of the underlying, reprice the option at each historical
                       scenario.  Captures empirical fat tails / skew.

All methods return a dict with:
    - VaR              : potential loss at the given confidence level (positive number)
    - ES (CVaR)        : expected loss conditional on exceeding VaR
    - PnL distribution : array of simulated/scenario P&L for plotting
    - method, confidence, horizon_days, n_scenarios

Conventions
-----------
- VaR and ES are reported as POSITIVE numbers representing the magnitude of loss
- Long position : holder loses if option value drops
- Position size is in contracts (1 contract = 1 underlying share unless specified)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ===================================================================
# Helper: scale volatility / drift to horizon
# ===================================================================
def _scale_to_horizon(sigma_annual: float, mu_annual: float, horizon_days: float):
    """Scale annual sigma & drift to the holding-period horizon."""
    h = horizon_days / 252.0  # business days
    sigma_h = sigma_annual * np.sqrt(h)
    mu_h = mu_annual * h
    return sigma_h, mu_h, h


# ===================================================================
# 1. Delta-Normal VaR (parametric)
# ===================================================================
def delta_normal_var(
    *,
    option_value: float,
    delta: float,
    gamma: float = 0.0,
    spot: float,
    sigma_annual: float,
    mu_annual: float = 0.0,
    horizon_days: float = 1.0,
    confidence: float = 0.95,
    position_size: float = 1.0,
    long: bool = True,
) -> dict:
    """
    Delta-Normal (with optional Gamma) VaR.

    Approximation:
        ΔP ≈ Δ·ΔS + ½·Γ·(ΔS)²

    where ΔS = S·(exp(μh + σh·Z) - 1) ≈ S·(μh + σh·Z + ½(σh·Z)²) under GBM.

    For pure Delta-Normal we ignore Gamma (set gamma=0):
        ΔP ~ N(Δ·S·μh, (Δ·S·σh)²)
        VaR_α = -(Δ·S·μh + Δ·S·σh·Φ⁻¹(1-α))
              = Δ·S·σh·Φ⁻¹(α) - Δ·S·μh   (for long position)

    With Gamma we use a delta-gamma normal approximation (Cornish-Fisher style)
    by simulating Z and computing ΔP analytically.
    """
    sigma_h, mu_h, _ = _scale_to_horizon(sigma_annual, mu_annual, horizon_days)

    sign = 1.0 if long else -1.0
    pos_delta = sign * delta * position_size
    pos_gamma = sign * gamma * position_size

    if abs(gamma) < 1e-12:
        # Pure Delta-Normal closed-form
        sigma_pnl = abs(pos_delta) * spot * sigma_h
        mu_pnl = pos_delta * spot * mu_h
        z = stats.norm.ppf(confidence)
        var = z * sigma_pnl - mu_pnl
        # Expected Shortfall (parametric normal)
        phi_z = stats.norm.pdf(z)
        es = sigma_pnl * phi_z / (1 - confidence) - mu_pnl
        # P&L distribution for plotting (analytic normal)
        n_grid = 5000
        z_grid = np.linspace(-5, 5, n_grid)
        pnl_grid = pos_delta * spot * (mu_h + sigma_h * z_grid)
        pdf = stats.norm.pdf(z_grid)
    else:
        # Delta-Gamma simulation (faster than full MC since we don't reprice)
        n_sim = 100_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n_sim)
        ds = spot * (np.exp(mu_h + sigma_h * z) - 1.0)
        pnl_grid = pos_delta * ds + 0.5 * pos_gamma * ds**2
        var = -np.percentile(pnl_grid, (1 - confidence) * 100)
        es = -pnl_grid[pnl_grid <= -var].mean() if (pnl_grid <= -var).any() else var
        pdf = None  # use histogram instead

    return {
        "method": "Delta-Normal" if abs(gamma) < 1e-12 else "Delta-Gamma",
        "VaR": float(max(var, 0.0)),
        "ES": float(max(es, 0.0)),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "position_size": position_size,
        "n_scenarios": 5000 if abs(gamma) < 1e-12 else 100_000,
        "pnl_dist": pnl_grid,
        "pdf": pdf,
        "option_value": option_value * position_size * sign,
    }


# ===================================================================
# 2. Monte Carlo VaR (full revaluation)
# ===================================================================
def monte_carlo_var(
    *,
    pricer,                       # callable: pricer(**kwargs) -> price
    base_kwargs: dict,            # base parameters for pricer
    spot_key: str = "S",
    T_key: str = "T",
    sigma_annual: float,
    mu_annual: float = 0.0,
    horizon_days: float = 1.0,
    confidence: float = 0.95,
    position_size: float = 1.0,
    long: bool = True,
    n_paths: int = 5_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo VaR with full revaluation.

    1. Simulate underlying price S_h at horizon h under risk-neutral or
       physical drift (user-specified mu_annual).
    2. Reprice option at S_h with reduced T (T - h).
    3. P&L = V(S_h, T-h) - V(S_0, T)
    4. VaR = -percentile(PnL, 1-α);  ES = mean(PnL | PnL ≤ -VaR)
    """
    spot_0 = float(base_kwargs[spot_key])
    T_0 = float(base_kwargs[T_key])
    h = horizon_days / 252.0
    if T_0 <= h:
        raise ValueError(
            f"Horizon ({horizon_days}d) exceeds option time-to-expiry "
            f"({T_0*252:.0f}d). Reduce horizon."
        )

    sigma_h = sigma_annual * np.sqrt(h)
    mu_h = mu_annual * h

    # Base price
    base_kw = dict(base_kwargs)
    v0 = float(pricer(**base_kw))

    # Simulate underlying
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_h = spot_0 * np.exp(mu_h - 0.5 * sigma_h**2 + sigma_h * z)

    # Reprice at horizon
    pnl = np.zeros(n_paths)
    T_h = T_0 - h
    sign = 1.0 if long else -1.0

    # Vectorize where possible — most pricers don't support vectorized inputs,
    # so we loop.  5000 paths is fast enough for closed-form pricers.
    for i in range(n_paths):
        kw = dict(base_kw)
        kw[spot_key] = float(S_h[i])
        kw[T_key] = float(T_h)
        try:
            v_i = float(pricer(**kw))
        except Exception:
            v_i = v0  # fall back if pricer fails on extreme spot
        pnl[i] = sign * position_size * (v_i - v0)

    var = -np.percentile(pnl, (1 - confidence) * 100)
    tail = pnl[pnl <= -var]
    es = -tail.mean() if len(tail) > 0 else var

    return {
        "method": "Monte Carlo",
        "VaR": float(max(var, 0.0)),
        "ES": float(max(es, 0.0)),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "position_size": position_size,
        "n_scenarios": n_paths,
        "pnl_dist": pnl,
        "pdf": None,
        "option_value": v0 * position_size * sign,
        "S_horizon": S_h,
    }


# ===================================================================
# 3. Historical VaR
# ===================================================================
def historical_var(
    *,
    pricer,
    base_kwargs: dict,
    spot_key: str = "S",
    T_key: str = "T",
    historical_returns: np.ndarray,    # daily log returns of underlying
    horizon_days: float = 1.0,
    confidence: float = 0.95,
    position_size: float = 1.0,
    long: bool = True,
) -> dict:
    """
    Historical Simulation VaR.

    1. Aggregate `horizon_days` consecutive daily log returns to get
       horizon returns.
    2. Apply each historical horizon return to the current spot to get
       scenario S_h values.
    3. Reprice option at each scenario; compute P&L.
    4. VaR = -percentile(PnL, 1-α)
    """
    spot_0 = float(base_kwargs[spot_key])
    T_0 = float(base_kwargs[T_key])
    h = horizon_days / 252.0
    if T_0 <= h:
        raise ValueError(
            f"Horizon ({horizon_days}d) exceeds option time-to-expiry "
            f"({T_0*252:.0f}d). Reduce horizon."
        )

    historical_returns = np.asarray(historical_returns, dtype=float)
    horizon_n = max(int(horizon_days), 1)

    # Build horizon returns by rolling sum
    if horizon_n == 1:
        horizon_returns = historical_returns
    else:
        # rolling sum of horizon_n daily log returns
        if len(historical_returns) < horizon_n:
            raise ValueError(
                f"Need at least {horizon_n} historical returns; got "
                f"{len(historical_returns)}."
            )
        cs = np.cumsum(historical_returns)
        horizon_returns = cs[horizon_n - 1:] - np.concatenate(
            ([0.0], cs[: -horizon_n])
        )

    # Apply to current spot
    S_h = spot_0 * np.exp(horizon_returns)

    # Base price
    base_kw = dict(base_kwargs)
    v0 = float(pricer(**base_kw))
    T_h = T_0 - h
    sign = 1.0 if long else -1.0

    pnl = np.zeros(len(S_h))
    for i, s_h in enumerate(S_h):
        kw = dict(base_kw)
        kw[spot_key] = float(s_h)
        kw[T_key] = float(T_h)
        try:
            v_i = float(pricer(**kw))
        except Exception:
            v_i = v0
        pnl[i] = sign * position_size * (v_i - v0)

    var = -np.percentile(pnl, (1 - confidence) * 100)
    tail = pnl[pnl <= -var]
    es = -tail.mean() if len(tail) > 0 else var

    return {
        "method": "Historical",
        "VaR": float(max(var, 0.0)),
        "ES": float(max(es, 0.0)),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "position_size": position_size,
        "n_scenarios": len(S_h),
        "pnl_dist": pnl,
        "pdf": None,
        "option_value": v0 * position_size * sign,
        "S_horizon": S_h,
    }


# ===================================================================
# Convenience: compute all methods
# ===================================================================
def compute_all_var_methods(
    *,
    pricer,
    base_kwargs: dict,
    option_value: float,
    delta: float,
    gamma: float,
    spot: float,
    sigma_annual: float,
    mu_annual: float = 0.0,
    horizon_days: float = 1.0,
    confidence: float = 0.95,
    position_size: float = 1.0,
    long: bool = True,
    spot_key: str = "S",
    T_key: str = "T",
    n_mc_paths: int = 5_000,
    historical_returns: np.ndarray | None = None,
) -> dict:
    """
    Run Delta-Normal, Delta-Gamma, Monte Carlo, and (if returns provided)
    Historical VaR all at once and return as a dict.
    """
    out = {}

    # Delta-Normal (no gamma)
    out["delta_normal"] = delta_normal_var(
        option_value=option_value,
        delta=delta, gamma=0.0,
        spot=spot, sigma_annual=sigma_annual, mu_annual=mu_annual,
        horizon_days=horizon_days, confidence=confidence,
        position_size=position_size, long=long,
    )

    # Delta-Gamma
    out["delta_gamma"] = delta_normal_var(
        option_value=option_value,
        delta=delta, gamma=gamma,
        spot=spot, sigma_annual=sigma_annual, mu_annual=mu_annual,
        horizon_days=horizon_days, confidence=confidence,
        position_size=position_size, long=long,
    )

    # Monte Carlo
    try:
        out["monte_carlo"] = monte_carlo_var(
            pricer=pricer, base_kwargs=base_kwargs,
            spot_key=spot_key, T_key=T_key,
            sigma_annual=sigma_annual, mu_annual=mu_annual,
            horizon_days=horizon_days, confidence=confidence,
            position_size=position_size, long=long,
            n_paths=n_mc_paths,
        )
    except Exception as e:
        out["monte_carlo"] = {"error": str(e)}

    # Historical (only if returns provided)
    if historical_returns is not None and len(historical_returns) > 0:
        try:
            out["historical"] = historical_var(
                pricer=pricer, base_kwargs=base_kwargs,
                spot_key=spot_key, T_key=T_key,
                historical_returns=historical_returns,
                horizon_days=horizon_days, confidence=confidence,
                position_size=position_size, long=long,
            )
        except Exception as e:
            out["historical"] = {"error": str(e)}

    return out
