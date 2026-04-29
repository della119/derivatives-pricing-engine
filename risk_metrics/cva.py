"""
Credit Valuation Adjustment (CVA) for Option Positions
========================================================
Adapted from Jon Gregory, "Counterparty Credit Risk" (Wiley 2010) and
the XVA toolkit (chapters 1-3).

Key formulas
------------
    Hazard rate (constant)        h ≈ s_CDS / LGD
    Survival probability          S(t) = exp(-h*t)
    Marginal default probability  PD(t_{i-1}, t_i) = S(t_{i-1}) - S(t_i)

    Exposure (long option holder) E(t) = V(t)        [always ≥ 0 for vanilla]
    Expected Exposure             EE(t) = E[V(t)]    [MC average across paths]
    Expected Positive Exposure    EPE   = (1/T) ∫ EE(t) dt

    CVA  =  LGD × Σ_i  D(t_i) × EE(t_i) × PD(t_{i-1}, t_i)
    DVA  =  LGD_own × Σ_i  D(t_i) × ENE(t_i) × PD_own(t_{i-1}, t_i)
    BCVA =  CVA + DVA          (DVA is a benefit, CVA is a cost)

Conventions
-----------
- All amounts are returned as POSITIVE numbers representing the magnitude.
- CVA is the cost to the option holder; deducted from the risk-free price.
- "Risky price" = Risk-free price - CVA
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


# ===================================================================
# Hazard-rate / survival curve
# ===================================================================
def hazard_rate_from_cds(cds_spread_bps: float, recovery_rate: float = 0.40) -> float:
    """
    Approximate constant hazard rate from a single CDS par spread.
        h ≈ spread / (1 - R) = spread / LGD
    cds_spread_bps : CDS spread in basis points (e.g., 150 = 150 bps = 1.50%)
    """
    s = cds_spread_bps / 10_000.0
    lgd = 1.0 - recovery_rate
    if lgd <= 0:
        return 0.0
    return s / lgd


def survival_probability(t: float, hazard: float) -> float:
    """S(t) = exp(-h * t)"""
    return float(np.exp(-hazard * t))


def marginal_default_probability(t_start: float, t_end: float, hazard: float) -> float:
    """P(default in (t_start, t_end]) = S(t_start) - S(t_end)"""
    return survival_probability(t_start, hazard) - survival_probability(t_end, hazard)


# ===================================================================
# CDS-bootstrapped piecewise-constant credit curve
# ===================================================================
class CreditCurve:
    """
    Piecewise-constant hazard-rate curve bootstrapped from CDS quotes.

    For simplicity we use the quick approximation
        h_i ≈ s_i / LGD
    rather than an iterative bootstrap. This is exact for a flat curve and
    accurate enough for typical CVA work.
    """

    def __init__(self, tenors: Sequence[float], hazard_rates: Sequence[float]):
        if len(tenors) != len(hazard_rates):
            raise ValueError("tenors and hazard_rates must have the same length")
        self.tenors = np.asarray(tenors, dtype=float)
        self.hazards = np.asarray(hazard_rates, dtype=float)

    @classmethod
    def from_cds_spreads(cls, tenors: Sequence[float], spreads_bps: Sequence[float],
                         recovery_rate: float = 0.40) -> "CreditCurve":
        hazards = [hazard_rate_from_cds(s, recovery_rate) for s in spreads_bps]
        return cls(tenors, hazards)

    @classmethod
    def flat(cls, hazard: float, max_tenor: float = 30.0) -> "CreditCurve":
        return cls([max_tenor], [hazard])

    def survival(self, t: float) -> float:
        """S(t) using piecewise-constant integration of the hazard rate."""
        if t <= 0:
            return 1.0
        accumulated = 0.0
        t_prev = 0.0
        for tk, hk in zip(self.tenors, self.hazards):
            if t <= tk:
                accumulated += hk * (t - t_prev)
                return float(np.exp(-accumulated))
            accumulated += hk * (tk - t_prev)
            t_prev = tk
        # Beyond last pillar — extrapolate flat using last hazard
        accumulated += self.hazards[-1] * (t - t_prev)
        return float(np.exp(-accumulated))

    def marginal_default_prob(self, t1: float, t2: float) -> float:
        return self.survival(t1) - self.survival(t2)


# ===================================================================
# Exposure profile via Monte Carlo (option holder's perspective)
# ===================================================================
def simulate_exposure_profile(
    *,
    pricer: Callable,
    base_kwargs: dict,
    spot_key: str = "S",
    T_key: str = "T",
    sigma_annual: float,
    mu_annual: float = 0.0,
    n_paths: int = 2_000,
    n_steps: int = 24,
    seed: int = 42,
) -> dict:
    """
    Simulate the underlying under GBM and reprice the option on a time grid
    to build an Expected Exposure profile.

    For a LONG option holder, exposure is the (non-negative) option value:
        E(t) = max(V(t), 0) = V(t)   for vanilla long positions.

    Returns
    -------
    dict with:
        times    : [0, dt, 2*dt, ..., T]            (n_steps+1 points)
        EE       : np.ndarray, expected exposure at each grid point
        PFE_975  : np.ndarray, 97.5th percentile exposure
        EPE      : float, time-averaged EE
        v0       : float, risk-free option value at t=0
    """
    spot_0 = float(base_kwargs[spot_key])
    T_0 = float(base_kwargs[T_key])

    times = np.linspace(0.0, T_0, n_steps + 1)
    dt = times[1] - times[0]

    rng = np.random.default_rng(seed)

    # GBM paths for the underlying
    log_S = np.full((n_paths, n_steps + 1), np.log(spot_0))
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    diffusion = sigma_annual * np.sqrt(dt)
    z = rng.standard_normal(size=(n_paths, n_steps))
    log_S[:, 1:] = np.log(spot_0) + np.cumsum(drift + diffusion * z, axis=1)
    S_paths = np.exp(log_S)

    # Reprice the option at each grid point on each path
    EE = np.zeros(n_steps + 1)
    PFE = np.zeros(n_steps + 1)

    # t=0 baseline
    v0 = float(pricer(**base_kwargs))
    EE[0] = max(v0, 0.0)
    PFE[0] = max(v0, 0.0)

    for k in range(1, n_steps + 1):
        t_k = times[k]
        T_remaining = max(T_0 - t_k, 1e-6)
        values = np.zeros(n_paths)
        for j in range(n_paths):
            kw = dict(base_kwargs)
            kw[spot_key] = float(S_paths[j, k])
            kw[T_key] = float(T_remaining)
            try:
                v = float(pricer(**kw))
            except Exception:
                v = 0.0
            values[j] = max(v, 0.0)
        EE[k] = values.mean()
        PFE[k] = np.percentile(values, 97.5)

    # Time-averaged EPE
    epe = float(np.trapz(EE, times) / max(T_0, 1e-9))

    return {
        "times": times,
        "EE": EE,
        "PFE_975": PFE,
        "EPE": epe,
        "v0": v0,
    }


# ===================================================================
# CVA / DVA / BCVA
# ===================================================================
def compute_cva(
    *,
    times: np.ndarray,
    ee_profile: np.ndarray,
    discount_rate: float,
    credit_curve: CreditCurve,
    recovery_rate: float = 0.40,
) -> float:
    """
    Discrete CVA:
        CVA = LGD × Σ_i  D(t_i) × EE(t_i) × [S(t_{i-1}) - S(t_i)]
    """
    lgd = 1.0 - recovery_rate
    cva = 0.0
    t_prev = 0.0
    for t, ee in zip(times, ee_profile):
        if t <= 0:
            t_prev = t
            continue
        df = float(np.exp(-discount_rate * t))
        dp = credit_curve.marginal_default_prob(t_prev, t)
        cva += df * ee * dp
        t_prev = t
    return lgd * cva


def compute_dva(
    *,
    times: np.ndarray,
    ene_profile: np.ndarray,
    discount_rate: float,
    own_credit_curve: CreditCurve,
    own_recovery: float = 0.40,
) -> float:
    """
    DVA from one's own default risk:
        DVA = LGD_own × Σ_i  D(t_i) × |ENE(t_i)| × dPD_own(t_{i-1}, t_i)

    For an option HOLDER, ENE is typically zero (long position can't owe
    counterparty more than zero on a vanilla option). Provided for symmetry
    when the position is short.
    """
    lgd = 1.0 - own_recovery
    dva = 0.0
    t_prev = 0.0
    for t, ene in zip(times, ene_profile):
        if t <= 0:
            t_prev = t
            continue
        df = float(np.exp(-discount_rate * t))
        dp = own_credit_curve.marginal_default_prob(t_prev, t)
        dva += df * abs(ene) * dp
        t_prev = t
    return lgd * dva


def compute_cva_for_option(
    *,
    pricer: Callable,
    base_kwargs: dict,
    spot_key: str = "S",
    T_key: str = "T",
    sigma_annual: float,
    discount_rate: float = 0.05,
    counterparty_cds_bps: float = 150.0,
    counterparty_recovery: float = 0.40,
    own_cds_bps: float = 0.0,
    own_recovery: float = 0.40,
    is_long: bool = True,
    notional_multiplier: float = 1.0,
    n_paths: int = 2_000,
    n_steps: int = 24,
    seed: int = 42,
    credit_curve: CreditCurve | None = None,
    own_credit_curve: CreditCurve | None = None,
) -> dict:
    """
    All-in-one CVA computation for a single option position.

    Parameters
    ----------
    pricer, base_kwargs : the option pricing function and its kwargs
    spot_key, T_key     : keys for spot and time-to-expiry in kwargs
    sigma_annual        : annual volatility for GBM exposure simulation
    discount_rate       : flat discount rate for B(0,t) = exp(-r t)
    counterparty_cds_bps: counterparty CDS spread in bps (drives hazard rate)
    counterparty_recovery: counterparty recovery rate (default 40%)
    own_cds_bps         : institution's own CDS spread (for DVA, 0 = no DVA)
    own_recovery        : institution's recovery rate
    is_long             : True if we are long the option (CVA applies)
                          False if short (DVA applies, exposure flips)
    notional_multiplier : scalar to multiply price/CVA by (e.g., 100 contracts)
    n_paths, n_steps    : MC parameters for exposure profile
    seed                : RNG seed
    credit_curve, own_credit_curve : optional pre-built curves; if None,
                          a flat curve is built from the bps spreads.

    Returns
    -------
    dict containing:
        v0, EE_profile, PFE_profile, times, EPE,
        CVA, DVA, BCVA, risky_price
    """
    # Build credit curves if not supplied
    if credit_curve is None:
        h_cp = hazard_rate_from_cds(counterparty_cds_bps, counterparty_recovery)
        credit_curve = CreditCurve.flat(h_cp)
    if own_credit_curve is None:
        h_own = hazard_rate_from_cds(own_cds_bps, own_recovery)
        own_credit_curve = CreditCurve.flat(h_own)

    # Simulate exposure profile (always run from holder's perspective)
    profile = simulate_exposure_profile(
        pricer=pricer,
        base_kwargs=base_kwargs,
        spot_key=spot_key,
        T_key=T_key,
        sigma_annual=sigma_annual,
        mu_annual=discount_rate,  # risk-neutral drift
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    times = profile["times"]
    EE = profile["EE"]
    PFE = profile["PFE_975"]
    v0 = profile["v0"]

    # For LONG position: counterparty owes us → CVA on EE
    #     ENE ≈ 0 (we never owe counterparty on a long vanilla option)
    # For SHORT position: we owe counterparty → CVA = 0, but DVA on |ENE|
    if is_long:
        ee_for_cva = EE
        ene_for_dva = np.zeros_like(EE)
    else:
        ee_for_cva = np.zeros_like(EE)
        ene_for_dva = EE  # we owe the option value to counterparty

    cva = compute_cva(
        times=times, ee_profile=ee_for_cva,
        discount_rate=discount_rate,
        credit_curve=credit_curve,
        recovery_rate=counterparty_recovery,
    )
    dva = compute_dva(
        times=times, ene_profile=ene_for_dva,
        discount_rate=discount_rate,
        own_credit_curve=own_credit_curve,
        own_recovery=own_recovery,
    )

    # Apply notional multiplier
    v0_total = v0 * notional_multiplier
    EE_total = EE * notional_multiplier
    PFE_total = PFE * notional_multiplier
    epe_total = profile["EPE"] * notional_multiplier
    cva_total = cva * notional_multiplier
    dva_total = dva * notional_multiplier
    bcva_total = cva_total - dva_total  # net cost (positive = net cost)

    # Risky price (long perspective): risk-free price - CVA + DVA
    sign = 1.0 if is_long else -1.0
    risky_price = sign * v0_total - cva_total + dva_total

    return {
        "v0": v0_total,
        "times": times,
        "EE_profile": EE_total,
        "PFE_profile": PFE_total,
        "EPE": epe_total,
        "CVA": cva_total,
        "DVA": dva_total,
        "BCVA": bcva_total,
        "risky_price": risky_price,
        "is_long": is_long,
        "notional": notional_multiplier,
        "counterparty_hazard": credit_curve.hazards[0] if len(credit_curve.hazards) == 1 else None,
        "own_hazard": own_credit_curve.hazards[0] if len(own_credit_curve.hazards) == 1 else None,
    }
