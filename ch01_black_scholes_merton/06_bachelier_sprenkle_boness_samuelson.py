"""
06_bachelier_sprenkle_boness_samuelson.py — BSM 前驱模型
=========================================================
【模型简介】
在 Black-Scholes (1973) 之前，多位学者提出了各自的期权定价方法。
本文件实现 Haug 书中 Section 1.3 介绍的四个历史模型：

1. Bachelier (1900)  — 假设价格服从正态分布（算术布朗运动）
2. Sprenkle (1964)   — 假设对数正态，但使用实际（物理）测度
3. Boness (1964)     — 类似 BSM 但用股票预期收益替代无风险利率
4. Samuelson (1965)  — 区分期权与股票的预期收益率

这些模型缺乏"风险中性定价"的关键概念，因此依赖主观参数（预期
收益率、风险溢价等），在实践中难以直接使用，但对理解 BSM 的
理论进步具有重要历史意义。

书中对应：Haug (2007), Chapter 1, Sections 1.3.1–1.3.4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# 1. Bachelier (1900) — 正态分布模型
# ═══════════════════════════════════════════════════════════════

def bachelier(F: float, K: float, T: float,
              sigma_abs: float, option_type: str = 'call') -> float:
    """
    Bachelier (1900) 期权定价公式（正态分布）。

    Bachelier 的博士论文首次给出期权定价的数学框架，假设
    标的资产价格 S_T 服从正态分布（算术布朗运动）：
        dS = σ_abs · dW  （无漂移）

    参数
    ----
    F        : 远期价格（= S·e^{(r-q)·T} 或期货价格）
    K        : 行权价格
    T        : 距到期日时间（年）
    sigma_abs: 价格的绝对（非百分比）年化波动率（如 5 美元/年^0.5）
    option_type : 'call' 或 'put'

    数学公式
    --------
    令 σ_T = σ_abs · √T，d = (F - K) / σ_T

    看涨：C_undiscounted = (F - K)·N(d) + σ_T·n(d)
    看跌：P_undiscounted = (K - F)·N(-d) + σ_T·n(d)

    注：本函数返回未折现价格；实际价格需乘以 e^{-r·T}
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    # σ_T = 到期时价格的标准差
    sigma_T = sigma_abs * sqrt(T)
    if sigma_T < 1e-10:
        if option_type.lower() == 'call':
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    d = (F - K) / sigma_T

    if option_type.lower() == 'call':
        # 正态分布下的期望正收益：标准化距离 × N(d) + 密度 × σ_T
        price = (F - K) * N(d) + sigma_T * n(d)
    elif option_type.lower() == 'put':
        price = (K - F) * N(-d) + sigma_T * n(d)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


def bachelier_discounted(S: float, K: float, T: float, r: float,
                         sigma_abs: float, q: float = 0.0,
                         option_type: str = 'call') -> float:
    """
    Bachelier 模型（含折现）。

    先计算远期价格 F = S·e^{(r-q)·T}，
    再对未折现期权价格进行折现 e^{-r·T}。

    参数
    ----
    S        : 现货价格
    K        : 行权价
    T        : 到期时间（年）
    r        : 无风险利率
    sigma_abs: 绝对波动率
    q        : 连续红利率（默认 0）
    option_type : 'call' 或 'put'
    """
    F = S * exp((r - q) * T)
    undiscounted = bachelier(F, K, T, sigma_abs, option_type)
    return exp(-r * T) * undiscounted


# ═══════════════════════════════════════════════════════════════
# 2. Sprenkle (1964) — 实际测度对数正态模型
# ═══════════════════════════════════════════════════════════════

def sprenkle(S: float, K: float, T: float,
             rho_growth: float, lambda_risk: float,
             sigma: float, option_type: str = 'call') -> float:
    """
    Sprenkle (1964) 期权定价公式。

    假设股价服从对数正态分布（与 BSM 相同），但使用实际
    预期收益率而非风险中性利率，并引入主观风险规避参数。

    参数
    ----
    S           : 当前股价
    K           : 行权价格
    T           : 到期时间（年）
    rho_growth  : 股票的预期年化增长率（实际测度，如 0.10 = 10%）
    lambda_risk : 风险规避/调整参数（主观，介于 0 和 1 之间）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    数学公式
    --------
    d₁ = [ln(S/K) + (ρ + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·e^{ρT}·N(d₁) - λ·K·N(d₂)
    看跌：P = λ·K·N(-d₂) - S·e^{ρT}·N(-d₁)

    注意：与 BSM 对比
    - BSM 中 e^{rT} 被 e^{ρT} 替代（用主观增长率）
    - BSM 中 e^{-rT} 被 λ 替代（风险调整折现，非客观折现）
    - 缺少了"风险中性测度变换"的理论依据
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # d₁, d₂：使用实际预期增长率 ρ，而非无风险利率 r
    d1 = (log(S / K) + (rho_growth + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        # S·e^{ρT} 是 T 期后股价的期望值；λ 是主观风险折现因子
        price = S * exp(rho_growth * T) * N(d1) - lambda_risk * K * N(d2)
    elif option_type.lower() == 'put':
        price = lambda_risk * K * N(-d2) - S * exp(rho_growth * T) * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


# ═══════════════════════════════════════════════════════════════
# 3. Boness (1964) — 用预期收益折现
# ═══════════════════════════════════════════════════════════════

def boness(S: float, K: float, T: float,
           rho: float, sigma: float, option_type: str = 'call') -> float:
    """
    Boness (1964) 期权定价公式。

    Boness 提出用股票自身的预期收益率（而非无风险利率）
    对期权支付进行折现，这是向 BSM 迈进的一步，但仍未
    实现风险中性定价。

    参数
    ----
    S           : 当前股价
    K           : 行权价格
    T           : 到期时间（年）
    rho         : 股票的预期年化收益率（如 0.12 = 12%）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    数学公式
    --------
    d₁ = [ln(S/K) + (ρ + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·N(d₁) - K·e^{-ρT}·N(d₂)
    看跌：P = K·e^{-ρT}·N(-d₂) - S·N(-d₁)

    差别：与 BSM 相比，Boness 用 ρ（股票预期收益率）
         替代 r（无风险利率）作为折现率，但对 S 的系数
         保持为 1（未折现），这在理论上不一致。
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (log(S / K) + (rho + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        # 用股票预期收益率 ρ 折现行权价，但股票本身未折现
        price = S * N(d1) - K * exp(-rho * T) * N(d2)
    elif option_type.lower() == 'put':
        price = K * exp(-rho * T) * N(-d2) - S * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


# ═══════════════════════════════════════════════════════════════
# 4. Samuelson (1965) — 区分股票与期权预期收益
# ═══════════════════════════════════════════════════════════════

def samuelson(S: float, K: float, T: float,
              rho: float, alpha: float,
              sigma: float, option_type: str = 'call') -> float:
    """
    Samuelson (1965) 期权定价公式。

    Samuelson 意识到期权的风险（杠杆效应）高于股票，
    因此期权的预期收益率 α 应高于股票的预期收益率 ρ。

    参数
    ----
    S           : 当前股价
    K           : 行权价格
    T           : 到期时间（年）
    rho         : 股票的预期年化收益率
    alpha       : 期权的预期年化收益率（alpha > rho，因为期权风险更高）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    数学公式
    --------
    d₁ = [ln(S/K) + (ρ + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·e^{(ρ-α)·T}·N(d₁) - K·e^{-α·T}·N(d₂)

    注：当 α = ρ = r（无风险利率）时，退化为 BSM 公式。
    BSM 的创新在于通过 delta 对冲消除了股票预期收益率，
    实现了"与 ρ、α 无关"的定价。
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (log(S / K) + (rho + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        # e^{(ρ-α)·T}：股票增长折现到期权折现率的净调整
        price = S * exp((rho - alpha) * T) * N(d1) - K * exp(-alpha * T) * N(d2)
    elif option_type.lower() == 'put':
        price = K * exp(-alpha * T) * N(-d2) - S * exp((rho - alpha) * T) * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


if __name__ == "__main__":
    print("=" * 60)
    print("BSM 前驱模型历史对比 — 数值示例")
    print("=" * 60)

    S, K, T, r, sigma = 100, 100, 0.5, 0.08, 0.25

    # ── Bachelier（用绝对波动率 = S·σ ≈ 100×0.25 = 25）──────────
    sigma_abs = S * sigma   # 转为绝对波动率
    b_call = bachelier_discounted(S, K, T, r, sigma_abs, 0.0, 'call')
    print(f"\nBachelier  Call (σ_abs={sigma_abs:.1f}) = {b_call:.4f}")

    # ── Sprenkle（假设 ρ=0.12, λ=0.9）──────────────────────────
    sp_call = sprenkle(S, K, T, rho_growth=0.12, lambda_risk=0.9, sigma=sigma, option_type='call')
    print(f"Sprenkle   Call (ρ=0.12, λ=0.9)    = {sp_call:.4f}")

    # ── Boness（假设 ρ=0.12）────────────────────────────────────
    bo_call = boness(S, K, T, rho=0.12, sigma=sigma, option_type='call')
    print(f"Boness     Call (ρ=0.12)            = {bo_call:.4f}")

    # ── Samuelson（假设 ρ=0.12, α=0.15）────────────────────────
    sa_call = samuelson(S, K, T, rho=0.12, alpha=0.15, sigma=sigma, option_type='call')
    print(f"Samuelson  Call (ρ=0.12, α=0.15)   = {sa_call:.4f}")

    # ── 当参数退化为风险中性时，Samuelson → BSM ─────────────────
    sa_bsm = samuelson(S, K, T, rho=r, alpha=r, sigma=sigma, option_type='call')
    from ch01_black_scholes_merton._01_black_scholes_1973 import black_scholes
    bsm_call = black_scholes(S, K, T, r, sigma, 'call')
    print(f"\nSamuelson(ρ=r, α=r) = {sa_bsm:.4f}  BSM = {bsm_call:.4f}  (应相等)")

    print(f"\n【历史意义】")
    print("BSM 的关键创新：通过 delta 对冲消除了对 ρ、α 等主观参数的依赖，")
    print("实现了仅依赖可观测参数（S, K, T, r, σ）的客观定价。")
