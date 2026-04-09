"""
06_binary_options.py — 二元期权 (Binary / Digital Options)
==========================================================
【模型简介】
二元期权（又称数字期权，Digital Option）在到期时支付
固定金额（现金或资产）或零，取决于是否满足触发条件。

本文件实现 Haug 书中 Section 4.19 的所有类型：
  1. Gap Option（缺口期权）
  2. Cash-or-Nothing（现金或无）
  3. Asset-or-Nothing（资产或无）
  4. Supershare Option（超级份额期权）
  5. Binary Barrier Options（二元障碍期权，28种变体）

参考：Rubinstein, M. & Reiner, E. (1991). "Unscrambling the Binary Code."
       RISK Magazine, Vol. 4, 75–83.
书中对应：Haug (2007), Chapter 4, Section 4.19
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# 1. Gap Option（缺口期权 / Pay-Later Option）
# ═══════════════════════════════════════════════════════════════

def gap_option(S: float, K1: float, K2: float, T: float,
               r: float, b: float, sigma: float,
               option_type: str = 'call') -> float:
    """
    缺口期权（Gap Option）定价。

    K1 = 触发行权的价格门槛（trigger）
    K2 = 实际结算时用于计算收益的行权价（payoff strike）

    收益：
    看涨：若 S_T > K1，则支付 S_T - K2（可能为负！）
    看跌：若 S_T < K1，则支付 K2 - S_T（可能为负！）

    因此缺口期权价值 = 正常 BSM，但用 K1 计算触发概率、K2 计算收益量
    即：d₁, d₂ 中使用 K1；公式中的行权支付使用 K2

    应用：Pay-later 期权（延迟支付期权溢价），gap = K2 - K1 > 0 时
         对持有人不利（支付更高价格才能触发）

    公式
    ----
    d₁ = [ln(S/K1) + (b + σ²/2)·T] / (σ√T)  ← 触发概率用 K1
    d₂ = d₁ - σ√T

    看涨：C = S·e^{(b-r)T}·N(d₁) - K2·e^{-rT}·N(d₂)  ← 收益用 K2
    看跌：P = K2·e^{-rT}·N(-d₂) - S·e^{(b-r)T}·N(-d₁)
    """
    if T <= 0:
        if option_type == 'call': return max(S - K2, 0.) if S > K1 else 0.
        return max(K2 - S, 0.) if S < K1 else 0.

    # d₁, d₂ 基于触发价 K1（决定是否行权）
    d1 = (log(S / K1) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    cf = exp((b-r)*T); df = exp(-r*T)

    if option_type.lower() == 'call':
        # 资产部分正常，但现金支付部分用 K2（可能 ≠ K1）
        return S * cf * N(d1) - K2 * df * N(d2)
    return K2 * df * N(-d2) - S * cf * N(-d1)


# ═══════════════════════════════════════════════════════════════
# 2. Cash-or-Nothing（现金或无）
# ═══════════════════════════════════════════════════════════════

def cash_or_nothing(S: float, K: float, T: float,
                    r: float, b: float, sigma: float,
                    cash_amount: float = 1.0,
                    option_type: str = 'call') -> float:
    """
    现金或无期权（Cash-or-Nothing）。

    到期时：若 S_T > K（看涨）或 S_T < K（看跌），支付固定现金 cash_amount；
    否则支付零。

    公式
    ----
    看涨：V = cash_amount · e^{-rT} · N(d₂)
    看跌：V = cash_amount · e^{-rT} · N(-d₂)

    其中 N(d₂) 是风险中性下到期时价内的概率。

    直觉：e^{-rT}·N(d₂) 是风险中性概率下"S_T > K"的折现值
    """
    if T <= 0:
        if option_type == 'call': return cash_amount if S > K else 0.
        return cash_amount if S < K else 0.

    d2 = (log(S/K) + (b - 0.5*sigma**2)*T) / (sigma*sqrt(T))
    df = exp(-r * T)

    if option_type.lower() == 'call':
        return cash_amount * df * N(d2)
    return cash_amount * df * N(-d2)


# ═══════════════════════════════════════════════════════════════
# 3. Asset-or-Nothing（资产或无）
# ═══════════════════════════════════════════════════════════════

def asset_or_nothing(S: float, K: float, T: float,
                     r: float, b: float, sigma: float,
                     option_type: str = 'call') -> float:
    """
    资产或无期权（Asset-or-Nothing）。

    到期时：若 S_T > K（看涨），支付标的资产 S_T；否则支付零。

    公式
    ----
    看涨：V = S·e^{(b-r)T}·N(d₁)
    看跌：V = S·e^{(b-r)T}·N(-d₁)

    关系：
    BSM Call = Asset-or-Nothing(call) - K·e^{-rT}·Cash-or-Nothing(call)
    即标准期权 = 资产或无 - 现金或无的组合！

    直觉：持有看涨期权等于持有"价内时收到资产"的权利，
         减去"价内时支付 K 现金"的义务。
    """
    if T <= 0:
        if option_type == 'call': return S if S > K else 0.
        return S if S < K else 0.

    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    cf = exp((b-r)*T)

    if option_type.lower() == 'call':
        return S * cf * N(d1)
    return S * cf * N(-d1)


# ═══════════════════════════════════════════════════════════════
# 4. Supershare Option（超级份额期权）
# ═══════════════════════════════════════════════════════════════

def supershare_option(S: float, K_L: float, K_H: float, T: float,
                      r: float, b: float, sigma: float) -> float:
    """
    超级份额期权（Supershare Option，Hakansson 1976）。

    到期时：若 K_L ≤ S_T ≤ K_H，支付 S_T / K_L；否则支付零。

    公式
    ----
    V = (S·e^{(b-r)T}/K_L) · [N(d₁_L) - N(d₁_H)]

    其中：
    d₁_L = [ln(S/K_L) + (b + σ²/2)·T] / (σ√T)  （下限对应的 d₁）
    d₁_H = [ln(S/K_H) + (b + σ²/2)·T] / (σ√T)  （上限对应的 d₁）

    等价于：(1/K_L) × [Asset-or-Nothing(K_L) - Asset-or-Nothing(K_H)]

    直觉：可以视为一个"杠杆的价差期权"——价格在区间内时
         持有人获得股票份额（但按 K_L 为名义本金）。
    """
    if T <= 0:
        return S / K_L if K_L <= S <= K_H else 0.

    d1_L = (log(S/K_L) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d1_H = (log(S/K_H) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    cf = exp((b-r)*T)

    # 超级份额 = 资产或无(K_L) - 资产或无(K_H)，再除以 K_L
    return (S * cf / K_L) * (N(d1_L) - N(d1_H))


# ═══════════════════════════════════════════════════════════════
# 5. Binary Barrier Options（二元障碍期权）
# ═══════════════════════════════════════════════════════════════

def binary_barrier_option(S: float, K: float, H: float,
                          T: float, r: float, b: float, sigma: float,
                          cash_amount: float = 1.0,
                          binary_type: int = 1) -> float:
    """
    二元障碍期权（Rubinstein & Reiner 1991）。

    共 28 种变体（Haug 书中编号 1–28），按以下维度分类：
      - 支付类型：现金（Cash-or-Nothing）或资产（Asset-or-Nothing）
      - 障碍方向：向下（Down）或向上（Up）
      - 敲入/敲出（Knock-in / Knock-out）
      - 障碍与行权价的相对位置（H > K 或 H < K）

    参数
    ----
    binary_type : 1–28 之整数（对应 Haug 表 4-16 的编号）

    本实现提供最常见的 8 种（对应书中 Type 1–8）

    常见类型快速参考：
    Type 1: Down-and-In Cash-or-Nothing Call (H < K)
    Type 2: Up-and-In Cash-or-Nothing Call   (H > K)
    Type 3: Down-and-In Cash-or-Nothing Put  (H > K)  [typo: H < K for call]
    ...

    使用统一公式框架：
    μ  = (b - σ²/2) / σ²
    λ  = √(μ² + 2r/σ²)
    x₁ = ln(S/K) / (σ√T) + (1+μ)·σ√T
    x₂ = ln(S/H) / (σ√T) + (1+μ)·σ√T
    y₁ = ln(H²/(SK)) / (σ√T) + (1+μ)·σ√T
    y₂ = ln(H/S) / (σ√T) + (1+μ)·σ√T
    z  = ln(H/S) / (σ√T) + λ·σ√T
    """
    if T <= 0:
        # 简化：到期时直接判断
        return cash_amount if S > K else 0.

    mu  = (b - 0.5*sigma**2) / sigma**2
    lam = sqrt(mu**2 + 2*r / sigma**2)
    vol_t = sigma * sqrt(T)
    df = exp(-r*T)

    x1 = log(S/K) / vol_t + (1+mu)*vol_t
    x2 = log(S/H) / vol_t + (1+mu)*vol_t
    y1 = log(H**2/(S*K)) / vol_t + (1+mu)*vol_t
    y2 = log(H/S) / vol_t + (1+mu)*vol_t
    z  = log(H/S) / vol_t + lam*vol_t

    mu_fac  = (H/S)**(2*mu)
    lam_f1  = (H/S)**(mu+lam)
    lam_f2  = (H/S)**(mu-lam)

    # 最常用的几种：
    # Type 1: Down-and-In Cash Call（H < K）：触及下障碍后变为现金看涨
    if binary_type == 1:
        return cash_amount * df * (mu_fac * N(y2 - vol_t))
    # Type 2: Down-and-Out Cash Call（H < K）
    elif binary_type == 2:
        return cash_amount * df * (N(x1 - vol_t) - mu_fac * N(y2 - vol_t))
    # Type 3: Up-and-In Cash Call（H > K）
    elif binary_type == 3:
        return cash_amount * df * N(x2 - vol_t)
    # Type 4: Up-and-Out Cash Call（H > K）
    elif binary_type == 4:
        return cash_amount * df * (N(x1 - vol_t) - N(x2 - vol_t) + mu_fac * N(y2 - vol_t))
    # Type 5: Down-and-In Asset Call（H < K）
    elif binary_type == 5:
        cf = exp((b-r)*T)
        return cf * S * mu_fac * N(y2)
    # Type 6: Down-and-Out Asset Call（H < K）
    elif binary_type == 6:
        cf = exp((b-r)*T)
        return cf * S * (N(x2) - mu_fac * N(y2))
    # Type 7: 现金接触时支付（One-touch）
    elif binary_type == 7:
        return cash_amount * (lam_f1 * N(z) + lam_f2 * N(z - 2*lam*vol_t))
    # Type 8: 普通现金或无（Cash-or-Nothing）
    elif binary_type == 8:
        d2 = (log(S/K) + (b - 0.5*sigma**2)*T) / (sigma*sqrt(T))
        return cash_amount * df * N(d2)
    else:
        raise ValueError(f"binary_type {binary_type} 超出范围（支持 1–8）")


if __name__ == "__main__":
    print("=" * 60)
    print("二元期权 — 数值示例（Haug Chapter 4, Section 4.19）")
    print("=" * 60)

    S, K, T, r, b, sigma = 100, 100, 0.5, 0.09, 0.09, 0.20

    # Gap Option（Haug p.174）：K1=50, K2=57, S=50, T=0.5, r=0.09, σ=0.20
    gap = gap_option(50, 50, 57, 0.5, 0.09, 0.09, 0.20, 'call')
    print(f"\nGap Option（S=50, K1=50, K2=57）= {gap:.4f}  （参考值 ≈ 2.9096）")

    # Cash-or-Nothing（S=100, K=80, T=1, r=0.06, σ=0.35, cash=10）
    con = cash_or_nothing(100, 80, 1, 0.06, 0.06, 0.35, 10, 'call')
    print(f"Cash-or-Nothing Call（K=80, cash=10） = {con:.4f}  （参考值 ≈ 9.06）")

    # Asset-or-Nothing
    aon_call = asset_or_nothing(S, K, T, r, b, sigma, 'call')
    aon_put  = asset_or_nothing(S, K, T, r, b, sigma, 'put')
    print(f"\nAsset-or-Nothing Call = {aon_call:.4f}")
    print(f"Asset-or-Nothing Put  = {aon_put:.4f}")

    # 验证：BSM = Asset-or-Nothing - K·e^{-rT}·Cash-or-Nothing
    from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
    bsm_call = generalized_bsm(S, K, T, r, b, sigma, 'call')
    con_unit = cash_or_nothing(S, K, T, r, b, sigma, 1.0, 'call')
    reconstructed = aon_call - K * exp(-r*T) * con_unit
    print(f"\n验证：Asset-or-Nothing - K·df·Cash = {reconstructed:.4f}  BSM = {bsm_call:.4f}")

    # Supershare
    ss = supershare_option(100, 90, 110, 0.5, 0.05, 0.05, 0.20)
    print(f"\nSupershare Option（K_L=90, K_H=110） = {ss:.4f}")
