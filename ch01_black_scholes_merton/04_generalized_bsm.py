"""
04_generalized_bsm.py — 广义 Black-Scholes-Merton 公式
=======================================================
【模型简介】
Haug (2007) 将 Black-Scholes、Merton、Black-76、Asay、
Garman-Kohlhagen 统一为一个"广义 BSM 公式"，通过
持有成本参数 b（cost-of-carry）区分不同标的资产：

  b = r         → 原始 Black-Scholes (1973)：无红利股票
  b = r - q     → Merton (1973)：连续红利股票/指数
  b = 0         → Black (1976)：期货/远期期权
  b = 0, r=0    → Asay (1982)：已保证金的期货期权
  b = r_d - r_f → Garman-Kohlhagen (1983)：外汇期权

这一统一框架是 Haug 书中贯穿始终的核心工具。

参考：Haug, E.G. (2007). "The Complete Guide to Option Pricing Formulas",
       2nd ed., Chapter 1, Section 1.1.6.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def generalized_bsm(S: float, K: float, T: float, r: float, b: float,
                    sigma: float, option_type: str = 'call') -> float:
    """
    广义 Black-Scholes-Merton 欧式期权定价公式。

    参数
    ----
    S           : 标的资产现价（股价、期货价、汇率等）
    K           : 行权价格
    T           : 距到期日时间（年）
    r           : 无风险利率（连续复利）
    b           : 持有成本（cost-of-carry）
                  b = r          → BSM（无红利股票）
                  b = r - q      → Merton（连续红利，q 为红利率）
                  b = 0          → Black-76（期货）
                  b = 0, r = 0   → Asay（已保证金期货期权）
                  b = r_d - r_f  → Garman-Kohlhagen（外汇，r_d=本币利率）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    返回
    ----
    float : 期权的理论价值

    数学公式
    --------
    d₁ = [ln(S/K) + (b + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·e^{(b-r)·T}·N(d₁) - K·e^{-r·T}·N(d₂)
    看跌：P = K·e^{-r·T}·N(-d₂) - S·e^{(b-r)·T}·N(-d₁)

    注意：e^{(b-r)·T} = e^{b·T} · e^{-r·T}
         当 b=r 时，e^{(b-r)·T} = 1，退化为 BSM
         当 b=0 时，e^{(b-r)·T} = e^{-r·T}，退化为 Black-76
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # ── 核心计算 ─────────────────────────────────────────────────
    # 持有成本调整因子：e^{(b-r)·T}
    # 经济含义：持有标的资产 T 年，净收益因子
    carry_factor = exp((b - r) * T)
    discount      = exp(-r * T)

    d1 = (log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        price = S * carry_factor * N(d1) - K * discount * N(d2)
    elif option_type.lower() == 'put':
        price = K * discount * N(-d2) - S * carry_factor * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


def black_scholes_stock(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes (1973)：无红利股票，b = r"""
    return generalized_bsm(S, K, T, r, b=r, sigma=sigma, option_type=option_type)


def merton_index(S, K, T, r, q, sigma, option_type='call'):
    """Merton (1973)：连续红利股票/指数，b = r - q"""
    return generalized_bsm(S, K, T, r, b=r-q, sigma=sigma, option_type=option_type)


def black_futures(F, K, T, r, sigma, option_type='call'):
    """Black (1976)：期货/远期，b = 0"""
    return generalized_bsm(F, K, T, r, b=0, sigma=sigma, option_type=option_type)


def asay_margined(F, K, T, sigma, option_type='call'):
    """Asay (1982)：已保证金期货期权，b = 0, r = 0（无折现）"""
    return generalized_bsm(F, K, T, r=0, b=0, sigma=sigma, option_type=option_type)


def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Garman-Kohlhagen (1983)：外汇期权。
    S = 即期汇率（本币/外币），r_d = 本币利率，r_f = 外币利率
    b = r_d - r_f（外币利率充当连续"红利"）
    """
    return generalized_bsm(S, K, T, r=r_d, b=r_d-r_f, sigma=sigma, option_type=option_type)


if __name__ == "__main__":
    print("=" * 60)
    print("广义 BSM 公式 — 统一验证各模型")
    print("=" * 60)

    # Black-Scholes: S=60, K=65, T=0.25, r=0.08, σ=0.30
    c1 = black_scholes_stock(60, 65, 0.25, 0.08, 0.30, 'call')
    print(f"\nBSM Call (S=60,K=65,T=0.25,r=0.08,σ=0.30) = {c1:.4f}  (≈2.1334)")

    # Merton: S=100, K=100, T=0.5, r=0.08, q=0.04, σ=0.20
    c2 = merton_index(100, 100, 0.5, 0.08, 0.04, 0.20, 'call')
    print(f"Merton Call (q=0.04)                       = {c2:.4f}  (≈8.2915)")

    # Black-76: F=105, K=100, T=0.5, r=0.10, σ=0.36
    c3 = black_futures(105, 100, 0.5, 0.10, 0.36, 'call')
    print(f"Black-76 Call (F=105)                      = {c3:.4f}  (≈15.6884)")

    # Asay: F=100, K=100, T=0.5, σ=0.20
    c4 = asay_margined(100, 100, 0.5, 0.20, 'call')
    print(f"Asay Margined Call (F=100, no discount)    = {c4:.4f}")

    # Garman-Kohlhagen: EUR/USD, S=1.56, K=1.60, T=0.5, r_d=0.06, r_f=0.08, σ=0.12
    c5 = garman_kohlhagen(1.56, 1.60, 0.5, 0.06, 0.08, 0.12, 'call')
    print(f"G-K FX Call (S=1.56,K=1.60)                = {c5:.4f}  (≈0.0291)")
