"""
02_merton_1973.py — Merton (1973) 连续红利股票/指数期权
=========================================================
【模型简介】
Merton (1973) 将 Black-Scholes 模型推广至支付连续红利的股票或
股票指数。核心修改：将股价 S 替换为调整后的远期价格
  F = S · e^{(r-q)·T}
其中 q 是连续红利率。

等价地：在 BSM 公式中将 r 替换为 (r - q) 作为"实际"增长率，
折现率仍用 r。这使得公式既适用于：
  - 支付连续红利的股票（如中国 A 股以当期分红率近似）
  - 股票指数（以指数股息率 q 近似连续红利）
  - 外汇期权（外币利率相当于连续"红利"，见 Garman-Kohlhagen）

参考：Merton, R.C. (1973). "Theory of Rational Option Pricing."
       Bell Journal of Economics, 4(1), 141–183.

书中对应：Haug (2007), Chapter 1, Section 1.1.2 (Options on Stock Indexes)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def merton(S: float, K: float, T: float, r: float, q: float,
           sigma: float, option_type: str = 'call') -> float:
    """
    Merton (1973) 连续红利股票/指数的欧式期权定价公式。

    参数
    ----
    S           : 当前股价（或指数点位）
    K           : 行权价格
    T           : 距到期日时间（年）
    r           : 无风险年利率（连续复利）
    q           : 连续红利率（年化，如 0.03 = 3%）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    返回
    ----
    float : 期权的理论价值

    数学公式
    --------
    d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·e^(-q·T)·N(d₁) - K·e^(-r·T)·N(d₂)
    看跌：P = K·e^(-r·T)·N(-d₂) - S·e^(-q·T)·N(-d₁)

    直觉
    ----
    S·e^(-q·T) 是持有股票到期的当前价值——扣除了期间
    以 q 速率"漏出"的红利，相当于用"红利调整后股价"代入 BSM。
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # ── 红利调整后的有效股价（现值）──────────────────────────────
    # 持股至到期日实际收到的现值 = S·e^(-q·T)
    S_adj = S * exp(-q * T)

    # ── d₁, d₂：注意用 (r-q) 而非 r 作为漂移 ──────────────────
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        # 等价写法：用 S·e^(-q·T) 替代 S·e^((b-r)·T)（b=r-q 时两者相同）
        price = S_adj * N(d1) - K * exp(-r * T) * N(d2)
    elif option_type.lower() == 'put':
        price = K * exp(-r * T) * N(-d2) - S_adj * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


def forward_price(S: float, T: float, r: float, q: float) -> float:
    """
    计算股票/指数的理论远期价格 F = S · e^{(r-q)·T}。

    远期价格是"今天签合约、T 年后交割"的公平交割价格。
    r - q 是净持有成本（融资成本 r 减去红利收益 q）。
    """
    return S * exp((r - q) * T)


if __name__ == "__main__":
    print("=" * 55)
    print("Merton (1973) 连续红利期权 — 数值示例")
    print("=" * 55)

    # 示例 1（书中 Haug p.4）：
    # S=100, K=100, T=0.5, r=0.08, q=0.04, σ=0.20
    S, K, T, r, q, sigma = 100, 100, 0.5, 0.08, 0.04, 0.20
    call = merton(S, K, T, r, q, sigma, 'call')
    put  = merton(S, K, T, r, q, sigma, 'put')
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, q={q}, σ={sigma}")
    print(f"看涨（Call）= {call:.4f}  （参考值 ≈ 8.2915）")
    print(f"看跌（Put） = {put:.4f}")

    # 验证：q=0 时退化为标准 BSM
    from ch01_black_scholes_merton._01_black_scholes_1973 import black_scholes
    call_no_div = merton(S, K, T, r, 0.0, sigma, 'call')
    bsm_call    = black_scholes(S, K, T, r, sigma, 'call')
    print(f"\n当 q=0 时，Merton = {call_no_div:.4f},  BSM = {bsm_call:.4f}")
    print(f"  差值（应≈0）= {abs(call_no_div - bsm_call):.2e}")

    # 示例 2：沪深 300 指数期权（假设参数）
    print(f"\n沪深300指数期权示例：")
    print(f"S=4000, K=4000, T=0.25, r=0.025, q=0.02, σ=0.25")
    c = merton(4000, 4000, 0.25, 0.025, 0.02, 0.25, 'call')
    p = merton(4000, 4000, 0.25, 0.025, 0.02, 0.25, 'put')
    print(f"看涨 = {c:.2f},  看跌 = {p:.2f}")
