"""
01_black_scholes_1973.py — Black-Scholes (1973) 期权定价模型
=============================================================
【模型简介】
Black 和 Scholes (1973) 发表了期权定价领域的突破性成果，给出了
欧式期权的解析定价公式。该模型假设：
  - 股票价格服从几何布朗运动（对数正态分布）
  - 无连续红利（原始版本）
  - 无摩擦市场，可持续动态对冲
  - 无风险利率和波动率为常数

公式核心思想：复制对冲组合（delta hedging），通过持有
Δ 份股票和借款来完全复制期权收益，无套利条件下得到定价。

参考：Black, F. & Scholes, M. (1973). "The Pricing of Options and
       Corporate Liabilities." Journal of Political Economy, 81(3), 637–654.

书中对应：Haug (2007), Chapter 1, Section 1.1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, option_type: str = 'call') -> float:
    """
    Black-Scholes (1973) 欧式期权定价公式。

    适用于：不支付红利的股票的欧式期权。

    参数
    ----
    S           : 当前股票价格（标的资产现价）
    K           : 行权价格（Strike Price）
    T           : 距到期日的时间，单位：年（如 0.5 = 6个月）
    r           : 无风险年利率（连续复利，如 0.05 = 5%）
    sigma       : 股票价格的年化波动率（如 0.20 = 20%）
    option_type : 'call'（看涨）或 'put'（看跌）

    返回
    ----
    float : 期权的理论价值

    数学公式
    --------
    d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨期权：C = S·N(d₁) - K·e^(-rT)·N(d₂)
    看跌期权：P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

    直觉解释
    --------
    - S·N(d₁)：期望收到股票的现值（概率加权）
    - K·e^(-rT)·N(d₂)：期望支付行权价的现值
    - N(d₂)：风险中性概率下，到期时期权价内（in-the-money）的概率
    - N(d₁)：delta，即对冲需要持有的股票份数
    """
    if T <= 0:
        # 到期：返回内在价值
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # ── 计算 d₁ 和 d₂ ──────────────────────────────────────────
    # d₁ 综合了价格对比（ln S/K）、时间价值（r·T）和波动率溢价（σ²/2·T）
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    # d₂ = d₁ 减去一个标准差，对应风险中性 ITM 概率
    d2 = d1 - sigma * sqrt(T)

    # ── 按期权类型计算价格 ────────────────────────────────────────
    if option_type.lower() == 'call':
        # 看涨：持有标的资产（S·N(d₁)）减去借款的现值（K·e^(-rT)·N(d₂)）
        price = S * N(d1) - K * exp(-r * T) * N(d2)
    elif option_type.lower() == 'put':
        # 看跌：行权收入的现值（K·e^(-rT)·N(-d₂)）减去标的资产价值（S·N(-d₁)）
        price = K * exp(-r * T) * N(-d2) - S * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际传入：{option_type}")

    return price


def put_call_parity_check(S: float, K: float, T: float, r: float,
                          sigma: float) -> dict:
    """
    验证看涨-看跌平价关系（Put-Call Parity）。

    平价公式：C - P = S - K·e^(-rT)
    即：看涨价格 - 看跌价格 = 股票现价 - 行权价的现值

    该关系是无套利定价的核心约束，与模型无关（适用于任何欧式期权）。

    返回
    ----
    dict 包含 call, put, 左侧(C-P), 右侧(S-K·e^(-rT)) 和误差
    """
    call = black_scholes(S, K, T, r, sigma, 'call')
    put  = black_scholes(S, K, T, r, sigma, 'put')
    lhs  = call - put                          # C - P
    rhs  = S - K * exp(-r * T)                 # S - K·e^{-rT}
    return {
        'call'  : call,
        'put'   : put,
        'C - P' : lhs,
        'S - K*exp(-rT)': rhs,
        'parity_error'  : abs(lhs - rhs)
    }


if __name__ == "__main__":
    print("=" * 55)
    print("Black-Scholes (1973) 期权定价 — 数值示例")
    print("=" * 55)

    # 书中 Example 1（Haug p.3）：
    # S=60, K=65, T=0.25, r=0.08, σ=0.30
    S, K, T, r, sigma = 60, 65, 0.25, 0.08, 0.30
    call = black_scholes(S, K, T, r, sigma, 'call')
    put  = black_scholes(S, K, T, r, sigma, 'put')
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"看涨期权（Call）价格 = {call:.4f}  （书中参考值 ≈ 2.1334）")
    print(f"看跌期权（Put） 价格 = {put:.4f}")

    # 验证 Put-Call Parity
    parity = put_call_parity_check(S, K, T, r, sigma)
    print(f"\nPut-Call Parity 验证：")
    print(f"  C - P         = {parity['C - P']:.6f}")
    print(f"  S - K·e^(-rT) = {parity['S - K*exp(-rT)']:.6f}")
    print(f"  平价误差       = {parity['parity_error']:.2e}  （应接近 0）")

    # 平值期权示例
    print(f"\n平值期权（ATM）：S=K=100, T=1, r=0.05, σ=0.20")
    c2 = black_scholes(100, 100, 1, 0.05, 0.20, 'call')
    p2 = black_scholes(100, 100, 1, 0.05, 0.20, 'put')
    print(f"看涨 = {c2:.4f},  看跌 = {p2:.4f}")
