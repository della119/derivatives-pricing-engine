"""
03_black_76.py — Black (1976) 期货/远期期权定价模型
=====================================================
【模型简介】
Black (1976) 将 BSM 模型适配到商品期货/远期合约上的期权定价。
核心差异：期货合约无需前期投资（持有成本 b = 0），因此：
  - 期货价格本身即"已折现"的预期现货价
  - 公式中用期货价格 F 直接替换，折现率用于贴现到期支付

Black-76 是大量利率衍生品（Caps、Floors、Swaptions）定价的标准框架。

参考：Black, F. (1976). "The Pricing of Commodity Contracts."
       Journal of Financial Economics, 3(1–2), 167–179.

书中对应：Haug (2007), Chapter 1, Section 1.1.3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def black_76(F: float, K: float, T: float, r: float,
             sigma: float, option_type: str = 'call') -> float:
    """
    Black (1976) 期货/远期期权定价公式。

    参数
    ----
    F           : 期货（或远期）价格
    K           : 行权价格
    T           : 距到期日时间（年）
    r           : 无风险年利率（连续复利），用于折现到期支付
    sigma       : 期货价格的年化波动率
    option_type : 'call' 或 'put'

    返回
    ----
    float : 期权的理论价值

    数学公式
    --------
    d₁ = [ln(F/K) + σ²/2 · T] / (σ·√T)
    d₂ = d₁ - σ·√T = [ln(F/K) - σ²/2 · T] / (σ·√T)

    看涨：C = e^(-r·T) · [F·N(d₁) - K·N(d₂)]
    看跌：P = e^(-r·T) · [K·N(-d₂) - F·N(-d₁)]

    与 BSM 的关系
    -------------
    Black-76 = BSM with b = 0（持有成本为零）
    即：把 BSM 中的 S·e^{b·T} 换成 F（期货价格即含持有成本的"远期价"）。
    等价地：generalized_bsm(F, K, T, r, b=0, sigma)
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    # ── 折现因子 ────────────────────────────────────────────────
    discount = exp(-r * T)

    # ── d₁, d₂：b=0，故无漂移调整项 ────────────────────────────
    d1 = (log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        # 期望收益 = E[max(F_T - K, 0)]，折现到今天
        price = discount * (F * N(d1) - K * N(d2))
    elif option_type.lower() == 'put':
        price = discount * (K * N(-d2) - F * N(-d1))
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


def implied_vol_black76(F: float, K: float, T: float, r: float,
                        market_price: float, option_type: str = 'call',
                        tol: float = 1e-7, max_iter: int = 200) -> float:
    """
    用 Black-76 反推隐含波动率（Newton-Raphson 方法）。

    参数
    ----
    F            : 期货价格
    K            : 行权价
    T            : 到期时间（年）
    r            : 无风险利率
    market_price : 市场报价
    option_type  : 'call' 或 'put'
    tol          : 收敛容差
    max_iter     : 最大迭代次数

    返回
    ----
    float : 隐含波动率（年化）
    """
    from math import log, sqrt, exp

    # 初始猜测：用 Brenner-Subrahmanyam 近似
    sigma = sqrt(2 * abs(log(F / K)) / T) if F != K else 0.5

    for _ in range(max_iter):
        price = black_76(F, K, T, r, sigma, option_type)
        # Vega = F · e^(-rT) · n(d1) · √T
        d1 = (log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
        vega = F * exp(-r * T) * n(d1) * sqrt(T)
        if abs(vega) < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(1e-6, sigma)   # 波动率保持正值

    return sigma


if __name__ == "__main__":
    print("=" * 55)
    print("Black (1976) 期货期权定价 — 数值示例")
    print("=" * 55)

    # 示例（Haug p.4）：原油期货期权
    # F=105, K=100, T=0.5, r=0.10, σ=0.36
    F, K, T, r, sigma = 105, 100, 0.5, 0.10, 0.36
    call = black_76(F, K, T, r, sigma, 'call')
    put  = black_76(F, K, T, r, sigma, 'put')
    print(f"\n参数：F={F}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"看涨（Call）= {call:.4f}  （参考值 ≈ 15.6884）")
    print(f"看跌（Put） = {put:.4f}")

    # 利率期权示例：欧元区利率上限（Caplet）
    print(f"\nCaplet 示例（利率期权）：F=0.065, K=0.06, T=1, r=0.05, σ=0.20")
    caplet_call = black_76(0.065, 0.060, 1.0, 0.05, 0.20, 'call')
    print(f"Caplet 价格（未乘名义本金和 day count）= {caplet_call:.6f}")

    # 隐含波动率反推示例
    print(f"\n隐含波动率反推：市场价格 = {call:.4f}")
    iv = implied_vol_black76(F, K, T, r, call, 'call')
    print(f"反推隐含波动率 = {iv:.6f}  （应等于 {sigma}）")
