"""
03_compound_options.py — 期权上的期权 (Compound Options)
========================================================
【模型简介】
复合期权（Compound Option）是以另一个期权为标的物的期权，
共有四种类型：
  - 看涨上看涨（Call on Call, CaC）：以行权价 K 购买看涨期权
  - 看涨上看跌（Call on Put, CaP）：以行权价 K 购买看跌期权
  - 看跌上看涨（Put on Call, PaC）：以行权价 K 卖出看涨期权
  - 看跌上看跌（Put on Put, PaP）：以行权价 K 卖出看跌期权

应用：
  - 公司债务/股权（Merton 模型扩展）
  - 两阶段投资决策（实物期权）
  - 汇率期权的期权

参考：Geske, R. (1979). "The Valuation of Compound Options."
       Journal of Financial Economics, 7, 63–81.
书中对应：Haug (2007), Chapter 4, Section 4.13
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n, cbnd
from scipy.optimize import brentq


def _gbsm_call(S, K, T, r, b, sigma):
    if T <= 0: return max(S - K, 0.0)
    d1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*exp((b-r)*T)*N(d1) - K*exp(-r*T)*N(d2)

def _gbsm_put(S, K, T, r, b, sigma):
    if T <= 0: return max(K - S, 0.0)
    d1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*N(-d2) - S*exp((b-r)*T)*N(-d1)


def compound_option(S: float, K_outer: float, K_inner: float,
                    T1: float, T2: float,
                    r: float, b: float, sigma: float,
                    outer_type: str = 'call',
                    inner_type: str = 'call') -> float:
    """
    复合期权（Geske 1979）统一定价函数。

    参数
    ----
    S          : 当前标的资产价格（内层期权的标的）
    K_outer    : 外层期权的行权价（购买/出售内层期权的价格）
    K_inner    : 内层期权的行权价
    T1         : 外层期权到期（年，T₁ < T₂）
    T2         : 内层期权到期（年）
    r          : 无风险利率
    b          : 持有成本
    sigma      : 年化波动率
    outer_type : 外层期权类型 'call' 或 'put'
    inner_type : 内层期权类型 'call' 或 'put'

    返回
    ----
    float : 复合期权价值

    算法
    ----
    1. 寻找临界资产价格 S*：使 T₁ 时内层期权价值 = K_outer
       即：BSM_inner(S*, K_inner, T₂-T₁, r, b, σ) = K_outer
    2. 计算复合正态分布项（涉及 cbnd 二维正态 CDF）
    3. 按四种类型组合符号

    公式（Call on Call 为例）
    ────────────────────────
    a₁ = [ln(S/S*) + (b + σ²/2)·T₁] / (σ·√T₁)
    a₂ = a₁ - σ·√T₁
    b₁ = [ln(S/K_inner) + (b + σ²/2)·T₂] / (σ·√T₂)
    b₂ = b₁ - σ·√T₂
    ρ  = √(T₁/T₂)  （两个时间点的相关系数）

    CaC = S·e^{(b-r)T₂}·M(a₁, b₁, ρ)
          - K_inner·e^{-rT₂}·M(a₂, b₂, ρ)
          - K_outer·e^{-rT₁}·N(a₂)
    """
    if T1 >= T2:
        raise ValueError(f"须满足 T1({T1}) < T2({T2})")
    if T1 <= 0:
        # 外层期权已到期：直接返回内层期权价值超过 K_outer 的部分
        inner_val = (_gbsm_call(S, K_inner, T2, r, b, sigma)
                     if inner_type == 'call'
                     else _gbsm_put(S, K_inner, T2, r, b, sigma))
        if outer_type == 'call':
            return max(inner_val - K_outer, 0.0)
        return max(K_outer - inner_val, 0.0)

    # ── 步骤1：寻找临界价格 S* ────────────────────────────────────
    # S* 满足：内层期权(S*, K_inner, T₂-T₁) = K_outer
    tau = T2 - T1

    def inner_val_at_s(Sc):
        if inner_type == 'call':
            return _gbsm_call(Sc, K_inner, tau, r, b, sigma)
        return _gbsm_put(Sc, K_inner, tau, r, b, sigma)

    def objective(Sc):
        return inner_val_at_s(Sc) - K_outer

    # 寻找合适的搜索区间
    try:
        # 确定 objective 在 [lo, hi] 内有不同符号
        lo, hi = 1e-6, S * 100
        if objective(lo) * objective(hi) > 0:
            # 若无解，外层期权可能始终价内或价外
            if inner_type == 'call':
                # 当 S→∞，call → ∞ > K_outer；当 S→0，call → 0 < K_outer
                # 所以必有解，调整搜索范围
                lo, hi = 1e-8, S * 1e6
            else:
                lo, hi = 1e-8, K_inner * 10
        S_star = brentq(objective, lo, hi, xtol=1e-8, maxiter=300)
    except ValueError:
        # 无临界价格：内层期权始终 > K_outer（outer call → 欧式）
        # 或始终 < K_outer（outer call → 0）
        inner_now = inner_val_at_s(S)
        if outer_type == 'call':
            return max(inner_now - K_outer, 0.0)
        return max(K_outer - inner_now, 0.0)

    # ── 步骤2：计算辅助 d 值 ──────────────────────────────────────
    # a₁, a₂：对应临界价格 S*（期限 T₁）
    a1 = (log(S / S_star) + (b + 0.5*sigma**2)*T1) / (sigma*sqrt(T1))
    a2 = a1 - sigma * sqrt(T1)

    # b₁, b₂：对应内层行权价 K_inner（期限 T₂）
    b1 = (log(S / K_inner) + (b + 0.5*sigma**2)*T2) / (sigma*sqrt(T2))
    b2 = b1 - sigma * sqrt(T2)

    # ρ = 相关系数 = √(T₁/T₂)
    rho = sqrt(T1 / T2)

    # ── 步骤3：按类型计算 ────────────────────────────────────────
    cf2 = exp((b - r) * T2)   # 持有成本因子（T₂）
    df1 = exp(-r * T1)         # 折现因子（T₁）
    df2 = exp(-r * T2)         # 折现因子（T₂）

    if outer_type == 'call' and inner_type == 'call':
        # 看涨上看涨（Call on Call）
        # 当 S_T₁ > S*（内层看涨有价值）→ 行权
        price = (S * cf2 * cbnd(a1, b1, rho)
                 - K_inner * df2 * cbnd(a2, b2, rho)
                 - K_outer * df1 * N(a2))

    elif outer_type == 'call' and inner_type == 'put':
        # 看涨上看跌（Call on Put）
        # 当 S_T₁ < S*（内层看跌有价值）→ 行权
        price = (K_inner * df2 * cbnd(-a2, -b2, rho)
                 - S * cf2 * cbnd(-a1, -b1, rho)
                 - K_outer * df1 * N(-a2))

    elif outer_type == 'put' and inner_type == 'call':
        # 看跌上看涨（Put on Call）
        price = (K_inner * df2 * cbnd(a2, -b2, -rho)
                 - S * cf2 * cbnd(a1, -b1, -rho)
                 + K_outer * df1 * N(a2))

    elif outer_type == 'put' and inner_type == 'put':
        # 看跌上看跌（Put on Put）
        price = (S * cf2 * cbnd(-a1, b1, -rho)
                 - K_inner * df2 * cbnd(-a2, b2, -rho)
                 + K_outer * df1 * N(-a2))
    else:
        raise ValueError(f"无效的期权类型组合：{outer_type} on {inner_type}")

    return max(price, 0.0)


if __name__ == "__main__":
    print("=" * 60)
    print("复合期权（Geske 1979）— 数值示例")
    print("=" * 60)

    # 示例（Haug p.133）：S=500, K_outer=50, K_inner=520, T1=0.25, T2=0.5
    S, K_o, K_i, T1, T2, r, b, sigma = 500, 50, 520, 0.25, 0.5, 0.08, 0.08, 0.35
    print(f"\n参数：S={S}, K_outer={K_o}, K_inner={K_i}, T1={T1}, T2={T2}")
    print(f"       r={r}, b={b}, σ={sigma}")

    cac = compound_option(S, K_o, K_i, T1, T2, r, b, sigma, 'call', 'call')
    cap = compound_option(S, K_o, K_i, T1, T2, r, b, sigma, 'call', 'put')
    pac = compound_option(S, K_o, K_i, T1, T2, r, b, sigma, 'put',  'call')
    pap = compound_option(S, K_o, K_i, T1, T2, r, b, sigma, 'put',  'put')
    print(f"\n  Call on Call = {cac:.4f}  （参考值 ≈ 17.6217）")
    print(f"  Call on Put  = {cap:.4f}")
    print(f"  Put  on Call = {pac:.4f}")
    print(f"  Put  on Put  = {pap:.4f}")

    # 验证 Put-Call Parity for compound options
    # CaC - PaC = C(S, K_inner, T2) - K_outer * exp(-r*T1)
    inner_call = _gbsm_call(S, K_i, T2, r, b, sigma)
    parity_lhs = cac - pac
    parity_rhs = inner_call - K_o * exp(-r * T1)
    print(f"\n  复合期权平价验证 (CaC - PaC = C - K·e^(-rT1)):")
    print(f"  左边 = {parity_lhs:.4f}, 右边 = {parity_rhs:.4f}, 误差 = {abs(parity_lhs-parity_rhs):.2e}")
