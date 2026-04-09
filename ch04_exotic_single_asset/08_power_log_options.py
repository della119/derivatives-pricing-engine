"""
08_power_log_options.py — 幂式期权与对数期权
============================================
【模型简介】
幂式合约和期权的收益是标的资产价格的幂次函数，
对数合约的收益是价格的对数。这些是构建方差互换
（Variance Swap）和波动率互换的重要基础工具。

涵盖：
  1. 幂式合约（Power Contract）：收益 = S_T^i
  2. 标准幂式期权（Standard Power / Asymmetric）：max(S^i - K, 0)
  3. 封顶幂式期权（Capped Power / Capped Asymmetric）：min(max(S^i - K,0), cap)
  4. 对称幂式期权（Powered / Symmetric）：max(S - K, 0)^i
  5. 对数合约（Log Contract）：ln(S_T)
  6. 对数期权（Log Option）：max(ln(S_T/F) - K_log, 0)

书中对应：Haug (2007), Chapter 4, Sections 4.4–4.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# 1. 幂式合约（Power Contract）
# ═══════════════════════════════════════════════════════════════

def power_contract(S: float, T: float, r: float, b: float,
                   sigma: float, i: float) -> float:
    """
    幂式合约（Power Contract）。

    收益 = S_T^i（在到期日支付标的资产价格的 i 次幂）

    参数
    ----
    S     : 当前股价
    T     : 到期时间（年）
    r     : 无风险利率
    b     : 持有成本
    sigma : 年化波动率
    i     : 幂次（可为小数，如 i=2 → 收益 = S_T²）

    公式
    ----
    V = S^i · e^{[(i-1)·(b + i·σ²/2) - r]·T}

    推导：E^Q[S_T^i] = S^i · exp{[i·b + i·(i-1)/2·σ²]·T}
         折现后：V = e^{-rT} · E^Q[S_T^i]
                  = S^i · exp{[(i·b + i(i-1)σ²/2) - r]·T}
                  = S^i · exp{[(i-1)·(b + iσ²/2) - r]·T}（化简）
    """
    exponent = (i - 1) * (b + i * sigma**2 / 2) - r
    return S**i * exp(exponent * T)


# ═══════════════════════════════════════════════════════════════
# 2. 标准幂式期权（Standard Power Option / Asymmetric）
# ═══════════════════════════════════════════════════════════════

def standard_power_option(S: float, K: float, T: float, r: float, b: float,
                          sigma: float, i: float,
                          option_type: str = 'call') -> float:
    """
    标准幂式期权（Standard Power Option，非对称幂式）。

    收益 = max(S_T^i - K, 0)  [看涨]
    收益 = max(K - S_T^i, 0)  [看跌]

    参数
    ----
    i : 幂次（如 i=2 → 股价平方的看涨期权）

    公式
    ----
    将 S^i 视为新的"资产"，其对数正态分布参数为：
    - 有效持有成本：b_i = i·b + i·(i-1)·σ²/2
    - 有效波动率：σ_i = i·σ
    - 有效"远期价格"：F_i = S^i · exp(b_i · T)

    看涨：C = e^{-rT} · [F_i · N(d₁) - K · N(d₂)]
    其中：d₁ = [ln(S^i/K) + (b_i + σ_i²/2)·T] / (σ_i·√T)
         d₂ = d₁ - σ_i·√T
    """
    if T <= 0:
        power = S**i
        if option_type == 'call': return max(power - K, 0.)
        return max(K - power, 0.)

    # 幂式变换后的有效参数
    b_i   = i * b + i * (i-1) * sigma**2 / 2   # 有效持有成本
    sigma_i = i * sigma                          # 有效波动率（线性放大）
    S_i   = S**i                                  # 有效"当前资产价格"

    d1 = (log(S_i / K) + (b_i + 0.5*sigma_i**2)*T) / (sigma_i*sqrt(T))
    d2 = d1 - sigma_i * sqrt(T)

    cf = exp((b_i - r)*T)
    df = exp(-r * T)

    if option_type.lower() == 'call':
        return S_i * cf * N(d1) - K * df * N(d2)
    return K * df * N(-d2) - S_i * cf * N(-d1)


# ═══════════════════════════════════════════════════════════════
# 3. 封顶幂式期权（Capped Power Option）
# ═══════════════════════════════════════════════════════════════

def capped_power_option(S: float, K: float, cap: float, T: float,
                        r: float, b: float, sigma: float, i: float,
                        option_type: str = 'call') -> float:
    """
    封顶幂式期权（Capped Power Option）。

    收益 = min(max(S_T^i - K, 0), cap)
         = max(S_T^i - K, 0) - max(S_T^i - K - cap, 0)

    等于：两个标准幂式期权的价差（牛市价差）
    """
    long  = standard_power_option(S, K,       T, r, b, sigma, i, option_type)
    short = standard_power_option(S, K + cap, T, r, b, sigma, i, option_type)
    return long - short


# ═══════════════════════════════════════════════════════════════
# 4. 对称幂式期权（Powered Option）
# ═══════════════════════════════════════════════════════════════

def powered_option(S: float, K: float, T: float, r: float, b: float,
                   sigma: float, i: int,
                   option_type: str = 'call') -> float:
    """
    对称幂式期权（Powered Option / Symmetric Power Option）。

    收益 = max(S_T - K, 0)^i  [看涨的 i 次幂]
    或    max(K - S_T, 0)^i   [看跌的 i 次幂]

    对于整数幂 i，可以用二项展开精确计算（Haug 2007）：
    E[max(S-K, 0)^i] = Σ_{j=0}^{i} C(i,j) · (-K)^{i-j} · E[S^j · 1_{S>K}]

    其中 E[S^j · 1_{S>K}] 是资产或无期权（Asset-or-Nothing）的 j 阶矩

    参数
    ----
    i : 正整数幂次（通常取 2 或 3）
    """
    from math import comb

    if T <= 0:
        x = max(S - K, 0.) if option_type == 'call' else max(K - S, 0.)
        return x**i

    # 使用 BSM-类公式的级数展开（对于 i≥1 的整数）
    d2 = (log(S/K) + (b - 0.5*sigma**2)*T) / (sigma*sqrt(T))

    # Σ_{j=0}^{i} C(i,j) · (-K)^{i-j} · moment_j
    total = 0.
    for j in range(i+1):
        # moment_j = E^Q[S_T^j · 1_{S_T > K}] (风险中性期望)
        # = S^j · exp{[j·b + j(j-1)σ²/2 - r]·T} · N(d1_j)
        # 其中 d1_j = [ln(S/K) + (b + (j-0.5)σ²)·T] / (σ√T)
        if j == 0:
            # E^Q[1_{S_T>K}] = e^{-rT} · N(d₂) ... 但这里要算贡献
            moment_j = exp(-r*T) * N(d2 if option_type=='call' else -d2)
        else:
            d1_j = (log(S/K) + (b + (j - 0.5)*sigma**2)*T) / (sigma*sqrt(T))
            exponent_j = (j*b + j*(j-1)*sigma**2/2 - r)*T
            moment_j = S**j * exp(exponent_j) * N(d1_j if option_type=='call' else -d1_j)

        sign = (-K)**(i - j)
        total += comb(i, j) * sign * moment_j

    return max(total, 0.)


# ═══════════════════════════════════════════════════════════════
# 5. 对数合约（Log Contract）
# ═══════════════════════════════════════════════════════════════

def log_contract(S: float, T: float, r: float, b: float, sigma: float) -> float:
    """
    对数合约（Log Contract）。

    收益 = ln(S_T)（到期时支付对数价格）

    公式
    ----
    V = e^{-rT} · E^Q[ln(S_T)]
      = e^{-rT} · [ln(S) + (b - σ²/2)·T]

    重要性：对数合约是方差互换（Variance Swap）复制策略的
           核心组成部分。方差互换价格 = -2·对数合约价格/T
    """
    return exp(-r*T) * (log(S) + (b - 0.5*sigma**2)*T)


def log_return_contract(S: float, T: float, r: float, b: float, sigma: float) -> float:
    """
    对数收益合约（Log Return Contract）。

    收益 = ln(S_T / S_0)（对数收益率）

    公式：V = e^{-rT} · (b - σ²/2) · T
    """
    return exp(-r*T) * (b - 0.5*sigma**2) * T


# ═══════════════════════════════════════════════════════════════
# 6. 对数期权（Log Option）
# ═══════════════════════════════════════════════════════════════

def log_option(S: float, K_log: float, T: float, r: float, b: float,
               sigma: float, option_type: str = 'call') -> float:
    """
    对数期权（Log Option）。

    收益 = max(ln(S_T/F) - K_log, 0)  [看涨]
    其中 F = S·e^{bT} 是远期价格

    直觉：以远期价格为基准，对数收益超过 K_log 才有收益

    公式
    ----
    由于 ln(S_T/F) ~ N(-(σ²/2)·T, σ²·T)，令：
    d = (-K_log - σ²/2·T) / (σ√T)

    看涨：C = e^{-rT} · [-(K_log + σ²/2·T)·N(d) + σ·√T·n(d)]
    看跌：P = e^{-rT} · [ (K_log + σ²/2·T)·N(-d) + σ·√T·n(d)]
    """
    if T <= 0:
        F = S * exp(b * T)
        log_ret = log(S/F) - K_log
        if option_type == 'call': return max(log_ret, 0.)
        return max(-log_ret, 0.)

    # 中心化的对数均值：E^Q[ln(S_T/F)] = -σ²T/2
    mu_log = -sigma**2 / 2 * T   # ln(S_T/F) 的期望
    vol_log = sigma * sqrt(T)     # ln(S_T/F) 的标准差

    d = (mu_log - K_log) / vol_log   # 即 (-σ²T/2 - K_log) / (σ√T)

    df = exp(-r*T)
    if option_type.lower() == 'call':
        return df * ((mu_log - K_log) * N(d) + vol_log * n(d))
    return df * ((K_log - mu_log) * N(-d) + vol_log * n(d))


if __name__ == "__main__":
    print("=" * 60)
    print("幂式期权 & 对数期权 — 数值示例（Haug Chapter 4）")
    print("=" * 60)

    S, K, T, r, b, sigma = 100, 100, 0.5, 0.08, 0.08, 0.30

    # 幂式合约
    pc = power_contract(S, T, r, b, sigma, i=2)
    print(f"\n幂式合约 (i=2, S²)：V = {pc:.2f}  （理论值 ≈ S²·e^{...}）")

    # 标准幂式期权
    spo_call = standard_power_option(S, 10000, T, r, b, sigma, i=2, option_type='call')
    spo_call2 = standard_power_option(S, 90**2, T, r, b, sigma, i=2, option_type='call')
    print(f"标准幂式看涨 (i=2, K=10000=100²)：C = {spo_call:.4f}")
    print(f"标准幂式看涨 (i=2, K=8100=90²)：C = {spo_call2:.4f}")

    # 封顶幂式
    cap_val = capped_power_option(S, 10000, 500, T, r, b, sigma, i=2)
    print(f"封顶幂式 (cap=500)：C = {cap_val:.4f}")

    # 对称幂式
    pow2 = powered_option(S, K, T, r, b, sigma, i=2, option_type='call')
    print(f"对称幂式 (i=2, (max(S-K,0))²)：V = {pow2:.4f}")

    # 对数合约和期权
    lc = log_contract(S, T, r, b, sigma)
    lr = log_return_contract(S, T, r, b, sigma)
    print(f"\n对数合约（收益=ln(S_T)）：V = {lc:.4f}")
    print(f"对数收益合约（收益=ln(S_T/S_0)）：V = {lr:.6f}")

    lo_call = log_option(S, 0.05, T, r, b, sigma, 'call')
    lo_put  = log_option(S, 0.05, T, r, b, sigma, 'put')
    print(f"对数期权（K_log=0.05）：看涨={lo_call:.6f}, 看跌={lo_put:.6f}")
