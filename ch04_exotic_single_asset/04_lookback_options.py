"""
04_lookback_options.py — 回望期权 (Lookback Options)
=====================================================
【模型简介】
回望期权（Lookback Option，又称"无悔期权"）的收益依赖于
整个持有期内标的资产价格的历史最大值或最小值。

两大类型：
  1. 浮动行权价回望期权（Floating Strike Lookback）
     - 看涨：S_T - min(S_t)  → 以历史最低价买入
     - 看跌：max(S_t) - S_T  → 以历史最高价卖出
  2. 固定行权价回望期权（Fixed Strike Lookback）
     - 看涨：max(0, max(S_t) - K)  → 以 K 为行权价的历史最高价期权
     - 看跌：max(0, K - min(S_t))  → 以 K 为行权价的历史最低价期权

参考：Goldman, B., Sosin, H. & Gatto, M.A. (1979). "Path Dependent Options:
       Buy at the Low, Sell at the High." Journal of Finance, 34, 1111–1127.
     Conze, A. & Viswanathan, R. (1991). "Path Dependent Options: The Case
       of Lookback Options." Journal of Finance, 46, 1893–1907.
书中对应：Haug (2007), Chapter 4, Section 4.15
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def _d1_lookback(S, S_extreme, T, b, sigma, sign=1):
    """计算回望期权中的标准 d₁ 值（sign=+1 用于标准项，-1 用于反射项）。"""
    return (log(S / S_extreme) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))


# ═══════════════════════════════════════════════════════════════
# 浮动行权价回望期权
# ═══════════════════════════════════════════════════════════════

def floating_lookback_call(S: float, S_min: float, T: float,
                           r: float, b: float, sigma: float) -> float:
    """
    浮动行权价回望看涨期权（Goldman et al 1979）。
    收益 = S_T - min_{0≤t≤T}(S_t)  [以历史最低价买入股票]

    参数
    ----
    S     : 当前股价
    S_min : 当前观测到的历史最低价（期权开始至今，初始时 S_min = S）
    T     : 距到期日时间（年）
    r     : 无风险利率
    b     : 持有成本
    sigma : 年化波动率

    数学公式
    --------
    a₁ = [ln(S/S_min) + (b + σ²/2)·T] / (σ√T)
    a₂ = a₁ - σ√T
    a₃ = [ln(S/S_min) + (-b + σ²/2)·T] / (σ√T)  [注意：-b]

    若 b ≠ 0：
    C = S·e^{(b-r)T}·N(a₁) - S_min·e^{-rT}·N(a₂)
       + S·e^{-rT}·σ²/(2b) · [(S/S_min)^{-2b/σ²}·N(a₃) - e^{bT}·N(a₁)]

    若 b = 0（期货期权）：用极限形式（含 T 项）
    """
    if T <= 0:
        return max(S - S_min, 0.0)

    # 确保 S_min ≤ S（历史最低价不超过当前价）
    S_min = min(S_min, S)

    a1 = (log(S / S_min) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    a2 = a1 - sigma * sqrt(T)
    # a₃ 使用 -b（反射部分的漂移取反）
    a3 = (log(S / S_min) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

    cf = exp((b - r) * T)
    df = exp(-r * T)

    if abs(b) > 1e-10:
        # 标准情形（b ≠ 0）
        correction = (S * df * sigma**2 / (2*b)
                      * ((S / S_min)**(-2*b/sigma**2) * N(a3) - exp(b*T) * N(a1)))
        price = S * cf * N(a1) - S_min * df * N(a2) + correction
    else:
        # b = 0 时的极限形式（期货期权）
        # 当 b→0：σ²/(2b)·[...]→ 使用 L'Hôpital 极限
        price = (S * df * (N(a1) + sigma*sqrt(T) * (n(a1) + a1*N(a1)))
                 - S_min * df * N(a2))

    return max(price, S - S_min)


def floating_lookback_put(S: float, S_max: float, T: float,
                          r: float, b: float, sigma: float) -> float:
    """
    浮动行权价回望看跌期权。
    收益 = max_{0≤t≤T}(S_t) - S_T  [以历史最高价卖出股票]

    参数
    ----
    S_max : 当前观测到的历史最高价（初始时 S_max = S）
    """
    if T <= 0:
        return max(S_max - S, 0.0)

    S_max = max(S_max, S)

    b1 = (log(S / S_max) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    b2 = b1 - sigma * sqrt(T)
    b3 = (log(S / S_max) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

    cf = exp((b - r) * T)
    df = exp(-r * T)

    if abs(b) > 1e-10:
        correction = -(S * df * sigma**2 / (2*b)
                       * (-(S / S_max)**(-2*b/sigma**2) * N(-b3) + exp(b*T) * N(-b1)))
        price = S_max * df * N(-b2) - S * cf * N(-b1) + correction
    else:
        price = (S_max * df * N(-b2)
                 - S * df * (N(-b1) + sigma*sqrt(T) * (n(b1) - b1*N(-b1))))

    return max(price, S_max - S)


# ═══════════════════════════════════════════════════════════════
# 固定行权价回望期权
# ═══════════════════════════════════════════════════════════════

def fixed_lookback_call(S: float, S_max: float, K: float, T: float,
                        r: float, b: float, sigma: float) -> float:
    """
    固定行权价回望看涨期权（Conze & Viswanathan 1991）。
    收益 = max(max(S_t) - K, 0)

    参数
    ----
    S_max : 当前已观测到的历史最高价
    K     : 固定行权价

    算法
    ----
    情形1：S_max > K
      → 等效于以 S_max 为行权价的浮动行权价期权 + (S_max-K)的折现
      → 分解为 BSM call(K) + 回望溢价

    情形2：S_max ≤ K
      → 标准 BSM 格式（以 K 为行权价）
    """
    if T <= 0:
        return max(S_max - K, 0.0)

    if S_max > K:
        # 以 S_max 为实际"目前已锁定"的最高价，再加上未来路径的最大值期权
        # = BSM_call(S, K, T) + 浮动回望溢价
        # 等价于：先考虑 S_max 已超过 K 的部分，然后叠加额外上行收益

        # d 值使用 S_max（当前观察到的最大值）
        c1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        c2 = c1 - sigma*sqrt(T)
        c3 = (log(S / S_max) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        c4 = c3 - sigma*sqrt(T)
        c5 = (log(S / S_max) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

        cf = exp((b-r)*T); df = exp(-r*T)

        if abs(b) > 1e-10:
            corr = -(S * df * sigma**2 / (2*b)
                     * (-(S/S_max)**(-2*b/sigma**2)*N(-c5) + exp(b*T)*N(-c3)))
            price = (S*cf*N(c1) - K*df*N(c2)
                     + S_max*df*N(-c4) - S*cf*N(-c3) + corr)
        else:
            price = (S*cf*N(c1) - K*df*N(c2)
                     + S_max*df*N(-c4) - S*cf*N(-c3))
        return max(price, S_max - K)
    else:
        # S_max ≤ K：标准 BSM 看涨 + 上行部分（K 以上的最大值期权）
        c1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        c2 = c1 - sigma*sqrt(T)
        c3 = (log(S / K) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

        cf = exp((b-r)*T); df = exp(-r*T)

        if abs(b) > 1e-10:
            corr = S * df * sigma**2/(2*b) * ((S/K)**(-2*b/sigma**2)*N(c3) - exp(b*T)*N(c1))
            price = S*cf*N(c1) - K*df*N(c2) + corr
        else:
            price = S*cf*N(c1) - K*df*N(c2)
        return max(price, 0.0)


def fixed_lookback_put(S: float, S_min: float, K: float, T: float,
                       r: float, b: float, sigma: float) -> float:
    """
    固定行权价回望看跌期权。
    收益 = max(K - min(S_t), 0)

    参数
    ----
    S_min : 当前已观测到的历史最低价
    K     : 固定行权价
    """
    if T <= 0:
        return max(K - S_min, 0.0)

    if S_min < K:
        # 已有部分价值（S_min < K），计算额外的下行收益
        p1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        p2 = p1 - sigma*sqrt(T)
        p3 = (log(S / S_min) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        p4 = p3 - sigma*sqrt(T)
        p5 = (log(S / S_min) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

        cf = exp((b-r)*T); df = exp(-r*T)

        if abs(b) > 1e-10:
            corr = S * df * sigma**2/(2*b) * ((S/S_min)**(-2*b/sigma**2)*N(p5) - exp(b*T)*N(p3))
            price = (K*df*N(-p2) - S*cf*N(-p1)
                     + S*cf*N(p3) - S_min*df*N(p4) + corr)
        else:
            price = K*df*N(-p2) - S*cf*N(-p1) + S*cf*N(p3) - S_min*df*N(p4)
        return max(price, K - S_min)
    else:
        # S_min ≥ K：标准 BSM 看跌 + 下行部分
        p1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        p2 = p1 - sigma*sqrt(T)
        p3 = (log(S / K) + (-b + 0.5*sigma**2)*T) / (sigma*sqrt(T))

        cf = exp((b-r)*T); df = exp(-r*T)

        if abs(b) > 1e-10:
            corr = -(S * df * sigma**2/(2*b) * (-(S/K)**(-2*b/sigma**2)*N(-p3) + exp(b*T)*N(-p1)))
            price = K*df*N(-p2) - S*cf*N(-p1) + corr
        else:
            price = K*df*N(-p2) - S*cf*N(-p1)
        return max(price, 0.0)


if __name__ == "__main__":
    print("=" * 60)
    print("回望期权 — 数值示例（Haug Chapter 4, Section 4.15）")
    print("=" * 60)

    # 浮动行权价看涨（Haug p.142）：S=120, S_min=100, T=0.5, r=0.10, b=0.10, σ=0.30
    S, S_min, S_max, T, r, b, sigma = 120, 100, 130, 0.5, 0.10, 0.10, 0.30
    fl_call = floating_lookback_call(S, S_min, T, r, b, sigma)
    fl_put  = floating_lookback_put(S, S_max, T, r, b, sigma)
    print(f"\n浮动行权价：S={S}, S_min={S_min}, S_max={S_max}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"  浮动看涨 = {fl_call:.4f}  （参考值 ≈ 25.3533）")
    print(f"  浮动看跌 = {fl_put:.4f}")

    # 固定行权价（Haug p.144）：S=120, S_max=130, K=110, T=0.5
    fx_call = fixed_lookback_call(S, S_max, 110, T, r, b, sigma)
    fx_put  = fixed_lookback_put(S, S_min, 110, T, r, b, sigma)
    print(f"\n固定行权价（K=110）：")
    print(f"  固定看涨（S_max=130） = {fx_call:.4f}")
    print(f"  固定看跌（S_min=100） = {fx_put:.4f}")

    # 特殊情形：期权刚开始，S_min = S_max = S（初始时）
    fl_call_init = floating_lookback_call(100, 100, 0.5, 0.10, 0.10, 0.30)
    print(f"\n初始时浮动看涨（S=S_min=100） = {fl_call_init:.4f}")
