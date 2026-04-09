"""
01_margrabe_exchange.py — 以一资产换另一资产期权 (Margrabe 1978)
================================================================
【模型简介】
Margrabe (1978) 给出了"以资产 S₂ 换取资产 S₁"的欧式期权的
精确定价公式。该模型是多资产期权定价的基石，也是展示
"相对定价"的经典例子。

收益 = max(Q₁·S₁ - Q₂·S₂, 0)
      = max(S₁ - S₂, 0)  （当 Q₁=Q₂=1 时）

关键洞察：以 S₂ 作为"计价单位"（numeraire），将二维问题
降维为一维，有效期权标的变为比率 S₁/S₂。

应用：
  - 价差期权（Spread Options）的特殊情形（K=0）
  - 股票互换（Equity Swaps）
  - 并购套利期权（Merger Arbitrage）
  - Quanto 期权定价的中间步骤

参考：Margrabe, W. (1978). "The Value of an Option to Exchange One
       Asset for Another." Journal of Finance, 33(1), 177–186.
书中对应：Haug (2007), Chapter 5, Section 5.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def margrabe_exchange(S1: float, S2: float,
                      Q1: float, Q2: float,
                      T: float, r: float,
                      b1: float, b2: float,
                      sigma1: float, sigma2: float,
                      rho: float) -> float:
    """
    Margrabe (1978) 资产交换期权。

    支付 Q₁ 单位 S₂ 换取 Q₁ 单位 S₁ 的欧式期权。
    收益 = max(Q₁·S₁ - Q₂·S₂, 0)

    参数
    ----
    S1, S2   : 两个资产的当前价格
    Q1, Q2   : 名义数量（单位）
    T        : 到期时间（年）
    r        : 无风险利率
    b1, b2   : 两个资产的持有成本（b = r-q 对应连续红利）
    sigma1   : 资产 1 的年化波动率
    sigma2   : 资产 2 的年化波动率
    rho      : 两资产对数收益率的相关系数 (-1, 1)

    返回
    ----
    float : 交换期权价值

    数学公式
    --------
    有效波动率（比率 S₁/S₂ 的波动率）：
    σ₁₂ = √(σ₁² + σ₂² - 2·ρ·σ₁·σ₂)

    d₁ = [ln(Q₁S₁/Q₂S₂) + (b₁ - b₂ + σ₁₂²/2)·T] / (σ₁₂·√T)
    d₂ = d₁ - σ₁₂·√T

    V = Q₁·S₁·e^{(b₁-r)T}·N(d₁) - Q₂·S₂·e^{(b₂-r)T}·N(d₂)

    关系
    ----
    - 当 S₂ = K（常数，σ₂=0，rho=0）时退化为 BSM（b₁ = r 时）
    - 当 b₁ = b₂ = 0（期货）时，与 Black-76 类似
    """
    if T <= 0:
        return max(Q1*S1 - Q2*S2, 0.)

    # ── 有效波动率：两资产价差的波动率 ──────────────────────────
    sigma12 = sqrt(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)

    if sigma12 < 1e-10:
        # 完全相关且等波动率：期权退化为内在价值
        return max(Q1*S1*exp((b1-r)*T) - Q2*S2*exp((b2-r)*T), 0.)

    # ── d₁, d₂ ─────────────────────────────────────────────────
    d1 = (log(Q1*S1 / (Q2*S2)) + (b1 - b2 + 0.5*sigma12**2)*T) / (sigma12*sqrt(T))
    d2 = d1 - sigma12 * sqrt(T)

    # ── 定价 ─────────────────────────────────────────────────────
    # S₁ 贡献（以 e^{(b₁-r)T} 折现）
    # S₂ 贡献（以 e^{(b₂-r)T} 折现）
    price = (Q1 * S1 * exp((b1-r)*T) * N(d1)
             - Q2 * S2 * exp((b2-r)*T) * N(d2))

    return max(price, 0.)


def spread_option_margrabe(S1: float, S2: float, T: float, r: float,
                           b1: float, b2: float,
                           sigma1: float, sigma2: float, rho: float) -> float:
    """
    价差期权（零行权价情形，K=0）：收益 = max(S₁ - S₂, 0)

    这是 Margrabe 的特殊情形（K=0），给出精确解。
    对于 K > 0 的价差期权，需要近似方法（见 Kirk 近似）。
    """
    return margrabe_exchange(S1, S2, 1., 1., T, r, b1, b2, sigma1, sigma2, rho)


def two_asset_correlation_call(S1: float, S2: float, K1: float, K2: float,
                               T: float, r: float,
                               b1: float, b2: float,
                               sigma1: float, sigma2: float, rho: float) -> float:
    """
    两资产相关期权（Two-Asset Correlation Option）。

    当 S₁_T > K₁ 且 S₂_T > K₂ 时，支付 (S₂_T - K₂)。
    即：S₁ 决定是否触发，S₂ 决定收益大小。

    公式
    ----
    d₁_j = [ln(S_j/K_j) + (b_j + σ_j²/2)·T] / (σ_j·√T)
    d₂_j = d₁_j - σ_j·√T

    V = S₂·e^{(b₂-r)T}·M(d₁₁, d₁₂, ρ) - K₂·e^{-rT}·M(d₂₁, d₂₂, ρ)

    其中 M(a, b, ρ) = 二元正态 CDF
    """
    from utils.common import cbnd

    if T <= 0:
        val = (S2 - K2) if S1 > K1 and S2 > K2 else 0.
        return max(val, 0.)

    d11 = (log(S1/K1) + (b1 + 0.5*sigma1**2)*T) / (sigma1*sqrt(T))
    d21 = d11 - sigma1*sqrt(T)
    d12 = (log(S2/K2) + (b2 + 0.5*sigma2**2)*T) / (sigma2*sqrt(T))
    d22 = d12 - sigma2*sqrt(T)

    return (S2 * exp((b2-r)*T) * cbnd(d12, d11, rho)
            - K2 * exp(-r*T)  * cbnd(d22, d21, rho))


def two_asset_correlation_put(S1: float, S2: float, K1: float, K2: float,
                              T: float, r: float,
                              b1: float, b2: float,
                              sigma1: float, sigma2: float, rho: float) -> float:
    """
    两资产相关期权（看跌）。

    当 S₁_T < K₁ 且 S₂_T < K₂ 时，支付 (K₂ - S₂_T)。
    """
    from utils.common import cbnd

    if T <= 0:
        val = (K2 - S2) if S1 < K1 and S2 < K2 else 0.
        return max(val, 0.)

    d11 = (log(S1/K1) + (b1 + 0.5*sigma1**2)*T) / (sigma1*sqrt(T))
    d21 = d11 - sigma1*sqrt(T)
    d12 = (log(S2/K2) + (b2 + 0.5*sigma2**2)*T) / (sigma2*sqrt(T))
    d22 = d12 - sigma2*sqrt(T)

    return (K2 * exp(-r*T)   * cbnd(-d22, -d21, rho)
            - S2 * exp((b2-r)*T) * cbnd(-d12, -d11, rho))


if __name__ == "__main__":
    print("=" * 60)
    print("Margrabe 交换期权 & 两资产相关期权 — 数值示例")
    print("=" * 60)

    # Margrabe 示例（Haug p.207）：
    # S1=100, S2=105, Q1=Q2=1, T=0.5, r=0.10, b1=b2=0.10, σ1=0.20, σ2=0.25, ρ=0.5
    S1, S2 = 100., 105.
    T, r = 0.5, 0.10
    b1, b2 = 0.10, 0.10
    sigma1, sigma2 = 0.20, 0.25
    rho = 0.5

    ex = margrabe_exchange(S1, S2, 1., 1., T, r, b1, b2, sigma1, sigma2, rho)
    print(f"\nMargrabe 交换期权：S1={S1}, S2={S2}, ρ={rho}")
    print(f"  价值 = {ex:.4f}  （参考值 ≈ 8.0537）")

    # 有效波动率
    sig12 = sqrt(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)
    print(f"  有效波动率 σ₁₂ = {sig12:.4f}")
    print(f"  直觉：相关性越高，波动率越低，期权越便宜")

    # 不同相关系数的影响
    print(f"\n相关系数 ρ 对期权价值的影响：")
    for r_ in [-0.9, -0.5, 0, 0.5, 0.9]:
        v = margrabe_exchange(S1, S2, 1., 1., T, r, b1, b2, sigma1, sigma2, r_)
        sig = sqrt(sigma1**2 + sigma2**2 - 2*r_*sigma1*sigma2)
        print(f"  ρ={r_:+.1f}: σ₁₂={sig:.4f}, 期权价值={v:.4f}")

    # 两资产相关期权示例（Haug p.206）
    print(f"\n两资产相关看涨：S1=52, S2=65, K1=50, K2=70, T=0.5, ρ=0.75")
    tac = two_asset_correlation_call(52, 65, 50, 70, 0.5, 0.10, 0.10, 0.10, 0.20, 0.30, 0.75)
    print(f"  价值 = {tac:.4f}")
