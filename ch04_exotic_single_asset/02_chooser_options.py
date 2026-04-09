"""
02_chooser_options.py — 选择权 (Chooser Options)
=================================================
【模型简介】
选择权（又称"随你期权"，As-You-Like-It Option）赋予持有人
在未来某时刻 t_c 选择持有看涨或看跌期权的权利。

两种类型：
  - 简单选择权（Simple Chooser）：两个期权有相同的 K 和 T
  - 复杂选择权（Complex Chooser）：允许不同 K 和 T

直觉：选择权价值 = 看涨 + 看跌（中至少较大者）的期望值，
因此常见定价分解为"欧式看涨 + 调整后的看跌"

参考：Rubinstein, M. (1991). "Options for the Undecided."
       RISK Magazine, Vol. 4, 43.
     Rubinstein, M. (1992). "Exotic Options." UC Berkeley Working Paper.
书中对应：Haug (2007), Chapter 4, Section 4.12
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n, cbnd
from scipy.optimize import brentq


def _gbsm(S, K, T, r, b, sigma, opt):
    """广义 BSM 辅助函数。"""
    if T <= 0:
        return max(S - K, 0.0) if opt == 'call' else max(K - S, 0.0)
    d1 = (log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    cf = exp((b - r) * T); df = exp(-r * T)
    if opt == 'call':
        return S * cf * N(d1) - K * df * N(d2)
    return K * df * N(-d2) - S * cf * N(-d1)


def simple_chooser(S: float, K: float, tc: float, T: float,
                   r: float, b: float, sigma: float) -> float:
    """
    简单选择权定价（Rubinstein 1991）。

    在时刻 tc，持有人可选择继续持有：
      - 看涨期权（K, T）或
      - 看跌期权（K, T）
    取两者中价值较高者。

    参数
    ----
    S     : 当前股价
    K     : 行权价（看涨和看跌相同）
    tc    : 选择时刻（年，tc < T）
    T     : 期权到期时刻（年）
    r     : 无风险利率
    b     : 持有成本
    sigma : 年化波动率

    数学公式
    --------
    利用 Put-Call Parity，在 tc 时刻：
      max(C, P) = C + max(0, P - C)
               = C + max(0, K·e^{-r(T-tc)} - S_{tc}·e^{(b-r)(T-tc)})

    因此选择权价值 = 欧式看涨(T) + 欧式看跌(K*, tc)
    其中 K* = K·e^{-(r-b)(T-tc)}（即调整后的等效行权价）

    d₁(T)  = [ln(S/K) + (b + σ²/2)·T] / (σ·√T)
    d₂(T)  = d₁(T) - σ·√T
    d*(tc) = [ln(S/K*) + (b + σ²/2)·tc] / (σ·√tc)
           = [ln(S/K) + (b - r + σ²/2)·tc + (b)(T-tc)] / (σ·√tc) [展开]

    Chooser = S·e^{(b-r)T}·N(d₁) - K·e^{-rT}·N(d₂)
             + K·e^{-rT}·N(-d**) - S·e^{(b-r)tc}·N(-d*)
    """
    if tc <= 0 or T <= 0 or tc >= T:
        raise ValueError(f"须满足 0 < tc({tc}) < T({T})")

    # ── 参数计算 ─────────────────────────────────────────────────
    tau = T - tc   # 选择后的剩余期

    # d₁, d₂ 对应完整期限 T 的标准欧式看涨
    d1 = (log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # 等效看跌行权价：K* = K·e^{-(r-b)·τ}（调整了选择后期间）
    K_star = K * exp(-(r - b) * tau)

    # d* 对应调整后看跌（以 K* 为行权价，期限 tc）
    d_star1 = (log(S / K_star) + (b + 0.5 * sigma**2) * tc) / (sigma * sqrt(tc))
    d_star2 = d_star1 - sigma * sqrt(tc)

    # ── 定价：看涨(T) + 调整看跌(tc, K*) ────────────────────────
    # 看涨部分（标准 BSM）
    call_part = (S * exp((b - r) * T) * N(d1)
                 - K * exp(-r * T) * N(d2))

    # 看跌部分（行权价 K*，期限 tc）— 使用广义 BSM 计算
    put_part = _gbsm(S, K_star, tc, r, b, sigma, 'put')

    return call_part + put_part


def complex_chooser(S: float, K_c: float, K_p: float,
                    T_c: float, T_p: float, tc: float,
                    r: float, b: float, sigma: float) -> float:
    """
    复杂选择权定价（Rubinstein 1992）。

    允许看涨和看跌有不同的行权价和到期日：
      - 看涨：行权价 K_c，到期 T_c
      - 看跌：行权价 K_p，到期 T_p
    在 tc 时选择两者之一（tc < min(T_c, T_p)）。

    参数
    ----
    S    : 当前股价
    K_c  : 看涨行权价
    K_p  : 看跌行权价
    T_c  : 看涨到期（年）
    T_p  : 看跌到期（年）
    tc   : 选择时刻（年）
    r    : 无风险利率
    b    : 持有成本
    sigma: 年化波动率

    算法
    ----
    在 tc 时：若 C(S_{tc}, K_c, T_c-tc) ≥ P(S_{tc}, K_p, T_p-tc)
              则选看涨，否则选看跌

    存在临界价格 S* 满足：C(S*, K_c, T_c-tc) = P(S*, K_p, T_p-tc)
    → 用数值方法（二分法）求解 S*

    定价公式（双正态积分）
    ────────────────────
    V = C部分（使用二维正态）+ P部分（使用二维正态）

    看涨部分：
    = S·e^{(b-r)Tc}·M(d₁,y₁,√(tc/Tc)) - Kc·e^{-rTc}·M(d₂,y₂,√(tc/Tc))

    看跌部分：
    = -S·e^{(b-r)Tp}·M(-d₃,-y₃,√(tc/Tp)) + Kp·e^{-rTp}·M(-d₄,-y₄,√(tc/Tp))
    """
    if tc <= 0 or tc >= min(T_c, T_p):
        raise ValueError("须满足 0 < tc < min(T_c, T_p)")

    # ── 寻找临界价格 S*（二分法）────────────────────────────────
    def objective(Sc):
        call_val = _gbsm(Sc, K_c, T_c - tc, r, b, sigma, 'call')
        put_val  = _gbsm(Sc, K_p, T_p - tc, r, b, sigma, 'put')
        return call_val - put_val

    try:
        # 搜索范围 [1e-4, 1e6]（对绝大多数实际情形足够）
        S_star = brentq(objective, S * 1e-4, S * 1e3, xtol=1e-8)
    except ValueError:
        # 退化情形：无法找到交叉点，保守地返回两者最大值
        return max(_gbsm(S, K_c, T_c, r, b, sigma, 'call'),
                   _gbsm(S, K_p, T_p, r, b, sigma, 'put'))

    # ── 辅助 d 值 ────────────────────────────────────────────────
    def _d1(T_):
        return (log(S / S_star) + (b + 0.5 * sigma**2) * tc) / (sigma * sqrt(tc))

    # y 值：对应临界价格 S* 的 d₁（按 tc 期）
    y1 = (log(S / S_star) + (b + 0.5 * sigma**2) * tc) / (sigma * sqrt(tc))
    y2 = y1 - sigma * sqrt(tc)

    # d 值：对应各自行权价的 d₁（按完整期 T_c、T_p）
    d1_c = (log(S / K_c) + (b + 0.5 * sigma**2) * T_c) / (sigma * sqrt(T_c))
    d2_c = d1_c - sigma * sqrt(T_c)
    d1_p = (log(S / K_p) + (b + 0.5 * sigma**2) * T_p) / (sigma * sqrt(T_p))
    d2_p = d1_p - sigma * sqrt(T_p)

    rho_c = sqrt(tc / T_c)   # 两个时间点的相关系数（看涨）
    rho_p = sqrt(tc / T_p)   # （看跌）

    # ── 看涨部分（S > S* 时选看涨）────────────────────────────────
    call_part = (S * exp((b - r) * T_c) * cbnd(d1_c, y1,  rho_c)
                 - K_c * exp(-r * T_c) * cbnd(d2_c, y2,  rho_c))

    # ── 看跌部分（S < S* 时选看跌）────────────────────────────────
    put_part = (-S * exp((b - r) * T_p) * cbnd(-d1_p, -y1, rho_p)
                + K_p * exp(-r * T_p)  * cbnd(-d2_p, -y2, rho_p))

    return call_part + put_part


if __name__ == "__main__":
    print("=" * 55)
    print("选择权 — 数值示例")
    print("=" * 55)

    # 简单选择权（Haug p.129）：S=50, K=50, tc=0.25, T=0.5, r=0.08, b=0.08, σ=0.25
    S, K, tc, T, r, b, sigma = 50, 50, 0.25, 0.5, 0.08, 0.08, 0.25
    sc = simple_chooser(S, K, tc, T, r, b, sigma)
    print(f"\n简单选择权：S={S}, K={K}, tc={tc}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"  价值 = {sc:.4f}  （参考值 ≈ 6.1071）")

    # 对比：看涨+看跌价格
    c = _gbsm(S, K, T, r, b, sigma, 'call')
    p = _gbsm(S, K, T, r, b, sigma, 'put')
    print(f"  看涨 = {c:.4f}, 看跌 = {p:.4f}, max = {max(c,p):.4f}")
    print(f"  （选择权 > max 因为选择权有等待价值）")

    # 复杂选择权
    print(f"\n复杂选择权：S=50, K_c=55, K_p=48, T_c=0.5, T_p=0.75, tc=0.25")
    S2, K_c, K_p = 50, 55, 48
    T_c, T_p, tc2, r2, b2, s2 = 0.5, 0.75, 0.25, 0.08, 0.08, 0.25
    cc = complex_chooser(S2, K_c, K_p, T_c, T_p, tc2, r2, b2, s2)
    print(f"  价值 = {cc:.4f}")
