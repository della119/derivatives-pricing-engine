"""
01_discrete_dividends.py — 离散红利期权定价
============================================
【模型简介】
连续红利（通过持有成本 b = r - q 处理）是一个近似。
实践中，股票的红利往往在特定日期以固定金额发放。
在这种情形下，BSM 假设受到破坏，需要专门处理。

本文件实现三种处理离散红利的方法：

A. 疏散红利法（Escrowed Dividend Model）
────────────────────────────────────────
将股价中的红利现值部分"扣除"，得到调整后的股价 S*：
S* = S - PV(D)  = S - Σ Dᵢ · e^{-r·tᵢ}

然后对 S* 用标准 BSM 定价。
适用于：少量、确定性红利；仅用于欧式期权。

B. Roll-Geske-Whaley (RGW) 精确模型
────────────────────────────────────
Roll (1977), Geske (1979), Whaley (1981) 给出含单一离散红利的
美式看涨期权的精确解析公式（利用复合期权）。

核心思路：
- 在红利发放前（τ = t - t_D），期权持有者可能选择提前行权
- 若 S_D > S_D*（临界价格），立即行权更有利（获取红利）
- 期权价值 = 欧式部分 + 提前行权溢价（通过 cbnd 计算）

C. Black 近似（Black 1975）
────────────────────────────
对美式看涨期权，取以下两个价格的最大值：
1. 无红利美式期权价格（BSM，T 为完整期限）
2. 欧式看涨期权（到最后红利日行权，T' = t_D）

实践中最常用的快速近似。

参考：
  - Roll, R. (1977). "An Analytic Valuation Formula for Unprotected
    American Call Options on Stocks with Known Dividends."
    Journal of Financial Economics, 5, 251–258.
  - Geske, R. (1979). "A Note on an Analytic Valuation Formula for
    Unprotected American Call Options on Stocks with Known Dividends."
    Journal of Financial Economics, 7, 375–380.
  - Whaley, R.E. (1981). "On the Valuation of American Call Options
    on Stocks with Known Dividends." Journal of Financial Economics,
    9, 207–211.
书中对应：Haug (2007), Chapter 9
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n, cbnd


# ═══════════════════════════════════════════════════════════════
# A. 疏散红利法（Escrowed Dividend Model）
# ═══════════════════════════════════════════════════════════════

def escrowed_dividend_option(S: float, K: float, T: float,
                              r: float, sigma: float,
                              dividends: list,   # [(t_i, D_i), ...]
                              option_type: str = 'call') -> float:
    """
    疏散红利法（Escrowed / Present Value of Dividends）欧式期权定价。

    将未来红利的现值从当前股价中扣除，得到调整股价 S*，
    然后对 S* 应用标准 BSM 公式。

    S* = S - Σ Dᵢ · e^{-r·tᵢ}  （仅扣除期权有效期内的红利）

    参数
    ----
    S          : 当前股价
    K          : 行权价
    T          : 到期时间（年）
    r          : 无风险利率
    sigma      : 股票波动率（调整后 S* 的波动率）
    dividends  : 红利列表 [(t₁, D₁), (t₂, D₂), ...]
                 t_i = 红利发放时间（年），D_i = 红利金额
    option_type : 'call' 或 'put'

    注意
    ----
    - 只扣除 t_i ≤ T 的红利
    - sigma 对应调整后股价 S* 的波动率（通常略高于调整前）
    - 仅适用于欧式期权（美式需用 RGW 方法）
    """
    # 计算期权有效期内所有红利的现值
    PV_div = sum(D * exp(-r*t) for t, D in dividends if 0 < t <= T)

    # 调整股价
    S_star = S - PV_div

    if S_star <= 0:
        raise ValueError(f"调整后股价 S* = {S_star:.4f} ≤ 0，红利过大")

    if T <= 0:
        if option_type == 'call': return max(S_star - K, 0.)
        return max(K - S_star, 0.)

    # 标准 BSM 公式（对 S*）
    d1 = (log(S_star/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    df = exp(-r*T)

    if option_type.lower() == 'call':
        return S_star * N(d1) - K * df * N(d2)
    return K * df * N(-d2) - S_star * N(-d1)


# ═══════════════════════════════════════════════════════════════
# B. Roll-Geske-Whaley（单次离散红利美式看涨）
# ═══════════════════════════════════════════════════════════════

def roll_geske_whaley_call(S: float, K: float, T: float,
                            r: float, sigma: float,
                            t_D: float, D: float) -> float:
    """
    Roll-Geske-Whaley (1977/1979/1981) 含单次离散红利的美式看涨期权。

    对于含单次离散红利 D（在 t_D 时发放）的美式看涨期权，
    提前行权只可能在红利发放前一刻（ t_D^- ）发生。

    算法：
    1. 找临界股价 S*：使得在 t_D 时立即行权与持有期权无差异
       S* - K = call_european(S*, K, T - t_D, r, 0, sigma)
       （红利发放后立即用 BSM 定价，因为之后无更多红利）
    2. 若 S ≤ 临界价格，期权肯定不提前行权
       → 使用疏散红利法（简化为欧式）
    3. 否则，使用 Geske 的双变量正态公式

    精确公式：
    令 S' = S - D·e^{-r·t_D}（调整红利）
    a₁ = [ln(S'/S*) + (r + σ²/2)·t_D] / (σ√t_D)
    a₂ = a₁ - σ√t_D
    b₁ = [ln(S'/K) + (r + σ²/2)·T] / (σ√T)
    b₂ = b₁ - σ√T
    ρ  = √(t_D/T)  （两段时间的相关系数）

    C = S'·N(b₁) - K·e^{-rT}·N(b₂) - (K-D)·e^{-r·t_D}·N(a₂)
        + S'·[cbnd(-a₁, b₁, -ρ) - cbnd(-a₁, a₁, ρ)]  (简化示意)

    实际公式更复杂，见代码实现。
    """
    from scipy.optimize import brentq

    # ── 红利调整股价 ──────────────────────────────────────
    S_adj = S - D * exp(-r * t_D)   # 扣除红利现值后的股价

    if t_D <= 0 or t_D >= T:
        # 红利在期权起始或到期后：退化为标准欧式 BSM
        d1 = (log(S_adj/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return S_adj * N(d1) - K * exp(-r*T) * N(d2)

    # ── 寻找临界股价 S*──────────────────────────────────────
    # 临界条件：S* - K = European_call(S*, K, T-t_D, r, r, sigma)
    # 即在红利发放后，立即行权（获得 S* - K）等于继续持有期权
    T2 = T - t_D   # 红利发放后剩余时间

    def critical_equation(S_star):
        if S_star <= 0: return -K
        d1_c = (log(S_star/K) + (r + 0.5*sigma**2)*T2) / (sigma*sqrt(T2))
        d2_c = d1_c - sigma*sqrt(T2)
        eu_val = S_star * N(d1_c) - K * exp(-r*T2) * N(d2_c)
        return (S_star + D - K) - eu_val   # S+D-K = eu_val 时无差异

    # 搜索临界价格
    try:
        I_star = brentq(critical_equation, 1e-4, S * 100, xtol=1e-8, maxiter=500)
    except ValueError:
        # 若找不到临界价格，提前行权从未有利 → 疏散红利法
        return escrowed_dividend_option(S, K, T, r, sigma, [(t_D, D)], 'call')

    # ── 若当前股价低于临界价格，提前行权无利 ───────────────
    if S_adj <= I_star - D:
        # 提前行权无利，用疏散红利法（欧式近似）
        return escrowed_dividend_option(S, K, T, r, sigma, [(t_D, D)], 'call')

    # ── RGW 精确公式 ──────────────────────────────────────────
    # 相关系数（两段时间）
    rho = sqrt(t_D / T)

    a1 = (log(S_adj / (I_star - D)) + (r + 0.5*sigma**2)*t_D) / (sigma*sqrt(t_D))
    a2 = a1 - sigma*sqrt(t_D)
    b1 = (log(S_adj / K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    b2 = b1 - sigma*sqrt(T)

    # RGW 公式（Geske 1979 的精确形式）
    C = (S_adj * N(b1)
         - K * exp(-r*T) * N(b2)
         - (I_star - D - K) * exp(-r*t_D) * N(a2)
         + S_adj * cbnd(b1, -a1, -rho)
         - K * exp(-r*T) * cbnd(b2, -a2, -rho))

    return max(C, max(S - K, 0.))


# ═══════════════════════════════════════════════════════════════
# C. Black (1975) 近似
# ═══════════════════════════════════════════════════════════════

def black_approximation_call(S: float, K: float, T: float,
                              r: float, sigma: float,
                              t_D: float, D: float) -> float:
    """
    Black (1975) 近似 — 含单次离散红利的美式看涨期权快速近似。

    取以下两者的较大值：
    1. 欧式看涨（到期 T，扣除红利现值）
    2. 欧式看涨（到红利日 t_D 行权，股价不扣减，行权价 K）
       即：提前行权（在 t_D^- 时）的价值

    公式：
    C_black = max(C_european(S_adj, K, T, r, σ),
                  C_european(S, K, t_D, r, σ))

    说明：第 2 项相当于在红利发放前行权，此时持有者获得
    股票（价值 S_D + D）减去行权价 K，期望收益通过 BSM 估算。

    参数
    ----
    t_D : 单次红利发放时间（< T）
    D   : 红利金额
    """
    # 方案一：扣除红利现值后的欧式期权（持有到 T）
    S_adj = S - D * exp(-r * t_D)
    d1_1 = (log(S_adj/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2_1 = d1_1 - sigma*sqrt(T)
    C1 = S_adj * N(d1_1) - K * exp(-r*T) * N(d2_1)

    # 方案二：到红利日前行权的欧式期权
    if t_D <= 0:
        C2 = max(S - K, 0.)
    else:
        d1_2 = (log(S/K) + (r + 0.5*sigma**2)*t_D) / (sigma*sqrt(t_D))
        d2_2 = d1_2 - sigma*sqrt(t_D)
        C2 = S * N(d1_2) - K * exp(-r*t_D) * N(d2_2)

    return max(C1, C2)


# ═══════════════════════════════════════════════════════════════
# D. 多次离散红利（使用疏散红利法 + 二叉树）
# ═══════════════════════════════════════════════════════════════

def discrete_dividend_tree(S: float, K: float, T: float,
                            r: float, sigma: float,
                            dividends: list,
                            N: int = 100,
                            option_type: str = 'call',
                            exercise: str = 'american') -> float:
    """
    多次离散红利期权定价（CRR 二叉树 + 疏散红利调整）。

    每个节点上的股价 = "增长部分"（无红利）+ 未来红利现值
    这样可以处理多次离散红利。

    参数
    ----
    dividends : [(t₁, D₁), (t₂, D₂), ...] 多次红利列表
    N         : 树步数
    exercise  : 'european' 或 'american'
    """
    import numpy as np

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1. / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)   # b=r（无红利）

    if not (0 < p < 1):
        raise ValueError(f"概率越界: p={p:.4f}")

    # 预计算每个步骤结束时点 t_k = k·dt 之后的红利现值
    # PV_after[k] = Σ_{t_i > k·dt} D_i · e^{-r·(t_i - k·dt)}
    def pv_future_dividends(t_now: float) -> float:
        return sum(D * exp(-r*(t - t_now))
                   for t, D in dividends if t > t_now)

    # ── 到期时股价节点（无红利的"增长部分"）─────────────
    S_growth_final = np.array([S * u**(2*j - N) for j in range(N+1)])
    # 到期时已无未来红利，实际股价 ≈ 增长部分
    S_final = S_growth_final  # （如有到期后红利，此项为 0）

    if option_type.lower() == 'call':
        V = np.maximum(S_final - K, 0.)
    else:
        V = np.maximum(K - S_final, 0.)

    # ── 向后递推 ──────────────────────────────────────────
    for step in range(N - 1, -1, -1):
        t_now = step * dt
        # 当前步的"增长部分"股价（不含红利）
        S_growth = np.array([S * u**(2*j - step) for j in range(step+1)])
        # 实际股价 = 增长部分 + 未来红利现值
        pv_div = pv_future_dividends(t_now)
        S_actual = S_growth + pv_div

        V_hold = disc * (p * V[1:step+2] + (1-p) * V[:step+1])

        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_ex = np.maximum(S_actual - K, 0.)
            else:
                V_ex = np.maximum(K - S_actual, 0.)
            V = np.maximum(V_hold, V_ex)
        else:
            V = V_hold

    return float(V[0])


if __name__ == "__main__":
    print("=" * 65)
    print("离散红利期权定价 — 数值示例（Haug Chapter 9）")
    print("=" * 65)

    # ── 参数（Haug p.155 示例）─────────────────────────────
    S, K, T, r, sigma = 80., 82., 0.25, 0.06, 0.30
    t_D, D = 1./12., 4.0   # 1 个月后发放 4 元红利

    print(f"\n参数：S={S}, K={K}, T={T}年, r={r:.0%}, σ={sigma:.0%}")
    print(f"      红利：t_D={t_D:.4f}年({int(t_D*12)}月), D={D}")

    # 疏散红利法（欧式）
    esc = escrowed_dividend_option(S, K, T, r, sigma, [(t_D, D)], 'call')
    PV_d = D * exp(-r*t_D)
    print(f"\n1. 疏散红利法（欧式看涨）：")
    print(f"   PV(红利) = {PV_d:.4f}")
    print(f"   S* = S - PV(D) = {S - PV_d:.4f}")
    print(f"   欧式看涨 = {esc:.4f}")

    # Black 近似（美式）
    blk = black_approximation_call(S, K, T, r, sigma, t_D, D)
    print(f"\n2. Black (1975) 近似（美式看涨）= {blk:.4f}")

    # RGW 精确（美式）
    try:
        rgw = roll_geske_whaley_call(S, K, T, r, sigma, t_D, D)
        print(f"\n3. Roll-Geske-Whaley（美式看涨精确）= {rgw:.4f}")
        print(f"   （参考值：Haug 2007 p.155 ≈ 4.3860）")
    except Exception as e:
        print(f"\n3. RGW 计算：{e}")

    # 二叉树（多次红利）
    tree_val = discrete_dividend_tree(S, K, T, r, sigma,
                                       [(t_D, D)], N=200, option_type='call',
                                       exercise='american')
    print(f"\n4. 二叉树（N=200，美式看涨）= {tree_val:.4f}")

    # ── 多次红利示例 ──────────────────────────────────────
    print(f"\n多次离散红利示例（每季度发放一次）：")
    S2, K2, T2, r2, sigma2 = 100., 100., 1., 0.08, 0.25
    dividends_multi = [(0.25, 2.), (0.50, 2.), (0.75, 2.)]  # 每季 2 元
    total_pv = sum(D*exp(-r2*t) for t,D in dividends_multi)
    print(f"  S={S2}, K={K2}, T={T2}年, r={r2:.0%}, σ={sigma2:.0%}")
    print(f"  红利：{dividends_multi}")
    print(f"  红利总现值 = {total_pv:.4f}")

    eu_multi = escrowed_dividend_option(S2, K2, T2, r2, sigma2, dividends_multi, 'call')
    am_multi  = discrete_dividend_tree(S2, K2, T2, r2, sigma2, dividends_multi,
                                        N=200, option_type='call', exercise='american')
    print(f"  欧式看涨（疏散红利法）= {eu_multi:.4f}")
    print(f"  美式看涨（二叉树N=200）= {am_multi:.4f}")
    print(f"  美式溢价 = {am_multi - eu_multi:.4f}")
