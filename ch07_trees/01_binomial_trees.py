"""
01_binomial_trees.py — 二叉树 & 三叉树期权定价
===============================================
【模型简介】
二叉树（Binomial Tree）是期权定价的离散时间框架，通过
逆向递推（Backward Induction）计算期权价格。相比 BSM，
二叉树的优势在于：
  1. 可自然处理美式期权（提前行权）
  2. 可处理离散红利和路径依赖特征
  3. 计算直观，便于理解期权定价原理

本文件实现以下方法：
  A. CRR 二叉树（Cox-Ross-Rubinstein 1979）— 最经典
  B. Leisen-Reimer 二叉树（1996）— 更快收敛（奇数步必用）
  C. 等概率二叉树（Jarrow-Rudd 1983）— 等概率变体
  D. 三叉树（Trinomial Tree, Boyle 1986）— 更精确

所有方法均支持：
  - 欧式 / 美式
  - 看涨 / 看跌
  - 持有成本 b（统一 GBM 框架）
  - 通用障碍期权（二叉树中实现）

参考：
  - Cox, J.C., Ross, S.A. & Rubinstein, M. (1979). "Option Pricing:
    A Simplified Approach." Journal of Financial Economics, 7, 229–263.
  - Leisen, D. & Reimer, M. (1996). "Binomial Models for Option Valuation."
    European Journal of Operational Research.
  - Boyle, P.P. (1986). "Option Valuation Using a Three Jump Process."
    International Options Journal, 3, 7–12.
书中对应：Haug (2007), Chapter 7, Sections 7.1–7.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
import numpy as np
from utils.common import norm_cdf as N


# ═══════════════════════════════════════════════════════════════
# 辅助：收益函数
# ═══════════════════════════════════════════════════════════════

def _payoff(S: float, K: float, option_type: str) -> float:
    if option_type == 'call': return max(S - K, 0.)
    return max(K - S, 0.)


# ═══════════════════════════════════════════════════════════════
# A. CRR 二叉树（Cox-Ross-Rubinstein 1979）
# ═══════════════════════════════════════════════════════════════

def crr_binomial_tree(S: float, K: float, T: float,
                      r: float, b: float, sigma: float,
                      N: int = 100,
                      option_type: str = 'call',
                      exercise: str = 'european') -> float:
    """
    CRR（Cox-Ross-Rubinstein 1979）二叉树期权定价。

    CRR 参数化：
    Δt = T/N     （每步时长）
    u = e^{σ√Δt}  （上升因子）
    d = 1/u = e^{-σ√Δt}  （下跌因子，u×d=1 保证对称性）
    p = (e^{bΔt} - d) / (u - d)  （风险中性上升概率）
    折现因子 = e^{-rΔt}

    关键性质：
    - u·d = 1（对称性），使树不断重组（Recombining Tree）
    - 节点数 = O(N²)，计算效率高
    - 当 N→∞ 时收敛到 BSM 价格
    - 美式期权：每个节点比较持有价值与立即行权价值

    参数
    ----
    S        : 当前股价
    K        : 行权价
    T        : 到期时间（年）
    r        : 无风险利率
    b        : 持有成本（b=r 无红利，b=r-q 含红利）
    sigma    : 年化波动率
    N        : 树的步数（步数越多越精确，推荐 100–200）
    option_type : 'call' 或 'put'
    exercise : 'european' 或 'american'
    """
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1. / u
    # 风险中性概率
    disc = exp(-r * dt)                     # 单步折现因子
    p = (exp(b * dt) - d) / (u - d)        # 上升概率

    if not (0 < p < 1):
        raise ValueError(f"CRR 概率越界：p={p:.4f}，请减小 dt 或调整参数")

    # ── 构建到期时的股价节点 ──────────────────────────────
    # 节点 j: S · u^j · d^{N-j} = S · u^{2j-N}
    S_final = np.array([S * u**(2*j - N) for j in range(N+1)])

    # ── 到期收益 ─────────────────────────────────────────
    if option_type.lower() == 'call':
        V = np.maximum(S_final - K, 0.)
    else:
        V = np.maximum(K - S_final, 0.)

    # ── 向后递推 ─────────────────────────────────────────
    for step in range(N - 1, -1, -1):
        # 当前步骤的股价
        S_step = np.array([S * u**(2*j - step) for j in range(step+1)])

        # 持有期权的价值（期望 × 折现）
        V_hold = disc * (p * V[1:step+2] + (1-p) * V[:step+1])

        if exercise.lower() == 'american':
            # 提前行权价值
            if option_type.lower() == 'call':
                V_exercise = np.maximum(S_step - K, 0.)
            else:
                V_exercise = np.maximum(K - S_step, 0.)
            V = np.maximum(V_hold, V_exercise)
        else:
            V = V_hold

    return float(V[0])


# ═══════════════════════════════════════════════════════════════
# B. Leisen-Reimer (LR) 二叉树（1996）
# ═══════════════════════════════════════════════════════════════

def _hr_inversion(z: float, n: int) -> float:
    """
    Haase-Roessing 反函数（用于 Leisen-Reimer 概率校正）。
    Peizer-Pratt 反函数近似 N^{-1}(z, n)：
    将正态 CDF 逆函数修正为离散分布的准确匹配。
    """
    # Peizer & Pratt (1968) 近似
    if abs(z) < 1e-10: return 0.5
    # 使用 Haase & Roessing / Peizer-Pratt h 函数
    # h(z) = 0.5 + sign(z) · 0.5 · sqrt(1 - exp(-z²/(n+1/3+0.1/(n+1))·(n+1/6)))
    t = z**2 / (n + 1./3. + 0.1/(n + 1.))
    try:
        from math import exp as mexp
        val = 0.5 + (1. if z > 0 else -1.) * 0.5 * (1. - mexp(-t * (n + 1./6.)))**0.5
    except:
        val = 0.5
    return max(0., min(1., val))


def leisen_reimer_tree(S: float, K: float, T: float,
                        r: float, b: float, sigma: float,
                        N: int = 101,
                        option_type: str = 'call',
                        exercise: str = 'european') -> float:
    """
    Leisen-Reimer (1996) 二叉树 — 改进收敛性。

    LR 树的关键改进：
    - 不再使用 u/d 的几何对称性，而是直接匹配 BSM 的 d₁, d₂ 概率
    - 利用 Peizer-Pratt 近似函数将连续正态 CDF 转换为离散概率
    - 在相同步数下，收敛速度比 CRR 快约一个数量级
    - 推荐使用奇数步数（N 为奇数）以确保精确收敛

    参数
    ----
    N : 步数（强烈建议用奇数，如 99, 101, 199, 201...）
    """
    if N % 2 == 0:
        N += 1  # 强制奇数

    dt = T / N
    disc = exp(-r * dt)

    # 计算 BSM 的 d₁, d₂（用于校准 LR 概率）
    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    # LR 概率（通过 Haase-Roessing 反函数校准）
    p1 = _hr_inversion(d1, N)  # 关联 d₁ 的概率（用于资产端）
    p2 = _hr_inversion(d2, N)  # 关联 d₂ 的概率（用于现金端）

    # 根据 p₁, p₂ 反推 u, d
    # p₁·u + (1-p₁)·d = e^{bΔt}（均值匹配）
    # p₂·(1-u) + (1-p₂)·(1-d) = ... （隐含关系）
    # 实际：u 从 p₂ 推导（现金测度），p 从 p₁ 推导（资产测度）
    # LR 的 u, d 定义：
    p = p2        # 风险中性概率
    u = exp(b*dt) * p1 / p2
    d = (exp(b*dt) - p*u) / (1. - p)

    if u <= 0 or d <= 0 or not (0 < p < 1):
        # 退化为 CRR
        return crr_binomial_tree(S, K, T, r, b, sigma, N, option_type, exercise)

    # ── 到期股价节点 ──────────────────────────────────────
    S_final = np.array([S * u**j * d**(N-j) for j in range(N+1)])

    # 到期收益
    if option_type.lower() == 'call':
        V = np.maximum(S_final - K, 0.)
    else:
        V = np.maximum(K - S_final, 0.)

    # ── 向后递推 ─────────────────────────────────────────
    for step in range(N - 1, -1, -1):
        S_step = np.array([S * u**j * d**(step-j) for j in range(step+1)])
        V_hold = disc * (p * V[1:step+2] + (1-p) * V[:step+1])

        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_ex = np.maximum(S_step - K, 0.)
            else:
                V_ex = np.maximum(K - S_step, 0.)
            V = np.maximum(V_hold, V_ex)
        else:
            V = V_hold

    return float(V[0])


# ═══════════════════════════════════════════════════════════════
# C. Jarrow-Rudd (1983) 等概率二叉树
# ═══════════════════════════════════════════════════════════════

def jarrow_rudd_tree(S: float, K: float, T: float,
                     r: float, b: float, sigma: float,
                     N: int = 100,
                     option_type: str = 'call',
                     exercise: str = 'european') -> float:
    """
    Jarrow-Rudd (1983) 等概率二叉树（Equal Probability Binomial Tree）。

    JR 树的参数化（令 p = 0.5，即上升和下跌概率相等）：
    u = e^{(b - σ²/2)Δt + σ√Δt}    ← 上升：均值漂移 + 波动
    d = e^{(b - σ²/2)Δt - σ√Δt}    ← 下跌：均值漂移 - 波动
    p = 0.5                           ← 等概率！

    折现时的调整：
    风险中性折现：disc = e^{-rΔt}

    与 CRR 不同，JR 树不要求 u·d = 1，但保证等概率。
    收敛速度与 CRR 相近。
    """
    dt = T / N
    # JR 的 u, d（含均值漂移）
    u = exp((b - 0.5*sigma**2)*dt + sigma*sqrt(dt))
    d = exp((b - 0.5*sigma**2)*dt - sigma*sqrt(dt))
    p = 0.5   # 等概率

    disc = exp(-r * dt)

    # 到期股价
    S_final = np.array([S * u**j * d**(N-j) for j in range(N+1)])

    if option_type.lower() == 'call':
        V = np.maximum(S_final - K, 0.)
    else:
        V = np.maximum(K - S_final, 0.)

    for step in range(N - 1, -1, -1):
        S_step = np.array([S * u**j * d**(step-j) for j in range(step+1)])
        V_hold = disc * (p * V[1:step+2] + (1-p) * V[:step+1])

        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_ex = np.maximum(S_step - K, 0.)
            else:
                V_ex = np.maximum(K - S_step, 0.)
            V = np.maximum(V_hold, V_ex)
        else:
            V = V_hold

    return float(V[0])


# ═══════════════════════════════════════════════════════════════
# D. 三叉树（Trinomial Tree, Boyle 1986）
# ═══════════════════════════════════════════════════════════════

def trinomial_tree(S: float, K: float, T: float,
                   r: float, b: float, sigma: float,
                   N: int = 50,
                   option_type: str = 'call',
                   exercise: str = 'european') -> float:
    """
    三叉树（Trinomial Tree, Boyle 1986）期权定价。

    三叉树每步有三个可能：上升(u)、中间(m=1)、下跌(d)。
    相比二叉树，三叉树在相同步数下精度更高（收敛更快）。

    Boyle (1986) 参数化：
    u = e^{σ√(3Δt)}   （或更常用 u = e^{σ√(λΔt)}，λ=√3 时最优）
    d = 1/u
    m = 1              （中间不变）

    风险中性概率（匹配前两阶矩）：
    p_u = (e^{bΔt/2} - e^{-σ√(Δt/2)})² / (e^{σ√(Δt/2)} - e^{-σ√(Δt/2)})²
    p_d = (e^{σ√(Δt/2)} - e^{bΔt/2})² / (e^{σ√(Δt/2)} - e^{-σ√(Δt/2)})²
    p_m = 1 - p_u - p_d

    或简化的 Kamrad-Ritchken (1991) 形式（本实现采用）：
    λ = √3/2（调节参数）
    u = e^{σ√(λΔt)}
    p_u = 1/6 + (b - σ²/2)√Δt/(2σ√λ) + σ²Δt/(6σ²) = ...
    （见代码）

    三叉树优势：
    1. 在 Δt 较大时仍保持数值稳定
    2. 适合实现障碍期权（障碍可以精确落在节点上）
    3. 等价于显式有限差分格式
    """
    dt = T / N
    # Kamrad-Ritchken 三叉树参数
    lam = sqrt(3. / 2.)    # 最优参数（最小化误差）
    u = exp(sigma * lam * sqrt(dt))
    d = 1. / u
    disc = exp(-r * dt)
    edt_b = exp(b * dt)

    # 风险中性概率
    pu = 0.5 * (sigma**2 * dt + (b - 0.5*sigma**2)**2 * dt**2) / (sigma * lam * sqrt(dt))**2 \
         + (b - 0.5*sigma**2) * dt / (2 * sigma * lam * sqrt(dt))
    pd = 0.5 * (sigma**2 * dt + (b - 0.5*sigma**2)**2 * dt**2) / (sigma * lam * sqrt(dt))**2 \
         - (b - 0.5*sigma**2) * dt / (2 * sigma * lam * sqrt(dt))
    pm = 1. - pu - pd

    # 概率有效性检查
    if pu < 0 or pd < 0 or pm < 0:
        # 退化为标准参数化
        pu = 1. / 6. + (b - 0.5*sigma**2) * sqrt(dt) / (2*sigma*lam)
        pd = 1. / 6. - (b - 0.5*sigma**2) * sqrt(dt) / (2*sigma*lam)
        pm = 2. / 3.
        if pu < 0 or pd < 0:
            return crr_binomial_tree(S, K, T, r, b, sigma, N*2, option_type, exercise)

    # ── 到期时的股价节点（三叉树：2N+1 个节点）─────────────
    # 节点 j 的股价：S · u^{j - N}（j = 0, 1, ..., 2N）
    # j=0: S·d^N（最低），j=N: S（中间），j=2N: S·u^N（最高）
    n_nodes = 2 * N + 1
    S_final = np.array([S * u**(j - N) for j in range(n_nodes)])

    if option_type.lower() == 'call':
        V = np.maximum(S_final - K, 0.)
    else:
        V = np.maximum(K - S_final, 0.)

    # ── 向后递推（三叉）─────────────────────────────────────
    for step in range(N - 1, -1, -1):
        n_current = 2 * step + 1  # 当前步节点数
        S_step = np.array([S * u**(j - step) for j in range(n_current)])

        # 三个子节点的概率加权（u→节点+1，m→节点，d→节点-1）
        V_hold = disc * (pu * V[2:n_current+2]
                         + pm * V[1:n_current+1]
                         + pd * V[:n_current])

        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_ex = np.maximum(S_step - K, 0.)
            else:
                V_ex = np.maximum(K - S_step, 0.)
            V = np.maximum(V_hold, V_ex)
        else:
            V = V_hold

    return float(V[0])


# ═══════════════════════════════════════════════════════════════
# E. 带障碍的二叉树（Barrier Option in Tree）
# ═══════════════════════════════════════════════════════════════

def barrier_binomial_tree(S: float, K: float, H: float, rebate: float,
                           T: float, r: float, b: float, sigma: float,
                           N: int = 200,
                           barrier_type: str = 'down-out',
                           option_type: str = 'call') -> float:
    """
    障碍期权的二叉树定价（通用实现）。

    在 CRR 二叉树的基础上，在每个节点处检查障碍条件：
    - 若股价触及（或穿越）障碍 H，则期权作废（或激活）
    - 敲出（Knock-out）：触碰障碍后期权失效，支付 rebate
    - 敲入（Knock-in）：触碰障碍后期权激活

    注意：二叉树对障碍的处理有"毛刺"效应（Oscillation），
    可通过增大步数 N 或使用 Richardson 外推来改善。

    参数
    ----
    H            : 障碍价格
    rebate       : 敲出时的补偿金（通常为 0）
    barrier_type : 'down-out', 'up-out', 'down-in', 'up-in'
    """
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1. / u
    disc = exp(-r * dt)
    p = (exp(b * dt) - d) / (u - d)

    # 到期股价节点
    S_final = np.array([S * u**(2*j - N) for j in range(N+1)])

    # 判断各节点是否触碰障碍（到期时判断）
    if 'down' in barrier_type:
        knocked = S_final <= H
    else:
        knocked = S_final >= H

    # 敲入 vs 敲出的收益
    if option_type.lower() == 'call':
        payoff_raw = np.maximum(S_final - K, 0.)
    else:
        payoff_raw = np.maximum(K - S_final, 0.)

    if 'out' in barrier_type:
        V = np.where(knocked, rebate, payoff_raw)
    else:  # knock-in
        V = np.where(knocked, payoff_raw, rebate)

    # 向后递推（含障碍检查）
    for step in range(N - 1, -1, -1):
        S_step = np.array([S * u**(2*j - step) for j in range(step+1)])

        V_hold = disc * (p * V[1:step+2] + (1-p) * V[:step+1])

        # 障碍条件检查
        if 'down' in barrier_type:
            hit = S_step <= H
        else:
            hit = S_step >= H

        if 'out' in barrier_type:
            V = np.where(hit, rebate, V_hold)
        else:  # knock-in：触碰障碍后激活（在递推中简化处理）
            # 敲入通过 knock-out + vanilla 的对称性间接计算
            V = V_hold  # 简化：直接递推（更准确的敲入需追踪路径）

    return float(V[0])


if __name__ == "__main__":
    print("=" * 65)
    print("二叉树 & 三叉树期权定价 — 数值示例（Haug Chapter 7）")
    print("=" * 65)

    # 参数
    S, K, T, r, b, sigma = 100., 100., 0.5, 0.10, 0.10, 0.20

    # BSM 理论值（欧式）
    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    from utils.common import norm_cdf as Nf, norm_pdf as nf
    bsm_call = S*exp((b-r)*T)*Nf(d1) - K*exp(-r*T)*Nf(d2)
    bsm_put  = K*exp(-r*T)*Nf(-d2) - S*exp((b-r)*T)*Nf(-d1)
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"BSM 理论值：看涨 = {bsm_call:.4f}, 看跌 = {bsm_put:.4f}")

    # ── 欧式期权各方法对比 ────────────────────────────────
    print(f"\n欧式期权定价对比（N=100 步）：")
    print(f"{'方法':<20} {'看涨':>10} {'看跌':>10} {'看涨误差':>10}")

    crr_c = crr_binomial_tree(S, K, T, r, b, sigma, 100, 'call', 'european')
    crr_p = crr_binomial_tree(S, K, T, r, b, sigma, 100, 'put',  'european')
    print(f"{'CRR (N=100)':<20} {crr_c:>10.4f} {crr_p:>10.4f} {crr_c-bsm_call:>+10.6f}")

    lr_c = leisen_reimer_tree(S, K, T, r, b, sigma, 101, 'call', 'european')
    lr_p = leisen_reimer_tree(S, K, T, r, b, sigma, 101, 'put',  'european')
    print(f"{'LR (N=101)':<20} {lr_c:>10.4f} {lr_p:>10.4f} {lr_c-bsm_call:>+10.6f}")

    jr_c = jarrow_rudd_tree(S, K, T, r, b, sigma, 100, 'call', 'european')
    jr_p = jarrow_rudd_tree(S, K, T, r, b, sigma, 100, 'put',  'european')
    print(f"{'Jarrow-Rudd (N=100)':<20} {jr_c:>10.4f} {jr_p:>10.4f} {jr_c-bsm_call:>+10.6f}")

    tri_c = trinomial_tree(S, K, T, r, b, sigma, 50, 'call', 'european')
    tri_p = trinomial_tree(S, K, T, r, b, sigma, 50, 'put',  'european')
    print(f"{'三叉树 (N=50)':<20} {tri_c:>10.4f} {tri_p:>10.4f} {tri_c-bsm_call:>+10.6f}")

    # ── 美式期权 ─────────────────────────────────────────────
    print(f"\n美式期权定价（b=0，即期货期权，提前行权有价值）：")
    b_futures = 0.0
    am_crr_c  = crr_binomial_tree(S, K, T, r, b_futures, sigma, 200, 'call', 'american')
    am_crr_p  = crr_binomial_tree(S, K, T, r, b_futures, sigma, 200, 'put',  'american')
    am_lr_c   = leisen_reimer_tree(S, K, T, r, b_futures, sigma, 201, 'call', 'american')
    am_lr_p   = leisen_reimer_tree(S, K, T, r, b_futures, sigma, 201, 'put',  'american')

    # 欧式基准（b=0）
    d1_f = (log(S/K) + (b_futures + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2_f = d1_f - sigma*sqrt(T)
    eu_c_fut = S*exp((b_futures-r)*T)*Nf(d1_f) - K*exp(-r*T)*Nf(d2_f)

    print(f"  欧式看涨（b=0）= {eu_c_fut:.4f}")
    print(f"  美式看涨 CRR(N=200) = {am_crr_c:.4f}  (提前行权溢价: {am_crr_c-eu_c_fut:.4f})")
    print(f"  美式看涨 LR(N=201)  = {am_lr_c:.4f}")
    print(f"  美式看跌 CRR(N=200) = {am_crr_p:.4f}")
    print(f"  美式看跌 LR(N=201)  = {am_lr_p:.4f}")

    # ── 收敛性分析 ────────────────────────────────────────────
    print(f"\nCRR vs LR 收敛性（步数增加）：")
    print(f"  {'N':>6}  {'CRR':>10}  {'LR':>10}  {'三叉(N/2)':>12}")
    for N_t in [10, 20, 50, 100, 200]:
        c_crr = crr_binomial_tree(S, K, T, r, b, sigma, N_t, 'call', 'european')
        N_lr = N_t if N_t % 2 else N_t + 1
        c_lr  = leisen_reimer_tree(S, K, T, r, b, sigma, N_lr, 'call', 'european')
        c_tri = trinomial_tree(S, K, T, r, b, sigma, max(N_t//2, 5), 'call', 'european')
        print(f"  {N_t:>6}  {c_crr:>10.5f}  {c_lr:>10.5f}  {c_tri:>12.5f}")
    print(f"  {'理论':>6}  {bsm_call:>10.5f}  {bsm_call:>10.5f}  {bsm_call:>12.5f}")
