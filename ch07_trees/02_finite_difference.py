"""
02_finite_difference.py — 有限差分法期权定价
=============================================
【模型简介】
有限差分法（Finite Difference Method, FDM）是通过将 Black-Scholes
偏微分方程（PDE）离散化来数值求解期权价格的方法：

BSM PDE（持有成本 b 形式）：
∂V/∂t + b·S·∂V/∂S + ½σ²S²·∂²V/∂S² - r·V = 0

边界条件：
  看涨到期：V(S, T) = max(S - K, 0)
  看涨上边界：V(S_max, t) ≈ S_max - K·e^{-r(T-t)}
  看涨下边界：V(0, t) = 0

三种主要离散化格式：

A. 显式格式（Explicit / FTCS）
──────────────────────────────
时间向前差分（Forward in Time）+ 空间中央差分（Central in Space）。
条件稳定：Δt ≤ 1/(σ²·N_S²)，步长受限。
优点：计算简单，每步只需矩阵向量乘法。

B. 隐式格式（Implicit / BTCS / Fully Implicit）
────────────────────────────────────────────────
时间向后差分（Backward in Time）。
无条件稳定（适合任何步长）。
每步需要求解三对角线性方程组。

C. Crank-Nicolson 格式（CN）
────────────────────────────
显式和隐式的等权平均（时间中央差分）。
无条件稳定 + 时间二阶精度（比隐式更准确）。
最常用的 FDM 期权定价格式。

本文件结构：
  1. 共用的网格设置函数
  2. 三种格式的欧式/美式期权定价
  3. 数值验证与精度对比

参考：
  - Wilmott, P., Dewynne, J. & Howison, S. (1993).
    "Option Pricing: Mathematical Models and Computation."
  - Crank, J. & Nicolson, P. (1947). "A Practical Method for
    Numerical Evaluation of Solutions of PDEs."
书中对应：Haug (2007), Chapter 7, Section 7.4–7.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
import numpy as np
from utils.common import norm_cdf as N


# ═══════════════════════════════════════════════════════════════
# 共用网格设置
# ═══════════════════════════════════════════════════════════════

def setup_grid(S: float, K: float, T: float, sigma: float,
               N_S: int = 200, N_T: int = 100,
               S_max_factor: float = 3.0):
    """
    建立 BSM FDM 价格-时间网格。

    参数
    ----
    S           : 当前股价（用于确定中心区域）
    K           : 行权价
    T           : 到期时间
    sigma       : 波动率（用于确定合适的 S_max）
    N_S         : 股价方向网格节点数
    N_T         : 时间方向步数
    S_max_factor: S_max = S_max_factor × K

    返回
    ----
    S_grid : 股价节点数组（长度 N_S+1）
    dt     : 时间步长
    dS     : 股价步长
    i_S0   : 当前股价 S 最近的网格节点索引
    """
    S_max = S_max_factor * K   # 上边界
    dS = S_max / N_S
    dt = T / N_T
    S_grid = np.linspace(0, S_max, N_S + 1)   # [0, dS, 2dS, ..., S_max]

    # 找到最接近当前股价 S 的网格节点
    i_S0 = int(round(S / dS))
    i_S0 = max(1, min(i_S0, N_S - 1))

    return S_grid, dt, dS, i_S0


# ═══════════════════════════════════════════════════════════════
# A. 显式格式（Explicit FDM）
# ═══════════════════════════════════════════════════════════════

def explicit_fdm(S: float, K: float, T: float,
                 r: float, b: float, sigma: float,
                 N_S: int = 200, N_T: int = 500,
                 option_type: str = 'call',
                 exercise: str = 'european') -> float:
    """
    显式有限差分法（Explicit / Forward-Time Central-Space）。

    BSM PDE 离散化（显式格式）：
    V_i^{n+1} - V_i^n     b·S_i       V_{i+1}^n - V_{i-1}^n
    ──────────────────  + ──────── · ──────────────────────────
           Δt               2          2ΔS
                        σ²·S_i²   V_{i+1}^n - 2V_i^n + V_{i-1}^n
                      + ──────── · ───────────────────────────────
                           2                   ΔS²
                        - r·V_i^n = 0

    解出 V_i^n（已知 V^{n+1}，求 V^n）：
    V_i^n = α_i·V_{i-1}^{n+1} + β_i·V_i^{n+1} + γ_i·V_{i+1}^{n+1}

    系数：
    α_i = Δt · (σ²S_i²/ΔS² - b·S_i/ΔS) / 2
    β_i = 1 - Δt · (σ²S_i²/ΔS² + r)
    γ_i = Δt · (σ²S_i²/ΔS² + b·S_i/ΔS) / 2

    稳定性条件：β_i ≥ 0 → Δt ≤ 1/(σ²(N_S)² + r)
    （N_T 需足够大）

    参数
    ----
    N_S : 空间网格数（推荐 ≥ 100）
    N_T : 时间步数（显式格式需要更多步数保证稳定）
    """
    S_max = 3.0 * K
    dS = S_max / N_S
    dt = T / N_T
    S_grid = np.linspace(0, S_max, N_S + 1)

    # 稳定性检查
    stability = dt * (sigma**2 * (N_S)**2 + r)
    if stability > 1.0:
        # 自动增加时间步数
        N_T = int(N_T * stability * 1.1) + 1
        dt = T / N_T

    # ── 到期收益（t = T 时的边界条件）────────────────────
    if option_type.lower() == 'call':
        V = np.maximum(S_grid - K, 0.)
    else:
        V = np.maximum(K - S_grid, 0.)

    # ── 从到期反向推演到现在（t = 0）──────────────────────
    for step in range(N_T):
        tau = (step + 1) * dt   # 离到期的时间
        V_new = np.zeros(N_S + 1)

        # 内部节点（i = 1, 2, ..., N_S-1）
        i_arr = np.arange(1, N_S)
        Si = S_grid[i_arr]

        alpha = dt * (0.5*sigma**2*Si**2/dS**2 - 0.5*b*Si/dS)
        beta  = 1. - dt * (sigma**2*Si**2/dS**2 + r)
        gamma = dt * (0.5*sigma**2*Si**2/dS**2 + 0.5*b*Si/dS)

        V_new[i_arr] = (alpha*V[i_arr-1] + beta*V[i_arr] + gamma*V[i_arr+1])

        # 边界条件
        V_new[0] = 0. if option_type == 'call' else K * exp(-r * (T - tau + dt))
        if option_type.lower() == 'call':
            V_new[N_S] = S_max - K * exp(-r * (T - tau + dt))
        else:
            V_new[N_S] = 0.

        # 美式提前行权
        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_new = np.maximum(V_new, np.maximum(S_grid - K, 0.))
            else:
                V_new = np.maximum(V_new, np.maximum(K - S_grid, 0.))

        V = V_new

    # 插值得到当前股价 S 对应的期权价格
    i_S0 = int(S / dS)
    i_S0 = max(1, min(i_S0, N_S - 1))
    # 线性插值
    frac = (S - S_grid[i_S0]) / dS
    return float(V[i_S0] * (1 - frac) + V[i_S0 + 1] * frac)


# ═══════════════════════════════════════════════════════════════
# B. 隐式格式（Fully Implicit FDM）
# ═══════════════════════════════════════════════════════════════

def implicit_fdm(S: float, K: float, T: float,
                 r: float, b: float, sigma: float,
                 N_S: int = 200, N_T: int = 100,
                 option_type: str = 'call',
                 exercise: str = 'european') -> float:
    """
    隐式有限差分法（Fully Implicit / Backward-Time Central-Space）。

    BSM PDE 离散化（隐式格式，已知 V^n 求 V^{n-1}）：
    注意：此处用"反向时间"记号，V^{n} 是时间步 n（从到期倒数）。

    每步求解三对角方程组：A · V^{n} = V^{n+1}

    系数矩阵 A（三对角）：
    a_i = -α_i  （下对角）
    b_i = 1 + α_i + γ_i + r·Δt  （主对角）
    c_i = -γ_i  （上对角）

    其中：
    α_i = Δt · (σ²S_i²/ΔS² - b·S_i/ΔS) / 2
    γ_i = Δt · (σ²S_i²/ΔS² + b·S_i/ΔS) / 2

    优点：无条件稳定（任何 Δt 都稳定）
    缺点：一阶时间精度（比 CN 低）
    """
    S_max = 3.0 * K
    dS = S_max / N_S
    dt = T / N_T
    S_grid = np.linspace(0, S_max, N_S + 1)
    N_inner = N_S - 1   # 内部节点数（排除 S=0 和 S=S_max）

    # ── 到期收益 ────────────────────────────────────────────
    if option_type.lower() == 'call':
        V = np.maximum(S_grid - K, 0.)
    else:
        V = np.maximum(K - S_grid, 0.)

    # 预计算三对角矩阵系数（内部节点 i=1,...,N_S-1）
    i_arr = np.arange(1, N_S)
    Si = S_grid[i_arr]
    alpha = dt * (0.5*sigma**2*Si**2/dS**2 - 0.5*b*Si/dS)
    gamma = dt * (0.5*sigma**2*Si**2/dS**2 + 0.5*b*Si/dS)
    beta  = 1. + dt * (sigma**2*Si**2/dS**2 + r)

    # 三对角矩阵的三条对角线
    lower = -alpha[1:]      # 下对角（从第 2 个内部节点开始）
    main  = beta            # 主对角
    upper = -gamma[:-1]     # 上对角（到倒数第 2 个内部节点）

    # ── 反向时间推演 ─────────────────────────────────────────
    for step in range(N_T):
        tau = (step + 1) * dt   # 离到期的时间（反向）

        # 右端向量（内部节点的当前值）
        rhs = V[1:N_S].copy()

        # 处理边界条件（隐式格式中边界进入右端向量）
        if option_type.lower() == 'call':
            bc_upper = S_max - K * exp(-r * (T - tau + dt))
        else:
            bc_upper = 0.
        bc_lower = 0. if option_type == 'call' else K * exp(-r * (T - tau + dt))

        rhs[0]  -= -alpha[0] * bc_lower    # 左边界影响第一个内部节点
        rhs[-1] -= -gamma[-1] * bc_upper   # 右边界影响最后一个内部节点

        # 求解三对角方程组（Thomas 算法，即追赶法）
        V_inner = _thomas_algorithm(lower, main.copy(), upper, rhs)

        # 组合完整解
        V_new = np.empty(N_S + 1)
        V_new[0]    = bc_lower
        V_new[N_S]  = bc_upper
        V_new[1:N_S] = V_inner

        # 美式提前行权
        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                V_new = np.maximum(V_new, np.maximum(S_grid - K, 0.))
            else:
                V_new = np.maximum(V_new, np.maximum(K - S_grid, 0.))

        V = V_new

    # 插值
    i_S0 = int(S / dS)
    i_S0 = max(1, min(i_S0, N_S - 1))
    frac = (S - S_grid[i_S0]) / dS
    return float(V[i_S0] * (1 - frac) + V[i_S0 + 1] * frac)


def _thomas_algorithm(lower: np.ndarray, main: np.ndarray,
                       upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Thomas 算法（追赶法）求解三对角线性方程组 Ax = b。

    三对角矩阵结构：
    [ main[0] upper[0]                    ]   [ x[0] ]   [ rhs[0] ]
    [ lower[0] main[1] upper[1]           ]   [ x[1] ]   [ rhs[1] ]
    [          lower[1] main[2] upper[2]  ] · [ x[2] ] = [ rhs[2] ]
    [                   ...               ]   [ ...  ]   [ ...    ]
    [                   lower[-1] main[-1]]   [x[-1] ]   [rhs[-1] ]

    算法复杂度：O(n)，是最高效的三对角求解方法。
    """
    n = len(main)
    m = main.copy()
    d = rhs.copy()

    # 前向消元
    for i in range(1, n):
        factor = lower[i-1] / m[i-1]
        m[i] -= factor * upper[i-1]
        d[i] -= factor * d[i-1]

    # 回代
    x = np.zeros(n)
    x[-1] = d[-1] / m[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - upper[i] * x[i+1]) / m[i]

    return x


# ═══════════════════════════════════════════════════════════════
# C. Crank-Nicolson 格式
# ═══════════════════════════════════════════════════════════════

def crank_nicolson_fdm(S: float, K: float, T: float,
                        r: float, b: float, sigma: float,
                        N_S: int = 200, N_T: int = 100,
                        option_type: str = 'call',
                        exercise: str = 'european') -> float:
    """
    Crank-Nicolson（CN）有限差分法——最常用的 FDM 格式。

    CN 格式是显式和隐式的等权平均（时间步 n 和 n+1 各取 0.5）：

    (V^{n+1} - V^n)/Δt = 0.5 · L(V^n) + 0.5 · L(V^{n+1})

    其中 L 是 BSM 的空间微分算子。

    等价的矩阵方程：
    A · V^{n+1} = B · V^n + 边界修正

    其中：
    A（求解矩阵）：主对角 = 1 + 0.5(α+γ+rΔt)，次对角 = -0.5α 或 -0.5γ
    B（显式部分）：主对角 = 1 - 0.5(α+γ+rΔt)，次对角 = +0.5α 或 +0.5γ

    系数：
    α_i = Δt·(σ²S_i²/ΔS² - b·S_i/ΔS)/2
    γ_i = Δt·(σ²S_i²/ΔS² + b·S_i/ΔS)/2

    CN 优势：
    1. 无条件稳定（比显式好）
    2. 时间 O(Δt²) 精度（比隐式好）
    3. 标准工业实现
    """
    S_max = 3.0 * K
    dS = S_max / N_S
    dt = T / N_T
    S_grid = np.linspace(0, S_max, N_S + 1)
    N_inner = N_S - 1

    # ── 到期收益 ────────────────────────────────────────────
    if option_type.lower() == 'call':
        V = np.maximum(S_grid - K, 0.)
    else:
        V = np.maximum(K - S_grid, 0.)

    # ── 预计算系数 ───────────────────────────────────────────
    i_arr = np.arange(1, N_S)
    Si = S_grid[i_arr]
    alpha = dt * (0.5*sigma**2*Si**2/dS**2 - 0.5*b*Si/dS)
    gamma = dt * (0.5*sigma**2*Si**2/dS**2 + 0.5*b*Si/dS)
    r_dt  = r * dt

    # CN 矩阵 A（隐式部分，乘以 0.5）
    a_lower = -0.5 * alpha[1:]
    a_main  =  1. + 0.5*(alpha + gamma) + 0.5*r_dt
    a_upper = -0.5 * gamma[:-1]

    # CN 矩阵 B（显式部分，乘以 0.5）
    b_lower =  0.5 * alpha[1:]
    b_main  =  1. - 0.5*(alpha + gamma) - 0.5*r_dt
    b_upper =  0.5 * gamma[:-1]

    # ── 反向时间推演 ─────────────────────────────────────────
    for step in range(N_T):
        tau_new = (step + 1) * dt    # 当前步（反向）离到期的时间
        tau_old = step * dt           # 上一步离到期的时间

        # 边界条件（时间中点的平均）
        if option_type.lower() == 'call':
            bc_lower_new = 0.
            bc_upper_new = S_max - K * exp(-r * (T - tau_new + dt))
            bc_lower_old = 0.
            bc_upper_old = S_max - K * exp(-r * (T - tau_old + dt))
        else:
            bc_lower_new = K * exp(-r * (T - tau_new + dt))
            bc_upper_new = 0.
            bc_lower_old = K * exp(-r * (T - tau_old + dt))
            bc_upper_old = 0.

        # 右端向量：B · V^{old}（内部节点）
        V_inner = V[1:N_S]
        rhs = (b_lower * np.roll(V_inner, 1)[1:]
               if len(V_inner) > 1 else np.zeros(0))

        # 完整的 B · V 计算（矩阵向量乘）
        rhs = np.zeros(N_inner)
        rhs[0]  = b_main[0]*V_inner[0] + b_upper[0]*V_inner[1]
        rhs[-1] = b_lower[-1]*V_inner[-2] + b_main[-1]*V_inner[-1]
        for i in range(1, N_inner - 1):
            rhs[i] = (b_lower[i-1]*V_inner[i-1]
                      + b_main[i]*V_inner[i]
                      + b_upper[i]*V_inner[i+1])

        # 边界修正（CN 格式下，两个时间点的平均）
        rhs[0]  += 0.5*alpha[0]*(bc_lower_old + bc_lower_new)
        rhs[-1] += 0.5*gamma[-1]*(bc_upper_old + bc_upper_new)

        # 求解三对角方程组 A · V^{new} = rhs
        V_inner_new = _thomas_algorithm(a_lower.copy(), a_main.copy(),
                                         a_upper.copy(), rhs)

        # 组合完整解
        V_new = np.empty(N_S + 1)
        V_new[0]    = bc_lower_new
        V_new[N_S]  = bc_upper_new
        V_new[1:N_S] = V_inner_new

        # 美式提前行权（projected SOR 或简单最大值）
        if exercise.lower() == 'american':
            if option_type.lower() == 'call':
                intrinsic = np.maximum(S_grid - K, 0.)
            else:
                intrinsic = np.maximum(K - S_grid, 0.)
            V_new = np.maximum(V_new, intrinsic)

        V = V_new

    # 线性插值得到 S 对应的期权价格
    i_S0 = int(S / dS)
    i_S0 = max(1, min(i_S0, N_S - 1))
    frac = (S - S_grid[i_S0]) / dS
    return float(V[i_S0] * (1 - frac) + V[i_S0 + 1] * frac)


# ═══════════════════════════════════════════════════════════════
# D. 有限差分法 Greeks（通过网格直接计算）
# ═══════════════════════════════════════════════════════════════

def fdm_greeks(S: float, K: float, T: float,
               r: float, b: float, sigma: float,
               method: str = 'CN',
               option_type: str = 'call',
               exercise: str = 'european') -> dict:
    """
    从有限差分网格直接提取 Greeks（Delta, Gamma, Theta）。

    有限差分网格天然提供局部导数估计：
    Delta ≈ (V_{i+1} - V_{i-1}) / (2ΔS)   ← 中央空间差分
    Gamma ≈ (V_{i+1} - 2V_i + V_{i-1}) / ΔS²  ← 二阶导数
    Theta ≈ (V^{n+1} - V^n) / Δt           ← 时间导数

    参数
    ----
    method : 'CN'（Crank-Nicolson）, 'implicit', 'explicit'
    """
    S_max = 3.0 * K
    N_S, N_T = 200, 100
    dS = S_max / N_S
    dt = T / N_T
    S_grid = np.linspace(0, S_max, N_S + 1)

    # 找到最近节点
    i_S0 = int(round(S / dS))
    i_S0 = max(2, min(i_S0, N_S - 2))

    # 计算当前期权价格
    if method == 'CN':
        V0 = crank_nicolson_fdm(S, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vup = crank_nicolson_fdm(S + dS, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vdn = crank_nicolson_fdm(S - dS, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vt  = crank_nicolson_fdm(S, K, T - dt, r, b, sigma, N_S, N_T, option_type, exercise)
    else:
        V0  = implicit_fdm(S, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vup = implicit_fdm(S + dS, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vdn = implicit_fdm(S - dS, K, T, r, b, sigma, N_S, N_T, option_type, exercise)
        Vt  = implicit_fdm(S, K, T - dt, r, b, sigma, N_S, N_T, option_type, exercise)

    delta = (Vup - Vdn) / (2 * dS)
    gamma = (Vup - 2*V0 + Vdn) / dS**2
    theta = (Vt - V0) / dt   # Δ时间 = dt 时的价格变化 / dt

    return {
        'price': V0,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,   # 注意：theta = ∂V/∂t（正值），实际 theta = -∂V/∂t（负值）
    }


if __name__ == "__main__":
    print("=" * 65)
    print("有限差分法（FDM）期权定价 — 数值示例（Haug Chapter 7）")
    print("=" * 65)

    S, K, T, r, b, sigma = 100., 100., 0.5, 0.10, 0.10, 0.20
    N_S, N_T = 200, 100

    # BSM 理论值
    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    from utils.common import norm_cdf as Nf
    bsm_call = S*exp((b-r)*T)*Nf(d1) - K*exp(-r*T)*Nf(d2)
    bsm_put  = K*exp(-r*T)*Nf(-d2) - S*exp((b-r)*T)*Nf(-d1)
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"BSM 理论值：看涨 = {bsm_call:.5f}, 看跌 = {bsm_put:.5f}")

    # ── 欧式期权对比 ─────────────────────────────────────────
    print(f"\n欧式期权定价对比（N_S={N_S}, N_T={N_T}）：")
    print(f"  {'方法':<22} {'看涨':>10} {'看跌':>10} {'看涨误差':>12}")

    ex_c = explicit_fdm(S, K, T, r, b, sigma, N_S, 1000, 'call', 'european')
    ex_p = explicit_fdm(S, K, T, r, b, sigma, N_S, 1000, 'put',  'european')
    print(f"  {'显式(N_T=1000)':<22} {ex_c:>10.5f} {ex_p:>10.5f} {ex_c-bsm_call:>+12.6f}")

    im_c = implicit_fdm(S, K, T, r, b, sigma, N_S, N_T, 'call', 'european')
    im_p = implicit_fdm(S, K, T, r, b, sigma, N_S, N_T, 'put',  'european')
    print(f"  {'隐式(N_T=100)':<22} {im_c:>10.5f} {im_p:>10.5f} {im_c-bsm_call:>+12.6f}")

    cn_c = crank_nicolson_fdm(S, K, T, r, b, sigma, N_S, N_T, 'call', 'european')
    cn_p = crank_nicolson_fdm(S, K, T, r, b, sigma, N_S, N_T, 'put',  'european')
    print(f"  {'Crank-Nicolson(N_T=100)':<22} {cn_c:>10.5f} {cn_p:>10.5f} {cn_c-bsm_call:>+12.6f}")

    # ── 美式期权 ─────────────────────────────────────────────
    print(f"\n美式期权（b=0 期货期权，提前行权有价值）：")
    b_f = 0.0
    am_cn_c = crank_nicolson_fdm(S, K, T, r, b_f, sigma, N_S, N_T, 'call', 'american')
    am_cn_p = crank_nicolson_fdm(S, K, T, r, b_f, sigma, N_S, N_T, 'put',  'american')

    d1f = (log(S/K) + (b_f + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2f = d1f - sigma*sqrt(T)
    eu_c_fut = S*exp((b_f-r)*T)*Nf(d1f) - K*exp(-r*T)*Nf(d2f)

    print(f"  欧式看涨（b=0）= {eu_c_fut:.5f}")
    print(f"  美式看涨 CN    = {am_cn_c:.5f}  （溢价: {am_cn_c-eu_c_fut:+.5f}）")
    print(f"  美式看跌 CN    = {am_cn_p:.5f}")

    # ── 收敛性分析（N_S 的影响）────────────────────────────
    print(f"\nCrank-Nicolson 空间网格收敛性（N_T=100）：")
    print(f"  {'N_S':>6}  {'看涨价格':>12}  {'BSM误差':>12}")
    for ns in [25, 50, 100, 200, 400]:
        v_cn = crank_nicolson_fdm(S, K, T, r, b, sigma, ns, 100, 'call', 'european')
        print(f"  {ns:>6}  {v_cn:>12.6f}  {v_cn-bsm_call:>+12.7f}")
