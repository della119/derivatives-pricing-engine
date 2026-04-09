"""
01_monte_carlo.py — 蒙特卡洛模拟期权定价
=========================================
【模型简介】
蒙特卡洛（Monte Carlo, MC）模拟通过模拟大量资产价格路径，
用路径收益的均值作为期权价格的估计值。

核心思想：
  E^Q[e^{-rT} · Payoff(S_T)] ≈ (1/N) · Σ_{i=1}^{N} e^{-rT} · Payoff(S_T^{(i)})

MC 的优势：
  - 可处理几乎任意复杂的收益结构（路径依赖、多资产）
  - 随着维度增加，误差增长率仅为 O(1/√N)（与维度无关）
  - 实现简单，易于理解和验证

MC 的劣势：
  - 收敛慢（相比解析公式）
  - 难以处理美式期权（需要 Longstaff-Schwartz 等方法）
  - 需要大量路径才能达到足够精度

本文件实现：
  A. 基础 MC（欧式期权）— GBM 路径模拟
  B. 方差缩减技术（Variance Reduction）
     - 对偶变量法（Antithetic Variates）
     - 控制变量法（Control Variates）
  C. 路径依赖期权 MC（亚式期权、障碍期权）
  D. 多资产期权 MC（相关资产联合路径）
  E. Longstaff-Schwartz 最小二乘 MC（美式期权）

参考：
  - Boyle, P.P. (1977). "Options: A Monte Carlo Approach."
    Journal of Financial Economics, 4, 323–338.
  - Longstaff, F.A. & Schwartz, E.S. (2001). "Valuing American
    Options by Simulation: A Simple Least-Squares Approach."
    Review of Financial Studies, 14(1), 113–147.
书中对应：Haug (2007), Chapter 8
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
import numpy as np
from utils.common import norm_cdf as N


# ═══════════════════════════════════════════════════════════════
# A. 基础 GBM 路径生成
# ═══════════════════════════════════════════════════════════════

def generate_gbm_paths(S: float, T: float, r: float, b: float,
                        sigma: float, n_paths: int, n_steps: int,
                        seed: int = None) -> np.ndarray:
    """
    生成 GBM（几何布朗运动）路径。

    离散化欧拉方案（精确离散化，无离散误差）：
    S_{t+Δt} = S_t · exp[(b - σ²/2)·Δt + σ·√Δt·Z]
    其中 Z ~ N(0,1)

    参数
    ----
    n_paths : 模拟路径数（如 100000）
    n_steps : 每条路径的时间步数（如 252 代表每日）
    seed    : 随机数种子（用于可重复结果）

    返回
    ----
    paths : shape (n_paths, n_steps+1) 的数组
            paths[:, 0] = S（初始价格）
            paths[:, -1] = S_T（到期价格）
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift = (b - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)

    # 生成标准正态随机数（n_paths × n_steps）
    Z = np.random.standard_normal((n_paths, n_steps))

    # 对数收益率
    log_returns = drift + diffusion * Z

    # 累积路径（log_returns 前加 log(S) 初始值）
    log_paths = np.concatenate(
        [np.full((n_paths, 1), log(S)), log_returns],
        axis=1
    )
    paths = np.exp(np.cumsum(log_paths, axis=1))
    return paths


# ═══════════════════════════════════════════════════════════════
# B. 欧式期权基础 MC
# ═══════════════════════════════════════════════════════════════

def mc_european(S: float, K: float, T: float,
                 r: float, b: float, sigma: float,
                 n_paths: int = 100_000, n_steps: int = 1,
                 option_type: str = 'call',
                 seed: int = 42) -> dict:
    """
    欧式期权蒙特卡洛定价（基础版本）。

    对于欧式期权，n_steps=1（只需到期时的股价）。
    多步骤路径用于路径依赖期权。

    返回
    ----
    dict：包含 price（价格估计）、std_error（标准误）、
          confidence_95（95% 置信区间）
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift    = (b - 0.5*sigma**2) * T
    diffusion = sigma * sqrt(T)

    # 欧式期权只需到期时的价格（单步）
    Z = np.random.standard_normal(n_paths)
    S_T = S * np.exp(drift + diffusion * Z)

    # 收益
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0.)
    else:
        payoffs = np.maximum(K - S_T, 0.)

    # 折现平均
    discounted = np.exp(-r * T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
    }


# ═══════════════════════════════════════════════════════════════
# C. 方差缩减：对偶变量法
# ═══════════════════════════════════════════════════════════════

def mc_antithetic(S: float, K: float, T: float,
                   r: float, b: float, sigma: float,
                   n_paths: int = 100_000,
                   option_type: str = 'call',
                   seed: int = 42) -> dict:
    """
    对偶变量法（Antithetic Variates）方差缩减。

    原理：对每个随机数 Z，同时计算 Z 和 -Z 两条路径的收益，
    取两者的平均。由于 Z 和 -Z 负相关，平均后的方差显著降低。

    eff_price = [Payoff(S_T(Z)) + Payoff(S_T(-Z))] / 2

    理论方差减少：若 Cov(Payoff(Z), Payoff(-Z)) < 0，
    则 Var(eff_price) = [Var + Cov] / 2 < Var/2

    对凸收益函数（如 Call），对偶变量法通常将方差减少 50%+。
    """
    if seed is not None:
        np.random.seed(seed)

    drift     = (b - 0.5*sigma**2) * T
    diffusion = sigma * sqrt(T)

    # 只生成 n_paths/2 个随机数，每个用 Z 和 -Z 两次
    n_half = n_paths // 2
    Z = np.random.standard_normal(n_half)

    S_T_pos = S * np.exp(drift + diffusion * Z)    # Z 路径
    S_T_neg = S * np.exp(drift - diffusion * Z)    # -Z 路径

    if option_type.lower() == 'call':
        payoffs_pos = np.maximum(S_T_pos - K, 0.)
        payoffs_neg = np.maximum(S_T_neg - K, 0.)
    else:
        payoffs_pos = np.maximum(K - S_T_pos, 0.)
        payoffs_neg = np.maximum(K - S_T_neg, 0.)

    # 对偶平均
    payoffs_avg = (payoffs_pos + payoffs_neg) / 2.
    discounted  = np.exp(-r*T) * payoffs_avg

    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_half)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
        'method': 'antithetic variates',
    }


# ═══════════════════════════════════════════════════════════════
# D. 方差缩减：控制变量法
# ═══════════════════════════════════════════════════════════════

def mc_control_variate(S: float, K: float, T: float,
                        r: float, b: float, sigma: float,
                        n_paths: int = 100_000,
                        option_type: str = 'call',
                        seed: int = 42) -> dict:
    """
    控制变量法（Control Variates）方差缩减。

    原理：选择一个与期权高度相关、但有已知解析值的"控制变量" C。
    然后用以下公式修正 MC 估计：

    V_adjusted = V_MC + β*(E[C] - C_MC)

    其中：β = Cov(V, C) / Var(C)（最优修正系数）
          E[C] = 控制变量的理论值（解析解）
          C_MC = 控制变量的 MC 估计值

    对于几何亚式期权（控制变量）控制算术亚式期权（目标）：
    几何亚式有闭合解，与算术亚式高度相关（ρ ≈ 0.99+）。

    本函数以欧式期权 BSM 价格作为控制变量（示例）：
    控制欧式 C → 目标亚式（粗略示范，实践中用几何亚式）。

    对于标准欧式期权，BSM 已是精确解，这里演示思路。
    """
    if seed is not None:
        np.random.seed(seed)

    drift     = (b - 0.5*sigma**2) * T
    diffusion = sigma * sqrt(T)

    Z = np.random.standard_normal(n_paths)
    S_T = S * np.exp(drift + diffusion * Z)

    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0.)
    else:
        payoffs = np.maximum(K - S_T, 0.)

    # 控制变量：S_T 本身（其期望为 S·e^{bT}）
    c_mc = S_T
    c_true = S * exp(b * T)    # E^Q[S_T] = S·e^{bT}

    # 最优 beta（协方差/方差）
    cov_matrix = np.cov(payoffs, c_mc)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.

    # 修正后的收益
    payoffs_cv = payoffs + beta * (c_true - c_mc)
    discounted = np.exp(-r*T) * payoffs_cv

    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
        'beta': beta,
        'method': 'control variates (delta hedge)',
    }


# ═══════════════════════════════════════════════════════════════
# E. 路径依赖期权 MC
# ═══════════════════════════════════════════════════════════════

def mc_asian_option(S: float, K: float, T: float,
                     r: float, b: float, sigma: float,
                     n_paths: int = 100_000, n_steps: int = 252,
                     average_type: str = 'arithmetic',
                     option_type: str = 'call',
                     seed: int = 42) -> dict:
    """
    亚式期权（Asian Option）蒙特卡洛定价。

    收益取决于路径上的平均股价（而非到期股价）：

    算术平均看涨：max(A_T - K, 0)，A_T = (1/n)·Σ S_{t_i}
    几何平均看涨：max(G_T - K, 0)，G_T = (Π S_{t_i})^{1/n}

    MC 是亚式期权最常用的定价方法（无简单闭合解）。
    几何平均亚式有闭合解（可作为算术平均的控制变量）。

    参数
    ----
    n_steps       : 路径步数（252 = 每日重采样，一年）
    average_type  : 'arithmetic' 或 'geometric'
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift     = (b - 0.5*sigma**2) * dt
    diffusion = sigma * sqrt(dt)

    # 生成路径
    Z = np.random.standard_normal((n_paths, n_steps))
    log_ret = drift + diffusion * Z
    log_paths = np.hstack([np.full((n_paths, 1), log(S)),
                            log_ret])
    paths = np.exp(np.cumsum(log_paths, axis=1))   # shape: (n_paths, n_steps+1)

    # 计算路径均值（不含 t=0 的初始价格，仅含 t=dt,...,T）
    S_path = paths[:, 1:]   # shape: (n_paths, n_steps)

    if average_type == 'arithmetic':
        avg = S_path.mean(axis=1)   # 算术平均
    else:  # geometric
        avg = np.exp(np.log(S_path).mean(axis=1))  # 几何平均

    if option_type.lower() == 'call':
        payoffs = np.maximum(avg - K, 0.)
    else:
        payoffs = np.maximum(K - avg, 0.)

    discounted = np.exp(-r*T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
        'n_steps': n_steps,
        'average_type': average_type,
    }


def mc_barrier_option(S: float, K: float, H: float, rebate: float,
                       T: float, r: float, b: float, sigma: float,
                       barrier_type: str = 'down-out',
                       option_type: str = 'call',
                       n_paths: int = 100_000, n_steps: int = 252,
                       seed: int = 42) -> dict:
    """
    障碍期权蒙特卡洛定价（连续监控近似）。

    通过多步路径模拟连续监控障碍（步数越多越精确）。

    连续障碍的离散化误差（Broadie-Glasserman-Kou 1997 修正）：
    对于下敲出看涨，修正后的障碍 H' = H · exp(±0.5826 · σ · √Δt)
    可减少离散误差，让少步数模拟更接近连续障碍价格。

    参数
    ----
    barrier_type : 'down-out', 'up-out', 'down-in', 'up-in'
    n_steps      : 障碍监控次数（步数越多，越接近连续监控）
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift     = (b - 0.5*sigma**2) * dt
    diffusion = sigma * sqrt(dt)

    # Broadie-Glasserman-Kou (1997) 离散化修正系数
    BGK_beta = 0.5826   # ≈ ζ(1/2) / √(2π)
    if 'down' in barrier_type:
        H_adj = H * exp(-BGK_beta * sigma * sqrt(dt))  # 调低障碍
    else:
        H_adj = H * exp(+BGK_beta * sigma * sqrt(dt))  # 调高障碍

    # 生成路径
    Z = np.random.standard_normal((n_paths, n_steps))
    log_ret = drift + diffusion * Z
    log_S0 = log(S)
    log_paths = np.hstack([np.full((n_paths, 1), log_S0),
                            log_ret])
    paths = np.exp(np.cumsum(log_paths, axis=1))   # (n_paths, n_steps+1)

    S_T = paths[:, -1]   # 到期股价

    # 障碍触碰检测
    if 'down' in barrier_type:
        knocked = (paths.min(axis=1) <= H_adj)   # 是否碰下障碍
    else:
        knocked = (paths.max(axis=1) >= H_adj)   # 是否碰上障碍

    # 收益计算
    if option_type.lower() == 'call':
        vanilla_payoff = np.maximum(S_T - K, 0.)
    else:
        vanilla_payoff = np.maximum(K - S_T, 0.)

    if 'out' in barrier_type:
        # 敲出：碰到障碍 → 收益为 rebate，否则为 vanilla
        payoffs = np.where(knocked, rebate, vanilla_payoff)
    else:
        # 敲入：碰到障碍 → vanilla，否则为 rebate
        payoffs = np.where(knocked, vanilla_payoff, rebate)

    discounted = np.exp(-r*T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
        'n_steps': n_steps,
        'barrier_type': barrier_type,
        'knock_ratio': knocked.mean(),
    }


# ═══════════════════════════════════════════════════════════════
# F. 多资产期权 MC（相关 GBM）
# ═══════════════════════════════════════════════════════════════

def mc_two_asset_option(S1: float, S2: float,
                         K: float, T: float,
                         r: float, b1: float, b2: float,
                         sigma1: float, sigma2: float, rho: float,
                         payoff_func,
                         n_paths: int = 100_000,
                         seed: int = 42) -> dict:
    """
    双资产期权蒙特卡洛定价（相关 GBM）。

    使用 Cholesky 分解生成相关正态随机数：
    Z₁ ~ N(0,1)
    Z₂ = ρ·Z₁ + √(1-ρ²)·Z₁'   （Z₁' 独立标准正态）

    参数
    ----
    b1, b2    : 两个资产的持有成本
    rho       : 相关系数
    payoff_func : 函数 (S1_T, S2_T) → payoff

    示例
    ----
    # 交换期权：Payoff = max(S1 - S2, 0)
    mc_two_asset_option(..., payoff_func=lambda s1,s2: np.maximum(s1-s2,0))
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成相关正态随机数（Cholesky 分解）
    Z1 = np.random.standard_normal(n_paths)
    Z2_indep = np.random.standard_normal(n_paths)
    Z2 = rho * Z1 + sqrt(1 - rho**2) * Z2_indep

    # 到期价格（精确 GBM 离散化）
    S1_T = S1 * np.exp((b1 - 0.5*sigma1**2)*T + sigma1*sqrt(T)*Z1)
    S2_T = S2 * np.exp((b2 - 0.5*sigma2**2)*T + sigma2*sqrt(T)*Z2)

    # 用户指定的收益函数
    payoffs = payoff_func(S1_T, S2_T)

    discounted = np.exp(-r*T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
    }


# ═══════════════════════════════════════════════════════════════
# G. Longstaff-Schwartz 最小二乘 MC（美式期权）
# ═══════════════════════════════════════════════════════════════

def mc_american_lsm(S: float, K: float, T: float,
                     r: float, b: float, sigma: float,
                     n_paths: int = 50_000, n_steps: int = 50,
                     option_type: str = 'put',
                     seed: int = 42) -> dict:
    """
    Longstaff-Schwartz (2001) 最小二乘蒙特卡洛（LSM）美式期权定价。

    LSM 算法核心思想：
    1. 生成股价路径（前向模拟）
    2. 从到期日向前递推（反向归纳）
    3. 在每个行权时点，通过最小二乘回归估计"持有期权"的条件期望
       E^Q[V_{t+1} | S_t, 价内]
    4. 比较立即行权价值与持有价值，决定是否提前行权

    回归基函数：Laguerre 多项式（常用），此处使用 1, S_t, S_t²

    注意：LSM 对看跌期权比看涨期权更常用（美式看涨通常不提前行权，
    除非有大额红利）。

    参数
    ----
    n_paths : 路径数（精度与路径数的平方根成正比）
    n_steps : 行权时点数（类似于二叉树的步数）
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    disc = exp(-r * dt)   # 单步折现因子

    # 生成完整路径
    Z = np.random.standard_normal((n_paths, n_steps))
    log_ret = (b - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z
    log_S0 = log(S)
    log_paths = np.hstack([np.full((n_paths, 1), log_S0),
                            log_ret])
    paths = np.exp(np.cumsum(log_paths, axis=1))  # (n_paths, n_steps+1)

    # 到期收益
    if option_type.lower() == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0.)
    else:
        payoffs = np.maximum(paths[:, -1] - K, 0.)

    # 保存每条路径的"现金流时间"（初始为到期日）
    cash_flows = payoffs.copy()
    # 时间折现索引：从到期折现到各行权时点
    cf_discount = np.ones(n_paths)   # 累计折现因子

    # ── 反向递推（从 n_steps-1 到 1）───────────────────────
    for step in range(n_steps - 1, 0, -1):
        S_now = paths[:, step]   # 当前时点的股价

        # 立即行权价值
        if option_type.lower() == 'put':
            intrinsic = np.maximum(K - S_now, 0.)
        else:
            intrinsic = np.maximum(S_now - K, 0.)

        # 只对价内路径进行回归
        in_the_money = intrinsic > 0
        if in_the_money.sum() < 10:
            cash_flows *= disc   # 直接折现，不行权
            continue

        # 持有期权的条件期望（通过最小二乘回归估计）
        S_itm = S_now[in_the_money]
        Y = cash_flows[in_the_money] * disc   # 折现后的持有价值

        # 基函数：[1, S, S², exp(-S/2)] （Laguerre 类）
        X = np.column_stack([
            np.ones(S_itm.shape),
            S_itm,
            S_itm**2,
        ])

        # 最小二乘回归估计持有价值
        try:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            holding_value = X @ coef
        except Exception:
            cash_flows *= disc
            continue

        # 行权决策：若立即行权价值 > 持有价值，则行权
        exercise_now = in_the_money.copy()
        exercise_now[in_the_money] = (intrinsic[in_the_money] > holding_value)

        # 更新现金流
        cash_flows[exercise_now]     = intrinsic[exercise_now]  # 行权
        cash_flows[~exercise_now]    *= disc                     # 继续持有（折现）

    # 最终价格：从 t=1 折现到 t=0 的平均值
    price = disc * cash_flows.mean()
    std_error = disc * cash_flows.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'n_paths': n_paths,
        'n_steps': n_steps,
        'method': 'Longstaff-Schwartz LSM',
    }


if __name__ == "__main__":
    print("=" * 65)
    print("蒙特卡洛模拟期权定价 — 数值示例（Haug Chapter 8）")
    print("=" * 65)

    S, K, T, r, b, sigma = 100., 100., 0.5, 0.10, 0.10, 0.20
    N = 100_000

    # BSM 理论值
    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    from utils.common import norm_cdf as Nf
    bsm_c = S*exp((b-r)*T)*Nf(d1) - K*exp(-r*T)*Nf(d2)
    bsm_p = K*exp(-r*T)*Nf(-d2) - S*exp((b-r)*T)*Nf(-d1)
    print(f"\nBSM理论值：看涨={bsm_c:.5f}, 看跌={bsm_p:.5f}")

    # ── 欧式 MC（基础）───────────────────────────────────────
    print(f"\n1. 欧式期权 MC 定价（N={N:,}条路径）：")
    mc_c = mc_european(S, K, T, r, b, sigma, N, option_type='call', seed=42)
    mc_p = mc_european(S, K, T, r, b, sigma, N, option_type='put',  seed=42)
    print(f"   基础MC 看涨: {mc_c['price']:.5f} ± {mc_c['std_error']:.5f}  (BSM误差: {mc_c['price']-bsm_c:+.5f})")
    print(f"   基础MC 看跌: {mc_p['price']:.5f} ± {mc_p['std_error']:.5f}")

    # ── 对偶变量法 ────────────────────────────────────────────
    print(f"\n2. 对偶变量法（方差缩减）：")
    av_c = mc_antithetic(S, K, T, r, b, sigma, N, 'call', seed=42)
    print(f"   对偶MC 看涨: {av_c['price']:.5f} ± {av_c['std_error']:.5f}")
    print(f"   标准误减少: {(1-av_c['std_error']/mc_c['std_error'])*100:.1f}%")

    # ── 亚式期权 MC ───────────────────────────────────────────
    print(f"\n3. 亚式期权（算术平均）MC：")
    asian_arith = mc_asian_option(S, K, T, r, b, sigma, N//5, 252, 'arithmetic', 'call', seed=42)
    asian_geo   = mc_asian_option(S, K, T, r, b, sigma, N//5, 252, 'geometric',  'call', seed=42)

    # 几何亚式闭合解（对照）
    from utils.common import norm_cdf as Nc
    b_A = 0.5*(b - sigma**2/6.)
    sigma_A = sigma / sqrt(3.)
    d1_g = (log(S/K) + (b_A + 0.5*sigma_A**2)*T) / (sigma_A*sqrt(T))
    d2_g = d1_g - sigma_A*sqrt(T)
    geo_closed = S*exp((b_A-r)*T)*Nc(d1_g) - K*exp(-r*T)*Nc(d2_g)

    print(f"   几何均值看涨 MC:   {asian_geo['price']:.5f} ± {asian_geo['std_error']:.5f}")
    print(f"   几何均值看涨 闭合:  {geo_closed:.5f}  (误差: {asian_geo['price']-geo_closed:+.5f})")
    print(f"   算术均值看涨 MC:   {asian_arith['price']:.5f} ± {asian_arith['std_error']:.5f}")

    # ── 障碍期权 MC ───────────────────────────────────────────
    print(f"\n4. 障碍期权（下敲出看涨, H=85）MC：")
    H = 85.
    bar_mc = mc_barrier_option(S, K, H, 0., T, r, b, sigma,
                                'down-out', 'call', N//5, 252, seed=42)
    # 闭合解参考（来自 ch04/05_barrier_options.py 的公式）
    print(f"   下敲出看涨 MC: {bar_mc['price']:.5f} ± {bar_mc['std_error']:.5f}")
    print(f"   敲出概率: {bar_mc['knock_ratio']:.4f} ({bar_mc['knock_ratio']*100:.2f}%的路径触碰障碍)")
    print(f"   普通看涨: {bsm_c:.5f}  (障碍导致价格下降: {bsm_c-bar_mc['price']:.5f})")

    # ── 双资产期权 MC ─────────────────────────────────────────
    print(f"\n5. 双资产交换期权（Margrabe）MC：")
    S1, S2, rho2 = 100., 105., 0.75
    sigma1, sigma2 = 0.20, 0.25
    sigma_12 = sqrt(sigma1**2 + sigma2**2 - 2*rho2*sigma1*sigma2)
    d1_m = (log(S1/S2) + 0.5*sigma_12**2*T) / (sigma_12*sqrt(T))
    d2_m = d1_m - sigma_12*sqrt(T)
    margrabe_ref = S1*exp((b-r)*T)*Nc(d1_m) - S2*exp((b-r)*T)*Nc(d2_m)

    mc_exch = mc_two_asset_option(S1, S2, K, T, r, b, b, sigma1, sigma2, rho2,
                                   lambda s1, s2: np.maximum(s1-s2, 0.), N, seed=42)
    print(f"   Margrabe 交换期权 MC:    {mc_exch['price']:.5f}")
    print(f"   Margrabe 交换期权 理论:  {margrabe_ref:.5f}")

    # ── LSM 美式期权 ──────────────────────────────────────────
    print(f"\n6. LSM 最小二乘 MC 美式看跌：")
    lsm = mc_american_lsm(S, K, T, r, b, sigma, n_paths=50_000, n_steps=50,
                           option_type='put', seed=42)
    print(f"   美式看跌 LSM:     {lsm['price']:.5f} ± {lsm['std_error']:.5f}")
    print(f"   欧式看跌 BSM:     {bsm_p:.5f}")
    print(f"   提前行权溢价:     {lsm['price']-bsm_p:.5f}")
