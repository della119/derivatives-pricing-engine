"""
01_merton_jump_diffusion.py — Merton (1976) 跳扩散期权定价
==========================================================
【模型简介】
Black-Scholes-Merton 假设资产价格服从连续扩散过程，无法捕捉
"跳跃"（如公司重大事件、市场崩盘）带来的价格突变。
Merton (1976) 在标准 GBM 基础上叠加了一个复合泊松过程（跳跃项）：

dS/S = (μ - λk̄) dt + σ dW + J dN(λ)

其中：
  - σ    = 扩散波动率（"正常"的连续波动）
  - λ    = 跳跃强度（单位时间的平均跳跃次数）
  - J    = 跳跃幅度（随机），J ~ LogNormal(γ, δ²)
  - k̄  = E[J-1] = e^{γ+δ²/2} - 1（跳跃平均净增益）
  - N(λ) = 泊松过程，强度为 λ

关键：泊松过程的"每次"跳跃对应独立的对数正态幅度
（即每次跳多大也是随机的），级数求和给出精确解。

定价公式（欧式期权）：
将期权价格表达为 n 次跳跃条件下 BSM 价格的泊松加权和：

V = Σ_{n=0}^{∞} [e^{-λ'T}·(λ'T)^n / n!] · BSM(S, K, T, r_n, σ_n)

其中：
  λ' = λ(1 + k̄)            ← 风险调整后的跳跃强度
  r_n = r - λk̄ + n·γ/T     ← 第 n 次跳跃场景的有效无风险利率
  σ_n = √(σ² + n·δ²/T)     ← 第 n 次跳跃场景的有效波动率
  γ   = ln(1+k̄) - δ²/2    ← 跳跃对数均值（确保 E[J]=1+k̄）

直觉：
  n=0 项：无跳跃，退化为标准 BSM
  n=1 项：一次跳跃，增加一次跳跃后的 BSM
  n→∞ 项：级数快速收敛（30–50 项已足够精确）

参考：Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns
       Are Discontinuous." Journal of Financial Economics, 3, 125–144.
书中对应：Haug (2007), Chapter 6, Section 6.3
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, factorial
from utils.common import norm_cdf as N


def _bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """标准 BSM 看涨（无红利，b=r 情形）。内部辅助函数。"""
    if sigma <= 0 or T <= 0:
        return max(S - K * exp(-r*T), 0.)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S * N(d1) - K * exp(-r*T) * N(d2)


def _bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """标准 BSM 看跌。"""
    call = _bsm_call(S, K, T, r, sigma)
    return call - S + K * exp(-r*T)


def merton_jump_diffusion(S: float, K: float, T: float,
                          r: float, sigma: float,
                          lam: float, gamma: float, delta: float,
                          option_type: str = 'call',
                          n_terms: int = 50) -> float:
    """
    Merton (1976) 跳扩散模型 — 欧式期权定价。

    参数
    ----
    S         : 当前股价
    K         : 行权价
    T         : 到期时间（年）
    r         : 无风险利率（连续复利）
    sigma     : 扩散波动率（不含跳跃的"正常"波动）
    lam       : 跳跃强度 λ（每年平均跳跃次数，典型值 0–5）
    gamma     : 跳跃对数幅度的均值 γ（ln(J) 的期望，正 = 向上跳）
    delta     : 跳跃对数幅度的标准差 δ（跳跃幅度的不确定性）
    option_type : 'call' 或 'put'
    n_terms   : 级数截断项数（50 项足够精确）

    返回
    ----
    float : 期权价格

    推导细节
    --------
    跳跃幅度 J 的分布：
    ln(J) ~ N(γ, δ²)
    期望跳幅：E[J] = 1 + k̄ = e^{γ + δ²/2}
    k̄ = E[J] - 1 = e^{γ + δ²/2} - 1

    风险调整强度：λ' = λ(1 + k̄) = λ·e^{γ + δ²/2}

    第 n 项（n 次跳跃的条件 BSM 参数）：
    r_n   = r - λk̄ + n·γ/T     ← 漂移补偿 + n 次跳跃贡献
    σ_n   = √(σ² + n·δ²/T)     ← 扩散方差 + n 次跳跃方差
    权重  = e^{-λ'T}·(λ'T)^n / n!  ← 泊松概率（n 次跳跃）
    """
    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 跳跃均值和漂移调整
    kbar = exp(gamma + 0.5*delta**2) - 1.0   # E[J] - 1，跳跃平均净增益
    lam_prime = lam * (1.0 + kbar)             # 风险调整后的泊松强度

    price = 0.0
    for n in range(n_terms):
        # 泊松权重：P(N(T) = n)
        poisson_weight = exp(-lam_prime*T) * (lam_prime*T)**n / factorial(n)
        if poisson_weight < 1e-15:
            break  # 权重极小时截断

        # 第 n 跳跃场景的有效参数
        r_n   = r - lam*kbar + n*gamma/T          # 有效利率
        sigma_n = sqrt(sigma**2 + n*delta**2/T)   # 有效波动率

        # BSM 定价（n 次跳跃场景下）
        if option_type.lower() == 'call':
            bsm_n = _bsm_call(S, K, T, r_n, sigma_n)
        else:
            bsm_n = _bsm_put(S, K, T, r_n, sigma_n)

        price += poisson_weight * bsm_n

    return price


def merton_jump_implied_vol(S: float, K: float, T: float,
                             r: float, sigma: float,
                             lam: float, gamma: float, delta: float,
                             option_type: str = 'call') -> float:
    """
    Merton 跳扩散模型的"等价隐含波动率"（Equivalent Implied BSM Vol）。

    计算跳扩散期权价格所对应的 BSM 隐含波动率，用于理解
    跳跃如何扭曲波动率曲面（微笑/偏斜）。

    通过二分法逆推：BSM(S, K, T, r, σ_imp) = merton_price
    """
    from scipy.optimize import brentq

    target = merton_jump_diffusion(S, K, T, r, sigma, lam, gamma, delta, option_type)

    if option_type.lower() == 'call':
        def objective(sig):
            return _bsm_call(S, K, T, r, sig) - target
        # 内在价值检验
        intrinsic = max(S - K*exp(-r*T), 0.)
    else:
        def objective(sig):
            return _bsm_put(S, K, T, r, sig) - target
        intrinsic = max(K*exp(-r*T) - S, 0.)

    if target <= intrinsic + 1e-10:
        return 0.0

    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return float('nan')


def jump_diffusion_volatility_smile(S: float, T: float, r: float,
                                    sigma: float,
                                    lam: float, gamma: float, delta: float,
                                    moneyness_range: tuple = (0.7, 1.3),
                                    n_points: int = 11) -> list:
    """
    计算 Merton 跳扩散模型在不同执行价下的隐含波动率曲线（波动率微笑）。

    返回：[(K/S, implied_vol), ...] 的列表
    """
    from math import linspace_like  # 不可用，用手动方式
    lo, hi = moneyness_range
    step = (hi - lo) / (n_points - 1)
    strikes = [S * (lo + i*step) for i in range(n_points)]

    smile = []
    for K in strikes:
        iv = merton_jump_implied_vol(S, K, T, r, sigma, lam, gamma, delta, 'call')
        smile.append((K/S, iv))
    return smile


if __name__ == "__main__":
    print("=" * 60)
    print("Merton (1976) 跳扩散模型 — 数值示例")
    print("=" * 60)

    # 基准参数（Haug p.65 示例）
    # S=100, K=100, T=0.5, r=0.05, σ=0.20（扩散）
    # λ=1（每年1次跳跃），γ=-0.10（向下跳），δ=0.20（跳幅不确定）
    S, K, T, r = 100., 100., 0.5, 0.05
    sigma = 0.20
    lam, gamma, delta = 1.0, -0.10, 0.20

    jd_call = merton_jump_diffusion(S, K, T, r, sigma, lam, gamma, delta, 'call')
    jd_put  = merton_jump_diffusion(S, K, T, r, sigma, lam, gamma, delta, 'put')
    bsm_call_pure = _bsm_call(S, K, T, r, sigma)  # 纯 BSM（无跳跃）

    kbar = exp(gamma + 0.5*delta**2) - 1.
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}")
    print(f"     σ={sigma}, λ={lam}, γ={gamma}, δ={delta}")
    print(f"     跳跃均值 k̄ = {kbar:.4f}  ({kbar*100:.2f}%，负数表示向下跳)")
    print(f"\n纯 BSM（无跳）看涨  = {bsm_call_pure:.4f}")
    print(f"Merton 跳扩散看涨  = {jd_call:.4f}")
    print(f"Merton 跳扩散看跌  = {jd_put:.4f}")
    print(f"Put-Call 平价验证：C-P = {jd_call-jd_put:.4f}, "
          f"理论 S-Ke^{{-rT}} = {S - K*exp(-r*T):.4f}")

    # 跳跃强度对期权价格的影响
    print(f"\n跳跃强度 λ 对期权价格的影响（ATM 看涨）：")
    print(f"  {'λ':>6}  {'k̄':>8}  {'跳扩散':>10}  {'BSM差异':>10}")
    for lam_t in [0, 0.5, 1, 2, 5]:
        v = merton_jump_diffusion(S, K, T, r, sigma, lam_t, gamma, delta, 'call')
        print(f"  {lam_t:>6.1f}  {kbar:>8.4f}  {v:>10.4f}  {v-bsm_call_pure:>+10.4f}")

    # 波动率微笑（跳跃产生的偏斜）
    print(f"\n执行价与隐含波动率（跳跃产生的波动率偏斜）：")
    print(f"  {'K/S':>6}  {'跳扩散价格':>12}  {'隐含波动率':>12}")
    for moneyness in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
        Kk = S * moneyness
        v  = merton_jump_diffusion(S, Kk, T, r, sigma, lam, gamma, delta, 'call')
        iv = merton_jump_implied_vol(S, Kk, T, r, sigma, lam, gamma, delta, 'call')
        print(f"  {moneyness:>6.2f}  {v:>12.4f}  {iv:>12.4f}")

    # 项数收敛验证
    print(f"\n级数项数收敛性（λ={lam}, γ={gamma}, δ={delta}）：")
    for n in [5, 10, 20, 50, 100]:
        v = merton_jump_diffusion(S, K, T, r, sigma, lam, gamma, delta, 'call', n)
        print(f"  n={n:>3} 项: 价格 = {v:.6f}")
