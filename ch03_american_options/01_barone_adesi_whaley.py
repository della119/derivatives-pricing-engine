"""
01_barone_adesi_whaley.py — Barone-Adesi & Whaley (1987) 美式期权近似
======================================================================
【模型简介】
Barone-Adesi & Whaley (1987) 提出了美式期权的二次近似定价方法（QA），
将美式期权价格分解为欧式价格加上提前行权溢价（Early Exercise Premium, EEP）。

关键洞察：提前行权溢价满足一个近似的偏微分方程，其解具有二次幂函数形式，
即 EEP ≈ A · (S/S*)^q，其中 S* 是最优提前行权边界。

适用范围：适用于广义 BSM 框架（股票、指数、期货、外汇）的美式期权，
尤其适合中短期期权（T ≤ 1 年）。

参考：Barone-Adesi, G. & Whaley, R.E. (1987). "Efficient Analytic
       Approximation of American Option Values."
       Journal of Finance, 42(2), 301–320.

书中对应：Haug (2007), Chapter 3, Section 3.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, inf
from utils.common import norm_cdf as N, norm_pdf as n
from scipy.optimize import brentq


def _bsm_call(S, K, T, r, b, sigma):
    """广义 BSM 看涨期权价格（内部辅助函数）。"""
    from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
    return generalized_bsm(S, K, T, r, b, sigma, 'call')


def _bsm_put(S, K, T, r, b, sigma):
    """广义 BSM 看跌期权价格（内部辅助函数）。"""
    from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
    return generalized_bsm(S, K, T, r, b, sigma, 'put')


def _bsm_delta_call(S, K, T, r, b, sigma):
    """广义 BSM 看涨 Delta（内部辅助函数）。"""
    d1 = (log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp((b - r) * T) * N(d1)


def _bsm_delta_put(S, K, T, r, b, sigma):
    """广义 BSM 看跌 Delta（内部辅助函数）。"""
    d1 = (log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp((b - r) * T) * (N(d1) - 1.0)


def barone_adesi_whaley(S: float, K: float, T: float, r: float, b: float,
                        sigma: float, option_type: str = 'call') -> float:
    """
    Barone-Adesi & Whaley (1987) 二次近似法 美式期权定价。

    参数
    ----
    S           : 当前标的资产价格
    K           : 行权价格
    T           : 距到期日时间（年）
    r           : 无风险利率（连续复利）
    b           : 持有成本
                  b = r     → 无红利股票
                  b = r-q   → 连续红利，q 为红利率
                  b = 0     → 期货
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    返回
    ----
    float : 美式期权价格

    算法说明
    --------
    设 M = 2r/σ², N = 2b/σ²

    对于看涨期权：
    ─ 当 b ≥ r 时（如无红利股票），不会提前行权，美式=欧式
    ─ 二次方程系数：q₂ = [-(N-1) + √((N-1)² + 4M/h)] / 2
      其中 h = 1 - e^{-rT}（时间衰减因子）
    ─ 寻找临界价格 S*（使提前行权恰好值得）：
      S* - K = BSM_call(S*) + (1 - e^{(b-r)T}·N(d₁(S*))) · S*/q₂
    ─ 定价：
      若 S < S*：C = BSM_call(S) + A₂·(S/S*)^q₂
      若 S ≥ S*：C = S - K（立即行权）
      其中 A₂ = S*/q₂ · (1 - e^{(b-r)T}·N(d₁(S*)))

    对于看跌期权：类似，但 q₁ < 0，且 S** < S*
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if option_type.lower() == 'call':
        return _baw_call(S, K, T, r, b, sigma)
    elif option_type.lower() == 'put':
        return _baw_put(S, K, T, r, b, sigma)
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


def _baw_call(S, K, T, r, b, sigma):
    """BAW 美式看涨期权（内部实现）。"""
    # 当 b ≥ r 时（无红利股票），永远不值得提前行权看涨期权
    # 直接返回欧式价格
    if b >= r:
        return _bsm_call(S, K, T, r, b, sigma)

    # ── 二次近似参数 ─────────────────────────────────────────────
    M  = 2 * r / (sigma ** 2)          # 折现因子参数
    N_ = 2 * b / (sigma ** 2)          # 持有成本参数
    h  = 1 - exp(-r * T)               # 时间衰减因子（趋近 1 时 T→∞）

    # q₂：二次方程 q² + (N-1)q - M/h = 0 的正根
    q2 = (-(N_ - 1) + sqrt((N_ - 1) ** 2 + 4 * M / h)) / 2

    # ── 寻找临界价格 S*（牛顿迭代） ─────────────────────────────
    # 条件：S* - K = BSM_call(S*) + (1 - e^{(b-r)T}·N(d₁(S*))) · S*/q₂
    def call_critical_eq(Sc):
        """S* 满足的方程：左边 = 右边，求解使函数值为 0 的 Sc"""
        bsm_c = _bsm_call(Sc, K, T, r, b, sigma)
        d1_sc = (log(Sc / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        lhs = Sc - K               # 立即行权收益
        rhs = bsm_c + (1 - exp((b - r) * T) * N(d1_sc)) * Sc / q2
        return lhs - rhs

    # S* 的搜索范围：从 K 到足够大的上界
    try:
        S_star = brentq(call_critical_eq, K * 1e-4, K * 1000, xtol=1e-8, maxiter=200)
    except ValueError:
        # 若无解（数值问题），返回欧式价格
        return _bsm_call(S, K, T, r, b, sigma)

    # ── 计算提前行权溢价系数 A₂ ─────────────────────────────────
    d1_star = (log(S_star / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    A2 = (S_star / q2) * (1 - exp((b - r) * T) * N(d1_star))

    # ── 最终定价 ─────────────────────────────────────────────────
    if S < S_star:
        # 持有期权（欧式价值 + 提前行权溢价的现值近似）
        return _bsm_call(S, K, T, r, b, sigma) + A2 * (S / S_star) ** q2
    else:
        # 立即行权更合算
        return max(S - K, 0.0)


def _baw_put(S, K, T, r, b, sigma):
    """BAW 美式看跌期权（内部实现）。"""
    # ── 二次近似参数 ─────────────────────────────────────────────
    M  = 2 * r / (sigma ** 2)
    N_ = 2 * b / (sigma ** 2)
    h  = 1 - exp(-r * T)

    # q₁：二次方程的负根（看跌期权对应 q₁ < 0）
    q1 = (-(N_ - 1) - sqrt((N_ - 1) ** 2 + 4 * M / h)) / 2

    # ── 寻找临界价格 S**（看跌期权触发点，S** < K） ─────────────
    def put_critical_eq(Sc):
        bsm_p = _bsm_put(Sc, K, T, r, b, sigma)
        d1_sc = (log(Sc / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        lhs = K - Sc               # 立即行权收益
        rhs = bsm_p - (1 - exp((b - r) * T) * (N(d1_sc) - 1)) * Sc / q1
        return lhs - rhs

    try:
        # S** 应在 (0, K) 范围内
        S_star_put = brentq(put_critical_eq, K * 1e-6, K, xtol=1e-8, maxiter=200)
    except ValueError:
        return _bsm_put(S, K, T, r, b, sigma)

    # ── A₁ 系数 ────────────────────────────────────────────────
    d1_star = (log(S_star_put / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    A1 = -(S_star_put / q1) * (1 - exp((b - r) * T) * (N(d1_star) - 1))

    # ── 最终定价 ─────────────────────────────────────────────────
    if S > S_star_put:
        # 持有期权（欧式价值 + 提前行权溢价）
        return _bsm_put(S, K, T, r, b, sigma) + A1 * (S / S_star_put) ** q1
    else:
        # 立即行权
        return max(K - S, 0.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Barone-Adesi & Whaley (1987) 美式期权近似 — 数值示例")
    print("=" * 60)

    # 示例（Haug p.98）：
    # S=100, K=100, T=0.5, r=0.10, b=0.10 (股票), σ=0.25
    S, K, T, r, b, sigma = 100, 100, 0.5, 0.10, 0.10, 0.25
    am_call = barone_adesi_whaley(S, K, T, r, b, sigma, 'call')
    am_put  = barone_adesi_whaley(S, K, T, r, b, sigma, 'put')
    eu_call = _bsm_call(S, K, T, r, b, sigma)
    eu_put  = _bsm_put(S, K, T, r, b, sigma)
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"美式看涨 = {am_call:.4f},  欧式看涨 = {eu_call:.4f},  提前行权溢价 = {am_call-eu_call:.4f}")
    print(f"美式看跌 = {am_put:.4f},  欧式看跌 = {eu_put:.4f},  提前行权溢价 = {am_put-eu_put:.4f}")

    # 连续红利股票（b = r - q = 0.10 - 0.05 = 0.05）
    print(f"\n连续红利股票（q=0.05, b=0.05）：")
    S, K, T, r, b, sigma = 100, 100, 0.5, 0.10, 0.05, 0.25
    am_call2 = barone_adesi_whaley(S, K, T, r, b, sigma, 'call')
    am_put2  = barone_adesi_whaley(S, K, T, r, b, sigma, 'put')
    eu_call2 = _bsm_call(S, K, T, r, b, sigma)
    eu_put2  = _bsm_put(S, K, T, r, b, sigma)
    print(f"美式看涨 = {am_call2:.4f},  欧式看涨 = {eu_call2:.4f},  EEP = {am_call2-eu_call2:.4f}")
    print(f"美式看跌 = {am_put2:.4f},  欧式看跌 = {eu_put2:.4f},  EEP = {am_put2-eu_put2:.4f}")

    # 期货期权（b=0）
    print(f"\n期货期权（b=0）：")
    S, K, T, r, b, sigma = 100, 100, 0.5, 0.10, 0.0, 0.35
    am_call3 = barone_adesi_whaley(S, K, T, r, b, sigma, 'call')
    am_put3  = barone_adesi_whaley(S, K, T, r, b, sigma, 'put')
    print(f"美式看涨 = {am_call3:.4f},  美式看跌 = {am_put3:.4f}")
    print(f"（参考值：call ≈ 8.20, put ≈ 8.38 Bjerksund）")
