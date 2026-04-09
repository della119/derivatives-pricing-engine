"""
05_garman_kohlhagen.py — Garman-Kohlhagen (1983) 外汇期权
==========================================================
【模型简介】
Garman & Kohlhagen (1983) 将 BSM 模型推广至外汇期权定价，
同年 Grabbe (1983) 也独立提出类似公式。

核心洞察：持有外汇头寸会以外币利率 r_f 产生连续"红利"
（类似于 Merton 模型中的股息率），因此只需将 Merton 中的
q 替换为外币利率 r_f 即可。

应用场景：
  - 场外 FX 期权（如 USD/CNY 期权）
  - 交易所外汇期权（PHLX 等）
  - 结构性产品中的货币期权

参考：Garman, M.B. & Kohlhagen, S.W. (1983). "Foreign Currency Option Values."
       Journal of International Money and Finance, 2, 231–237.

书中对应：Haug (2007), Chapter 1, Section 1.1.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def garman_kohlhagen(S: float, K: float, T: float,
                     r_d: float, r_f: float,
                     sigma: float, option_type: str = 'call') -> float:
    """
    Garman-Kohlhagen (1983) 外汇欧式期权定价。

    参数
    ----
    S           : 即期汇率（本币/外币，如 USD/EUR = 1.10）
    K           : 行权汇率
    T           : 距到期日时间（年）
    r_d         : 本币（domestic）无风险利率（连续复利）
    r_f         : 外币（foreign）无风险利率（连续复利）
    sigma       : 汇率的年化波动率
    option_type : 'call'（看涨，买入外币权利）或 'put'（看跌）

    返回
    ----
    float : 期权价值（以本币计价）

    数学公式
    --------
    d₁ = [ln(S/K) + (r_d - r_f + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    看涨：C = S·e^{-r_f·T}·N(d₁) - K·e^{-r_d·T}·N(d₂)
    看跌：P = K·e^{-r_d·T}·N(-d₂) - S·e^{-r_f·T}·N(-d₁)

    解释
    ----
    - S·e^{-r_f·T}：外币资产（S）扣除外币利率"漏出"后的现值
      （持有 1 单位外币到期会"产生" r_f 利率，类似股息）
    - K·e^{-r_d·T}：行权时支付本币 K 的现值
    - 看涨期权赋予持有人以 K 本币买入 1 单位外币的权利
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # ── 外币利率充当连续红利率 ──────────────────────────────────
    # e^{-r_f·T}：折现外币面值（外币利率可视为"外币红利"）
    foreign_pv = exp(-r_f * T)
    domestic_df = exp(-r_d * T)

    d1 = (log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        price = S * foreign_pv * N(d1) - K * domestic_df * N(d2)
    elif option_type.lower() == 'put':
        price = K * domestic_df * N(-d2) - S * foreign_pv * N(-d1)
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


def fx_put_call_parity(S, K, T, r_d, r_f, sigma):
    """
    外汇期权的看涨-看跌平价关系：
    C - P = S·e^{-r_f·T} - K·e^{-r_d·T}
    """
    call = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'call')
    put  = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'put')
    lhs = call - put
    rhs = S * exp(-r_f * T) - K * exp(-r_d * T)
    return {'call': call, 'put': put, 'parity_lhs': lhs,
            'parity_rhs': rhs, 'error': abs(lhs - rhs)}


if __name__ == "__main__":
    print("=" * 55)
    print("Garman-Kohlhagen (1983) 外汇期权 — 数值示例")
    print("=" * 55)

    # 示例（Haug p.6）：EUR/USD
    # S=1.56, K=1.60, T=0.5, r_d=0.06, r_f=0.08, σ=0.12
    S, K, T, r_d, r_f, sigma = 1.56, 1.60, 0.5, 0.06, 0.08, 0.12
    call = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'call')
    put  = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'put')
    print(f"\n参数：S={S}, K={K}, T={T}, r_d={r_d}, r_f={r_f}, σ={sigma}")
    print(f"看涨（Call）= {call:.4f}  （参考值 ≈ 0.0291）")
    print(f"看跌（Put） = {put:.4f}")

    parity = fx_put_call_parity(S, K, T, r_d, r_f, sigma)
    print(f"\nPut-Call Parity 误差 = {parity['error']:.2e}")

    # USD/CNY 示例（假设参数）
    print(f"\nUSD/CNY 示例（假设参数）：")
    print(f"S=7.20, K=7.30, T=0.25, r_d=0.02, r_f=0.05, σ=0.05")
    c = garman_kohlhagen(7.20, 7.30, 0.25, 0.02, 0.05, 0.05, 'call')
    p = garman_kohlhagen(7.20, 7.30, 0.25, 0.02, 0.05, 0.05, 'put')
    print(f"USD看涨 = {c:.4f},  USD看跌 = {p:.4f}")
