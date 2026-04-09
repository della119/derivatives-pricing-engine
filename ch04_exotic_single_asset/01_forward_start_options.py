"""
01_forward_start_options.py — 远期开始期权 (Forward Start Options)
===================================================================
【模型简介】
远期开始期权（Forward Start Option）是一种在未来某时刻 t₁ 才
开始生效、且行权价按届时股价的比例 α 确定的期权。

典型用途：
  - 员工股票期权激励计划（每年重置行权价）
  - Cliquet（棘轮）期权的基本构成单元
  - 为将来某时刻设计 ATM 期权

定价逻辑：由于 t₁ 时的行权价 K = α·S_{t₁} 与 t₁ 时的股价成
比例，今天看来该期权等价于：以今天的股价 S 为基础、打折为
α·S 的远期起始 BSM 公式。

参考：Rubinstein, M. (1991). "Pay Now, Choose Later."
       RISK Magazine, Vol. 4, 13.
书中对应：Haug (2007), Chapter 4, Section 4.6
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def forward_start_option(S: float, alpha: float, t1: float, T: float,
                         r: float, b: float, sigma: float,
                         option_type: str = 'call') -> float:
    """
    远期开始期权定价（Rubinstein 1991）。

    参数
    ----
    S           : 当前标的资产价格
    alpha       : 行权价占 t₁ 时股价的比例
                  alpha = 1.0  → t₁ 时平值（ATM）
                  alpha < 1.0  → t₁ 时价内（ITM，有利于看涨）
                  alpha > 1.0  → t₁ 时价外（OTM）
    t1          : 期权开始生效的时刻（年，t₁ < T）
    T           : 期权到期时刻（年）
    r           : 无风险利率（连续复利）
    b           : 持有成本（b=r 无红利股票，b=r-q 连续红利）
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    返回
    ----
    float : 期权当前价值

    数学公式
    --------
    τ = T - t₁（期权生效后的剩余有效期）

    d₁ = [ln(1/α) + (b + σ²/2)·τ] / (σ·√τ)
    d₂ = d₁ - σ·√τ

    看涨：C = S·e^{(b-r)·T} · [N(d₁) - α·e^{-r·τ}·N(d₂)]
           = S·e^{(b-r)·t₁} · e^{(b-r)·τ} · [N(d₁) - α·e^{-r·τ}·N(d₂)]

    直觉
    ----
    在 t₁ 时，期权等同于持有一个到期 τ 的 BSM 期权（K=α·S_{t₁}）。
    因此今日价值 = S·e^{(b-r)·t₁}（持有到 t₁ 的调整因子）
                  × BSM(1, α, τ, r, b, σ) （按单位化股价计算）
    """
    # τ = 期权生效后的实际有效期
    tau = T - t1
    if tau <= 0:
        raise ValueError(f"T({T}) 须大于 t1({t1})")

    # ── d₁, d₂ 基于单位化价格（S/S_{t₁} = 1），行权价为 α ──────
    d1 = (log(1.0 / alpha) + (b + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    # ── 持有到 t₁ 的折现因子 ──────────────────────────────────────
    # e^{(b-r)·T} = e^{(b-r)·t₁} × e^{(b-r)·τ}，整合为一步
    carry_total = exp((b - r) * T)

    if option_type.lower() == 'call':
        # 单位股价的 BSM 看涨 × 持有调整
        price = S * carry_total * N(d1) - S * alpha * exp(-r * T) * exp(b * t1) * N(d2)
        # 等价简洁写法：
        price = S * exp((b - r) * t1) * (
            exp((b - r) * tau) * N(d1) - alpha * exp(-r * tau) * N(d2)
        )
    elif option_type.lower() == 'put':
        price = S * exp((b - r) * t1) * (
            alpha * exp(-r * tau) * N(-d2) - exp((b - r) * tau) * N(-d1)
        )
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return price


if __name__ == "__main__":
    print("=" * 55)
    print("远期开始期权 — 数值示例")
    print("=" * 55)

    # 示例（Haug p.122）：S=60, α=1.1, t₁=0.25, T=1, r=0.08, b=0.08, σ=0.30
    S, alpha, t1, T, r, b, sigma = 60, 1.1, 0.25, 1.0, 0.08, 0.08, 0.30
    call = forward_start_option(S, alpha, t1, T, r, b, sigma, 'call')
    put  = forward_start_option(S, alpha, t1, T, r, b, sigma, 'put')
    print(f"\n参数：S={S}, α={alpha}, t₁={t1}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"看涨（Call）= {call:.4f}  （参考值 ≈ 4.4064）")
    print(f"看跌（Put） = {put:.4f}")

    # ATM 远期开始期权（α=1）
    print(f"\nATM 远期开始（α=1）：S=100, t₁=0.5, T=1, r=0.05, b=0.05, σ=0.20")
    c_atm = forward_start_option(100, 1.0, 0.5, 1.0, 0.05, 0.05, 0.20, 'call')
    print(f"看涨 = {c_atm:.4f}")
