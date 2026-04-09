"""
01_bsm_greeks.py — BSM 期权希腊值（Greeks）全集
================================================
【模型简介】
希腊值（Greeks）衡量期权价格对各种市场变量的敏感度，是期权
风险管理和对冲的核心工具。本文件基于广义 BSM 框架（持有成本 b）
实现全部一阶、二阶、三阶希腊值。

持有成本参数 b 的含义（同 generalized_bsm.py）：
  b = r         → 无红利股票
  b = r - q     → 连续红利股票/指数
  b = 0         → 期货
  b = r_d - r_f → 外汇

【希腊值速查表】
Delta  (Δ)  : ∂V/∂S        价格对股价的一阶敏感度
Gamma  (Γ)  : ∂²V/∂S²      Delta 对股价的敏感度
Vega   (ν)  : ∂V/∂σ        价格对波动率的敏感度
Theta  (Θ)  : ∂V/∂t        价格随时间的衰减
Rho    (ρ)  : ∂V/∂r        价格对利率的敏感度
Vanna       : ∂Delta/∂σ = ∂Vega/∂S
Charm       : ∂Delta/∂t（Delta 随时间的衰减）
Vomma  (Volga): ∂Vega/∂σ   Vega 对波动率的二阶敏感度
Speed       : ∂Gamma/∂S
Zomma       : ∂Gamma/∂σ
Color       : ∂Gamma/∂t

书中对应：Haug (2007), Chapter 2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def _d1d2(S, K, T, b, sigma):
    """计算广义 BSM 的 d₁ 和 d₂，供所有希腊值函数复用。"""
    d1 = (log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


# ═══════════════════════════════════════════════════════════════
# 一阶希腊值
# ═══════════════════════════════════════════════════════════════

def delta(S: float, K: float, T: float, r: float, b: float,
          sigma: float, option_type: str = 'call') -> float:
    """
    Delta (Δ) = ∂V/∂S — 期权价格对标的资产价格的一阶导数。

    经济含义
    --------
    - 代表期权的"等效持仓"：Delta = 0.5 意味着持有该期权
      的风险等同于持有 0.5 份标的资产
    - 是 delta 对冲的核心：做空 Delta 份标的资产可使组合
      对小幅价格变动免疫（delta-neutral）
    - 近似于期权到期价内（ITM）的概率（风险中性下为 N(d₁)）

    公式
    ----
    看涨：Δ_c = e^{(b-r)·T} · N(d₁)
    看跌：Δ_p = e^{(b-r)·T} · (N(d₁) - 1) = -e^{(b-r)·T} · N(-d₁)

    范围：看涨 ∈ (0, 1)，看跌 ∈ (-1, 0)
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = _d1d2(S, K, T, b, sigma)
    carry = exp((b - r) * T)

    if option_type.lower() == 'call':
        return carry * N(d1)
    elif option_type.lower() == 'put':
        return carry * (N(d1) - 1.0)
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


def gamma(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Gamma (Γ) = ∂²V/∂S² — Delta 对标的资产价格的一阶导数。

    经济含义
    --------
    - Gamma 越大，Delta 变化越快，对冲越难（需要频繁再平衡）
    - 买入期权（long option）：Gamma > 0，价格变动使 Delta 朝有利方向变化
    - 在平值（ATM）期权和到期日临近时 Gamma 最大
    - 对看涨和看跌期权相同（Put-Call Symmetry）

    公式
    ----
    Γ = e^{(b-r)·T} · n(d₁) / (S · σ · √T)

    其中 n(d₁) = 正态密度函数（衡量在 d₁ 处的"概率质量"）
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, b, sigma)
    return exp((b - r) * T) * n(d1) / (S * sigma * sqrt(T))


def gammaP(S: float, K: float, T: float, r: float, b: float,
           sigma: float) -> float:
    """
    GammaP = Γ · S / 100 — 百分比 Gamma（每 1% 股价变动的 Delta 变化）。

    GammaP 比 Gamma 更直观：它表示当股价变动 1% 时，
    Delta 变化多少个百分点，与绝对价格无关。
    """
    return gamma(S, K, T, r, b, sigma) * S / 100.0


def vega(S: float, K: float, T: float, r: float, b: float,
         sigma: float) -> float:
    """
    Vega (ν) = ∂V/∂σ — 期权价格对波动率的一阶导数。

    经济含义
    --------
    - Vega 衡量波动率变动 1 个单位（100%）时期权价值的变化
    - 实际中常表示为"每 1% 波动率变动时的价格变化"（需除以 100）
    - 买入期权（无论看涨看跌）Vega > 0：高波动率对持有人有利
    - 在平值期权和到期时间较长时 Vega 最大

    公式
    ----
    ν = S · e^{(b-r)·T} · n(d₁) · √T

    注：对看涨和看跌期权 Vega 相同（因 Put-Call Parity）
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, b, sigma)
    return S * exp((b - r) * T) * n(d1) * sqrt(T)


def vegaP(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    VegaP = ν · σ / 10 — 百分比 Vega（每 10% 相对波动率变动的价值变化）。

    适合对比不同行权价、不同到期日期权的波动率风险。
    """
    return vega(S, K, T, r, b, sigma) * sigma / 10.0


def theta(S: float, K: float, T: float, r: float, b: float,
          sigma: float, option_type: str = 'call') -> float:
    """
    Theta (Θ) = ∂V/∂t — 期权价值随时间的变化率（时间价值衰减）。

    经济含义
    --------
    - Theta < 0（对于普通期权）：随着时间流逝，期权价值下降
    - 越接近到期日，ATM 期权的时间价值衰减越快
    - 卖出期权（short option）可从 Theta 中获益
    - 注意：这里计算的是每日历年的衰减；实际每日衰减 ≈ Theta/365

    公式（看涨，基于广义 BSM）
    --------
    Θ_c = -[S·e^{(b-r)T}·n(d₁)·σ/(2√T)]
           - (b-r)·S·e^{(b-r)T}·N(d₁)
           - r·K·e^{-rT}·N(d₂)

    看跌：将 N(d₁) 替换为 N(-d₁)，N(d₂) 替换为 N(-d₂)，
          并调整符号
    """
    if T <= 0:
        return 0.0

    d1, d2 = _d1d2(S, K, T, b, sigma)
    carry   = exp((b - r) * T)
    discount = exp(-r * T)

    # 所有期权共同的时间衰减项（波动率相关）
    common = -S * carry * n(d1) * sigma / (2 * sqrt(T))

    if option_type.lower() == 'call':
        theta_val = (common
                     - (b - r) * S * carry * N(d1)
                     - r * K * discount * N(d2))
    elif option_type.lower() == 'put':
        theta_val = (common
                     + (b - r) * S * carry * N(-d1)
                     + r * K * discount * N(-d2))
    else:
        raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")

    return theta_val


def rho(S: float, K: float, T: float, r: float, b: float,
        sigma: float, option_type: str = 'call') -> float:
    """
    Rho (ρ) = ∂V/∂r — 期权价格对无风险利率的一阶导数。

    经济含义
    --------
    - 利率上升：看涨期权价值上升（远期价格上升），看跌下降
    - 深度价内（ITM）长期期权对利率最敏感
    - 对股票期权（b=r），Rho 为正（看涨）或负（看跌）

    公式
    ----
    看涨：ρ_c = +T · K · e^{-rT} · N(d₂)
    看跌：ρ_p = -T · K · e^{-rT} · N(-d₂)
    """
    if T <= 0:
        return 0.0

    _, d2 = _d1d2(S, K, T, b, sigma)

    if option_type.lower() == 'call':
        return T * K * exp(-r * T) * N(d2)
    elif option_type.lower() == 'put':
        return -T * K * exp(-r * T) * N(-d2)
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


def phi_carry_rho(S: float, K: float, T: float, r: float, b: float,
                  sigma: float, option_type: str = 'call') -> float:
    """
    Phi / Rho-2 = ∂V/∂b — 期权价格对持有成本的敏感度。

    对外汇期权（b = r_d - r_f）：Phi = ∂V/∂r_d（本币利率敏感度）

    公式
    ----
    看涨：Phi_c = +T · S · e^{(b-r)T} · N(d₁)
    看跌：Phi_p = -T · S · e^{(b-r)T} · N(-d₁)
    """
    if T <= 0:
        return 0.0

    d1, _ = _d1d2(S, K, T, b, sigma)
    carry = exp((b - r) * T)

    if option_type.lower() == 'call':
        return T * S * carry * N(d1)
    elif option_type.lower() == 'put':
        return -T * S * carry * N(-d1)
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


# ═══════════════════════════════════════════════════════════════
# 二阶希腊值（Cross-Greeks）
# ═══════════════════════════════════════════════════════════════

def vanna(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Vanna = ∂Delta/∂σ = ∂Vega/∂S — Delta 对波动率的敏感度。

    经济含义
    --------
    - 衡量当波动率变化时，Delta 如何变化（对冲比率的波动率风险）
    - 也反映当股价变化时，Vega 如何变化
    - 对于 delta/vega 中性对冲组合，Vanna 会暴露剩余风险

    公式
    ----
    Vanna = -e^{(b-r)T} · n(d₁) · d₂ / σ
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, b, sigma)
    return -exp((b - r) * T) * n(d1) * d2 / sigma


def charm(S: float, K: float, T: float, r: float, b: float,
          sigma: float, option_type: str = 'call') -> float:
    """
    Charm = ∂Delta/∂t = -∂²V/∂S∂T — Delta 随时间的变化率。

    经济含义
    --------
    - 预测对冲比率在无价格变动时如何随时间变化
    - 对于做市商维持 delta-neutral 组合非常重要

    公式
    ----
    看涨：Charm_c = e^{(b-r)T} · [n(d₁) · (2bT - d₂·σ√T)/(2Tσ√T) + (b-r)·N(d₁)]
    看跌：Charm_p = e^{(b-r)T} · [n(d₁) · (2bT - d₂·σ√T)/(2Tσ√T) - (b-r)·N(-d₁)]
    """
    if T <= 0:
        return 0.0

    d1, d2 = _d1d2(S, K, T, b, sigma)
    carry   = exp((b - r) * T)
    vol_t   = sigma * sqrt(T)

    # 共同项（来自密度函数部分）
    density_term = n(d1) * (2 * b * T - d2 * vol_t) / (2 * T * vol_t)

    if option_type.lower() == 'call':
        return carry * (density_term + (b - r) * N(d1))
    elif option_type.lower() == 'put':
        return carry * (density_term - (b - r) * N(-d1))
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


def vomma(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Vomma / Volga = ∂Vega/∂σ = ∂²V/∂σ² — Vega 对波动率的二阶导数。

    经济含义
    --------
    - 衡量 Vega 对波动率变化的敏感度（凸性）
    - Vomma > 0 时，随波动率上升 Vega 增加（有利于持有者）
    - 对波动率交易（volatility trading）至关重要

    公式
    ----
    Vomma = Vega · d₁ · d₂ / σ
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, b, sigma)
    return vega(S, K, T, r, b, sigma) * d1 * d2 / sigma


def speed(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Speed = ∂Gamma/∂S = ∂³V/∂S³ — Gamma 对股价的一阶导数。

    经济含义
    --------
    - 衡量当股价变化时，对冲需要调整的 Gamma 头寸大小
    - 用于三阶 Gamma 对冲

    公式
    ----
    Speed = -Gamma/S · (d₁/(σ√T) + 1)
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, b, sigma)
    g = gamma(S, K, T, r, b, sigma)
    return -g / S * (d1 / (sigma * sqrt(T)) + 1)


def zomma(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Zomma = ∂Gamma/∂σ — Gamma 对波动率的一阶导数。

    经济含义
    --------
    - 衡量当波动率变化时，Gamma 如何变化
    - 对于 Gamma-Vega 联合对冲策略非常重要

    公式
    ----
    Zomma = Gamma · (d₁·d₂ - 1) / σ
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, b, sigma)
    g = gamma(S, K, T, r, b, sigma)
    return g * (d1 * d2 - 1) / sigma


def color(S: float, K: float, T: float, r: float, b: float,
          sigma: float) -> float:
    """
    Color = ∂Gamma/∂t — Gamma 对时间的一阶导数（每日 Gamma 衰减）。

    公式
    ----
    Color = -e^{(b-r)T}·n(d₁)/(2S·T·σ√T) ·
            [2bT + 1 + d₁·(2bT - d₂·σ√T)/(σ√T)]
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, b, sigma)
    vol_t = sigma * sqrt(T)
    carry = exp((b - r) * T)
    inner = 2 * b * T + 1 + d1 * (2 * b * T - d2 * vol_t) / vol_t
    return -carry * n(d1) / (2 * S * T * vol_t) * inner


def ultima(S: float, K: float, T: float, r: float, b: float,
           sigma: float) -> float:
    """
    Ultima = ∂³V/∂σ³ — Vomma 对波动率的一阶导数（第三阶波动率敏感度）。

    公式
    ----
    Ultima = -Vega/(σ²) · [d₁·d₂·(1 - d₁·d₂) + d₁² + d₂²]
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, b, sigma)
    v = vega(S, K, T, r, b, sigma)
    return -v / (sigma ** 2) * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)


# ═══════════════════════════════════════════════════════════════
# 概率希腊值
# ═══════════════════════════════════════════════════════════════

def itm_probability(S: float, K: float, T: float, r: float, b: float,
                    sigma: float, option_type: str = 'call') -> float:
    """
    风险中性下到期时期权价内（ITM）的概率。

    公式
    ----
    P(S_T > K) = N(d₂)  （看涨）
    P(S_T < K) = N(-d₂) （看跌）

    注意：这是 d₂ 对应的概率，不是 d₁（d₁ 对应的是 delta）
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        return 1.0 if S < K else 0.0

    _, d2 = _d1d2(S, K, T, b, sigma)

    if option_type.lower() == 'call':
        return N(d2)
    return N(-d2)


def elasticity(S: float, K: float, T: float, r: float, b: float,
               sigma: float, option_type: str = 'call') -> float:
    """
    Elasticity (弹性) = Delta · S / V — 期权价格对股价的百分比弹性。

    经济含义
    --------
    弹性衡量股价变动 1% 时期权价格变动的百分比，反映期权的杠杆效应。
    对于价外（OTM）期权，弹性通常很大（如 5~20x）。
    """
    from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
    V = generalized_bsm(S, K, T, r, b, sigma, option_type)
    if abs(V) < 1e-10:
        return float('inf')
    d = delta(S, K, T, r, b, sigma, option_type)
    return d * S / V


# ═══════════════════════════════════════════════════════════════
# 数值希腊值（有限差分，用于验证解析公式）
# ═══════════════════════════════════════════════════════════════

def numerical_delta(pricing_func, dS=0.01, **kwargs):
    """中心差分法计算 Delta ≈ (V(S+dS) - V(S-dS)) / (2·dS)"""
    S = kwargs['S']
    v_up   = pricing_func(**{**kwargs, 'S': S + dS})
    v_down = pricing_func(**{**kwargs, 'S': S - dS})
    return (v_up - v_down) / (2 * dS)


def numerical_gamma(pricing_func, dS=0.01, **kwargs):
    """中心差分法计算 Gamma ≈ (V(S+dS) - 2V(S) + V(S-dS)) / dS²"""
    S = kwargs['S']
    v_up   = pricing_func(**{**kwargs, 'S': S + dS})
    v_mid  = pricing_func(**kwargs)
    v_down = pricing_func(**{**kwargs, 'S': S - dS})
    return (v_up - 2 * v_mid + v_down) / (dS ** 2)


def numerical_vega(pricing_func, dsigma=0.001, **kwargs):
    """中心差分法计算 Vega ≈ (V(σ+dσ) - V(σ-dσ)) / (2·dσ)"""
    s = kwargs['sigma']
    v_up   = pricing_func(**{**kwargs, 'sigma': s + dsigma})
    v_down = pricing_func(**{**kwargs, 'sigma': s - dsigma})
    return (v_up - v_down) / (2 * dsigma)


def numerical_theta(pricing_func, dT=1/365, **kwargs):
    """前向差分法计算 Theta ≈ (V(T-dT) - V(T)) / dT"""
    T = kwargs['T']
    if T - dT <= 0:
        return pricing_func(**{**kwargs, 'T': max(T - dT, 1e-6)}) - pricing_func(**kwargs)
    v_now  = pricing_func(**kwargs)
    v_later = pricing_func(**{**kwargs, 'T': T - dT})
    return (v_later - v_now) / dT


if __name__ == "__main__":
    print("=" * 60)
    print("BSM 希腊值全集 — 数值示例（Haug Chapter 2）")
    print("=" * 60)

    # 示例参数：S=105, K=100, T=0.5, r=0.10, b=0.10, σ=0.36
    S, K, T, r, b, sigma = 105, 100, 0.5, 0.10, 0.10, 0.36

    call_delta = delta(S, K, T, r, b, sigma, 'call')
    call_gamma = gamma(S, K, T, r, b, sigma)
    call_vega  = vega(S, K, T, r, b, sigma)
    call_theta = theta(S, K, T, r, b, sigma, 'call')
    call_rho   = rho(S, K, T, r, b, sigma, 'call')

    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"\n【一阶希腊值】")
    print(f"  Delta  = {call_delta:.6f}  （参考值 ≈ 0.5946）")
    print(f"  Vega   = {call_vega:.6f}  （每单位波动率变化）")
    print(f"  Theta  = {call_theta:.6f}  （每年时间衰减）")
    print(f"  Rho    = {call_rho:.6f}")
    print(f"\n【二阶希腊值】")
    print(f"  Gamma  = {call_gamma:.6f}")
    print(f"  Vanna  = {vanna(S, K, T, r, b, sigma):.6f}")
    print(f"  Charm  = {charm(S, K, T, r, b, sigma, 'call'):.6f}")
    print(f"  Vomma  = {vomma(S, K, T, r, b, sigma):.6f}")
    print(f"\n【三阶希腊值】")
    print(f"  Speed  = {speed(S, K, T, r, b, sigma):.6f}")
    print(f"  Zomma  = {zomma(S, K, T, r, b, sigma):.6f}")
    print(f"  Color  = {color(S, K, T, r, b, sigma):.6f}")
    print(f"  Ultima = {ultima(S, K, T, r, b, sigma):.6f}")
    print(f"\n【概率希腊值】")
    print(f"  ITM Prob (Call) = {itm_probability(S, K, T, r, b, sigma, 'call'):.6f}")
    print(f"\n【看跌期权 Greeks】")
    print(f"  Put Delta = {delta(S, K, T, r, b, sigma, 'put'):.6f}")
    print(f"  Put Theta = {theta(S, K, T, r, b, sigma, 'put'):.6f}")
    print(f"  Put Rho   = {rho(S, K, T, r, b, sigma, 'put'):.6f}")

    # ── 验证：解析 Delta vs 数值 Delta ──────────────────────────
    from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
    def gbs_call(**kw): return generalized_bsm(kw['S'], K, T, r, b, sigma, 'call')
    num_delta = numerical_delta(gbs_call, dS=0.01, S=S)
    print(f"\n【验证：解析 Delta vs 数值 Delta】")
    print(f"  解析 Delta = {call_delta:.8f}")
    print(f"  数值 Delta = {num_delta:.8f}")
    print(f"  误差       = {abs(call_delta - num_delta):.2e}")
