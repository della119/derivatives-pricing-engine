"""
03_stochastic_vol_variance_swaps.py — 随机波动率模型 & 方差互换
===============================================================
【模型简介】

A. Hull-White (1987) 随机波动率期权定价
────────────────────────────────────────
Hull & White (1987) 允许波动率本身也是随机游走：
  dS/S = μ dt + √V dW₁
  dV   = μ_V V dt + ξ V dW₂
  Corr(dW₁, dW₂) = ρ_vS  （通常 ρ_vS = 0 或接近 0）

当 ρ_vS = 0（波动率与股价不相关）时，Hull-White 给出解析近似：

C ≈ C_BSM(σ̄) + 展开修正

其中 σ̄ 是未来方差的"均值"，修正项用到方差的均值和方差。

Stein-Stein (1991)、Heston (1993) 给出更完整的随机波动率框架。
本文件实现 Hull-White 近似（基于方差序列展开），以及
Heston (1993) 半解析定价（特征函数方法）。

B. 方差互换与波动率互换（Variance & Volatility Swaps）
───────────────────────────────────────────────────────
方差互换（Variance Swap）：
  到期收益 = N · (σ²_realized - K_var)
  其中 σ²_realized 是实现方差，K_var 是固定名义方差（合约方差）

无套利复制价格（Demeterfi et al 1999）：
  K_var = (2/T) · [r·T - (S₀·e^{rT}/F - 1) + ln(F/S₀)
           - Σ(OTM期权价格的积分)]

简化对数合约近似：
  K_var ≈ (2/T) · ln(F/K*)  （其中 F 是远期价格，K* 是参考行权价）

更实用的复制：通过离散期权价格的梯形积分近似波动率曲面积分。

C. Bates (1996) 跳扩散随机波动率
──────────────────────────────────
组合 Heston 随机波动率 + Merton 跳扩散，是市场标准模型之一。
本文件给出简化版本（仅波动率近似修正）。

参考：
  - Hull, J. & White, A. (1987). "The Pricing of Options on Assets
    with Stochastic Volatilities." Journal of Finance, 42(2), 281–300.
  - Heston, S.L. (1993). "A Closed-Form Solution for Options with
    Stochastic Volatility with Applications to Bond and Currency Options."
    Review of Financial Studies, 6(2), 327–343.
  - Demeterfi et al (1999). "A Guide to Volatility and Variance Swaps."
    Journal of Derivatives.
书中对应：Haug (2007), Chapter 6, Sections 6.8–6.11
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, pi
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. Hull-White (1987) 随机波动率近似
# ═══════════════════════════════════════════════════════════════

def hull_white_stochastic_vol(S: float, K: float, T: float,
                               r: float, b: float,
                               sigma0: float,
                               v_of_v: float,
                               rho_sv: float = 0.,
                               option_type: str = 'call') -> float:
    """
    Hull-White (1987) 随机波动率期权定价近似。

    假设方差过程 V(t) 满足对数正态（或几何布朗运动）：
    dV = μ_V V dt + ξ V dW_V

    其中：
      V₀    = sigma0²  = 初始方差
      ξ     = v_of_v   = "波动率的波动率"（ vol of vol）
      ρ_sv  = rho_sv   = 股价与波动率的相关系数

    Hull-White 近似（ρ_sv = 0 时精确，ρ_sv ≠ 0 时为一阶近似）：

    C ≈ C_BSM(σ̄) + c₁·ψ₂ + c₂·ψ₃ + ...

    其中 σ̄ 是 T 期间平均方差的期望值的根：
    σ̄ = sigma0  （假设 E[V̄] ≈ V₀ = sigma0²，简化）

    完整展开（对 ξ 的二阶修正）：
    E[V̄] = V₀
    Var[V̄] = V₀² · ξ² · [e^{ξ²T} - ξ²T - 1] / (ξ^4·T²/2) （精确）
    简化：Var[V̄] ≈ V₀² · ξ² · T / 3  （小 ξ 展开）

    参数
    ----
    sigma0  : 初始年化波动率（√V₀）
    v_of_v  : 波动率的波动率 ξ（即方差的扩散系数）
    rho_sv  : 股票与波动率的相关系数（ρ，Heston 中关键参数）
    """
    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 基础 BSM 价格
    d1 = (log(S/K) + (b + 0.5*sigma0**2)*T) / (sigma0*sqrt(T))
    d2 = d1 - sigma0*sqrt(T)
    cf = exp((b-r)*T); df = exp(-r*T)

    if option_type.lower() == 'call':
        bsm = S*cf*N(d1) - K*df*N(d2)
    else:
        bsm = K*df*N(-d2) - S*cf*N(-d1)

    # Hull-White 二阶修正（ξ 的二阶项）
    # ψ₂ = ∂²C/∂σ²|_σ₀ × (1/2) × Var[V̄]
    # ∂²C_BSM/∂V = (S·e^{(b-r)T}·n(d₁)·d₁·d₂) / (2·V·σ√T)
    V0 = sigma0**2
    var_vbar = V0**2 * v_of_v**2 * T / 3.0   # 简化近似

    # 二阶偏导数修正（关于方差 V 的二阶导）
    d_numerator = S * cf * n(d1) * d1 * d2
    if abs(sigma0) > 1e-8:
        d2CV2 = d_numerator / (2 * V0 * sigma0 * sqrt(T))
    else:
        d2CV2 = 0.

    hw_correction = 0.5 * d2CV2 * var_vbar

    # rho_sv 修正（一阶项）
    # 若 ρ ≠ 0：额外修正 ≈ ρ·ξ·(...) 项，简化处理
    if abs(rho_sv) > 1e-8:
        # 一阶 ρ 修正：ΔC ≈ ρ·ξ·T·(∂C/∂σ)·σ₀/σ_T
        vega_factor = S * cf * n(d1) * sqrt(T)   # BSM vega（对 σ）
        rho_correction = rho_sv * v_of_v * T * vega_factor * sigma0 / (sigma0*sqrt(T))
        # 这是一个简化项；Heston 模型提供精确结果
    else:
        rho_correction = 0.

    return max(bsm + hw_correction + rho_correction, 0.)


# ═══════════════════════════════════════════════════════════════
# B. Heston (1993) 半解析定价
#    特征函数方法（傅里叶变换）
# ═══════════════════════════════════════════════════════════════

def heston_option(S: float, K: float, T: float,
                  r: float, q: float,
                  v0: float, kappa: float, theta: float,
                  xi: float, rho: float,
                  option_type: str = 'call',
                  n_integration: int = 100) -> float:
    """
    Heston (1993) 随机波动率模型 — 半解析期权定价。

    Heston 模型：
    dS/S = (r - q) dt + √V dW₁
    dV   = κ(θ - V) dt + ξ√V dW₂
    dW₁·dW₂ = ρ dt

    其中：
      κ  = kappa  = 均值回归速度（vol 回归到长期均值的速度）
      θ  = theta  = 长期波动率方差（长期均值）
      ξ  = xi     = vol of vol（波动率的波动率）
      ρ  = rho    = 股价与波动率的相关系数（通常 ρ < 0）
      v0          = 初始方差（V₀ = sigma₀²）

    Feller 条件（确保方差保持正数）：2κθ ≥ ξ²

    半解析定价（特征函数积分）：
    C = S·e^{-qT}·P₁ - K·e^{-rT}·P₂

    其中 P₁, P₂ 通过以下积分得到：
    P_j = 1/2 + (1/π) · ∫₀^∞ Re[e^{-iφln(K)}·φ_j(φ)] / (iφ) dφ

    φ_j(φ) 是 Heston 特征函数（闭合形式），通过龙格-库塔或梯形积分近似。

    参数
    ----
    v0            : 初始方差（注意：非波动率，是方差！σ₀² = v0）
    kappa         : 均值回归速度 κ（>0）
    theta         : 长期方差均值 θ
    xi            : 波动率的波动率 ξ（vol of vol）
    rho           : 相关系数 ρ（stock与vol，通常 -1 < ρ < 0）
    n_integration : 梯形积分节点数（越多越精确，100 通常足够）
    """
    import numpy as np

    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 特征函数 φ_j(φ; j=1,2)
    # Heston (1993) 原始形式（保持数值稳定的变体）
    def heston_char_func(phi, j):
        """
        Heston 特征函数 φ_j(φ)。
        j=1: P₁ 使用的特征函数（资产测度）
        j=2: P₂ 使用的特征函数（风险中性测度）
        """
        if j == 1:
            u = 0.5
            b_j = kappa - rho * xi
        else:
            u = -0.5
            b_j = kappa

        a = kappa * theta
        d = np.sqrt((rho*xi*phi*1j - b_j)**2 - xi**2*(2*u*phi*1j - phi**2))
        g = (b_j - rho*xi*phi*1j + d) / (b_j - rho*xi*phi*1j - d)

        # 对数特征函数（避免溢出）
        log_S  = np.log(S)
        log_term = np.log((1 - g*np.exp(d*T)) / (1 - g)) if abs(g) > 1e-12 else d*T

        C_func = ((r - q)*phi*1j*T
                  + (a/xi**2)*((b_j - rho*xi*phi*1j + d)*T - 2*log_term))
        D_func = ((b_j - rho*xi*phi*1j + d)/xi**2
                  * (1 - np.exp(d*T)) / (1 - g*np.exp(d*T)))

        return np.exp(C_func + D_func*v0 + 1j*phi*log_S)

    # 数值积分（梯形法则，上限 ω_max = 200 通常足够）
    omega_max = 200.
    dw = omega_max / n_integration
    omegas = np.linspace(dw, omega_max, n_integration)  # 避免 ω=0 处奇点

    log_K = log(K)

    def integrand_P(phi, j):
        cf = heston_char_func(phi, j)
        return np.real(np.exp(-1j * phi * log_K) * cf / (1j * phi))

    # P₁（资产测度，j=1）
    P1_values = np.array([integrand_P(w, 1) for w in omegas])
    P1 = 0.5 + (1./pi) * np.trapz(P1_values, omegas)

    # P₂（风险中性测度，j=2）
    P2_values = np.array([integrand_P(w, 2) for w in omegas])
    P2 = 0.5 + (1./pi) * np.trapz(P2_values, omegas)

    # Heston 期权价格
    call = S * exp(-q*T) * P1 - K * exp(-r*T) * P2

    if option_type.lower() == 'call':
        return max(call, 0.)
    # Put-Call 平价
    put = call - S*exp(-q*T) + K*exp(-r*T)
    return max(put, 0.)


# ═══════════════════════════════════════════════════════════════
# C. 方差互换（Variance Swap）定价
# ═══════════════════════════════════════════════════════════════

def variance_swap_strike(S: float, T: float, r: float,
                          call_strikes: list, call_prices: list,
                          put_strikes: list, put_prices: list,
                          K_atm: float = None) -> float:
    """
    方差互换公允方差（公允 K_var）计算。

    基于 Demeterfi-Derman-Kamal-Zou (1999) 静态复制方法：
    K_var = (2/T) · Σ [C(K)/K² · ΔK] 的离散积分近似

    公允方差 = 连续期权价格积分（OTM 期权 / K² 之和）：
    K_var = (2/T) · [∫₀^{F} P(K)/K² dK + ∫_F^∞ C(K)/K² dK]

    其中 F = S·e^{rT} 是远期价格

    参数
    ----
    call_strikes  : 看涨期权行权价列表（K > F 的 OTM 部分）
    call_prices   : 对应的看涨期权价格
    put_strikes   : 看跌期权行权价列表（K < F 的 OTM 部分）
    put_prices    : 对应的看跌期权价格
    K_atm         : ATM 行权价（若 None 使用 F）

    返回
    ----
    float : 公允方差互换率（年化方差，如 0.04 代表 σ=20%）
    """
    F = S * exp(r * T)   # 远期价格
    if K_atm is None:
        K_atm = F

    # 使用梯形积分近似
    # OTM 看跌（K ≤ F）
    put_integral = 0.
    sorted_puts = sorted(zip(put_strikes, put_prices))
    for i in range(len(sorted_puts) - 1):
        K1, P1 = sorted_puts[i]
        K2, P2 = sorted_puts[i+1]
        if K1 > F: continue
        K2 = min(K2, F)   # 截断在 F 处
        delta_K = K2 - K1
        # 梯形：(P1/K1² + P2/K2²) / 2 × ΔK
        put_integral += (P1/K1**2 + P2/K2**2) / 2. * delta_K

    # OTM 看涨（K ≥ F）
    call_integral = 0.
    sorted_calls = sorted(zip(call_strikes, call_prices))
    for i in range(len(sorted_calls) - 1):
        K1, C1 = sorted_calls[i]
        K2, C2 = sorted_calls[i+1]
        if K2 < F: continue
        K1 = max(K1, F)   # 截断在 F 处
        delta_K = K2 - K1
        call_integral += (C1/K1**2 + C2/K2**2) / 2. * delta_K

    K_var = (2. / T) * (put_integral + call_integral)
    return K_var


def variance_swap_log_approx(S: float, T: float, r: float,
                              sigma_atm: float) -> float:
    """
    方差互换的简单对数近似（用于快速估算）。

    当只有 ATM 隐含波动率时的粗略近似：
    K_var ≈ σ_ATM²

    更精确的近似（考虑偏斜）：
    K_var ≈ σ_ATM² · (1 + skew_factor)

    本函数返回 K_var = σ_ATM²（零阶近似）。
    在实践中，方差互换率 ≈ ATM^2 通常用于校核。
    """
    return sigma_atm**2


def volatility_swap_approx(K_var: float, T: float,
                            sigma: float, kappa: float, theta: float) -> float:
    """
    波动率互换（Volatility Swap）近似定价。

    波动率互换收益：σ_realized - K_vol（以波动率而非方差计）

    波动率互换的公允率 K_vol 与方差互换率 K_var 的近似关系：
    K_vol ≈ √K_var - Var(σ_realized) / (8 · K_var^{3/2})

    Jensen 不等式修正：由于 E[√V] < √E[V]，波动率互换率
    总是略低于方差互换率的平方根。

    参数
    ----
    K_var   : 公允方差互换率
    T       : 到期时间
    sigma   : 当前波动率（√V₀）
    kappa   : 均值回归速度（OU/Heston 参数）
    theta   : 长期方差均值

    返回
    ----
    float : 波动率互换的近似公允率
    """
    # Var[σ_realized] 的 Heston 近似
    # 在均值回归 vol 模型下，方差过程的方差：
    # Var[V̄_T] ≈ V₀·ξ² / (2·κ·T) · (1 - e^{-2κT}) （简化）
    # 这里用简单的 Jensen 修正
    sqrt_Kvar = sqrt(K_var)
    # Jensen 修正项（近似）：Var[V̄] 的粗略估算
    var_sigma = K_var * (theta / 8.) * (1. / sqrt_Kvar)  # 粗略近似
    K_vol = sqrt_Kvar - var_sigma
    return K_vol


if __name__ == "__main__":
    print("=" * 65)
    print("随机波动率模型 & 方差互换 — 数值示例（Haug Chapter 6）")
    print("=" * 65)

    # ── Hull-White 近似 ───────────────────────────────────────
    S, K, T = 100., 100., 0.5
    r, b, sigma0 = 0.10, 0.10, 0.20
    v_of_v, rho_sv = 0.30, 0.0

    from utils.common import norm_cdf as N_func, norm_pdf as n_func
    hw_call = hull_white_stochastic_vol(S, K, T, r, b, sigma0, v_of_v, rho_sv, 'call')
    hw_put  = hull_white_stochastic_vol(S, K, T, r, b, sigma0, v_of_v, rho_sv, 'put')

    # BSM 基准
    d1 = (log(S/K) + (b + 0.5*sigma0**2)*T) / (sigma0*sqrt(T))
    d2 = d1 - sigma0*sqrt(T)
    bsm_c = S*exp((b-r)*T)*N_func(d1) - K*exp(-r*T)*N_func(d2)

    print(f"\nHull-White 随机波动率近似")
    print(f"  S={S}, K={K}, T={T}, σ₀={sigma0}, ξ(vol_of_vol)={v_of_v}")
    print(f"  BSM 基准看涨 = {bsm_c:.4f}")
    print(f"  HW 随机波动率看涨 = {hw_call:.4f}  (修正量: {hw_call-bsm_c:+.4f})")
    print(f"  HW 随机波动率看跌 = {hw_put:.4f}")

    # ── Heston 模型 ───────────────────────────────────────────
    # Heston (1993) 论文参数
    S_h, K_h, T_h = 100., 100., 1.0
    r_h, q_h = 0.05, 0.0
    v0    = 0.04    # 初始方差（σ₀ = 20%）
    kappa = 2.0     # 均值回归速度
    theta = 0.04    # 长期方差均值（=v0, ATM）
    xi    = 0.30    # vol of vol
    rho   = -0.70   # 股价与波动率负相关（常见市场特征）

    print(f"\nHeston (1993) 随机波动率模型")
    print(f"  S={S_h}, K={K_h}, T={T_h}, r={r_h}")
    print(f"  v₀={v0}(σ₀={v0**0.5:.2%}), κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}")
    print(f"  Feller条件 2κθ={2*kappa*theta:.4f} {'≥' if 2*kappa*theta >= xi**2 else '<'} ξ²={xi**2:.4f}")

    try:
        hc = heston_option(S_h, K_h, T_h, r_h, q_h, v0, kappa, theta, xi, rho, 'call')
        hp = heston_option(S_h, K_h, T_h, r_h, q_h, v0, kappa, theta, xi, rho, 'put')
        print(f"  Heston 看涨 = {hc:.4f}")
        print(f"  Heston 看跌 = {hp:.4f}")

        # 不同相关系数下的期权价格（体现偏斜效应）
        print(f"\n  ρ 对期权价格的影响（体现波动率偏斜）：")
        for rho_t in [-0.9, -0.7, -0.5, 0.0, 0.5]:
            c_t = heston_option(S_h, K_h, T_h, r_h, q_h, v0, kappa, theta, xi, rho_t, 'call')
            print(f"    ρ={rho_t:+.1f}: 看涨={c_t:.4f}")
    except Exception as e:
        print(f"  [Heston 计算错误: {e}]")

    # ── 方差互换 ──────────────────────────────────────────────
    print(f"\n方差互换（Variance Swap）")
    # 模拟一组期权价格
    from math import exp as mexp
    S_vs, r_vs, T_vs = 100., 0.05, 1.
    F = S_vs * mexp(r_vs * T_vs)
    sigma_vs = 0.20  # ATM 波动率

    # 生成简化的 BSM 期权价格（实践中来自市场）
    put_strikes  = [70., 75., 80., 85., 90., 95.]
    call_strikes = [105., 110., 115., 120., 125., 130.]

    def simple_bsm_put(Kp):
        d2_ = (log(S_vs/Kp) + (r_vs-0.5*sigma_vs**2)*T_vs) / (sigma_vs*sqrt(T_vs))
        return Kp*mexp(-r_vs*T_vs)*N_func(-d2_) - S_vs*N_func(-d2_-sigma_vs*sqrt(T_vs))

    def simple_bsm_call(Kc):
        d1_ = (log(S_vs/Kc) + (r_vs+0.5*sigma_vs**2)*T_vs) / (sigma_vs*sqrt(T_vs))
        d2_ = d1_ - sigma_vs*sqrt(T_vs)
        return S_vs*N_func(d1_) - Kc*mexp(-r_vs*T_vs)*N_func(d2_)

    put_prices  = [max(simple_bsm_put(K), 0.) for K in put_strikes]
    call_prices = [max(simple_bsm_call(K), 0.) for K in call_strikes]

    K_var = variance_swap_strike(S_vs, T_vs, r_vs, call_strikes, call_prices,
                                  put_strikes, put_prices)
    print(f"  复制方差互换率 K_var = {K_var:.6f}  (={sqrt(K_var)*100:.2f}% 等价波动率)")
    print(f"  ATM 波动率近似 = {sigma_vs**2:.6f}  (={sigma_vs*100:.2f}%)")
    print(f"  说明：两者接近说明复制成功（BSM 无偏斜时应相等）")

    K_vol_approx = volatility_swap_approx(K_var, T_vs, sigma_vs, 2., 0.04)
    print(f"  波动率互换近似率 K_vol = {K_vol_approx:.4f}  ({K_vol_approx*100:.2f}%)")
    print(f"  Jensen 修正: K_vol < √K_var = {sqrt(K_var):.4f}（必然成立）")
