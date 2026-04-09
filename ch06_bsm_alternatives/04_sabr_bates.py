"""
04_sabr_bates.py — SABR 模型 & Bates 跳扩散随机波动率
======================================================
【模型简介】

A. SABR 模型（Stochastic Alpha Beta Rho, Hagan et al 2002）
────────────────────────────────────────────────────────────
SABR 是利率衍生品领域最广泛使用的随机波动率模型之一，
同样适用于外汇和股票期权。

过程方程：
  dF = α·F^β dW₁          （远期价格 F，无漂移）
  dα = ν·α  dW₂           （波动率 α，对数正态）
  dW₁·dW₂ = ρ dt

参数含义：
  α（alpha）= 初始波动率（stochastic volatility 的初始水平）
  β（beta）  = 弹性参数：β=1 对数正态，β=0 正态，β=0.5 CIR 型
  ρ（rho）   = F 与 α 的相关系数（负值 → 负偏斜）
  ν（nu）    = 波动率的波动率（vol of vol，控制峰度）

Hagan et al (2002) 近似隐含波动率公式：
σ_imp(F, K) ≈ [复杂分析表达式]（见代码）

SABR 的主要用途：
1. 生成波动率微笑/偏斜（利率、FX 市场）
2. 校准到市场报价的 swaption 波动率曲面
3. Delta 对冲（SABR Delta 与 BSM Delta 不同）

B. Bates (1996) 模型：跳扩散 + 随机波动率
───────────────────────────────────────────
Bates 模型结合：
- Heston 随机波动率（均值回归，OU 过程）
- Merton 跳扩散（复合泊松过程）

过程方程：
  dS/S = (r-q-λk̄) dt + √V dW₁ + J dN(λ)
  dV   = κ(θ-V) dt + ξ√V dW₂
  dW₁·dW₂ = ρ dt

Bates 通过特征函数方法给出半解析定价（类似 Heston）。
本文件实现简化版本（Heston + 跳扩散的叠加近似）。

参考：
  - Hagan, P.S. et al (2002). "Managing Smile Risk."
    Wilmott Magazine, July, 84–108.
  - Bates, D.S. (1996). "Jumps and Stochastic Volatility:
    Exchange Rate Processes Implicit in Deutsche Mark Options."
    Review of Financial Studies, 9(1), 69–107.
书中对应：Haug (2007), Chapter 6, Sections 6.5, 6.9
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, pi
import numpy as np
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. SABR 隐含波动率公式（Hagan et al 2002）
# ═══════════════════════════════════════════════════════════════

def sabr_implied_vol(F: float, K: float, T: float,
                      alpha: float, beta: float,
                      rho: float, nu: float) -> float:
    """
    SABR 模型隐含波动率近似（Hagan et al 2002 公式）。

    这是 SABR 模型最重要的输出：给定模型参数，
    计算在 (F, K, T) 处的 Black-76 隐含波动率 σ_B。

    Hagan 近似公式（精确到 O(ε²)）：

    令 F_mid = (F·K)^{(1-β)/2}（几何中间点）
       z = (ν/α) · (FK)^{(1-β)/2} · ln(F/K)
       χ(z) = ln[(√(1-2ρz+z²)+z-ρ)/(1-ρ)]

    ATM 时（F=K，即 z=0 时）：
    σ_B^{ATM} = α / (F^{1-β}) · [1 + ((1-β)²σ²/24 + ρβνα/4 + (2-3ρ²)ν²/24)·T]

    非 ATM：
    σ_B = σ_B^{ATM} · z/χ(z) · [修正项]

    参数
    ----
    F     : 远期价格（或当前股价）
    K     : 行权价
    T     : 到期时间（年）
    alpha : 初始随机波动率 α（> 0）
    beta  : 弹性参数 β（∈ [0, 1]）
    rho   : 相关系数 ρ（∈ (-1, 1)）
    nu    : vol of vol ν（≥ 0）

    返回
    ----
    float : Black-76 隐含波动率（年化）
    """
    # 处理 ATM 极限（F ≈ K）
    if abs(F - K) < 1e-10 * F:
        # ATM 公式（更稳定）
        F_mid = F**(1. - beta)
        term1 = alpha / F_mid
        correction = ((1.-beta)**2 * alpha**2 / (24.*F**(2.-2.*beta))
                      + rho*beta*nu*alpha / (4.*F_mid)
                      + (2.-3.*rho**2) * nu**2 / 24.)
        return term1 * (1. + correction * T)

    # 非 ATM 情况
    log_FK = log(F / K)
    FK_mid = (F * K)**(0.5 * (1. - beta))   # (FK)^{(1-β)/2}

    # z 参数
    z = (nu / alpha) * FK_mid * log_FK

    # χ(z)
    denom_chi = sqrt(1. - 2.*rho*z + z**2) + z - rho
    if abs(denom_chi) < 1e-12:
        chi_z = 1.0
    else:
        chi_z = log(denom_chi / (1. - rho))
        if abs(chi_z) < 1e-12:
            chi_z = 1.0

    # 分子（含小 β 修正）
    numerator = alpha
    # 分母修正项（β 引起的）
    beta_correction = (1. + (1.-beta)**2 * log_FK**2 / 24.
                       + (1.-beta)**4 * log_FK**4 / 1920.)

    denom_base = FK_mid * beta_correction

    # 长期修正项（T 的 O(ε²) 项）
    long_term = (1. + ((1.-beta)**2 * alpha**2 / (24.*FK_mid**2)
                       + rho*beta*nu*alpha / (4.*FK_mid)
                       + (2.-3.*rho**2) * nu**2 / 24.) * T)

    sigma_B = (numerator / denom_base) * (z / chi_z) * long_term
    return max(sigma_B, 1e-10)


def sabr_option(F: float, K: float, T: float, r: float,
                 alpha: float, beta: float, rho: float, nu: float,
                 option_type: str = 'call') -> float:
    """
    SABR 模型期权定价（通过隐含波动率代入 Black-76）。

    流程：
    1. 用 sabr_implied_vol 计算 SABR 隐含波动率 σ_B
    2. 将 σ_B 代入 Black-76 公式计算期权价格

    参数
    ----
    F     : 远期价格
    K     : 行权价
    T     : 到期时间
    r     : 无风险利率（用于折现）
    """
    sigma_B = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)

    # Black-76 定价
    if T <= 0:
        if option_type == 'call': return exp(-r*T) * max(F - K, 0.)
        return exp(-r*T) * max(K - F, 0.)

    d1 = (log(F/K) + 0.5*sigma_B**2*T) / (sigma_B*sqrt(T))
    d2 = d1 - sigma_B*sqrt(T)
    df = exp(-r*T)

    if option_type.lower() == 'call':
        return df * (F * N(d1) - K * N(d2))
    return df * (K * N(-d2) - F * N(-d1))


def sabr_calibrate(F: float, T: float,
                    strikes: list, market_vols: list,
                    beta: float = 0.5,
                    initial_guess: tuple = None) -> tuple:
    """
    SABR 参数校准（最小化隐含波动率拟合误差）。

    给定市场隐含波动率，通过非线性优化找到最优 (α, ρ, ν)。
    β 通常预设（从历史数据估算或市场惯例）。

    参数
    ----
    strikes      : 行权价列表
    market_vols  : 对应的市场隐含波动率列表
    beta         : 预设弹性参数（通常 0 ≤ β ≤ 1）
    initial_guess: (α₀, ρ₀, ν₀) 初始猜测值

    返回
    ----
    (alpha, rho, nu) : 校准后的 SABR 参数
    """
    from scipy.optimize import minimize

    if initial_guess is None:
        # 简单初始猜测
        atm_vol = market_vols[len(market_vols)//2]
        F_beta = F**(1. - beta)
        alpha0 = atm_vol * F_beta   # 近似 ATM 关系
        initial_guess = (alpha0, -0.3, 0.3)

    def objective(params):
        alpha_t, rho_t, nu_t = params
        if alpha_t <= 0 or nu_t <= 0 or not (-0.999 < rho_t < 0.999):
            return 1e10
        total_err = 0.
        for K, mv in zip(strikes, market_vols):
            sv = sabr_implied_vol(F, K, T, alpha_t, beta, rho_t, nu_t)
            total_err += (sv - mv)**2
        return total_err

    result = minimize(objective, initial_guess,
                      method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000})
    alpha_opt, rho_opt, nu_opt = result.x
    return alpha_opt, rho_opt, nu_opt


# ═══════════════════════════════════════════════════════════════
# B. Bates (1996) 跳扩散 + 随机波动率模型
# ═══════════════════════════════════════════════════════════════

def bates_option(S: float, K: float, T: float,
                  r: float, q: float,
                  v0: float, kappa: float, theta: float,
                  xi: float, rho: float,
                  lam: float, mu_j: float, delta_j: float,
                  option_type: str = 'call',
                  n_integration: int = 100) -> float:
    """
    Bates (1996) 模型：Heston 随机波动率 + Merton 跳扩散。

    模型：
    dS/S = (r - q - λk̄) dt + √V dW₁ + (e^J - 1) dN(λ)
    dV   = κ(θ - V) dt + ξ√V dW₂
    J ~ N(μ_j, δ_j²)（跳跃的对数收益服从正态分布）

    其中 k̄ = E[e^J - 1] = e^{μ_j + δ_j²/2} - 1（期望跳幅）

    定价通过修正的特征函数（在 Heston 基础上叠加跳跃项）：

    φ_Bates(φ) = φ_Heston(φ) × exp[λT(e^{iφμ_j - φ²δ_j²/2} - 1 - iφk̄)]

    参数
    ----
    v0, kappa, theta, xi, rho : Heston 参数（同 ch06/03）
    lam    : 跳跃强度 λ（每年平均跳跃次数）
    mu_j   : 跳跃对数收益均值 μ_j
    delta_j: 跳跃对数收益标准差 δ_j
    """
    import numpy as np_

    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    kbar = exp(mu_j + 0.5*delta_j**2) - 1.   # 期望跳幅

    def bates_char_func(phi, j):
        """Bates 特征函数 = Heston φ × 跳跃修正项"""
        if j == 1:
            u = 0.5; b_j = kappa - rho*xi
        else:
            u = -0.5; b_j = kappa

        a = kappa * theta
        d = np_.sqrt((rho*xi*phi*1j - b_j)**2 - xi**2*(2*u*phi*1j - phi**2))
        g = (b_j - rho*xi*phi*1j + d) / (b_j - rho*xi*phi*1j - d)

        log_term = np_.log((1 - g*np_.exp(d*T))/(1-g)) if abs(g) > 1e-12 else d*T

        C_h = ((r-q)*phi*1j*T
               + (a/xi**2)*((b_j - rho*xi*phi*1j + d)*T - 2*log_term))
        D_h = ((b_j - rho*xi*phi*1j + d)/xi**2
               * (1 - np_.exp(d*T))/(1 - g*np_.exp(d*T)))

        heston_cf = np_.exp(C_h + D_h*v0 + 1j*phi*np_.log(S))

        # Bates 跳跃修正：exp[λT(e^{iφμ_j - φ²δ_j²/2} - 1 - iφk̄)]
        jump_correction = np_.exp(
            lam * T * (np_.exp(1j*phi*mu_j - 0.5*delta_j**2*phi**2) - 1. - 1j*phi*kbar)
        )

        return heston_cf * jump_correction

    # 数值积分（与 Heston 相同）
    omega_max = 200.
    dw = omega_max / n_integration
    omegas = np_.linspace(dw, omega_max, n_integration)
    log_K = log(K)

    def integrand_P(phi, j):
        cf = bates_char_func(phi, j)
        return np_.real(np_.exp(-1j*phi*log_K)*cf / (1j*phi))

    P1_vals = np_.array([integrand_P(w, 1) for w in omegas])
    P1 = 0.5 + (1./pi)*np_.trapz(P1_vals, omegas)
    P2_vals = np_.array([integrand_P(w, 2) for w in omegas])
    P2 = 0.5 + (1./pi)*np_.trapz(P2_vals, omegas)

    call = S*exp(-q*T)*P1 - K*exp(-r*T)*P2

    if option_type.lower() == 'call':
        return max(call, 0.)
    return max(call - S*exp(-q*T) + K*exp(-r*T), 0.)


def bates_approximation(S: float, K: float, T: float,
                         r: float, q: float,
                         v0: float, kappa: float, theta: float,
                         xi: float, rho: float,
                         lam: float, mu_j: float, delta_j: float,
                         option_type: str = 'call',
                         n_jump_terms: int = 10) -> float:
    """
    Bates 模型近似（Heston × 跳扩散 Poisson 求和）。

    将跳扩散部分展开为泊松级数，每一项使用独立的 Heston 价格：

    C_Bates ≈ Σ_{n=0}^{∞} [e^{-λT}(λT)^n/n!] × C_Heston(σ_n², r_n)

    其中跳跃调整参数：
    r_n   = r - λk̄ + n·μ_j/T   （有效利率）
    σ_n²  = v0 + n·δ_j²/T       （有效初始方差）

    这是 Bates 的近似版本（在小跳跃频率下精确），
    避免了复杂的数值积分。

    参数
    ----
    n_jump_terms : 泊松级数截断项数（10 项通常足够）
    """
    from math import factorial

    kbar = exp(mu_j + 0.5*delta_j**2) - 1.

    # 简化版：对每个跳跃项用 BSM（近似 Heston → BSM）
    # 这是近似近似，适合演示；精确实现需要完整 Heston 特征函数
    price = 0.
    for n in range(n_jump_terms):
        poisson_w = exp(-lam*T) * (lam*T)**n / factorial(n)
        if poisson_w < 1e-15:
            break

        r_n   = r - q - lam*kbar + n*mu_j/T
        v_n   = max(v0 + n*delta_j**2/T, 1e-8)
        sigma_n = sqrt(v_n)

        d1 = (log(S/K) + (r_n + 0.5*v_n)*T) / (sigma_n*sqrt(T))
        d2 = d1 - sigma_n*sqrt(T)

        if option_type.lower() == 'call':
            bsm_n = S*exp(-q*T)*N(d1) - K*exp(-r_n*T)*N(d2)
        else:
            bsm_n = K*exp(-r_n*T)*N(-d2) - S*exp(-q*T)*N(-d1)

        price += poisson_w * bsm_n

    return max(price, 0.)


if __name__ == "__main__":
    print("=" * 65)
    print("SABR 模型 & Bates 跳扩散随机波动率 — 数值示例")
    print("=" * 65)

    # ── SABR 隐含波动率微笑 ───────────────────────────────
    F0, T_s = 0.05, 1.0    # 远期利率 5%，1 年
    alpha_s, beta_s, rho_s, nu_s = 0.20, 0.5, -0.30, 0.40

    print(f"\nSABR 波动率微笑")
    print(f"  F={F0:.2%}, T={T_s}, α={alpha_s}, β={beta_s}, ρ={rho_s}, ν={nu_s}")
    print(f"  {'K/F':>6}  {'K':>8}  {'SABR σ_impl':>14}  {'ATM相对':>12}")
    atm_vol = sabr_implied_vol(F0, F0, T_s, alpha_s, beta_s, rho_s, nu_s)
    for m in [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]:
        K_t = F0 * m
        sv  = sabr_implied_vol(F0, K_t, T_s, alpha_s, beta_s, rho_s, nu_s)
        print(f"  {m:>6.2f}  {K_t:>8.4f}  {sv:>14.4f}  {sv-atm_vol:>+12.4f}")

    print(f"\n  ATM 隐含波动率 = {atm_vol:.4f}")

    # ρ 对波动率偏斜的影响
    print(f"\n  不同 ρ 对 OTM 看跌（K=0.04）的隐含波动率：")
    K_otm = 0.04
    for rho_t in [-0.9, -0.5, -0.3, 0., 0.3, 0.5]:
        sv_otm = sabr_implied_vol(F0, K_otm, T_s, alpha_s, beta_s, rho_t, nu_s)
        sv_atm = sabr_implied_vol(F0, F0,    T_s, alpha_s, beta_s, rho_t, nu_s)
        print(f"    ρ={rho_t:+.1f}: σ_ATM={sv_atm:.4f}, σ_OTM_put={sv_otm:.4f}, 偏斜={sv_otm-sv_atm:+.4f}")

    # SABR 期权定价
    r_s = 0.03; K_s = 0.048
    sc  = sabr_option(F0, K_s, T_s, r_s, alpha_s, beta_s, rho_s, nu_s, 'call')
    sp  = sabr_option(F0, K_s, T_s, r_s, alpha_s, beta_s, rho_s, nu_s, 'put')
    print(f"\n  利率期权（K={K_s:.2%}）: 看涨={sc:.6f}, 看跌={sp:.6f}")

    # ── Bates 模型 ────────────────────────────────────────────
    print(f"\nBates (1996) 跳扩散随机波动率模型")
    S_b, K_b, T_b = 100., 100., 1.0
    r_b, q_b = 0.05, 0.0
    # Heston 参数
    v0=0.04; kappa_b=2.; theta_b=0.04; xi_b=0.30; rho_b=-0.70
    # 跳跃参数
    lam_b=0.5; mu_j=-0.10; delta_j=0.15

    kbar_b = exp(mu_j + 0.5*delta_j**2) - 1.
    print(f"  Heston: v₀={v0}, κ={kappa_b}, θ={theta_b}, ξ={xi_b}, ρ={rho_b}")
    print(f"  跳扩散: λ={lam_b}, μ_j={mu_j}, δ_j={delta_j}, k̄={kbar_b:.4f}")

    bates_c = bates_approximation(S_b, K_b, T_b, r_b, q_b, v0, kappa_b, theta_b,
                                   xi_b, rho_b, lam_b, mu_j, delta_j, 'call')
    bates_p = bates_approximation(S_b, K_b, T_b, r_b, q_b, v0, kappa_b, theta_b,
                                   xi_b, rho_b, lam_b, mu_j, delta_j, 'put')

    # Heston 单独（λ=0，无跳跃）
    from utils.common import norm_cdf as Nf
    d1_b = (log(S_b/K_b) + (r_b-q_b + 0.5*v0)*T_b) / (sqrt(v0)*sqrt(T_b))
    d2_b = d1_b - sqrt(v0)*sqrt(T_b)
    bsm_b = S_b*exp(-q_b*T_b)*Nf(d1_b) - K_b*exp(-r_b*T_b)*Nf(d2_b)

    print(f"\n  BSM（σ=√v₀={sqrt(v0):.2%}）: {bsm_b:.4f}")
    print(f"  Bates 近似 看涨: {bates_c:.4f}  （跳跃影响: {bates_c-bsm_b:+.4f}）")
    print(f"  Bates 近似 看跌: {bates_p:.4f}")

    # 跳跃强度对期权的影响
    print(f"\n  跳跃强度 λ 对 ATM 看涨期权价格的影响：")
    for lam_t in [0., 0.2, 0.5, 1., 2.]:
        v = bates_approximation(S_b, K_b, T_b, r_b, q_b, v0, kappa_b, theta_b,
                                 xi_b, rho_b, lam_t, mu_j, delta_j, 'call')
        print(f"    λ={lam_t:.1f}: Bates看涨={v:.4f}  (vs BSM diff: {v-bsm_b:+.4f})")
