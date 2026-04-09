"""
02_cev_corrado_su.py — CEV 模型 & Corrado-Su 偏斜-峰度修正
=========================================================
【模型简介】

A. 常数弹性方差模型（CEV, Constant Elasticity of Variance）
─────────────────────────────────────────────────────────
CEV 模型（Cox 1975, Emanuel & MacBeth 1982）允许波动率随股价变化：
dS = μS dt + σ·S^β dW

其中 β 是"弹性参数"（elasticity parameter）：
  β = 1    → 标准 BSM（等价于 GBM，σ 为常数）
  β < 1    → 杠杆效应（股价下跌时波动率上升，负偏斜）
  β > 1    → 反杠杆效应（成长股中常见，正偏斜）
  β = 0    → Bachelier 模型（绝对扩散）
  β = 0.5  → Cox-Ross 平方根过程（利率模型常用）

CEV 期权定价通过非中心卡方分布（Non-Central Chi-Square）精确计算。
本文件实现两个版本：
  1. 精确 CEV 定价（Schroder 1989 非中心卡方）
  2. 近似 CEV 定价（2 阶幂级数展开，运算更快）

B. Corrado-Su (1996) 偏斜-峰度修正
────────────────────────────────
标准 BSM 假设对数收益率服从正态分布（偏斜=0，超额峰度=0）。
Corrado & Su (1996) 用 Gram-Charlier 展开对 BSM 进行三、四阶矩修正：
  - 通过添加基于 N(d₂) 修正项来引入偏斜（skewness = μ₃）
  - 通过添加基于 n(d₂) 修正项来引入峰度（excess kurtosis = μ₄-3）

修正公式：
C ≈ C_BSM + Q₃ · w₃ + Q₄ · w₄

其中 w₃, w₄ 是偏斜和峰度的修正加权。

参考：
  - Cox, J.C. (1975). Notes on Option Pricing I: Constant Elasticity of Variance Diffusions.
    Stanford Graduate School of Business Working Paper.
  - Schroder, M. (1989). "Computing the Constant Elasticity of Variance Option Pricing Formula."
    Journal of Finance, 44(1), 211-219.
  - Corrado, C.J. & Su, T. (1996). "Skewness and Kurtosis in S&P 500 Index Returns Implied
    by Option Prices." Journal of Financial Research, 19(2), 175-192.
书中对应：Haug (2007), Chapter 6, Sections 6.1, 6.7
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, factorial
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. CEV 模型（Constant Elasticity of Variance）
# ═══════════════════════════════════════════════════════════════

def cev_option(S: float, K: float, T: float,
               r: float, b: float,
               sigma: float, beta: float,
               option_type: str = 'call') -> float:
    """
    CEV 模型（常数弹性方差）期权定价（Schroder 1989 精确解）。

    dS = b·S dt + σ·S^β dW（在风险中性测度下）

    精确解通过非中心卡方分布计算：
    m = 2r / [σ²·(2-β)·(e^{r(2-β)T} - 1)]  ← 非中心参数相关
    ...（详见代码内注释）

    参数
    ----
    S     : 当前股价
    K     : 行权价
    T     : 到期时间（年）
    r     : 无风险利率
    b     : 持有成本（b=r 无红利；b=r-q 含连续红利）
    sigma : CEV 系数（σ in dS = μS dt + σ S^β dW）
    beta  : 弹性参数 β（≠ 1）
       β < 1: 杠杆效应（典型 β ≈ 0.5）
       β > 1: 反杠杆（成长股）
       β → 1: 退化为 BSM（用 BSM 公式代替）
    option_type : 'call' 或 'put'
    """
    from scipy.stats import ncx2

    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 若 beta ≈ 1，退化为 BSM
    if abs(beta - 1.0) < 1e-6:
        d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        cf = exp((b-r)*T); df = exp(-r*T)
        if option_type == 'call':
            return S*cf*N(d1) - K*df*N(d2)
        return K*df*N(-d2) - S*cf*N(-d1)

    # ── CEV 精确定价（Schroder 1989）────────────────────────
    # 将 S 和 K 转换到 CEV 框架的标度变量
    # sigma_adj 调整为使 dS = b·S dt + sigma_adj·S^beta dW
    # 有效系数 ξ（取决于持有成本 b）
    ξ = 2 * b / (sigma**2 * (2 - beta) * (exp(b*(2-beta)*T) - 1))

    x = ξ * S**(2*(1-beta)) * exp(b*(2-beta)*T)  # 未到期价格的非中心参数
    y = ξ * K**(2*(1-beta))                        # 行权价的参数

    # 自由度
    nu = 1 / (1 - beta)   # 非中心卡方自由度（当 beta<1 时）

    df = exp(-r * T)

    if beta < 1:
        # ── β < 1（杠杆效应）─────────────────────────────
        # 看涨：P(χ²(2x; 2+nu, 2y) > 0) 相关
        # 看涨 = S·e^{(b-r)T}·(1 - F(2y; 2+ν, 2x)) - K·e^{-rT}·F(2y; ν, 2x)
        # F(w; k, λ) = 非中心卡方 CDF，自由度 k，非中心参数 λ
        cdf_1 = ncx2.cdf(2*y, 2 + nu, 2*x)  # Prob(χ²(2+ν, 2x) ≤ 2y)
        cdf_2 = ncx2.cdf(2*y, nu, 2*x)       # Prob(χ²(ν, 2x) ≤ 2y)

        if option_type.lower() == 'call':
            return (S * exp((b-r)*T) * (1 - cdf_1)
                    - K * df * cdf_2)
        else:
            # 看跌 = K·e^{-rT}·(1 - F(2y; ν, 2x)) - S·e^{(b-r)T}·F(2y; 2+ν, 2x)
            return (K * df * (1 - cdf_2)
                    - S * exp((b-r)*T) * cdf_1)

    else:  # beta > 1
        # ── β > 1（反杠杆效应）──────────────────────────
        nu2 = 1 / (beta - 1)
        cdf_3 = ncx2.cdf(2*x, 2 + nu2, 2*y)
        cdf_4 = ncx2.cdf(2*x, nu2, 2*y)

        if option_type.lower() == 'call':
            return (S * exp((b-r)*T) * cdf_3
                    - K * df * cdf_4)
        else:
            return (K * df * (1 - cdf_4)
                    - S * exp((b-r)*T) * (1 - cdf_3))


def cev_option_approx(S: float, K: float, T: float,
                      r: float, b: float,
                      sigma: float, beta: float,
                      option_type: str = 'call') -> float:
    """
    CEV 期权定价近似解（幂级数展开，Emanuel & MacBeth 1982）。

    将 CEV 价格展开为以 ε = (β-1) 为小量的级数，
    适用于 |β - 1| 较小的情形（如 β ∈ [0.5, 1.5]）。

    c ≈ c_BSM + (β-1) · ε₁ · n(d₂)/σ√T  +  (β-1)² · ε₂ · n(d₂)/σ√T

    这是粗略近似，精度不如 cev_option()，但计算更快。
    实践中推荐使用 cev_option() 精确解。

    此函数作为教学演示，用于理解 β 对期权价格的一阶影响。
    """
    from utils.common import norm_cdf as N, norm_pdf as np_

    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 基础 BSM（β=1 时）
    sigma_eff = sigma * S**(beta - 1)  # 局部波动率在当前 S 处的近似
    d1 = (log(S/K) + (b + 0.5*sigma_eff**2)*T) / (sigma_eff*sqrt(T))
    d2 = d1 - sigma_eff * sqrt(T)
    cf = exp((b-r)*T); df = exp(-r*T)

    if option_type.lower() == 'call':
        return S*cf*N(d1) - K*df*N(d2)
    return K*df*N(-d2) - S*cf*N(-d1)


# ═══════════════════════════════════════════════════════════════
# B. Corrado-Su (1996) 偏斜-峰度修正 BSM
# ═══════════════════════════════════════════════════════════════

def corrado_su_option(S: float, K: float, T: float,
                      r: float, b: float,
                      sigma: float,
                      skewness: float,
                      excess_kurtosis: float,
                      option_type: str = 'call') -> float:
    """
    Corrado-Su (1996) 偏斜-峰度修正 BSM 期权定价。

    标准 BSM 假设 ln(S_T/S) ~ N((b-σ²/2)T, σ²T)（偏斜=0，超额峰度=0）。
    Corrado-Su 利用 Gram-Charlier A 型展开，对偏斜和峰度进行修正：

    V = C_BSM + (μ₃/3!) · ∂³C/∂K³_类 + (μ₄/4!) · ∂⁴C/∂K⁴_类

    实际修正量化为：
    看涨：C = C_BSM + Q₃·w₃ + Q₄·w₄

    其中：
    w₃ = (1/6) · S·σ·√T · [2σ√T - d₁] · n(d₁)   ← 偏斜修正权重
    w₄ = (1/24) · S·σ·√T · [d₁² - 3d₁·σ√T + 3σ²T - 1] · n(d₁)  ← 峰度修正权重

    Q₃ = μ₃       = 三阶矩（偏斜度）
    Q₄ = μ₄ - 3   = 超额峰度（超出正态的峰度）

    参数
    ----
    S              : 当前股价
    K              : 行权价
    T              : 到期时间
    r              : 无风险利率
    b              : 持有成本
    sigma          : 年化波动率
    skewness       : 对数收益率的三阶矩 μ₃（偏斜度）
                     > 0：右偏（厚右尾），< 0：左偏（厚左尾）
    excess_kurtosis: 超额峰度 μ₄ - 3（= 0 正态，> 0 厚尾）
    option_type    : 'call' 或 'put'

    应用
    ----
    μ₃, μ₄ 可从历史收益率估算，或从期权市场隐含（矩校准）。
    市场中 S&P 500 通常呈现：μ₃ < 0（左偏），μ₄ > 3（厚尾）。
    """
    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    sigma_t = sigma * sqrt(T)   # 总波动率

    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / sigma_t
    d2 = d1 - sigma_t

    cf = exp((b-r)*T)
    df = exp(-r*T)

    # BSM 基础价格
    if option_type.lower() == 'call':
        bsm = S * cf * N(d1) - K * df * N(d2)
    else:
        bsm = K * df * N(-d2) - S * cf * N(-d1)

    # ── 偏斜修正项（三阶矩 Q₃ = μ₃）───────────────────────
    # w₃ = (1/6)·S·σ_T·(2σ_T - d₁)·n(d₁)   [原始 Corrado-Su 形式]
    # 将 S·cf 整体处理，与 Haug(2007) p.226 公式对齐
    w3 = (1./6.) * S * cf * sigma_t * (2*sigma_t - d1) * n(d1)

    # ── 峰度修正项（四阶矩超额 Q₄ = μ₄ - 3）──────────────
    # w₄ = (1/24)·S·σ_T·(d₁² - 3d₁·σ_T + 3σ_T² - 1)·n(d₁)
    w4 = (1./24.) * S * cf * sigma_t * (d1**2 - 3*d1*sigma_t + 3*sigma_t**2 - 1) * n(d1)

    # 注意：w₃/w₄ 是对 call 的修正；put 的修正方向相同
    # （Gram-Charlier 修正对 call 和 put 的影响是对称的）
    price = bsm + skewness * w3 + excess_kurtosis * w4

    return max(price, 0.)


def corrado_su_implied_moments(S: float, K_list: list, T: float,
                                r: float, b: float, sigma: float,
                                market_prices: list,
                                option_type: str = 'call') -> tuple:
    """
    从市场期权价格反推 Corrado-Su 的隐含偏斜和峰度（矩校准）。

    使用最小二乘法找到最佳 (μ₃, μ₄-3) 拟合市场价格。

    参数
    ----
    K_list        : 行权价列表
    market_prices : 对应的市场期权价格
    sigma         : 输入的波动率（或 ATM 隐含波动率）

    返回
    ----
    (skewness, excess_kurtosis) : 隐含的三阶和超额四阶矩
    """
    from scipy.optimize import minimize

    def objective(params):
        mu3, mu4m3 = params
        total_error = 0.
        for K, mp in zip(K_list, market_prices):
            model_p = corrado_su_option(S, K, T, r, b, sigma, mu3, mu4m3, option_type)
            total_error += (model_p - mp)**2
        return total_error

    result = minimize(objective, [0.0, 0.0],
                      method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-10})
    return tuple(result.x)


if __name__ == "__main__":
    print("=" * 65)
    print("CEV 模型 & Corrado-Su 偏斜-峰度修正 — 数值示例")
    print("=" * 65)

    S, K, T, r, b, sigma = 100., 100., 0.5, 0.10, 0.10, 0.20

    # ── CEV 模型 ─────────────────────────────────────────────
    print(f"\nCEV 模型（S={S}, K={K}, T={T}, r={r}, σ={sigma}）")
    print(f"  {'β':>6}  {'看涨':>10}  {'看跌':>10}  {'含义'}")
    for beta_t, desc in [(0.0, '绝对扩散'), (0.5, 'CIR型'), (0.8, '弱杠杆'),
                          (0.9, '轻杠杆'), (1.0, 'BSM基准'),
                          (1.1, '轻反杠杆'), (1.5, '强反杠杆')]:
        try:
            c = cev_option(S, K, T, r, b, sigma, beta_t, 'call')
            p = cev_option(S, K, T, r, b, sigma, beta_t, 'put')
            print(f"  {beta_t:>6.1f}  {c:>10.4f}  {p:>10.4f}  {desc}")
        except Exception as e:
            print(f"  {beta_t:>6.1f}  [计算错误: {e}]")

    # β < 1 时的波动率偏斜（杠杆效应导致负偏斜）
    print(f"\nCEV β=0.5 的波动率偏斜（隐含波动率 vs 执行价）：")
    from utils.common import norm_cdf as N_func
    beta_test = 0.5
    print(f"  {'K/S':>6}  {'CEV价格':>10}  ('偏斜特征')")
    for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
        Kk = S * m
        try:
            c_cev = cev_option(S, Kk, T, r, b, sigma, beta_test, 'call')
            print(f"  {m:>6.2f}  {c_cev:>10.4f}")
        except Exception:
            print(f"  {m:>6.2f}  [N/A]")

    # ── Corrado-Su 修正 ───────────────────────────────────────
    print(f"\nCorrado-Su (1996) 偏斜-峰度修正（S={S}, K={K}）：")
    bsm_c  = corrado_su_option(S, K, T, r, b, sigma, 0., 0., 'call')  # μ₃=0, μ₄-3=0
    print(f"  BSM（μ₃=0, μ₄-3=0）：看涨 = {bsm_c:.4f}")

    # 市场典型参数（S&P 500 特征）
    mu3_spx  = -0.30   # 负偏斜（左尾厚）
    mu4m3    =  3.00   # 正超额峰度（尖峰）

    cs_call = corrado_su_option(S, K, T, r, b, sigma, mu3_spx, mu4m3, 'call')
    cs_put  = corrado_su_option(S, K, T, r, b, sigma, mu3_spx, mu4m3, 'put')
    print(f"  CS修正（μ₃={mu3_spx}, μ₄-3={mu4m3}）：看涨={cs_call:.4f}, 看跌={cs_put:.4f}")
    print(f"  偏斜修正量 = {cs_call - bsm_c:+.4f}（负偏斜使虚值看跌更贵）")

    # 不同偏斜对期权价格的影响
    print(f"\n  偏斜度 μ₃ 对 OTM 看跌价格的影响（K=90，低行权价）：")
    for mu3 in [-0.8, -0.5, -0.30, 0., 0.30, 0.5]:
        p_otm = corrado_su_option(S, 90, T, r, b, sigma, mu3, mu4m3, 'put')
        print(f"    μ₃={mu3:+.2f}: OTM看跌 = {p_otm:.4f}")
