"""
第13章：概率分布理论与期权定价
Chapter 13: Probability Distributions and Option Pricing

本文件实现：
1. 正态分布（Normal Distribution）及其参数估计
2. 对数正态分布（Log-Normal Distribution）—— BSM 的基础
3. 泊松分布（Poisson Distribution）—— 跳扩散的跳跃计数
4. 非中心卡方分布（Non-central Chi-Square）—— CEV 模型精确解
5. Gram-Charlier A 型展开（偏度/峰度修正）
6. Edgeworth 级数展开（高阶矩修正）
7. 稳定分布（Lévy / Alpha-Stable）—— 厚尾资产收益建模
8. 混合正态分布（Mixture of Normals）—— 波动率聚类建模
9. 广义双曲分布（GHD）—— 含偏度和厚尾
10. 矩计算工具（均值/方差/偏度/峰度）

参考：
  - Haug (2007) "The Complete Guide to Option Pricing Formulas" Ch.13
  - Johnson, Kotz & Balakrishnan "Continuous Univariate Distributions"
  - Fama (1965) "The Behavior of Stock Market Prices" JB
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from scipy.special import gamma, kv, factorial
from scipy.optimize import minimize, brentq
import warnings

from utils.common import norm_cdf, norm_pdf


# ============================================================
# 1. 正态分布（Normal / Gaussian Distribution）
# ============================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    正态分布概率密度函数

    f(x) = (1 / (σ√(2π))) · exp(-(x-μ)² / (2σ²))

    参数说明：
        x     : 随机变量值
        mu    : 均值
        sigma : 标准差

    返回：
        f(x) 在给定参数下的值
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    正态分布累积分布函数

    F(x) = Φ((x-μ)/σ)

    使用误差函数（erf）实现：
        Φ(z) = (1 + erf(z/√2)) / 2
    """
    z = (x - mu) / sigma
    return 0.5 * (1.0 + np.math.erf(z / np.sqrt(2)))


def normal_quantile(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    正态分布分位数（逆 CDF）

    Q(p) = μ + σ·Φ⁻¹(p)

    Beasley-Springer-Moro 近似（单精度）；高精度使用 scipy.stats.norm.ppf

    参数说明：
        p : 概率值 ∈ (0, 1)
    """
    from utils.common import norm_inv
    return mu + sigma * norm_inv(p)


def normal_fit_mle(data: np.ndarray) -> tuple[float, float]:
    """
    正态分布最大似然估计

    MLE 解析解（正态分布存在闭合 MLE）：
        μ̂ = x̄  （样本均值）
        σ̂ = √(Σ(xᵢ-x̄)²/n)  （样本标准差，最大似然用 n 非 n-1）

    参数说明：
        data : 样本数据数组

    返回：
        (mu_hat, sigma_hat)
    """
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.std(data, ddof=0))  # MLE 用 ddof=0
    return mu_hat, sigma_hat


def normal_moments(mu: float = 0.0, sigma: float = 1.0) -> dict:
    """
    正态分布各阶矩

    E[X]         = μ
    Var[X]       = σ²
    Skewness     = 0（对称）
    Kurtosis     = 3（超额峰度 = 0）
    E[X^n]       n 为偶数 → (n-1)!! · σⁿ（对标准正态）

    返回：
        dict 含均值、方差、偏度、峰度
    """
    return {
        'mean': mu,
        'variance': sigma ** 2,
        'std': sigma,
        'skewness': 0.0,
        'kurtosis': 3.0,        # 总峰度
        'excess_kurtosis': 0.0, # 超额峰度（Fisher 定义）
        'mgf_coeff': f'exp(μt + σ²t²/2)',  # 矩母函数
    }


# ============================================================
# 2. 对数正态分布（Log-Normal Distribution）
# ============================================================

def lognormal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    对数正态分布概率密度函数

    若 ln(X) ~ N(μ, σ²)，则 X ~ LogNormal(μ, σ²)

    f(x) = (1/(xσ√(2π))) · exp(-(ln(x)-μ)²/(2σ²))，x > 0

    在 BSM 框架中：
        S_T = S_0 · exp[(b - σ²/2)T + σ√T·Z]
        即 ln(S_T) ~ N(ln(S_0) + (b-σ²/2)T, σ²T)

    参数说明：
        x     : 随机变量值（必须 > 0）
        mu    : 对数均值（即 ln(X) 的均值）
        sigma : 对数标准差
    """
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.log(x) - mu) / sigma) ** 2)


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    对数正态分布累积分布函数

    F(x) = Φ((ln(x) - μ) / σ)

    BSM 公式中 N(d2) 即为 S_T > K 的风险中性概率（经过漂移调整）
    """
    if x <= 0:
        return 0.0
    return norm_cdf((np.log(x) - mu) / sigma)


def lognormal_moments(mu: float, sigma: float) -> dict:
    """
    对数正态分布的统计矩

    E[X]     = exp(μ + σ²/2)
    Var[X]   = (exp(σ²) - 1) · exp(2μ + σ²)
    Skewness = (exp(σ²)+2) · √(exp(σ²)-1)
    Kurt     = exp(4σ²) + 2exp(3σ²) + 3exp(2σ²) - 6

    这些矩关系在期权定价中极为重要：
    E[X] = F（远期价格）在风险中性测度下等于 S_0·e^{bT}

    返回：
        dict 含各阶矩
    """
    e_sigma2 = np.exp(sigma ** 2)
    mean = np.exp(mu + 0.5 * sigma ** 2)
    var = (e_sigma2 - 1) * np.exp(2 * mu + sigma ** 2)
    skew = (e_sigma2 + 2) * np.sqrt(e_sigma2 - 1)
    kurt = e_sigma2 ** 4 + 2 * e_sigma2 ** 3 + 3 * e_sigma2 ** 2 - 6

    return {
        'mean': mean,
        'variance': var,
        'std': np.sqrt(var),
        'skewness': skew,
        'kurtosis': kurt + 3,      # 总峰度
        'excess_kurtosis': kurt,   # 超额峰度
        'median': np.exp(mu),
        'mode': np.exp(mu - sigma ** 2),
    }


def lognormal_option_price(F: float, K: float, T: float,
                             r: float, sigma: float,
                             option_type: str = 'call') -> float:
    """
    对数正态分布下的欧式期权价格（Black-76 等价形式）

    直接从对数正态 CDF 推导出 Black-Scholes 公式：
        C = e^{-rT} · E_Q[max(S_T - K, 0)]
          = e^{-rT} · [E_Q[S_T · 1_{S_T>K}] - K · Q(S_T > K)]
          = e^{-rT} · [F·N(d1) - K·N(d2)]

    这揭示了 BSM 定价的概率论本质：
        Q(S_T > K) = N(d2)  风险中性概率
        E_Q[S_T/F | S_T>K] = N(d1)/N(d2)  截断期望

    参数说明：
        F  : 远期价格（S_0·e^{bT} 或直接输入远期）
        K  : 行权价
        T  : 到期时间
        r  : 无风险利率（折现用）
        sigma : 对数正态标准差（年化波动率）
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(F - K, 0.0)
        else:
            return max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = np.exp(-r * T)

    if option_type.lower() == 'call':
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


# ============================================================
# 3. 泊松分布（Poisson Distribution）
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    """
    泊松分布概率质量函数

    P(X = k) = (λᵏ · e^{-λ}) / k!

    在跳扩散模型（Merton 1976）中：
        跳跃次数 N(T) ~ Poisson(λT)
        P(N(T)=k) = (λT)ᵏ · e^{-λT} / k!

    参数说明：
        k   : 非负整数（跳跃次数）
        lam : 泊松强度（单位时间内的期望跳跃次数）
    """
    if k < 0 or not isinstance(k, (int, np.integer)):
        return 0.0
    return np.exp(-lam) * (lam ** k) / float(factorial(k, exact=True))


def poisson_moments(lam: float) -> dict:
    """
    泊松分布各阶矩

    E[X]     = λ
    Var[X]   = λ
    Skewness = 1/√λ
    Kurtosis = 3 + 1/λ  (总峰度)

    特点：均值等于方差（均值越大，分布越接近正态）

    返回：
        dict 含各阶矩
    """
    return {
        'mean': lam,
        'variance': lam,
        'std': np.sqrt(lam),
        'skewness': 1.0 / np.sqrt(lam) if lam > 0 else np.inf,
        'excess_kurtosis': 1.0 / lam if lam > 0 else np.inf,
    }


def poisson_cdf(k: int, lam: float) -> float:
    """
    泊松分布累积分布函数

    P(X ≤ k) = Σᵢ₌₀ᵏ (λⁱ·e^{-λ}/i!)

    用于跳扩散模型中截断级数求和（实践中取 k_max=20 足够精确）
    """
    return float(stats.poisson.cdf(k, lam))


# ============================================================
# 4. 非中心卡方分布（Non-central Chi-Square）
# ============================================================

def noncentralchisq_pdf(x: float, df: float, nc: float) -> float:
    """
    非中心卡方分布概率密度函数

    若 Z₁,...,Zₖ i.i.d. N(μᵢ, 1)，则 Σ Zᵢ² ~ χ²(k, λ=Σμᵢ²)

    f(x; k, λ) = Σₙ₌₀^∞ [e^{-λ/2}(λ/2)ⁿ/n!] · f_χ²(k+2n)(x)

    在 CEV 模型中，股价的转换分布恰好是非中心卡方

    参数说明：
        x  : 随机变量值（≥ 0）
        df : 自由度（degrees of freedom）
        nc : 非中心参数 λ（non-centrality parameter，≥ 0）
    """
    if x < 0:
        return 0.0
    return float(stats.ncx2.pdf(x, df, nc))


def noncentralchisq_cdf(x: float, df: float, nc: float) -> float:
    """
    非中心卡方分布累积分布函数

    P(X ≤ x; k, λ) = Σₙ₌₀^∞ [e^{-λ/2}(λ/2)ⁿ/n!] · F_χ²(k+2n)(x)

    CEV 模型（β < 1）中：
        df = 2/(1-β)² (当 β < 1)
        nc = 2κS_{t}^{(1-β)²/2} / (σ²(1-β)²(e^{κT}-1))  (漂移修正后)

    通过 scipy.stats.ncx2.cdf 实现，精度极高
    """
    if x < 0:
        return 0.0
    return float(stats.ncx2.cdf(x, df, nc))


def noncentralchisq_quantile(p: float, df: float, nc: float) -> float:
    """
    非中心卡方分布分位数

    用于 CEV 模型的精确期权定价中的 CDF 反转
    """
    return float(stats.ncx2.ppf(p, df, nc))


def noncentralchisq_moments(df: float, nc: float) -> dict:
    """
    非中心卡方分布各阶矩

    E[X]     = k + λ
    Var[X]   = 2(k + 2λ)
    Skewness = 2√2 · (k + 3λ) / (k + 2λ)^{3/2}
    Kurt     = 12(k + 4λ) / (k + 2λ)² + 3

    返回：
        dict 含各阶矩（k=df, λ=nc）
    """
    k, lam = df, nc
    mean = k + lam
    var = 2 * (k + 2 * lam)
    skew = 2 * np.sqrt(2) * (k + 3 * lam) / (k + 2 * lam) ** 1.5
    kurt = 12 * (k + 4 * lam) / (k + 2 * lam) ** 2 + 3

    return {
        'mean': mean,
        'variance': var,
        'std': np.sqrt(var),
        'skewness': skew,
        'kurtosis': kurt,
        'excess_kurtosis': kurt - 3,
    }


# ============================================================
# 5. Gram-Charlier A 型展开
# ============================================================

def gram_charlier_pdf(x: float, mu: float, sigma: float,
                       skew: float, excess_kurt: float) -> float:
    """
    Gram-Charlier A 型概率密度函数展开

    GC-A 展开在标准正态 φ(z) 的基础上，用 Hermite 多项式修正：
        f(x) ≈ φ(z)/σ · [1 + (κ₃/6)·H₃(z) + (κ₄/24)·H₄(z)]

    其中：
        z    = (x - μ) / σ
        κ₃   = 偏度（skewness）
        κ₄   = 超额峰度（excess kurtosis）
        H₃(z) = z³ - 3z  （三阶 Hermite 多项式）
        H₄(z) = z⁴ - 6z² + 3  （四阶 Hermite 多项式）

    GC-A 展开的局限：
        可能在尾部产生负值（需额外约束 κ₃² ≤ κ₄+2 等）

    Corrado-Su (1996) 期权定价即基于此展开

    参数说明：
        mu, sigma     : 均值和标准差
        skew          : 偏度系数 κ₃
        excess_kurt   : 超额峰度系数 κ₄
    """
    z = (x - mu) / sigma
    # Hermite 多项式（概率论型）
    H3 = z ** 3 - 3 * z
    H4 = z ** 4 - 6 * z ** 2 + 3
    phi_z = norm_pdf(z)
    correction = 1 + (skew / 6) * H3 + (excess_kurt / 24) * H4
    return max(phi_z * correction / sigma, 0.0)


def gram_charlier_cdf(x: float, mu: float, sigma: float,
                       skew: float, excess_kurt: float) -> float:
    """
    Gram-Charlier A 型累积分布函数

    F(x) ≈ Φ(z) - φ(z)·[(κ₃/6)·H₂(z) + (κ₄/24)·H₃(z)]

    其中：
        H₂(z) = z² - 1（对 H₃ 积分 → -(z²-1)φ(z)）
        H₃(z) = z³ - 3z（对 H₄ 积分 → -(z³-3z)φ(z)）

    参数说明：同 gram_charlier_pdf
    """
    z = (x - mu) / sigma
    phi_z = norm_pdf(z)
    Phi_z = norm_cdf(z)
    # 积分后的 Hermite 修正
    H2_corr = z ** 2 - 1       # ∫H₃dz / d(z)
    H3_corr = z ** 3 - 3 * z   # ∫H₄dz / d(z)
    result = Phi_z - phi_z * ((skew / 6) * H2_corr + (excess_kurt / 24) * H3_corr)
    return float(np.clip(result, 0.0, 1.0))


def gram_charlier_option_price(S: float, K: float, T: float,
                                r: float, b: float, sigma: float,
                                skew: float, excess_kurt: float,
                                option_type: str = 'call') -> float:
    """
    Corrado-Su (1996) Gram-Charlier 修正期权定价

    在 BSM 的基础上加入偏度/峰度修正项：
        C = C_BSM + (κ₃/6)·w₃ + (κ₄/24)·w₄

    其中：
        w₃ = S·e^{(b-r)T}·σ_T·(2σ_T - d₁)·φ(d₁)
             （偏度修正：跳跃或厚尾引起的波动率微笑）
        w₄ = S·e^{(b-r)T}·σ_T·(d₁²-3d₁σ_T+3σ_T²-1)·φ(d₁)
             （峰度修正）

    参数说明：
        S, K    : 现价、行权价
        T, r, b : 到期时间、无风险利率、持有成本
        sigma   : 对数正态波动率
        skew    : 超额偏度（与正态的差异，κ₃-0=κ₃）
        excess_kurt : 超额峰度（κ₄-0=κ₄，正态为0）
    """
    sqrt_T = np.sqrt(T)
    sigma_T = sigma * sqrt_T
    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / sigma_T
    d2 = d1 - sigma_T
    cf = np.exp((b - r) * T)
    df = np.exp(-r * T)

    # 基础 BSM 价格
    if option_type.lower() == 'call':
        bsm = S * cf * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
        bsm = K * df * norm_cdf(-d2) - S * cf * norm_cdf(-d1)

    # Gram-Charlier 修正项
    phi_d1 = norm_pdf(d1)
    w3 = S * cf * sigma_T * (2 * sigma_T - d1) * phi_d1
    w4 = S * cf * sigma_T * (d1 ** 2 - 3 * d1 * sigma_T + 3 * sigma_T ** 2 - 1) * phi_d1

    # 看跌期权的修正符号相同（从 put-call parity 导出）
    correction = (skew / 6) * w3 + (excess_kurt / 24) * w4
    if option_type.lower() == 'put':
        correction = -correction  # put 修正符号相反

    return max(bsm + correction, 0.0)


# ============================================================
# 6. Edgeworth 级数展开
# ============================================================

def edgeworth_pdf(x: float, mu: float, sigma: float,
                   kappa3: float, kappa4: float,
                   kappa5: float = 0.0, kappa6: float = 0.0) -> float:
    """
    Edgeworth 级数展开概率密度函数（四至六阶修正）

    Edgeworth 展开相比 Gram-Charlier 有更好的渐近性质：
        f(x) ≈ φ(z)/σ · [1
              + (κ₃/6)·He₃(z)
              + (κ₄/24)·He₄(z) + (κ₃²/72)·He₆(z)
              + (κ₅/120)·He₅(z) + (κ₃κ₄/144)·He₇(z) + (κ₃³/1296)·He₉(z)]

    Hermite 多项式（概率论型 He_n）：
        He₃ = z³ - 3z
        He₄ = z⁴ - 6z² + 3
        He₅ = z⁵ - 10z³ + 15z
        He₆ = z⁶ - 15z⁴ + 45z² - 15
        He₇ = z⁷ - 21z⁵ + 105z³ - 105z
        He₉ = z⁹ - 36z⁷ + 378z⁵ - 1260z³ + 945z

    参数说明：
        kappa3 : 三阶累积量（=偏度·σ³）
        kappa4 : 四阶累积量（=超额峰度·σ⁴）
        kappa5 : 五阶累积量（通常设为0）
        kappa6 : 六阶累积量（通常设为0）
    """
    z = (x - mu) / sigma

    # Hermite 多项式
    He3 = z ** 3 - 3 * z
    He4 = z ** 4 - 6 * z ** 2 + 3
    He5 = z ** 5 - 10 * z ** 3 + 15 * z
    He6 = z ** 6 - 15 * z ** 4 + 45 * z ** 2 - 15
    He7 = z ** 7 - 21 * z ** 5 + 105 * z ** 3 - 105 * z
    He9 = z ** 9 - 36 * z ** 7 + 378 * z ** 5 - 1260 * z ** 3 + 945 * z

    # 转换为标准化累积量
    lam3 = kappa3 / sigma ** 3  # 偏度
    lam4 = kappa4 / sigma ** 4  # 超额峰度
    lam5 = kappa5 / sigma ** 5
    lam6 = kappa6 / sigma ** 6

    phi_z = norm_pdf(z)
    correction = (1
                  + (lam3 / 6) * He3
                  + (lam4 / 24) * He4 + (lam3 ** 2 / 72) * He6
                  + (lam5 / 120) * He5
                  + (lam3 * lam4 / 144) * He7
                  + (lam3 ** 3 / 1296) * He9)

    return max(phi_z * correction / sigma, 0.0)


def edgeworth_cdf(x: float, mu: float, sigma: float,
                   kappa3: float, kappa4: float) -> float:
    """
    Edgeworth 级数展开累积分布函数（二阶截断）

    F(x) ≈ Φ(z) - φ(z) · [(κ₃/6)·He₂(z) + (κ₄/24)·He₃(z) + (κ₃²/72)·He₅(z)]

    其中 He₂(z) = z²-1，He₅(z) = z⁵-10z³+15z

    参数说明：
        kappa3 : 偏度 × σ³
        kappa4 : 超额峰度 × σ⁴
    """
    z = (x - mu) / sigma
    lam3 = kappa3 / sigma ** 3
    lam4 = kappa4 / sigma ** 4

    phi_z = norm_pdf(z)
    Phi_z = norm_cdf(z)

    He2 = z ** 2 - 1
    He3 = z ** 3 - 3 * z
    He5 = z ** 5 - 10 * z ** 3 + 15 * z

    correction = phi_z * ((lam3 / 6) * He2 + (lam4 / 24) * He3 + (lam3 ** 2 / 72) * He5)
    result = Phi_z - correction
    return float(np.clip(result, 0.0, 1.0))


# ============================================================
# 7. 稳定分布（Alpha-Stable / Lévy Distribution）
# ============================================================

def stable_distribution_params(alpha: float, beta: float,
                                 gamma: float, delta: float) -> dict:
    """
    Alpha-Stable 分布参数概述

    稳定分布 S(α, β, γ, δ) 由四个参数描述：
        α ∈ (0, 2] : 稳定性指数（越小尾部越厚，α=2 退化为正态）
        β ∈ [-1, 1]: 偏度参数（β=0 对称）
        γ > 0      : 尺度参数（类比标准差，但方差在 α<2 时无穷大）
        δ ∈ (-∞,∞) : 位置参数（类比均值，但 α≤1 时均值无穷大）

    特殊情形：
        S(2, 0, γ, δ) = N(δ, 2γ²)  正态分布
        S(1, 0, γ, δ) = Cauchy(δ, γ)  柯西分布
        S(0.5, 1, γ, δ) = Lévy(δ, γ)  Lévy 分布

    金融含义：
        Fama (1965) 发现股票收益的特征指数 α ≈ 1.7（非正态）
        α < 2 意味着方差无穷大（不适合直接用 BS 公式）

    参数说明：
        alpha, beta, gamma, delta : 稳定分布四参数

    返回：
        参数摘要 dict（含存在性条件）
    """
    result = {
        'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta,
        'mean_exists': alpha > 1,
        'variance_exists': alpha >= 2,
        'all_moments_exist': alpha >= 2,
    }
    if alpha > 1:
        result['mean'] = delta
    if alpha >= 2:
        result['variance'] = 2 * gamma ** 2
    return result


def levy_pdf(x: float, mu: float = 0.0, c: float = 1.0) -> float:
    """
    Lévy 分布概率密度函数（α=0.5, β=1 的稳定分布）

    f(x; μ, c) = √(c/(2π)) · exp(-c/(2(x-μ))) / (x-μ)^{3/2}，x > μ

    Lévy 分布是单边稳定分布（只有右尾），
    在金融中可用于建模首次穿越时间等

    参数说明：
        mu : 位置参数（下界）
        c  : 尺度参数
    """
    if x <= mu:
        return 0.0
    y = x - mu
    return np.sqrt(c / (2 * np.pi)) * np.exp(-c / (2 * y)) / y ** 1.5


def stable_characteristic_function(t: float, alpha: float, beta: float,
                                     gamma: float, delta: float) -> complex:
    """
    Alpha-Stable 分布的特征函数（S1 参数化）

    ln φ(t) = iδt - γ|t|^α · (1 - iβ·sign(t)·tan(πα/2))，α ≠ 1
    ln φ(t) = iδt - γ|t| · (1 + iβ·(2/π)·sign(t)·ln|t|)，α = 1

    特征函数是稳定分布的定义基础，因为稳定分布没有解析 PDF（除少数特例）

    参数说明：
        t : 特征函数的参数

    返回：
        复数特征函数值
    """
    if abs(t) < 1e-12:
        return complex(1.0, 0.0)

    sign_t = np.sign(t)
    abs_t = abs(t)

    if abs(alpha - 1.0) > 1e-6:
        tan_term = beta * np.tan(np.pi * alpha / 2)
        log_cf = (1j * delta * t
                  - gamma * abs_t ** alpha * (1 - 1j * sign_t * tan_term))
    else:
        log_term = (2 / np.pi) * beta * sign_t * np.log(abs_t)
        log_cf = (1j * delta * t
                  - gamma * abs_t * (1 + 1j * log_term))

    return np.exp(log_cf)


def stable_pdf_numerical(x: float, alpha: float, beta: float,
                          gamma: float, delta: float,
                          n_points: int = 2000) -> float:
    """
    数值积分计算 Alpha-Stable 分布 PDF

    通过特征函数的 Fourier 逆变换：
        f(x) = (1/2π) ∫ φ(t) · e^{-itx} dt

    实现：使用梯形法数值积分（t ∈ [-T_max, T_max]）

    参数说明：
        n_points : 积分点数（越多越精确，但耗时）

    返回：
        f(x) 的数值近似值
    """
    T_max = 50.0 / max(gamma, 1e-6)
    t_vals = np.linspace(-T_max, T_max, n_points)
    dt = t_vals[1] - t_vals[0]

    integrand = np.array([
        stable_characteristic_function(t, alpha, beta, gamma, delta) * np.exp(-1j * t * x)
        for t in t_vals
    ])
    pdf_val = np.real(np.trapz(integrand, t_vals)) / (2 * np.pi)
    return max(pdf_val, 0.0)


# ============================================================
# 8. 混合正态分布（Mixture of Normals）
# ============================================================

def mixture_normal_pdf(x: float, weights: list, mus: list, sigmas: list) -> float:
    """
    混合正态分布概率密度函数

    f(x) = Σᵢ wᵢ · N(x; μᵢ, σᵢ²)

    混合正态分布（Gaussian Mixture Model）能灵活地捕捉：
    1. 多峰分布（bi-modal）
    2. 厚尾（fat tails）—— 高方差分量贡献
    3. 偏度（skewness）—— 非对称权重
    4. 波动率聚类 —— 时变混合权重

    在期权定价中：
        条件期望 E[S_T | regime] 可得到半解析期权价格
        C = Σᵢ wᵢ · BSM(S, K, T, r, σᵢ)

    参数说明：
        weights : 混合权重列表（Σwᵢ = 1）
        mus     : 各分量均值
        sigmas  : 各分量标准差
    """
    w = np.array(weights)
    w = w / w.sum()
    total = 0.0
    for wt, mu_i, sig_i in zip(w, mus, sigmas):
        total += wt * normal_pdf(x, mu_i, sig_i)
    return total


def mixture_normal_option_price(S: float, K: float, T: float,
                                  r: float,
                                  weights: list, mus: list, sigmas: list,
                                  option_type: str = 'call') -> float:
    """
    混合正态分布下的欧式期权定价

    当底层对数收益率服从混合正态分布时：
        C = Σᵢ wᵢ · BSM(S, K, T, r, σᵢ, bᵢ)

    其中每个分量对应不同的"状态"（低波动/高波动/跳跃）

    这是 Merton 跳扩散模型的连续时间版本，
    也与 Regime-Switching 模型等价（在积分后）

    参数说明：
        weights : 各正态分量在收益率上的权重
        mus     : 各分量对数漂移（年化）
        sigmas  : 各分量对数波动率（年化）

    返回：
        混合分布下的期权价格
    """
    w = np.array(weights)
    w = w / w.sum()

    total_price = 0.0
    for wt, mu_i, sigma_i in zip(w, mus, sigmas):
        # 每个分量用对应参数的 Black 公式
        b_i = mu_i + r  # 持有成本 = 漂移 + 无风险利率
        F_i = S * np.exp(b_i * T)
        price_i = lognormal_option_price(F_i, K, T, r, sigma_i, option_type)
        total_price += wt * price_i

    return total_price


def mixture_normal_fit_em(data: np.ndarray,
                           n_components: int = 2,
                           max_iter: int = 100,
                           tol: float = 1e-6) -> dict:
    """
    混合正态分布 EM 算法拟合

    E 步（期望步）：计算后验责任度
        γᵢₙ = wᵢ·N(xₙ; μᵢ, σᵢ²) / Σⱼ wⱼ·N(xₙ; μⱼ, σⱼ²)

    M 步（最大化步）：更新参数
        wᵢ   = (1/N) Σₙ γᵢₙ
        μᵢ   = Σₙ γᵢₙ xₙ / Σₙ γᵢₙ
        σᵢ²  = Σₙ γᵢₙ (xₙ-μᵢ)² / Σₙ γᵢₙ

    参数说明：
        data         : 样本数据（日收益率等）
        n_components : 混合分量数
        max_iter     : 最大迭代次数
        tol          : 收敛容忍度（对数似然变化）

    返回：
        dict 含 weights, mus, sigmas, log_likelihood
    """
    n = len(data)
    # 初始化：用 K-means 风格的随机分配
    np.random.seed(42)
    idx = np.random.choice(n, n_components, replace=False)
    mus = data[idx].copy()
    sigmas = np.full(n_components, np.std(data))
    weights = np.ones(n_components) / n_components

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E 步
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * stats.norm.pdf(data, mus[k], sigmas[k])
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-300)
        resp /= resp_sum

        # M 步
        Nk = resp.sum(axis=0)
        for k in range(n_components):
            if Nk[k] > 1e-8:
                weights[k] = Nk[k] / n
                mus[k] = np.dot(resp[:, k], data) / Nk[k]
                sigmas[k] = np.sqrt(np.dot(resp[:, k], (data - mus[k]) ** 2) / Nk[k])
                sigmas[k] = max(sigmas[k], 1e-6)

        # 对数似然
        ll = 0.0
        for k in range(n_components):
            ll += weights[k] * stats.norm.pdf(data, mus[k], sigmas[k])
        log_ll = float(np.sum(np.log(np.maximum(ll, 1e-300))))

        if abs(log_ll - prev_ll) < tol:
            break
        prev_ll = log_ll

    return {
        'weights': weights.tolist(),
        'mus': mus.tolist(),
        'sigmas': sigmas.tolist(),
        'log_likelihood': prev_ll,
        'n_iterations': iteration + 1,
    }


# ============================================================
# 9. 矩计算与分布诊断工具
# ============================================================

def sample_moments(data: np.ndarray) -> dict:
    """
    计算样本的前四阶矩（均值/方差/偏度/峰度）

    使用无偏估计量（Bessel 修正 n-1）：
        μ̂₁ = x̄ = (1/n)Σxᵢ
        μ̂₂ = s² = Σ(xᵢ-x̄)²/(n-1)  （方差，无偏）
        μ̂₃ = [n/((n-1)(n-2))] Σ((xᵢ-x̄)/s)³  （Fisher 偏度，无偏）
        μ̂₄ = [(n(n+1))/((n-1)(n-2)(n-3))]·Σ((xᵢ-x̄)/s)⁴
              - 3(n-1)²/((n-2)(n-3))  （Fisher-Pearson 超额峰度，无偏）

    Jarque-Bera 正态性检验：
        JB = n/6 · [S² + (E-3)²/4]  ~ χ²(2) 渐近

    参数说明：
        data : 数据数组

    返回：
        dict 含各统计量和 JB 检验
    """
    n = len(data)
    mu = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    z = (data - mu) / max(s, 1e-12)

    skew = float(np.mean(z ** 3) * n * (n - 1) / max((n - 1) * (n - 2), 1)) \
        if n > 2 else 0.0
    # 用 scipy 的无偏估计
    skew_unbiased = float(stats.skew(data, bias=False))
    kurt_excess = float(stats.kurtosis(data, bias=False))  # Fisher 超额峰度

    # Jarque-Bera 检验统计量
    jb_stat = n / 6 * (skew_unbiased ** 2 + kurt_excess ** 2 / 4)
    jb_pvalue = float(1 - stats.chi2.cdf(jb_stat, df=2))

    return {
        'n': n,
        'mean': mu,
        'std': s,
        'variance': s ** 2,
        'skewness': skew_unbiased,
        'excess_kurtosis': kurt_excess,
        'kurtosis': kurt_excess + 3.0,
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'is_normal_005': bool(jb_pvalue > 0.05),
    }


def moments_to_lognormal_params(mean: float, variance: float) -> tuple[float, float]:
    """
    从均值和方差反推对数正态参数 (μ, σ)

    由对数正态矩：
        E[X]   = exp(μ + σ²/2) = m
        Var[X] = (exp(σ²)-1)·m² = v

    求解：
        σ²  = ln(1 + v/m²)
        μ   = ln(m) - σ²/2

    用于将市场估计（均值/方差）转换为 BSM 模型参数

    参数说明：
        mean, variance : 目标均值和方差

    返回：
        (mu, sigma)
    """
    sigma_sq = np.log(1 + variance / mean ** 2)
    mu = np.log(mean) - 0.5 * sigma_sq
    return float(mu), float(np.sqrt(sigma_sq))


def cumulants_to_moments(kappa1: float, kappa2: float,
                           kappa3: float, kappa4: float) -> dict:
    """
    累积量（Cumulant）转换为中心矩

    累积量生成函数 K(t) = ln M(t)，
    κₙ 为 K(t) 关于 t 在 0 处的 n 阶导数

    转换关系：
        μ₁ = κ₁（均值）
        μ₂ = κ₂（方差）
        μ₃ = κ₃（三阶中心矩 = 偏度·σ³）
        μ₄ = κ₄ + 3κ₂²（四阶中心矩 = (超额峰度+3)·σ⁴）

    对正态分布：κₙ = 0 for n ≥ 3

    参数说明：
        kappa1,..,kappa4 : 一到四阶累积量

    返回：
        dict 含各阶矩和标准化统计量
    """
    mu1 = kappa1
    mu2 = kappa2   # 方差
    mu3 = kappa3   # 三阶中心矩
    mu4 = kappa4 + 3 * kappa2 ** 2  # 四阶中心矩

    sigma = np.sqrt(max(mu2, 0))
    skew = mu3 / sigma ** 3 if sigma > 0 else 0.0
    kurt_excess = kappa4 / sigma ** 4 if sigma > 0 else 0.0

    return {
        'mean': mu1,
        'variance': mu2,
        'std': sigma,
        'skewness': skew,
        'excess_kurtosis': kurt_excess,
        'central_moment_3': mu3,
        'central_moment_4': mu4,
    }


# ============================================================
# 示例与测试
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("第13章（一）：概率分布基础")
    print("=" * 60)

    # ── 1. 正态分布 ─────────────────────────────────────────────────
    print("\n── 正态分布 ──")
    x_vals = [-2, -1, 0, 1, 2]
    print("z    PDF(0,1)    CDF(0,1)")
    for x in x_vals:
        print(f"{x:+.0f}  {normal_pdf(x):.6f}  {normal_cdf(x):.6f}")
    mom = normal_moments(0, 1)
    print(f"矩：均值={mom['mean']}, 方差={mom['variance']}, 偏度={mom['skewness']}, 峰度={mom['kurtosis']}")

    # ── 2. 对数正态分布 ─────────────────────────────────────────────
    print("\n── 对数正态分布（股价建模）──")
    S0, sigma_ln, T_ln = 100.0, 0.20, 1.0
    r_ln = 0.05
    mu_ln = np.log(S0) + (r_ln - 0.5 * sigma_ln ** 2) * T_ln
    sigma_T = sigma_ln * np.sqrt(T_ln)
    lnm = lognormal_moments(mu_ln, sigma_T)
    print(f"S_0=100, σ=20%, T=1 → E[S_T]={lnm['mean']:.4f}（应≈{S0*np.exp(r_ln*T_ln):.4f}）")

    # BSM 看涨期权验证
    call_price = lognormal_option_price(S0 * np.exp(r_ln * T_ln), 100.0, T_ln, r_ln, sigma_ln)
    print(f"ATM 看涨（Black-76）：{call_price:.4f}")

    # ── 3. 泊松分布 ─────────────────────────────────────────────────
    print("\n── 泊松分布（跳跃模型）──")
    lam_jump = 2.0
    print(f"λ={lam_jump}，P(N=k)：")
    for k in range(6):
        print(f"  P(N={k}) = {poisson_pmf(k, lam_jump):.6f}")
    pm = poisson_moments(lam_jump)
    print(f"矩：均值={pm['mean']}, 方差={pm['variance']}, 偏度={pm['skewness']:.4f}")

    # ── 4. 非中心卡方分布 ───────────────────────────────────────────
    print("\n── 非中心卡方分布（CEV 模型）──")
    df_ncx, nc_ncx = 4.0, 5.0
    x_ncx = 8.0
    print(f"χ²({df_ncx:.0f}, nc={nc_ncx:.0f}) at x={x_ncx}:")
    print(f"  PDF = {noncentralchisq_pdf(x_ncx, df_ncx, nc_ncx):.6f}")
    print(f"  CDF = {noncentralchisq_cdf(x_ncx, df_ncx, nc_ncx):.6f}")
    ncx_m = noncentralchisq_moments(df_ncx, nc_ncx)
    print(f"  均值={ncx_m['mean']:.2f}, 方差={ncx_m['variance']:.2f}")

    # ── 5. Gram-Charlier 展开 ────────────────────────────────────────
    print("\n── Gram-Charlier A 型展开 ──")
    skew_gc, kurt_gc = -0.5, 2.0  # 负偏度（看跌偏斜）、正超额峰度
    S_gc, K_gc = 100.0, 100.0
    T_gc, r_gc, b_gc = 0.5, 0.05, 0.05
    sigma_gc = 0.20

    price_bsm = lognormal_option_price(S_gc * np.exp(b_gc * T_gc), K_gc, T_gc, r_gc, sigma_gc)
    price_gc = gram_charlier_option_price(S_gc, K_gc, T_gc, r_gc, b_gc, sigma_gc, skew_gc, kurt_gc)
    print(f"BSM 看涨：{price_bsm:.4f}")
    print(f"GC 修正（偏度={skew_gc}, 超额峰度={kurt_gc}）：{price_gc:.4f}")
    print(f"GC 修正幅度：{price_gc - price_bsm:+.4f}")

    # ── 6. Edgeworth 展开 ────────────────────────────────────────────
    print("\n── Edgeworth 级数展开 ──")
    sigma_ew = 1.0
    kappa3_ew = -0.5 * sigma_ew ** 3  # 偏度 × σ³
    kappa4_ew = 2.0 * sigma_ew ** 4   # 超额峰度 × σ⁴
    x_ew = 0.0
    pdf_norm = normal_pdf(x_ew, 0, sigma_ew)
    pdf_ew = edgeworth_pdf(x_ew, 0.0, sigma_ew, kappa3_ew, kappa4_ew)
    cdf_ew = edgeworth_cdf(x_ew, 0.0, sigma_ew, kappa3_ew, kappa4_ew)
    print(f"x=0：正态 PDF={pdf_norm:.6f}，Edgeworth PDF={pdf_ew:.6f}")
    print(f"x=0：Edgeworth CDF={cdf_ew:.6f}（正态=0.5000）")

    # ── 7. 混合正态 ─────────────────────────────────────────────────
    print("\n── 混合正态分布 ──")
    weights_mn = [0.80, 0.20]
    mus_mn = [0.0005, -0.01]       # 低波动均值 / 高波动均值（跳跃）
    sigmas_mn = [0.01, 0.04]       # 低波动 / 高波动

    # 生成混合数据
    n_data = 500
    mask = np.random.random(n_data) < 0.80
    data_mn = np.where(mask,
                        np.random.normal(mus_mn[0], sigmas_mn[0], n_data),
                        np.random.normal(mus_mn[1], sigmas_mn[1], n_data))

    em_result = mixture_normal_fit_em(data_mn, n_components=2)
    print(f"EM 拟合（{em_result['n_iterations']}次迭代）：")
    for i in range(2):
        print(f"  分量{i+1}: w={em_result['weights'][i]:.3f}, "
              f"μ={em_result['mus'][i]:.5f}, σ={em_result['sigmas'][i]:.5f}")

    # ── 8. 样本矩诊断 ────────────────────────────────────────────────
    print("\n── 样本矩诊断（日收益率）──")
    daily_returns = np.random.standard_t(df=5, size=1000) * 0.01
    sm = sample_moments(daily_returns)
    print(f"样本量: {sm['n']}, 均值: {sm['mean']:.6f}")
    print(f"标准差: {sm['std']:.6f}, 偏度: {sm['skewness']:.4f}, 超额峰度: {sm['excess_kurtosis']:.4f}")
    print(f"Jarque-Bera: stat={sm['jarque_bera_stat']:.2f}, p值={sm['jarque_bera_pvalue']:.6f}")
    print(f"正态分布（5%）: {sm['is_normal_005']}")

    # ── 9. 累积量转换 ────────────────────────────────────────────────
    print("\n── 累积量 → 矩 转换 ──")
    κ1, κ2, κ3, κ4 = 0.0005, 0.0001, -1e-6, 3e-8
    moments_dict = cumulants_to_moments(κ1, κ2, κ3, κ4)
    print(f"κ₁={κ1}, κ₂={κ2}, κ₃={κ3}, κ₄={κ4}")
    print(f"偏度: {moments_dict['skewness']:.4f}, 超额峰度: {moments_dict['excess_kurtosis']:.4f}")

    # ── 10. 对数正态参数反推 ─────────────────────────────────────────
    print("\n── 从均值/方差反推对数正态参数 ──")
    target_mean, target_var = 105.0, 400.0  # E[S_T]=105, Var[S_T]=400
    mu_ln2, sigma_ln2 = moments_to_lognormal_params(target_mean, target_var)
    print(f"E[S]={target_mean}, Var[S]={target_var}")
    print(f"→ 对数均值 μ={mu_ln2:.6f}, 对数标准差 σ={sigma_ln2:.6f}")
    check_m = lognormal_moments(mu_ln2, sigma_ln2)
    print(f"验证：E[X]={check_m['mean']:.4f}（应={target_mean}）, "
          f"Var[X]={check_m['variance']:.4f}（应={target_var}）")
