"""
第13章（二）：资产收益率分布与风险度量
Chapter 13 (Part II): Return Distributions and Risk Measures

本文件实现：
1. 学生 t 分布（厚尾收益建模）
2. 广义误差分布（GED / Generalized Error Distribution）
3. 偏斜正态分布（Skew-Normal）
4. 偏斜 t 分布（Hansen 1994 / Azzalini-Capitanio）
5. Johnson SU 分布（四矩灵活拟合）
6. Cornish-Fisher 展开（VaR/ES 修正）
7. 参数化 VaR 与 Expected Shortfall（ES / CVaR）
8. 历史模拟法 VaR 与 ES
9. 极值理论（GEV / GPD 分布）
10. 分布拟合优度检验（KS 检验、Anderson-Darling、QQ 图数据）
11. 收益率分布的期权隐含矩提取

参考：
  - Haug (2007) "The Complete Guide to Option Pricing Formulas" Ch.13
  - McNeil, Frey, Embrechts "Quantitative Risk Management"
  - Hansen (1994) "Autoregressive Conditional Density Estimation"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func, beta as beta_func
from scipy.optimize import minimize, brentq
import warnings

from utils.common import norm_cdf, norm_pdf


# ============================================================
# 1. 学生 t 分布（Student's t-Distribution）
# ============================================================

def student_t_pdf(x: float, nu: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    学生 t 分布概率密度函数（位置-尺度参数化）

    f(x; ν, μ, σ) = Γ((ν+1)/2) / [Γ(ν/2)·√(νπ)·σ] · (1 + (x-μ)²/(νσ²))^{-(ν+1)/2}

    金融含义：
        收益率分布的尾部比正态分布厚（ν < 30 时显著）
        ν → ∞ 时退化为正态分布
        ν = 1 时为柯西分布（均值不存在）
        ν = 4 时超额峰度 = 2（典型日收益率 ν ≈ 4-8）

    参数说明：
        nu    : 自由度（ν > 2 时方差存在，ν > 4 时峰度存在）
        mu    : 位置参数（均值，ν > 1 时存在）
        sigma : 尺度参数
    """
    z = (x - mu) / sigma
    coeff = gamma_func((nu + 1) / 2) / (gamma_func(nu / 2) * np.sqrt(nu * np.pi) * sigma)
    return float(coeff * (1 + z ** 2 / nu) ** (-(nu + 1) / 2))


def student_t_cdf(x: float, nu: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    学生 t 分布累积分布函数

    通过不完全 Beta 函数实现，调用 scipy.stats.t

    参数说明：同 student_t_pdf
    """
    return float(stats.t.cdf(x, df=nu, loc=mu, scale=sigma))


def student_t_quantile(p: float, nu: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    学生 t 分布分位数（逆 CDF）

    用于 t 分布 VaR 计算：VaR_α = μ + σ·t⁻¹(α; ν)
    """
    return float(stats.t.ppf(p, df=nu, loc=mu, scale=sigma))


def student_t_moments(nu: float, mu: float = 0.0, sigma: float = 1.0) -> dict:
    """
    学生 t 分布各阶矩

    E[X]       = μ                 (ν > 1)
    Var[X]     = σ² · ν/(ν-2)     (ν > 2)
    Skewness   = 0                 (对称，ν > 3)
    Kurt_excess = 6/(ν-4)         (ν > 4)

    返回：
        dict 含各阶矩和存在性条件
    """
    result = {'nu': nu, 'mu': mu, 'sigma': sigma}
    if nu > 1:
        result['mean'] = mu
    else:
        result['mean'] = None  # 均值不存在
    if nu > 2:
        result['variance'] = sigma ** 2 * nu / (nu - 2)
        result['std'] = sigma * np.sqrt(nu / (nu - 2))
    else:
        result['variance'] = np.inf
    result['skewness'] = 0.0 if nu > 3 else None
    if nu > 4:
        result['excess_kurtosis'] = 6.0 / (nu - 4)
    else:
        result['excess_kurtosis'] = np.inf if nu > 2 else None
    return result


def student_t_fit_mle(data: np.ndarray) -> tuple[float, float, float]:
    """
    学生 t 分布最大似然估计（数值优化）

    最大化：
        L(ν, μ, σ) = Σₙ ln f(xₙ; ν, μ, σ)

    参数说明：
        data : 收益率样本（日度或周度）

    返回：
        (nu_hat, mu_hat, sigma_hat)
    """
    # 使用 scipy 的 t.fit（它返回 (df, loc, scale)）
    nu_hat, mu_hat, sigma_hat = stats.t.fit(data, floc=None)
    return float(nu_hat), float(mu_hat), float(sigma_hat)


# ============================================================
# 2. 广义误差分布（GED / Power Exponential Distribution）
# ============================================================

def ged_pdf(x: float, nu: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    广义误差分布概率密度函数

    f(x; ν, μ, σ) = ν / (2·λ·Γ(1/ν)) · exp(-(|x-μ|/(λσ))^ν)

    其中 λ = [2^{-2/ν}·Γ(1/ν)/Γ(3/ν)]^{1/2}（使 Var=σ²）

    特殊情形：
        ν = 2  : 正态分布
        ν = 1  : 双指数/拉普拉斯分布（厚尾）
        ν → ∞  : 均匀分布

    GED 常用于 GARCH 模型的条件分布（Nelson 1991），
    ν ≈ 1.5 时匹配典型金融收益的尖峰厚尾特征

    参数说明：
        nu    : 形状参数（ν > 0）
        mu    : 位置参数
        sigma : 尺度参数
    """
    lam = np.sqrt(2 ** (-2 / nu) * gamma_func(1 / nu) / gamma_func(3 / nu))
    z = abs(x - mu) / (lam * sigma)
    coeff = nu / (2 * lam * sigma * gamma_func(1 / nu))
    return float(coeff * np.exp(-z ** nu))


def ged_moments(nu: float, sigma: float = 1.0) -> dict:
    """
    广义误差分布各阶矩

    E[X]       = 0（对称）
    Var[X]     = σ²
    Skewness   = 0（对称）
    Kurt_excess = Γ(5/ν)·Γ(1/ν) / Γ(3/ν)² - 3

    参数说明：
        nu    : 形状参数
        sigma : 尺度参数
    """
    kurt = gamma_func(5 / nu) * gamma_func(1 / nu) / gamma_func(3 / nu) ** 2 - 3
    return {
        'mean': 0.0,
        'variance': sigma ** 2,
        'std': sigma,
        'skewness': 0.0,
        'excess_kurtosis': float(kurt),
    }


# ============================================================
# 3. Hansen 偏斜 t 分布（Hansen 1994）
# ============================================================

def hansen_skew_t_pdf(x: float, nu: float, lam: float,
                       mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Hansen (1994) 偏斜 t 分布概率密度函数

    这是金融时间序列模型（DCC、GARCH-GJR）中最常用的偏斜厚尾分布：

        f(x; ν, λ) = bc · (1 + 1/(ν-2) · ((bx+a)/(1±λ))²)^{-(ν+1)/2}

    其中：
        a = 4λc(ν-2)/(ν-1)
        b = √(1 + 3λ² - a²)
        c = Γ((ν+1)/2) / (√(π(ν-2))·Γ(ν/2))
        + 号用于 x < -a/b，- 号用于 x ≥ -a/b

    特点：
        λ = 0 → 对称 t 分布
        λ > 0 → 右偏（正偏度）
        λ < 0 → 左偏（负偏度，股票常见）

    参数说明：
        nu  : 自由度（ν > 2）
        lam : 偏斜参数（λ ∈ (-1, 1)）
        mu, sigma : 位置、尺度
    """
    if nu <= 2 or abs(lam) >= 1:
        return 0.0
    z = (x - mu) / sigma
    c = gamma_func((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(max(1 + 3 * lam ** 2 - a ** 2, 1e-12))

    # 分段定义
    if z < -a / b:
        kernel = (1 + 1 / (nu - 2) * ((b * z + a) / (1 - lam)) ** 2) ** (-(nu + 1) / 2)
    else:
        kernel = (1 + 1 / (nu - 2) * ((b * z + a) / (1 + lam)) ** 2) ** (-(nu + 1) / 2)

    return float(b * c / sigma * kernel)


def hansen_skew_t_moments(nu: float, lam: float, sigma: float = 1.0) -> dict:
    """
    Hansen 偏斜 t 分布各阶矩

    均值 = a/b（非零，取决于偏斜参数）
    方差 = 1 - a²（标准化后）
    偏度：由 λ 控制（λ < 0 → 左偏）
    峰度：类似 t 分布但受 λ 修正

    返回：
        dict 含参数摘要
    """
    if nu <= 2:
        return {}
    c = gamma_func((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma_func(nu / 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(max(1 + 3 * lam ** 2 - a ** 2, 1e-12))

    mean_std = -a / b  # 标准化分布的均值
    var_std = 1.0      # 标准化分布的方差（按定义）

    return {
        'lambda': lam, 'nu': nu,
        'standardized_mean': float(mean_std),
        'standardized_variance': float(var_std),
        'mean': float(mu_loc + mean_std * sigma for mu_loc in [0.0]),  # 占位
        'note': '位置参数=0时的矩；实际均值=mu + sigma * mean_std',
    }


# ============================================================
# 4. Johnson SU 分布
# ============================================================

def johnson_su_pdf(x: float, xi: float, lam: float,
                    gamma: float, delta: float) -> float:
    """
    Johnson SU 分布概率密度函数

    Johnson SU（Unbounded）分布变换：
        z = γ + δ·sinh⁻¹((x-ξ)/λ)
        X = ξ + λ·sinh((Z-γ)/δ)，Z ~ N(0,1)

    PDF：
        f(x) = δ / (λ·√(2π)) · 1/√(1+((x-ξ)/λ)²) · exp(-z²/2)

    特点：
        可以精确匹配任意偏度和峰度（四参数灵活性）
        尾部比正态厚，但比 t 分布轻（有限矩）
        在金融中常用于拟合汇率收益率分布

    参数说明：
        xi    : 位置参数（ξ）
        lam   : 尺度参数（λ > 0）
        gamma : 形状参数 1（γ，偏斜控制）
        delta : 形状参数 2（δ > 0，厚尾控制）
    """
    if lam <= 0 or delta <= 0:
        return 0.0
    y = (x - xi) / lam
    z = gamma + delta * np.arcsinh(y)
    coeff = delta / (lam * np.sqrt(2 * np.pi) * np.sqrt(1 + y ** 2))
    return float(coeff * np.exp(-0.5 * z ** 2))


def johnson_su_cdf(x: float, xi: float, lam: float,
                    gamma: float, delta: float) -> float:
    """
    Johnson SU 分布累积分布函数

    F(x) = Φ(γ + δ·sinh⁻¹((x-ξ)/λ))

    因为变换后服从正态，CDF 有解析形式
    """
    y = (x - xi) / lam
    z = gamma + delta * np.arcsinh(y)
    return float(norm_cdf(z))


def johnson_su_moments(xi: float, lam: float,
                        gamma: float, delta: float) -> dict:
    """
    Johnson SU 分布各阶矩（解析公式）

    令 ω = exp(1/δ²)：
        E[X]     = ξ - λ·ω^{1/2}·sinh(γ/δ)
        Var[X]   = (λ²/2)·(ω-1)·(ω·cosh(2γ/δ)+1)
        Skewness : 由 γ 控制（γ=0 → 对称）

    参数说明：同 johnson_su_pdf
    """
    omega = np.exp(1.0 / delta ** 2)
    mean = xi - lam * np.sqrt(omega) * np.sinh(gamma / delta)
    var = (lam ** 2 / 2) * (omega - 1) * (omega * np.cosh(2 * gamma / delta) + 1)
    std = np.sqrt(max(var, 0))
    return {
        'mean': float(mean),
        'variance': float(var),
        'std': float(std),
        'omega': float(omega),
        'parameters': {'xi': xi, 'lambda': lam, 'gamma': gamma, 'delta': delta},
    }


def johnson_su_fit(data: np.ndarray) -> dict:
    """
    Johnson SU 分布矩匹配法拟合

    使用前四阶矩匹配（均值/方差/偏度/超额峰度）数值求解 (γ, δ)：
        给定样本矩，通过 scipy.optimize 匹配 Johnson SU 理论矩

    参数说明：
        data : 收益率样本

    返回：
        dict 含 (xi, lambda, gamma, delta) 参数
    """
    m1 = float(np.mean(data))
    m2 = float(np.var(data, ddof=1))
    m3_norm = float(stats.skew(data, bias=False))   # 偏度
    m4_norm = float(stats.kurtosis(data, bias=False))  # 超额峰度

    # 简化：用对称 Johnson SU（gamma=0）匹配均值/方差/峰度
    # 超额峰度 κ₄ = ω⁴ + 2ω³ + 3ω² - 6，ω = exp(1/δ²)
    # 给定 κ₄，数值求解 δ
    def kurt_eq(delta):
        if delta <= 0:
            return 1e6
        omega = np.exp(1.0 / delta ** 2)
        kurt_theo = omega ** 4 + 2 * omega ** 3 + 3 * omega ** 2 - 6
        return kurt_theo - (m4_norm + 3)  # 目标：总峰度

    try:
        delta_hat = brentq(kurt_eq, 0.1, 10.0, xtol=1e-8)
    except ValueError:
        delta_hat = 1.0  # 回退值

    omega_hat = np.exp(1.0 / delta_hat ** 2)
    gamma_hat = -m3_norm / abs(m3_norm) * delta_hat / 2 if m3_norm != 0 else 0.0

    # 从均值/方差匹配 xi, lambda
    var_theo = (1 / 2) * (omega_hat - 1) * (omega_hat * np.cosh(2 * gamma_hat / delta_hat) + 1)
    if var_theo > 1e-10:
        lam_hat = np.sqrt(m2 / var_theo)
    else:
        lam_hat = np.sqrt(m2)
    xi_hat = m1 + lam_hat * np.sqrt(omega_hat) * np.sinh(gamma_hat / delta_hat)

    return {
        'xi': float(xi_hat),
        'lambda': float(lam_hat),
        'gamma': float(gamma_hat),
        'delta': float(delta_hat),
        'target_skewness': m3_norm,
        'target_excess_kurtosis': m4_norm,
    }


# ============================================================
# 5. Cornish-Fisher 展开（VaR/ES 修正）
# ============================================================

def cornish_fisher_quantile(p: float, mu: float, sigma: float,
                              skew: float, excess_kurt: float) -> float:
    """
    Cornish-Fisher 展开修正分位数

    在正态分位数基础上加入偏度/峰度修正：
        z_CF = z_p
             + (z_p² - 1)/6 · κ₃
             + (z_p³ - 3z_p)/24 · κ₄
             - (2z_p³ - 5z_p)/36 · κ₃²

    其中 z_p = Φ⁻¹(p) 为正态分位数，κ₃ = 偏度，κ₄ = 超额峰度

    CF 展开广泛应用于：
    1. 历史模拟法的偏度/峰度修正
    2. RiskMetrics 修正 VaR（Modified VaR）
    3. 资产配置中的 CVaR 近似

    参数说明：
        p             : 置信水平（如 VaR 99% → p=0.01）
        mu, sigma     : 分布均值和标准差
        skew          : 偏度
        excess_kurt   : 超额峰度

    返回：
        修正分位数（Modified Quantile）
    """
    from utils.common import norm_inv
    z_p = norm_inv(p)

    # Cornish-Fisher 修正
    z_cf = (z_p
            + (z_p ** 2 - 1) / 6 * skew
            + (z_p ** 3 - 3 * z_p) / 24 * excess_kurt
            - (2 * z_p ** 3 - 5 * z_p) / 36 * skew ** 2)

    return float(mu + sigma * z_cf)


def modified_var(mu: float, sigma: float, skew: float, excess_kurt: float,
                  alpha: float = 0.01) -> float:
    """
    修正 VaR（Modified Value-at-Risk）—— Zangari (1996)

    通过 Cornish-Fisher 展开修正正态 VaR：
        MVaR_α = -(μ + σ·z_CF(α))  （取负数，损失为正）

    正态 VaR 假设收益率正态分布，
    修正 VaR 纳入实际偏度（负偏 → 更厚左尾 → 损失更大）
    和超额峰度（峰度越高 → 极端事件更频繁）

    参数说明：
        mu, sigma     : 收益率均值和标准差
        skew          : 偏度（负值代表左偏）
        excess_kurt   : 超额峰度
        alpha         : 显著性水平（如 0.01 = 99% VaR）

    返回：
        修正 VaR（正值 = 损失）
    """
    q = cornish_fisher_quantile(alpha, mu, sigma, skew, excess_kurt)
    return -q  # 损失取正


def modified_es(mu: float, sigma: float, skew: float, excess_kurt: float,
                 alpha: float = 0.01, n_simulations: int = 100000) -> float:
    """
    修正期望损失（Modified Expected Shortfall / CVaR）

    ES_α = E[X | X ≤ VaR_α] = 期望亏损条件于亏损超过 VaR

    通过蒙特卡洛近似（使用 Cornish-Fisher 转换的伪随机数）：
        1. 生成正态随机数 Z ~ N(0,1)
        2. 应用 CF 变换近似目标分布
        3. 计算超过 VaR 的条件均值

    参数说明：
        n_simulations : 模拟路径数

    返回：
        修正 ES（正值 = 期望损失）
    """
    np.random.seed(42)
    z = np.random.standard_normal(n_simulations)

    # CF 近似变换（将标准正态近似映射到目标分布）
    z_cf = (z
            + (z ** 2 - 1) / 6 * skew
            + (z ** 3 - 3 * z) / 24 * excess_kurt
            - (2 * z ** 3 - 5 * z) / 36 * skew ** 2)

    returns = mu + sigma * z_cf
    var_est = modified_var(mu, sigma, skew, excess_kurt, alpha)
    losses_exceeding = returns[returns < -var_est]

    if len(losses_exceeding) == 0:
        return var_est
    return float(-np.mean(losses_exceeding))


# ============================================================
# 6. VaR 与 Expected Shortfall（多种方法）
# ============================================================

def historical_var(returns: np.ndarray, alpha: float = 0.01) -> float:
    """
    历史模拟法 VaR

    直接从历史收益率样本提取分位数：
        VaR_α = -Q_α(returns)  = 第 α·100 百分位数的负值

    优点：无分布假设，自然包含厚尾/偏度
    缺点：依赖历史数据，对稀有事件估计不准

    参数说明：
        returns : 历史收益率序列（日度）
        alpha   : 显著性水平（0.01 = 99% VaR）

    返回：
        历史 VaR（正值 = 损失）
    """
    return float(-np.percentile(returns, alpha * 100))


def historical_es(returns: np.ndarray, alpha: float = 0.01) -> float:
    """
    历史模拟法期望损失（Expected Shortfall）

    ES_α = -E[X | X ≤ Q_α] = 超过 VaR 的条件期望损失

    ES 是一致性风险度量（VaR 不满足次可加性）

    参数说明：
        returns : 历史收益率序列
        alpha   : 显著性水平

    返回：
        历史 ES（正值 = 期望损失）
    """
    var = historical_var(returns, alpha)
    tail_returns = returns[returns < -var]
    if len(tail_returns) == 0:
        return var
    return float(-np.mean(tail_returns))


def parametric_var_normal(mu: float, sigma: float, alpha: float = 0.01,
                           horizon: int = 1) -> float:
    """
    参数化 VaR（正态分布假设）

    VaR_α,T = -(μT - z_{1-α}·σ√T)

    √T 规则（Basel III）：日 VaR 乘以 √10 得 10 日 VaR

    参数说明：
        mu, sigma : 日收益率均值和标准差
        alpha     : 显著性水平
        horizon   : 持有期（天数）

    返回：
        参数化 VaR
    """
    from utils.common import norm_inv
    z_alpha = norm_inv(alpha)
    mu_T = mu * horizon
    sigma_T = sigma * np.sqrt(horizon)
    return float(-(mu_T + z_alpha * sigma_T))


def parametric_var_t(mu: float, sigma: float, nu: float,
                      alpha: float = 0.01, horizon: int = 1) -> float:
    """
    参数化 VaR（学生 t 分布假设）

    VaR_α,T = -(μT - t_{α,ν}·σ_adj·√T)

    其中 σ_adj = σ·√(ν/(ν-2)) 确保方差等于 σ²

    t 分布 VaR 在尾部比正态 VaR 更保守（更大的损失估计）

    参数说明：
        nu : 自由度（从历史数据 MLE 估计）

    返回：
        t 分布 VaR
    """
    if nu <= 2:
        return np.inf
    t_alpha = float(stats.t.ppf(alpha, df=nu))
    # 调整尺度参数（t 分布标准差调整）
    sigma_adj = sigma / np.sqrt(nu / (nu - 2))
    mu_T = mu * horizon
    sigma_T = sigma_adj * np.sqrt(horizon)
    return float(-(mu_T + t_alpha * sigma_T))


def parametric_es_normal(mu: float, sigma: float, alpha: float = 0.01,
                           horizon: int = 1) -> float:
    """
    参数化 ES（正态分布假设）

    ES_α = -μT + σ√T · φ(z_α) / α

    其中 φ 为标准正态 PDF，z_α = Φ⁻¹(α)

    正态分布下 ES = VaR × (φ(z_α)/(α·|z_α|))（ES > VaR）

    参数说明：
        同 parametric_var_normal
    """
    from utils.common import norm_inv
    z_alpha = norm_inv(alpha)
    mu_T = mu * horizon
    sigma_T = sigma * np.sqrt(horizon)
    es = -(mu_T - sigma_T * norm_pdf(z_alpha) / alpha)
    return float(es)


def parametric_es_t(mu: float, sigma: float, nu: float,
                     alpha: float = 0.01, horizon: int = 1) -> float:
    """
    参数化 ES（学生 t 分布假设）

    ES_α = -μT + σ√T · (f_t(t_α;ν)/α) · (ν + t_α²) / (ν-1)

    其中 f_t 为 t 分布 PDF，t_α 为 α 分位数

    t 分布 ES 在 ν 较小时显著大于正态 ES（尾部更厚）

    参数说明：
        nu : 自由度
    """
    if nu <= 1:
        return np.inf
    t_alpha = float(stats.t.ppf(alpha, df=nu))
    sigma_adj = sigma / np.sqrt(nu / (nu - 2)) if nu > 2 else sigma
    mu_T = mu * horizon
    sigma_T = sigma_adj * np.sqrt(horizon)

    pdf_t = float(stats.t.pdf(t_alpha, df=nu))
    es_component = -sigma_T * pdf_t * (nu + t_alpha ** 2) / ((nu - 1) * alpha)
    return float(-mu_T + es_component)


# ============================================================
# 7. 极值理论（Extreme Value Theory）
# ============================================================

def gev_pdf(x: float, xi: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    广义极值分布（GEV）概率密度函数

    GEV 统一了三种极值分布：
        ξ > 0 : Fréchet（厚尾，股票收益 ξ ≈ 0.2-0.4）
        ξ = 0 : Gumbel（轻尾，最大值渐近）
        ξ < 0 : Weibull（有界上尾）

    f(x) = (1/σ) · t(x)^{ξ+1} · exp(-t(x))

    其中 t(x) = (1 + ξ·(x-μ)/σ)^{-1/ξ}  (ξ ≠ 0)
              = exp(-(x-μ)/σ)              (ξ = 0)

    GEV 用于月度/年度最大/最小损失的分布建模

    参数说明：
        xi    : 形状参数（尾部指数）
        mu    : 位置参数
        sigma : 尺度参数（> 0）
    """
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        # Gumbel 极限
        t = np.exp(-z)
        return float(np.exp(-z - t) / sigma)
    else:
        u = 1 + xi * z
        if u <= 0:
            return 0.0
        t = u ** (-1 / xi)
        return float(t ** (xi + 1) * np.exp(-t) / sigma)


def gev_cdf(x: float, xi: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    广义极值分布累积分布函数

    F(x) = exp(-t(x))

    参数说明：同 gev_pdf
    """
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return float(np.exp(-np.exp(-z)))
    else:
        u = 1 + xi * z
        if u <= 0:
            return 0.0 if xi > 0 else 1.0
        return float(np.exp(-u ** (-1 / xi)))


def gpd_pdf(x: float, xi: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    广义 Pareto 分布（GPD）概率密度函数

    GPD 是超越阈值（POT, Peaks-Over-Threshold）方法的基础：
    当 X - u | X > u （超越量）的分布在 u → ∞ 时收敛到 GPD

    f(x) = (1/σ) · (1 + ξ·(x-μ)/σ)^{-(1/ξ+1)}  (ξ ≠ 0)
         = (1/σ) · exp(-(x-μ)/σ)                  (ξ = 0)

    在 VaR/ES 估计中：
        超越 VaR 的尾部分布 ~ GPD(ξ, u, σ_u)
        ES_α = VaR_α / (1-ξ) + (σ - ξ·u) / (1-ξ)

    参数说明：
        xi    : 形状参数（ξ < 0.5 时 ES 存在）
        mu    : 阈值（通常设为 VaR 估计值）
        sigma : 超越尺度参数
    """
    z = (x - mu) / sigma
    if z < 0:
        return 0.0
    if abs(xi) < 1e-8:
        return float(np.exp(-z) / sigma)
    else:
        u = 1 + xi * z
        if u <= 0:
            return 0.0
        return float(u ** (-(1 / xi + 1)) / sigma)


def gpd_fit_mle(exceedances: np.ndarray) -> tuple[float, float]:
    """
    GPD 最大似然估计（POT 方法）

    对超越量 y = x - u 拟合 GPD(ξ, σ)：
        L(ξ, σ) = -n·ln(σ) - (1+1/ξ) Σ ln(1 + ξ·yᵢ/σ)  (ξ ≠ 0)

    参数说明：
        exceedances : 超越阈值的样本（即 y = x - threshold > 0）

    返回：
        (xi_hat, sigma_hat)
    """
    y = np.asarray(exceedances, dtype=float)
    y = y[y > 0]

    def neg_loglik(params):
        xi, sigma = params
        if sigma <= 0:
            return 1e10
        if xi == 0:
            return len(y) * np.log(sigma) + np.sum(y / sigma)
        z = 1 + xi * y / sigma
        if np.any(z <= 0):
            return 1e10
        return len(y) * np.log(sigma) + (1 + 1 / xi) * np.sum(np.log(z))

    res = minimize(neg_loglik, [0.1, np.mean(y)],
                   method='L-BFGS-B',
                   bounds=[(-0.5, 1.0), (1e-6, None)])
    return float(res.x[0]), float(res.x[1])


def evt_var_es(data: np.ndarray, threshold_pct: float = 0.90,
               alpha: float = 0.01) -> dict:
    """
    极值理论 VaR 和 ES 估计（POT 方法）

    步骤：
    1. 确定阈值 u = threshold_pct 分位数
    2. 对超越量 y = x - u 拟合 GPD(ξ, σ_u)
    3. 外推：
       VaR_α = u + (σ_u/ξ)·[((1-threshold_pct)/α)^ξ - 1]
       ES_α  = VaR_α / (1-ξ) + (σ_u - ξ·u) / (1-ξ)

    EVT 方法的优势：对极端尾部（1% 以下）估计更准确，
    不依赖分布假设（由极值定理保证渐近有效性）

    参数说明：
        data           : 损失序列（正值 = 损失）
        threshold_pct  : 阈值分位数
        alpha          : VaR/ES 显著性水平

    返回：
        dict 含 EVT-VaR, EVT-ES 及拟合参数
    """
    losses = np.asarray(data, dtype=float)
    u = float(np.quantile(losses, threshold_pct))
    exceedances = losses[losses > u] - u

    if len(exceedances) < 10:
        return {'error': f'超越量样本量不足（{len(exceedances)} < 10）'}

    xi_hat, sigma_hat = gpd_fit_mle(exceedances)
    n = len(losses)
    n_u = len(exceedances)

    # EVT VaR
    if abs(xi_hat) > 1e-6:
        var_evt = u + (sigma_hat / xi_hat) * ((n * (1 - threshold_pct) / (n_u * alpha)) ** xi_hat - 1)
    else:
        var_evt = u + sigma_hat * np.log(n * (1 - threshold_pct) / (n_u * alpha))

    # EVT ES（需要 ξ < 1）
    if xi_hat < 1:
        es_evt = var_evt / (1 - xi_hat) + (sigma_hat - xi_hat * u) / (1 - xi_hat)
    else:
        es_evt = np.inf

    return {
        'threshold': u,
        'n_exceedances': n_u,
        'xi': xi_hat,
        'sigma': sigma_hat,
        'var_evt': float(var_evt),
        'es_evt': float(es_evt),
    }


# ============================================================
# 8. 分布拟合与检验工具
# ============================================================

def ks_test(data: np.ndarray, distribution: str = 'norm',
             params: tuple = None) -> dict:
    """
    Kolmogorov-Smirnov 正态性检验

    检验统计量：D_n = sup_x |F_n(x) - F(x)|
    （经验 CDF 与理论 CDF 的最大绝对偏差）

    H₀: 数据来自指定分布
    拒绝域：D_n > D_n,α（临界值）

    参数说明：
        distribution : 'norm', 't', 'lognorm' 等（scipy.stats 支持的分布）
        params       : 分布参数元组（None = 从数据估计）

    返回：
        dict 含 KS 统计量和 p 值
    """
    dist = getattr(stats, distribution)
    if params is None:
        params = dist.fit(data)

    stat, pval = stats.kstest(data, distribution, args=params)
    return {
        'statistic': float(stat),
        'pvalue': float(pval),
        'reject_h0_005': bool(pval < 0.05),
        'distribution': distribution,
        'params': params,
    }


def anderson_darling_test(data: np.ndarray) -> dict:
    """
    Anderson-Darling 正态性检验

    对尾部更敏感（比 KS 检验功效更高）：
        A² = -n - (1/n) Σ(2i-1)[ln(z_i) + ln(1-z_{n+1-i})]

    其中 z_i = Φ((x_(i) - μ̂)/σ̂)（排序后的标准化观测值）

    参数说明：
        data : 样本数据

    返回：
        dict 含 AD 统计量和临界值
    """
    result = stats.anderson(data, dist='norm')
    return {
        'statistic': float(result.statistic),
        'critical_values': {f'{sl}%': cv for sl, cv in zip(result.significance_level, result.critical_values)},
        'reject_005': bool(result.statistic > result.critical_values[2]),  # 5% 显著性水平
    }


def qq_plot_data(data: np.ndarray, distribution: str = 'norm',
                  params: tuple = None) -> dict:
    """
    Q-Q 图数据生成（用于可视化拟合优度）

    Q-Q 图：将样本分位数 vs 理论分位数绘图
    若数据服从该分布，点应落在 y=x 直线上
    尾部偏离指示厚尾（S 形曲线）或偏度（上下非对称偏离）

    参数说明：
        distribution : 参考分布
        params       : 分布参数（None = 从数据估计）

    返回：
        dict 含 theoretical_quantiles, sample_quantiles
    """
    n = len(data)
    dist = getattr(stats, distribution)
    if params is None:
        params = dist.fit(data)

    sorted_data = np.sort(data)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = dist.ppf(probs, *params)

    return {
        'theoretical_quantiles': theoretical.tolist(),
        'sample_quantiles': sorted_data.tolist(),
        'n': n,
        'distribution': distribution,
    }


# ============================================================
# 9. 期权隐含矩提取
# ============================================================

def implied_moments_from_options(strikes: np.ndarray,
                                  call_prices: np.ndarray,
                                  put_prices: np.ndarray,
                                  S: float, F: float, r: float, T: float) -> dict:
    """
    从期权价格提取风险中性密度的各阶矩（Bakshi, Kapadia, Madan 2003）

    BKM (2003) 方法：通过加权积分构建矩：
        E^Q[R]   ≈ 方差风险溢价（通过 V(K)）
        E^Q[R²]  ≈ 波动率互换（通过 W(K)）
        E^Q[R³]  ≈ 偏度互换（通过 X(K)）
        E^Q[R⁴]  ≈ 峰度互换（通过 Y(K)）

    V(K) = ∫₀^∞ 2(1 - ln(K/F)) / K² · C(K)dK   (OTM call)
         + ∫₀^∞ 2(1 + ln(F/K)) / K² · P(K)dK   (OTM put)

    参数说明：
        strikes    : 行权价数组（升序）
        call_prices: 对应的 OTM 看涨期权价格
        put_prices : 对应的 OTM 看跌期权价格
        S          : 当前股价
        F          : 远期价格
        r          : 无风险利率
        T          : 到期时间

    返回：
        dict 含风险中性偏度、峰度等隐含矩
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    P = np.asarray(put_prices, dtype=float)
    df = np.exp(-r * T)

    # 分离 OTM calls 和 puts
    otm_call_mask = K >= F
    otm_put_mask = K < F

    # BKM 权重函数
    # V(K) 权重
    def w_V_call(k): return 2 * (1 - np.log(k / F)) / k ** 2
    def w_V_put(k): return 2 * (1 + np.log(F / k)) / k ** 2

    # W(K)（三阶矩）权重
    def w_W_call(k): return (6 * np.log(k / F) - 3 * np.log(k / F) ** 2) / k ** 2
    def w_W_put(k): return (-6 * np.log(F / k) - 3 * np.log(F / k) ** 2) / k ** 2

    # X(K)（四阶矩）权重
    def w_X_call(k): return (12 * np.log(k / F) ** 2 - 4 * np.log(k / F) ** 3) / k ** 2
    def w_X_put(k): return (12 * np.log(F / k) ** 2 + 4 * np.log(F / k) ** 3) / k ** 2

    def integrate_options(w_call_fn, w_put_fn):
        """梯形积分计算 BKM 矩"""
        total = 0.0
        # OTM calls
        K_c = K[otm_call_mask]
        C_c = C[otm_call_mask]
        if len(K_c) > 1:
            total += np.trapz(w_call_fn(K_c) * C_c * df, K_c)
        # OTM puts
        K_p = K[otm_put_mask]
        P_p = P[otm_put_mask]
        if len(K_p) > 1:
            total += np.trapz(w_put_fn(K_p) * P_p * df, K_p)
        return total

    try:
        V = integrate_options(w_V_call, w_V_put)
        W = integrate_options(w_W_call, w_W_put)
        X = integrate_options(w_X_call, w_X_put)

        # 风险中性期望对数收益
        mu_Q = np.exp(r * T) - 1 - np.exp(r * T) * V / 2 - np.exp(r * T) * W / 6 - np.exp(r * T) * X / 24

        # 风险中性矩（Bakshi et al 公式）
        var_Q = np.exp(r * T) * V - mu_Q ** 2
        skew_Q = (np.exp(r * T) * W - 3 * mu_Q * np.exp(r * T) * V + 2 * mu_Q ** 3) / max(var_Q ** 1.5, 1e-12)
        kurt_Q = (np.exp(r * T) * X - 4 * mu_Q * np.exp(r * T) * W
                  + 6 * mu_Q ** 2 * np.exp(r * T) * V - 3 * mu_Q ** 4) / max(var_Q ** 2, 1e-12)

        return {
            'risk_neutral_mean': float(mu_Q),
            'risk_neutral_variance': float(var_Q),
            'risk_neutral_vol': float(np.sqrt(max(var_Q, 0))),
            'risk_neutral_skewness': float(skew_Q),
            'risk_neutral_excess_kurtosis': float(kurt_Q - 3),
            'V': float(V), 'W': float(W), 'X': float(X),
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================
# 示例与测试
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("第13章（二）：资产收益率分布与风险度量")
    print("=" * 60)

    # 生成模拟日收益率（t 分布 + 轻微左偏）
    nu_true = 5.0
    sim_returns = stats.t.rvs(df=nu_true, loc=0.0005, scale=0.012, size=1000)

    # ── 1. 学生 t 分布 ──────────────────────────────────────────────
    print("\n── 学生 t 分布拟合 ──")
    nu_hat, mu_hat, sigma_hat = student_t_fit_mle(sim_returns)
    print(f"MLE 估计：ν={nu_hat:.2f}（真实={nu_true:.1f}），μ={mu_hat:.6f}，σ={sigma_hat:.6f}")
    t_mom = student_t_moments(nu_hat)
    print(f"超额峰度 = {t_mom.get('excess_kurtosis', 'N/A'):.4f}（正态=0）")

    # ── 2. GED 分布 ─────────────────────────────────────────────────
    print("\n── 广义误差分布（GED）──")
    for nu_ged in [1.0, 1.5, 2.0, 3.0]:
        g = ged_moments(nu_ged, 1.0)
        print(f"ν={nu_ged:.1f}: 超额峰度={g['excess_kurtosis']:.4f}"
              f"{'（正态）' if nu_ged == 2.0 else ''}")

    # ── 3. Johnson SU 分布 ──────────────────────────────────────────
    print("\n── Johnson SU 分布拟合 ──")
    jsu_params = johnson_su_fit(sim_returns)
    print(f"拟合参数：ξ={jsu_params['xi']:.6f}, λ={jsu_params['lambda']:.6f}, "
          f"γ={jsu_params['gamma']:.6f}, δ={jsu_params['delta']:.6f}")
    jsu_mom = johnson_su_moments(jsu_params['xi'], jsu_params['lambda'],
                                  jsu_params['gamma'], jsu_params['delta'])
    print(f"拟合均值={jsu_mom['mean']:.6f}, 标准差={jsu_mom['std']:.6f}")

    # ── 4. Cornish-Fisher 修正 VaR ───────────────────────────────────
    print("\n── Cornish-Fisher 修正 VaR（99% 置信水平）──")
    mu_r = float(np.mean(sim_returns))
    sigma_r = float(np.std(sim_returns, ddof=1))
    skew_r = float(stats.skew(sim_returns, bias=False))
    kurt_r = float(stats.kurtosis(sim_returns, bias=False))

    var_norm = parametric_var_normal(mu_r, sigma_r, alpha=0.01)
    var_t = parametric_var_t(mu_r, sigma_r, nu=nu_hat, alpha=0.01)
    var_cf = modified_var(mu_r, sigma_r, skew_r, kurt_r, alpha=0.01)
    var_hist = historical_var(sim_returns, alpha=0.01)

    print(f"正态 VaR(99%):   {var_norm:.6f}")
    print(f"t 分布 VaR(99%): {var_t:.6f}")
    print(f"CF 修正 VaR(99%): {var_cf:.6f}  （偏度={skew_r:.3f}, 超额峰度={kurt_r:.3f}）")
    print(f"历史 VaR(99%):   {var_hist:.6f}")

    # ── 5. Expected Shortfall 对比 ───────────────────────────────────
    print("\n── Expected Shortfall（ES）对比 ──")
    es_norm = parametric_es_normal(mu_r, sigma_r, alpha=0.01)
    es_t = parametric_es_t(mu_r, sigma_r, nu=nu_hat, alpha=0.01)
    es_hist = historical_es(sim_returns, alpha=0.01)

    print(f"正态 ES(99%):   {es_norm:.6f}")
    print(f"t 分布 ES(99%): {es_t:.6f}")
    print(f"历史 ES(99%):   {es_hist:.6f}")
    print(f"ES/VaR 比（正态）: {es_norm / var_norm:.4f}（应≈1.17 for 99%）")

    # ── 6. 极值理论 VaR/ES ──────────────────────────────────────────
    print("\n── 极值理论（EVT）VaR/ES ──")
    # 将收益率转为损失（取负）
    losses = -sim_returns
    evt_result = evt_var_es(losses, threshold_pct=0.90, alpha=0.01)
    if 'error' not in evt_result:
        print(f"GPD 形状参数 ξ={evt_result['xi']:.4f}（>0 为厚尾）")
        print(f"超越量数={evt_result['n_exceedances']}")
        print(f"EVT VaR(99%): {evt_result['var_evt']:.6f}")
        print(f"EVT ES(99%):  {evt_result['es_evt']:.6f}")

    # ── 7. GEV 分布 ─────────────────────────────────────────────────
    print("\n── 广义极值分布（GEV）──")
    # 月度最大损失
    n_months = 10
    monthly_max = [np.max(-sim_returns[i*100:(i+1)*100]) for i in range(n_months)]
    print(f"月度最大损失样本: {[f'{x:.4f}' for x in monthly_max[:5]]}...")

    for xi_test, name in [(0.3, 'Fréchet（厚尾）'), (0.0, 'Gumbel（中尾）'), (-0.3, 'Weibull（有界）')]:
        pdf_val = gev_pdf(0.03, xi=xi_test, mu=0.02, sigma=0.005)
        print(f"{name}（ξ={xi_test:.1f}）: PDF(0.03) = {pdf_val:.4f}")

    # ── 8. 正态性检验 ────────────────────────────────────────────────
    print("\n── 正态性检验 ──")
    ks_result = ks_test(sim_returns, 'norm')
    print(f"KS 检验（正态）: stat={ks_result['statistic']:.4f}, "
          f"p={ks_result['pvalue']:.6f}, 拒绝H₀(5%)={ks_result['reject_h0_005']}")

    ad_result = anderson_darling_test(sim_returns)
    print(f"AD 检验: stat={ad_result['statistic']:.4f}, "
          f"拒绝H₀(5%)={ad_result['reject_005']}")

    # ── 9. Hansen 偏斜 t 分布 ────────────────────────────────────────
    print("\n── Hansen 偏斜 t 分布 ──")
    nu_sk, lam_sk = 6.0, -0.20  # 自由度6，轻微左偏
    x_test = 0.0
    pdf_sk = hansen_skew_t_pdf(x_test, nu_sk, lam_sk, 0.0, 0.012)
    pdf_norm_ref = normal_pdf(x_test, 0.0, 0.012)
    print(f"x=0: 偏斜 t PDF = {pdf_sk:.6f}，正态 PDF = {pdf_norm_ref:.6f}")
    print(f"峰值比（偏斜t/正态）: {pdf_sk / pdf_norm_ref:.4f}")

    # ── 10. 期权隐含矩（BKM 方法）─────────────────────────────────
    print("\n── 期权隐含矩（BKM 2003）──")
    S_bkm = 100.0
    F_bkm = 102.0  # 远期价格（含利率）
    r_bkm = 0.05
    T_bkm = 0.25

    # 构造模拟期权价格（用正态近似生成）
    strikes_bkm = np.array([85., 90., 95., 100., 105., 110., 115.])
    # 简化：用 BSM 价格生成模拟数据（实践中来自市场）
    from utils.common import norm_cdf as Phi
    sigma_bkm = 0.20
    call_prices_bkm = []
    put_prices_bkm = []
    for K_bkm in strikes_bkm:
        sqT = np.sqrt(T_bkm)
        d1 = (np.log(F_bkm / K_bkm) + 0.5 * sigma_bkm ** 2 * T_bkm) / (sigma_bkm * sqT)
        d2 = d1 - sigma_bkm * sqT
        df_bkm = np.exp(-r_bkm * T_bkm)
        c = df_bkm * (F_bkm * Phi(d1) - K_bkm * Phi(d2))
        p = df_bkm * (K_bkm * Phi(-d2) - F_bkm * Phi(-d1))
        call_prices_bkm.append(c)
        put_prices_bkm.append(p)

    bkm = implied_moments_from_options(
        strikes_bkm, np.array(call_prices_bkm), np.array(put_prices_bkm),
        S_bkm, F_bkm, r_bkm, T_bkm
    )
    if 'error' not in bkm:
        print(f"风险中性波动率: {bkm['risk_neutral_vol']:.4f}（应≈{sigma_bkm:.4f}）")
        print(f"风险中性偏度: {bkm['risk_neutral_skewness']:.4f}（对称模型应≈0）")
        print(f"风险中性超额峰度: {bkm['risk_neutral_excess_kurtosis']:.4f}")
