"""
01_volatility_estimation.py — 波动率估计方法
============================================
【模型简介】
波动率（Volatility）是期权定价中最关键的输入参数，却不可直接观测。
本文件实现从历史价格数据中估计波动率的主要方法：

A. 历史波动率（Historical Volatility）
──────────────────────────────────────
  1. 收盘价收益率法（Close-to-Close）— 最经典
  2. Parkinson (1980) 极值估计量（High-Low）— 方差减少 5倍
  3. Garman-Klass (1980) 估计量（OHLC）— 最高效利用日内数据
  4. Rogers-Satchell (1991) 估计量（含漂移修正）
  5. Yang-Zhang (2000) 估计量（最优 OHLC 估计）

B. 指数加权移动平均（EWMA）
────────────────────────────
  J.P. Morgan RiskMetrics 模型：
  σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
  典型 λ = 0.94（日数据），0.97（月数据）

C. GARCH(1,1) 模型
────────────────────
  Bollerslev (1986)：
  σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
  参数约束：ω > 0，α ≥ 0，β ≥ 0，α+β < 1
  长期方差：σ²_∞ = ω / (1 - α - β)

D. 隐含波动率曲面（Implied Volatility Surface）
────────────────────────────────────────────────
  通过期权市场价格反推的波动率：
  - 逐点隐含波动率（Newton-Raphson 方法）
  - 波动率曲面插值与拟合（SVI 参数化）

E. 波动率期限结构（Volatility Term Structure）
───────────────────────────────────────────────
  VIX 式方差互换公式（离散化近似）：
  σ² = (2/T) · Σ [ΔK/K²] · e^{rT} · option_price
  σ^{T1→T2} = √[(σ²_{T2}·T2 - σ²_{T1}·T1)/(T2-T1)]  ← 远期波动率

参考：
  - Parkinson, M. (1980). "The Extreme Value Method for Estimating
    the Variance of the Rate of Return." Journal of Business, 53, 61–65.
  - Garman, M.B. & Klass, M.J. (1980). "On the Estimation of Security
    Price Volatilities from Historical Data." Journal of Business, 53, 67–78.
  - Yang, D. & Zhang, Q. (2000). "Drift-Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices." Journal of Business, 73, 477–491.
书中对应：Haug (2007), Chapter 12
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, pi
import numpy as np
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. 历史波动率估计量
# ═══════════════════════════════════════════════════════════════

def close_to_close_vol(closes: list, ann_factor: float = 252.) -> float:
    """
    收盘价对数收益率波动率（Close-to-Close Historical Volatility）。

    最经典、最简单的历史波动率估计：
    rᵢ = ln(Cᵢ / Cᵢ₋₁)
    σ² = Σ(rᵢ - r̄)² / (n-1)
    σ_annual = σ_daily × √ann_factor

    缺点：仅用收盘价，丢弃了日内极值信息，效率低。

    参数
    ----
    closes     : 收盘价序列（时间顺序排列）
    ann_factor : 年化因子（252=工作日, 365=日历日, 12=月度）

    返回
    ----
    float : 年化历史波动率
    """
    if len(closes) < 2:
        raise ValueError("需要至少 2 个收盘价")

    log_rets = [log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    n_obs = len(log_rets)
    mean_ret = sum(log_rets) / n_obs
    variance = sum((r - mean_ret)**2 for r in log_rets) / (n_obs - 1)
    return sqrt(variance * ann_factor)


def parkinson_vol(highs: list, lows: list, ann_factor: float = 252.) -> float:
    """
    Parkinson (1980) 极值波动率估计量（High-Low Estimator）。

    利用日内最高价和最低价，比收盘价法效率高 ~5 倍：

    σ² = 1/(4·ln2) · Σ[ln(Hᵢ/Lᵢ)]² / n

    其中 1/(4·ln2) ≈ 0.3607（修正偏差因子）

    假设：无漂移（μ=0），连续交易（无隔夜跳跃）。
    实践中隔夜缺口会使估计量产生偏差。

    参数
    ----
    highs : 日最高价序列
    lows  : 日最低价序列
    """
    if len(highs) != len(lows):
        raise ValueError("高低价序列长度不一致")
    n_obs = len(highs)
    factor = 1. / (4. * log(2.))
    var_daily = factor * sum((log(H/L))**2 for H, L in zip(highs, lows)) / n_obs
    return sqrt(var_daily * ann_factor)


def garman_klass_vol(opens: list, highs: list, lows: list, closes: list,
                      ann_factor: float = 252.) -> float:
    """
    Garman-Klass (1980) 波动率估计量（OHLC Estimator）。

    最大化利用 OHLC 数据（开高低收），效率比收盘价法高 ~7 倍：

    σ²_GK = Σ [0.5·ln(H/L)² - (2ln2-1)·ln(C/O)²] / n

    其中：
    - 0.5·ln(H/L)² → Parkinson 项（极值贡献）
    - -(2ln2-1)·ln(C/O)² → 开收价修正（去除漂移偏差）
    - 系数 2ln2-1 ≈ 0.3863

    假设：无隔夜跳跃（opens = previous close）。

    参数
    ----
    opens, highs, lows, closes : OHLC 价格序列（等长）
    """
    n_obs = len(closes)
    gk_sum = 0.
    for O, H, L, C in zip(opens, highs, lows, closes):
        if O <= 0 or H <= 0 or L <= 0 or C <= 0:
            continue
        gk_sum += (0.5 * log(H/L)**2
                   - (2.*log(2.) - 1.) * log(C/O)**2)

    var_daily = gk_sum / n_obs
    return sqrt(max(var_daily, 0.) * ann_factor)


def rogers_satchell_vol(opens: list, highs: list, lows: list, closes: list,
                         ann_factor: float = 252.) -> float:
    """
    Rogers-Satchell (1991) 波动率估计量（漂移无偏 OHLC 估计）。

    Garman-Klass 假设无漂移，Rogers-Satchell 对存在趋势的市场更准确：

    σ²_RS = Σ [ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)] / n

    特点：
    - 对任意漂移方向无偏（适合牛市/熊市）
    - 不需要开盘价等于前收盘价
    - 不处理隔夜跳跃（跳跃会被低估）
    """
    n_obs = len(closes)
    rs_sum = 0.
    for O, H, L, C in zip(opens, highs, lows, closes):
        if O <= 0 or H <= 0 or L <= 0 or C <= 0:
            continue
        rs_sum += (log(H/C) * log(H/O)
                   + log(L/C) * log(L/O))

    var_daily = rs_sum / n_obs
    return sqrt(max(var_daily, 0.) * ann_factor)


def yang_zhang_vol(opens: list, highs: list, lows: list, closes: list,
                    ann_factor: float = 252., k: float = 0.34) -> float:
    """
    Yang-Zhang (2000) 波动率估计量（最优 OHLC，含隔夜跳跃修正）。

    Yang-Zhang 是处理隔夜跳跃的最优无偏估计量：

    σ²_YZ = σ²_overnight + k·σ²_open_close + (1-k)·σ²_RS

    其中：
    σ²_overnight = 隔夜收益方差（close → next open）
    σ²_open_close = 开收益方差（open → close）
    σ²_RS         = Rogers-Satchell 日内方差
    k = 0.34/(1 + (n+1)/(n-1))  ← 最优权重（默认 0.34）

    Yang-Zhang 同时捕获：
    1. 隔夜跳跃（gaps）
    2. 日内价格运动
    3. 开盘价偏斜（Garman-Klass 的弱点）
    """
    n_obs = len(closes)
    if n_obs < 2:
        raise ValueError("Yang-Zhang 需要至少 2 天数据")

    # 最优权重
    k_opt = 0.34 / (1. + (n_obs + 1.) / (n_obs - 1.))
    if k != 0.34:
        k_opt = k  # 使用用户自定义 k

    # 隔夜收益：ln(Oᵢ / Cᵢ₋₁)
    overnight_rets = [log(opens[i]/closes[i-1]) for i in range(1, n_obs)]
    mean_overnight = sum(overnight_rets) / len(overnight_rets)
    var_overnight  = sum((r - mean_overnight)**2 for r in overnight_rets) / (n_obs - 1)

    # 开收日内收益：ln(Cᵢ / Oᵢ)
    open_close_rets = [log(closes[i]/opens[i]) for i in range(n_obs)]
    mean_oc = sum(open_close_rets) / n_obs
    var_oc  = sum((r - mean_oc)**2 for r in open_close_rets) / (n_obs - 1)

    # Rogers-Satchell 日内方差
    rs_sum = 0.
    for O, H, L, C in zip(opens, highs, lows, closes):
        rs_sum += (log(H/C)*log(H/O) + log(L/C)*log(L/O))
    var_rs = rs_sum / n_obs

    # 组合
    var_yz = var_overnight + k_opt * var_oc + (1. - k_opt) * var_rs
    return sqrt(max(var_yz, 0.) * ann_factor)


# ═══════════════════════════════════════════════════════════════
# B. EWMA（指数加权移动平均）
# ═══════════════════════════════════════════════════════════════

def ewma_volatility(returns: list, lam: float = 0.94,
                     ann_factor: float = 252.) -> list:
    """
    EWMA 波动率估计（J.P. Morgan RiskMetrics 模型）。

    递推公式：
    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    其中 λ（lambda）是衰减因子：
    - λ = 0.94：日数据（RiskMetrics 标准）
    - λ = 0.97：月数据
    - 等效半衰期 t_{1/2} = ln(0.5) / ln(λ)

    优点：
    - 更新简单（只需前一步方差和当前收益）
    - 对近期数据赋予更高权重
    - 自动"遗忘"旧数据（无固定窗口长度问题）

    缺点：
    - 不能均值回归（方差可以无限增长）
    - λ 需要外生设定

    参数
    ----
    returns    : 对数收益率序列
    lam        : 衰减因子 λ（0 < λ < 1）
    ann_factor : 年化因子

    返回
    ----
    list : 每个时间点的年化 EWMA 波动率序列
    """
    n = len(returns)
    if n < 2:
        raise ValueError("需要至少 2 个收益率")

    # 初始方差 = 历史收益率的方差
    sigma2 = sum(r**2 for r in returns) / n

    sigma2_series = []
    for r in returns:
        sigma2 = lam * sigma2 + (1. - lam) * r**2
        sigma2_series.append(sqrt(sigma2 * ann_factor))

    halflife = log(0.5) / log(lam)
    return sigma2_series


def ewma_correlation(returns1: list, returns2: list,
                      lam: float = 0.94) -> list:
    """
    EWMA 相关系数估计。

    递推公式（协方差-方差同步更新）：
    Cov_t  = λ·Cov_{t-1}  + (1-λ)·r1_{t-1}·r2_{t-1}
    Var1_t = λ·Var1_{t-1} + (1-λ)·r1²_{t-1}
    Var2_t = λ·Var2_{t-1} + (1-λ)·r2²_{t-1}
    ρ_t    = Cov_t / √(Var1_t · Var2_t)

    参数
    ----
    返回：EWMA 相关系数时间序列
    """
    n = min(len(returns1), len(returns2))
    cov12 = sum(r1*r2 for r1, r2 in zip(returns1, returns2)) / n
    var1  = sum(r**2 for r in returns1) / n
    var2  = sum(r**2 for r in returns2) / n

    rho_series = []
    for r1, r2 in zip(returns1, returns2):
        cov12 = lam * cov12 + (1. - lam) * r1 * r2
        var1  = lam * var1  + (1. - lam) * r1**2
        var2  = lam * var2  + (1. - lam) * r2**2
        denom = sqrt(var1 * var2)
        rho   = cov12 / denom if denom > 1e-12 else 0.
        rho_series.append(max(-1., min(1., rho)))

    return rho_series


# ═══════════════════════════════════════════════════════════════
# C. GARCH(1,1) 模型
# ═══════════════════════════════════════════════════════════════

def garch11_fit(returns: list,
                init_params: tuple = None) -> dict:
    """
    GARCH(1,1) 参数估计（最大似然估计，MLE）。

    模型：
    r_t = σ_t · ε_t,   ε_t ~ N(0,1)
    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    约束：ω > 0, α ≥ 0, β ≥ 0, α+β < 1

    对数似然函数：
    ℓ = -½ Σ [ln(σ²_t) + r²_t/σ²_t] + 常数

    参数
    ----
    init_params : (ω₀, α₀, β₀) 初始猜测（若 None 自动设定）

    返回
    ----
    dict : {omega, alpha, beta, long_run_var, persistence,
            log_likelihood, aic, bic}
    """
    from scipy.optimize import minimize

    n = len(returns)
    ret_arr = np.array(returns)

    # 初始猜测
    if init_params is None:
        var0 = float(np.var(ret_arr))
        omega0 = var0 * 0.10
        alpha0 = 0.10
        beta0  = 0.85
    else:
        omega0, alpha0, beta0 = init_params

    def neg_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        sigma2 = np.var(ret_arr)  # 初始方差
        log_lik = 0.
        for r in ret_arr:
            sigma2 = omega + alpha * r**2 + beta * sigma2
            if sigma2 <= 0:
                return 1e10
            log_lik += -0.5 * (log(sigma2) + r**2 / sigma2)

        return -log_lik  # 最小化负对数似然

    # 优化
    result = minimize(neg_log_likelihood,
                      [omega0, alpha0, beta0],
                      method='L-BFGS-B',
                      bounds=[(1e-10, None), (0., 0.999), (0., 0.999)],
                      options={'ftol': 1e-10, 'gtol': 1e-8})

    omega, alpha, beta = result.x
    persistence = alpha + beta
    long_run_var = omega / (1. - persistence) if persistence < 1 else float('inf')

    # 计算 AIC/BIC
    log_lik = -result.fun
    k = 3  # 参数个数
    aic = 2*k - 2*log_lik
    bic = k*log(n) - 2*log_lik

    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'persistence': persistence,
        'long_run_vol': sqrt(long_run_var * 252.),  # 年化长期波动率
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'converged': result.success,
    }


def garch11_forecast(omega: float, alpha: float, beta: float,
                      last_vol: float, last_ret: float,
                      horizon: int = 10,
                      ann_factor: float = 252.) -> list:
    """
    GARCH(1,1) 波动率预测（多步前向）。

    h 步超前预测（方差）：
    E[σ²_{t+h}] = σ²_∞ + (α+β)^{h-1} · (σ²_{t+1} - σ²_∞)
    其中 σ²_∞ = ω/(1-α-β)（长期均值，均值回归）

    σ²_{t+1} = ω + α·r²_t + β·σ²_t（下一步确定性预测）

    参数
    ----
    last_vol : 最新的日波动率（非年化，如 0.01 = 1%/天）
    last_ret : 最新的日收益率（如 -0.02 = -2%/天）

    返回
    ----
    list : 未来 horizon 步的年化波动率预测序列
    """
    sigma2_last = last_vol**2   # 注意：输入是日波动率，需要平方
    persistence = alpha + beta
    long_run_var = omega / (1. - persistence) if persistence < 1. else sigma2_last

    # 下一步预测
    sigma2_next = omega + alpha * last_ret**2 + beta * sigma2_last

    forecasts = []
    for h in range(1, horizon + 1):
        if h == 1:
            sigma2_h = sigma2_next
        else:
            sigma2_h = long_run_var + persistence**(h-1) * (sigma2_next - long_run_var)

        vol_annual = sqrt(max(sigma2_h, 0.) * ann_factor)
        forecasts.append(vol_annual)

    return forecasts


# ═══════════════════════════════════════════════════════════════
# D. 隐含波动率计算
# ═══════════════════════════════════════════════════════════════

def implied_vol_newton(market_price: float, S: float, K: float, T: float,
                        r: float, b: float,
                        option_type: str = 'call',
                        tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Newton-Raphson 方法求隐含波动率。

    目标：找 σ 使得 BSM(S, K, T, r, b, σ) = market_price

    更新公式：
    σ_{n+1} = σ_n - [BSM(σ_n) - price] / Vega(σ_n)

    Vega = ∂BSM/∂σ = S·e^{(b-r)T}·n(d₁)·√T

    参数
    ----
    tol      : 收敛容差
    max_iter : 最大迭代次数

    返回
    ----
    float : 隐含波动率（若不收敛返回 nan）
    """
    if T <= 0 or market_price <= 0:
        return float('nan')

    # 检查价格范围
    intrinsic = max(S*exp((b-r)*T) - K*exp(-r*T), 0.) if option_type == 'call' \
                else max(K*exp(-r*T) - S*exp((b-r)*T), 0.)
    if market_price < intrinsic - 1e-6:
        return float('nan')

    # 初始猜测（Brenner-Subrahmanyam 近似）
    sigma = sqrt(2.*pi/T) * market_price / S

    for _ in range(max_iter):
        if sigma <= 0:
            sigma = 1e-6

        d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        cf = exp((b-r)*T); df = exp(-r*T)

        if option_type.lower() == 'call':
            price_model = S*cf*N(d1) - K*df*N(d2)
        else:
            price_model = K*df*N(-d2) - S*cf*N(-d1)

        # BSM Vega
        vega = S * cf * n(d1) * sqrt(T)
        if abs(vega) < 1e-12:
            break

        diff = price_model - market_price
        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega
        sigma = max(sigma, 1e-6)   # 保证非负

    return sigma if abs(sigma) < 10. else float('nan')


def vol_surface(S: float, T_list: list, K_list: list,
                 r: float, b: float,
                 price_matrix: list,
                 option_type: str = 'call') -> list:
    """
    构建隐含波动率曲面。

    输入一个市场期权价格矩阵，输出对应的隐含波动率矩阵。

    参数
    ----
    T_list       : 到期时间列表（行）
    K_list       : 行权价列表（列）
    price_matrix : market_price[i][j]，T_list[i] 和 K_list[j] 对应的期权价格

    返回
    ----
    list : iv_surface[i][j] — 对应的隐含波动率
    """
    iv_surface = []
    for i, T in enumerate(T_list):
        iv_row = []
        for j, K in enumerate(K_list):
            price = price_matrix[i][j]
            iv = implied_vol_newton(price, S, K, T, r, b, option_type)
            iv_row.append(iv)
        iv_surface.append(iv_row)
    return iv_surface


# ═══════════════════════════════════════════════════════════════
# E. SVI 波动率参数化（Gatheral 2004）
# ═══════════════════════════════════════════════════════════════

def svi_implied_vol(k: float,
                     a: float, b: float, rho: float,
                     m: float, sigma: float) -> float:
    """
    SVI（Stochastic Volatility Inspired）隐含总方差参数化。
    Gatheral (2004) 提出，在利率衍生品和股权市场广泛使用。

    SVI 参数化总方差 w(k) = σ²(k)·T：
    w(k) = a + b·[ρ·(k-m) + √((k-m)² + σ²)]

    其中：
    k   = ln(K/F)（对数执行比）
    a   = 整体水平（控制整体波动率高低）
    b   = 斜率/展宽（控制微笑坡度）
    ρ   = 偏度（-1 < ρ < 1，控制微笑不对称性）
    m   = 最小点（控制微笑水平位置，类似 ATM）
    σ   = 曲率参数（控制微笑宽度/峰度）

    无套利条件（Durrleman 条件）：
    b(1+|ρ|) ≤ 4 时保证无蝴蝶套利

    参数
    ----
    k = ln(K/F)（对数执行比），ATM 时 k=0
    a, b, rho, m, sigma : SVI 参数

    返回
    ----
    float : 隐含总方差 w(k)
    """
    return a + b * (rho*(k - m) + sqrt((k - m)**2 + sigma**2))


def svi_calibrate(log_moneyness: list, total_variance: list,
                   weights: list = None) -> dict:
    """
    校准 SVI 参数（加权最小二乘）。

    输入：市场隐含总方差（=σ²·T）和对应的对数执行比
    输出：最优 SVI 参数 (a, b, ρ, m, σ)

    参数
    ----
    log_moneyness  : [ln(K₁/F), ln(K₂/F), ...] 列表
    total_variance : [σ₁²T, σ₂²T, ...] 市场总方差列表
    weights        : 拟合权重（若 None 则等权）
    """
    from scipy.optimize import minimize, differential_evolution

    k_arr = np.array(log_moneyness)
    w_mkt = np.array(total_variance)
    wt = np.ones(len(k_arr)) if weights is None else np.array(weights)

    def objective(params):
        a, b, rho, m, sig = params
        if b < 0 or sig <= 0 or not (-1 < rho < 1):
            return 1e10
        if b * (1 + abs(rho)) > 4:
            return 1e10  # 违反 Durrleman 无套利条件
        w_svi = np.array([svi_implied_vol(k, a, b, rho, m, sig) for k in k_arr])
        if np.any(w_svi <= 0):
            return 1e10
        return float(np.sum(wt * (w_svi - w_mkt)**2))

    # 全局优化（差分进化）
    bounds = [(0., max(w_mkt)*2),   # a
              (0., 2.),              # b
              (-0.999, 0.999),       # rho
              (min(k_arr), max(k_arr)),  # m
              (0.001, 2.)]           # sigma

    try:
        result = differential_evolution(objective, bounds,
                                         seed=42, maxiter=500, tol=1e-8)
        a, b, rho, m, sig = result.x
    except Exception:
        a, b, rho, m, sig = 0.04, 0.10, -0.30, 0., 0.20

    return {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sig,
            'fun': result.fun if 'result' in dir() else float('nan')}


# ═══════════════════════════════════════════════════════════════
# F. 波动率期限结构（Term Structure）与远期波动率
# ═══════════════════════════════════════════════════════════════

def forward_volatility(sigma_T1: float, T1: float,
                        sigma_T2: float, T2: float) -> float:
    """
    远期波动率（Forward Volatility）。

    从 T1 到 T2 的远期波动率（通过方差插值）：
    σ²_{T1→T2} = (σ²_{T2}·T2 - σ²_{T1}·T1) / (T2 - T1)

    直觉：总方差 = σ²·T 可以分解为两段的方差之和：
    σ²_{T2}·T2 = σ²_{T1}·T1 + σ²_{T1→T2}·(T2-T1)

    参数
    ----
    sigma_T1 : [0, T1] 的年化波动率（总或平均）
    T1       : 第一到期时间
    sigma_T2 : [0, T2] 的年化波动率
    T2       : 第二到期时间（T2 > T1）

    返回
    ----
    float : [T1, T2] 区间的远期年化波动率
    """
    if T2 <= T1:
        raise ValueError("T2 必须大于 T1")
    var_T1 = sigma_T1**2 * T1
    var_T2 = sigma_T2**2 * T2
    var_fwd = (var_T2 - var_T1) / (T2 - T1)
    if var_fwd < 0:
        raise ValueError(f"远期方差为负（{var_fwd:.4f}）：波动率期限结构违反无套利条件")
    return sqrt(var_fwd)


def vix_style_variance(S: float, T: float, r: float,
                        put_strikes: list, put_prices: list,
                        call_strikes: list, call_prices: list,
                        K0: float = None) -> float:
    """
    VIX 式隐含方差计算（CBOE VIX 方法论）。

    CBOE VIX 公式（连续积分的离散化）：
    σ²_VIX = (2/T) · Σ [ΔKᵢ/Kᵢ²] · e^{rT} · Q(Kᵢ) - (1/T)·(F/K₀ - 1)²

    其中：
    Q(Kᵢ) = OTM 期权价格（K < F 用看跌，K > F 用看涨）
    K₀    = 最接近 F（远期价格）的行权价
    ΔKᵢ   = 相邻行权价的间距（梯形积分）
    F     = 由看涨-看跌平价推算的远期价格

    参数
    ----
    put_strikes/put_prices   : OTM 看跌期权（K < F）
    call_strikes/call_prices : OTM 看涨期权（K > F）
    K0                       : 最近 ATM 行权价（若 None 自动估算）
    """
    F = S * exp(r * T)  # 远期价格（简化，实际从 put-call 平价推导）

    if K0 is None:
        # 找最接近 F 的行权价
        all_K = sorted(put_strikes + call_strikes)
        K0 = min(all_K, key=lambda K: abs(K - F))

    # 合并 OTM 期权（去除重复的 ATM）
    options = []
    for K, P in zip(put_strikes, put_prices):
        if K <= K0:
            options.append((K, P))
    for K, C in zip(call_strikes, call_prices):
        if K >= K0:
            options.append((K, C))

    options.sort()
    if not options:
        return float('nan')

    # 计算 ΔK（边界使用单侧差分）
    var_sum = 0.
    for i, (K, Q) in enumerate(options):
        if i == 0:
            dK = options[1][0] - options[0][0]
        elif i == len(options) - 1:
            dK = options[-1][0] - options[-2][0]
        else:
            dK = (options[i+1][0] - options[i-1][0]) / 2.

        var_sum += dK / K**2 * Q

    var_sum *= exp(r*T)

    # ATM 修正项（F/K₀ - 1)²
    atm_correction = (F/K0 - 1.)**2

    sigma2_vix = (2./T) * var_sum - (1./T) * atm_correction
    return max(sigma2_vix, 0.)


# ═══════════════════════════════════════════════════════════════
# G. 波动率锥（Volatility Cone）
# ═══════════════════════════════════════════════════════════════

def volatility_cone(price_history: list, windows: list = None,
                     ann_factor: float = 252.,
                     quantiles: list = None) -> dict:
    """
    波动率锥（Volatility Cone）分析。

    波动率锥展示不同历史窗口下的滚动波动率分布，
    用于判断当前期权隐含波动率是否被高估或低估。

    对于每个窗口长度 w：
    1. 计算所有滚动窗口的历史波动率
    2. 统计分布的分位数（如第 10、25、50、75、90 百分位）

    实践中：
    - 若当前 IV > 历史波动率的第 75 百分位：IV 偏高（卖期权）
    - 若当前 IV < 历史波动率的第 25 百分位：IV 偏低（买期权）

    参数
    ----
    price_history : 历史价格序列（足够长，如 2-5 年日数据）
    windows       : 滚动窗口长度列表（如 [21, 42, 63, 126, 252]）
    quantiles     : 分位数列表（默认 [0.10, 0.25, 0.50, 0.75, 0.90]）

    返回
    ----
    dict : {window_size: {quantile: vol_value, ...}, ...}
    """
    if windows is None:
        windows = [10, 21, 42, 63, 126, 252]
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    # 计算对数收益率
    log_rets = [log(price_history[i]/price_history[i-1])
                for i in range(1, len(price_history))]

    cone = {}
    for w in windows:
        if w >= len(log_rets):
            continue
        # 滚动窗口历史波动率
        rolling_vols = []
        for start in range(len(log_rets) - w + 1):
            window_rets = log_rets[start:start+w]
            mean_r = sum(window_rets) / w
            var_w  = sum((r - mean_r)**2 for r in window_rets) / (w - 1)
            rolling_vols.append(sqrt(var_w * ann_factor))

        rolling_vols.sort()
        n_vols = len(rolling_vols)

        cone[w] = {}
        for q in quantiles:
            idx = min(int(q * n_vols), n_vols - 1)
            cone[w][f'p{int(q*100)}'] = rolling_vols[idx]

        cone[w]['current'] = rolling_vols[-1]  # 最近窗口的波动率
        cone[w]['min'] = rolling_vols[0]
        cone[w]['max'] = rolling_vols[-1]

    return cone


if __name__ == "__main__":
    print("=" * 65)
    print("波动率估计方法 — 数值示例（Haug Chapter 12）")
    print("=" * 65)

    # ── 模拟价格数据（生成测试用途）──────────────────────────
    np.random.seed(42)
    n_days = 252
    S0 = 100.
    true_vol = 0.20
    dt = 1./252.

    # 生成 GBM 价格路径（含日内 OHLC）
    prices = [S0]
    for _ in range(n_days):
        S_prev = prices[-1]
        # 日末价格
        r_day = (0.05 - 0.5*true_vol**2)*dt + true_vol*sqrt(dt)*np.random.randn()
        prices.append(S_prev * exp(r_day))

    closes = prices[1:]
    opens  = prices[:-1]   # 简化：开盘 = 前收盘（无隔夜跳跃）
    highs  = [max(o, c) * (1 + abs(np.random.randn() * true_vol * sqrt(dt) * 0.5))
               for o, c in zip(opens, closes)]
    lows   = [min(o, c) * (1 - abs(np.random.randn() * true_vol * sqrt(dt) * 0.5))
               for o, c in zip(opens, closes)]

    print(f"\n真实波动率（模拟）= {true_vol:.2%}，样本长度 = {n_days} 天")

    # ── 历史波动率估计量对比 ──────────────────────────────────
    print(f"\n历史波动率估计量对比：")
    c2c = close_to_close_vol(closes)
    park = parkinson_vol(highs, lows)
    gk   = garman_klass_vol(opens, highs, lows, closes)
    rs   = rogers_satchell_vol(opens, highs, lows, closes)
    yz   = yang_zhang_vol(opens, highs, lows, closes)

    for name, vol in [("收盘价法 (C2C)", c2c), ("Parkinson (HL)", park),
                       ("Garman-Klass (OHLC)", gk), ("Rogers-Satchell", rs),
                       ("Yang-Zhang", yz)]:
        bias = vol - true_vol
        print(f"  {name:<24} = {vol:.4f} ({vol:.2%})  偏差: {bias:+.4f}")

    # ── EWMA 波动率 ───────────────────────────────────────────
    log_rets = [log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    ewma_vols = ewma_volatility(log_rets, lam=0.94)
    print(f"\nEWMA 波动率（λ=0.94）：")
    print(f"  最终估计 = {ewma_vols[-1]:.4f} ({ewma_vols[-1]:.2%})")
    print(f"  最大值   = {max(ewma_vols):.4f} ({max(ewma_vols):.2%})")
    print(f"  最小值   = {min(ewma_vols):.4f} ({min(ewma_vols):.2%})")

    # ── GARCH(1,1) ────────────────────────────────────────────
    print(f"\nGARCH(1,1) 参数估计：")
    try:
        g = garch11_fit(log_rets)
        print(f"  ω = {g['omega']:.6f}")
        print(f"  α = {g['alpha']:.4f}  （ARCH 效应，新息冲击权重）")
        print(f"  β = {g['beta']:.4f}   （GARCH 效应，波动率持续性）")
        print(f"  α+β = {g['persistence']:.4f}  （持续性，<1 保证平稳）")
        print(f"  长期年化波动率 = {g['long_run_vol']:.4f} ({g['long_run_vol']:.2%})")
        print(f"  AIC = {g['aic']:.2f},  收敛: {g['converged']}")

        # GARCH 预测
        forecasts = garch11_forecast(g['omega'], g['alpha'], g['beta'],
                                      ewma_vols[-1]/sqrt(252.), log_rets[-1], horizon=10)
        print(f"\n  GARCH(1,1) 未来 10 天波动率预测（年化）：")
        for h, vf in enumerate(forecasts, 1):
            print(f"    h={h:>2} 天: {vf:.4f} ({vf:.2%})")
    except Exception as e:
        print(f"  [GARCH 计算: {e}]")

    # ── 隐含波动率 ────────────────────────────────────────────
    print(f"\n隐含波动率（Newton-Raphson 反推）：")
    from utils.common import norm_cdf as Nf
    S0_iv, K_iv, T_iv, r_iv, b_iv, sigma_iv = 100., 100., 0.5, 0.10, 0.10, 0.20
    d1 = (log(S0_iv/K_iv)+(b_iv+0.5*sigma_iv**2)*T_iv)/(sigma_iv*sqrt(T_iv))
    d2 = d1-sigma_iv*sqrt(T_iv)
    mkt_price = S0_iv*exp((b_iv-r_iv)*T_iv)*Nf(d1) - K_iv*exp(-r_iv*T_iv)*Nf(d2)
    iv = implied_vol_newton(mkt_price, S0_iv, K_iv, T_iv, r_iv, b_iv)
    print(f"  BSM 理论价格 = {mkt_price:.4f}（σ={sigma_iv:.2%}）")
    print(f"  反推隐含波动率 = {iv:.6f}（误差: {iv-sigma_iv:.2e}）")

    # ── 远期波动率 ────────────────────────────────────────────
    print(f"\n远期波动率（波动率期限结构）：")
    t1_vol, t2_vol = 0.18, 0.22
    T1, T2 = 0.25, 0.50
    fwd_vol = forward_volatility(t1_vol, T1, t2_vol, T2)
    print(f"  σ(0→{T1})={t1_vol:.2%}, σ(0→{T2})={t2_vol:.2%}")
    print(f"  远期波动率 σ({T1}→{T2}) = {fwd_vol:.4f} ({fwd_vol:.2%})")
    print(f"  验证: {t1_vol**2*T1:.6f} + {fwd_vol**2*(T2-T1):.6f} = {t2_vol**2*T2:.6f}")
