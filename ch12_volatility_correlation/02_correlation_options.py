"""
第12章：相关性估计与相关性期权
Chapter 12: Correlation Estimation and Correlation Options

本文件实现：
1. 历史相关性估计（滚动窗口 / EWMA）
2. DCC-GARCH 动态条件相关性模型（Engle 2002）
3. 隐含相关性（从指数与成分股期权中提取）
4. 相关性互换（Correlation Swap）定价
5. 最差表现期权（Worst-of / Best-of）中的相关性敏感性
6. 色散交易（Dispersion Trade）盈亏分析
7. 相关性的期限结构插值

参考：
  - Engle (2002) "Dynamic Conditional Correlation" JBES
  - Haug (2007) "The Complete Guide to Option Pricing Formulas" Ch.12
  - Demeterfi et al (1999) variance swap replication
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

from utils.common import norm_cdf, norm_pdf, cbnd


# ============================================================
# 1. 历史相关性估计
# ============================================================

def historical_correlation(returns1: np.ndarray,
                            returns2: np.ndarray,
                            window: int = None) -> float | np.ndarray:
    """
    历史相关性估计（滚动窗口 Pearson 相关系数）

    参数说明：
        returns1, returns2 : 两个资产的对数收益率序列
        window             : 滚动窗口长度（None = 全样本）

    数学公式：
        ρ = Cov(r1, r2) / (σ₁ · σ₂)
        Cov = Σ(r1_t - μ1)(r2_t - μ2) / (n-1)

    返回：
        window=None → 单一相关系数标量
        window=N   → 长度为 len-N+1 的滚动相关序列
    """
    r1 = np.asarray(returns1, dtype=float)
    r2 = np.asarray(returns2, dtype=float)

    if window is None:
        # 全样本 Pearson 相关
        rho = np.corrcoef(r1, r2)[0, 1]
        return rho
    else:
        n = len(r1)
        if window > n:
            raise ValueError("窗口大小不能超过样本长度")
        roll_corr = np.full(n - window + 1, np.nan)
        for i in range(n - window + 1):
            sub1 = r1[i: i + window]
            sub2 = r2[i: i + window]
            roll_corr[i] = np.corrcoef(sub1, sub2)[0, 1]
        return roll_corr


def ewma_correlation_series(returns1: np.ndarray,
                             returns2: np.ndarray,
                             lam: float = 0.94) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EWMA 动态相关性估计（RiskMetrics 方法）

    递推公式：
        Var1_t   = λ·Var1_{t-1} + (1-λ)·r1_{t-1}²
        Var2_t   = λ·Var2_{t-1} + (1-λ)·r2_{t-1}²
        Cov12_t  = λ·Cov12_{t-1} + (1-λ)·r1_{t-1}·r2_{t-1}
        ρ_t      = Cov12_t / √(Var1_t · Var2_t)

    参数说明：
        lam : EWMA 衰减因子（通常 0.94 日频）

    返回：
        (ewma_rho, ewma_var1, ewma_var2) — 各时间步的值
    """
    r1 = np.asarray(returns1, dtype=float)
    r2 = np.asarray(returns2, dtype=float)
    n = len(r1)

    var1 = np.zeros(n)
    var2 = np.zeros(n)
    cov12 = np.zeros(n)
    rho = np.zeros(n)

    # 用第一个观测值初始化
    var1[0] = r1[0] ** 2
    var2[0] = r2[0] ** 2
    cov12[0] = r1[0] * r2[0]

    for t in range(1, n):
        var1[t] = lam * var1[t - 1] + (1 - lam) * r1[t - 1] ** 2
        var2[t] = lam * var2[t - 1] + (1 - lam) * r2[t - 1] ** 2
        cov12[t] = lam * cov12[t - 1] + (1 - lam) * r1[t - 1] * r2[t - 1]
        denom = np.sqrt(var1[t] * var2[t])
        rho[t] = cov12[t] / denom if denom > 1e-12 else 0.0

    return rho, var1, var2


# ============================================================
# 2. DCC-GARCH 动态条件相关性（Engle 2002）
# ============================================================

def dcc_garch_fit(returns1: np.ndarray,
                  returns2: np.ndarray,
                  a_init: float = 0.05,
                  b_init: float = 0.90) -> dict:
    """
    DCC-GARCH(1,1) 两步估计（Engle 2002）

    第一步：对每个序列分别拟合 GARCH(1,1)，得到标准化残差 ε_t
    第二步：对标准化残差拟合 DCC 参数 (a, b)

    DCC 递推（Q 矩阵）：
        Q_t = (1 - a - b)·Q̄ + a·ε_{t-1}·ε_{t-1}' + b·Q_{t-1}
        R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

    其中 Q̄ = 无条件相关矩阵 E[ε_t ε_t']

    参数说明：
        a_init, b_init : DCC 参数初始值

    返回：
        dict 含 DCC 参数 (a, b)、时变相关序列 rho_t、标准化残差
    """
    r1 = np.asarray(returns1, dtype=float)
    r2 = np.asarray(returns2, dtype=float)
    n = len(r1)

    # ── 第一步：各自 GARCH(1,1) 拟合 ──────────────────────────────
    def garch11_loglik(params, returns):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        h = np.var(returns)
        ll = 0.0
        for t in range(1, len(returns)):
            h = omega + alpha * returns[t - 1] ** 2 + beta * h
            if h <= 0:
                return 1e10
            ll += 0.5 * (np.log(h) + returns[t] ** 2 / h)
        return ll

    res1 = minimize(garch11_loglik, [1e-6, 0.05, 0.90], args=(r1,),
                    method='L-BFGS-B',
                    bounds=[(1e-8, None), (0, 0.5), (0, 0.9999)])
    res2 = minimize(garch11_loglik, [1e-6, 0.05, 0.90], args=(r2,),
                    method='L-BFGS-B',
                    bounds=[(1e-8, None), (0, 0.5), (0, 0.9999)])

    def garch11_variance_series(params, returns):
        omega, alpha, beta = params
        h_series = np.zeros(len(returns))
        h_series[0] = np.var(returns)
        for t in range(1, len(returns)):
            h_series[t] = omega + alpha * returns[t - 1] ** 2 + beta * h_series[t - 1]
        return np.maximum(h_series, 1e-12)

    h1 = garch11_variance_series(res1.x, r1)
    h2 = garch11_variance_series(res2.x, r2)

    # 标准化残差
    eps1 = r1 / np.sqrt(h1)
    eps2 = r2 / np.sqrt(h2)

    # ── 第二步：DCC 参数估计 ──────────────────────────────────────
    Q_bar = np.array([[np.mean(eps1 ** 2), np.mean(eps1 * eps2)],
                      [np.mean(eps1 * eps2), np.mean(eps2 ** 2)]])

    def dcc_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(1, n):
            eps_vec = np.array([eps1[t - 1], eps2[t - 1]])
            Q = (1 - a - b) * Q_bar + a * np.outer(eps_vec, eps_vec) + b * Q
            # 标准化 Q → R
            sqrt_diag = np.sqrt(np.diag(Q))
            R = Q / np.outer(sqrt_diag, sqrt_diag)
            rho_t = R[0, 1]
            det_R = max(1 - rho_t ** 2, 1e-10)
            # 条件 log-likelihood 贡献
            ll += 0.5 * (np.log(det_R) +
                         (eps1[t] ** 2 - 2 * rho_t * eps1[t] * eps2[t] + eps2[t] ** 2) / det_R -
                         eps1[t] ** 2 - eps2[t] ** 2)
        return ll

    res_dcc = minimize(dcc_loglik, [a_init, b_init],
                       method='L-BFGS-B',
                       bounds=[(0.001, 0.3), (0.5, 0.999)])
    a_opt, b_opt = res_dcc.x

    # 重建时变相关序列
    Q = Q_bar.copy()
    rho_t = np.zeros(n)
    for t in range(1, n):
        eps_vec = np.array([eps1[t - 1], eps2[t - 1]])
        Q = (1 - a_opt - b_opt) * Q_bar + a_opt * np.outer(eps_vec, eps_vec) + b_opt * Q
        sqrt_diag = np.sqrt(np.diag(Q))
        R = Q / np.outer(sqrt_diag, sqrt_diag)
        rho_t[t] = np.clip(R[0, 1], -0.9999, 0.9999)

    return {
        'a': a_opt,
        'b': b_opt,
        'rho_t': rho_t,
        'eps1': eps1,
        'eps2': eps2,
        'garch1_params': res1.x,
        'garch2_params': res2.x,
        'h1': h1,
        'h2': h2,
    }


def dcc_forecast(dcc_result: dict, h_ahead: int = 5) -> np.ndarray:
    """
    DCC-GARCH 相关性多步预测

    均值回归公式（条件期望）：
        E[ρ_{t+h}] = ρ_∞ + (a+b)^{h-1} · (ρ_{t+1} - ρ_∞)

    其中 ρ_∞ = Q̄[0,1] / √(Q̄[0,0]·Q̄[1,1])（无条件长期相关）

    参数说明：
        dcc_result : dcc_garch_fit 的返回值
        h_ahead    : 预测步数

    返回：
        长度为 h_ahead 的预测相关序列
    """
    a = dcc_result['a']
    b = dcc_result['b']
    rho_t = dcc_result['rho_t']

    # 无条件（长期）相关 — 从标准化残差估计
    eps1 = dcc_result['eps1']
    eps2 = dcc_result['eps2']
    rho_inf = np.mean(eps1 * eps2) / np.sqrt(np.mean(eps1 ** 2) * np.mean(eps2 ** 2))
    rho_inf = np.clip(rho_inf, -0.9999, 0.9999)

    rho_current = rho_t[-1]
    forecasts = np.zeros(h_ahead)
    for h in range(1, h_ahead + 1):
        forecasts[h - 1] = rho_inf + (a + b) ** (h - 1) * (rho_current - rho_inf)

    return forecasts


# ============================================================
# 3. 隐含相关性（从指数与成分股期权提取）
# ============================================================

def implied_correlation_index(index_vol: float,
                               component_vols: np.ndarray,
                               weights: np.ndarray) -> float:
    """
    从指数隐含波动率与成分股隐含波动率估算隐含相关性

    推导：
        σ²_index = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ

    假设所有成分股之间的相关系数相等（pairwise equal correlation ρ）：
        σ²_index = ρ · (Σᵢ wᵢ σᵢ)² + (1-ρ) · Σᵢ wᵢ² σᵢ²

    求解 ρ：
        ρ = (σ²_index - Σᵢ wᵢ² σᵢ²) / [(Σᵢ wᵢ σᵢ)² - Σᵢ wᵢ² σᵢ²]

    参数说明：
        index_vol       : 指数隐含波动率（年化）
        component_vols  : 各成分股隐含波动率数组
        weights         : 各成分股权重（求和为1）

    返回：
        隐含相关系数 ρ ∈ (-1, 1)
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(component_vols, dtype=float)

    # 规范化权重
    w = w / w.sum()

    sigma_index_sq = index_vol ** 2
    weighted_sum_sq = np.sum((w * sigma) ** 2)          # Σ wᵢ² σᵢ²
    weighted_sum_cross = (np.dot(w, sigma)) ** 2        # (Σ wᵢ σᵢ)²

    denominator = weighted_sum_cross - weighted_sum_sq
    if abs(denominator) < 1e-12:
        return 1.0  # 所有成分波动率相等时退化

    rho = (sigma_index_sq - weighted_sum_sq) / denominator
    return float(np.clip(rho, -0.9999, 0.9999))


def implied_correlation_two_assets(S1: float, S2: float,
                                    K1: float, K2: float,
                                    T: float,
                                    r: float, b: float,
                                    sigma1: float, sigma2: float,
                                    spread_price: float,
                                    option_type: str = 'call') -> float:
    """
    从双资产价差期权市场价格反推隐含相关系数

    价差期权（Margrabe 公式）：
        C_spread = BSM(S1, S2, K=K2-K1, σ_eff)
        σ²_eff = σ₁² + σ₂² - 2ρσ₁σ₂

    二分法（Brent）反推 ρ，使得理论价格 = 市场价格

    参数说明：
        S1, S2        : 两个资产当前价格
        K1, K2        : 行权价
        T             : 到期时间（年）
        r             : 无风险利率
        b             : 持有成本
        sigma1,sigma2 : 各资产波动率
        spread_price  : 价差期权市场价格
        option_type   : 'call' or 'put'

    返回：
        隐含相关系数
    """
    from scipy.optimize import brentq

    def spread_price_model(rho):
        sigma_eff = np.sqrt(max(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2, 1e-8))
        d1 = (np.log(S1 / S2) + (b + 0.5 * sigma_eff ** 2) * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        cf = np.exp((b - r) * T)
        df = np.exp(-r * T)
        if option_type.lower() == 'call':
            price = S1 * cf * norm_cdf(d1) - S2 * df * norm_cdf(d2)
        else:
            price = S2 * df * norm_cdf(-d2) - S1 * cf * norm_cdf(-d1)
        return price - spread_price

    try:
        rho_impl = brentq(spread_price_model, -0.9999, 0.9999, xtol=1e-8)
    except ValueError:
        rho_impl = np.nan
    return rho_impl


# ============================================================
# 4. 相关性互换（Correlation Swap）
# ============================================================

def correlation_swap_payoff(realized_corr: float,
                             strike_corr: float,
                             notional: float) -> float:
    """
    相关性互换到期收益

    收益 = Notional × (ρ_realized - ρ_strike)

    相关性互换允许投资者直接买卖平均成对相关性，
    无需承担任何方向性市场风险（delta 中性）

    参数说明：
        realized_corr : 到期时实现的平均成对相关系数
        strike_corr   : 互换约定的相关系数（参考值）
        notional      : 名义本金

    返回：
        互换买方（多相关）的净收益
    """
    return notional * (realized_corr - strike_corr)


def average_pairwise_correlation(returns_matrix: np.ndarray) -> float:
    """
    计算多资产组合的平均成对相关系数

    用于相关性互换的实现值结算

    公式：
        ρ̄ = (2 / (N(N-1))) · Σᵢ<ⱼ ρᵢⱼ

    参数说明：
        returns_matrix : 形状 (T, N) 的收益率矩阵，T=时间步，N=资产数

    返回：
        平均成对相关系数标量
    """
    R = np.corrcoef(returns_matrix.T)  # N×N 相关矩阵
    n = R.shape[0]
    # 提取上三角元素（排除对角线）
    upper_tri = R[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def correlation_swap_fair_strike(returns_matrix: np.ndarray,
                                  window: int = None) -> float:
    """
    相关性互换的公平 strike（历史均值近似）

    在实务中，相关性互换的 fair strike 通常由以下方式估算：
    1. 历史平均成对相关系数（此函数实现）
    2. 由隐含相关性曲面插值得到

    参数说明：
        returns_matrix : (T, N) 收益率矩阵
        window         : 用于估算的历史窗口长度（None=全样本）

    返回：
        公平 strike 相关系数
    """
    if window is not None:
        returns_matrix = returns_matrix[-window:]
    return average_pairwise_correlation(returns_matrix)


# ============================================================
# 5. 色散交易（Dispersion Trade）P&L 分析
# ============================================================

def dispersion_pnl(index_realized_var: float,
                   component_realized_vars: np.ndarray,
                   weights: np.ndarray,
                   vega_index: float,
                   vegas_components: np.ndarray) -> dict:
    """
    色散交易盈亏分解

    色散交易策略：
        - 做空指数方差互换（卖出指数波动率）
        - 做多成分股方差互换（买入个股波动率）

    P&L 分解：
        总收益 = Vega_index × (σ²_index_strike - σ²_index_realized)
                + Σᵢ Vega_i × (σ²_i_realized - σ²_i_strike)

    相关性 P&L：
        σ²_index = ρ̄ · (Σᵢ wᵢ σᵢ)² + (1-ρ̄) · Σᵢ wᵢ² σᵢ²

        若实现相关性 < 隐含相关性 → 色散交易盈利

    参数说明：
        index_realized_var      : 指数已实现方差
        component_realized_vars : 各成分股已实现方差（数组）
        weights                 : 成分股权重
        vega_index              : 指数方差互换 vega（每1方差的美元盈亏）
        vegas_components        : 成分股方差互换 vega 数组

    返回：
        dict 含各项盈亏分解
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    comp_vars = np.asarray(component_realized_vars, dtype=float)
    v_comp = np.asarray(vegas_components, dtype=float)

    # 按权重重建的"预期"指数方差（在某个假设相关性下）
    weighted_var_sum = np.sum(w ** 2 * comp_vars)         # Σ wᵢ² σᵢ²
    weighted_vol_sum_sq = np.dot(w, np.sqrt(comp_vars)) ** 2  # (Σ wᵢ σᵢ)²

    # 指数 P&L（做空指数方差，strike 为 0 时的相对盈亏）
    index_pnl = vega_index * (-index_realized_var)

    # 成分股 P&L（做多各成分股方差互换）
    component_pnl_total = np.sum(v_comp * comp_vars)

    # 相关性贡献：指数方差超出纯权重方差的部分 → 来自相关性
    implied_rho_contribution = index_realized_var - weighted_var_sum
    correlation_effect = implied_rho_contribution / max(weighted_vol_sum_sq - weighted_var_sum, 1e-10)

    return {
        'index_pnl': index_pnl,
        'component_pnl': component_pnl_total,
        'net_pnl': index_pnl + component_pnl_total,
        'realized_correlation_est': float(correlation_effect),
        'index_realized_var': index_realized_var,
        'weighted_component_var': weighted_var_sum,
    }


# ============================================================
# 6. 相关性对最差/最好表现期权的敏感性（Correlation Greeks）
# ============================================================

def best_of_two_option(S1: float, S2: float,
                        K: float, T: float,
                        r: float, b1: float, b2: float,
                        sigma1: float, sigma2: float,
                        rho: float,
                        option_type: str = 'call') -> float:
    """
    双资产最好表现看涨/看跌期权（Stulz 1982）

    定价公式（看涨）：
        C_best = S1·e^{(b1-r)T}·N(d1, e1; ρ1)
               + S2·e^{(b2-r)T}·N(d2, e2; ρ2)
               - K·e^{-rT}·N(-e1, -e2; ρ)
               + Call_Margrabe(S1, S2)

    其中：
        d1 = [ln(S1/K) + (b1+σ1²/2)T] / (σ1√T)
        d2 = [ln(S2/K) + (b2+σ2²/2)T] / (σ2√T)
        e1 = [ln(S1/S2) + (b1-b2+σ12²/2)T] / (σ12√T)，σ12=√(σ1²+σ2²-2ρσ1σ2)
        ρ1 = (σ1-ρσ2)/σ12，ρ2 = (σ2-ρσ1)/σ12

    参数说明：
        S1, S2   : 两个资产现价
        K        : 行权价
        T        : 到期时间
        r        : 无风险利率
        b1, b2   : 各资产持有成本
        sigma1,sigma2 : 各资产波动率
        rho      : 两资产相关系数

    返回：
        期权价格
    """
    sigma12 = np.sqrt(max(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2, 1e-8))

    d1 = (np.log(S1 / K) + (b1 + 0.5 * sigma1 ** 2) * T) / (sigma1 * np.sqrt(T))
    d2 = (np.log(S2 / K) + (b2 + 0.5 * sigma2 ** 2) * T) / (sigma2 * np.sqrt(T))
    e1 = (np.log(S1 / S2) + (b1 - b2 + 0.5 * sigma12 ** 2) * T) / (sigma12 * np.sqrt(T))
    e2 = e1 - sigma12 * np.sqrt(T)

    rho1 = (sigma1 - rho * sigma2) / sigma12
    rho2 = (sigma2 - rho * sigma1) / sigma12

    if option_type.lower() == 'call':
        price = (S1 * np.exp((b1 - r) * T) * cbnd(d1, e1, rho1)
                 + S2 * np.exp((b2 - r) * T) * cbnd(d2, e2, rho2)
                 - K * np.exp(-r * T) * (1 - cbnd(-d1, -d2, rho)))
    else:
        # 最好表现看跌 = 最好表现看涨 - (S1_forward + S2_forward - K_disc - 交换价值)
        # 用 put-call parity 近似
        call = (S1 * np.exp((b1 - r) * T) * cbnd(d1, e1, rho1)
                + S2 * np.exp((b2 - r) * T) * cbnd(d2, e2, rho2)
                - K * np.exp(-r * T) * (1 - cbnd(-d1, -d2, rho)))
        # Best-of put-call parity: C_best - P_best = S1·e^{(b1-r)T} + S2·e^{(b2-r)T} - K·e^{-rT}
        # (对最好者而言不完全成立，此为近似）
        price = call - (S1 * np.exp((b1 - r) * T) + S2 * np.exp((b2 - r) * T) - K * np.exp(-r * T))
    return max(price, 0.0)


def correlation_delta(S1: float, S2: float,
                       K: float, T: float,
                       r: float, b1: float, b2: float,
                       sigma1: float, sigma2: float,
                       rho: float,
                       option_type: str = 'call',
                       drho: float = 0.001) -> float:
    """
    相关性 Delta（期权价格对相关系数的敏感度）

    ∂C/∂ρ ≈ [C(ρ+Δρ) - C(ρ-Δρ)] / (2Δρ)

    对于最好表现期权，相关性越高 → 最好表现价值越低
    对于最差表现期权，相关性越高 → 最差表现价值越低
    （高相关时两个资产趋同，分散化效应降低）

    返回：
        期权对相关系数的一阶偏导
    """
    price_up = best_of_two_option(S1, S2, K, T, r, b1, b2, sigma1, sigma2,
                                   min(rho + drho, 0.9999), option_type)
    price_dn = best_of_two_option(S1, S2, K, T, r, b1, b2, sigma1, sigma2,
                                   max(rho - drho, -0.9999), option_type)
    return (price_up - price_dn) / (2 * drho)


# ============================================================
# 7. 相关性期限结构插值
# ============================================================

def correlation_term_structure(maturities: np.ndarray,
                                correlations: np.ndarray,
                                target_T: float,
                                method: str = 'linear') -> float:
    """
    相关性期限结构插值

    类似波动率期限结构，相关性也存在期限结构：
    短期相关性（市场危机时飙升）vs 长期均值回归

    方差加权插值（保证方差连续性）：
        ρ(T)·T 在期限上线性插值（类似远期方差）

    参数说明：
        maturities   : 已知到期日数组（升序）
        correlations : 对应的相关系数数组
        target_T     : 目标到期日
        method       : 'linear'（线性插值）或 'variance_weighted'

    返回：
        目标到期日的相关系数
    """
    T = np.asarray(maturities, dtype=float)
    rho = np.asarray(correlations, dtype=float)

    if target_T <= T[0]:
        return float(rho[0])
    if target_T >= T[-1]:
        return float(rho[-1])

    if method == 'variance_weighted':
        # 用 ρ·T 插值（类似方差期限结构）
        rho_T = rho * T
        rho_T_interp = np.interp(target_T, T, rho_T)
        return float(np.clip(rho_T_interp / target_T, -0.9999, 0.9999))
    else:
        # 简单线性插值
        return float(np.interp(target_T, T, rho))


def forward_correlation(T1: float, T2: float,
                         rho_T1: float, rho_T2: float) -> float:
    """
    远期相关系数（Forward Correlation）

    类比远期方差公式：
        ρ_fwd(T1→T2) = (ρ_T2·T2 - ρ_T1·T1) / (T2 - T1)

    注意：此公式为近似，严格的 forward correlation 涉及协方差矩阵分解

    参数说明：
        T1, T2     : 两个到期时间（T1 < T2）
        rho_T1/T2  : 对应的相关系数

    返回：
        T1 到 T2 的远期相关系数
    """
    if T2 <= T1:
        raise ValueError("T2 必须大于 T1")
    rho_fwd = (rho_T2 * T2 - rho_T1 * T1) / (T2 - T1)
    return float(np.clip(rho_fwd, -0.9999, 0.9999))


# ============================================================
# 8. 多资产相关矩阵的正定性修正
# ============================================================

def nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    寻找最近正定相关矩阵（Higham 2002 算法）

    在市场数据中，由于数据频率不同步、缺失值等原因，
    估算出的相关矩阵可能不是正定的（PD），
    导致 Cholesky 分解失败（蒙特卡洛模拟无法进行）。

    Higham 算法：交替投影到对称矩阵集合和正半定矩阵集合

    参数说明：
        A : N×N 相关矩阵（可能非正定）

    返回：
        最近正定相关矩阵
    """
    n = A.shape[0]
    B = (A + A.T) / 2  # 对称化

    # 迭代交替投影
    Y = B.copy()
    delta_S = np.zeros_like(B)

    for _ in range(1000):
        R = Y - delta_S
        # 投影到正半定矩阵（截断负特征值）
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-8)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        delta_S = X - R
        # 投影到单位对角线（相关矩阵）
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        if np.max(np.abs(Y_new - Y)) < 1e-10:
            Y = Y_new
            break
        Y = Y_new

    # 确保对角线严格为 1
    np.fill_diagonal(Y, 1.0)
    # 确保对称
    Y = (Y + Y.T) / 2
    return Y


def check_positive_definite(corr_matrix: np.ndarray) -> dict:
    """
    检查相关矩阵的正定性并给出诊断

    参数说明：
        corr_matrix : N×N 相关矩阵

    返回：
        dict 含正定性标志、最小特征值、条件数
    """
    eigvals = np.linalg.eigvalsh(corr_matrix)
    min_eig = float(np.min(eigvals))
    cond_number = float(np.max(np.abs(eigvals)) / max(np.min(np.abs(eigvals)), 1e-12))

    return {
        'is_positive_definite': bool(min_eig > 0),
        'is_positive_semidefinite': bool(min_eig >= 0),
        'min_eigenvalue': min_eig,
        'max_eigenvalue': float(np.max(eigvals)),
        'condition_number': cond_number,
        'eigenvalues': eigvals,
    }


# ============================================================
# 示例与测试
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("第12章：相关性估计与相关性期权")
    print("=" * 60)

    # ── 1. 生成模拟收益率数据 ──────────────────────────────────────
    n_days = 500
    true_rho = 0.60
    sigma1, sigma2 = 0.20 / np.sqrt(252), 0.25 / np.sqrt(252)
    cov_matrix = np.array([[sigma1 ** 2, true_rho * sigma1 * sigma2],
                            [true_rho * sigma1 * sigma2, sigma2 ** 2]])
    L = np.linalg.cholesky(cov_matrix)
    Z = np.random.standard_normal((n_days, 2))
    returns = Z @ L.T

    r1, r2 = returns[:, 0], returns[:, 1]

    # ── 2. 历史相关性 ───────────────────────────────────────────────
    print("\n── 历史相关性估计 ──")
    rho_full = historical_correlation(r1, r2)
    print(f"全样本 Pearson 相关: {rho_full:.4f}  (真实值: {true_rho:.4f})")

    roll_corr = historical_correlation(r1, r2, window=60)
    print(f"60日滚动相关（最新）: {roll_corr[-1]:.4f}")
    print(f"60日滚动相关（均值）: {np.nanmean(roll_corr):.4f}")

    # ── 3. EWMA 相关性 ──────────────────────────────────────────────
    print("\n── EWMA 相关性 ──")
    ewma_rho, ewma_v1, ewma_v2 = ewma_correlation_series(r1, r2, lam=0.94)
    print(f"EWMA 相关（最新）: {ewma_rho[-1]:.4f}")
    print(f"EWMA 相关（均值）: {np.nanmean(ewma_rho[50:]):.4f}")

    # ── 4. DCC-GARCH ────────────────────────────────────────────────
    print("\n── DCC-GARCH(1,1) ──")
    dcc_res = dcc_garch_fit(r1, r2)
    print(f"DCC 参数: a={dcc_res['a']:.4f}, b={dcc_res['b']:.4f}")
    print(f"DCC 相关（最新）: {dcc_res['rho_t'][-1]:.4f}")
    print(f"DCC 相关（均值）: {np.mean(dcc_res['rho_t'][10:]):.4f}")

    forecasts = dcc_forecast(dcc_res, h_ahead=5)
    print(f"DCC 5步预测: {forecasts}")

    # ── 5. 隐含相关性 ───────────────────────────────────────────────
    print("\n── 隐含相关性（从指数与成分股）──")
    # 假设 3 个成分股
    w = np.array([0.50, 0.30, 0.20])
    comp_vols = np.array([0.20, 0.25, 0.30])
    # 给定隐含相关 0.40，计算理论指数波动率
    rho_assumed = 0.40
    sigma_idx_sq = (rho_assumed * (np.dot(w, comp_vols)) ** 2
                    + (1 - rho_assumed) * np.sum(w ** 2 * comp_vols ** 2))
    sigma_idx = np.sqrt(sigma_idx_sq)
    print(f"假设相关={rho_assumed:.2f} → 指数波动率={sigma_idx:.4f}")

    rho_impl = implied_correlation_index(sigma_idx, comp_vols, w)
    print(f"反推隐含相关: {rho_impl:.4f}  (应≈{rho_assumed:.4f})")

    # ── 6. 相关性互换 ───────────────────────────────────────────────
    print("\n── 相关性互换 ──")
    # 多资产返回矩阵
    returns_matrix = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[0.04, 0.02, 0.015],
             [0.02, 0.0625, 0.02],
             [0.015, 0.02, 0.09]],
        size=252
    )
    avg_corr = average_pairwise_correlation(returns_matrix)
    fair_strike = correlation_swap_fair_strike(returns_matrix)
    print(f"平均成对相关: {avg_corr:.4f}")
    print(f"相关互换公平 strike: {fair_strike:.4f}")

    realized = 0.35
    pnl = correlation_swap_payoff(realized, fair_strike, notional=1_000_000)
    print(f"相关互换收益（ρ_realized={realized:.2f}）: ${pnl:,.0f}")

    # ── 7. 最好表现期权 ─────────────────────────────────────────────
    print("\n── 最好表现期权（Stulz 1982）──")
    S1, S2, K = 100.0, 100.0, 100.0
    T_opt = 0.50
    r_opt = 0.05
    sig1, sig2 = 0.20, 0.25
    for rho_test in [0.0, 0.3, 0.6, 0.9]:
        price = best_of_two_option(S1, S2, K, T_opt, r_opt, r_opt, r_opt,
                                    sig1, sig2, rho_test, 'call')
        corr_d = correlation_delta(S1, S2, K, T_opt, r_opt, r_opt, r_opt,
                                    sig1, sig2, rho_test, 'call')
        print(f"ρ={rho_test:.1f}  →  价格={price:.4f},  ∂C/∂ρ={corr_d:.4f}")

    # ── 8. 相关性期限结构 ───────────────────────────────────────────
    print("\n── 相关性期限结构 ──")
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    corrs = np.array([0.70, 0.65, 0.60, 0.55])
    for target in [0.75, 1.5]:
        rho_interp = correlation_term_structure(mats, corrs, target, 'variance_weighted')
        print(f"T={target:.2f} 插值相关: {rho_interp:.4f}")

    fwd_corr = forward_correlation(0.5, 1.0, corrs[1], corrs[2])
    print(f"远期相关 0.5→1.0: {fwd_corr:.4f}")

    # ── 9. 正定性修正 ───────────────────────────────────────────────
    print("\n── 相关矩阵正定性修正 ──")
    bad_corr = np.array([[1.0, 0.90, 0.85],
                          [0.90, 1.0, 0.95],
                          [0.85, 0.95, 1.0]])
    diag = check_positive_definite(bad_corr)
    print(f"原始矩阵正定？{diag['is_positive_definite']}  最小特征值: {diag['min_eigenvalue']:.6f}")

    good_corr = nearest_positive_definite(bad_corr)
    diag2 = check_positive_definite(good_corr)
    print(f"修正后正定？{diag2['is_positive_definite']}  最小特征值: {diag2['min_eigenvalue']:.6f}")
    print(f"修正后相关矩阵:\n{good_corr.round(4)}")
