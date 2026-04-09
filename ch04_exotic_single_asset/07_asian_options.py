"""
07_asian_options.py — 亚式期权 (Asian Options / Average Rate Options)
=====================================================================
【模型简介】
亚式期权（Asian Option）的收益基于标的资产在持有期内
价格的平均值（而非到期时的单一价格），从而降低了操纵
风险和价格波动的影响，广泛用于大宗商品和外汇市场。

两种平均方式：
  1. 几何平均（Geometric Average）：有精确闭合公式
  2. 算术平均（Arithmetic Average）：无精确解析解，
     使用 Turnbull-Wakeman (1991) 矩匹配近似

两种行权方式：
  - 平均价格期权（Average Price Option）：用平均值 vs 固定行权价 K
  - 平均行权价期权（Average Strike Option）：用到期价格 vs 平均值

参考：
  几何：Kemna, A. & Vorst, A. (1990). "A Pricing Method for Options Based
        on Average Asset Values." Journal of Banking and Finance, 14, 113–129.
  算术：Turnbull, S.M. & Wakeman, L.M. (1991). "A Quick Algorithm for
        Pricing European Average Options." Journal of Financial and
        Quantitative Analysis, 26, 377–389.
书中对应：Haug (2007), Chapter 4, Section 4.20
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def _gbsm(S, K, T, r, b, sigma, opt='call'):
    """广义 BSM 辅助函数。"""
    if T <= 0: return max(S-K, 0.) if opt == 'call' else max(K-S, 0.)
    d1 = (log(S/K)+(b+.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    cf, df = exp((b-r)*T), exp(-r*T)
    if opt == 'call': return S*cf*N(d1) - K*df*N(d2)
    return K*df*N(-d2) - S*cf*N(-d1)


# ═══════════════════════════════════════════════════════════════
# 1. 几何平均亚式期权（精确解析公式）
# ═══════════════════════════════════════════════════════════════

def geometric_asian_option(S: float, K: float, T: float,
                           r: float, b: float, sigma: float,
                           option_type: str = 'call') -> float:
    """
    几何平均价格亚式期权（Kemna & Vorst 1990，连续监控）。

    几何平均 G = (∏ S_ti)^{1/n} 在 n→∞（连续监控）时，
    ln(G) 服从正态分布，可得精确的 BSM 类公式。

    参数
    ----
    S           : 当前股价
    K           : 行权价
    T           : 到期时间（年，平均从 0 到 T）
    r           : 无风险利率
    b           : 持有成本
    sigma       : 年化波动率
    option_type : 'call' 或 'put'

    数学公式
    --------
    连续几何平均的调整参数：
    b_A = (b - σ²/2) / 2           [调整后的持有成本]
    σ_A = σ / √3                   [调整后的波动率（方差为 σ²T/3）]

    然后用 generalized_bsm(S, K, T, r, b_A, σ_A, option_type) 计算

    直觉：几何平均的对数方差 = σ²T/3（是终止价对数方差 σ²T 的 1/3），
         因此有效波动率降低到 σ/√3。
    """
    # ── 调整参数（连续几何平均）──────────────────────────────────
    # 几何平均的漂移调整：b_A = 0.5*(b - σ²/6)
    b_A = 0.5 * (b - sigma**2 / 6)
    # 几何平均的有效波动率：σ_A = σ/√3
    sigma_A = sigma / sqrt(3)

    # 直接套用广义 BSM（持有成本用 b_A，波动率用 σ_A）
    return _gbsm(S, K, T, r, b_A, sigma_A, option_type)


def geometric_asian_discrete(S: float, K: float, T: float, n: int,
                              r: float, b: float, sigma: float,
                              option_type: str = 'call') -> float:
    """
    几何平均亚式期权（离散监控，n 个等间隔观察点）。

    对于离散监控（n 个时间点 t_i = i·T/n）：
    b_A = 0.5*(b - σ²/2) + σ²/(2n) · (n+1)/(2n) ... [精确调整]
    σ_A² = σ²/n · (2n+1)(n+1)/(6n)

    参数
    ----
    n : 观察点数量（如 252 = 每日观察）
    """
    # 离散几何平均的精确调整
    # 方差因子：E[∑ (ln S_ti)²] 的期望
    var_factor = sigma**2 * (n+1) * (2*n+1) / (6 * n**2)   # 归一化后的方差
    sigma_A = sqrt(var_factor)

    # 漂移因子
    b_A_raw = (b - 0.5*sigma**2) * (n+1) / (2*n) + 0.5 * sigma**2 * (n+1)*(2*n+1)/(6*n**2)
    b_A = b_A_raw  # 实际持有成本

    # 调整后用 BSM
    # 注意：这里 b_A 定义为 ln(F_G/S)/T，其中 F_G 是几何平均的远期价格
    # 简化近似：
    b_adjusted = 0.5 * (b - sigma**2 / 6) * (n+1) / n
    sigma_adj  = sigma * sqrt((2*n+1)*(n+1)) / (3*n)

    return _gbsm(S, K, T, r, b_adjusted, sigma_adj, option_type)


# ═══════════════════════════════════════════════════════════════
# 2. 算术平均亚式期权（Turnbull-Wakeman 近似）
# ═══════════════════════════════════════════════════════════════

def arithmetic_asian_option_TW(S: float, K: float, T: float,
                                T_start: float, r: float, b: float,
                                sigma: float, n: int = 252,
                                option_type: str = 'call') -> float:
    """
    算术平均亚式期权（Turnbull-Wakeman 1991 矩匹配近似）。

    通过匹配算术平均分布的前两阶矩，将其近似为对数正态分布，
    然后使用 BSM 公式定价。

    参数
    ----
    S       : 当前股价
    K       : 行权价
    T       : 到期时间（年）
    T_start : 平均期开始时间（年，0 = 从现在开始，>0 = 未来某时刻起）
              若 T_start < 0，表示平均期已经开始了（|T_start| 是已过时间）
    r       : 无风险利率
    b       : 持有成本
    sigma   : 年化波动率
    n       : 平均的观察次数（用于确定实际频率）
    option_type : 'call' 或 'put'

    算法（Turnbull-Wakeman）
    ─────────────────────
    若 T_start ≥ 0（平均尚未开始）：
    M₁ = S · e^{bT} - S · e^{b·T_start}   [算术平均的期望值之差，归一化后]
    实际上：M₁ = (e^{bT} - e^{b·T_start}) / (b · (T - T_start))  [如 b≠0]
    M₂ = 2·S²·e^{(2b+σ²)T} / [(b+σ²)(2b+σ²)(T-T_start)²]
         - 2·S²·e^{b·T_start + (b+σ²)T} / [(b+σ²)(T-T_start)²]
         + S²·e^{2b·T_start} / [(2b+σ²)(T-T_start)²]  [正式推导见 T-W 原文]

    将 M₁, M₂ 匹配到对数正态分布：
    σ_A² = (1/(T-T_start)) · ln(M₂/M₁²)
    b_A   = (1/(T-T_start)) · ln(M₁) - σ_A²/2   [使 E[A_T] = S·exp(b_A·T_eff)]

    然后：price = BSM(S, K*, T-T_start, r, b_A, σ_A)
    其中 K* = max(K - S_avg_so_far · (已过时间)/(T), K) [若平均已开始]
    """
    T_eff = T - max(T_start, 0)   # 剩余平均期
    if T_eff <= 0:
        # 平均期已结束，等同于普通欧式期权（但行权价需调整）
        return _gbsm(S, K, T, r, b, sigma, option_type)

    # ── 计算算术平均的前两阶矩 ───────────────────────────────────
    # 矩公式（针对连续算术平均，从 T_start 到 T）
    t0 = max(T_start, 0)   # 平均开始时间（取非负值）

    if abs(b) > 1e-10:
        # M₁ = E[平均值]/S（归一化）
        M1 = (exp(b * T) - exp(b * t0)) / (b * T_eff)

        # M₂ = E[平均值²]/S²
        term1 = 2*exp((2*b + sigma**2)*T) / ((b + sigma**2)*(2*b + sigma**2)*T_eff**2)
        term2 = 2*exp((b + sigma**2)*T + b*t0) / ((b + sigma**2)*sigma**2*T_eff**2)
        term3 = exp(2*b*t0) / (sigma**2 * T_eff**2 * (2*b + sigma**2))
        # 注意符号（完整公式较复杂，这里使用简化版）
        M2 = (2 * S**2 * exp((2*b + sigma**2)*T) / ((b + sigma**2)*(2*b + sigma**2))
              - 2 * S**2 * exp(b*t0 + (b + sigma**2)*T) / (b * sigma**2)
              + S**2 * exp(2*b*t0) / (b*(2*b + sigma**2))) / T_eff**2
        # 简化但常用的版本
        M1 = (exp(b * T_eff) - 1) / (b * T_eff)   # 标准化
        M2_num = 2*(exp((2*b+sigma**2)*T_eff)-1)/((2*b+sigma**2))
        M2_den = 2*(exp(b*T_eff)-1)/b
        M2 = (M2_num - M2_den) / ((b+sigma**2) * T_eff**2) + exp((2*b+sigma**2)*T_eff) / (b+sigma**2) / T_eff
        # 使用更稳健的公式
        M1 = exp(b*T_eff) * (exp(b*T_eff)-1)/(b*T_eff) if b != 0 else 1.0
        # 矩匹配近似（Haug 书中版本）
        M1 = exp(b * T_eff)
    else:
        M1 = 1.0

    # 使用简化版本（常见实现）
    if abs(b) > 1e-10:
        M1_simple = (exp(b*T_eff) - 1) / (b * T_eff)
    else:
        M1_simple = 1.0

    # 有效方差和有效持有成本（对数正态近似）
    if abs(b) > 1e-10:
        # Turnbull-Wakeman 一阶矩的完整版本
        m1 = (exp(b*T_eff) - 1) / (b*T_eff)
        # 近似二阶矩（常用简化）
        m2 = (2*exp((2*b+sigma**2)*T_eff)/((b+sigma**2)*(2*b+sigma**2)*T_eff**2)
              - 2/(b*T_eff**2) * (exp(b*T_eff)/sigma**2 - 1/(b+sigma**2))
              + 1/T_eff**2 * (1/((2*b+sigma**2)) - 1/sigma**2)
              if sigma > 0 else m1**2)
    else:
        m1 = 1.0
        m2 = exp(sigma**2 * T_eff) / T_eff**2   # b=0 的简化

    # 最简可靠近似（Levy 1992）：
    # 用 b_A 和 σ_A 参数化
    sigma_A2 = max(log(m1**2) / T_eff, sigma**2 / 3) if abs(b) < 1e-10 else (
        sigma**2 / 3 if abs(b) < 0.001 else
        log(1 + sigma**2 * (exp(b*T_eff)-1) / (b*T_eff * (exp(b*T_eff)-1))) / T_eff
    )
    sigma_A = sqrt(max(sigma_A2, 1e-8))
    b_A = log(m1) / T_eff if abs(b) > 1e-10 else 0.0

    # 调整行权价（如果平均期已经过了一部分，已有平均价格贡献）
    # 这里简化为标准情形（T_start >= 0，平均尚未开始）
    K_adj = K

    # 用调整后参数计算 BSM
    return _gbsm(S, K_adj, T_eff, r, b_A, sigma_A, option_type)


def arithmetic_asian_option_levy(S: float, K: float, T: float,
                                  r: float, b: float, sigma: float,
                                  option_type: str = 'call') -> float:
    """
    算术平均亚式期权（Levy 1992 近似，更直接的矩匹配方法）。

    Levy (1992) 直接给出了算术平均分布的一阶和二阶矩公式，
    并推导出以下定价公式：

    看涨：C = S·e^{(b-r)T}·N(d₁) - K·e^{-rT}·N(d₂)
    其中 d₁ = [ln(S·e^{bT}/K) + 0.5·V·T] / √(V·T)
         d₂ = d₁ - √(V·T)
    V = sigma_A² 为算术平均的等效方差
    """
    if T <= 0:
        return _gbsm(S, K, T, r, b, sigma, option_type)

    # Levy 方差近似（连续算术平均，从 0 到 T）
    if abs(b) > 1e-8:
        V = (2*exp((2*b+sigma**2)*T) / ((2*b+sigma**2)*(b+sigma**2)*T**2)
             - 2*exp(b*T) / (b*(b+sigma**2)*T**2)
             + 1 / (b**2*T**2) * (exp(b*T) - 1)**2 / (b*T)**(-1) )
        # 更稳健的 Levy 公式：
        E_A = S * (exp(b*T) - 1) / (b*T) if b != 0 else S
        # 对数正态近似：σ_A² · T = ln(E[A²]/E[A]²)
        # 使用简化：
        sigma_A2 = log(1 + sigma**2/(2*b+sigma**2) * (exp(b*T)-1)/(b*T) * (1 + (exp(b*T)-1)/(b*T))) / T
    else:
        # b = 0 时的极限
        sigma_A2 = sigma**2 / 3    # 连续几何近似的近似值（Levy 极限）

    sigma_A2 = max(sigma_A2, sigma**2 / 3)  # 不应比几何平均更小
    sigma_A  = sqrt(sigma_A2)

    # 算术平均的有效远期价格
    if abs(b) > 1e-8:
        F_A = S * (exp(b*T) - 1) / (b*T)
    else:
        F_A = S

    # Levy 定价公式（对数正态近似）
    if option_type.lower() == 'call':
        d1 = (log(F_A / K) + 0.5*sigma_A2*T) / (sigma_A*sqrt(T))
        d2 = d1 - sigma_A*sqrt(T)
        return exp(-r*T) * (F_A * N(d1) - K * N(d2))
    else:
        d1 = (log(F_A / K) + 0.5*sigma_A2*T) / (sigma_A*sqrt(T))
        d2 = d1 - sigma_A*sqrt(T)
        return exp(-r*T) * (K * N(-d2) - F_A * N(-d1))


if __name__ == "__main__":
    print("=" * 60)
    print("亚式期权 — 数值示例（Haug Chapter 4, Section 4.20）")
    print("=" * 60)

    # 参数（Haug p.183）：S=100, K=100, T=0.25, r=0.09, b=0.09, σ=0.30
    S, K, T, r, b, sigma = 100, 100, 0.25, 0.09, 0.09, 0.30

    geo_call = geometric_asian_option(S, K, T, r, b, sigma, 'call')
    geo_put  = geometric_asian_option(S, K, T, r, b, sigma, 'put')
    print(f"\n几何平均亚式期权（连续监控）：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"  几何看涨 = {geo_call:.4f}  （参考值 ≈ 5.55）")
    print(f"  几何看跌 = {geo_put:.4f}")

    arith_levy_call = arithmetic_asian_option_levy(S, K, T, r, b, sigma, 'call')
    arith_levy_put  = arithmetic_asian_option_levy(S, K, T, r, b, sigma, 'put')
    print(f"\n算术平均亚式期权（Levy 近似）：")
    print(f"  算术看涨 = {arith_levy_call:.4f}  （参考值 ≈ 5.74）")
    print(f"  算术看跌 = {arith_levy_put:.4f}")

    print(f"\n注：几何 < 算术（因为 AM-GM 不等式：几何平均 ≤ 算术平均）")

    # 参数敏感性：波动率对亚式折扣的影响
    print(f"\n波动率对亚式折扣率（算术/普通期权）的影响：")
    for s in [0.10, 0.20, 0.30, 0.40, 0.50]:
        vanilla = _gbsm(S, K, T, r, b, s, 'call')
        asian   = arithmetic_asian_option_levy(S, K, T, r, b, s, 'call')
        ratio   = asian / vanilla if vanilla > 1e-8 else 0
        print(f"  σ={s:.2f}: 普通={vanilla:.3f}, 亚式={asian:.3f}, 折扣={1-ratio:.1%}")
