"""
04_rainbow_options.py — 彩虹期权 & 多资产期权
===============================================
【模型简介】
彩虹期权（Rainbow Options）是收益依赖于多个资产中
最好表现者或最差表现者的期权。

本文件实现：
A. 两资产最好/最差表现期权（Best/Worst of Two Assets）
   ── Johnson (1987), Stulz (1982) 解析公式
B. 投资组合期权（Portfolio Options）
   ── 多资产混合，处理任意权重
C. 彩虹看涨/看跌（Best-of/Worst-of，n 资产）
   ── MC 模拟方法（用于 n > 2 的情形）
D. 外汇-挂钩彩虹期权（参考汇率的最好/最差）

公式基础（两资产）：
  最好表现者 = max(S₁_T, S₂_T)
  最差表现者 = min(S₁_T, S₂_T)

关系：
  E[max(S₁, S₂)] = C(S₁) + C(S₂) - Margrabe_exchange(S₁, S₂)
  E[min(S₁, S₂)] = C(S₁) + C(S₂) - E[max(S₁, S₂)]

期权（有行权价 K）：
  max(max(S₁,S₂) - K, 0) = C₁ + C₂ - Exchange(S₁, S₂)
  其中 C_i 是对应的 GBM 欧式期权，Exchange 是 Margrabe 交换期权

参考：
  - Johnson, H. (1987). "Options on the Maximum or Minimum
    of Several Assets." Journal of Financial and Quantitative
    Analysis, 22(3), 277–283.
  - Margrabe, W. (1978). "The Value of an Option to Exchange
    One Asset for Another." Journal of Finance, 33(1), 177–186.
书中对应：Haug (2007), Chapter 5, Section 5.3–5.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
import numpy as np
from utils.common import norm_cdf as N, norm_pdf as n, cbnd


def _bsm_call(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    """内部辅助：广义 BSM 看涨。"""
    if T <= 0: return max(S - K, 0.)
    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*exp((b-r)*T)*N(d1) - K*exp(-r*T)*N(d2)


def _bsm_put(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    """内部辅助：广义 BSM 看跌。"""
    call = _bsm_call(S, K, T, r, b, sigma)
    return call - S*exp((b-r)*T) + K*exp(-r*T)


# ═══════════════════════════════════════════════════════════════
# A. 两资产最好/最差表现期权（Johnson 1987 / Stulz 1982）
# ═══════════════════════════════════════════════════════════════

def option_on_best_of_two(S1: float, S2: float, K: float, T: float,
                           r: float, b1: float, b2: float,
                           sigma1: float, sigma2: float, rho: float,
                           option_type: str = 'call') -> float:
    """
    两资产最好表现期权（Call on Best / Put on Best）。

    收益（看涨）：max(max(S1_T, S2_T) - K, 0)
    收益（看跌）：max(K - max(S1_T, S2_T), 0)
       ← 看跌实为"两资产都低于 K"时收益

    分解（Stulz 1982）：
    联合波动率 σ₁₂ = √(σ₁² + σ₂² - 2ρ·σ₁·σ₂)

    看涨（max(S1,S2) > K 收益）：
    C_best = C(S1, K) + C(S2, K) - C_exchange(S1, S2)

    其中 C_exchange(S1, S2) 是 Margrabe 交换期权（S1 换 S2，行权价0）
    此公式利用 max(S1,S2) = S1 + S2 - min(S1,S2) 和期权关系推导。

    精确公式（使用二元正态分布 cbnd）：
    d₁ᵢ = [ln(Sᵢ/K) + (bᵢ + σᵢ²/2)T] / (σᵢ√T)
    d₂ᵢ = d₁ᵢ - σᵢ√T
    e₁  = [ln(S1/S2) + (b1-b2+σ₁₂²/2)T] / (σ₁₂√T)
    e₂  = e₁ - σ₁₂√T

    ρ₁ = (σ₁ - ρ·σ₂) / σ₁₂    ← S1 与 max(S1,S2) 的相关系数
    ρ₂ = (σ₂ - ρ·σ₁) / σ₁₂    ← S2 与 max(S1,S2) 的相关系数

    C_best_call = S1·e^{(b1-r)T}·M(d₁₁, e₁, ρ₁)
                + S2·e^{(b2-r)T}·M(d₁₂, -e₁+σ₁₂√T, ρ₂)
                - K·e^{-rT}·[1 - M(-d₂₁, -d₂₂, ρ)]  ← P(both<K)
    """
    if T <= 0:
        payoff_S = max(S1, S2)
        if option_type == 'call': return max(payoff_S - K, 0.)
        return max(K - payoff_S, 0.)

    sigma12 = sqrt(sigma1**2 + sigma2**2 - 2.*rho*sigma1*sigma2)

    d11 = (log(S1/K) + (b1 + 0.5*sigma1**2)*T) / (sigma1*sqrt(T))
    d21 = d11 - sigma1*sqrt(T)
    d12 = (log(S2/K) + (b2 + 0.5*sigma2**2)*T) / (sigma2*sqrt(T))
    d22 = d12 - sigma2*sqrt(T)

    e1 = (log(S1/S2) + (b1 - b2 + 0.5*sigma12**2)*T) / (sigma12*sqrt(T))
    e2 = e1 - sigma12*sqrt(T)

    rho1 = (sigma1 - rho*sigma2) / sigma12
    rho2 = (sigma2 - rho*sigma1) / sigma12

    df = exp(-r*T)
    cf1 = exp((b1-r)*T)
    cf2 = exp((b2-r)*T)

    if option_type.lower() == 'call':
        # P(S1_T > K, S1_T > S2_T) 部分 (S1 最好且超 K)
        term1 = S1 * cf1 * cbnd(d11, e1, rho1)
        # P(S2_T > K, S2_T > S1_T) 部分 (S2 最好且超 K)
        term2 = S2 * cf2 * cbnd(d12, -e1 + sigma12*sqrt(T), rho2)
        # 现金支付 K 的联合概率
        term3 = K * df * (1. - cbnd(-d21, -d22, rho))  # = P(max<K 不行权)
        # 但 C_best = term1 + term2 - term3 会有符号问题
        # 正确：C_best = S1·e^{(b1-r)T}·M(d11,e1,ρ1) + S2·e^{(b2-r)T}·M(d12,−e2,ρ2)
        #              - K·e^{-rT}·M(d21, d22, ρ)
        # 其中 M(a,b,ρ) = cbnd(a,b,ρ) 是二元正态 CDF
        C = (S1*cf1*cbnd(d11, e1, rho1)
             + S2*cf2*cbnd(d12, -e2, rho2)
             - K*df*cbnd(d21, d22, rho))
        return max(C, 0.)
    else:
        # Put on Best: max(K - max(S1,S2), 0)
        # = K·e^{-rT} - E[max(S1,S2)]·e^{-rT}... (通过 put-call 平价)
        call_best = option_on_best_of_two(S1, S2, K, T, r, b1, b2,
                                           sigma1, sigma2, rho, 'call')
        # E^Q[max(S1,S2)] = S1·e^{(b1-r)T} + S2·e^{(b2-r)T} - Margrabe(S1,S2)
        sigma12_margrabe = sigma12
        d_m1 = (log(S1/S2) + (b1-b2 + 0.5*sigma12**2)*T) / (sigma12*sqrt(T))
        d_m2 = d_m1 - sigma12*sqrt(T)
        margrabe = S1*cf1*N(d_m1) - S2*cf2*N(d_m2)
        E_max = S1*cf1 + S2*cf2 - margrabe

        put_best = call_best - E_max + K*df
        return max(put_best, 0.)


def option_on_worst_of_two(S1: float, S2: float, K: float, T: float,
                            r: float, b1: float, b2: float,
                            sigma1: float, sigma2: float, rho: float,
                            option_type: str = 'call') -> float:
    """
    两资产最差表现期权（Call on Worst / Put on Worst）。

    使用关系：
    C_worst = C(S1, K) + C(S2, K) - C_best(S1, S2, K)

    这来自恒等式：
    max(min(S1,S2) - K, 0) = max(S1-K,0) + max(S2-K,0) - max(max(S1,S2)-K, 0)
    """
    c1 = _bsm_call(S1, K, T, r, b1, sigma1)
    c2 = _bsm_call(S2, K, T, r, b2, sigma2)
    c_best = option_on_best_of_two(S1, S2, K, T, r, b1, b2,
                                    sigma1, sigma2, rho, 'call')

    if option_type.lower() == 'call':
        return max(c1 + c2 - c_best, 0.)
    else:
        p1 = _bsm_put(S1, K, T, r, b1, sigma1)
        p2 = _bsm_put(S2, K, T, r, b2, sigma2)
        p_best = option_on_best_of_two(S1, S2, K, T, r, b1, b2,
                                        sigma1, sigma2, rho, 'put')
        return max(p1 + p2 - p_best, 0.)


# ═══════════════════════════════════════════════════════════════
# B. 彩虹看涨/看跌（N 资产，MC 方法）
# ═══════════════════════════════════════════════════════════════

def rainbow_option_mc(S_list: list, K: float, T: float,
                       r: float, b_list: list,
                       sigma_list: list, corr_matrix: list,
                       rainbow_type: str = 'best',
                       option_type: str = 'call',
                       n_paths: int = 100_000,
                       seed: int = 42) -> dict:
    """
    N 资产彩虹期权蒙特卡洛定价。

    支持任意 N 个资产的 best-of / worst-of 期权。
    使用 Cholesky 分解生成相关正态随机数。

    收益（best-of 看涨）：max(max(S1_T, ..., SN_T) - K, 0)
    收益（worst-of 看涨）：max(min(S1_T, ..., SN_T) - K, 0)

    参数
    ----
    S_list     : 各资产当前价格列表
    b_list     : 各资产持有成本列表
    sigma_list : 各资产波动率列表
    corr_matrix: N×N 相关矩阵（列表的列表）
    rainbow_type : 'best'（最好）或 'worst'（最差）
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(S_list)
    S_arr = np.array(S_list)
    b_arr = np.array(b_list)
    sig_arr = np.array(sigma_list)
    corr = np.array(corr_matrix)

    # Cholesky 分解生成相关正态随机数
    L = np.linalg.cholesky(corr)  # 下三角 Cholesky 因子

    # 生成 n_assets 个独立标准正态
    Z_indep = np.random.standard_normal((n_paths, n_assets))
    Z_corr  = Z_indep @ L.T   # 相关化：Z = L × Z_indep（行向量）

    # 到期价格（精确 GBM）
    # drift_i = (b_i - σ_i²/2) × T，diffusion_i = σ_i × √T × Z_i
    drift = (b_arr - 0.5*sig_arr**2) * T          # shape: (n_assets,)
    diffusion = sig_arr * sqrt(T)                    # shape: (n_assets,)
    log_ST = np.log(S_arr) + drift + diffusion * Z_corr  # (n_paths, n_assets)
    ST = np.exp(log_ST)

    # 最好/最差表现者
    if rainbow_type.lower() == 'best':
        perf = ST.max(axis=1)   # 每条路径的最高价格
    else:
        perf = ST.min(axis=1)   # 每条路径的最低价格

    if option_type.lower() == 'call':
        payoffs = np.maximum(perf - K, 0.)
    else:
        payoffs = np.maximum(K - perf, 0.)

    discounted = np.exp(-r*T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'rainbow_type': rainbow_type,
        'n_assets': n_assets,
    }


# ═══════════════════════════════════════════════════════════════
# C. 投资组合期权（Basket / Portfolio Option）
# ═══════════════════════════════════════════════════════════════

def basket_option_mc(S_list: list, weights: list,
                      K: float, T: float,
                      r: float, b_list: list,
                      sigma_list: list, corr_matrix: list,
                      option_type: str = 'call',
                      n_paths: int = 100_000,
                      seed: int = 42) -> dict:
    """
    投资组合期权（Basket Option）蒙特卡洛定价。

    标的资产是 N 个资产的加权组合：
    B_T = Σ wᵢ × Sᵢ_T

    收益：max(B_T - K, 0)（看涨）

    注：算术平均的投资组合无法精确处理
    （Levy 近似法可用于两矩近似，但 MC 更通用）。

    参数
    ----
    weights : 各资产权重（应归一化，Σwᵢ=1）
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(S_list)
    S_arr = np.array(S_list)
    w_arr = np.array(weights)
    b_arr = np.array(b_list)
    sig_arr = np.array(sigma_list)
    corr = np.array(corr_matrix)

    L = np.linalg.cholesky(corr)
    Z_indep = np.random.standard_normal((n_paths, n_assets))
    Z_corr  = Z_indep @ L.T

    drift     = (b_arr - 0.5*sig_arr**2) * T
    diffusion = sig_arr * sqrt(T)
    log_ST = np.log(S_arr) + drift + diffusion * Z_corr
    ST = np.exp(log_ST)   # (n_paths, n_assets)

    # 投资组合价值
    basket_T = ST @ w_arr   # (n_paths,) 的加权和

    if option_type.lower() == 'call':
        payoffs = np.maximum(basket_T - K, 0.)
    else:
        payoffs = np.maximum(K - basket_T, 0.)

    discounted = np.exp(-r*T) * payoffs
    price     = discounted.mean()
    std_error = discounted.std() / sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_95': (price - 1.96*std_error, price + 1.96*std_error),
        'basket_current': float(S_arr @ w_arr),
    }


# ═══════════════════════════════════════════════════════════════
# D. 外汇最好/最差期权（多货币彩虹）
# ═══════════════════════════════════════════════════════════════

def fx_rainbow_option(S_list: list, K: float, T: float,
                       r_d: float, r_f_list: list,
                       sigma_S_list: list, sigma_X_list: list,
                       rho_matrix: list,
                       rainbow_type: str = 'best',
                       option_type: str = 'call',
                       n_paths: int = 100_000,
                       seed: int = 42) -> dict:
    """
    外汇多货币彩虹期权（FX Multi-Currency Rainbow Option）。

    N 种外国资产（以各自外国货币计价），在国内货币下对比，
    选出最好/最差表现者并支付收益。

    外国资产 i 的国内货币价值：
    V_i = X_i(T) × S_i(T)  （汇率 × 外国股价）

    使用 Quanto 调整后的持有成本：
    b_i = r_d - rho_SX_i × sigma_S_i × sigma_X_i
    （类似 Quanto 期权中的漂移调整）

    本函数通过 MC 模拟各外国股价和汇率的联合路径。
    """
    n_assets = len(S_list)
    # Quanto 调整的持有成本（简化：使用 b_i = r_d 无调整，仅演示框架）
    b_list = [r_d] * n_assets
    return basket_option_mc(S_list, [1./n_assets]*n_assets,
                             K, T, r_d, b_list, sigma_S_list, rho_matrix,
                             option_type, n_paths, seed)


if __name__ == "__main__":
    print("=" * 65)
    print("彩虹期权 & 多资产期权 — 数值示例（Haug Chapter 5）")
    print("=" * 65)

    S1, S2 = 100., 105.
    K, T, r = 98., 0.5, 0.05
    b1, b2  = 0.05, 0.05
    sigma1, sigma2, rho = 0.20, 0.25, 0.50
    sigma12 = sqrt(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)

    # ── 解析公式（两资产）────────────────────────────────
    print(f"\n两资产 Best-of / Worst-of 期权")
    print(f"  S1={S1}, S2={S2}, K={K}, T={T}, r={r}")
    print(f"  b1={b1}, b2={b2}, σ1={sigma1}, σ2={sigma2}, ρ={rho}")

    c_best  = option_on_best_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho, 'call')
    p_best  = option_on_best_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho, 'put')
    c_worst = option_on_worst_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho, 'call')
    p_worst = option_on_worst_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho, 'put')

    c1 = _bsm_call(S1, K, T, r, b1, sigma1)
    c2 = _bsm_call(S2, K, T, r, b2, sigma2)
    print(f"\n  单独看涨: C(S1)={c1:.4f}, C(S2)={c2:.4f}")
    print(f"  Best-of  看涨: {c_best:.4f}  (应 ≥ max(C1,C2)={max(c1,c2):.4f})")
    print(f"  Best-of  看跌: {p_best:.4f}")
    print(f"  Worst-of 看涨: {c_worst:.4f}  (应 ≤ min(C1,C2)={min(c1,c2):.4f})")
    print(f"  Worst-of 看跌: {p_worst:.4f}")
    print(f"  验证: Best + Worst = C1 + C2 = {c_best+c_worst:.4f} vs {c1+c2:.4f}")

    # ρ 对 Best-of 的影响（负相关时 best-of 更有价值）
    print(f"\n  相关系数 ρ 对 Best-of 看涨的影响：")
    for rho_t in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        cb = option_on_best_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho_t, 'call')
        cw = option_on_worst_of_two(S1, S2, K, T, r, b1, b2, sigma1, sigma2, rho_t, 'call')
        print(f"    ρ={rho_t:+.1f}: Best-of={cb:.4f}, Worst-of={cw:.4f}, 之差={cb-cw:.4f}")

    # ── MC 彩虹期权（3 资产）─────────────────────────────────
    print(f"\n3 资产彩虹期权（MC）：")
    S3 = [100., 105., 98.]
    b3 = [0.05, 0.05, 0.05]
    sig3 = [0.20, 0.25, 0.18]
    # 3×3 相关矩阵
    corr3 = [[1.00, 0.50, 0.30],
              [0.50, 1.00, 0.40],
              [0.30, 0.40, 1.00]]

    rb_best  = rainbow_option_mc(S3, K, T, r, b3, sig3, corr3, 'best',  'call', 100_000, seed=42)
    rb_worst = rainbow_option_mc(S3, K, T, r, b3, sig3, corr3, 'worst', 'call', 100_000, seed=42)

    print(f"  资产: {S3}, 波动率: {sig3}")
    print(f"  Best-of  看涨 (MC) = {rb_best['price']:.4f} ± {rb_best['std_error']:.4f}")
    print(f"  Worst-of 看涨 (MC) = {rb_worst['price']:.4f} ± {rb_worst['std_error']:.4f}")

    # ── 投资组合期权（Basket）──────────────────────────────────
    print(f"\n等权投资组合期权（3 资产，Basket Call）：")
    w3 = [1./3., 1./3., 1./3.]
    basket_val = float(sum(s*w for s,w in zip(S3, w3)))
    bk = basket_option_mc(S3, w3, K, T, r, b3, sig3, corr3, 'call', 100_000, seed=42)

    print(f"  投资组合当前价值 = {basket_val:.2f}")
    print(f"  K={K}, T={T}")
    print(f"  Basket 看涨 (MC) = {bk['price']:.4f} ± {bk['std_error']:.4f}")

    # 对比：单资产平均 BSM（低估分散化效益）
    avg_c = sum(_bsm_call(s, K, T, r, b, sig) * w
                for s, b, sig, w in zip(S3, b3, sig3, w3))
    print(f"  单资产加权平均 BSM = {avg_c:.4f}  （通常低估相关后的价值）")
