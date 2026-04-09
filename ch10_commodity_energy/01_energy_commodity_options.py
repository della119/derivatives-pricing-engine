"""
01_energy_commodity_options.py — 能源与商品期权定价
====================================================
【模型简介】
商品（能源、农产品、金属）期权定价与股票期权的关键区别：
1. 便利收益（Convenience Yield）：持有实物商品有"储存省去的
   好处"，相当于负成本的红利
2. 储存成本（Storage Cost）：仓储、保险等费用，相当于正红利
3. 价格均值回归（Mean Reversion）：商品价格往往围绕长期均衡
   水平波动，而非纯随机游走（GBM）
4. 季节性（Seasonality）：天然气、电力等能源需求有季节规律

本文件实现：
  A. Black-76（期货期权）— 商品期权的基础模型
  B. Miltersen-Schwartz (1998) 两因素模型（期货价格+便利收益）
  C. Schwartz (1997) 单因素均值回归模型（连续时间）
  D. 扩展 Gabillon 模型（短期/长期价格结构）
  E. 能源互换定价（Energy Swap，固定价格与浮动价格互换）

参考：
  - Black, F. (1976). "The Pricing of Commodity Contracts."
    Journal of Financial Economics, 3, 167–179.
  - Miltersen, K. & Schwartz, E. (1998). "Pricing of Options on
    Commodity Futures with Stochastic Term Structures of Convenience
    Yields and Interest Rates." Journal of Financial and
    Quantitative Analysis, 33, 33–59.
  - Schwartz, E. (1997). "The Stochastic Behavior of Commodity Prices."
    Journal of Finance, 52(3), 923–973.
书中对应：Haug (2007), Chapter 10
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. Black-76 期货期权（商品期权基础）
# ═══════════════════════════════════════════════════════════════

def black_76_commodity(F: float, K: float, T: float,
                        r: float, sigma: float,
                        option_type: str = 'call') -> float:
    """
    Black-76 期货期权（商品期货期权标准模型）。

    假设期货价格服从 GBM（无漂移，因为期货价格已经是远期价格）：
    dF/F = σ dW

    该模型等价于持有成本 b = 0 的广义 BSM。
    广泛用于：
    - 原油期货期权
    - 天然气期货期权
    - 农产品期货期权
    - 金属期货期权

    公式
    ----
    d₁ = [ln(F/K) + σ²T/2] / (σ√T)
    d₂ = d₁ - σ√T

    看涨：C = e^{-rT}[F·N(d₁) - K·N(d₂)]
    看跌：P = e^{-rT}[K·N(-d₂) - F·N(-d₁)]

    参数
    ----
    F     : 期货价格（当前报价）
    K     : 行权价
    T     : 到期时间（年）
    r     : 无风险利率（用于折现，期货无资金成本）
    sigma : 期货价格的年化波动率

    应用示例
    --------
    WTI 原油期货 @ $70/桶，行权价 $72，3 月期，σ=30% → call ≈ ...
    """
    if T <= 0:
        if option_type == 'call': return max(F - K, 0.)
        return max(K - F, 0.)

    d1 = (log(F/K) + 0.5*sigma**2*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    df = exp(-r*T)

    if option_type.lower() == 'call':
        return df * (F * N(d1) - K * N(d2))
    return df * (K * N(-d2) - F * N(-d1))


def commodity_option_spot(S: float, K: float, T: float,
                           r: float, u: float, y: float,
                           sigma: float,
                           option_type: str = 'call') -> float:
    """
    现货价格商品期权（含储存成本和便利收益）。

    商品现货价格的成本收益：
    持有成本 b = r + u - y
    其中：
      u = 储存成本率（年化，如 u = 0.03 表示 3%/年的仓储费）
      y = 便利收益率（年化，如 y = 0.05 表示 5%/年）
      b = r + u - y（净持有成本，等价于 BSM 中的 b）

    通过广义 BSM 定价（b = r + u - y）。

    参数
    ----
    S     : 商品现货价格
    u     : 储存成本率（年化比例）
    y     : 便利收益率（年化比例）
    """
    b = r + u - y   # 净持有成本

    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    d1 = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    cf = exp((b-r)*T); df = exp(-r*T)

    if option_type.lower() == 'call':
        return S*cf*N(d1) - K*df*N(d2)
    return K*df*N(-d2) - S*cf*N(-d1)


# ═══════════════════════════════════════════════════════════════
# B. Schwartz (1997) 单因素均值回归模型
# ═══════════════════════════════════════════════════════════════

def schwartz_mean_reversion_option(S: float, K: float, T: float,
                                    r: float, sigma: float,
                                    kappa: float, mu: float, lam: float,
                                    option_type: str = 'call') -> float:
    """
    Schwartz (1997) 单因素均值回归商品期权定价。

    商品价格的 OU（Ornstein-Uhlenbeck）均值回归过程：
    d(ln S) = κ(μ̄ - ln S) dt + σ dW

    其中：
      κ   = kappa = 均值回归速度（值越大，回归越快）
      μ   = 风险中性长期均值（对数价格）
      λ   = 市场价格的风险溢价（风险调整）
      μ̄  = μ - λ/κ（风险中性测度下的长期均值）

    期权定价通过等价的远期价格计算：

    远期价格（期货价格）：
    F(t, T) = exp[e^{-κT}·ln(S) + μ̄·(1-e^{-κT}) + 0.5·σ²/κ·(1-e^{-2κT})/2]

    期货期权波动率：
    σ_T = σ·√[(1-e^{-2κT})/(2κ·T)]

    然后对期货 F 使用 Black-76 定价。

    参数
    ----
    S     : 当前商品价格
    K     : 行权价
    T     : 到期时间（年）
    r     : 无风险利率
    sigma : 商品价格波动率
    kappa : 均值回归速度 κ（>0）
    mu    : 风险中性长期对数均值 μ̄（不是物理测度下的 μ）
    lam   : 市场风险溢价 λ（通常从市场期货曲线校准）
    """
    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 风险中性远期价格（Schwartz 1997 公式）
    alpha = exp(-kappa * T)
    mu_star = mu - lam / kappa   # 风险中性长期均值

    # E^Q[ln(S_T)] = α·ln(S) + μ*·(1-α)
    E_lnS = alpha * log(S) + mu_star * (1. - alpha)
    # Var^Q[ln(S_T)] = σ²·(1-e^{-2κT})/(2κ)
    var_lnS = sigma**2 * (1. - alpha**2) / (2. * kappa)

    # 风险中性期货价格：F = e^{E[ln S] + 0.5·Var[ln S]}
    F = exp(E_lnS + 0.5 * var_lnS)

    # 期货期权的有效波动率（Black-76 框架）
    sigma_T = sigma * sqrt((1. - alpha**2) / (2. * kappa * T))

    # 使用 Black-76 定价
    return black_76_commodity(F, K, T, r, sigma_T, option_type)


# ═══════════════════════════════════════════════════════════════
# C. Miltersen-Schwartz (1998) 两因素模型
# ═══════════════════════════════════════════════════════════════

def miltersen_schwartz_option(F: float, K: float, T: float,
                               T_futures: float,
                               r: float,
                               sigma_F: float, sigma_r: float, sigma_y: float,
                               rho_Fr: float, rho_Fy: float, rho_ry: float,
                               kappa_y: float,
                               option_type: str = 'call') -> float:
    """
    Miltersen-Schwartz (1998) 两因素商品期权定价。

    该模型允许：
    - 便利收益 y 服从均值回归（OU 过程）
    - 无风险利率 r 随机（Vasicek 利率过程）
    - 期货价格 F 与 y、r 的相关性

    期货期权的有效波动率：
    σ²_eff = σ_F² + (σ_r/κ_r)²·[(1-e^{-κ_r·T})²/(T·...)] + ...
             + 2·ρ_Fr·σ_F·(...) + ...

    本实现采用简化版（假设利率为常数），仅保留便利收益随机性：
    σ²_eff = σ_F² + σ_y²/(κ_y²T) · (κ_y·T - 1 + e^{-κ_y·T})²/... (简化)
             + 2·ρ_Fy·σ_F·σ_y/κ_y · (1 - e^{-κ_y·T})/T·T + ...

    注意：本函数是简化版，完整实现需要更复杂的积分计算。

    参数
    ----
    F         : 当前期货价格（到期 T_futures 的期货合约）
    K         : 行权价
    T         : 期权到期时间（≤ T_futures）
    T_futures : 期货合约到期时间
    sigma_F   : 期货价格波动率
    sigma_r   : 利率波动率
    sigma_y   : 便利收益波动率
    rho_Fr    : F 与 r 的相关系数
    rho_Fy    : F 与 y 的相关系数
    rho_ry    : r 与 y 的相关系数
    kappa_y   : 便利收益的均值回归速度
    """
    if T <= 0:
        if option_type == 'call': return max(F - K, 0.)
        return max(K - F, 0.)

    # 便利收益均值回归带来的额外方差（简化计算）
    # 精确公式需要对 e^{-κT} 积分，这里用近似
    def ms_integral(kappa: float, t: float) -> float:
        """(1-e^{-κT})²/κ 的积分近似"""
        if abs(kappa) < 1e-6:
            return t
        return (2./kappa**2) * (kappa*t - 1 + exp(-kappa*t))

    # 有效方差（简化 Miltersen-Schwartz）
    # 仅考虑主要项
    tau = T  # 期权有效期

    var_F = sigma_F**2 * tau
    var_y_contrib = (sigma_y / kappa_y)**2 * ms_integral(kappa_y, tau)
    cross_Fy = 2 * rho_Fy * sigma_F * sigma_y / kappa_y * (1 - exp(-kappa_y*tau))

    total_var = var_F + var_y_contrib + cross_Fy
    sigma_eff = sqrt(max(total_var / tau, 1e-10))

    # 用 Black-76 定价
    return black_76_commodity(F, K, T, r, sigma_eff, option_type)


# ═══════════════════════════════════════════════════════════════
# D. 能源互换（Energy Swap）定价
# ═══════════════════════════════════════════════════════════════

def energy_swap_value(F_curve: list, K_fixed: float,
                       r: float, N_quantity: float = 1.0) -> float:
    """
    能源互换（Energy Swap / Commodity Swap）定价。

    固定价格接受方的价值：
    固定方收到浮动价格（现货/期货价格 F_t），支付固定价格 K_fixed。

    V = N · Σ_{i=1}^{n} e^{-r·t_i} · (F_i - K_fixed)

    公允互换价格（NPV=0 时的 K）：
    K_fair = Σ(e^{-r·t_i}·F_i) / Σ(e^{-r·t_i})
             = 期货价格的折现加权平均

    参数
    ----
    F_curve    : [(t₁, F₁), (t₂, F₂), ...] 期货价格曲线（时间, 期货价格）
    K_fixed    : 固定价格（合约规定）
    r          : 无风险利率（用于折现）
    N_quantity : 合约名义数量

    返回
    ----
    float : 接受浮动、支付固定方的互换价值
    """
    total_value = 0.
    for t, F in F_curve:
        discount = exp(-r * t)
        total_value += discount * (F - K_fixed)
    return N_quantity * total_value


def fair_swap_price(F_curve: list, r: float) -> float:
    """
    计算公允互换价格（使互换 NPV = 0 的固定价格）。

    K_fair = Σ(e^{-r·t_i}·F_i) / Σ(e^{-r·t_i})
           = 折现加权平均期货价格
    """
    numerator   = sum(exp(-r*t) * F for t, F in F_curve)
    denominator = sum(exp(-r*t)     for t, F in F_curve)
    return numerator / denominator if denominator > 1e-12 else 0.


def energy_swaption(F_avg_fwd: float, K: float, T_expiry: float,
                     r: float, sigma_swap: float,
                     notional: float = 1.0,
                     option_type: str = 'call') -> float:
    """
    能源互换期权（Energy Swaption）定价。

    能源互换期权赋予持有人在到期日 T_expiry 以固定价格 K
    进入一个能源互换合同的权利。

    近似定价：
    将互换的浮动腿视为远期价格 F_avg_fwd（到期时的互换价值远期），
    使用 Black-76 框架定价。

    看涨：持有人若互换价格 > K，则行权，以 K 进入（浮动接受方）
    看跌：持有人若互换价格 < K，则行权，以 K 进入（固定接受方）

    参数
    ----
    F_avg_fwd  : 互换固定价格的远期价格（= 公允互换价格的远期）
    K          : 互换期权的行权价（固定价格）
    T_expiry   : 期权到期时间
    sigma_swap : 互换价格的隐含波动率
    """
    # 使用 Black-76 定价互换期权
    price = black_76_commodity(F_avg_fwd, K, T_expiry, r, sigma_swap, option_type)
    return notional * price


# ═══════════════════════════════════════════════════════════════
# E. 商品期权隐含便利收益（Implied Convenience Yield）
# ═══════════════════════════════════════════════════════════════

def implied_convenience_yield(S: float, F: float, T: float,
                               r: float, u: float = 0.) -> float:
    """
    从现货价格和期货价格反推隐含便利收益。

    持有成本关系（期货定价公式）：
    F = S · e^{(r + u - y)·T}

    解出 y：
    y = r + u - ln(F/S) / T

    参数
    ----
    S  : 现货价格
    F  : 期货价格（到期 T 的合约）
    r  : 无风险利率
    u  : 储存成本率（年化）

    返回
    ----
    float : 隐含便利收益率（年化）
    """
    return r + u - log(F/S) / T


if __name__ == "__main__":
    print("=" * 65)
    print("能源与商品期权定价 — 数值示例（Haug Chapter 10）")
    print("=" * 65)

    # ── Black-76 期货期权 ─────────────────────────────────
    F, K, T, r, sigma = 70., 70., 0.5, 0.05, 0.30
    print(f"\nBlack-76 期货期权（原油期权类）")
    print(f"  F={F}, K={K}, T={T}, r={r:.0%}, σ={sigma:.0%}")
    b76_c = black_76_commodity(F, K, T, r, sigma, 'call')
    b76_p = black_76_commodity(F, K, T, r, sigma, 'put')
    print(f"  看涨 = {b76_c:.4f},  看跌 = {b76_p:.4f}")
    print(f"  Put-Call 平价：C-P = {b76_c-b76_p:.4f}, "
          f"F·e^{{-rT}}-K·e^{{-rT}} = {(F-K)*exp(-r*T):.4f}")

    # 不同行权价下的期权价格
    print(f"\n  不同执行价下的期货期权价格（F={F}, T={T}年）：")
    print(f"  {'K':>6}  {'看涨':>10}  {'看跌':>10}  {'Delta_call':>12}")
    for K_t in [55, 60, 65, 70, 75, 80, 85]:
        c = black_76_commodity(F, K_t, T, r, sigma, 'call')
        p = black_76_commodity(F, K_t, T, r, sigma, 'put')
        d1t = (log(F/K_t) + 0.5*sigma**2*T)/(sigma*sqrt(T))
        from utils.common import norm_cdf as Nc
        delta_c = exp(-r*T) * Nc(d1t)
        print(f"  {K_t:>6}  {c:>10.4f}  {p:>10.4f}  {delta_c:>12.4f}")

    # ── 现货期权（含储存成本和便利收益）────────────────────
    S_cm = 70.; u_cm = 0.02; y_cm = 0.05   # 储存成本 2%，便利收益 5%
    b_cm = r + u_cm - y_cm
    print(f"\n现货商品期权（含持有成本）")
    print(f"  S={S_cm}, K={K}, T={T}, 储存={u_cm:.0%}, 便利收益={y_cm:.0%}")
    print(f"  净持有成本 b = r+u-y = {b_cm:.4f}")
    sp_c = commodity_option_spot(S_cm, K, T, r, u_cm, y_cm, sigma, 'call')
    print(f"  现货看涨 = {sp_c:.4f}")

    # 隐含便利收益
    F_market = 68.  # 市场观察的期货价格（低于现货，说明便利收益高）
    y_impl = implied_convenience_yield(S_cm, F_market, T, r, u_cm)
    print(f"\n  从 S={S_cm}, F={F_market} 推断隐含便利收益 = {y_impl:.4f} ({y_impl:.2%}/年)")

    # ── Schwartz 均值回归模型 ─────────────────────────────
    print(f"\nSchwartz (1997) 均值回归商品期权")
    S_s = 70.; K_s = 70.; T_s = 0.5
    kappa_s = 1.5     # 均值回归速度（适中）
    mu_s    = log(70.)  # 长期均衡对数价格（= ln(70)）
    lam_s   = 0.10      # 风险溢价（正值意味着商品价格风险溢价）
    sigma_s = 0.35

    mr_call = schwartz_mean_reversion_option(S_s, K_s, T_s, r, sigma_s,
                                              kappa_s, mu_s, lam_s, 'call')
    b76_ref  = black_76_commodity(S_s * exp(-r*T_s), K_s, T_s, r, sigma_s, 'call')
    print(f"  S={S_s}, K={K_s}, T={T_s}, κ={kappa_s}, μ={mu_s:.4f}, λ={lam_s}")
    print(f"  均值回归看涨 = {mr_call:.4f}")
    print(f"  Black-76 参考 = {b76_ref:.4f}  （均值回归使波动率衰减）")

    # ── 能源互换 ─────────────────────────────────────────
    print(f"\n能源互换定价")
    # 期货价格曲线（例如天然气，单位：$/MMBTU）
    F_curve = [(0.25, 4.20), (0.50, 4.35), (0.75, 4.50), (1.00, 4.60)]
    r_swap  = 0.05

    K_fair = fair_swap_price(F_curve, r_swap)
    print(f"  期货曲线：{F_curve}")
    print(f"  公允互换价格 K_fair = {K_fair:.4f}")

    K_fixed = 4.40  # 合约固定价格
    V_swap  = energy_swap_value(F_curve, K_fixed, r_swap, N_quantity=10000)
    print(f"  合约固定价格 K={K_fixed}，名义量=10000 MMBTU")
    print(f"  浮动接受方互换价值 = {V_swap:.2f}  ({'盈利' if V_swap > 0 else '亏损'})")

    # 互换期权
    V_swaption = energy_swaption(K_fair, K_fixed, 0.25, r_swap, 0.20, 10000, 'call')
    print(f"\n  能源互换期权（T_expiry=0.25, σ=20%, 名义量=10000）")
    print(f"  互换看涨期权价值 = {V_swaption:.2f}")
