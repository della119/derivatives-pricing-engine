"""
01_interest_rate_options.py — 利率衍生品定价
============================================
【模型简介】
利率期权的定价框架与股票期权有本质区别：
1. 标的资产（利率）本身不能交易，需要通过债券/互换来实现
2. 利率通常不能为负（但现实中可以）
3. 利率之间高度相关，需要期限结构模型
4. 主要产品：利率上限（Cap）、利率下限（Floor）、互换期权（Swaption）

本文件实现：
  A. Black-76 模型 — 远期利率和互换利率期权
     - 利率上限（Cap）= 多个 Caplet 的总和
     - 利率下限（Floor）= 多个 Floorlet 的总和
     - 欧式互换期权（Swaption）
  B. Vasicek (1977) 债券期权 — 短期利率模型下的解析解
  C. Hull-White (1990/1993) 扩展 Vasicek 债券期权
  D. Ho-Lee (1986) 债券期权
  E. Black-Derman-Toy (BDT, 1990) 利率树（简化版）

参考：
  - Black, F. (1976). "The Pricing of Commodity Contracts."
    Journal of Financial Economics, 3, 167–179.
  - Vasicek, O. (1977). "An Equilibrium Characterization of the
    Term Structure." Journal of Financial Economics, 5, 177–188.
  - Hull, J. & White, A. (1990/1993). One-factor Hull-White model.
  - Ho, T.S.Y. & Lee, S.B. (1986). "Term Structure Movements and
    Pricing Interest Rate Contingent Claims." Journal of Finance,
    41(5), 1011–1029.
书中对应：Haug (2007), Chapter 11
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. Black-76 利率期权框架
# ═══════════════════════════════════════════════════════════════

def caplet_floorlet(F: float, K: float, T: float,
                    r: float, sigma: float,
                    tau: float = 0.25,
                    notional: float = 1.0,
                    option_type: str = 'cap') -> float:
    """
    利率 Caplet / Floorlet 定价（Black-76 模型）。

    Caplet（单个上限合约）：
    在时间 T + τ 结算，基于 T 时观察的 LIBOR/SOFR 利率。
    若浮动利率 L_T > K（行权价），则支付 (L_T - K) × τ × 名义本金。

    Black-76 公式（远期利率 L_T 服从对数正态）：
    d₁ = [ln(F/K) + σ²T/2] / (σ√T)
    d₂ = d₁ - σ√T

    Caplet = N·τ·e^{-r(T+τ)} · [F·N(d₁) - K·N(d₂)]
    Floorlet = N·τ·e^{-r(T+τ)} · [K·N(-d₂) - F·N(-d₁)]

    参数
    ----
    F          : 远期利率（Forward Rate for [T, T+τ]）
    K          : 行权利率（Strike Rate）
    T          : 利率观察时间（期权到期时间）
    r          : 无风险折现利率
    sigma      : 远期利率的波动率（利率波动率）
    tau        : 结算期间（如 0.25 = 季度，0.5 = 半年）
    notional   : 名义本金
    option_type: 'cap'（上限）或 'floor'（下限）
    """
    if T <= 0:
        if option_type == 'cap':
            payoff = max(F - K, 0.) * tau * notional
        else:
            payoff = max(K - F, 0.) * tau * notional
        return payoff * exp(-r * (T + tau))   # 到结算日的折现

    d1 = (log(F/K) + 0.5*sigma**2*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    # 结算日在 T+τ，需折现到今天
    df = exp(-r * (T + tau)) * tau * notional

    if option_type.lower() in ('cap', 'caplet'):
        return df * (F * N(d1) - K * N(d2))
    return df * (K * N(-d2) - F * N(-d1))


def interest_rate_cap_floor(forward_curve: list,
                             K: float, r: float, sigma: float,
                             tau: float = 0.25,
                             notional: float = 1.0,
                             option_type: str = 'cap') -> float:
    """
    利率上限 / 下限（Cap / Floor）定价。

    Cap = Σ Caplet_i（所有期间的 Caplet 之和）
    Floor = Σ Floorlet_i

    Cap-Floor 平价（类似 Put-Call 平价）：
    Cap - Floor = 浮动利率互换价值（接受浮动方）

    参数
    ----
    forward_curve : [(T_i, F_i), ...] 远期利率曲线
                    T_i = 利率观察时间（期权到期），F_i = 远期利率
                    结算时间 = T_i + tau
    K             : 行权利率
    r             : 折现利率（或用折现曲线）
    sigma         : 利率波动率（假设各期相同，即 flat vol surface）
    tau           : 结算期间（季度=0.25，半年=0.5，年=1）
    notional      : 名义本金

    返回
    ----
    float : Cap/Floor 总价值
    """
    total = 0.
    for T_i, F_i in forward_curve:
        if T_i <= 0:
            continue
        cl = caplet_floorlet(F_i, K, T_i, r, sigma, tau, notional, option_type)
        total += cl
    return total


def swaption_black76(F_swap: float, K: float, T_expiry: float,
                      r: float, sigma: float,
                      annuity: float,
                      payer_receiver: str = 'payer') -> float:
    """
    欧式互换期权（European Swaption）定价（Black-76 模型）。

    互换期权赋予持有人在 T_expiry 时以固定利率 K 进入
    利率互换合同的权利。

    支付型互换期权（Payer Swaption）：
    - 持有人有权成为固定利率支付方（浮动利率接受方）
    - 若到期互换利率 > K（收益），则行权
    - 价值 = A(T) · Black76Call(F_swap, K)

    接受型互换期权（Receiver Swaption）：
    - 持有人有权成为固定利率接受方
    - 若到期互换利率 < K（收益），则行权

    公式
    ----
    d₁ = [ln(F_swap/K) + σ²·T_expiry/2] / (σ·√T_expiry)
    d₂ = d₁ - σ·√T_expiry

    支付型：V = A · [F_swap·N(d₁) - K·N(d₂)]
    接受型：V = A · [K·N(-d₂) - F_swap·N(-d₁)]

    其中 A = 年金因子（Annuity Factor）= Σ τᵢ·B(0, Tᵢ)
    B(0, Tᵢ) = 到 Tᵢ 的折现因子

    参数
    ----
    F_swap       : 远期互换利率（Forward Swap Rate）
    K            : 互换期权行权利率（Strike Swap Rate）
    T_expiry     : 互换期权到期时间
    r            : 无风险利率（用于简化近似）
    sigma        : 互换利率波动率（Swaption 隐含波动率）
    annuity      : 年金因子 A（预先计算，取决于互换期限和折现曲线）
    payer_receiver : 'payer'（支付型）或 'receiver'（接受型）
    """
    if T_expiry <= 0:
        if payer_receiver == 'payer':
            return annuity * max(F_swap - K, 0.)
        return annuity * max(K - F_swap, 0.)

    d1 = (log(F_swap/K) + 0.5*sigma**2*T_expiry) / (sigma*sqrt(T_expiry))
    d2 = d1 - sigma*sqrt(T_expiry)

    if payer_receiver.lower() == 'payer':
        return annuity * (F_swap * N(d1) - K * N(d2))
    return annuity * (K * N(-d2) - F_swap * N(-d1))


def annuity_factor(r: float, T_start: float, T_end: float,
                   payment_freq: int = 4) -> float:
    """
    年金因子计算（平坦收益率曲线近似）。

    A = Σ_{i=1}^{n} τ · e^{-r·T_i}

    其中 T_i = T_start + i·τ，τ = 1/payment_freq

    参数
    ----
    T_start      : 互换开始时间（=期权到期时间）
    T_end        : 互换结束时间
    payment_freq : 每年支付次数（4=季度，2=半年，1=年）
    """
    tau = 1.0 / payment_freq
    n_payments = int(round((T_end - T_start) * payment_freq))
    annuity = 0.
    for i in range(1, n_payments + 1):
        T_i = T_start + i * tau
        annuity += tau * exp(-r * T_i)
    return annuity


# ═══════════════════════════════════════════════════════════════
# B. Vasicek (1977) 债券期权
# ═══════════════════════════════════════════════════════════════

def vasicek_bond_price(r0: float, T: float,
                        kappa: float, theta: float, sigma: float) -> float:
    """
    Vasicek (1977) 债券价格（零息债券）。

    Vasicek 短利率过程：
    dr = κ(θ - r) dt + σ dW

    解析债券价格：
    P(0, T) = A(T) · e^{-B(T)·r₀}

    其中：
    B(T) = (1 - e^{-κT}) / κ
    A(T) = exp[(B(T) - T)·(κ²θ - σ²/2)/κ² - σ²·B(T)²/(4κ)]

    参数
    ----
    r0    : 当前短期利率
    T     : 债券到期时间（年）
    kappa : 均值回归速度
    theta : 长期均衡利率
    sigma : 利率波动率
    """
    if T <= 0:
        return 1.0   # 到期债券价值为 1

    B = (1. - exp(-kappa*T)) / kappa
    A = exp((B - T) * (kappa**2 * theta - 0.5*sigma**2) / kappa**2
            - sigma**2 * B**2 / (4.*kappa))
    return A * exp(-B * r0)


def vasicek_bond_option(r0: float, K: float, T_option: float, T_bond: float,
                         kappa: float, theta: float, sigma: float,
                         option_type: str = 'call') -> float:
    """
    Vasicek (1977) 债券期权（欧式零息债券期权）。

    到期为 T_bond 的零息债券，上面挂一个到期为 T_option 的欧式期权。

    Jamshidian (1989) 精确公式：
    有效波动率：
    σ_P = σ · B(T_bond - T_option) · √[(1-e^{-2κ·T_option})/(2κ)]
        = σ/κ · (1 - e^{-κ(T_bond-T_option)}) · √[(1-e^{-2κT_option})/(2κ)]

    h = (1/σ_P) · ln[P(0,T_bond) / (K·P(0,T_option))] + σ_P/2

    看涨：C = P(0,T_bond)·N(h) - K·P(0,T_option)·N(h - σ_P)
    看跌：P = K·P(0,T_option)·N(-h + σ_P) - P(0,T_bond)·N(-h)

    参数
    ----
    r0        : 当前短期利率
    K         : 期权行权价（债券价格，如 0.95）
    T_option  : 期权到期时间
    T_bond    : 债券到期时间（> T_option）
    kappa, theta, sigma : Vasicek 参数
    """
    if T_option <= 0:
        # 期权已到期
        P_bond = vasicek_bond_price(r0, T_bond, kappa, theta, sigma)
        if option_type == 'call': return max(P_bond - K, 0.)
        return max(K - P_bond, 0.)

    # 两个关键债券价格
    P_T_option = vasicek_bond_price(r0, T_option, kappa, theta, sigma)
    P_T_bond   = vasicek_bond_price(r0, T_bond,   kappa, theta, sigma)

    # 有效波动率
    B_diff = (1. - exp(-kappa*(T_bond - T_option))) / kappa  # B(T_bond - T_option)
    sigma_P = sigma * B_diff * sqrt((1. - exp(-2.*kappa*T_option)) / (2.*kappa))

    if sigma_P < 1e-10:
        if option_type == 'call': return max(P_T_bond - K*P_T_option, 0.)
        return max(K*P_T_option - P_T_bond, 0.)

    h = log(P_T_bond / (K * P_T_option)) / sigma_P + sigma_P / 2.

    if option_type.lower() == 'call':
        return P_T_bond * N(h) - K * P_T_option * N(h - sigma_P)
    return K * P_T_option * N(-h + sigma_P) - P_T_bond * N(-h)


# ═══════════════════════════════════════════════════════════════
# C. Hull-White (1990/1993) 扩展 Vasicek
# ═══════════════════════════════════════════════════════════════

def hull_white_bond_option(P_T_option: float, P_T_bond: float,
                            K: float, T_option: float, T_bond: float,
                            kappa: float, sigma: float,
                            option_type: str = 'call') -> float:
    """
    Hull-White (1990) 扩展 Vasicek 模型债券期权。

    Hull-White 对 Vasicek 的改进：通过时间依赖的漂移 θ(t)
    使模型与当前期限结构精确拟合（校准初始债券价格）。

    当模型与当前期限结构校准后，定价公式形式上与 Vasicek 相同，
    但输入的是市场观察的折现因子 P(0, T)，而非模型预测值。

    有效波动率（Hull-White）：
    σ_P = σ/κ · (1 - e^{-κ(T_bond-T_option)}) · √[(1-e^{-2κ·T_option})/(2κ)]

    参数
    ----
    P_T_option : 市场折现因子 P(0, T_option)（从折现曲线读取）
    P_T_bond   : 市场折现因子 P(0, T_bond)
    K          : 债券期权行权价
    kappa      : 均值回归速度（从市场校准）
    sigma      : 利率波动率（从 swaption/cap 市场校准）
    """
    B_diff = (1. - exp(-kappa*(T_bond - T_option))) / kappa
    sigma_P = (sigma / kappa) * B_diff * sqrt((1. - exp(-2.*kappa*T_option)) / (2.*kappa))

    if sigma_P < 1e-10:
        if option_type == 'call': return max(P_T_bond - K*P_T_option, 0.)
        return max(K*P_T_option - P_T_bond, 0.)

    h = log(P_T_bond / (K * P_T_option)) / sigma_P + sigma_P / 2.

    if option_type.lower() == 'call':
        return P_T_bond * N(h) - K * P_T_option * N(h - sigma_P)
    return K * P_T_option * N(-h + sigma_P) - P_T_bond * N(-h)


def hull_white_caplet(F: float, K: float, T: float,
                       r: float, kappa: float, sigma: float,
                       tau: float = 0.25,
                       notional: float = 1.0) -> float:
    """
    Hull-White Caplet 定价（利用 Jamshidian 分解）。

    在 Hull-White 框架下，Caplet 等价于债券看跌期权：
    Caplet(T, T+τ) = (1 + K·τ) · PutOption on Bond(0, T+τ)
                      with K* = 1/(1 + K·τ)

    本函数使用 Black-76 近似（当 HW 校准到 flat vol 时等价）：
    """
    # 转换为债券看跌期权
    K_bond = 1. / (1. + K * tau)    # 债券期权行权价
    notional_adj = notional * (1. + K * tau)  # 调整名义本金

    # 有效波动率（Hull-White 框架）
    B = (1. - exp(-kappa * tau)) / kappa
    sigma_F = sigma * B * sqrt((1. - exp(-2.*kappa*T)) / (2.*kappa))

    # 折现因子
    P_T      = exp(-r * T)
    P_T_tau  = exp(-r * (T + tau))

    if sigma_F < 1e-10:
        return notional_adj * max(K_bond - P_T_tau/P_T, 0.) * P_T

    # 债券看跌期权 → Caplet
    h = log(P_T_tau / (K_bond * P_T)) / sigma_F + sigma_F/2.

    bond_put = K_bond * P_T * N(-h + sigma_F) - P_T_tau * N(-h)
    return notional_adj * bond_put


# ═══════════════════════════════════════════════════════════════
# D. Ho-Lee (1986) 债券期权
# ═══════════════════════════════════════════════════════════════

def ho_lee_bond_option(P_T_option: float, P_T_bond: float,
                        K: float, T_option: float, T_bond: float,
                        sigma: float,
                        option_type: str = 'call') -> float:
    """
    Ho-Lee (1986) 模型债券期权。

    Ho-Lee 是最简单的无套利利率树模型，连续时间版本：
    dr = θ(t) dt + σ dW  （无均值回归，κ = 0）

    有效波动率（κ → 0 时 Vasicek 的极限）：
    σ_P = σ · (T_bond - T_option) · √T_option

    定价公式与 Vasicek 相同（Jamshidian 公式）。

    参数
    ----
    P_T_option : 折现因子 P(0, T_option)
    P_T_bond   : 折现因子 P(0, T_bond)
    K          : 行权价
    T_option   : 期权到期
    T_bond     : 债券到期
    sigma      : 利率波动率
    """
    # Ho-Lee：κ=0 时的有效波动率
    sigma_P = sigma * (T_bond - T_option) * sqrt(T_option)

    if sigma_P < 1e-10:
        if option_type == 'call': return max(P_T_bond - K*P_T_option, 0.)
        return max(K*P_T_option - P_T_bond, 0.)

    h = log(P_T_bond / (K * P_T_option)) / sigma_P + sigma_P / 2.

    if option_type.lower() == 'call':
        return P_T_bond * N(h) - K * P_T_option * N(h - sigma_P)
    return K * P_T_option * N(-h + sigma_P) - P_T_bond * N(-h)


if __name__ == "__main__":
    print("=" * 65)
    print("利率衍生品定价 — 数值示例（Haug Chapter 11）")
    print("=" * 65)

    # ── Caplet / Cap 定价 ─────────────────────────────────
    F_rate, K_rate, T_cap, r_rf, sigma_rate = 0.055, 0.05, 1.0, 0.05, 0.20
    tau, notional = 0.25, 1_000_000.

    print(f"\nCaplet 定价（Black-76）")
    print(f"  远期利率 F={F_rate:.2%}, 行权利率 K={K_rate:.2%}, T={T_cap}年")
    print(f"  r={r_rf:.2%}, σ={sigma_rate:.2%}, 结算期={tau}年, 名义本金={notional:,.0f}")

    cap_val = caplet_floorlet(F_rate, K_rate, T_cap, r_rf, sigma_rate, tau, notional, 'cap')
    floor_val = caplet_floorlet(F_rate, K_rate, T_cap, r_rf, sigma_rate, tau, notional, 'floor')
    print(f"  Caplet 价值 = {cap_val:,.2f}")
    print(f"  Floorlet 价值 = {floor_val:,.2f}")

    # 利率互换价值（浮动利率接受方）
    swap_pv = notional * tau * (F_rate - K_rate) * exp(-r_rf * (T_cap + tau))
    print(f"  Cap - Floor 差 = {cap_val-floor_val:,.2f},  互换价值 = {swap_pv:,.2f}  （应相等）")

    # 多期 Cap
    print(f"\n利率 Cap（4期，1年，季度结算，K={K_rate:.2%}）：")
    forward_curve = [(0.25, 0.051), (0.50, 0.053), (0.75, 0.055), (1.00, 0.057)]
    cap_total  = interest_rate_cap_floor(forward_curve, K_rate, r_rf, sigma_rate,
                                          tau, notional, 'cap')
    floor_total = interest_rate_cap_floor(forward_curve, K_rate, r_rf, sigma_rate,
                                           tau, notional, 'floor')
    print(f"  远期利率曲线：{forward_curve}")
    print(f"  Cap 总价值  = {cap_total:,.2f}")
    print(f"  Floor 总价值 = {floor_total:,.2f}")

    # ── 互换期权（Swaption）─────────────────────────────────
    print(f"\n互换期权（Swaption，Black-76）")
    F_sw, K_sw, T_exp, sigma_sw = 0.055, 0.05, 1.0, 0.20
    r_sw = 0.05
    # 年金因子：5 年互换，季度支付
    A = annuity_factor(r_sw, T_exp, T_exp + 5., payment_freq=4)
    print(f"  远期互换利率 F={F_sw:.2%}, 行权价 K={K_sw:.2%}, T_expiry={T_exp}年")
    print(f"  5年互换，季度付息，年金因子 A = {A:.4f}")

    sw_payer   = swaption_black76(F_sw, K_sw, T_exp, r_sw, sigma_sw, A, 'payer')
    sw_receiver = swaption_black76(F_sw, K_sw, T_exp, r_sw, sigma_sw, A, 'receiver')
    print(f"  支付型互换期权价值 = {sw_payer:.6f}")
    print(f"  接受型互换期权价值 = {sw_receiver:.6f}")
    # 检验：payer - receiver = 固定利率互换价值
    swap_val = A * (F_sw - K_sw)
    print(f"  Payer-Receiver = {sw_payer-sw_receiver:.6f},  互换价值 = {swap_val:.6f}（应相等）")

    # ── Vasicek 债券期权 ──────────────────────────────────
    print(f"\nVasicek (1977) 债券期权")
    r0, kappa_v, theta_v, sigma_v = 0.05, 0.30, 0.05, 0.03
    T_opt, T_bnd = 1., 5.   # 1 年期权，5 年债券

    P_1 = vasicek_bond_price(r0, T_opt, kappa_v, theta_v, sigma_v)
    P_5 = vasicek_bond_price(r0, T_bnd, kappa_v, theta_v, sigma_v)
    print(f"  参数: r₀={r0:.2%}, κ={kappa_v}, θ={theta_v:.2%}, σ={sigma_v:.2%}")
    print(f"  P(0,1) = {P_1:.4f},  P(0,5) = {P_5:.4f}")

    K_bond = 0.85   # 行权价格为债券价格的 0.85（单位：1元本金）
    vas_call = vasicek_bond_option(r0, K_bond, T_opt, T_bnd, kappa_v, theta_v, sigma_v, 'call')
    vas_put  = vasicek_bond_option(r0, K_bond, T_opt, T_bnd, kappa_v, theta_v, sigma_v, 'put')
    print(f"  债券期权 K={K_bond}: 看涨={vas_call:.6f}, 看跌={vas_put:.6f}")

    # ── Hull-White 债券期权 ───────────────────────────────
    print(f"\nHull-White 债券期权（使用市场折现因子）")
    # 从市场折现曲线读取
    r_curve = 0.05
    P_T1 = exp(-r_curve * T_opt)
    P_T5 = exp(-r_curve * T_bnd)
    kappa_hw = 0.10; sigma_hw = 0.01

    hw_call = hull_white_bond_option(P_T1, P_T5, K_bond, T_opt, T_bnd, kappa_hw, sigma_hw, 'call')
    hw_put  = hull_white_bond_option(P_T1, P_T5, K_bond, T_opt, T_bnd, kappa_hw, sigma_hw, 'put')
    print(f"  κ={kappa_hw}, σ={sigma_hw}, P(0,1)={P_T1:.4f}, P(0,5)={P_T5:.4f}")
    print(f"  Hull-White 看涨={hw_call:.6f}, 看跌={hw_put:.6f}")

    # ── Ho-Lee 债券期权 ───────────────────────────────────
    print(f"\nHo-Lee (1986) 债券期权（无均值回归）")
    sigma_hl = 0.01
    hl_call = ho_lee_bond_option(P_T1, P_T5, K_bond, T_opt, T_bnd, sigma_hl, 'call')
    hl_put  = ho_lee_bond_option(P_T1, P_T5, K_bond, T_opt, T_bnd, sigma_hl, 'put')
    print(f"  σ={sigma_hl}")
    print(f"  Ho-Lee 看涨={hl_call:.6f}, 看跌={hl_put:.6f}")
    # Put-Call 平价验证
    parity = P_T5 - K_bond * P_T1
    print(f"  Put-Call平价: C-P={hl_call-hl_put:.6f},  P(0,5)-K·P(0,1)={parity:.6f}")
