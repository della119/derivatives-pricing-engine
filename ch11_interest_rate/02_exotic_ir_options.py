"""
02_exotic_ir_options.py — 奇异利率期权
=======================================
【模型简介】
奇异利率期权是在标准利率衍生品基础上增加特殊条件的合约。
本文件实现：

A. 利率上限期权（Caption）
   ── 在上限（Cap）上的期权：以固定价格买/卖一个 Cap
   ── 也称为"期权的期权"（Compound Interest Rate Option）

B. 利率下限期权（Floortion）
   ── 在下限（Floor）上的期权

C. 利差期权（Spread Option on Rates）
   ── 收益基于两个利率之差（如 10Y - 2Y 利差）
   ── Bachelier 模型（利率利差可以为负，适合正态分布）

D. 利率区间累积期权（Range Accrual）
   ── 在利率位于特定区间 [L, U] 内时，每天累计利息

E. 数字利率期权（Digital / Binary Rate Option）
   ── 若到期利率高/低于行权利率，支付固定金额

F. 可赎回/可回售债券（Callable/Putable Bond）
   ── 通过 Hull-White 模型定价（嵌入债券期权）

参考：
  - Brigo, D. & Mercurio, F. (2006). "Interest Rate Models —
    Theory and Practice." Springer.
  - Hull, J. (2021). "Options, Futures, and Other Derivatives."
    Pearson.
书中对应：Haug (2007), Chapter 11
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp, pi
from utils.common import norm_cdf as N, norm_pdf as n


# ═══════════════════════════════════════════════════════════════
# A. 利率上限期权（Caption）
# ═══════════════════════════════════════════════════════════════

def caption(F_cap: float, K_cap_price: float, T_expiry: float,
             r: float, sigma_cap: float,
             option_type: str = 'call') -> float:
    """
    利率上限期权（Caption）定价——Cap 价格的欧式期权。

    Caption 赋予持有人以价格 K_cap_price 购买（看涨）或
    出售（看跌）一个利率上限（Cap）的权利。

    应用：
    - 借款人购买 Caption（看涨）：锁定未来购买利率保护的成本
    - 当预期利率波动率上升时有利

    定价：
    使用 Black-76 对 Cap 价格建模。
    F_cap = 当前 Cap 的市场价格（作为"期货价格"）
    K_cap_price = Caption 的行权价格（购买 Cap 的价格）

    Caption(call) = e^{-rT}[F_cap·N(d₁) - K_cap_price·N(d₂)]

    其中：
    d₁ = [ln(F_cap/K_cap_price) + σ_cap²T/2] / (σ_cap√T)
    d₂ = d₁ - σ_cap√T

    参数
    ----
    F_cap       : Cap 的当前价格（隐含远期价格）
    K_cap_price : Caption 行权价（购买 Cap 的固定价格）
    T_expiry    : Caption 到期时间
    sigma_cap   : Cap 价格的波动率
    """
    if T_expiry <= 0:
        if option_type == 'call': return max(F_cap - K_cap_price, 0.)
        return max(K_cap_price - F_cap, 0.)

    d1 = (log(F_cap / K_cap_price) + 0.5*sigma_cap**2*T_expiry) / (sigma_cap*sqrt(T_expiry))
    d2 = d1 - sigma_cap*sqrt(T_expiry)
    df = exp(-r*T_expiry)

    if option_type.lower() == 'call':
        return df * (F_cap * N(d1) - K_cap_price * N(d2))
    return df * (K_cap_price * N(-d2) - F_cap * N(-d1))


def floortion(F_floor: float, K_floor_price: float, T_expiry: float,
               r: float, sigma_floor: float,
               option_type: str = 'call') -> float:
    """
    利率下限期权（Floortion）定价——Floor 价格的欧式期权。

    与 Caption 完全对称，只是标的是 Floor 而非 Cap。
    实现与 Caption 相同，仅标签不同。
    """
    return caption(F_floor, K_floor_price, T_expiry, r, sigma_floor, option_type)


# ═══════════════════════════════════════════════════════════════
# B. 利率利差期权（Spread Option on Two Interest Rates）
# ═══════════════════════════════════════════════════════════════

def rate_spread_option(F1: float, F2: float, K: float,
                        T: float, r: float,
                        sigma1: float, sigma2: float, rho: float,
                        option_type: str = 'call') -> float:
    """
    利率利差期权（Interest Rate Spread Option）。

    收益（看涨）：max(F1_T - F2_T - K, 0)
    例如：10 年收益率 - 2 年收益率 > K（正向利差期权）

    利差 D = F1 - F2 可以为负（因此不能直接用 Black-76）。
    使用 Bachelier 模型（正态分布）定价：

    D_T ~ N(F1-F2, σ_D²T)
    其中 σ_D² = σ₁² + σ₂² - 2ρσ₁σ₂（利差波动率）

    Bachelier 公式：
    C = e^{-rT} · [(F1-F2-K)·N(d) + σ_D·√T·n(d)]
    d = (F1-F2-K) / (σ_D·√T)

    参数
    ----
    F1, F2  : 两个远期利率
    K       : 利差的行权价（如 K=0 表示利率反转期权）
    sigma1  : F1 的波动率（lognormal 近似）
    sigma2  : F2 的波动率
    rho     : 两利率的相关系数（通常正相关）
    """
    if T <= 0:
        spread = F1 - F2
        if option_type == 'call': return max(spread - K, 0.)
        return max(K - spread, 0.)

    # 利差波动率（Margrabe 近似）
    sigma_D = sqrt(max(sigma1**2 * F1**2 + sigma2**2 * F2**2
                       - 2.*rho*sigma1*sigma2*F1*F2, 1e-10)) / max(abs(F1 - F2), F1*0.01)

    # 实际：对于利率利差，通常直接用绝对波动率（Bachelier 模型）
    # sigma_D_abs = σ₁·F₁ + σ₂·F₂（近似，用于正态模型）
    sigma_D_abs = sqrt((sigma1*F1)**2 + (sigma2*F2)**2 - 2.*rho*sigma1*F1*sigma2*F2)

    spread = F1 - F2
    df = exp(-r*T)

    d = (spread - K) / (sigma_D_abs * sqrt(T))

    if option_type.lower() == 'call':
        return df * ((spread - K)*N(d) + sigma_D_abs*sqrt(T)*n(d))
    return df * ((K - spread)*N(-d) + sigma_D_abs*sqrt(T)*n(-d))


# ═══════════════════════════════════════════════════════════════
# C. 利率区间累积期权（Range Accrual Note / Option）
# ═══════════════════════════════════════════════════════════════

def range_accrual_note(F: float, L: float, U: float,
                        T: float, r: float, sigma: float,
                        face_value: float = 1.0,
                        coupon_rate: float = None,
                        n_obs: int = 252) -> dict:
    """
    利率区间累积票据（Range Accrual Note）近似定价。

    Range Accrual：每个观测日，若参考利率 r_t ∈ [L, U]，
    则累计利息（否则不计利息）。

    近似定价（对数正态远期利率分布假设）：
    E[在区间天数占比] ≈ ∫ Prob(L ≤ r_t ≤ U | 信息) 对时间积分

    在每个时间点 t，远期利率 F_t 服从对数正态：
    Prob(L ≤ F_t ≤ U) = N(d₂(U)) - N(d₂(L))
    其中 d₂(X) = [ln(F/X) - σ²t/2] / (σ√t)

    参数
    ----
    F          : 当前参考远期利率
    L, U       : 利率区间（累积利息的范围）
    T          : 票据期限（年）
    r          : 无风险利率（折现）
    sigma      : 利率波动率
    face_value : 票据面值
    coupon_rate: 区间内的票息率（若 None，近似为当前市场利率）
    n_obs      : 每年观测次数（252 = 每日，4 = 季度）

    返回
    ----
    dict: 包含预期区间天数、累积票息现值、票据价值
    """
    if coupon_rate is None:
        coupon_rate = F   # 默认使用当前远期利率

    dt_obs = T / n_obs   # 每次观测的时间间隔

    # 在每个观测时点 t_i 的区间概率
    prob_in_range = 0.
    for i in range(1, n_obs + 1):
        t_i = i * dt_obs

        # 在 t_i 时点，Prob(L ≤ F_{t_i} ≤ U)
        # 使用对数正态分布（Black 模型的前向分布）
        if sigma * sqrt(t_i) > 1e-8:
            # Prob(L ≤ F_t ≤ U) = N(d(U)) - N(d(L))
            # 注意：用 d₂ 形式（风险中性概率）
            def prob_above(X):
                if X <= 0: return 1.0
                d = (log(F/X) - 0.5*sigma**2*t_i) / (sigma*sqrt(t_i))
                return N(d)

            prob_i = prob_above(L) - prob_above(U)
        else:
            prob_i = 1. if L <= F <= U else 0.

        prob_in_range += max(prob_i, 0.)

    # 期望区间天数/年
    expected_ratio = prob_in_range / n_obs

    # 累积票息现值
    # 每日：discount to maturity
    coupon_pv = coupon_rate * T * expected_ratio * face_value * exp(-r*T)

    # 票据价值 = 面值现值 + 累积票息现值
    note_value = face_value * exp(-r*T) + coupon_pv

    return {
        'note_value': note_value,
        'coupon_pv': coupon_pv,
        'expected_in_range_ratio': expected_ratio,
        'expected_accrual_days': expected_ratio * T * 252,  # 折算为天数（年化）
    }


# ═══════════════════════════════════════════════════════════════
# D. 数字利率期权（Digital Rate Option / Binary）
# ═══════════════════════════════════════════════════════════════

def digital_rate_option(F: float, K: float, T: float,
                         r: float, sigma: float,
                         payment: float = 1.0,
                         option_type: str = 'call') -> float:
    """
    数字利率期权（Digital / Binary Interest Rate Option）。

    若到期时参考利率 F_T > K（看涨），支付固定金额 payment；
    否则支付 0。

    Black-76 数字期权定价：
    Digital_call = payment × e^{-rT} × N(d₂)
    其中 d₂ = [ln(F/K) - σ²T/2] / (σ√T)

    应用：
    - 保本结构性产品（当利率达标时支付票息）
    - 利率区间赌注

    参数
    ----
    payment : 行权时的固定支付金额（如 0.01 表示 1%）
    """
    if T <= 0:
        if option_type == 'call':
            return payment if F > K else 0.
        return payment if F < K else 0.

    d2 = (log(F/K) - 0.5*sigma**2*T) / (sigma*sqrt(T))
    df = exp(-r*T)

    if option_type.lower() == 'call':
        return payment * df * N(d2)     # N(d₂) = 风险中性 Prob(F_T > K)
    return payment * df * N(-d2)


# ═══════════════════════════════════════════════════════════════
# E. 可赎回 / 可回售债券（Callable / Putable Bond）
# ═══════════════════════════════════════════════════════════════

def callable_bond_value(coupon_rate: float, face: float,
                         T_maturity: float, T_call: float,
                         K_call: float,
                         r: float, kappa: float, sigma: float,
                         n_coupons: int = None) -> dict:
    """
    可赎回债券（Callable Bond）近似定价。

    可赎回债券 = 普通债券 - 内嵌赎回期权（Call Option）
    发行人拥有在 T_call 时以 K_call 赎回债券的权利。

    V_callable = V_straight_bond - V_call_option(bond)

    使用 Hull-White 框架的债券期权公式近似。

    参数
    ----
    coupon_rate : 票息率（年化，连续复利）
    face        : 面值
    T_maturity  : 债券到期时间
    T_call      : 赎回日期（< T_maturity）
    K_call      : 赎回价格（通常接近面值，如 1.0 或 1.01）
    r           : 当前短期利率
    kappa       : Hull-White 均值回归速度
    sigma       : 利率波动率
    n_coupons   : 到期前的票息次数（若 None，连续计算）
    """
    from ch11_interest_rate.interest_rate_options import (
        vasicek_bond_price, vasicek_bond_option
    )

    # 普通债券价值（零息债券加票息之和）
    if n_coupons is None:
        # 连续计算：积分近似为离散和
        n_coupons = int(T_maturity * 4)   # 季度付息

    dt_c = T_maturity / n_coupons
    straight_bond = 0.
    for i in range(1, n_coupons + 1):
        t_i = i * dt_c
        coupon_i = coupon_rate * dt_c * face
        discount_i = vasicek_bond_price(r, t_i, kappa, r, sigma)  # 近似
        straight_bond += coupon_i * discount_i

    # 加面值本金
    straight_bond += face * vasicek_bond_price(r, T_maturity, kappa, r, sigma)

    # 内嵌赎回期权（债券看涨期权）
    P_T_call = vasicek_bond_price(r, T_call, kappa, r, sigma)
    P_T_mat  = vasicek_bond_price(r, T_maturity, kappa, r, sigma)

    # 修正：赎回价 K_call 是债券的面值比例
    K_bond = K_call  # 实际是债券价格（绝对值）/面值

    # 简化：用 Vasicek 债券期权公式
    call_option = vasicek_bond_option(r, K_bond/face, T_call, T_maturity, kappa, r, sigma, 'call')
    call_option *= face  # 转换为绝对金额

    callable_value = straight_bond - call_option

    return {
        'straight_bond_value': straight_bond,
        'embedded_call_value': call_option,
        'callable_bond_value': callable_value,
        'call_premium': straight_bond - callable_value,
    }


def putable_bond_value(coupon_rate: float, face: float,
                        T_maturity: float, T_put: float,
                        K_put: float,
                        r: float, kappa: float, sigma: float) -> dict:
    """
    可回售债券（Putable Bond）近似定价。

    可回售债券 = 普通债券 + 内嵌回售期权（Put Option）
    持有人拥有在 T_put 时以 K_put 卖回债券的权利。

    V_putable = V_straight_bond + V_put_option(bond)
    """
    from ch11_interest_rate.interest_rate_options import (
        vasicek_bond_price, vasicek_bond_option
    )

    n_coupons = int(T_maturity * 4)
    dt_c = T_maturity / n_coupons
    straight_bond = 0.
    for i in range(1, n_coupons + 1):
        t_i = i * dt_c
        straight_bond += coupon_rate*dt_c*face * vasicek_bond_price(r, t_i, kappa, r, sigma)
    straight_bond += face * vasicek_bond_price(r, T_maturity, kappa, r, sigma)

    put_option = vasicek_bond_option(r, K_put/face, T_put, T_maturity, kappa, r, sigma, 'put')
    put_option *= face

    return {
        'straight_bond_value': straight_bond,
        'embedded_put_value': put_option,
        'putable_bond_value': straight_bond + put_option,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("奇异利率期权 — 数值示例（Haug Chapter 11）")
    print("=" * 65)

    # ── Caption（上限期权）────────────────────────────────────
    print(f"\nCaption（Cap 上的期权）：")
    F_cap    = 0.05   # 当前 2 年 ATM Cap 价格（以利率表示，这里简化为 cap 价值比例）
    K_cap    = 0.04   # Caption 行权价（购买 Cap 的价格）
    T_caption = 0.5   # Caption 到期时间
    r_disc   = 0.04   # 折现利率
    sigma_cap = 0.25  # Cap 价格波动率

    cap_call = caption(F_cap, K_cap, T_caption, r_disc, sigma_cap, 'call')
    cap_put  = caption(F_cap, K_cap, T_caption, r_disc, sigma_cap, 'put')
    print(f"  F_cap={F_cap:.2%}, K={K_cap:.2%}, T={T_caption}, σ={sigma_cap:.0%}")
    print(f"  Caption 看涨（右买 Cap）= {cap_call:.6f}")
    print(f"  Caption 看跌（右卖 Cap）= {cap_put:.6f}")

    # ── 利率利差期权 ──────────────────────────────────────────
    print(f"\n利率利差期权（10年-2年利差）：")
    F10, F2 = 0.040, 0.025   # 10年、2年远期利率
    K_spread = 0.010          # 行权利差：1%
    T_sp = 1.0
    sig10, sig2, rho_sr = 0.20, 0.25, 0.85  # 利率高相关

    spread_c = rate_spread_option(F10, F2, K_spread, T_sp, r_disc, sig10, sig2, rho_sr, 'call')
    spread_p = rate_spread_option(F10, F2, K_spread, T_sp, r_disc, sig10, sig2, rho_sr, 'put')
    curr_spread = F10 - F2
    print(f"  F10={F10:.2%}, F2={F2:.2%}, 当前利差={curr_spread:.2%}")
    print(f"  行权利差 K={K_spread:.2%}, T={T_sp}年, ρ={rho_sr}")
    print(f"  利差看涨（利差扩大）= {spread_c:.6f}")
    print(f"  利差看跌（利差收窄）= {spread_p:.6f}")

    # 曲线倒挂期权（K=0，利差变为负数则获益）
    inversion = rate_spread_option(F10, F2, 0., T_sp, r_disc, sig10, sig2, rho_sr, 'put')
    print(f"  曲线倒挂期权（K=0 看跌利差）= {inversion:.6f}")

    # ── 区间累积票据 ──────────────────────────────────────────
    print(f"\n利率区间累积票据（Range Accrual）：")
    F_rn, L_rn, U_rn = 0.045, 0.030, 0.060  # 参考利率=4.5%，区间[3%, 6%]
    T_rn = 1.; r_rn = 0.04; sig_rn = 0.20
    face_rn = 1.0; cpn_rn = 0.05  # 5% 票息

    ran = range_accrual_note(F_rn, L_rn, U_rn, T_rn, r_rn, sig_rn,
                              face_rn, cpn_rn, n_obs=252)
    print(f"  参考利率={F_rn:.2%}, 区间=[{L_rn:.2%}, {U_rn:.2%}]")
    print(f"  T={T_rn}年, 票息={cpn_rn:.2%}/年, 面值={face_rn}")
    print(f"  预期在区间内占比    = {ran['expected_in_range_ratio']:.4f} ({ran['expected_in_range_ratio']*100:.1f}%)")
    print(f"  预期累积天数        = {ran['expected_accrual_days']:.1f} 天")
    print(f"  票息现值            = {ran['coupon_pv']:.6f}")
    print(f"  票据总价值          = {ran['note_value']:.6f}")

    # ── 数字利率期权 ──────────────────────────────────────────
    print(f"\n数字利率期权（Binary Rate Option）：")
    F_dig, K_dig = 0.045, 0.05
    T_dig, sig_dig = 0.5, 0.20
    pmt = 0.01   # 到期支付 1%

    dig_c = digital_rate_option(F_dig, K_dig, T_dig, r_disc, sig_dig, pmt, 'call')
    dig_p = digital_rate_option(F_dig, K_dig, T_dig, r_disc, sig_dig, pmt, 'put')
    print(f"  F={F_dig:.2%}, K={K_dig:.2%}, T={T_dig}, payment={pmt:.2%}")
    print(f"  数字看涨（利率↑超 K 得款）= {dig_c:.6f}")
    print(f"  数字看跌（利率↓低 K 得款）= {dig_p:.6f}")
    print(f"  验证: C+P = {dig_c+dig_p:.6f}，应 = payment×e^{{-rT}} = {pmt*exp(-r_disc*T_dig):.6f}")

    # ── 可赎回债券 ────────────────────────────────────────────
    print(f"\n可赎回债券（Callable Bond）：")
    r0 = 0.05; kappa_cb = 0.20; sigma_cb = 0.02
    cpn_cb = 0.06; face_cb = 100.; T_mat = 5.; T_call_cb = 2.; K_call_cb = 102.

    try:
        cb = callable_bond_value(cpn_cb, face_cb, T_mat, T_call_cb, K_call_cb,
                                  r0, kappa_cb, sigma_cb)
        print(f"  票息率={cpn_cb:.0%}, 面值={face_cb}, 到期={T_mat}年")
        print(f"  赎回日={T_call_cb}年, 赎回价={K_call_cb}")
        print(f"  普通债券价值   = {cb['straight_bond_value']:.4f}")
        print(f"  内嵌赎回期权   = {cb['embedded_call_value']:.4f}")
        print(f"  可赎回债券价值 = {cb['callable_bond_value']:.4f}")
    except Exception as e:
        print(f"  [计算中: {e}]")
