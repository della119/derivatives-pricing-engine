"""
03_currency_translated_fx_options.py — 货币转换期权 / 外汇挂钩股票期权
======================================================================
【模型简介】
当投资者在一种货币下对以另一种货币计价的资产定价时，
需要考虑汇率风险（货币翻译风险）。本文件实现 Haug (2007)
第 5 章中与外汇相关的多资产期权：

1. 外国股票期权（本币结算）
   ── "外国股票期权以国内货币结算"
   ── 持有人同时承担股价涨跌和汇率波动两重风险

2. 固定汇率外国股票期权（Quanto 期权）
   ── 行权时汇率锁定为初始汇率（消除汇率风险）
   ── 广泛应用于国际市场间 Quanto ETF / Quanto Futures

3. 外汇-股票联动期权（Equity-Linked Foreign Exchange）
   ── 只有当外国股票价格上涨时，外汇期权才行权
   ── 本质：股票二元期权 + 条件外汇期权的组合

4. 并购套利外汇期权（Takeover Foreign Exchange Option）
   ── 并购中：支付本币、收到外国股票；
   ── 外汇期权只有在并购成功时（S_T > K_s）才行权

参考：
  - Reiner, E. (1992). "Quanto Mechanics." RISK Magazine.
  - Haug, E.G. (2007). "The Complete Guide to Option Pricing Formulas",
    Chapter 5, Section 5.6–5.9.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, cbnd


# ═══════════════════════════════════════════════════════════════
# 1. 外国股票期权（本币结算）
#    Foreign Equity Options Struck in Domestic Currency
# ═══════════════════════════════════════════════════════════════

def foreign_equity_option_domestic(S: float, K: float, T: float,
                                   r_d: float, r_f: float,
                                   sigma_S: float, sigma_X: float,
                                   rho: float,
                                   option_type: str = 'call') -> float:
    """
    外国股票期权，以国内货币结算（Domestic-Currency Foreign Equity Option）。

    期权以国内货币计价，标的是外国股票（以外国货币计价）。
    收益（本币） = X_T · max(S_T - K, 0)（看涨）
    其中 X_T 是到期时的汇率（外国货币/本国货币），
    这里假设已将外国股价折算为本国货币：
    S_d = X · S_f（以本国货币计量的外国股价）

    模型假设：
    - S  = X₀ · S_f₀（本国货币计的当前外国股价）
    - K  = 以本国货币计的行权价
    - sigma_S = 外国股票价格的波动率（外国货币下）
    - sigma_X = 汇率（X = 本/外）的波动率
    - rho  = S_f 与 X 的相关系数

    有效波动率（本国货币下外国股价的波动率）：
    σ_eff = √(σ_S² + σ_X² + 2·ρ·σ_S·σ_X)

    持有成本：b = r_d - r_f（无抛补利率平价）

    参数
    ----
    S       : 以本国货币计的外国股票当前价格（= X₀ · S_f₀）
    K       : 以本国货币计的行权价
    T       : 到期时间（年）
    r_d     : 国内无风险利率
    r_f     : 国外无风险利率
    sigma_S : 外国股票波动率（外国货币下）
    sigma_X : 汇率波动率
    rho     : 外国股票收益率与汇率变化的相关系数
    option_type : 'call' 或 'put'

    公式
    ----
    σ = √(σ_S² + σ_X² + 2·ρ·σ_S·σ_X)   ← 本币下的综合波动率
    b = r_d - r_f                          ← 持有成本
    d₁ = [ln(S/K) + (b + σ²/2)·T] / (σ√T)
    d₂ = d₁ - σ√T

    看涨：C = S·e^{(b-r_d)T}·N(d₁) - K·e^{-r_d·T}·N(d₂)
    看跌：P = K·e^{-r_d·T}·N(-d₂) - S·e^{(b-r_d)T}·N(-d₁)
    """
    if T <= 0:
        if option_type == 'call': return max(S - K, 0.)
        return max(K - S, 0.)

    # 综合波动率（本国货币下）
    sigma = sqrt(sigma_S**2 + sigma_X**2 + 2*rho*sigma_S*sigma_X)
    b = r_d - r_f  # 无抛补利率平价

    d1 = (log(S / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    df = exp(-r_d * T)
    cf = exp((b - r_d)*T)

    if option_type.lower() == 'call':
        return S * cf * N(d1) - K * df * N(d2)
    return K * df * N(-d2) - S * cf * N(-d1)


# ═══════════════════════════════════════════════════════════════
# 2. Quanto 期权（固定汇率外国股票期权）
#    Fixed Exchange Rate Foreign Equity Option (Quanto)
# ═══════════════════════════════════════════════════════════════

def quanto_option(S_f: float, K_f: float, T: float,
                  r_d: float, r_f: float,
                  sigma_S: float, sigma_X: float,
                  rho: float, X0: float,
                  option_type: str = 'call') -> float:
    """
    Quanto 期权（固定汇率外国股票期权）。

    Quanto 期权以国内货币结算，但汇率在期初锁定为 X₀
    （而非到期时的市场汇率 X_T）。
    这消除了期权持有人的汇率风险。

    收益（本币） = X₀ · max(S_f_T - K_f, 0)（看涨）

    关键调整：汇率-股票相关性使得外国股票在国内测度下的
    有效漂移需要加入 Quanto 调整项 -ρ·σ_S·σ_X（"Quanto 对冲"）。

    参数
    ----
    S_f     : 外国股票的当前价格（以外国货币计）
    K_f     : 以外国货币计的行权价
    T       : 到期时间（年）
    r_d     : 国内无风险利率
    r_f     : 国外无风险利率
    sigma_S : 外国股票（外国货币下）的年化波动率
    sigma_X : 汇率波动率
    rho     : 外国股票与汇率的相关系数（通常为负：外币贬值时外股下跌）
    X0      : 锁定的固定汇率（外/国内，用于将收益转换为本币）
    option_type : 'call' 或 'put'

    公式
    ----
    Quanto 调整后的持有成本：
    b = r_f - ρ·σ_S·σ_X      ← r_f 减去相关性调整

    d₁ = [ln(S_f/K_f) + (b + σ_S²/2)·T] / (σ_S·√T)
    d₂ = d₁ - σ_S·√T

    看涨：C = X₀ · e^{-r_d·T} · [S_f·e^{(b-r_f)T}·N(d₁) - K_f·N(d₂)]
              （折现到国内，以 X₀ 换算）

    等价写法：
    C = X₀ · [S_f·e^{-r_d·T}·e^{(b-r_f)T}·N(d₁) - K_f·e^{-r_d·T}·N(d₂)]
    """
    if T <= 0:
        payoff = max(S_f - K_f, 0.) if option_type == 'call' else max(K_f - S_f, 0.)
        return X0 * payoff

    # Quanto 持有成本（外国测度下调整）
    b = r_f - rho * sigma_S * sigma_X

    d1 = (log(S_f / K_f) + (b + 0.5*sigma_S**2)*T) / (sigma_S*sqrt(T))
    d2 = d1 - sigma_S*sqrt(T)

    # 国内折现因子 × 固定汇率 X₀
    df_d = exp(-r_d * T)
    # 外国股票的有效折现：e^{(b-r_f)T} = e^{-rho*sigma_S*sigma_X*T}
    cf   = exp((b - r_f)*T)

    if option_type.lower() == 'call':
        return X0 * df_d * (S_f * exp(b*T) * N(d1) - K_f * N(d2))
    return X0 * df_d * (K_f * N(-d2) - S_f * exp(b*T) * N(-d1))


# ═══════════════════════════════════════════════════════════════
# 3. 外汇-股票联动期权
#    Equity-Linked Foreign Exchange Option
# ═══════════════════════════════════════════════════════════════

def equity_linked_fx_option(S_f: float, X: float,
                             K_X: float, K_S: float,
                             T: float, r_d: float, r_f: float,
                             sigma_S: float, sigma_X: float,
                             rho: float,
                             option_type: str = 'call') -> float:
    """
    外汇-股票联动期权（Equity-Linked Foreign Exchange Option）。

    只有当外国股票价格 S_f_T > K_S 时，外汇期权才行权。
    即：需要同时满足股票上涨条件和外汇有利方向。

    收益（本币看涨）：S_f_T · max(X_T - K_X, 0)  如果 S_f_T > K_S
    注：S_f_T 起"数量"作用（面值调整），X_T 是汇率期权的标的。

    等价拆分：
    V = S_f · e^{(b_S - r_d)T} · M(d₁_S, d₁_X, ρ)
      - K_X · e^{-r_d·T} · S_f · e^{(b_S - r_d)T} · M(d₂_S, d₁_X - σ_X√T, ρ) / S_f
    （两种资产同时正向行权）

    简化版（Haug 2007）：
    条件外汇期权 = 外汇期权价格 × Prob(S_f_T > K_S | 联合分布)

    参数
    ----
    S_f     : 外国股票当前价格（外国货币）
    X       : 当前汇率（外国货币兑本国货币，即 1 外 = X 本）
    K_X     : 外汇期权的行权价（汇率）
    K_S     : 触发外汇期权行权的股票价格门槛
    T       : 到期时间
    r_d     : 国内无风险利率
    r_f     : 国外无风险利率
    sigma_S : 外国股票波动率
    sigma_X : 汇率波动率
    rho     : S_f 与 X 的相关系数
    option_type : 'call'（外汇看涨）或 'put'（外汇看跌）

    公式（双变量正态分布方法）
    ----
    b_S = r_f（外国股票持有成本），b_X = r_d - r_f（汇率持有成本）

    d₁_S  = [ln(S_f/K_S) + (b_S + σ_S²/2)·T] / (σ_S√T)
    d₂_S  = d₁_S - σ_S√T
    d₁_X  = [ln(X/K_X) + (b_X + σ_X²/2)·T] / (σ_X√T)
    d₂_X  = d₁_X - σ_X√T

    看涨（外汇call，股票上涨条件）：
    V = S_f · e^{(b_S-r_f)T} · X · e^{(b_X-r_d)T} · M(d₁_S, d₁_X, ρ)
      - K_X · e^{-r_d·T} · S_f · e^{(b_S-r_f)T} · M(d₂_S, d₂_X, ρ)
    （其中 S_f 提供"数量"，X 是外汇）
    """
    if T <= 0:
        # 两个条件同时满足
        stock_cond = S_f > K_S
        if option_type == 'call':
            fx_payoff = max(X - K_X, 0.)
        else:
            fx_payoff = max(K_X - X, 0.)
        return S_f * fx_payoff if stock_cond else 0.

    b_S = r_f          # 外国股票持有成本（外国测度下）
    b_X = r_d - r_f    # 汇率持有成本（无抛补利率平价）

    d1_S = (log(S_f / K_S) + (b_S + 0.5*sigma_S**2)*T) / (sigma_S*sqrt(T))
    d2_S = d1_S - sigma_S*sqrt(T)
    d1_X = (log(X / K_X) + (b_X + 0.5*sigma_X**2)*T) / (sigma_X*sqrt(T))
    d2_X = d1_X - sigma_X*sqrt(T)

    # S_f 的增长因子（以本国货币体系表达）
    # X₀ = X（当前汇率），将 S_f 转换到本国货币基础
    # 本函数返回以本国货币表示的期权价值
    df_d = exp(-r_d * T)

    if option_type.lower() == 'call':
        # 资产部分：X_T · S_f_T 的期望（价内部分）
        term1 = X * S_f * exp((b_X + b_S)*T - r_d*T) * cbnd(d1_S, d1_X, rho)
        # 现金部分：K_X · S_f_T 的期望（价内部分）
        term2 = K_X * S_f * exp((b_S - r_f)*T) * df_d * cbnd(d2_S, d2_X, rho)
        return term1 - term2
    else:
        # 外汇看跌：S_f_T · max(K_X - X_T, 0)
        term1 = K_X * S_f * exp((b_S - r_f)*T) * df_d * cbnd(d2_S, -d2_X, -rho)
        term2 = X * S_f * exp((b_X + b_S)*T - r_d*T) * cbnd(d1_S, -d1_X, -rho)
        return term1 - term2


# ═══════════════════════════════════════════════════════════════
# 4. 并购套利外汇期权
#    Takeover Foreign Exchange Option
# ═══════════════════════════════════════════════════════════════

def takeover_fx_option(S_f: float, X: float,
                       K_S: float, K_X: float,
                       T: float, r_d: float, r_f: float,
                       sigma_S: float, sigma_X: float,
                       rho: float) -> float:
    """
    并购套利外汇期权（Takeover Foreign Exchange Option）。

    并购套利背景：
    收购方（国内公司）计划收购外国公司，以固定价格 K_S 收购
    外国股票，但以本国货币结算。
    只有当外国股票价格 S_f_T > K_S（并购成功/有利可图）时，
    才会执行汇率对冲期权（外汇看涨）。

    收益 = max(X_T - K_X, 0)  如果 S_f_T > K_S（并购发生）
           0                   如果 S_f_T ≤ K_S（并购取消）

    这是一个"条件外汇期权"——外汇期权仅在特定股票条件下有效。

    与 equity_linked_fx_option 的区别：
    - 本函数中，S_f 仅作为触发条件（无数量调整）
    - equity_linked_fx_option 中，S_f_T 同时作为数量权重

    参数
    ----
    S_f     : 外国股票当前价格（外国货币）
    X       : 当前汇率
    K_S     : 并购触发价格（股票行权价）
    K_X     : 外汇期权行权价
    T       : 到期时间
    r_d     : 国内无风险利率
    r_f     : 国外无风险利率
    sigma_S : 外国股票波动率
    sigma_X : 汇率波动率
    rho     : 相关系数

    公式（联合概率方法）
    ----
    b_S = r_f，b_X = r_d - r_f

    d₁_S  = [ln(S_f/K_S) + (b_S + σ_S²/2)·T] / (σ_S√T)
    d₂_S  = d₁_S - σ_S√T
    d₁_X  = [ln(X/K_X) + (b_X + σ_X²/2)·T] / (σ_X√T)
    d₂_X  = d₁_X - σ_X√T

    V = X·e^{(b_X-r_d)T}·M(d₁_S, d₁_X, ρ) - K_X·e^{-r_d·T}·M(d₂_S, d₂_X, ρ)

    注：本函数返回本国货币计价的期权价格
    """
    if T <= 0:
        stock_cond = S_f > K_S
        return max(X - K_X, 0.) if stock_cond else 0.

    b_S = r_f
    b_X = r_d - r_f

    d1_S = (log(S_f / K_S) + (b_S + 0.5*sigma_S**2)*T) / (sigma_S*sqrt(T))
    d2_S = d1_S - sigma_S*sqrt(T)
    d1_X = (log(X / K_X)   + (b_X + 0.5*sigma_X**2)*T) / (sigma_X*sqrt(T))
    d2_X = d1_X - sigma_X*sqrt(T)

    df_d = exp(-r_d * T)

    # 资产部分：X_T 的条件期望（X_T > K_X 且 S_f_T > K_S）
    term1 = X * exp((b_X - r_d)*T) * cbnd(d1_S, d1_X, rho)
    # 现金部分：K_X 的条件期望
    term2 = K_X * df_d * cbnd(d2_S, d2_X, rho)

    return term1 - term2


# ═══════════════════════════════════════════════════════════════
# 5. 简化版 Quanto（GBM 直接法）
#    Quanto Forward / Quanto Futures Pricing
# ═══════════════════════════════════════════════════════════════

def quanto_forward(S_f: float, T: float,
                   r_d: float, r_f: float,
                   sigma_S: float, sigma_X: float,
                   rho: float, X0: float) -> float:
    """
    Quanto 远期价格（以本国货币结算的外国资产远期）。

    在 Quanto 框架下，国内测度（Q^d）下外国资产的漂移率为：
    μ^d = r_f - ρ·σ_S·σ_X

    Quanto 远期价格（本国货币计）：
    F_quanto = X₀ · S_f · e^{(r_f - ρ·σ_S·σ_X)·T} · e^{-r_d·T}
    ÷ e^{-r_d·T} （折现）= X₀ · S_f · e^{(r_f - ρ·σ_S·σ_X - r_d)·T}

    不折现的远期价格：
    F = X₀ · S_f · e^{(r_f - ρ·σ_S·σ_X)·T}

    参数
    ----
    返回：Quanto 远期的无折现价格（本国货币）
    """
    # Quanto 调整漂移：r_f - rho*sigma_S*sigma_X
    quanto_drift = r_f - rho * sigma_S * sigma_X
    return X0 * S_f * exp(quanto_drift * T)


if __name__ == "__main__":
    print("=" * 65)
    print("货币转换期权 / Quanto 期权 — 数值示例（Haug Chapter 5）")
    print("=" * 65)

    # ── 外国股票期权（本币结算）──────────────────────────────
    # S_d = X₀·S_f = 1.5·100 = 150（本国货币计），K=160，T=1
    # r_d=0.08, r_f=0.05, σ_S=0.20, σ_X=0.12, ρ=0.30
    S_d = 150.0  # 本国货币计的当前外国股价
    K_d = 160.0
    T   = 1.0
    r_d, r_f = 0.08, 0.05
    sigma_S, sigma_X = 0.20, 0.12
    rho = 0.30

    c_dom = foreign_equity_option_domestic(S_d, K_d, T, r_d, r_f,
                                           sigma_S, sigma_X, rho, 'call')
    p_dom = foreign_equity_option_domestic(S_d, K_d, T, r_d, r_f,
                                           sigma_S, sigma_X, rho, 'put')
    sigma_eff = (sigma_S**2 + sigma_X**2 + 2*rho*sigma_S*sigma_X)**0.5
    print(f"\n1. 外国股票期权（本币结算）")
    print(f"   S_d={S_d}, K={K_d}, T={T}, r_d={r_d}, r_f={r_f}")
    print(f"   σ_S={sigma_S}, σ_X={sigma_X}, ρ={rho}")
    print(f"   有效波动率 σ_eff = {sigma_eff:.4f}")
    print(f"   看涨 = {c_dom:.4f},  看跌 = {p_dom:.4f}")

    # 平价验证：C - P = S·e^{-r_f·T} - K·e^{-r_d·T}
    b = r_d - r_f
    from math import exp as mexp
    parity = S_d * mexp((b - r_d)*T) - K_d * mexp(-r_d*T)
    print(f"   Put-Call 平价 C-P = {c_dom - p_dom:.4f}, 理论 = {parity:.4f}")

    # ── Quanto 期权 ──────────────────────────────────────────
    # 外国股票 S_f=100（外币），K_f=100，T=1
    # r_d=0.08, r_f=0.05, σ_S=0.20, σ_X=0.12, ρ=-0.30（常见负相关）
    # 固定汇率 X₀=1.5
    S_f = 100.; K_f = 100.; X0 = 1.5; rho_q = -0.30

    qc = quanto_option(S_f, K_f, T, r_d, r_f, sigma_S, sigma_X, rho_q, X0, 'call')
    qp = quanto_option(S_f, K_f, T, r_d, r_f, sigma_S, sigma_X, rho_q, X0, 'put')
    b_q = r_f - rho_q * sigma_S * sigma_X
    print(f"\n2. Quanto 期权")
    print(f"   S_f={S_f}, K_f={K_f}, X₀={X0}, ρ={rho_q}")
    print(f"   Quanto 调整持有成本 b = {b_q:.4f}  (= r_f - ρ·σ_S·σ_X)")
    print(f"   看涨 = {qc:.4f},  看跌 = {qp:.4f}")

    # 不同相关系数对 Quanto 的影响
    print(f"   相关系数对 Quanto 看涨的影响：")
    for rho_test in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        v = quanto_option(S_f, K_f, T, r_d, r_f, sigma_S, sigma_X, rho_test, X0, 'call')
        b_test = r_f - rho_test * sigma_S * sigma_X
        print(f"     ρ={rho_test:+.1f}: b={b_test:.4f}, 看涨={v:.4f}")

    # ── 并购套利外汇期权 ──────────────────────────────────
    # 外国股票 S_f=100，K_S=105（并购触发），汇率 X=1.5，K_X=1.55
    # T=0.5, r_d=0.08, r_f=0.05, σ_S=0.20, σ_X=0.12, ρ=0.40
    S_f2=100.; X2=1.5; K_S2=105.; K_X2=1.55; T2=0.5; rho2=0.40

    tko = takeover_fx_option(S_f2, X2, K_S2, K_X2, T2, r_d, r_f,
                              sigma_S, sigma_X, rho2)
    print(f"\n4. 并购套利外汇期权")
    print(f"   S_f={S_f2}, X={X2}, K_S={K_S2}, K_X={K_X2}, T={T2}, ρ={rho2}")
    print(f"   并购条件外汇看涨价值 = {tko:.4f}")

    # Quanto 远期
    qfwd = quanto_forward(S_f, T, r_d, r_f, sigma_S, sigma_X, rho_q, X0)
    print(f"\n5. Quanto 远期价格（本国货币，不折现）= {qfwd:.4f}")
    print(f"   普通远期 = {X0 * S_f * mexp((r_d - r_f)*T):.4f}  （对比）")
