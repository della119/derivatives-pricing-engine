"""
05_barrier_options.py — 障碍期权 (Barrier Options)
===================================================
【模型简介】
障碍期权（Barrier Option）的存续取决于标的资产价格是否在期权
有效期内触及预设的障碍价格 H。

四种基本类型（8 个变种）：
  ─ 向下敲入看涨（Down-and-In Call, DIC） ：价格跌至 H 才生效
  ─ 向下敲出看涨（Down-and-Out Call, DOC）：价格跌至 H 即作废
  ─ 向上敲入看涨（Up-and-In Call, UIC）   ：价格涨至 H 才生效
  ─ 向上敲出看涨（Up-and-Out Call, UOC）  ：价格涨至 H 即作废
  ─ 以及对应的四种看跌期权

关键关系：敲入 + 敲出 = 普通欧式期权（无套利条件）

参考：Merton, R.C. (1973). "Theory of Rational Option Pricing." Bell J Econ.
     Reiner, E. & Rubinstein, M. (1991). "Breaking Down the Barriers."
       RISK Magazine, Vol. 4, 28–35.
书中对应：Haug (2007), Chapter 4, Section 4.17
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n


def _gbsm(S, K, T, r, b, sigma, opt='call'):
    if T <= 0: return max(S-K, 0.) if opt=='call' else max(K-S, 0.)
    d1 = (log(S/K)+(b+.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    cf, df = exp((b-r)*T), exp(-r*T)
    if opt == 'call': return S*cf*N(d1) - K*df*N(d2)
    return K*df*N(-d2) - S*cf*N(-d1)


def barrier_option(S: float, K: float, H: float, rebate: float,
                   T: float, r: float, b: float, sigma: float,
                   barrier_type: str = 'down-and-out-call') -> float:
    """
    标准连续障碍期权（Reiner & Rubinstein 1991）。

    参数
    ----
    S            : 当前标的资产价格
    K            : 行权价格
    H            : 障碍价格（触发水平）
    rebate       : 敲出时支付的补偿金（rebate），敲入未成时补偿
    T            : 距到期日时间（年）
    r            : 无风险利率
    b            : 持有成本
    sigma        : 年化波动率
    barrier_type : 下面 8 种之一（不区分大小写）：
                   'down-and-out-call' | 'down-and-in-call'
                   'up-and-out-call'   | 'up-and-in-call'
                   'down-and-out-put'  | 'down-and-in-put'
                   'up-and-out-put'    | 'up-and-in-put'

    返回
    ----
    float : 期权价值

    公式核心参数
    ------------
    μ = (b - σ²/2) / σ²
    λ = √(μ² + 2r/σ²)
    x₁ = ln(S/K)/(σ√T) + (1+μ)·σ√T          → 基于 K 的 d₁
    x₂ = ln(S/H)/(σ√T) + (1+μ)·σ√T          → 基于 H 的 d₁
    y₁ = ln(H²/(SK))/(σ√T) + (1+μ)·σ√T      → 反射障碍项
    y₂ = ln(H/S)/(σ√T)  + (1+μ)·σ√T         → 反射 H 项
    z  = ln(H/S)/(σ√T) + λ·σ√T               → rebate 项

    公式（8 种类型由符号 φ 和 η 控制）
    ─────────────────────────────────
    φ = +1（看涨）或 -1（看跌）
    η = +1（向下障碍）或 -1（向上障碍）

    用辅助函数 A, B, C, D, E, F 构建各类期权价格
    """
    if T <= 0:
        bt = barrier_type.lower()
        s_greater_h = S > H
        s_less_h    = S < H
        vanilla = max(S-K, 0.) if 'call' in bt else max(K-S, 0.)
        if 'down' in bt:
            if 'out' in bt: return 0. if s_less_h else vanilla
            if 'in'  in bt: return vanilla if s_less_h else 0.
        if 'up' in bt:
            if 'out' in bt: return 0. if s_greater_h else vanilla
            if 'in'  in bt: return vanilla if s_greater_h else 0.
        return vanilla

    bt = barrier_type.lower()

    # 确定方向符号
    phi = 1.0 if 'call' in bt else -1.0     # 看涨 +1，看跌 -1
    eta = 1.0 if 'down' in bt else -1.0     # 向下 +1，向上 -1

    # ── 核心参数 ─────────────────────────────────────────────────
    mu  = (b - 0.5*sigma**2) / sigma**2     # 漂移参数
    lam = sqrt(mu**2 + 2*r / sigma**2)      # 折现调整参数
    vol_t = sigma * sqrt(T)

    # 标准 d₁ 类参数
    x1 = log(S/K)  / vol_t + (1 + mu) * vol_t   # 基于 K
    x2 = log(S/H)  / vol_t + (1 + mu) * vol_t   # 基于 H

    # 反射（镜像）障碍参数：用 H²/(SK) 和 H/S
    y1 = log(H**2 / (S*K)) / vol_t + (1 + mu) * vol_t   # 基于 H²/(SK)
    y2 = log(H/S)  / vol_t + (1 + mu) * vol_t            # 基于 H/S

    # rebate 项
    z  = log(H/S)  / vol_t + lam * vol_t

    # ── 辅助构建块 ───────────────────────────────────────────────
    cf = exp((b-r)*T); df = exp(-r*T)
    mu_factor = (H/S)**(2*mu)       # (H/S)^{2μ}
    lam_factor = (H/S)**(mu+lam)    # (H/S)^{μ+λ}
    lam_factor2 = (H/S)**(mu-lam)   # (H/S)^{μ-λ}

    # A: 标准 BSM 看涨/跌（按 φ 决定方向）
    A = (phi * S * cf * N(phi*x1)
         - phi * K * df * N(phi*(x1 - vol_t)))

    # B: 以 H 为障碍的调整项
    B = (phi * S * cf * N(phi*x2)
         - phi * K * df * N(phi*(x2 - vol_t)))

    # C: 反射项（含 H 的幂次）
    C = (phi * S * cf * mu_factor * N(eta*y1)
         - phi * K * df * mu_factor * N(eta*(y1 - vol_t)))

    # D: 另一个反射项
    D = (phi * S * cf * mu_factor * N(eta*y2)
         - phi * K * df * mu_factor * N(eta*(y2 - vol_t)))

    # E: rebate 项（到期时）
    E = (rebate * df
         * (N(eta*(x2 - vol_t)) - mu_factor * N(eta*(y2 - vol_t))))

    # F: rebate 项（触及时立即支付）
    F = (rebate
         * (lam_factor * N(eta*z) + lam_factor2 * N(eta*(z - 2*lam*vol_t))))

    # ── 按障碍类型组合 ────────────────────────────────────────────
    if 'down' in bt and 'out' in bt and 'call' in bt:
        # Down-and-Out Call：H < K 时 C_di = A - B + C - D + E
        # H ≥ K（极少见）时直接用 E
        if H <= K:
            return A - B + C - D + E
        else:
            return B - D + E

    elif 'down' in bt and 'in' in bt and 'call' in bt:
        if H <= K:
            return B - C + D + E  # 注意：DOC+DIC = Vanilla → DIC = Vanilla - DOC
        else:
            vanilla = _gbsm(S, K, T, r, b, sigma, 'call')
            doc = max(barrier_option(S, K, H, rebate, T, r, b, sigma, 'down-and-out-call'), 0.)
            return max(vanilla - doc + rebate * exp(-r*T) * N(eta*(x2-vol_t)), 0.)

    elif 'up' in bt and 'out' in bt and 'call' in bt:
        if H >= K:
            return A - B + C - D + E
        else:
            return E

    elif 'up' in bt and 'in' in bt and 'call' in bt:
        vanilla = _gbsm(S, K, T, r, b, sigma, 'call')
        uoc = barrier_option(S, K, H, rebate, T, r, b, sigma, 'up-and-out-call')
        return vanilla - uoc

    elif 'down' in bt and 'out' in bt and 'put' in bt:
        if H <= K:
            return A - B + C - D + E
        else:
            return E

    elif 'down' in bt and 'in' in bt and 'put' in bt:
        vanilla = _gbsm(S, K, T, r, b, sigma, 'put')
        dop = barrier_option(S, K, H, rebate, T, r, b, sigma, 'down-and-out-put')
        return vanilla - dop

    elif 'up' in bt and 'out' in bt and 'put' in bt:
        if H >= K:
            return A - B + C - D + E
        else:
            return B - D + E

    elif 'up' in bt and 'in' in bt and 'put' in bt:
        vanilla = _gbsm(S, K, T, r, b, sigma, 'put')
        uop = barrier_option(S, K, H, rebate, T, r, b, sigma, 'up-and-out-put')
        return vanilla - uop

    raise ValueError(f"未知障碍类型：{barrier_type}")


def double_barrier_option(S: float, K: float, L: float, U: float,
                          T: float, r: float, b: float, sigma: float,
                          option_type: str = 'call',
                          rebate_L: float = 0.0,
                          rebate_U: float = 0.0,
                          n_terms: int = 5) -> float:
    """
    双障碍期权（Double Barrier Option）。

    在上障碍 U 和下障碍 L 之间存续；触及任一障碍则敲出。
    使用无穷级数展开（实际中取 n_terms = 5 已足够精确）。

    参数
    ----
    S         : 当前股价
    K         : 行权价
    L         : 下障碍（L < S < U）
    U         : 上障碍
    T         : 到期时间（年）
    r, b, sigma: 标准参数
    option_type: 'call' 或 'put'
    rebate_L  : 触及下障碍时的补偿
    rebate_U  : 触及上障碍时的补偿
    n_terms   : 级数展开项数（取 ±n）

    公式
    ----
    使用对称反射原理：
    C = Σ_{n=-N}^{N} [F_n(S,K,U,L) + G_n(S,U,L)]
    每项 F_n 是两个 BSM 类项的线性组合，G_n 为 rebate 贡献。
    """
    if T <= 0:
        vanilla = max(S-K, 0.) if option_type == 'call' else max(K-S, 0.)
        return vanilla if L < S < U else 0.

    if S <= L or S >= U:
        return 0.

    phi = 1. if option_type == 'call' else -1.
    vol_t = sigma * sqrt(T)
    mu    = (b - 0.5*sigma**2) / sigma**2
    cf = exp((b-r)*T); df = exp(-r*T)

    price = 0.
    for m in range(-n_terms, n_terms+1):
        lm = (U/L)**m                  # (U/L)^m
        # 位移量：m 次反射后的节点
        d1_n = log(S * lm**2 / K) / vol_t + (mu + 1)*vol_t
        d2_n = d1_n - vol_t
        d3_n = log(S * lm**2 / U) / vol_t + (mu + 1)*vol_t
        d4_n = d3_n - vol_t

        F_m = (phi * S * cf * lm**(2*(mu+1)) * (N(phi*d1_n) - N(phi*d3_n))
               - phi * K * df * lm**(2*mu)   * (N(phi*d2_n) - N(phi*d4_n)))
        price += F_m

    return max(price, 0.)


if __name__ == "__main__":
    print("=" * 60)
    print("障碍期权 — 数值示例（Haug Chapter 4, Section 4.17）")
    print("=" * 60)

    # 示例（Haug p.154）：S=100, K=100, H=95, T=0.5, r=0.10, b=0.10, σ=0.25, rebate=3
    S, K, H, rebate, T, r, b, sigma = 100, 100, 95, 3, 0.5, 0.10, 0.10, 0.25

    types = ['down-and-out-call', 'down-and-in-call',
             'up-and-out-call',   'up-and-in-call',
             'down-and-out-put',  'down-and-in-put',
             'up-and-out-put',    'up-and-in-put']

    # 上障碍用 H=110
    H_up = 110
    print(f"\nS={S}, K={K}, H_down={H}, H_up={H_up}, rebate={rebate}, T={T}, r={r}, b={b}, σ={sigma}")
    for bt in types:
        h = H if 'down' in bt else H_up
        v = barrier_option(S, K, h, rebate, T, r, b, sigma, bt)
        print(f"  {bt:<28} = {v:.4f}")

    # 敲入+敲出 = 普通欧式期权（验证）
    vanilla_call = _gbsm(S, K, T, r, b, sigma, 'call')
    dic = barrier_option(S, K, H, rebate, T, r, b, sigma, 'down-and-in-call')
    doc = barrier_option(S, K, H, rebate, T, r, b, sigma, 'down-and-out-call')
    print(f"\n平价验证：DIC + DOC = {dic + doc:.4f},  Vanilla = {vanilla_call:.4f}")

    # 双障碍期权
    print(f"\n双障碍看涨（L=95, U=115）= {double_barrier_option(S, K, 95, 115, T, r, b, sigma, 'call'):.4f}")
