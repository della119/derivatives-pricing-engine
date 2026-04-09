"""
09_soft_barrier_partial_options.py — 软障碍期权 & 分段障碍期权
==============================================================
【模型简介】

A. 软障碍期权（Soft Barrier Options）
───────────────────────────────────
标准障碍期权在价格触及障碍时立即失效（"硬"障碍）。
软障碍期权（Hart & Ross 1994, Rubinstein & Reiner 1991）
在障碍区间 [L, H] 内逐渐失效：

敲出比例 = (H - S) / (H - L)（下敲出，S ∈ [L, H] 时）

实际上软障碍是多个行权价不同的硬障碍期权的积分组合。
闭合解通过对障碍价格积分得到。

B. 部分时间障碍期权（Partial Time Barrier Options）
──────────────────────────────────────────────────
标准障碍期权在整个期权有效期内监控障碍。
部分时间障碍期权只在部分时间段监控：

类型1（Start 型）：[0, t₁]（期初监控，t₁ < T）
类型2（End 型）：  [t₁, T]（期末监控，t₁ > 0）

Heynen & Kat (1994) 给出闭合解析公式，使用二元正态分布。

C. 外部障碍期权（Outside Barrier Options）
───────────────────────────────────────────
障碍条件由资产 S₂ 触发，但收益基于资产 S₁（Heynen & Kat 1994）：
- 当 S₂_t 触及障碍 H 时，S₁ 的期权被激活或失效

D. 杠杆式障碍期权（Leveraged Barrier, 逐渐消亡型）
────────────────────────────────────────────────────
期权价值随着股价接近障碍而逐渐减少（非跳跃式）。

参考：
  - Heynen, R.C. & Kat, H.M. (1994). "Partial Barrier Options."
    Journal of Financial Engineering, 3(3/4), 253–274.
  - Hart, I. & Ross, M. (1994). "Striking Continuity."
    RISK, June, 51–56.
书中对应：Haug (2007), Chapter 4, Sections 4.20–4.24
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n, cbnd


def _gbsm(S, K, T, r, b, sigma, cp):
    """广义 BSM 辅助函数，cp=1 看涨，cp=-1 看跌。"""
    if T <= 0: return max(cp*(S-K), 0.)
    d1 = (log(S/K) + (b+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return cp*(S*exp((b-r)*T)*N(cp*d1) - K*exp(-r*T)*N(cp*d2))


# ═══════════════════════════════════════════════════════════════
# A. 软障碍期权（Soft / Gradual Barrier）
# ═══════════════════════════════════════════════════════════════

def soft_barrier_option(S: float, K: float,
                         L: float, U: float,
                         T: float, r: float, b: float, sigma: float,
                         barrier_type: str = 'down-out',
                         option_type: str = 'call') -> float:
    """
    软障碍期权（Soft Barrier Option, Hart & Ross 1994）。

    在 [L, U] 区间内：
    敲出：若 S ∈ [L, U]，期权名义价值按比例缩减
          若 S < L（下穿），完全失效
          若 S > U（从未进入），完全有效

    缩减比例（下敲出）：
    若 L ≤ S ≤ U：remaining = (S - L) / (U - L)
    若 S > U：remaining = 1（未触及）
    若 S < L：remaining = 0（完全失效）

    闭合解（Rubinstein & Reiner 1991 推广）：
    软障碍期权 = ∫_L^U (1 - (H-S)/(H-L)) × 点击障碍的条件期权价格

    近似实现：使用两个硬障碍期权的差：
    C_soft(L, U) = C_hard(L) - C_hard(U) × [(价值分配)]

    更精确：通过 Heaviside 积分
    C_soft = C_vanilla - [C_L - C_U × (U-K)/(U-L)] × 对应调整

    参数
    ----
    L : 软障碍下限
    U : 软障碍上限（L < U，宽度 = U - L）
    barrier_type : 'down-out' 或 'up-out'
    """
    if T <= 0:
        intrinsic = max(S - K, 0.) if option_type == 'call' else max(K - S, 0.)
        # 应用软障碍
        if barrier_type == 'down-out':
            if S < L: return 0.
            if S <= U: return intrinsic * (S - L) / (U - L)
        else:
            if S > U: return 0.
            if S >= L: return intrinsic * (U - S) / (U - L)
        return intrinsic

    cp = 1 if option_type == 'call' else -1

    # 软障碍闭合近似（两个硬障碍期权的线性组合）
    # 软障碍 ≈ (1/(U-L)) × ∫_L^U C_hard(H') dH'
    # 通过辛普森数值积分近似
    n_quad = 20
    dH = (U - L) / n_quad
    total = 0.
    for i in range(n_quad + 1):
        H_i = L + i * dH
        w = 1. if i == 0 or i == n_quad else (2. if i % 2 == 0 else 4.)
        # 以 H_i 为障碍的硬障碍期权价格
        c_hard_i = hard_barrier_at(S, K, H_i, T, r, b, sigma, barrier_type, option_type)
        total += w * c_hard_i * dH / 3.

    # 软障碍 = 积分 / (U-L) × 名义缩减权重
    return total / (U - L)


def hard_barrier_at(S: float, K: float, H: float, T: float,
                     r: float, b: float, sigma: float,
                     barrier_type: str = 'down-out',
                     option_type: str = 'call') -> float:
    """
    标准硬障碍期权定价（Reiner-Rubinstein 1991 公式，简化版）。
    内部辅助函数。
    """
    if T <= 0:
        intrinsic = max(S - K, 0.) if option_type == 'call' else max(K - S, 0.)
        if 'down' in barrier_type and 'out' in barrier_type:
            return intrinsic if S > H else 0.
        if 'up' in barrier_type and 'out' in barrier_type:
            return intrinsic if S < H else 0.
        return intrinsic

    eta = 1. if 'down' in barrier_type else -1.
    phi = 1. if option_type == 'call' else -1.
    lam = sqrt((b/sigma - 0.5*sigma)**2 + 2.*r/sigma**2) if r > 0 else abs(b/sigma - 0.5*sigma)

    x1 = log(S/K)/(sigma*sqrt(T)) + (1. + lam)*sigma*sqrt(T) if K > 0 else 0.
    x2 = log(S/H)/(sigma*sqrt(T)) + (1. + lam)*sigma*sqrt(T)
    y1 = log(H**2/(S*K))/(sigma*sqrt(T)) + (1.+lam)*sigma*sqrt(T)
    y2 = log(H/S)/(sigma*sqrt(T)) + (1.+lam)*sigma*sqrt(T)

    mu  = (b - 0.5*sigma**2) / sigma**2
    df  = exp(-r*T)
    cfS = exp((b-r)*T)

    A1 = phi*S*cfS*N(phi*x1) - phi*K*df*N(phi*(x1-sigma*sqrt(T)))
    A2 = phi*S*cfS*N(phi*x2) - phi*K*df*N(phi*(x2-sigma*sqrt(T)))
    B1 = phi*S*cfS*(H/S)**(2*(mu+1))*N(eta*y1) - phi*K*df*(H/S)**(2*mu)*N(eta*(y1-sigma*sqrt(T)))
    B2 = phi*S*cfS*(H/S)**(2*(mu+1))*N(eta*y2) - phi*K*df*(H/S)**(2*mu)*N(eta*(y2-sigma*sqrt(T)))

    if 'down' in barrier_type and 'out' in barrier_type:
        if S <= H: return 0.
        if K > H:
            return A1 - B1
        else:
            return A2 - B2 + (A1 - A2) - (B1 - B2)
    elif 'up' in barrier_type and 'out' in barrier_type:
        if S >= H: return 0.
        if K < H:
            return A1 - B1
        else:
            return 0.
    elif 'down' in barrier_type and 'in' in barrier_type:
        vanilla = _gbsm(S, K, T, r, b, sigma, phi)
        return vanilla - hard_barrier_at(S, K, H, T, r, b, sigma, 'down-out', option_type)
    else:  # up-in
        vanilla = _gbsm(S, K, T, r, b, sigma, phi)
        return vanilla - hard_barrier_at(S, K, H, T, r, b, sigma, 'up-out', option_type)


# ═══════════════════════════════════════════════════════════════
# B. 部分时间障碍期权（Partial Time Barrier）
# ═══════════════════════════════════════════════════════════════

def partial_time_barrier_option(S: float, K: float, H: float,
                                  T: float, t1: float,
                                  r: float, b: float, sigma: float,
                                  barrier_type: str = 'down-out-end',
                                  option_type: str = 'call') -> float:
    """
    部分时间障碍期权（Partial Time Barrier, Heynen & Kat 1994）。

    只在部分时间区间内监控障碍：
    - 'down-out-start': [0, t₁] 期间监控下障碍（t₁ < T）
    - 'down-out-end':   [t₁, T] 期间监控下障碍（t₁ > 0）
    - 'up-out-start', 'up-out-end': 上障碍对应版本

    核心：Heynen & Kat 使用二元正态分布 cbnd 给出半解析公式。

    参数
    ----
    T  : 期权到期时间
    t1 : 障碍监控起始/终止时间
    barrier_type : 障碍类型（含 start 或 end）
    """
    if T <= 0:
        intrinsic = max(S - K, 0.) if option_type == 'call' else max(K - S, 0.)
        return intrinsic

    # 标记：方向
    if 'down' in barrier_type:
        eta = 1.
        phi = 1. if option_type == 'call' else -1.
    else:
        eta = -1.
        phi = 1. if option_type == 'call' else -1.

    lam = (b/sigma + 0.5*sigma)    # 实际上是 b/sigma + sigma/2
    mu  = (b - 0.5*sigma**2)
    df  = exp(-r*T)
    cfS = exp((b-r)*T)

    # 相关系数（两段时间）
    rho_t = sqrt(t1 / T)   # corr(W_{t1}, W_T) = √(t1/T)

    # 关键参数
    d1  = (log(S/K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2  = d1 - sigma*sqrt(T)
    f1  = (log(S/H) + (b + 0.5*sigma**2)*t1) / (sigma*sqrt(t1))
    f2  = f1 - sigma*sqrt(t1)
    g1  = (log(S/H) + (b + 0.5*sigma**2)*T)  / (sigma*sqrt(T))
    g2  = g1 - sigma*sqrt(T)
    h1  = (log(H**2/(S*K)) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    h2  = h1 - sigma*sqrt(T)
    e1  = (log(H/S) - (b + 0.5*sigma**2)*t1) / (sigma*sqrt(t1))
    e2  = e1 + sigma*sqrt(t1)

    if 'end' in barrier_type:
        # 监控区间 [t1, T]（后段监控）
        # 期初有效，到期可能失效
        if 'down' in barrier_type and 'out' in barrier_type:
            if S <= H: return 0.
            # Heynen-Kat 公式（简化版）
            A = phi*S*cfS*(N(phi*d1) - cbnd(phi*g1, eta*e1, -phi*eta*rho_t))
            B = phi*K*df*(N(phi*d2) - cbnd(phi*g2, eta*e2, -phi*eta*rho_t))
            C = phi*S*cfS*(H/S)**(2*(b/sigma**2+0.5)) * cbnd(eta*(g1 + 2*log(H/S)/(sigma*sqrt(T))),
                                                                -eta*f1, -phi*eta*rho_t)
            D = phi*K*df*(H/S)**(2*b/sigma**2-1) * cbnd(eta*(g2 + 2*log(H/S)/(sigma*sqrt(T))),
                                                          -eta*f2, -phi*eta*rho_t)
            return max(A - B - C + D, 0.)
        elif 'up' in barrier_type and 'out' in barrier_type:
            if S >= H: return 0.
            eta_u = -1.
            A = phi*S*cfS*(N(phi*d1) - cbnd(phi*g1, eta_u*e1, -phi*eta_u*rho_t))
            B = phi*K*df*(N(phi*d2) - cbnd(phi*g2, eta_u*e2, -phi*eta_u*rho_t))
            return max(A - B, 0.)
        else:
            # 敲入 = vanilla - 敲出
            vanilla = _gbsm(S, K, T, r, b, sigma, phi)
            ko_type = barrier_type.replace('in', 'out')
            ko = partial_time_barrier_option(S, K, H, T, t1, r, b, sigma, ko_type, option_type)
            return max(vanilla - ko, 0.)

    else:
        # 监控区间 [0, t1]（前段监控）
        if 'down' in barrier_type and 'out' in barrier_type:
            if S <= H: return 0.
            A = phi*S*cfS*(N(phi*d1) - cbnd(phi*f1, eta*d1, phi*eta*rho_t))
            B = phi*K*df*(N(phi*d2) - cbnd(phi*f2, eta*d2, phi*eta*rho_t))
            C = phi*S*cfS*(H/S)**(2*(b/sigma**2+0.5)) * (cbnd(-eta*f1, eta*h1, -phi*eta*rho_t))
            D = phi*K*df*(H/S)**(2*b/sigma**2-1) * (cbnd(-eta*f2, eta*h2, -phi*eta*rho_t))
            return max(A - B - C + D, 0.)
        elif 'up' in barrier_type and 'out' in barrier_type:
            if S >= H: return 0.
            eta_u = -1.
            A = phi*S*cfS*(N(phi*d1) - cbnd(phi*f1, eta_u*d1, phi*eta_u*rho_t))
            B = phi*K*df*(N(phi*d2) - cbnd(phi*f2, eta_u*d2, phi*eta_u*rho_t))
            return max(A - B, 0.)
        else:
            vanilla = _gbsm(S, K, T, r, b, sigma, phi)
            ko_type = barrier_type.replace('in', 'out')
            ko = partial_time_barrier_option(S, K, H, T, t1, r, b, sigma, ko_type, option_type)
            return max(vanilla - ko, 0.)


# ═══════════════════════════════════════════════════════════════
# C. 两障碍期权（Double Barrier）— 解析级数解
# ═══════════════════════════════════════════════════════════════

def double_barrier_option(S: float, K: float,
                           L: float, U: float,
                           T: float, r: float, b: float, sigma: float,
                           option_type: str = 'call',
                           n_terms: int = 5) -> float:
    """
    双障碍期权（Double Barrier Option, Ikeda & Kunitomo 1992）。

    同时设有上障碍 U 和下障碍 L（L < S < U）：
    只要价格触及 L 或 U 中的任何一个，期权失效（双敲出）。

    精确级数解（反射原理）：
    C = Σ_{n=-∞}^{∞} [A_n - B_n]

    其中每一项通过 BSM 形式计算，利用反射原理处理两个吸收边界。

    收敛很快（n_terms = 5 通常足够）。
    """
    if T <= 0:
        intrinsic = max(S - K, 0.) if option_type == 'call' else max(K - S, 0.)
        return intrinsic if L < S < U else 0.

    if S <= L or S >= U:
        return 0.

    phi = 1. if option_type == 'call' else -1.
    lam = (b - 0.5*sigma**2) / sigma**2

    # 级数方法（Ikeda-Kunitomo）
    total = 0.
    log_LU = log(U / L)

    for n in range(-n_terms, n_terms + 1):
        d1n = (log(S * (U/L)**(2*n) / K) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2n = d1n - sigma*sqrt(T)
        e1n = (log(L**(2*n+2) / (S * K * U**(2*n))) + (b + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        e2n = e1n - sigma*sqrt(T)

        factor_A = (U/L)**(n*(b/sigma**2 - 0.5))
        factor_B = (L/S)**(2*(n*b/sigma**2 - n + 0.5*n**2*sigma**2/sigma**2))

        term_A = (factor_A * (phi*S*exp((b-r)*T)*N(phi*d1n) - phi*K*exp(-r*T)*N(phi*d2n)))
        term_B = (factor_B * (phi*S*exp((b-r)*T)*(L/S)**(2*(b/sigma**2+0.5))*N(phi*e1n)
                              - phi*K*exp(-r*T)*(L/S)**(2*b/sigma**2)*N(phi*e2n)))

        total += (term_A - term_B)

    return max(total, 0.)


if __name__ == "__main__":
    print("=" * 65)
    print("软障碍 & 部分时间障碍期权 — 数值示例（Haug Chapter 4）")
    print("=" * 65)

    S, K, T, r, b, sigma = 100., 100., 1.0, 0.08, 0.04, 0.25

    # BSM 普通看涨（参考）
    bsm_c = _gbsm(S, K, T, r, b, sigma, 1)
    print(f"\n参数：S={S}, K={K}, T={T}, r={r}, b={b}, σ={sigma}")
    print(f"BSM 普通看涨 = {bsm_c:.4f}")

    # ── 软障碍期权 ────────────────────────────────────────────
    print(f"\n软障碍期权（下敲出，软障碍区间 [85, 95]）：")
    L, U = 85., 95.
    H_hard = 90.   # 中间值的硬障碍对比
    soft = soft_barrier_option(S, K, L, U, T, r, b, sigma, 'down-out', 'call')
    hard = hard_barrier_at(S, K, H_hard, T, r, b, sigma, 'down-out', 'call')
    print(f"  软障碍 [L={L}, U={U}]  看涨 = {soft:.4f}")
    print(f"  硬障碍 H={H_hard}        看涨 = {hard:.4f}  （参考）")
    print(f"  BSM 普通看涨             = {bsm_c:.4f}")

    # ── 部分时间障碍期权（后段监控）────────────────────────
    print(f"\n部分时间障碍期权（后半段 [t1, T] 监控，t1=T/2）：")
    H_pt = 95.
    t1 = T / 2.  # 后半段监控

    pt_end = partial_time_barrier_option(S, K, H_pt, T, t1, r, b, sigma,
                                          'down-out-end', 'call')
    pt_start = partial_time_barrier_option(S, K, H_pt, T, t1, r, b, sigma,
                                            'down-out-start', 'call')
    full_hard = hard_barrier_at(S, K, H_pt, T, r, b, sigma, 'down-out', 'call')

    print(f"  H={H_pt}, t1={t1}（半程）")
    print(f"  全程障碍（连续监控）   = {full_hard:.4f}")
    print(f"  前段监控 [0, t1]       = {pt_start:.4f}  (应 ≥ 全程)")
    print(f"  后段监控 [t1, T]       = {pt_end:.4f}   (应 ≥ 全程，因监控时间短)")
    print(f"  BSM 普通看涨           = {bsm_c:.4f}")

    # ── 双障碍期权 ────────────────────────────────────────────
    print(f"\n双障碍期权（L=85, U=120，下敲出+上敲出）：")
    L_db, U_db = 85., 120.
    db_call = double_barrier_option(S, K, L_db, U_db, T, r, b, sigma, 'call')
    db_put  = double_barrier_option(S, K, L_db, U_db, T, r, b, sigma, 'put')
    print(f"  看涨 = {db_call:.4f}  看跌 = {db_put:.4f}")

    # 不同障碍宽度的影响
    print(f"\n  障碍宽度对双障碍看涨的影响（对称障碍 S±d）：")
    for d in [5, 10, 15, 20, 30, 50]:
        v = double_barrier_option(S, K, S-d, S+d, T, r, b, sigma, 'call')
        print(f"    ±{d}: L={S-d}, U={S+d}: 双障碍看涨 = {v:.4f}")
    print(f"    无障碍: 普通 BSM = {bsm_c:.4f}")
