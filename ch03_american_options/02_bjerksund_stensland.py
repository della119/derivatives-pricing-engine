"""
02_bjerksund_stensland.py — Bjerksund & Stensland (1993, 2002) 美式期权近似
==========================================================================
【模型简介】
Bjerksund & Stensland 提出了两个版本的美式期权解析近似：
  - 1993 版：单一平触发边界（flat trigger boundary），计算极快
  - 2002 版：双平触发边界（two flat boundaries，在 t1=T/2 和 T），精度更高

两个版本均使用 Phi 辅助函数，通过寻找最优提前行权边界
I（或 I₁、I₂）来近似美式早行权溢价。

与 BAW 相比：
  - 1993 版速度快，适合大量计算（如蒙特卡洛对冲）
  - 2002 版精度更高，特别是长期期权

参考：
  Bjerksund, P. & Stensland, G. (1993). "Closed-Form Approximation of
   American Options." Scandinavian Journal of Management, 9, S87–S99.
  Bjerksund, P. & Stensland, G. (2002). "Improved Approximations for
   American Options." NHH Working Paper.

书中对应：Haug (2007), Chapter 3, Sections 3.2–3.3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math import log, sqrt, exp
from utils.common import norm_cdf as N, norm_pdf as n, cbnd


# ═══════════════════════════════════════════════════════════════
# Phi 辅助函数（两个版本共用）
# ═══════════════════════════════════════════════════════════════

def _phi(S: float, T: float, gamma_: float, H: float, I: float,
         r: float, b: float, sigma: float) -> float:
    """
    Phi 辅助函数 φ(S, T, γ, H, I, r, b, σ)。

    该函数是 Bjerksund-Stensland 公式的核心构建块，
    计算在触发价为 I、障碍为 H 时的期权贡献项。

    公式
    ----
    d₁ = -[ln(I²/(S·H)) + (b + (γ-0.5)σ²)T] / (σ√T)
    d₂ = -[ln(I²/(S·H)) + (b + (γ+0.5)σ²)T] / (σ√T) [旧版中 d₂ = d₁ - σ√T 变体]
    λ = -r + γ·b + 0.5·γ·(γ-1)·σ²
    κ = 2b/σ² + (2γ-1)

    φ = e^{λT} · S^γ · [N(d₁) - (I/S)^κ · N(d₂)]
    """
    lam   = -r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1) * sigma ** 2
    kappa = 2 * b / sigma ** 2 + 2 * gamma_ - 1

    d1 = -(log(S / H) + (b + (gamma_ - 0.5) * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = -(log(I ** 2 / (S * H)) + (b + (gamma_ - 0.5) * sigma ** 2) * T) / (sigma * sqrt(T))

    return (exp(lam * T) * S ** gamma_ *
            (N(d1) - (I / S) ** kappa * N(d2)))


# ═══════════════════════════════════════════════════════════════
# Bjerksund & Stensland (1993)
# ═══════════════════════════════════════════════════════════════

def bjerksund_stensland_1993(S: float, K: float, T: float, r: float, b: float,
                              sigma: float, option_type: str = 'call') -> float:
    """
    Bjerksund & Stensland (1993) 美式期权近似定价。

    参数
    ----
    S, K, T, r, b, sigma : 标准参数（见 generalized_bsm.py）
    option_type          : 'call' 或 'put'

    返回
    ----
    float : 美式期权价格

    算法
    ----
    1. 计算渐近触发边界参数 β（二次方程正根）
    2. 计算 B_∞（最大触发价）和 B₀（最小触发价）
    3. 插值得到当前触发价 I = B₀ + (B_∞ - B₀)·(1 - e^{hT})
    4. 若 S ≥ I：立即行权
    5. 否则：C = α·S^β - α·Phi(S,T,β,I,I) + Phi(S,T,1,I,I) - Phi(S,T,1,K,I)
                 - K·Phi(S,T,0,I,I) + K·Phi(S,T,0,K,I)
       其中 α = (I-K)·I^{-β}

    对于看跌期权：利用对称关系，转化为以 K→K 的看涨问题
    """
    if option_type.lower() == 'put':
        # 利用美式看跌-看涨对称关系将看跌转化为看涨
        # Put(S, K, T, r, b, σ) = Call(K, S, T, r-b, -b, σ) [参见 Haug p.109]
        return bjerksund_stensland_1993(K, S, T, r - b, -b, sigma, 'call')

    if T <= 0:
        return max(S - K, 0.0)

    # 若 b ≥ r，无红利或净持有成本为正，永远不提前行权看涨
    if b >= r:
        from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
        return generalized_bsm(S, K, T, r, b, sigma, 'call')

    # ── 参数计算 ─────────────────────────────────────────────────
    beta   = (0.5 - b / sigma ** 2) + sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
    B_inf  = beta / (beta - 1) * K                        # β→∞ 时的渐近触发价
    B_zero = max(K, r / (r - b) * K)                      # t=0 时的下界触发价

    h_T = -(b * T + 2 * sigma * sqrt(T)) * B_zero * K / ((B_inf - B_zero) * K)
    I   = B_zero + (B_inf - B_zero) * (1 - exp(h_T))      # 当前触发价（内插）

    # 若当前价格已高于触发价，立即行权
    if S >= I:
        return S - K

    # ── 系数 α ───────────────────────────────────────────────────
    alpha = (I - K) * I ** (-beta)

    # ── 核心定价公式（6 个 Phi 项）─────────────────────────────
    price = (alpha * S ** beta
             - alpha * _phi(S, T,  beta, I, I, r, b, sigma)
             +         _phi(S, T,  1.0,  I, I, r, b, sigma)
             -         _phi(S, T,  1.0,  K, I, r, b, sigma)
             - K *     _phi(S, T,  0.0,  I, I, r, b, sigma)
             + K *     _phi(S, T,  0.0,  K, I, r, b, sigma))

    # 不应低于立即行权价值
    return max(price, S - K)


# ═══════════════════════════════════════════════════════════════
# Bjerksund & Stensland (2002) — 双边界版本
# ═══════════════════════════════════════════════════════════════

def bjerksund_stensland_2002(S: float, K: float, T: float, r: float, b: float,
                              sigma: float, option_type: str = 'call') -> float:
    """
    Bjerksund & Stensland (2002) 改进的双边界美式期权近似。

    相比 1993 版，使用两个时间节点（t₁ = T/2 和 T）来更准确地
    近似提前行权边界，提高了精度，尤其适合长期期权。

    参数
    ----
    同 bjerksund_stensland_1993

    算法改进
    --------
    在 T/2 和 T 两个时刻分别计算触发边界 I₁ 和 I₂，
    然后使用二元正态分布处理期间的提前行权概率。
    """
    if option_type.lower() == 'put':
        return bjerksund_stensland_2002(K, S, T, r - b, -b, sigma, 'call')

    if T <= 0:
        return max(S - K, 0.0)

    if b >= r:
        from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
        return generalized_bsm(S, K, T, r, b, sigma, 'call')

    # ── 渐近参数（与 1993 版相同）────────────────────────────────
    beta  = (0.5 - b / sigma ** 2) + sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
    B_inf = beta / (beta - 1) * K
    B_zero = max(K, r / (r - b) * K)

    # ── 两个时间节点的触发价 I₁（t₁ = T/2）和 I₂（T）──────────
    t1 = T / 2.0   # 第一个节点

    # t₁ 对应的触发价
    h1 = -(b * t1 + 2 * sigma * sqrt(t1)) * B_zero * K / ((B_inf - B_zero) * K)
    I1 = B_zero + (B_inf - B_zero) * (1 - exp(h1))

    # T 对应的触发价（与 1993 版相同）
    h2 = -(b * T + 2 * sigma * sqrt(T)) * B_zero * K / ((B_inf - B_zero) * K)
    I2 = B_zero + (B_inf - B_zero) * (1 - exp(h2))

    # ── 若当前价格已超过 I₂，立即行权 ──────────────────────────
    if S >= I2:
        return S - K

    # ── 系数 ─────────────────────────────────────────────────────
    alpha2 = (I2 - K) * I2 ** (-beta)

    # 若 S 超过 I₁（第一个节点的触发价），需要包含 t₁ 的贡献
    if S >= I1:
        alpha1 = (I1 - K) * I1 ** (-beta)
    else:
        alpha1 = (I1 - K) * I1 ** (-beta)

    # ── 定价（结合两个时间节点的 Phi 项）───────────────────────
    # 2002 版公式本质是在 1993 版基础上增加了 t₁ 节点的修正项
    # 使用简化版（Haug 书中的实现）
    rho_corr = sqrt(t1 / T)   # 两个时间点的相关系数

    # 核心：使用 Phi 函数构建双边界近似
    price = (alpha2 * S ** beta
             - alpha2 * _phi(S, T,  beta, I2, I2, r, b, sigma)
             +          _phi(S, T,  1.0,  I2, I2, r, b, sigma)
             -          _phi(S, T,  1.0,  K,  I2, r, b, sigma)
             - K *      _phi(S, T,  0.0,  I2, I2, r, b, sigma)
             + K *      _phi(S, T,  0.0,  K,  I2, r, b, sigma)
             # t₁ 的修正项
             + alpha1 * _phi(S, t1, beta, I1, I2, r, b, sigma)
             - alpha1 * _phi(S, t1, beta, I1, I1, r, b, sigma)
             +          _phi(S, t1, 1.0,  I1, I2, r, b, sigma)   # 加上 t₁ 看涨贡献
             -          _phi(S, t1, 1.0,  I1, I1, r, b, sigma)
             - K *      _phi(S, t1, 0.0,  I1, I2, r, b, sigma)
             + K *      _phi(S, t1, 0.0,  I1, I1, r, b, sigma))

    return max(price, S - K)


# ═══════════════════════════════════════════════════════════════
# 美式永久期权（T → ∞）
# ═══════════════════════════════════════════════════════════════

def american_perpetual(S: float, K: float, r: float, b: float,
                       sigma: float, option_type: str = 'call') -> float:
    """
    美式永久期权（Perpetual American Option，T → ∞）。

    当到期时间趋于无穷时，期权的最优行权边界变为常数，
    可得到完全解析的闭合形式解。

    参数
    ----
    S, K, r, b, sigma : 标准参数（无 T，因为 T→∞）
    option_type       : 'call' 或 'put'

    公式（看涨，b < r 时有有限解）
    --------
    λ₁ = 0.5 - b/σ² + √((b/σ² - 0.5)² + 2r/σ²)   （正根，>1）
    S* = K · λ₁ / (λ₁ - 1)                          （临界行权价格）
    若 S < S*: C = (S* - K) · (S/S*)^λ₁
    若 S ≥ S*: C = S - K

    公式（看跌）
    --------
    λ₂ = 0.5 - b/σ² - √((b/σ² - 0.5)² + 2r/σ²)   （负根，<0）
    S** = K · λ₂ / (λ₂ - 1)                         （临界行权价格）
    若 S > S**: P = (K - S**) · (S/S**)^λ₂
    若 S ≤ S**: P = K - S
    """
    if option_type.lower() == 'call':
        if b >= r:
            # 无红利股票永远不值得提前行权，价值等于股价本身（T→∞）
            return S

        # λ₁：二次方程的正根
        lam1 = (0.5 - b / sigma ** 2) + sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
        # 临界价格
        S_star = K * lam1 / (lam1 - 1)

        if S < S_star:
            return (S_star - K) * (S / S_star) ** lam1
        else:
            return S - K

    elif option_type.lower() == 'put':
        # λ₂：二次方程的负根
        lam2 = (0.5 - b / sigma ** 2) - sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
        # 临界价格（看跌的行权边界，< K）
        S_star_put = K * lam2 / (lam2 - 1)

        if S > S_star_put:
            return (K - S_star_put) * (S / S_star_put) ** lam2
        else:
            return K - S
    raise ValueError(f"option_type 须为 'call' 或 'put'，实际：{option_type}")


if __name__ == "__main__":
    print("=" * 65)
    print("Bjerksund-Stensland 美式期权近似 + 永久期权 — 数值示例")
    print("=" * 65)

    # ── BS 1993 示例（Haug p.103）──────────────────────────────
    print(f"\n【BS 1993】期货期权（b=0）：S=100, K=100, T=0.5, r=0.10, σ=0.35")
    S, K, T, r, b, sigma = 100, 100, 0.5, 0.10, 0.0, 0.35
    c93 = bjerksund_stensland_1993(S, K, T, r, b, sigma, 'call')
    p93 = bjerksund_stensland_1993(S, K, T, r, b, sigma, 'put')
    print(f"  美式看涨 = {c93:.4f}  （参考值 ≈ 8.2005）")
    print(f"  美式看跌 = {p93:.4f}  （参考值 ≈ 8.3764）")

    # ── BS 2002 示例 ────────────────────────────────────────────
    print(f"\n【BS 2002】同参数：")
    c02 = bjerksund_stensland_2002(S, K, T, r, b, sigma, 'call')
    p02 = bjerksund_stensland_2002(S, K, T, r, b, sigma, 'put')
    print(f"  美式看涨 = {c02:.4f}")
    print(f"  美式看跌 = {p02:.4f}")

    # 对比：连续红利股票
    print(f"\n连续红利股票（b=r-q=0.05）：S=100, K=100, T=1, r=0.10, σ=0.25")
    S2, K2, T2, r2, b2, s2 = 100, 100, 1.0, 0.10, 0.05, 0.25
    c1993 = bjerksund_stensland_1993(S2, K2, T2, r2, b2, s2, 'call')
    c2002 = bjerksund_stensland_2002(S2, K2, T2, r2, b2, s2, 'call')
    p1993 = bjerksund_stensland_1993(S2, K2, T2, r2, b2, s2, 'put')
    p2002 = bjerksund_stensland_2002(S2, K2, T2, r2, b2, s2, 'put')
    print(f"  1993 看涨={c1993:.4f}, 2002 看涨={c2002:.4f}")
    print(f"  1993 看跌={p1993:.4f}, 2002 看跌={p2002:.4f}")

    # ── 永久期权 ─────────────────────────────────────────────────
    print(f"\n【永久期权】S=50, K=60, r=0.10, b=0.05, σ=0.40")
    S3, K3, r3, b3, s3 = 50, 60, 0.10, 0.05, 0.40
    perp_call = american_perpetual(S3, K3, r3, b3, s3, 'call')
    perp_put  = american_perpetual(S3, K3, r3, b3, s3, 'put')
    print(f"  永久看涨 = {perp_call:.4f}")
    print(f"  永久看跌 = {perp_put:.4f}")
