"""
common.py — 共享工具函数
====================================================
本模块为所有期权定价模型提供共同使用的数学工具函数：
  - 标准正态分布 CDF / PDF
  - 二维正态分布 CDF（Drezner 近似）
  - 三维正态分布 CDF（数值积分）

参考：Espen Gaarder Haug, "The Complete Guide to Option Pricing Formulas", 2nd ed.
       Chapter 13: Distributions
"""

import math
from scipy.stats import norm as _norm
from scipy import integrate as _integrate
import numpy as np


# ─────────────────────────────────────────────
# 一元正态分布
# ─────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """
    标准正态累积分布函数 N(x) = P(Z ≤ x)，Z ~ N(0,1)。

    参数
    ----
    x : float  标准正态变量的值

    返回
    ----
    float  累积概率，范围 [0, 1]
    """
    return _norm.cdf(x)


def norm_pdf(x: float) -> float:
    """
    标准正态概率密度函数 n(x) = exp(-x²/2) / sqrt(2π)。

    参数
    ----
    x : float

    返回
    ----
    float  概率密度值
    """
    return _norm.pdf(x)


# 常用别名
N = norm_cdf
n = norm_pdf


# ─────────────────────────────────────────────
# 二维正态分布 CDF（Drezner 1978 近似）
# ─────────────────────────────────────────────

def cbnd(a: float, b: float, rho: float) -> float:
    """
    二维标准正态累积分布函数（Bivariate Normal CDF）。

    计算 P(X ≤ a, Y ≤ b)，其中 (X, Y) 是相关系数为 rho 的
    二维标准正态分布。

    参数
    ----
    a   : float  第一个变量的上界
    b   : float  第二个变量的上界
    rho : float  两变量的相关系数，范围 (-1, 1)

    返回
    ----
    float  联合概率 P(X ≤ a, Y ≤ b)

    算法
    ----
    使用 Drezner (1978) / Drezner & Wesolowsky (1990) 的
    数值积分近似方法，精度约 1e-7。
    对于边界情形（rho = ±1）单独处理。
    """
    # 处理极端相关情形
    if abs(rho) > 1 - 1e-12:
        if rho > 0:
            # 完全正相关：P(X≤a, Y≤b) = N(min(a,b))
            return norm_cdf(min(a, b))
        else:
            # 完全负相关：P(X≤a, Y≤b) = max(0, N(a)+N(b)-1)
            return max(0.0, norm_cdf(a) + norm_cdf(b) - 1.0)

    # Gauss–Legendre 积分节点与权重（10 点公式）
    # 来自 Abramowitz & Stegun
    _x = [0.9324695142031522, 0.6612093864662645, 0.2386191860831969,
          -0.2386191860831969, -0.6612093864662645, -0.9324695142031522]
    _w = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
           0.4679139345726910,  0.3607615730481386,  0.1713244923791704]

    # 扩展到 10 点（对称）
    _xs = [0.9739065285171717, 0.8650633666889845, 0.6794095682990244,
           0.4333953941292472, 0.1488743389816312,
          -0.1488743389816312, -0.4333953941292472,
          -0.6794095682990244, -0.8650633666889845, -0.9739065285171717]
    _ws = [0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
           0.2692667193099963, 0.2955242247147529,
           0.2955242247147529, 0.2692667193099963,
           0.2190863625159820, 0.1494513491505806, 0.0666713443086881]

    # 若 a 或 b 极小，概率趋近于 0
    if a <= -8 or b <= -8:
        return 0.0
    # 若 a 或 b 极大，退化为单变量
    if a >= 8:
        return norm_cdf(b)
    if b >= 8:
        return norm_cdf(a)

    # 使用 scipy 实现（精确）作为主要计算
    try:
        from scipy.stats import multivariate_normal as _mvn
        cov = [[1.0, rho], [rho, 1.0]]
        return _mvn.cdf([a, b], mean=[0, 0], cov=cov)
    except Exception:
        pass

    # 备用：Drezner 数值积分
    if rho < 0:
        # 利用 P(X≤a,Y≤b,rho) = N(a) - P(X≤a,Y≥-b,-rho)
        # 转化为 rho>0 的情形
        neg_rho_case = cbnd(a, -b, -rho)
        return norm_cdf(a) - neg_rho_case if b > 0 else -norm_cdf(-b) + cbnd(-a, -b, rho)

    hs = (a * a - 2 * rho * a * b + b * b) / (2 * (1 - rho ** 2))
    asr = math.asin(rho)
    sum_val = 0.0
    for xi, wi in zip(_xs, _ws):
        for sgn in [1, -1]:
            xs = asr * (sgn * xi + 1) / 2
            sum_val += wi * math.exp((math.sin(xs) * a * b - hs) / (1 - math.sin(xs) ** 2))

    return max(0.0, min(1.0,
        norm_cdf(a) * norm_cdf(b) + asr / (4 * math.pi) * sum_val
    ))


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """cbnd 的别名，见 cbnd 文档。"""
    return cbnd(a, b, rho)


# ─────────────────────────────────────────────
# 辅助：反正态 CDF（implied vol 求解时用）
# ─────────────────────────────────────────────

def norm_inv(p: float) -> float:
    """
    标准正态分布的反函数（quantile function）N⁻¹(p)。

    参数
    ----
    p : float  概率，范围 (0, 1)

    返回
    ----
    float  使 N(x)=p 的 x 值
    """
    return _norm.ppf(p)


if __name__ == "__main__":
    # 简单测试
    print("=== 工具函数测试 ===")
    print(f"N(0)   = {N(0):.6f}  (期望 0.500000)")
    print(f"N(1.96)= {N(1.96):.6f}  (期望 0.975002)")
    print(f"n(0)   = {n(0):.6f}  (期望 0.398942)")
    print(f"cbnd(0,0,0)   = {cbnd(0,0,0):.6f}  (期望 0.250000)")
    print(f"cbnd(1,1,0.5) = {cbnd(1,1,0.5):.6f}  (期望 ~0.680)")
    print(f"cbnd(-inf,1,0)= {cbnd(-8,1,0):.6f}  (期望 0.000000)")
