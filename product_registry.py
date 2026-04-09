"""
Derivatives Product Registry
=============================
Defines all derivative products with metadata for the pricing engine UI.
"""
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("vanilla",        "Vanilla Options"),
    ("american",       "American Options"),
    ("exotic_single",  "Exotic — Single Asset"),
    ("exotic_two",     "Exotic — Two Assets"),
    ("alt_models",     "Alternative Models"),
    ("numerical",      "Numerical Methods"),
    ("commodity",      "Commodity & Energy"),
    ("interest_rate",  "Interest Rate"),
    ("discrete_div",   "Discrete Dividends"),
]


# ---------------------------------------------------------------------------
# Param helper
# ---------------------------------------------------------------------------
def P(key, label, default, min_val, max_val, step, help_text=""):
    return {
        "key": key, "label": label, "default": default,
        "min": min_val, "max": max_val, "step": step, "help": help_text,
    }

# Common parameter sets
_S   = P("S", "Spot Price (S)", 100.0, 0.01, 1e6, 1.0, "Current underlying price")
_K   = P("K", "Strike Price (K)", 100.0, 0.01, 1e6, 1.0, "Option strike price")
_T   = P("T", "Time to Expiry (T)", 1.0, 0.001, 30.0, 0.01, "Years to expiration")
_r   = P("r", "Risk-free Rate (r)", 0.05, -0.20, 1.0, 0.001, "Annualized risk-free rate")
_b   = P("b", "Cost of Carry (b)", 0.05, -1.0, 1.0, 0.001, "Cost-of-carry parameter")
_sig = P("sigma", "Volatility (σ)", 0.20, 0.001, 5.0, 0.01, "Annualized volatility")
_q   = P("q", "Dividend Yield (q)", 0.02, 0.0, 1.0, 0.001, "Continuous dividend yield")


# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------
PRODUCTS = OrderedDict()

# ========================= VANILLA OPTIONS =========================

PRODUCTS["black_scholes"] = {
    "name": "Black-Scholes (1973)",
    "category": "vanilla",
    "desc": "European options on non-dividend-paying stocks — the foundational option pricing model.",
    "module": "ch01_black_scholes_merton.01_black_scholes_1973",
    "func": "black_scholes",
    "params": [_S, _K, _T, _r, _sig],
    "has_option_type": True,
    "formula": r"C = S \, N(d_1) - K \, e^{-rT} N(d_2)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K) + (r + \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """The Black-Scholes (1973) model derives a closed-form solution for European option prices by
constructing a self-financing, continuously rebalanced hedge portfolio. Under risk-neutral pricing,
the expected payoff is discounted at the risk-free rate. The model solves the Black-Scholes PDE
subject to the terminal payoff boundary condition.""",
    "assumptions": [
        "The underlying follows geometric Brownian motion (GBM) with constant drift and volatility",
        "No dividends are paid during the option's life",
        "Markets are frictionless — no transaction costs, taxes, or short-selling restrictions",
        "Continuous trading is possible",
        "The risk-free interest rate is constant over the option's life",
        "Returns are log-normally distributed",
    ],
    "limitations": [
        "Cannot capture the implied volatility smile/skew observed in markets",
        "Not applicable for dividend-paying stocks (use Merton 1973 instead)",
        "Assumes continuous hedging which is impossible in practice",
        "Underestimates tail risk due to the log-normal assumption",
        "Constant volatility assumption is unrealistic for longer-dated options",
    ],
    "recommendations": "Best suited for short-dated European options on non-dividend stocks. For dividend-paying stocks, use Merton (1973). For volatility smile effects, consider Heston or SABR models.",
}

PRODUCTS["merton_1973"] = {
    "name": "Merton (1973) — Continuous Dividend",
    "category": "vanilla",
    "desc": "European options on assets paying a continuous dividend yield q.",
    "module": "ch01_black_scholes_merton.02_merton_1973",
    "func": "merton",
    "params": [_S, _K, _T, _r, _q, _sig],
    "has_option_type": True,
    "formula": r"C = S \, e^{-qT} N(d_1) - K \, e^{-rT} N(d_2)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K) + (r - q + \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """Merton (1973) extends Black-Scholes to assets paying a continuous dividend yield q.
The spot price is replaced by the dividend-adjusted forward S·e^{(r-q)T}, reflecting that the holder
forgoes dividend income. Equivalent to generalized BSM with cost-of-carry b = r − q.""",
    "assumptions": [
        "Continuous dividend yield q is constant over the option's life",
        "All Black-Scholes assumptions apply (GBM, constant vol, no frictions)",
        "Dividends are reinvested continuously in the underlying",
    ],
    "limitations": [
        "Continuous dividend assumption may be inaccurate for individual stocks with discrete dividends",
        "Better suited for index options where aggregate dividends approximate a continuous yield",
        "Same volatility smile limitations as Black-Scholes",
    ],
    "recommendations": "Use for equity index options or stocks with frequent small dividends. For stocks with known discrete dividends, use the discrete dividend models (Ch.9).",
}

PRODUCTS["black_76"] = {
    "name": "Black-76 — Futures Options",
    "category": "vanilla",
    "desc": "European options on futures or forward contracts.",
    "module": "ch01_black_scholes_merton.03_black_76",
    "func": "black_76",
    "params": [
        P("F", "Futures Price (F)", 100.0, 0.01, 1e6, 1.0, "Current futures/forward price"),
        _K, _T, _r, _sig,
    ],
    "has_option_type": True,
    "formula": r"C = e^{-rT}\bigl[F \, N(d_1) - K \, N(d_2)\bigr]",
    "formula_detail": r"""d_1 = \frac{\ln(F/K) + \tfrac{\sigma^2}{2}T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """Black (1976) adapts the Black-Scholes framework for options on futures/forwards.
Since the futures price already incorporates carry costs, the cost-of-carry is zero (b = 0).
The futures price replaces the spot price, and only discounting at the risk-free rate remains.""",
    "assumptions": [
        "Futures price follows GBM with constant volatility",
        "No margin requirements affect option pricing",
        "Risk-free rate is constant",
        "European exercise only",
    ],
    "limitations": [
        "Does not account for daily mark-to-market of futures margins",
        "Constant volatility assumption — same smile/skew limitations",
        "For American-style futures options, use BAW or binomial tree with b=0",
    ],
    "recommendations": "Standard model for exchange-traded commodity, energy, and interest rate futures options. Widely used for cap/floor and swaption pricing in interest rate markets.",
}

PRODUCTS["garman_kohlhagen"] = {
    "name": "Garman-Kohlhagen — FX Options",
    "category": "vanilla",
    "desc": "European options on foreign exchange rates.",
    "module": "ch01_black_scholes_merton.05_garman_kohlhagen",
    "func": "garman_kohlhagen",
    "params": [
        P("S", "Spot FX Rate (S)", 1.20, 0.0001, 1e4, 0.0001, "Current spot exchange rate"),
        P("K", "Strike Rate (K)", 1.20, 0.0001, 1e4, 0.0001, "Strike exchange rate"),
        _T,
        P("r_d", "Domestic Rate (r_d)", 0.05, -0.20, 1.0, 0.001, "Domestic risk-free rate"),
        P("r_f", "Foreign Rate (r_f)", 0.03, -0.20, 1.0, 0.001, "Foreign risk-free rate"),
        _sig,
    ],
    "has_option_type": True,
    "formula": r"C = S \, e^{-r_f T} N(d_1) - K \, e^{-r_d T} N(d_2)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K) + (r_d - r_f + \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """Garman-Kohlhagen (1983) extends BSM to FX options by treating the foreign interest rate
as a continuous dividend yield. The cost-of-carry is b = r_d − r_f (interest rate differential).
The model prices the right to exchange domestic currency for foreign currency at a fixed rate.""",
    "assumptions": [
        "FX rate follows GBM with constant volatility",
        "Both domestic and foreign risk-free rates are constant",
        "No capital controls or transaction costs",
        "Covered interest rate parity holds",
    ],
    "limitations": [
        "FX markets exhibit pronounced volatility smile — constant vol is a simplification",
        "Interest rate differentials may vary significantly over the option's life",
        "Does not capture jump risk in FX rates (e.g., central bank interventions)",
    ],
    "recommendations": "Standard model for vanilla FX options. For smile-consistent pricing, use SABR or Heston calibrated to the FX vol surface.",
}

PRODUCTS["generalized_bsm"] = {
    "name": "Generalized BSM — Cost of Carry",
    "category": "vanilla",
    "desc": "Unified BSM framework using cost-of-carry parameter b for any asset class.",
    "module": "ch01_black_scholes_merton.04_generalized_bsm",
    "func": "generalized_bsm",
    "params": [_S, _K, _T, _r, _b, _sig],
    "has_option_type": True,
    "formula": r"C = S \, e^{(b-r)T} N(d_1) - K \, e^{-rT} N(d_2)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K) + (b + \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """The generalized BSM unifies all Black-Scholes variants through the cost-of-carry parameter b:
• b = r: non-dividend stock (Black-Scholes 1973)
• b = r − q: continuous dividend stock (Merton 1973)
• b = 0: futures (Black-76)
• b = r_d − r_f: FX (Garman-Kohlhagen)
• b = r + storage − convenience: commodities""",
    "assumptions": [
        "Underlying follows GBM with constant drift, volatility, and cost-of-carry",
        "Frictionless, continuously tradable markets",
        "Constant risk-free rate and cost-of-carry",
    ],
    "limitations": [
        "Same fundamental limitations as BSM — constant parameters, no smile",
        "The cost-of-carry b must be correctly specified for the asset class",
    ],
    "recommendations": "Use as the universal building block for European option pricing across all asset classes. Most exotic option formulas in this library are built on the generalized BSM.",
}

# ========================= AMERICAN OPTIONS =========================

PRODUCTS["baw"] = {
    "name": "Barone-Adesi-Whaley (1987)",
    "category": "american",
    "desc": "Fast quadratic approximation for American options with early exercise premium.",
    "module": "ch03_american_options.01_barone_adesi_whaley",
    "func": "barone_adesi_whaley",
    "params": [_S, _K, _T, _r, _b, _sig],
    "has_option_type": True,
    "formula": r"C_{Am} = C_{Eur} + A_2 \left(\frac{S}{S^*}\right)^{q_2} \quad \text{if } S < S^*",
    "formula_detail": r"""q_2 = \frac{-(n-1) + \sqrt{(n-1)^2 + 4M/K_q}}{2}, \quad n = \frac{2(r-b)}{\sigma^2}, \quad M = \frac{2r}{\sigma^2}""",
    "methodology": """BAW (1987) decomposes the American option price into its European value plus an early exercise
premium. The premium is derived by solving a quadratic approximation to the free-boundary PDE.
The critical stock price S* (early exercise boundary) is found iteratively using Newton's method.""",
    "assumptions": [
        "Same GBM and constant parameter assumptions as generalized BSM",
        "Early exercise boundary is approximated by a single critical price",
        "Quadratic approximation to the early exercise premium",
    ],
    "limitations": [
        "Accuracy decreases for deep ITM options or very long maturities",
        "Approximation error is typically < 0.1% for practical parameter ranges",
        "For higher accuracy, use Bjerksund-Stensland 2002 or binomial trees",
    ],
    "recommendations": "The industry workhorse for fast American option pricing. Excellent speed-accuracy tradeoff for real-time applications. Use binomial trees (N≥200) when higher precision is needed.",
    "param_definitions": {
        "S": "Current spot price of the underlying asset",
        "K": "Strike (exercise) price",
        "T": "Time to expiration in years",
        "r": "Annualized risk-free rate (continuously compounded)",
        "b": "Cost-of-carry rate (b = r for non-dividend stock, b = r − q for continuous dividend)",
        "σ": "Annualized volatility of the underlying",
        "C_Eur": "European call price (computed via generalized BSM as the base value)",
        "S*": "Critical stock price — the boundary above which early exercise is optimal",
        "A₂": "Early exercise premium coefficient: A₂ = (S*/q₂) · [1 − N(d₁(S*))]",
        "q₂": "Positive root of the quadratic characteristic equation: q₂ = [−(n−1) + √((n−1)² + 4M/Kq)] / 2",
        "n": "Auxiliary parameter: n = 2(r − b) / σ²",
        "M": "Auxiliary parameter: M = 2r / σ²",
        "Kq": "Discounting factor for the quadratic: Kq = 1 − e^{−rT}",
        "d₁(S*)": "Standard BSM d₁ evaluated at S = S* (critical boundary)",
    },
}

PRODUCTS["bs2002"] = {
    "name": "Bjerksund-Stensland (2002)",
    "category": "american",
    "desc": "Improved American option approximation with dual flat exercise boundaries.",
    "module": "ch03_american_options.02_bjerksund_stensland",
    "func": "bjerksund_stensland_2002",
    "params": [_S, _K, _T, _r, _b, _sig],
    "has_option_type": True,
    "formula": r"C_{Am} = \alpha_2 S^{\beta} - \alpha_2 \phi(S, t_1, \beta, I_2, I_2) + \phi(S, t_1, 1, I_2, I_2) - \ldots",
    "formula_detail": r"""\text{Uses two flat boundaries } I_1, I_2 \text{ to approximate the early exercise region.}""",
    "methodology": """Bjerksund-Stensland (2002) improves upon the 1993 version by using two flat exercise
boundaries instead of one. The option's life is split into two periods at t₁, each with its
own constant critical price. This captures more of the boundary curvature and reduces pricing error.""",
    "assumptions": [
        "Same GBM assumptions as generalized BSM",
        "Early exercise boundary approximated by two flat segments",
        "Analytical solution uses the ϕ function involving bivariate normal CDF",
    ],
    "limitations": [
        "Still an approximation — maximum error typically < 0.02% for practical ranges",
        "Slightly slower than BAW due to bivariate normal evaluations",
        "For exotic exercise features, numerical methods are required",
    ],
    "recommendations": "More accurate than BAW, especially for long-dated options. Recommended when sub-basis-point accuracy matters but full numerical methods are too slow.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "β": "Coefficient in the power function S^β, derived from model parameters",
        "α₁, α₂": "Scaling coefficients for the two exercise boundaries: αᵢ = −(Iᵢ/β) · [1 − e^{(b−r)(T−t₁)} N(d₁(Iᵢ))]",
        "I₁, I₂": "Two flat early exercise boundaries (I₂ for the full period, I₁ for the first half) — found iteratively",
        "t₁": "Mid-point of the option's life (t₁ = T/2), used to split the problem into two sub-periods",
        "φ(S, t, γ, H, I)": "Auxiliary function involving the bivariate normal CDF, used to compute the probability of reaching the boundary I before time t",
    },
}

PRODUCTS["perpetual"] = {
    "name": "American Perpetual Option",
    "category": "american",
    "desc": "Closed-form American option with infinite time to expiry.",
    "module": "ch03_american_options.02_bjerksund_stensland",
    "func": "american_perpetual",
    "params": [
        _S, _K, _r, _b, _sig,
    ],
    "has_option_type": True,
    "no_T": True,
    "formula": r"C_{\infty} = \left(\frac{S}{S^*_{\infty}}\right)^{h_1} \cdot (S^*_{\infty} - K)",
    "formula_detail": r"""h_1 = \frac{1}{2} - \frac{b}{\sigma^2} + \sqrt{\left(\frac{b}{\sigma^2} - \frac{1}{2}\right)^2 + \frac{2r}{\sigma^2}}""",
    "methodology": """The perpetual American option (T → ∞) has a time-independent closed-form solution.
The critical exercise boundary is constant, and the option value depends on the power function
S^h where h is determined by the characteristic equation of the BSM ODE.""",
    "assumptions": [
        "Infinite time to expiry — the option never expires",
        "All BSM assumptions (constant r, b, σ)",
        "Provides an upper bound for finite-maturity American options",
    ],
    "limitations": [
        "Purely theoretical — no traded option has infinite maturity",
        "Useful mainly as a benchmark or bound for finite-life American options",
    ],
    "recommendations": "Use as an analytical benchmark. The perpetual option value provides an upper bound for any finite-maturity American option with the same parameters.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "S*_∞": "Optimal (constant) exercise boundary for the perpetual option — the price at which it is always optimal to exercise immediately",
        "h₁": "Power exponent solving the ODE characteristic equation: h₁ = ½ − b/σ² + √[(b/σ² − ½)² + 2r/σ²]",
        "C_∞": "Perpetual call value — the limiting value as T → ∞",
    },
}

# ========================= EXOTIC — SINGLE ASSET =========================

PRODUCTS["forward_start"] = {
    "name": "Forward Start Options",
    "category": "exotic_single",
    "desc": "Option that activates at a future date with strike set as a fraction of the spot at activation.",
    "module": "ch04_exotic_single_asset.01_forward_start_options",
    "func": "forward_start_option",
    "params": [
        _S,
        P("alpha", "Strike Ratio (α)", 1.0, 0.01, 5.0, 0.01, "Strike = α × S at activation"),
        P("t1", "Activation Time (t₁)", 0.25, 0.001, 29.0, 0.01, "Time when option activates"),
        P("T", "Final Expiry (T)", 1.0, 0.01, 30.0, 0.01, "Time to final expiration"),
        _r, _b, _sig,
    ],
    "has_option_type": True,
    "formula": r"V = S \, e^{(b-r)t_1} \cdot \text{BSM}(S=1, K=\alpha, T-t_1, r, b, \sigma)",
    "formula_detail": r"""\text{At } t_1, \text{ the strike is set to } K = \alpha \cdot S_{t_1}.""",
    "methodology": """A forward start option has a deferred start date t₁. At t₁, the strike is set to α·S(t₁).
Pricing uses the scaling property of BSM: the value today is S·e^{(b-r)t₁} times a European
option with S=1, K=α evaluated over the remaining life T−t₁.""",
    "assumptions": [
        "GBM dynamics and constant parameters throughout",
        "Strike ratio α is predetermined",
        "Common in employee stock option plans (ESOs)",
    ],
    "limitations": [
        "Constant volatility may not reflect the actual forward volatility",
        "Does not capture smile dynamics between today and t₁",
    ],
    "recommendations": "Used in executive compensation (reload options) and structured products. For more accurate pricing, consider forward-start options under stochastic volatility.",
    "param_definitions": {
        "S": "Current spot price today (before activation)",
        "α": "Strike ratio — at activation time t₁ the strike is set to K = α · S(t₁). α=1 gives ATM forward-start",
        "t₁": "Activation time in years — the future date when the option starts and the strike is fixed",
        "T": "Final expiration date in years (T > t₁)",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "e^{(b−r)t₁}": "Discount/carry factor from today to activation date",
        "BSM(1, α, T−t₁, r, b, σ)": "Standard BSM price evaluated with S=1, K=α over remaining life T−t₁ — scales linearly with S by the log-normality of GBM",
    },
}

PRODUCTS["simple_chooser"] = {
    "name": "Simple Chooser Option",
    "category": "exotic_single",
    "desc": "At the choice date, the holder selects whether the option becomes a call or a put.",
    "module": "ch04_exotic_single_asset.02_chooser_options",
    "func": "simple_chooser",
    "params": [
        _S, _K,
        P("tc", "Choice Date (tc)", 0.25, 0.001, 29.0, 0.01, "Time when call/put choice is made"),
        P("T", "Expiry (T)", 1.0, 0.01, 30.0, 0.01, "Final expiration date"),
        _r, _b, _sig,
    ],
    "has_option_type": False,
    "formula": r"V = \text{BSM}_{call}(S,K,T) + e^{(b-r)T} S \, N(-d_1^*) \cdot (\ldots)",
    "formula_detail": r"""\text{Decomposed into a call plus a put adjustment using put-call parity at the choice date.}""",
    "methodology": """At the choice date tc, the holder picks max(Call, Put). By put-call parity,
this decomposes into a European call with maturity T plus a European put with maturity tc
(strike-adjusted). The simple chooser has a closed-form using this decomposition.""",
    "assumptions": [
        "Same strike K and expiry T for both the call and put component",
        "Choice date tc < T",
        "Standard BSM assumptions",
    ],
    "limitations": [
        "The 'simple' version requires identical K and T for call and put",
        "For different strikes/expiries, use the complex chooser (requires 2D integration)",
    ],
    "recommendations": "Useful when the directional view is uncertain. The premium is always higher than a single call or put due to the optionality of choice.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Common strike for both call and put components",
        "tc": "Choice date — the time (in years) at which the holder decides call or put",
        "T": "Final expiration date of the underlying option (T > tc)",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "BSM_call(S,K,T)": "European call price with maturity T (the primary call component)",
        "d₁*": "d₁ evaluated at the adjusted strike K* = K · e^{−(b−r)(T−tc)} and maturity tc — the put-call parity decomposition point",
        "N(·)": "Standard normal cumulative distribution function",
    },
}

PRODUCTS["compound"] = {
    "name": "Compound Options",
    "category": "exotic_single",
    "desc": "An option on an option — four types: call-on-call, call-on-put, put-on-call, put-on-put.",
    "module": "ch04_exotic_single_asset.03_compound_options",
    "func": "compound_option",
    "params": [
        _S,
        P("K_outer", "Outer Strike (K₁)", 5.0, 0.0, 1e5, 0.1, "Strike of the outer option"),
        P("K_inner", "Inner Strike (K₂)", 100.0, 0.01, 1e6, 1.0, "Strike of the inner (underlying) option"),
        P("T1", "Outer Expiry (T₁)", 0.25, 0.001, 29.0, 0.01, "Outer option expiration"),
        P("T2", "Inner Expiry (T₂)", 1.0, 0.01, 30.0, 0.01, "Inner option expiration"),
        _r, _b, _sig,
    ],
    "has_option_type": False,
    "extra_choices": {
        "outer_type": {"label": "Outer Type", "options": ["call", "put"], "default": "call"},
        "inner_type": {"label": "Inner Type", "options": ["call", "put"], "default": "call"},
    },
    "formula": r"V = S\,e^{(b-r)T_2}\,M(d_1, d_2; \rho) - K_2\,e^{-rT_2}\,M(d_3, d_4; \rho) - (\pm)\,K_1\,e^{-rT_1}\,N(\pm d_3)",
    "formula_detail": r"""\rho = \sqrt{T_1 / T_2}, \quad M(\cdot,\cdot;\rho) = \text{bivariate normal CDF}""",
    "methodology": """A compound option gives the right to buy/sell another option at T₁. The pricing involves
the bivariate normal distribution M(a,b;ρ) where ρ = √(T₁/T₂). At T₁, the holder exercises
only if the inner option's value exceeds the outer strike K₁. The critical stock price S* at T₁
is found by solving BSM(S*, K₂, T₂−T₁) = K₁.""",
    "assumptions": [
        "Standard BSM assumptions",
        "T₁ < T₂ (outer expires before inner)",
        "No early exercise (European compound)",
    ],
    "limitations": [
        "Sensitive to volatility assumptions over two time horizons",
        "Bivariate normal approximation may introduce small numerical errors",
        "American-style compounds require numerical methods",
    ],
    "recommendations": "Used in installment options, corporate finance (real options), and sequential investment decisions. The four types (CaC, CaP, PaC, PaP) cover all combinations.",
    "param_definitions": {
        "S": "Current spot price",
        "K₁": "Outer strike — premium paid at T₁ to acquire the inner option",
        "K₂": "Inner strike — the strike of the underlying (inner) option at T₂",
        "T₁": "Outer option expiry — when the compound option is exercised (T₁ < T₂)",
        "T₂": "Inner option expiry — final maturity of the underlying option",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "M(a, b; ρ)": "Bivariate standard normal CDF — probability that two correlated standard normals are both below (a, b)",
        "ρ": "Correlation between the two integration variables: ρ = √(T₁/T₂)",
        "d₁, d₂": "Integration limits involving ln(S/S*) where S* is the critical stock price at T₁",
        "d₃, d₄": "Integration limits involving ln(S/K₂)",
        "S*": "Critical stock price at T₁ — the level at which the inner option value equals the outer strike K₁",
        "N(·)": "Standard normal CDF",
    },
}

PRODUCTS["floating_lookback"] = {
    "name": "Floating Strike Lookback",
    "category": "exotic_single",
    "desc": "Strike is set at the minimum (call) or maximum (put) of the underlying over the option's life.",
    "module": "ch04_exotic_single_asset.04_lookback_options",
    "func": "floating_lookback_call",
    "params": [
        _S,
        P("S_min", "Running Min (S_min)", 90.0, 0.01, 1e6, 1.0, "Minimum spot observed so far"),
        _T, _r, _b, _sig,
    ],
    "has_option_type": False,
    "is_call_only": True,
    "alt_func": {"put": "floating_lookback_put", "put_param_swap": {"S_min": ("S_max", "Running Max (S_max)", 110.0)}},
    "formula": r"C = S\,e^{(b-r)T}N(a_1) - S_{\min}\,e^{-rT}N(a_2) + S\,e^{-rT}\frac{\sigma^2}{2b}\left[\ldots\right]",
    "formula_detail": r"""a_1 = \frac{\ln(S/S_{\min}) + (b + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad a_2 = a_1 - \sigma\sqrt{T}""",
    "methodology": """Floating lookback call pays S_T − min(S_t) at expiry — the holder buys at the lowest price.
The put pays max(S_t) − S_T — selling at the highest. Closed-form solutions exist using the
distribution of the running extremum of GBM.""",
    "assumptions": [
        "Continuous monitoring of the extremum (no discrete gaps)",
        "Standard BSM assumptions",
    ],
    "limitations": [
        "Very expensive due to perfect hindsight — typical premium is 1.5-2× a vanilla option",
        "Continuous monitoring overstates value vs. discrete monitoring in practice",
        "Greeks can be discontinuous near current extremum",
    ],
    "recommendations": "Popular in structured products. For discrete monitoring, use Monte Carlo simulation. The premium reflects the maximum possible intrinsic value.",
    "param_definitions": {
        "S": "Current spot price",
        "S_min": "Running minimum of the spot price observed so far (from inception to today)",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "a₁": "Moneyness measure: a₁ = [ln(S/S_min) + (b + σ²/2)T] / (σ√T)",
        "a₂": "Strike-adjusted measure: a₂ = a₁ − σ√T",
        "σ²/2b": "Scaling factor in the third term (volatility-to-carry ratio); undefined if b = 0 — use limit",
        "N(·)": "Standard normal CDF",
    },
}

PRODUCTS["fixed_lookback"] = {
    "name": "Fixed Strike Lookback",
    "category": "exotic_single",
    "desc": "Lookback option with a fixed strike — payoff based on the running extremum vs. strike.",
    "module": "ch04_exotic_single_asset.04_lookback_options",
    "func": "fixed_lookback_call",
    "params": [
        _S,
        P("S_max", "Running Max (S_max)", 110.0, 0.01, 1e6, 1.0, "Maximum spot observed so far"),
        _K, _T, _r, _b, _sig,
    ],
    "has_option_type": False,
    "is_call_only": True,
    "alt_func": {"put": "fixed_lookback_put", "put_param_swap": {"S_max": ("S_min", "Running Min (S_min)", 90.0)}},
    "formula": r"C = \max(S_{\max}-K,0)\,e^{-rT} + (\text{continuation value})",
    "formula_detail": r"""\text{When } S_{\max} \geq K: \text{ guaranteed payoff plus upside potential}""",
    "methodology": """Fixed lookback call pays max(max(S_t) − K, 0). Unlike the floating version, there is a fixed
strike K. If the running max already exceeds K, part of the payoff is locked in. The formula
involves conditional expectations of the running maximum.""",
    "assumptions": [
        "Continuous monitoring of the running maximum/minimum",
        "Standard BSM assumptions",
    ],
    "limitations": [
        "Continuous monitoring assumption overstates value vs. discrete observation",
        "Can be very expensive when current extremum is far in-the-money",
    ],
    "recommendations": "Less common than floating lookbacks in structured products but useful for guaranteed minimum return strategies.",
    "param_definitions": {
        "S": "Current spot price",
        "S_max": "Running maximum of the spot price observed from inception to today",
        "K": "Fixed strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "max(S_max−K, 0)·e^{−rT}": "Discounted intrinsic value already locked in — guaranteed minimum payoff",
        "Continuation value": "Additional option value from the possibility that S_max will increase further before expiry",
    },
}

PRODUCTS["barrier"] = {
    "name": "Standard Barrier Options",
    "category": "exotic_single",
    "desc": "Option that activates (knock-in) or deactivates (knock-out) when the spot hits a barrier.",
    "module": "ch04_exotic_single_asset.05_barrier_options",
    "func": "barrier_option",
    "params": [
        _S, _K,
        P("H", "Barrier Level (H)", 90.0, 0.01, 1e6, 1.0, "Knock-in or knock-out barrier"),
        P("rebate", "Rebate", 0.0, 0.0, 1e6, 0.1, "Cash rebate if knocked out"),
        _T, _r, _b, _sig,
    ],
    "has_option_type": False,
    "extra_choices": {
        "barrier_type": {
            "label": "Barrier Type",
            "options": [
                "down-and-out-call", "down-and-in-call",
                "up-and-out-call", "up-and-in-call",
                "down-and-out-put", "down-and-in-put",
                "up-and-out-put", "up-and-in-put",
            ],
            "default": "down-and-out-call",
        },
    },
    "formula": r"V = \eta\phi\left[x_1 Se^{(b-r)T}N(\eta d_1) - Ke^{-rT}N(\eta d_2) + \ldots \right]",
    "formula_detail": r"""\mu = \frac{b - \sigma^2/2}{\sigma^2}, \quad \lambda = \sqrt{\mu^2 + \frac{2r}{\sigma^2}}""",
    "formula_extra": [
        r"""\begin{aligned}
A &= \phi\,S\,e^{(b-r)T}N(\phi x_1) - \phi\,K\,e^{-rT}N(\phi x_1 - \phi\sigma\sqrt{T})\\
B &= \phi\,S\,e^{(b-r)T}N(\phi x_2) - \phi\,K\,e^{-rT}N(\phi x_2 - \phi\sigma\sqrt{T})\\
C &= \phi\,S\,(H/S)^{2(\mu+1)}e^{(b-r)T}N(\eta y_1) - \phi\,K\,(H/S)^{2\mu}e^{-rT}N(\eta y_1 - \eta\sigma\sqrt{T})\\
D &= \phi\,S\,(H/S)^{2(\mu+1)}e^{(b-r)T}N(\eta y_2) - \phi\,K\,(H/S)^{2\mu}e^{-rT}N(\eta y_2 - \eta\sigma\sqrt{T})\\
E &= \text{Rebate}\cdot e^{-rT}\!\left[N(\eta x_2-\eta\sigma\sqrt{T}) - (H/S)^{2\mu}N(\eta y_2-\eta\sigma\sqrt{T})\right]\\
F &= \text{Rebate}\!\left[(H/S)^{\mu+\lambda}N(\eta z) + (H/S)^{\mu-\lambda}N(\eta z - 2\eta\lambda\sigma\sqrt{T})\right]
\end{aligned}""",
        r"""\begin{aligned}
x_1 &= \tfrac{\ln(S/K)}{\sigma\sqrt{T}} + (1{+}\mu)\sigma\sqrt{T}, \quad
x_2 = \tfrac{\ln(S/H)}{\sigma\sqrt{T}} + (1{+}\mu)\sigma\sqrt{T}\\
y_1 &= \tfrac{\ln(H^2/SK)}{\sigma\sqrt{T}} + (1{+}\mu)\sigma\sqrt{T}, \quad
y_2 = \tfrac{\ln(H/S)}{\sigma\sqrt{T}} + (1{+}\mu)\sigma\sqrt{T}\\
z &= \tfrac{\ln(H/S)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
\end{aligned}""",
        r"""\begin{array}{ll}
\textbf{8 Barrier Type Formulas} & (\phi{=}{+1}\text{ call},\ \phi{=}{-1}\text{ put};\ \eta{=}{+1}\text{ down},\ \eta{=}{-1}\text{ up})\\[4pt]
\text{Down-and-out call} & A - C + F \;(K{\ge}H),\quad B - D + F \;(K{<}H)\\
\text{Down-and-in call}  & C + E \;(K{\ge}H),\quad A - B + D + E \;(K{<}H)\\
\text{Up-and-out call}   & F \;(K{\ge}H),\quad A - B + C - D + F \;(K{<}H)\\
\text{Up-and-in call}    & A - C + D + E \;(K{\ge}H),\quad B + E \;(K{<}H)\\
\text{Down-and-out put}  & A - B + C - D + F \;(K{\ge}H),\quad F \;(K{<}H)\\
\text{Down-and-in put}   & B - D + E \;(K{\ge}H),\quad A - C + D + E \;(K{<}H)\\
\text{Up-and-out put}    & B - D + F \;(K{\ge}H),\quad A - C + F \;(K{<}H)\\
\text{Up-and-in put}     & A - B + D + E \;(K{\ge}H),\quad C + E \;(K{<}H)
\end{array}""",
    ],
    "methodology": """Barrier options (Merton 1973, Reiner-Rubinstein 1991) use the reflection principle of
Brownian motion. The 8 standard types combine: direction (up/down) × action (in/out) × option (call/put).
Knock-in + knock-out = vanilla (parity). The formula uses powers of (H/S) as reflection factors.""",
    "assumptions": [
        "Continuous barrier monitoring",
        "Standard BSM assumptions (GBM, constant params)",
        "No gap risk (price cannot jump through the barrier)",
    ],
    "limitations": [
        "Continuous monitoring overstates knock-out probability vs. discrete monitoring",
        "Highly sensitive to volatility near the barrier (vega can change sign)",
        "Gap risk (jumps through barrier) is not captured — consider Monte Carlo with jumps",
        "Delta can spike near the barrier, making hedging expensive",
    ],
    "recommendations": "Among the most widely traded exotics. For discrete monitoring, use the Broadie-Glasserman-Kou correction or Monte Carlo. In-out parity is a useful check: KI + KO = Vanilla.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Option strike price",
        "H": "Barrier level — the price threshold that triggers knock-in or knock-out",
        "rebate": "Cash payment received immediately (for KO) or at expiry (for KI) if the barrier is hit without payoff",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "η (eta)": "Option type sign: η = +1 for call, η = −1 for put",
        "φ (phi)": "Barrier direction sign: φ = +1 for down barrier (H < S), φ = −1 for up barrier (H > S)",
        "μ": "Reflection exponent: μ = (b − σ²/2) / σ²  — governs the (H/S)^{2μ} reflection factors",
        "λ": "Modified drift: λ = √(μ² + 2r/σ²)",
        "x₁": "Moneyness of spot vs. strike: x₁ = ln(S/K)/(σ√T) + (1+μ)σ√T",
        "y₁": "Moneyness of spot vs. barrier: y₁ = ln(H/S)/(σ√T) + (1+μ)σ√T  (used in knock-out pricing)",
        "y": "Barrier reflected distance: y = ln(H²/(SK))/(σ√T) + (1+μ)σ√T",
        "N(·)": "Standard normal CDF",
        "A, B, C, D": "Four building-block sub-formulas combined with signs to produce each of the 8 barrier types",
    },
}

PRODUCTS["double_barrier"] = {
    "name": "Double Barrier Options",
    "category": "exotic_single",
    "desc": "Option with both upper and lower barriers — knocked out if either is breached.",
    "module": "ch04_exotic_single_asset.05_barrier_options",
    "func": "double_barrier_option",
    "params": [
        _S, _K,
        P("L", "Lower Barrier (L)", 80.0, 0.01, 1e6, 1.0, "Lower knock-out barrier"),
        P("U", "Upper Barrier (U)", 120.0, 0.01, 1e6, 1.0, "Upper knock-out barrier"),
        _T, _r, _b, _sig,
    ],
    "has_option_type": True,
    "formula": r"V = \sum_{n=-\infty}^{\infty} \left[ f_n^{(1)}(d_1, d_2) - f_n^{(2)}(d_3, d_4) \right]",
    "formula_detail": r"""\text{Infinite series using image method — typically 5 terms suffice for convergence.}""",
    "methodology": """Double barrier options knock out if the spot hits either the lower barrier L or upper barrier U.
Priced using an infinite series based on the image (reflection) method. The series converges quickly
(typically 5 terms), with each term involving standard normal CDFs.""",
    "assumptions": [
        "Continuous monitoring of both barriers",
        "L < S < U at inception",
        "Standard BSM assumptions",
    ],
    "limitations": [
        "Requires L < S < U — no rebate differentiation by barrier in the basic formula",
        "Convergence of the series should be verified for extreme parameters",
    ],
    "recommendations": "Popular in FX structured products (range accumulators). The narrow corridor between barriers makes these much cheaper than vanilla options.",
    "param_definitions": {
        "S": "Current spot price (must satisfy L < S < U at inception)",
        "K": "Option strike price",
        "L": "Lower knock-out barrier — option terminates if S falls to or below L",
        "U": "Upper knock-out barrier — option terminates if S rises to or above U",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "n": "Summation index (integer from −∞ to +∞); typically |n| ≤ 5 terms needed",
        "f_n^(1), f_n^(2)": "Image-method terms for the upper/lower reflection; each involves standard normal CDFs evaluated at distances d₁ through d₄",
        "d₁, d₂, d₃, d₄": "Standardized log-price distances reflecting both barriers: e.g., d₁ = [ln(S·U^{2n}/L^{2n+2}/K) + (b+σ²/2)T] / (σ√T)",
        "N(·)": "Standard normal CDF",
    },
}

PRODUCTS["cash_or_nothing"] = {
    "name": "Cash-or-Nothing Binary",
    "category": "exotic_single",
    "desc": "Pays a fixed cash amount if the option expires in-the-money, zero otherwise.",
    "module": "ch04_exotic_single_asset.06_binary_options",
    "func": "cash_or_nothing",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("cash_amount", "Cash Payout", 1.0, 0.01, 1e6, 1.0, "Fixed cash amount paid if ITM"),
    ],
    "has_option_type": True,
    "formula": r"C = Q \, e^{-rT} N(d_2)",
    "formula_detail": r"""d_2 = \frac{\ln(S/K) + (b - \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad Q = \text{cash amount}""",
    "methodology": """The cash-or-nothing option pays Q if S_T > K (call) or S_T < K (put). The value
is the discounted risk-neutral probability of ending ITM times the payout. This is a pure bet
on the direction with a discontinuous payoff.""",
    "assumptions": [
        "Standard BSM assumptions",
        "Digital payoff — no partial settlement",
    ],
    "limitations": [
        "Discontinuous payoff creates extreme gamma/vega near expiry at-the-money",
        "Very difficult to hedge near maturity if spot is close to strike",
        "Model risk is high due to payoff discontinuity",
    ],
    "recommendations": "Building block for more complex structures. In practice, often approximated by tight call/put spreads to reduce hedging risk. Watch for pin risk near expiry.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike — the threshold that determines ITM/OTM",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "Q": "Fixed cash payout received if the option expires ITM (cash_amount input)",
        "e^{−rT}": "Discount factor to present-value the fixed payout",
        "d₂": "Risk-neutral moneyness: d₂ = [ln(S/K) + (b − σ²/2)T] / (σ√T) — represents the risk-neutral probability of expiring ITM",
        "N(d₂)": "Risk-neutral probability of expiring ITM (for call) — the probability that the fixed payout Q is received",
    },
}

PRODUCTS["asset_or_nothing"] = {
    "name": "Asset-or-Nothing Binary",
    "category": "exotic_single",
    "desc": "Pays the asset value S_T if expiring ITM, zero otherwise.",
    "module": "ch04_exotic_single_asset.06_binary_options",
    "func": "asset_or_nothing",
    "params": [_S, _K, _T, _r, _b, _sig],
    "has_option_type": True,
    "formula": r"C = S \, e^{(b-r)T} N(d_1)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K) + (b + \tfrac{\sigma^2}{2})T}{\sigma\sqrt{T}}""",
    "methodology": """Pays S_T if ITM at expiry. Value = forward price of S times the probability of S_T > K
under the stock-price measure (N(d₁)). Together with cash-or-nothing, these are the two
building blocks of the BSM formula: Vanilla Call = Asset-or-Nothing − K × Cash-or-Nothing.""",
    "assumptions": [
        "Standard BSM assumptions",
        "Payoff is the full asset value, not a fixed amount",
    ],
    "limitations": [
        "Same hedging difficulties as cash-or-nothing near expiry",
        "The payoff grows with S, unlike the bounded cash-or-nothing",
    ],
    "recommendations": "Useful decomposition: BSM Call = AssetOrNothing(Call) − K × CashOrNothing(Call). Provides insight into the delta component of option value.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike — determines ITM/OTM",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "e^{(b−r)T}": "Growth factor under the asset measure — converts the spot to the forward-measure probability weight",
        "d₁": "Asset-measure moneyness: d₁ = [ln(S/K) + (b + σ²/2)T] / (σ√T)",
        "N(d₁)": "Probability that S_T > K under the asset (stock-price) measure — determines the expected asset delivery",
    },
}

PRODUCTS["gap_option"] = {
    "name": "Gap Options",
    "category": "exotic_single",
    "desc": "Binary-style option with separate trigger strike and payoff strike.",
    "module": "ch04_exotic_single_asset.06_binary_options",
    "func": "gap_option",
    "params": [
        _S,
        P("K1", "Trigger Strike (K₁)", 100.0, 0.01, 1e6, 1.0, "Strike that determines if option pays"),
        P("K2", "Payoff Strike (K₂)", 105.0, 0.01, 1e6, 1.0, "Strike used in payoff calculation"),
        _T, _r, _b, _sig,
    ],
    "has_option_type": True,
    "formula": r"C = S\,e^{(b-r)T}N(d_1) - K_2\,e^{-rT}N(d_2)",
    "formula_detail": r"""d_1 = \frac{\ln(S/K_1) + (b + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """A gap option triggers based on K₁ but the payoff is computed using K₂.
Call pays (S_T − K₂) when S_T > K₁. The gap K₂ − K₁ can be positive or negative,
and the payoff can actually be negative for the holder.""",
    "assumptions": [
        "Standard BSM assumptions",
        "The holder must accept the payoff (even if negative) when triggered",
    ],
    "limitations": [
        "Payoff can be negative — unlike standard options",
        "Requires careful analysis of K₁ vs K₂ relationship",
    ],
    "recommendations": "Useful in insurance-linked products and credit derivatives where trigger and settlement levels differ.",
    "param_definitions": {
        "S": "Current spot price",
        "K₁": "Trigger strike — the level that determines whether the option pays (S_T > K₁ for call)",
        "K₂": "Payoff strike — used in the payoff calculation: payout = S_T − K₂ (which can be negative if K₂ > K₁)",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "d₁": "Moneyness using trigger strike K₁: d₁ = [ln(S/K₁) + (b + σ²/2)T] / (σ√T)",
        "d₂": "d₂ = d₁ − σ√T (using K₁ for triggering)",
        "Gap": "The difference K₂ − K₁; can be positive (reduces payout) or negative (increases payout)",
    },
}

PRODUCTS["geometric_asian"] = {
    "name": "Geometric Average Asian",
    "category": "exotic_single",
    "desc": "Asian option with geometric average price — closed-form solution available.",
    "module": "ch04_exotic_single_asset.07_asian_options",
    "func": "geometric_asian_option",
    "params": [_S, _K, _T, _r, _b, _sig],
    "has_option_type": True,
    "formula": r"V = \text{BSM}(S, K, T, r, b_A, \sigma_A)",
    "formula_detail": r"""b_A = \frac{1}{2}\left(b - \frac{\sigma^2}{6}\right), \quad \sigma_A = \frac{\sigma}{\sqrt{3}}""",
    "methodology": """The geometric average of a log-normal process is itself log-normal, enabling a closed-form
BSM-like solution. The adjusted volatility σ_A = σ/√3 and cost-of-carry b_A reflect the
averaging effect. Serves as a control variate for pricing arithmetic Asians.""",
    "assumptions": [
        "Continuous geometric averaging",
        "Standard BSM assumptions",
    ],
    "limitations": [
        "Geometric average ≤ arithmetic average, so this underprices arithmetic Asians",
        "Rarely traded directly — used mainly as a control variate or benchmark",
    ],
    "recommendations": "Primary use is as a control variate for Monte Carlo pricing of arithmetic Asians. The closed form is exact for the geometric average case.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Original cost-of-carry rate for the underlying",
        "σ": "Underlying annualized volatility",
        "b_A": "Adjusted cost-of-carry for geometric averaging: b_A = ½(b − σ²/6) — reflects that averaging reduces the drift",
        "σ_A": "Adjusted volatility for geometric averaging: σ_A = σ/√3 — lower because averaging smooths the price path",
        "BSM(S, K, T, r, b_A, σ_A)": "Standard BSM evaluated with the averaging-adjusted parameters (b_A, σ_A) instead of the raw (b, σ)",
    },
}

PRODUCTS["arithmetic_asian"] = {
    "name": "Arithmetic Average Asian (TW)",
    "category": "exotic_single",
    "desc": "Asian option based on arithmetic average price — Turnbull-Wakeman moment-matching approximation.",
    "module": "ch04_exotic_single_asset.07_asian_options",
    "func": "arithmetic_asian_option_TW",
    "params": [
        _S, _K, _T,
        P("T_start", "Averaging Start", 0.0, 0.0, 29.0, 0.01, "When averaging begins (0 = from inception)"),
        _r, _b, _sig,
        P("n", "Observation Points", 252, 1, 10000, 1, "Number of averaging observations"),
    ],
    "has_option_type": True,
    "formula": r"V \approx \text{BSM}(S, K, T, r, b_A, \sigma_A) \text{ with moment-matched } b_A, \sigma_A",
    "formula_detail": r"""\text{Match first two moments of arithmetic average to a log-normal distribution.}""",
    "methodology": """Turnbull-Wakeman (1991) approximates the arithmetic average's distribution as log-normal
by matching its first two moments. The adjusted BSM parameters capture the variance reduction
from averaging. More accurate than Levy's approximation for a wider parameter range.""",
    "assumptions": [
        "Arithmetic average is approximately log-normally distributed",
        "Discrete observations approximated by continuous averaging adjustment",
        "Standard BSM assumptions for the underlying",
    ],
    "limitations": [
        "Approximation error increases for long maturities and high volatilities",
        "Does not account for seasoning (partial averaging already done)",
        "For higher accuracy, use Monte Carlo with geometric Asian control variate",
    ],
    "recommendations": "The most widely traded Asian option type. Common in commodity and energy markets where the average price reduces manipulation risk.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Final expiration date in years",
        "T_start": "Start of the averaging period (0 = from inception; T_start > 0 means averaging begins in the future)",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate for the underlying",
        "σ": "Underlying annualized volatility",
        "n": "Number of discrete observation/averaging points",
        "b_A": "Moment-matched cost-of-carry: derived by matching the first moment of the arithmetic average to a log-normal — depends on b, σ, T, T_start",
        "σ_A": "Moment-matched volatility: derived by matching the second moment of the arithmetic average — lower than σ due to averaging",
        "M₁, M₂": "First and second raw moments of the arithmetic average under risk-neutral measure — used to derive b_A and σ_A",
    },
}

PRODUCTS["standard_power"] = {
    "name": "Standard Power Options",
    "category": "exotic_single",
    "desc": "Option on the i-th power of the underlying: max(S^i − K, 0).",
    "module": "ch04_exotic_single_asset.08_power_log_options",
    "func": "standard_power_option",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("i", "Power (i)", 2.0, 0.1, 10.0, 0.1, "Power exponent applied to S"),
    ],
    "has_option_type": True,
    "formula": r"C = e^{-rT}\left[\hat{S}\,N(d_1) - K\,N(d_2)\right]",
    "formula_detail": r"""\hat{S} = S^i \exp\{[(ib + i(i-1)\sigma^2/2) - r]T\}, \quad \sigma_i = i\sigma""",
    "methodology": """The power option replaces S with S^i in the payoff. Under GBM, S^i is also log-normal
with adjusted drift and volatility (σ_i = i·σ). The BSM framework applies with these
modified parameters.""",
    "assumptions": [
        "Standard BSM assumptions",
        "Power i can be any positive real number",
    ],
    "limitations": [
        "Extreme leverage for i > 2 — very sensitive to volatility",
        "May require capping to be practically tradeable",
    ],
    "recommendations": "Provides leveraged exposure. Often combined with caps (capped power option) to limit risk. Used in structured products for enhanced returns.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike applied to S^i",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility of S (not S^i)",
        "i": "Power exponent — the payoff is max(S_T^i − K, 0). i=1 gives a vanilla call; i=2 is the standard power option",
        "Ŝ": "Modified forward: Ŝ = S^i · exp{[(ib + i(i−1)σ²/2) − r]T} — the risk-neutral 'forward' of S^i",
        "σ_i": "Effective volatility of S^i: σ_i = i·σ — volatility scales linearly with the power",
        "d₁": "d₁ = [ln(S^i/K) + (ib + i(i−1)σ²/2 + σ_i²/2 − r)T] / (σ_i√T)",
        "d₂": "d₂ = d₁ − σ_i√T",
    },
}

# ========================= EXOTIC — TWO ASSETS =========================

PRODUCTS["margrabe"] = {
    "name": "Margrabe Exchange Option",
    "category": "exotic_two",
    "desc": "Right to exchange one asset for another — no fixed strike, no cash settlement.",
    "module": "ch05_exotic_two_assets.01_margrabe_exchange",
    "func": "margrabe_exchange",
    "params": [
        P("S1", "Asset 1 Price (S₁)", 100.0, 0.01, 1e6, 1.0, "Price of asset to receive"),
        P("S2", "Asset 2 Price (S₂)", 100.0, 0.01, 1e6, 1.0, "Price of asset to give up"),
        P("Q1", "Quantity 1 (Q₁)", 1.0, 0.01, 1e6, 0.01, "Quantity of asset 1"),
        P("Q2", "Quantity 2 (Q₂)", 1.0, 0.01, 1e6, 0.01, "Quantity of asset 2"),
        _T, _r,
        P("b1", "Carry 1 (b₁)", 0.05, -1.0, 1.0, 0.001, "Cost of carry for asset 1"),
        P("b2", "Carry 2 (b₂)", 0.05, -1.0, 1.0, 0.001, "Cost of carry for asset 2"),
        P("sigma1", "Vol 1 (σ₁)", 0.20, 0.001, 5.0, 0.01, "Volatility of asset 1"),
        P("sigma2", "Vol 2 (σ₂)", 0.20, 0.001, 5.0, 0.01, "Volatility of asset 2"),
        P("rho", "Correlation (ρ)", 0.5, -1.0, 1.0, 0.01, "Correlation between assets"),
    ],
    "has_option_type": False,
    "formula": r"V = Q_1 S_1 e^{(b_1-r)T} N(d_1) - Q_2 S_2 e^{(b_2-r)T} N(d_2)",
    "formula_detail": r"""\sigma = \sqrt{\sigma_1^2 - 2\rho\sigma_1\sigma_2 + \sigma_2^2}, \quad d_1 = \frac{\ln(Q_1 S_1/Q_2 S_2) + (b_1 - b_2 + \sigma^2/2)T}{\sigma\sqrt{T}}""",
    "methodology": """Margrabe (1978) prices the right to exchange Q₂ units of asset 2 for Q₁ units of asset 1.
The combined volatility σ = √(σ₁² − 2ρσ₁σ₂ + σ₂²) is the volatility of the relative price.
No risk-free rate appears in the payoff (it's a pure exchange), but carry costs matter.""",
    "assumptions": [
        "Both assets follow correlated GBM",
        "Constant volatilities and correlation",
        "No frictions or constraints on holding either asset",
    ],
    "limitations": [
        "Correlation is assumed constant — in reality it varies, especially in stress",
        "Only applicable to European-style exchange",
    ],
    "recommendations": "Foundation for spread options, relative value trades, and M&A deal contingent pricing. Kirk's approximation extends this to non-zero strikes.",
    "param_definitions": {
        "S₁": "Price of the asset being received (Q₁ units)",
        "S₂": "Price of the asset being given up (Q₂ units)",
        "Q₁": "Quantity of asset 1 received upon exercise",
        "Q₂": "Quantity of asset 2 delivered upon exercise",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b₁": "Cost-of-carry for asset 1",
        "b₂": "Cost-of-carry for asset 2",
        "σ₁": "Volatility of asset 1",
        "σ₂": "Volatility of asset 2",
        "ρ": "Correlation between the returns of asset 1 and asset 2",
        "σ": "Combined (relative) volatility: σ = √(σ₁² − 2ρσ₁σ₂ + σ₂²) — volatility of the log-price ratio ln(Q₁S₁ / Q₂S₂)",
        "d₁": "d₁ = [ln(Q₁S₁/Q₂S₂) + (b₁ − b₂ + σ²/2)T] / (σ√T)",
        "d₂": "d₂ = d₁ − σ√T",
    },
}

PRODUCTS["kirk_spread"] = {
    "name": "Kirk's Spread Option",
    "category": "exotic_two",
    "desc": "Option on the spread between two assets: max(S₁ − S₂ − K, 0).",
    "module": "ch05_exotic_two_assets.02_spread_max_min_options",
    "func": "spread_option_kirk",
    "params": [
        P("F1", "Forward 1 (F₁)", 100.0, 0.01, 1e6, 1.0, "Forward/futures price of asset 1"),
        P("F2", "Forward 2 (F₂)", 90.0, 0.01, 1e6, 1.0, "Forward/futures price of asset 2"),
        P("K", "Strike (K)", 5.0, -1e6, 1e6, 0.1, "Spread strike"),
        _T, _r,
        P("sigma1", "Vol 1 (σ₁)", 0.20, 0.001, 5.0, 0.01, "Volatility of asset 1"),
        P("sigma2", "Vol 2 (σ₂)", 0.25, 0.001, 5.0, 0.01, "Volatility of asset 2"),
        P("rho", "Correlation (ρ)", 0.5, -1.0, 1.0, 0.01, "Correlation between assets"),
    ],
    "has_option_type": True,
    "formula": r"V \approx e^{-rT}\left[F_1 N(d_1) - (F_2 + K) N(d_2)\right]",
    "formula_detail": r"""\sigma_{kirk} = \sqrt{\sigma_1^2 - 2\rho\sigma_1\sigma_2 \frac{F_2}{F_2+K} + \sigma_2^2\left(\frac{F_2}{F_2+K}\right)^2}""",
    "methodology": """Kirk (1995) approximates the spread option by treating S₂ + K as a single asset and
applying Margrabe's formula. The effective volatility σ_kirk accounts for the correlation
and the ratio F₂/(F₂+K). Accurate for most practical parameter ranges.""",
    "assumptions": [
        "Both underlyings follow correlated GBM",
        "Approximation works best when K is not too large relative to F₂",
        "European exercise",
    ],
    "limitations": [
        "Approximation quality degrades for very large |K|/F₂ ratios",
        "Cannot handle K < −F₂ (denominator becomes negative)",
        "For exact pricing, use 2D numerical integration or Monte Carlo",
    ],
    "recommendations": "Industry standard for energy spread options (crack spreads, spark spreads). For K=0, use exact Margrabe formula instead.",
    "param_definitions": {
        "F₁": "Forward/futures price of asset 1 (the long leg)",
        "F₂": "Forward/futures price of asset 2 (the short leg)",
        "K": "Spread strike — the fixed spread level to be exceeded for the option to pay",
        "T": "Time to expiration in years",
        "r": "Risk-free rate for discounting",
        "σ₁": "Volatility of asset 1",
        "σ₂": "Volatility of asset 2",
        "ρ": "Correlation between asset 1 and asset 2 returns",
        "σ_kirk": "Effective spread volatility: σ_kirk = √(σ₁² − 2ρσ₁σ₂·F₂/(F₂+K) + σ₂²·(F₂/(F₂+K))²) — approximation treating F₂+K as one asset",
        "d₁": "d₁ = [ln(F₁/(F₂+K)) + σ_kirk²·T/2] / (σ_kirk·√T)",
        "d₂": "d₂ = d₁ − σ_kirk·√T",
    },
}

PRODUCTS["option_max"] = {
    "name": "Option on Maximum of Two Assets",
    "category": "exotic_two",
    "desc": "Call or put on max(S₁, S₂) — rainbow option (Stulz 1982).",
    "module": "ch05_exotic_two_assets.02_spread_max_min_options",
    "func": "option_on_max",
    "params": [
        P("S1", "Asset 1 (S₁)", 100.0, 0.01, 1e6, 1.0),
        P("S2", "Asset 2 (S₂)", 105.0, 0.01, 1e6, 1.0),
        _K, _T, _r,
        P("b1", "Carry 1 (b₁)", 0.05, -1.0, 1.0, 0.001),
        P("b2", "Carry 2 (b₂)", 0.05, -1.0, 1.0, 0.001),
        P("sigma1", "Vol 1 (σ₁)", 0.20, 0.001, 5.0, 0.01),
        P("sigma2", "Vol 2 (σ₂)", 0.20, 0.001, 5.0, 0.01),
        P("rho", "Correlation (ρ)", 0.5, -1.0, 1.0, 0.01),
    ],
    "has_option_type": True,
    "formula": r"C = C_1 + C_2 - C_{\min}",
    "formula_detail": r"""\text{Uses bivariate normal CDF for the joint probability of (S_1, S_2).}""",
    "methodology": """Stulz (1982) prices European options on the maximum of two correlated assets using the
bivariate normal distribution. Call on max = call on S₁ + call on S₂ − call on min (by identity).
Lower correlation → higher value (more diversification benefit).""",
    "assumptions": [
        "Both assets follow correlated GBM with constant parameters",
        "European exercise only",
    ],
    "limitations": [
        "Bivariate normal assumption — no jumps or stochastic volatility",
        "Correlation is assumed constant",
        "Extension to 3+ assets requires multivariate normal or MC",
    ],
    "recommendations": "Used in best-of options, outperformance certificates, and multi-asset structured products.",
    "param_definitions": {
        "S₁, S₂": "Current spot prices of the two underlying assets",
        "K": "Common strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b₁, b₂": "Cost-of-carry rates for asset 1 and asset 2 respectively",
        "σ₁, σ₂": "Volatilities of asset 1 and asset 2",
        "ρ": "Correlation between the two asset returns",
        "C₁": "Vanilla call on S₁ alone (at strike K, maturity T)",
        "C₂": "Vanilla call on S₂ alone (at strike K, maturity T)",
        "C_min": "Call on the minimum of S₁ and S₂ — computed via Stulz's bivariate normal formula",
        "M(a, b; ρ)": "Bivariate standard normal CDF — joint probability that both standardized variables are below (a, b)",
        "Identity": "C_max = C₁ + C₂ − C_min (exact, by the max + min = S₁ + S₂ payoff identity)",
    },
}

PRODUCTS["option_min"] = {
    "name": "Option on Minimum of Two Assets",
    "category": "exotic_two",
    "desc": "Call or put on min(S₁, S₂) — rainbow option (Stulz 1982).",
    "module": "ch05_exotic_two_assets.02_spread_max_min_options",
    "func": "option_on_min",
    "params": [
        P("S1", "Asset 1 (S₁)", 100.0, 0.01, 1e6, 1.0),
        P("S2", "Asset 2 (S₂)", 95.0, 0.01, 1e6, 1.0),
        _K, _T, _r,
        P("b1", "Carry 1 (b₁)", 0.05, -1.0, 1.0, 0.001),
        P("b2", "Carry 2 (b₂)", 0.05, -1.0, 1.0, 0.001),
        P("sigma1", "Vol 1 (σ₁)", 0.20, 0.001, 5.0, 0.01),
        P("sigma2", "Vol 2 (σ₂)", 0.20, 0.001, 5.0, 0.01),
        P("rho", "Correlation (ρ)", 0.5, -1.0, 1.0, 0.01),
    ],
    "has_option_type": True,
    "formula": r"C_{\min} = C_1 + C_2 - C_{\max}",
    "formula_detail": r"""\text{Complement of max: min + max = S_1 + S_2 in payout terms.}""",
    "methodology": """Prices options on the minimum of two assets. By the identity max + min = S₁ + S₂, the
min option can be derived from the max option. Higher correlation → higher min option value
(less dispersion between assets).""",
    "assumptions": [
        "Both assets follow correlated GBM",
        "European exercise",
    ],
    "limitations": [
        "Same as option on max — constant correlation, bivariate normal",
    ],
    "recommendations": "Used in worst-of options and capital protection structures. The min is the cheaper component of rainbow structures.",
    "param_definitions": {
        "S₁, S₂": "Current spot prices of the two underlying assets",
        "K": "Common strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b₁, b₂": "Cost-of-carry rates for asset 1 and asset 2",
        "σ₁, σ₂": "Volatilities of asset 1 and asset 2",
        "ρ": "Correlation between asset 1 and asset 2 returns",
        "C_max": "Call on the maximum of S₁ and S₂ (from Stulz's formula)",
        "C₁, C₂": "Vanilla calls on each individual asset",
        "Identity": "C_min = C₁ + C₂ − C_max (exact identity from max + min = S₁ + S₂ payoff)",
        "M(a, b; ρ)": "Bivariate standard normal CDF",
    },
}

PRODUCTS["quanto"] = {
    "name": "Quanto Options",
    "category": "exotic_two",
    "desc": "Foreign equity option with a pre-agreed FX conversion rate — eliminates currency risk.",
    "module": "ch05_exotic_two_assets.03_currency_translated_fx_options",
    "func": "quanto_option",
    "params": [
        P("S_f", "Foreign Asset (S_f)", 100.0, 0.01, 1e6, 1.0, "Foreign asset price in foreign currency"),
        P("K_f", "Strike (K_f)", 100.0, 0.01, 1e6, 1.0, "Strike in foreign currency"),
        _T,
        P("r_d", "Domestic Rate (r_d)", 0.05, -0.20, 1.0, 0.001),
        P("r_f", "Foreign Rate (r_f)", 0.03, -0.20, 1.0, 0.001),
        P("sigma_S", "Asset Vol (σ_S)", 0.20, 0.001, 5.0, 0.01),
        P("sigma_X", "FX Vol (σ_X)", 0.10, 0.001, 5.0, 0.01),
        P("rho", "Correlation (ρ)", -0.3, -1.0, 1.0, 0.01, "Corr(asset, FX rate)"),
        P("X0", "FX Rate (X₀)", 1.20, 0.0001, 1e4, 0.0001, "Pre-agreed conversion rate"),
    ],
    "has_option_type": True,
    "formula": r"V = X_0 \, e^{-r_d T}\left[F_q N(d_1) - K_f N(d_2)\right]",
    "formula_detail": r"""F_q = S_f \exp\{(r_d - r_f - \rho\sigma_S\sigma_X)T\}, \quad d_1 = \frac{\ln(S_f/K_f) + (r_d - r_f - \rho\sigma_S\sigma_X + \sigma_S^2/2)T}{\sigma_S\sqrt{T}}""",
    "methodology": """A quanto converts the foreign payoff to domestic currency at a fixed rate X₀, eliminating
FX risk for the holder. The quanto adjustment −ρ·σ_S·σ_X modifies the drift of the foreign asset,
reflecting the covariance between asset and FX returns.""",
    "assumptions": [
        "Foreign asset and FX rate follow correlated GBM",
        "Constant correlation, volatilities, and interest rates",
        "Fixed conversion rate X₀ predetermined at inception",
    ],
    "limitations": [
        "Quanto adjustment depends heavily on correlation, which is unstable",
        "Model risk from correlation estimation",
        "Does not capture stochastic correlation or FX smile",
    ],
    "recommendations": "Common in cross-border structured products (Nikkei options settled in USD). The quanto adjustment can significantly affect the price — check correlation sensitivity.",
    "param_definitions": {
        "S_f": "Current foreign asset price expressed in foreign currency",
        "K_f": "Strike price in foreign currency",
        "T": "Time to expiration in years",
        "r_d": "Domestic risk-free rate (currency of settlement)",
        "r_f": "Foreign risk-free rate (currency of the asset)",
        "σ_S": "Volatility of the foreign asset price",
        "σ_X": "Volatility of the FX rate (domestic per foreign)",
        "ρ": "Correlation between the foreign asset returns and the FX rate changes",
        "X₀": "Pre-agreed FX conversion rate fixed at inception — eliminates FX exposure for the holder",
        "F_q": "Quanto-adjusted forward: F_q = S_f · exp{(r_d − r_f − ρ·σ_S·σ_X)·T}",
        "Quanto adjustment": "−ρ·σ_S·σ_X — modifies the drift to compensate for the covariance between asset and FX; negative ρ increases the effective forward",
        "d₁": "d₁ = [ln(S_f/K_f) + (r_d − r_f − ρσ_Sσ_X + σ_S²/2)T] / (σ_S√T)",
        "d₂": "d₂ = d₁ − σ_S√T",
    },
}

PRODUCTS["two_asset_corr_call"] = {
    "name": "Two-Asset Correlation Option",
    "category": "exotic_two",
    "desc": "Pays (S₂ − K₂) if S₁ > K₁ at expiry — asset 1 triggers, asset 2 pays.",
    "module": "ch05_exotic_two_assets.01_margrabe_exchange",
    "func": "two_asset_correlation_call",
    "params": [
        P("S1", "Trigger Asset (S₁)", 100.0, 0.01, 1e6, 1.0, "Asset 1 — must exceed K₁ to trigger"),
        P("S2", "Payoff Asset (S₂)", 100.0, 0.01, 1e6, 1.0, "Asset 2 — payoff based on this"),
        P("K1", "Trigger Strike (K₁)", 100.0, 0.01, 1e6, 1.0, "Trigger threshold for S₁"),
        P("K2", "Payoff Strike (K₂)", 100.0, 0.01, 1e6, 1.0, "Payoff strike for S₂"),
        _T, _r,
        P("b1", "Carry 1 (b₁)", 0.05, -1.0, 1.0, 0.001),
        P("b2", "Carry 2 (b₂)", 0.05, -1.0, 1.0, 0.001),
        P("sigma1", "Vol 1 (σ₁)", 0.20, 0.001, 5.0, 0.01),
        P("sigma2", "Vol 2 (σ₂)", 0.20, 0.001, 5.0, 0.01),
        P("rho", "Correlation (ρ)", 0.5, -1.0, 1.0, 0.01),
    ],
    "has_option_type": False,
    "formula": r"V = S_2 e^{(b_2-r)T} M(d_1^+, d_2^+; \rho) - K_2 e^{-rT} M(d_1^-, d_2^-; \rho)",
    "formula_detail": r"""M(\cdot,\cdot;\rho) = \text{bivariate normal CDF}""",
    "methodology": """The two-asset correlation option has a payoff that depends on both assets: the call pays
max(S₂ − K₂, 0) only if S₁ > K₁. Uses bivariate normal CDF for the joint probability.""",
    "assumptions": [
        "Both assets follow correlated GBM",
        "European exercise",
    ],
    "limitations": [
        "Constant correlation assumption",
        "Sensitive to the correlation estimate",
    ],
    "recommendations": "Used in outperformance and conditional payout structures.",
    "param_definitions": {
        "S₁": "Trigger asset — the option only pays if S₁ > K₁ at expiry",
        "S₂": "Payoff asset — the payoff is (S₂ − K₂) when triggered",
        "K₁": "Trigger threshold for asset 1",
        "K₂": "Payoff strike for asset 2",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b₁, b₂": "Cost-of-carry rates for asset 1 and asset 2",
        "σ₁, σ₂": "Volatilities of asset 1 and asset 2",
        "ρ": "Correlation between the two asset returns",
        "M(a, b; ρ)": "Bivariate standard normal CDF — joint probability used for the conditional payoff",
        "d₁⁺, d₂⁺": "Upper integration limits: d₁⁺ = [ln(S₂/K₂) + (b₂ + σ₂²/2)T]/(σ₂√T), d₂⁺ = [ln(S₁/K₁) + (b₁ + σ₁²/2)T]/(σ₁√T)",
        "d₁⁻, d₂⁻": "Lower limits: subtract σ√T from the respective d⁺ values",
    },
}

# ========================= ALTERNATIVE MODELS =========================

PRODUCTS["merton_jump"] = {
    "name": "Merton Jump Diffusion (1976)",
    "category": "alt_models",
    "desc": "Extends BSM with Poisson-distributed jumps in the underlying — captures crash risk.",
    "module": "ch06_bsm_alternatives.01_merton_jump_diffusion",
    "func": "merton_jump_diffusion",
    "params": [
        _S, _K, _T, _r, _sig,
        P("lam", "Jump Intensity (λ)", 1.0, 0.0, 20.0, 0.1, "Expected jumps per year"),
        P("gamma", "Jump Mean (γ)", -0.05, -1.0, 1.0, 0.01, "Mean log-jump size"),
        P("delta", "Jump Vol (δ)", 0.10, 0.001, 2.0, 0.01, "Std dev of log-jump size"),
    ],
    "has_option_type": True,
    "formula": r"V = \sum_{n=0}^{\infty} \frac{e^{-\lambda'T}(\lambda'T)^n}{n!} \cdot \text{BSM}(S, K, T, r_n, \sigma_n)",
    "formula_detail": r"""\lambda' = \lambda(1+\gamma), \quad r_n = r - \lambda\gamma + n\ln(1+\gamma)/T, \quad \sigma_n^2 = \sigma^2 + n\delta^2/T""",
    "methodology": """Merton (1976) adds Poisson-distributed jumps to GBM. The option price is an infinite weighted
sum of BSM prices, where each term corresponds to exactly n jumps occurring. The weights are
Poisson probabilities. Typically 20-50 terms suffice for convergence.""",
    "assumptions": [
        "Diffusion plus compound Poisson jump process",
        "Jump sizes are log-normally distributed",
        "Jump risk is diversifiable (not priced) — original Merton assumption",
        "Between jumps, standard GBM dynamics",
    ],
    "limitations": [
        "Jump parameters (λ, γ, δ) are difficult to estimate from market data",
        "Diversifiable jump risk assumption is debatable for market-wide crashes",
        "Produces implied vol smile but may not match market smile shape exactly",
    ],
    "recommendations": "Good for capturing crash risk and explaining short-dated smile. For a more flexible smile, combine with stochastic vol (see Bates model).",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "σ": "Diffusion (continuous) volatility — the non-jump component of total vol",
        "λ": "Jump intensity — expected number of jumps per year (Poisson arrival rate)",
        "γ": "Mean log-jump size — average log-return per jump (negative = downward bias)",
        "δ": "Standard deviation of log-jump size — uncertainty in individual jump magnitude",
        "λ'": "Risk-neutral jump intensity: λ' = λ(1 + γ) — adjusted for the expected log-jump under Q",
        "r_n": "n-jump risk-free rate: r_n = r − λγ + n·ln(1+γ)/T",
        "σ_n": "n-jump volatility: σ_n² = σ² + n·δ²/T — total variance including n jumps",
        "e^{−λ'T}(λ'T)^n/n!": "Poisson probability weight for exactly n jumps in [0, T]",
        "BSM(S, K, T, r_n, σ_n)": "Standard BSM evaluated with the n-jump adjusted parameters",
        "N": "Number of Poisson terms summed (typically 20–50 terms suffice for convergence)",
    },
}

PRODUCTS["cev"] = {
    "name": "CEV Model",
    "category": "alt_models",
    "desc": "Constant Elasticity of Variance — volatility depends on the price level.",
    "module": "ch06_bsm_alternatives.02_cev_corrado_su",
    "func": "cev_option",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("beta", "Elasticity (β)", 0.5, 0.01, 2.0, 0.01, "CEV exponent: σ(S) = σ·S^(β-1)"),
    ],
    "has_option_type": True,
    "formula": r"\text{CEV: } dS = (b)S\,dt + \sigma S^{\beta}\,dW",
    "formula_detail": r"""\beta < 1: \text{negative skew (leverage effect)}, \quad \beta = 1: \text{BSM}, \quad \beta > 1: \text{positive skew}""",
    "methodology": """The CEV model allows local volatility to depend on the stock price level: σ(S) = σ·S^{β−1}.
When β < 1 (e.g., 0.5), lower prices have higher volatility — the leverage effect — generating
negative skew. When β = 1, it reduces to BSM. Priced using the non-central chi-squared distribution.""",
    "assumptions": [
        "Local volatility is a power function of S",
        "β is constant over time",
        "No jumps",
    ],
    "limitations": [
        "One-parameter extension of BSM — limited flexibility",
        "For β < 1, the process can hit zero (absorption boundary)",
        "Cannot capture term structure of skew independently",
    ],
    "recommendations": "Simple way to introduce skew. β ≈ 0.5 for equities (leverage effect). For richer dynamics, consider stochastic volatility models.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Scale parameter of the CEV volatility function (not the same as BSM σ)",
        "β": "CEV elasticity exponent — determines how volatility changes with price: σ(S) = σ·S^{β−1}. β=1: BSM (constant vol); β<1: vol decreases as S rises (leverage); β>1: vol increases with S",
        "σ(S)": "Local volatility: σ(S) = σ·S^{β−1} — the instantaneous vol at price level S",
        "Non-central χ²": "Pricing formula uses the non-central chi-squared distribution (not closed-form in BSM sense)",
    },
}

PRODUCTS["corrado_su"] = {
    "name": "Corrado-Su — Skewness & Kurtosis",
    "category": "alt_models",
    "desc": "Adjusts BSM for non-normal returns using skewness and excess kurtosis parameters.",
    "module": "ch06_bsm_alternatives.02_cev_corrado_su",
    "func": "corrado_su_option",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("skewness", "Skewness (μ₃)", -0.5, -5.0, 5.0, 0.01, "Return distribution skewness"),
        P("excess_kurtosis", "Excess Kurtosis (μ₄)", 3.0, -2.0, 50.0, 0.1, "Excess kurtosis above normal"),
    ],
    "has_option_type": True,
    "formula": r"V = V_{BSM} + \mu_3 Q_3 + \mu_4 Q_4",
    "formula_detail": r"""Q_3 = \frac{1}{3!}S\sigma\sqrt{T}(2\sigma\sqrt{T}-d)\cdot n(d), \quad Q_4 = \frac{1}{4!}S\sigma\sqrt{T}(d^2-1-3\sigma\sqrt{T}(d-\sigma\sqrt{T}))\cdot n(d)""",
    "methodology": """Corrado-Su (1996) uses a Gram-Charlier expansion to adjust BSM for non-zero skewness and
excess kurtosis. The correction terms Q₃ (skewness) and Q₄ (kurtosis) are added to the
BSM price, providing a first-order approximation to non-normal returns.""",
    "assumptions": [
        "Returns are approximately described by a Gram-Charlier expansion",
        "Higher moments (>4th) are negligible",
        "Skewness and kurtosis are constant",
    ],
    "limitations": [
        "Gram-Charlier can produce negative probabilities for extreme skew/kurtosis",
        "Only first-order correction — may not fit deep OTM options well",
        "Parameters must be estimated from market data",
    ],
    "recommendations": "Quick way to adjust BSM for observed non-normality. Useful for back-of-envelope smile adjustments. For production use, prefer Heston or SABR.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility (same as BSM σ)",
        "μ₃ (skewness)": "Third standardized moment of returns; negative = left-skewed distribution (heavier left tail)",
        "μ₄ (excess kurtosis)": "Fourth standardized moment minus 3; positive = heavier tails than normal (leptokurtic)",
        "V_BSM": "Standard BSM option price — the base value before moment corrections",
        "Q₃": "Skewness correction term: Q₃ = S·σ√T·(2σ√T − d)·n(d) / 3! — proportional to μ₃",
        "Q₄": "Kurtosis correction term: Q₄ = S·σ√T·(d² − 1 − 3σ√T(d − σ√T))·n(d) / 4! — proportional to μ₄",
        "d": "Standardized moneyness: d = [ln(S/K) + (b + σ²/2)T] / (σ√T) (equivalent to d₁)",
        "n(d)": "Standard normal PDF evaluated at d: n(d) = (1/√(2π))·e^{−d²/2}",
    },
}

PRODUCTS["heston"] = {
    "name": "Heston Stochastic Volatility",
    "category": "alt_models",
    "desc": "Volatility follows a mean-reverting square-root process — captures smile and term structure.",
    "module": "ch06_bsm_alternatives.03_stochastic_vol_variance_swaps",
    "func": "heston_option",
    "params": [
        _S, _K, _T, _r,
        P("q", "Dividend Yield (q)", 0.0, 0.0, 1.0, 0.001),
        P("v0", "Initial Variance (v₀)", 0.04, 0.0001, 5.0, 0.001, "Current instantaneous variance"),
        P("kappa", "Mean Reversion (κ)", 2.0, 0.01, 20.0, 0.1, "Speed of variance mean reversion"),
        P("theta", "Long-run Var (θ)", 0.04, 0.0001, 5.0, 0.001, "Long-run variance level"),
        P("xi", "Vol of Vol (ξ)", 0.30, 0.01, 5.0, 0.01, "Volatility of variance process"),
        P("rho", "Correlation (ρ)", -0.70, -1.0, 1.0, 0.01, "Corr(stock, variance)"),
    ],
    "has_option_type": True,
    "formula": r"V = S e^{-qT} P_1 - K e^{-rT} P_2",
    "formula_detail": r"""P_j = \frac{1}{2} + \frac{1}{\pi}\int_0^{\infty} \text{Re}\left[\frac{e^{-iu\ln K}\phi_j(u)}{iu}\right]du, \quad dv = \kappa(\theta - v)dt + \xi\sqrt{v}\,dW^v""",
    "methodology": """Heston (1993) models variance v as a CIR process: dv = κ(θ−v)dt + ξ√v dW^v with
correlation ρ between stock and variance Brownian motions. The characteristic function has a
semi-analytical form, enabling fast pricing via Fourier inversion (numerical integration).
Negative ρ generates the leverage effect (negative skew).""",
    "assumptions": [
        "Variance follows a CIR (square-root) mean-reverting process",
        "Constant correlation between spot and variance innovations",
        "No jumps (for jumps, see Bates model)",
        "Feller condition: 2κθ > ξ² ensures variance stays positive",
    ],
    "limitations": [
        "5 parameters (κ, θ, ξ, ρ, v₀) require calibration to the vol surface",
        "May not fit short-term smile accurately without jumps",
        "Characteristic function inversion requires careful numerical integration",
        "Calibration is a non-convex optimization problem",
    ],
    "recommendations": "Industry standard for equity and FX vol smile modeling. Widely used in structured products pricing. Combine with jumps (Bates model) for better short-term smile fit.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "q": "Continuous dividend yield",
        "v₀": "Initial (current) instantaneous variance — v₀ = σ_spot², so σ_spot = √v₀",
        "κ (kappa)": "Mean reversion speed of variance — how quickly v reverts to θ; higher κ = faster reversion",
        "θ (theta)": "Long-run variance level — the equilibrium variance as t → ∞; long-run vol = √θ",
        "ξ (xi)": "Volatility of variance ('vol of vol') — governs smile curvature; higher ξ = more pronounced smile",
        "ρ (rho)": "Correlation between stock price and variance innovations; ρ < 0 produces negative skew (leverage effect)",
        "P₁, P₂": "Risk-neutral probabilities obtained by inverting the characteristic function via Fourier transform",
        "φ_j(u)": "Characteristic function of the log-price under Heston — a complex-valued function of frequency u; P_j = ½ + (1/π)∫Re[e^{−iu·ln(K)}φ_j(u)/iu]du",
        "Feller condition": "2κθ > ξ² — ensures the CIR variance process stays strictly positive",
        "dv": "Variance SDE: dv = κ(θ−v)dt + ξ√v·dW^v",
    },
}

PRODUCTS["sabr"] = {
    "name": "SABR Model",
    "category": "alt_models",
    "desc": "Stochastic Alpha Beta Rho — the standard interest rate smile model.",
    "module": "ch06_bsm_alternatives.04_sabr_bates",
    "func": "sabr_option",
    "params": [
        P("F", "Forward Price (F)", 100.0, 0.01, 1e6, 1.0, "Forward/futures price"),
        _K, _T, _r,
        P("alpha", "Alpha (α)", 0.20, 0.001, 5.0, 0.01, "Initial stochastic volatility level"),
        P("beta", "Beta (β)", 0.5, 0.0, 1.0, 0.01, "CEV exponent (0=normal, 1=lognormal)"),
        P("rho", "Rho (ρ)", -0.3, -0.999, 0.999, 0.01, "Corr(forward, vol)"),
        P("nu", "Nu (ν)", 0.40, 0.001, 5.0, 0.01, "Vol of vol"),
    ],
    "has_option_type": True,
    "formula": r"\sigma_{SABR}(K) = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)} \cdot \left(1 + \epsilon\right)",
    "formula_detail": r"""z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln(F/K), \quad x(z) = \ln\left(\frac{\sqrt{1-2\rho z+z^2}+z-\rho}{1-\rho}\right)""",
    "methodology": """SABR (Hagan et al. 2002) is a stochastic volatility model where both the forward and its
volatility are stochastic. The Hagan formula provides an analytical approximation for the
implied volatility as a function of strike, producing a smile/skew. β controls backbone,
ρ controls skew direction, ν controls smile curvature.""",
    "assumptions": [
        "Forward follows dF = αF^β dW¹, volatility follows dα = ν·α dW² with corr ρ",
        "Hagan's formula is an asymptotic expansion (accurate for moderate T)",
        "β is typically fixed and other params (α, ρ, ν) are calibrated",
    ],
    "limitations": [
        "Hagan's approximation breaks down for very long maturities or extreme strikes",
        "Arbitrage-free versions (e.g., SABR with no-arbitrage conditions) are more complex",
        "Backbone β must be chosen carefully (0.5 typical for rates, 1.0 for FX)",
    ],
    "recommendations": "The standard model for interest rate smile (swaptions, caps). Widely used in rates trading desks. For equities, Heston is more common.",
    "param_definitions": {
        "F": "Current forward/futures price of the underlying",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate (for discounting in Black-76 wrapper)",
        "α": "Initial stochastic volatility level — the current instantaneous vol of the forward",
        "β": "CEV backbone exponent (0 = normal/Bachelier, 0.5 = common choice for rates, 1 = lognormal/Black); controls how vol scales with F",
        "ρ": "Correlation between forward price and volatility innovations — controls smile skew direction",
        "ν (nu)": "Volatility of volatility — controls the smile curvature/wings; higher ν = more pronounced smile",
        "σ_SABR(K)": "Implied Black vol as a function of strike K — the SABR formula outputs this, which is then used in Black-76 for pricing",
        "z": "Scaled log-moneyness: z = (ν/α)·(FK)^{(1−β)/2}·ln(F/K)",
        "x(z)": "Log-CEV transformation: x(z) = ln{[√(1−2ρz+z²) + z − ρ] / (1−ρ)}",
        "ε": "Higher-order correction terms accounting for β, ρ, ν effects on the asymptotic expansion",
    },
}

PRODUCTS["bates"] = {
    "name": "Bates Jump-Diffusion SV",
    "category": "alt_models",
    "desc": "Heston + Merton jumps — stochastic volatility with jumps for extreme smile fitting.",
    "module": "ch06_bsm_alternatives.04_sabr_bates",
    "func": "bates_option",
    "params": [
        _S, _K, _T, _r,
        P("q", "Dividend Yield (q)", 0.0, 0.0, 1.0, 0.001),
        P("v0", "Initial Variance (v₀)", 0.04, 0.0001, 5.0, 0.001),
        P("kappa", "Mean Reversion (κ)", 2.0, 0.01, 20.0, 0.1),
        P("theta", "Long-run Var (θ)", 0.04, 0.0001, 5.0, 0.001),
        P("xi", "Vol of Vol (ξ)", 0.30, 0.01, 5.0, 0.01),
        P("rho", "Correlation (ρ)", -0.70, -1.0, 1.0, 0.01),
        P("lam", "Jump Intensity (λ)", 1.0, 0.0, 20.0, 0.1),
        P("mu_j", "Jump Mean (μ_J)", -0.05, -1.0, 1.0, 0.01),
        P("delta_j", "Jump Vol (δ_J)", 0.10, 0.001, 2.0, 0.01),
    ],
    "has_option_type": True,
    "formula": r"V = S e^{-qT} P_1 - K e^{-rT} P_2 \quad \text{(Heston CF + Poisson jumps)}",
    "formula_detail": r"""\text{Characteristic function combines Heston's SV with Merton's jump component.}""",
    "methodology": """Bates (1996) combines Heston's stochastic variance with Merton's Poisson jumps. The
characteristic function has a semi-analytical form combining both features. This provides
maximum flexibility for fitting the full implied volatility surface.""",
    "assumptions": [
        "Heston SV dynamics + compound Poisson jumps",
        "Jump sizes are log-normally distributed",
        "8 parameters require careful calibration",
    ],
    "limitations": [
        "Many parameters — risk of overfitting",
        "Calibration is computationally demanding",
        "Parameter identification can be poor (multiple local optima)",
    ],
    "recommendations": "Use when Heston alone cannot fit short-term smile. The jump component captures sudden large moves while SV handles the term structure.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "q": "Continuous dividend yield",
        "v₀": "Initial instantaneous variance (v₀ = σ_spot²)",
        "κ": "Speed of mean reversion of variance",
        "θ": "Long-run variance level",
        "ξ": "Volatility of variance (vol of vol)",
        "ρ": "Correlation between stock price and variance shocks",
        "λ": "Poisson jump intensity — expected jumps per year",
        "μ_J": "Mean log-jump size per jump (negative for crash-bias)",
        "δ_J": "Standard deviation of log-jump size per jump",
        "P₁, P₂": "Risk-neutral probabilities computed via Fourier inversion of the combined Heston + jump characteristic function",
        "Combined CF": "φ(u) = exp{Heston_CF(u) + Jump_CF(u)} — the jump contribution adds −λT[e^{iuμ_J − u²δ_J²/2} − 1] to the Heston exponent",
    },
}

# ========================= NUMERICAL METHODS =========================

PRODUCTS["crr_binomial"] = {
    "name": "CRR Binomial Tree",
    "category": "numerical",
    "desc": "Cox-Ross-Rubinstein (1979) binomial lattice — versatile numerical method for European and American options.",
    "module": "ch07_trees.01_binomial_trees",
    "func": "crr_binomial_tree",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("N", "Tree Steps (N)", 200, 10, 2000, 10, "Number of time steps in the tree"),
    ],
    "has_option_type": True,
    "extra_choices": {
        "exercise": {"label": "Exercise Style", "options": ["european", "american"], "default": "european"},
    },
    "formula": r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{b\Delta t} - d}{u - d}",
    "formula_detail": r"""\text{Backward induction: } V_{i,j} = e^{-r\Delta t}[p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}]""",
    "methodology": """The CRR tree discretizes the continuous GBM into a recombining binomial lattice. At each step,
the price moves up by factor u or down by d=1/u. The risk-neutral probability p ensures
no-arbitrage. American exercise is handled by comparing continuation value with intrinsic value
at each node. Convergence: O(1/N).""",
    "assumptions": [
        "Lattice approximates continuous GBM",
        "Risk-neutral pricing",
        "Recombining tree (ud = 1)",
    ],
    "limitations": [
        "Oscillatory convergence for European options — use odd N",
        "For barrier options near barrier-node boundaries, use adaptive mesh",
        "Slow for high accuracy — O(N²) complexity",
    ],
    "recommendations": "The pedagogical standard. Use N=200+ for production accuracy. For faster convergence, use Leisen-Reimer. For barriers, use the barrier tree variant.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "N": "Number of time steps in the tree (accuracy improves as N increases; use N≥200)",
        "Δt": "Length of each time step: Δt = T / N",
        "u": "Up-move factor: u = e^{σ√Δt} — multiplicative price increase per step",
        "d": "Down-move factor: d = 1/u = e^{−σ√Δt} — multiplicative price decrease per step (recombining property: ud = 1)",
        "p": "Risk-neutral up-probability: p = (e^{bΔt} − d) / (u − d) — ensures no-arbitrage",
        "1−p": "Risk-neutral down-probability",
        "V_{i,j}": "Option value at time step i, price node j — computed by backward induction from expiry",
        "e^{−rΔt}": "Discount factor applied at each step",
    },
}

PRODUCTS["leisen_reimer"] = {
    "name": "Leisen-Reimer Tree",
    "category": "numerical",
    "desc": "Improved binomial tree with faster convergence — no oscillation.",
    "module": "ch07_trees.01_binomial_trees",
    "func": "leisen_reimer_tree",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("N", "Tree Steps (N)", 201, 11, 2001, 10, "Number of steps (must be odd)"),
    ],
    "has_option_type": True,
    "extra_choices": {
        "exercise": {"label": "Exercise Style", "options": ["european", "american"], "default": "european"},
    },
    "formula": r"\text{Uses Peizer-Pratt inversion for } p, p' \text{ to center the tree on strike}",
    "formula_detail": r"""\text{Convergence: } O(1/N^2) \text{ — much faster than CRR's } O(1/N)""",
    "methodology": """Leisen-Reimer (1996) constructs the binomial tree so that the strike K coincides with a
node, eliminating the oscillatory convergence of CRR. Uses the Peizer-Pratt inversion to
set p and u. Achieves O(1/N²) convergence — 10× fewer steps for the same accuracy.""",
    "assumptions": [
        "Same as CRR — GBM approximation via binomial lattice",
        "N must be odd for the centering to work",
    ],
    "limitations": [
        "Slightly more complex implementation than CRR",
        "Same O(N²) complexity per step",
    ],
    "recommendations": "Preferred over CRR when accuracy matters. N=101 Leisen-Reimer ≈ N=1000 CRR in accuracy.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "N": "Number of time steps (must be odd to ensure the strike K falls exactly on a node)",
        "Δt": "Time step size: Δt = T / N",
        "p (LR)": "Risk-neutral probability computed via the Peizer-Pratt (1968) inversion of the normal CDF, ensuring the tree is centered on d₁ and d₂",
        "p' (LR)": "Modified probability used for the asset-measure component (parallel to CRR's p but centered on d₁)",
        "h(x, n)": "Peizer-Pratt inversion function mapping a standard normal quantile to a binomial probability for finite n",
        "Convergence": "O(1/N²) vs CRR's O(1/N) — achieved by centering the tree on the strike, eliminating the main oscillation error",
    },
}

PRODUCTS["trinomial_tree"] = {
    "name": "Trinomial Tree",
    "category": "numerical",
    "desc": "Three-branch lattice with up, middle, and down moves — smoother convergence than binomial.",
    "module": "ch07_trees.01_binomial_trees",
    "func": "trinomial_tree",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("N", "Tree Steps (N)", 100, 10, 1000, 10, "Number of time steps"),
    ],
    "has_option_type": True,
    "extra_choices": {
        "exercise": {"label": "Exercise Style", "options": ["european", "american"], "default": "european"},
    },
    "formula": r"u = e^{\sigma\sqrt{2\Delta t}}, \quad d = 1/u, \quad m = 1",
    "formula_detail": r"""p_u = \left(\frac{e^{b\Delta t/2} - e^{-\sigma\sqrt{\Delta t/2}}}{e^{\sigma\sqrt{\Delta t/2}} - e^{-\sigma\sqrt{\Delta t/2}}}\right)^2""",
    "methodology": """Boyle (1986) extends the binomial tree to three branches. The extra degree of freedom improves
convergence and naturally handles dividend adjustments. Equivalent to an explicit finite
difference scheme on a log-price grid.""",
    "assumptions": [
        "Three possible moves at each step: up, middle, down",
        "Recombining lattice (trinomial)",
        "Risk-neutral probabilities sum to 1 and match first two moments of GBM",
    ],
    "limitations": [
        "O(N²) complexity per step",
        "For exotic path-dependent options, non-recombining version needed",
    ],
    "recommendations": "Good alternative to binomial trees. Natural connection to explicit finite differences. Use for American options and barrier options.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "N": "Number of time steps",
        "Δt": "Time step size: Δt = T / N",
        "u": "Up-move factor: u = e^{σ√(2Δt)} — larger than CRR's u to accommodate the middle branch",
        "d": "Down-move factor: d = 1/u = e^{−σ√(2Δt)}",
        "m": "Middle-move factor = 1 (price stays flat for the middle branch)",
        "p_u": "Up-move risk-neutral probability: p_u = [(e^{bΔt/2} − e^{−σ√(Δt/2)}) / (e^{σ√(Δt/2)} − e^{−σ√(Δt/2)})]²",
        "p_d": "Down-move probability: p_d = [(e^{σ√(Δt/2)} − e^{bΔt/2}) / (e^{σ√(Δt/2)} − e^{−σ√(Δt/2)})]²",
        "p_m": "Middle-move probability: p_m = 1 − p_u − p_d",
    },
}

PRODUCTS["crank_nicolson"] = {
    "name": "Crank-Nicolson Finite Difference",
    "category": "numerical",
    "desc": "Implicit-explicit hybrid FDM — second-order accurate in both time and space.",
    "module": "ch07_trees.02_finite_difference",
    "func": "crank_nicolson_fdm",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("N_S", "Price Grid Points", 200, 50, 2000, 50, "Number of spatial grid points"),
        P("N_T", "Time Steps", 100, 10, 2000, 10, "Number of time steps"),
    ],
    "has_option_type": True,
    "extra_choices": {
        "exercise": {"label": "Exercise Style", "options": ["european", "american"], "default": "european"},
    },
    "formula": r"\frac{V_j^{n+1} - V_j^n}{\Delta t} = \frac{1}{2}[\mathcal{L}V^{n+1} + \mathcal{L}V^n]",
    "formula_detail": r"""\mathcal{L}V = \frac{1}{2}\sigma^2 S^2 V_{SS} + bS V_S - rV""",
    "methodology": """Crank-Nicolson averages the explicit and implicit finite difference schemes, achieving
second-order accuracy in both time O(Δt²) and space O(ΔS²). Unconditionally stable (no CFL
restriction on Δt/ΔS²). Solved via tridiagonal matrix inversion (Thomas algorithm).""",
    "assumptions": [
        "BSM PDE discretized on a finite grid",
        "Boundary conditions at S=0 and S=S_max",
        "For American: projection onto exercise constraint at each time step",
    ],
    "limitations": [
        "Requires careful boundary condition specification",
        "Can exhibit oscillations (Rannacher time-stepping fixes this)",
        "Grid resolution needed near barriers or discontinuities",
    ],
    "recommendations": "The gold standard for 1D PDE-based pricing. Excellent for American options, barrier options, and any problem expressible as a PDE. Use at least 200 spatial points.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "N_S": "Number of spatial (price) grid points — more points give finer resolution near the strike",
        "N_T": "Number of time steps — more steps improve time accuracy; total nodes = N_S × N_T",
        "Δt": "Time step size: Δt = T / N_T",
        "ΔS": "Price step size: ΔS = S_max / N_S (where S_max is the grid upper boundary, typically 3–5× S)",
        "ℒV": "BSM differential operator: ℒV = ½σ²S²·V_SS + bS·V_S − rV",
        "V_SS": "Second derivative of option value w.r.t. S (Gamma term in the PDE)",
        "V_S": "First derivative of option value w.r.t. S (Delta term in the PDE)",
        "½[ℒV^{n+1} + ℒV^n]": "Crank-Nicolson averaging — half implicit (n+1) + half explicit (n) — gives O(Δt²) accuracy and unconditional stability",
        "Thomas algorithm": "Efficient O(N) tridiagonal solver used at each time step — exploits the banded structure of the discretized PDE",
    },
}

PRODUCTS["mc_european"] = {
    "name": "Monte Carlo — European",
    "category": "numerical",
    "desc": "Price European options via Monte Carlo simulation with variance reduction.",
    "module": "ch08_monte_carlo.01_monte_carlo",
    "func": "mc_european",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("n_paths", "Number of Paths", 100000, 1000, 1000000, 10000, "Simulation paths"),
    ],
    "has_option_type": True,
    "returns_dict": True,
    "formula": r"V = e^{-rT} \frac{1}{N}\sum_{i=1}^{N} \max(\phi(S_T^{(i)} - K), 0)",
    "formula_detail": r"""S_T = S \exp\left[(b - \sigma^2/2)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim N(0,1)""",
    "methodology": """Generate N random terminal prices from GBM and average the discounted payoffs. Standard
error = σ_payoff / √N. Variance reduction (antithetic variates, control variates) can
reduce the required number of paths by 2-10×.""",
    "assumptions": [
        "GBM dynamics for the underlying",
        "Payoff depends only on terminal price (no path dependence for basic MC)",
        "Law of large numbers ensures convergence",
    ],
    "limitations": [
        "Convergence rate O(1/√N) — slow for high accuracy",
        "Standard error quantifies precision, not accuracy of the model",
        "Not directly applicable to American options (use LSM variant)",
    ],
    "recommendations": "Most flexible pricing method — works for any payoff structure. Use 100,000+ paths for production. Combine with variance reduction for efficiency.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "N (n_paths)": "Number of simulation paths — standard error ∝ 1/√N; 100,000 paths gives ~0.3% precision",
        "S_T^(i)": "Simulated terminal price for path i: S_T = S·exp[(b − σ²/2)T + σ√T·Z]",
        "Z": "Standard normal random variable Z ~ N(0,1) — one per path",
        "φ": "Option type sign: φ = +1 for call, φ = −1 for put",
        "e^{−rT}": "Discount factor to present-value the average payoff",
        "Std Error": "Standard error of the estimate: σ_payoff / √N — shown in output to quantify simulation noise",
        "95% CI": "95% confidence interval: price ± 1.96 × Std Error",
    },
}

PRODUCTS["mc_american_lsm"] = {
    "name": "Monte Carlo — American (LSM)",
    "category": "numerical",
    "desc": "Longstaff-Schwartz Least-Squares Monte Carlo for American options.",
    "module": "ch08_monte_carlo.01_monte_carlo",
    "func": "mc_american_lsm",
    "params": [
        _S, _K, _T, _r, _b, _sig,
        P("n_paths", "Number of Paths", 50000, 1000, 500000, 5000, "Simulation paths"),
        P("n_steps", "Time Steps", 50, 10, 500, 10, "Exercise dates"),
    ],
    "has_option_type": True,
    "returns_dict": True,
    "formula": r"V = \max\left(h(S_t), \; E^*\left[e^{-r\Delta t} V_{t+1} \,|\, S_t\right]\right)",
    "formula_detail": r"""\text{Continuation value estimated by regressing } e^{-r\Delta t}V_{t+1} \text{ on basis functions of } S_t""",
    "methodology": """Longstaff-Schwartz (2001) estimates the optimal exercise strategy via backward regression.
At each exercise date, the continuation value is approximated by regressing discounted future
cashflows on polynomial basis functions of the current state. Exercise if intrinsic > continuation.""",
    "assumptions": [
        "GBM paths discretized at n_steps exercise dates",
        "Polynomial regression captures the true continuation value",
        "Bermudan approximation of continuous American exercise",
    ],
    "limitations": [
        "Biased low (sub-optimal exercise) — typically 5-20 basis points",
        "Requires many paths (50,000+) for stable regression",
        "Multi-asset LSM suffers from curse of dimensionality",
    ],
    "recommendations": "Standard method for American-style exotic options. Use 50,000+ paths and 50+ time steps. Polynomial degree 2-3 is usually sufficient.",
    "param_definitions": {
        "S": "Current spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "b": "Cost-of-carry rate",
        "σ": "Annualized volatility",
        "n_paths": "Number of simulated price paths — more paths reduce regression noise and improve accuracy",
        "n_steps": "Number of Bermudan exercise dates — approximation to continuous American exercise; 50+ steps recommended",
        "h(S_t)": "Intrinsic (exercise) value at time t: max(S_t − K, 0) for a call",
        "E*[e^{−rΔt}V_{t+1} | S_t]": "Continuation value — expected discounted future payoff if not exercised now, estimated by regression",
        "Regression": "Least-squares regression of discounted future cashflows on basis functions of S_t (e.g., 1, S_t, S_t²) to estimate the continuation value",
        "Exercise boundary": "The critical S_t above which early exercise is optimal (where h(S_t) > continuation value)",
        "Backward induction": "Algorithm proceeds from final expiry backwards in time, updating exercise decisions at each step",
    },
}

# ========================= COMMODITY & ENERGY =========================

PRODUCTS["commodity_spot"] = {
    "name": "Commodity Spot Option",
    "category": "commodity",
    "desc": "European option on a commodity spot price with storage costs and convenience yield.",
    "module": "ch10_commodity_energy.01_energy_commodity_options",
    "func": "commodity_option_spot",
    "params": [
        _S, _K, _T, _r,
        P("u", "Storage Cost (u)", 0.02, 0.0, 1.0, 0.001, "Annual storage cost rate"),
        P("y", "Convenience Yield (y)", 0.05, 0.0, 1.0, 0.001, "Annual convenience yield"),
        _sig,
    ],
    "has_option_type": True,
    "formula": r"V = \text{BSM}(S, K, T, r, b=r+u-y, \sigma)",
    "formula_detail": r"""b = r + u - y \quad \text{(cost of carry for commodities)}""",
    "methodology": """Commodity options use the generalized BSM with cost-of-carry b = r + storage − convenience yield.
Storage costs increase carry cost, while convenience yield (benefit of holding physical commodity)
decreases it. If b > 0, contango; if b < 0, backwardation.""",
    "assumptions": [
        "Spot price follows GBM with commodity-specific carry",
        "Storage costs and convenience yield are constant",
        "No delivery or quality issues",
    ],
    "limitations": [
        "Convenience yield is hard to observe directly",
        "Mean-reverting commodity prices violate GBM — see Schwartz model",
    ],
    "recommendations": "For simple commodity options. For mean-reverting prices (energy, agri), use the Schwartz model.",
}

PRODUCTS["schwartz_mr"] = {
    "name": "Schwartz Mean-Reverting",
    "category": "commodity",
    "desc": "Commodity option with Ornstein-Uhlenbeck mean-reverting log-price dynamics.",
    "module": "ch10_commodity_energy.01_energy_commodity_options",
    "func": "schwartz_mean_reversion_option",
    "params": [
        _S, _K, _T, _r, _sig,
        P("kappa", "Mean Reversion (κ)", 1.0, 0.01, 20.0, 0.1, "Speed of mean reversion"),
        P("mu", "Long-run Mean (μ)", 4.6, 0.0, 20.0, 0.1, "Long-run equilibrium log-price"),
        P("lam", "Market Price of Risk (λ)", 0.0, -5.0, 5.0, 0.01, "Risk premium for commodity"),
    ],
    "has_option_type": True,
    "formula": r"V = S \exp\{(\alpha - r)T\} N(d_1) - K e^{-rT} N(d_2)",
    "formula_detail": r"""\alpha = (1-e^{-\kappa T})(\mu - \lambda/\kappa - \sigma^2/2\kappa)/T, \quad \sigma_T^2 = \sigma^2(1-e^{-2\kappa T})/(2\kappa T)""",
    "methodology": """Schwartz (1997) one-factor model assumes log-price follows an Ornstein-Uhlenbeck process,
mean-reverting to a long-run level. The effective volatility decreases with maturity due to
mean reversion, naturally producing a downward-sloping term structure of volatility.""",
    "assumptions": [
        "Log-price mean-reverts to equilibrium level μ",
        "Constant mean reversion speed κ and long-run volatility σ",
        "Risk premium λ adjusts the drift under the physical measure",
    ],
    "limitations": [
        "Single-factor — cannot capture multi-factor term structure dynamics",
        "Constant parameters may not fit all maturities simultaneously",
    ],
    "recommendations": "Standard model for energy commodities (oil, gas, power). The mean reversion naturally fits the term structure of futures volatility.",
    "param_definitions": {
        "S": "Current commodity spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "σ": "Short-term (instantaneous) volatility of the log-price",
        "κ": "Speed of mean reversion — how quickly the log-price reverts to μ; higher κ = faster reversion and lower long-dated vol",
        "μ": "Long-run equilibrium log-price (e.g., μ=4.6 corresponds to spot ~100)",
        "λ": "Market price of commodity risk — adjusts the risk-neutral drift; λ=0 means no risk premium",
        "α": "Risk-neutral drift parameter: α = (1−e^{−κT})(μ − λ/κ − σ²/(2κ)) / T — the effective adjusted growth rate",
        "σ_T": "Effective maturity-adjusted volatility: σ_T² = σ²(1−e^{−2κT})/(2κT) — decreases with T due to mean reversion",
        "d₁, d₂": "Moneyness measures using α and σ_T instead of the standard b and σ",
    },
}

PRODUCTS["energy_swaption"] = {
    "name": "Energy Swaption",
    "category": "commodity",
    "desc": "Option on an energy swap — right to enter a fixed-for-floating energy swap.",
    "module": "ch10_commodity_energy.01_energy_commodity_options",
    "func": "energy_swaption",
    "params": [
        P("F_avg_fwd", "Average Forward (F)", 50.0, 0.01, 1e4, 0.1, "Average forward price over swap period"),
        P("K", "Swap Fixed Price (K)", 50.0, 0.01, 1e4, 0.1, "Fixed price in the swap"),
        P("T_expiry", "Expiry (T)", 0.5, 0.001, 30.0, 0.01, "Time to swaption expiry"),
        _r,
        P("sigma_swap", "Swap Vol (σ)", 0.25, 0.001, 5.0, 0.01, "Volatility of the swap rate"),
        P("notional", "Notional", 1.0, 0.01, 1e9, 1.0, "Contract notional"),
    ],
    "has_option_type": True,
    "formula": r"V = \text{notional} \cdot e^{-rT}\left[F\,N(d_1) - K\,N(d_2)\right]",
    "formula_detail": r"""\text{Black-76 applied to the average forward swap rate.}""",
    "methodology": """Energy swaptions are priced using Black-76 applied to the swap rate. The average forward
price over the swap delivery period is used as the forward. The volatility represents
the uncertainty of the swap rate, not individual forward prices.""",
    "assumptions": [
        "Swap rate is log-normally distributed",
        "Black-76 framework",
        "Single volatility for the average forward",
    ],
    "limitations": [
        "Ignores correlation structure between individual forward periods",
        "For structured swaps, multi-factor models may be needed",
    ],
    "recommendations": "Standard market practice for energy swaptions. Used extensively in natural gas and power markets.",
    "param_definitions": {
        "F": "Average forward price over the swap delivery period — the 'forward swap rate' for energy",
        "K": "Fixed price in the swap agreement (the swap strike)",
        "T": "Time to swaption expiry in years",
        "r": "Risk-free discount rate",
        "σ": "Volatility of the average forward swap rate (implied from market swaption prices)",
        "notional": "Contract notional — scales the dollar value of the swaption",
        "d₁": "d₁ = [ln(F/K) + σ²T/2] / (σ√T)",
        "d₂": "d₂ = d₁ − σ√T",
        "N(d₁), N(d₂)": "Standard normal CDF — risk-neutral probabilities of the swap being in-the-money",
    },
}

# ========================= INTEREST RATE =========================

PRODUCTS["caplet"] = {
    "name": "Caplet / Floorlet",
    "category": "interest_rate",
    "desc": "Interest rate caplet (call on rate) or floorlet (put on rate) using Black-76.",
    "module": "ch11_interest_rate.01_interest_rate_options",
    "func": "caplet_floorlet",
    "params": [
        P("F", "Forward Rate (F)", 0.05, -0.10, 1.0, 0.001, "Forward LIBOR/SOFR rate"),
        P("K", "Strike Rate (K)", 0.05, -0.10, 1.0, 0.001, "Cap/floor strike rate"),
        P("T", "Expiry (T)", 1.0, 0.01, 30.0, 0.01, "Time to caplet/floorlet expiry"),
        P("r", "Discount Rate (r)", 0.04, -0.10, 1.0, 0.001, "Risk-free discount rate"),
        P("sigma", "Rate Vol (σ)", 0.20, 0.001, 5.0, 0.01, "Black volatility of forward rate"),
        P("tau", "Accrual Period (τ)", 0.25, 0.01, 2.0, 0.01, "Day count fraction (e.g., 0.25 for quarterly)"),
        P("notional", "Notional", 1000000.0, 1.0, 1e12, 1000.0, "Notional principal"),
    ],
    "has_option_type": False,
    "extra_choices": {
        "option_type": {"label": "Type", "options": ["cap", "floor"], "default": "cap"},
    },
    "formula": r"V_{caplet} = N \cdot \tau \cdot e^{-rT}\left[F\,N(d_1) - K\,N(d_2)\right]",
    "formula_detail": r"""d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}""",
    "methodology": """Caplets and floorlets are the building blocks of interest rate caps and floors. Priced
using Black-76 applied to the forward rate. A cap is a strip of caplets. The payoff at T+τ
is N·τ·max(F−K, 0) for a caplet, discounted to today.""",
    "assumptions": [
        "Forward rate is log-normally distributed (Black model)",
        "Single volatility for each caplet",
        "No correlation between forward rates (for individual caplet pricing)",
    ],
    "limitations": [
        "Log-normal assumption may be inadequate for negative rates — use Bachelier/shifted Black",
        "Each caplet uses a different (flat) vol — need to strip the vol surface",
        "Does not capture rate smile — use SABR for smile-consistent pricing",
    ],
    "recommendations": "Standard market model for cap/floor pricing. For smile, calibrate SABR to each caplet maturity.",
    "param_definitions": {
        "F": "Forward LIBOR/SOFR rate for the accrual period",
        "K": "Cap/floor strike rate",
        "T": "Time to caplet/floorlet expiry in years",
        "r": "Risk-free discount rate",
        "σ": "Black (log-normal) volatility of the forward rate",
        "τ (tau)": "Accrual period day-count fraction (e.g., 0.25 for 3-month/quarterly, 0.5 for semi-annual, 1.0 for annual)",
        "N (notional)": "Notional principal — the dollar payoff is N·τ·max(F_T − K, 0)",
        "N·τ": "Cash flow multiplier — converts the rate difference to a dollar amount",
        "d₁": "d₁ = [ln(F/K) + σ²T/2] / (σ√T)",
        "d₂": "d₂ = d₁ − σ√T",
        "e^{−rT}": "Discount factor from payoff date T+τ approximately to today",
    },
}

PRODUCTS["swaption"] = {
    "name": "Swaption (Black-76)",
    "category": "interest_rate",
    "desc": "European option to enter an interest rate swap — payer or receiver.",
    "module": "ch11_interest_rate.01_interest_rate_options",
    "func": "swaption_black76",
    "params": [
        P("F_swap", "Forward Swap Rate", 0.05, -0.10, 1.0, 0.001, "Par forward swap rate"),
        P("K", "Strike Rate (K)", 0.05, -0.10, 1.0, 0.001, "Strike swap rate"),
        P("T_expiry", "Option Expiry (T)", 1.0, 0.01, 30.0, 0.01, "Time to swaption expiry"),
        P("r", "Discount Rate (r)", 0.04, -0.10, 1.0, 0.001),
        P("sigma", "Swap Rate Vol (σ)", 0.20, 0.001, 5.0, 0.01, "Black vol of forward swap rate"),
        P("annuity", "Annuity Factor", 4.5, 0.01, 100.0, 0.01, "PV01 of the underlying swap (DV01)"),
    ],
    "has_option_type": False,
    "extra_choices": {
        "payer_receiver": {"label": "Type", "options": ["payer", "receiver"], "default": "payer"},
    },
    "formula": r"V = A \cdot \left[F\,N(\phi d_1) - K\,N(\phi d_2)\right]",
    "formula_detail": r"""A = \text{annuity factor}, \quad \phi = +1\text{ (payer)}, -1\text{ (receiver)}""",
    "methodology": """A swaption gives the right to enter a swap at a fixed rate K. Payer swaption = right to pay
fixed (call on swap rate). Priced via Black-76 applied to the forward swap rate, weighted by
the annuity factor (PV of the swap's fixed leg payments).""",
    "assumptions": [
        "Forward swap rate is log-normally distributed",
        "Annuity factor is deterministic (frozen at time zero)",
        "European exercise",
    ],
    "limitations": [
        "Frozen annuity assumption introduces errors for long-dated swaptions",
        "Log-normal assumption inadequate for negative rates",
        "Does not capture the swaption smile — use SABR or shifted-SABR",
    ],
    "recommendations": "The standard swaption pricing model. Annuity factor can be computed from the swap's fixed leg schedule and discount factors.",
    "param_definitions": {
        "F_swap": "Current par forward swap rate — the fixed rate that makes the underlying swap fair today",
        "K": "Strike swap rate — the fixed rate the swaption holder will pay/receive if exercised",
        "T": "Time to swaption expiry in years",
        "r": "Risk-free rate used for discounting",
        "σ": "Black vol of the forward swap rate",
        "A (annuity)": "Annuity factor = Σ τ_i · P(0, t_i) = PV of 1 unit of the fixed leg per period; also called DV01/PV01. Scales the swaption payoff to dollars",
        "φ": "Payer/receiver sign: φ = +1 for payer swaption (right to pay fixed), φ = −1 for receiver (right to receive fixed)",
        "d₁": "d₁ = [ln(F/K) + σ²T/2] / (σ√T)",
        "d₂": "d₂ = d₁ − σ√T",
        "N(φd₁), N(φd₂)": "Standard normal CDF applied with payer/receiver sign",
    },
}

PRODUCTS["vasicek_bond_option"] = {
    "name": "Vasicek Bond Option",
    "category": "interest_rate",
    "desc": "European option on a zero-coupon bond under the Vasicek short-rate model.",
    "module": "ch11_interest_rate.01_interest_rate_options",
    "func": "vasicek_bond_option",
    "params": [
        P("r0", "Current Short Rate (r₀)", 0.05, -0.10, 1.0, 0.001, "Current instantaneous short rate"),
        P("K", "Strike Price (K)", 0.95, 0.01, 2.0, 0.001, "Bond strike price"),
        P("T_option", "Option Expiry", 1.0, 0.01, 30.0, 0.01, "Time to option expiry"),
        P("T_bond", "Bond Maturity", 5.0, 0.01, 50.0, 0.1, "Maturity of the underlying bond"),
        P("kappa", "Mean Reversion (κ)", 0.3, 0.01, 10.0, 0.01, "Speed of mean reversion"),
        P("theta", "Long-run Rate (θ)", 0.05, -0.10, 1.0, 0.001, "Long-run equilibrium rate"),
        P("sigma", "Rate Vol (σ)", 0.01, 0.001, 0.5, 0.001, "Volatility of short rate"),
    ],
    "has_option_type": True,
    "formula": r"V = P(0,T_B) N(h) - K \cdot P(0,T_O) N(h - \sigma_P)",
    "formula_detail": r"""\sigma_P = \frac{\sigma}{\kappa}(1-e^{-\kappa(T_B-T_O)})\sqrt{\frac{1-e^{-2\kappa T_O}}{2\kappa}}""",
    "methodology": """Under Vasicek (1977), the short rate follows an Ornstein-Uhlenbeck process:
dr = κ(θ − r)dt + σ dW. Bond prices are exponential-affine in r, and bond options have
a Black-Scholes-like closed form with bond price volatility σ_P determined by model parameters.""",
    "assumptions": [
        "Short rate follows mean-reverting Gaussian process",
        "Bond prices are log-normally distributed given the short rate",
        "Parameters (κ, θ, σ) are constant",
    ],
    "limitations": [
        "Vasicek allows negative interest rates (may or may not be desirable)",
        "Single-factor model — cannot capture yield curve twists",
        "Calibration to the current yield curve requires time-dependent θ (→ Hull-White)",
    ],
    "recommendations": "Good for understanding rate dynamics and bond option pricing. For production use with term structure fitting, upgrade to Hull-White.",
    "param_definitions": {
        "r₀": "Current instantaneous short rate",
        "K": "Strike price of the bond option (as a fraction of par, e.g., 0.95 = 95 cents per dollar)",
        "T_option": "Time to option expiry in years",
        "T_bond": "Maturity of the underlying zero-coupon bond in years (T_bond > T_option)",
        "κ": "Speed of mean reversion of the short rate; higher κ = faster pull back to θ",
        "θ": "Long-run equilibrium short rate level",
        "σ": "Volatility of the short rate (instantaneous std dev of dr)",
        "P(0, T)": "Zero-coupon bond price today maturing at T: P(0,T) = A(T)·e^{−B(T)·r₀} under Vasicek (exponential-affine)",
        "B(T)": "Duration factor: B(T) = (1 − e^{−κT}) / κ",
        "A(T)": "Level factor in bond pricing formula: A(T) = exp{(B(T)−T)(κ²θ − σ²/2)/κ² − σ²B(T)²/(4κ)}",
        "σ_P": "Bond price volatility (vol of P(T_O, T_B)): σ_P = (σ/κ)(1−e^{−κ(T_B−T_O)})·√[(1−e^{−2κT_O})/(2κ)]",
        "h": "Moneyness: h = ln[P(0,T_B) / (K·P(0,T_O))] / σ_P + σ_P/2",
    },
}

# ========================= DISCRETE DIVIDENDS =========================

PRODUCTS["escrowed_div"] = {
    "name": "Escrowed Dividend Method",
    "category": "discrete_div",
    "desc": "European option on a stock paying known discrete dividends — PV-adjusted spot price.",
    "module": "ch09_discrete_dividends.01_discrete_dividends",
    "func": "escrowed_dividend_option",
    "params": [
        _S, _K, _T, _r, _sig,
    ],
    "has_option_type": True,
    "extra_params": {
        "dividends": {
            "label": "Dividends [(time, amount), ...]",
            "type": "dividend_list",
            "default": [(0.25, 2.0), (0.75, 2.0)],
            "help": "List of (time, cash amount) dividend pairs",
        },
    },
    "formula": r"V = \text{BSM}(S - PV(D), K, T, r, \sigma)",
    "formula_detail": r"""PV(D) = \sum_i D_i \, e^{-r t_i}, \quad S^* = S - PV(D)""",
    "methodology": """The escrowed dividend method subtracts the present value of all known dividends from the
spot price, then applies BSM to the adjusted spot S*. Simple but effective for European options
when dividend dates and amounts are known.""",
    "assumptions": [
        "Known discrete dividend amounts and dates",
        "Dividends are certain (no dividend risk)",
        "European exercise (for American, the dividend may trigger early exercise)",
    ],
    "limitations": [
        "S* can become negative for high dividends — problematic for BSM",
        "Volatility of S* differs from volatility of S (term structure effect)",
        "Not suitable for American options — use Roll-Geske-Whaley or binomial tree",
    ],
    "recommendations": "Simplest approach for European options with known dividends. For American options, use the discrete dividend tree or Roll-Geske-Whaley.",
    "param_definitions": {
        "S": "Current cum-dividend spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "σ": "Volatility of the ex-dividend spot S*",
        "D_i": "i-th discrete cash dividend amount",
        "t_i": "Payment time of the i-th dividend in years",
        "PV(D)": "Present value of all dividends paid before expiry: PV(D) = Σ D_i · e^{−r·t_i}",
        "S*": "Dividend-adjusted (ex-dividend) spot: S* = S − PV(D) — the component of S that follows GBM",
        "BSM(S*, K, T, r, σ)": "Standard Black-Scholes applied to the adjusted spot S* — same formula as vanilla with spot replaced by S*",
    },
}

PRODUCTS["discrete_div_tree"] = {
    "name": "Discrete Dividend Tree",
    "category": "discrete_div",
    "desc": "Binomial tree handling known discrete dividend drops at specified dates.",
    "module": "ch09_discrete_dividends.01_discrete_dividends",
    "func": "discrete_dividend_tree",
    "params": [
        _S, _K, _T, _r, _sig,
        P("N", "Tree Steps (N)", 100, 10, 2000, 10, "Number of time steps"),
    ],
    "has_option_type": True,
    "extra_choices": {
        "exercise": {"label": "Exercise Style", "options": ["european", "american"], "default": "american"},
    },
    "extra_params": {
        "dividends": {
            "label": "Dividends [(time, amount), ...]",
            "type": "dividend_list",
            "default": [(0.25, 2.0), (0.75, 2.0)],
            "help": "List of (time, cash amount) dividend pairs",
        },
    },
    "formula": r"\text{At dividend node: } S_{ex} = S_{cum} - D_i",
    "formula_detail": r"""\text{Standard binomial backward induction with price drops at dividend dates.}""",
    "methodology": """The tree is constructed with dividend-adjusted prices. At each dividend date, the stock
price drops by the dividend amount. Between dividends, standard CRR up/down moves apply.
For American options, early exercise is checked at each node (especially just before dividends).""",
    "assumptions": [
        "Known dividend amounts and dates",
        "Standard CRR tree between dividend dates",
        "Dividends cause discrete price drops",
    ],
    "limitations": [
        "Non-recombining at dividend nodes (requires interpolation or approximation)",
        "Computational cost increases with more dividend dates",
    ],
    "recommendations": "The most accurate method for American options on individual stocks with known dividends. Standard in equity derivatives desks.",
    "param_definitions": {
        "S": "Current cum-dividend spot price",
        "K": "Strike price",
        "T": "Time to expiration in years",
        "r": "Risk-free rate",
        "σ": "Annualized volatility between dividend dates",
        "N": "Number of binomial time steps in the tree",
        "D_i": "i-th discrete cash dividend amount",
        "t_i": "Ex-dividend date of the i-th dividend in years",
        "S_cum": "Stock price just before a dividend — the cum-dividend price at node j",
        "S_ex": "Stock price immediately after dividend payment: S_ex = S_cum − D_i",
        "CRR step": "Between dividend dates, standard CRR up/down moves apply: u = e^{σ√Δt}, d = 1/u",
        "American exercise": "At each node, V = max(intrinsic value, continuation value) — especially important just before ex-dividend dates where early exercise may be optimal",
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_products_by_category():
    """Return dict of category_id -> list of (product_id, product_dict)."""
    result = OrderedDict()
    for cat_id, cat_name in CATEGORIES:
        items = [(pid, p) for pid, p in PRODUCTS.items() if p["category"] == cat_id]
        if items:
            result[cat_id] = items
    return result


def get_category_name(cat_id):
    """Return display name for a category id."""
    for cid, cname in CATEGORIES:
        if cid == cat_id:
            return cname
    return cat_id
