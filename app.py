"""
Derivatives Pricing Engine
===========================
Streamlit-based professional UI for option pricing, analysis, and Greeks.
Built on Haug (2007) "The Complete Guide to Option Pricing Formulas".
"""
import sys
import os
import importlib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from product_registry import (
    PRODUCTS, CATEGORIES, get_products_by_category, get_category_name,
)

# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="Derivatives Pricing Engine by Della 😊",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS
# =====================================================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
        padding-bottom: 0.3rem;
        border-bottom: 3px solid #0066cc;
        margin-bottom: 1rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 400;
        margin: 0;
        opacity: 0.9;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-left: 4px solid #0066cc;
        padding-left: 0.8rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #1a1a2e;
    }
    /* Info boxes */
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.92rem;
    }
    .assumption-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #eee;
    }
    .limitation-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #fee;
        color: #c0392b;
    }
    /* Greek table */
    .greek-positive { color: #27ae60; font-weight: 600; }
    .greek-negative { color: #e74c3c; font-weight: 600; }
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* compact params */
    .stNumberInput > div > div > input { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Helpers
# =====================================================================
@st.cache_resource
def load_func(module_path, func_name):
    """Lazy-import a pricing function."""
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def safe_price(func, kwargs):
    """Call pricing function and handle errors gracefully."""
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, str(e)


def compute_numerical_greeks(func, base_kwargs, option_type_key="option_type",
                              S_key="S", sigma_key="sigma", T_key="T", r_key="r"):
    """Compute Greeks via central finite differences (bump-and-reprice)."""
    greeks = {}
    kw = dict(base_kwargs)

    # Get base price
    base_val, err = safe_price(func, kw)
    if err or base_val is None:
        return greeks, base_val

    # Delta: dV/dS
    if S_key in kw:
        S = kw[S_key]
        dS = max(S * 0.001, 0.01)
        kw_up = dict(kw); kw_up[S_key] = S + dS
        kw_dn = dict(kw); kw_dn[S_key] = S - dS
        v_up, _ = safe_price(func, kw_up)
        v_dn, _ = safe_price(func, kw_dn)
        if v_up is not None and v_dn is not None:
            delta = (v_up - v_dn) / (2 * dS)
            gamma = (v_up - 2 * base_val + v_dn) / (dS ** 2)
            greeks["Delta"] = delta
            greeks["Gamma"] = gamma

    # Vega: dV/dσ (per 1% move)
    if sigma_key in kw:
        sig = kw[sigma_key]
        dsig = 0.001
        kw_up = dict(kw); kw_up[sigma_key] = sig + dsig
        kw_dn = dict(kw); kw_dn[sigma_key] = max(sig - dsig, 0.0001)
        v_up, _ = safe_price(func, kw_up)
        v_dn, _ = safe_price(func, kw_dn)
        if v_up is not None and v_dn is not None:
            greeks["Vega"] = (v_up - v_dn) / (2 * dsig) * 0.01  # per 1% vol

    # Theta: -dV/dT (per day)
    if T_key in kw and kw[T_key] > 1/365:
        T = kw[T_key]
        dT = 1 / 365
        kw_short = dict(kw); kw_short[T_key] = T - dT
        v_short, _ = safe_price(func, kw_short)
        if v_short is not None:
            greeks["Theta"] = -(base_val - v_short) / 1  # per day

    # Rho: dV/dr (per 1% move)
    if r_key in kw:
        r = kw[r_key]
        dr = 0.0001
        kw_up = dict(kw); kw_up[r_key] = r + dr
        kw_dn = dict(kw); kw_dn[r_key] = r - dr
        v_up, _ = safe_price(func, kw_up)
        v_dn, _ = safe_price(func, kw_dn)
        if v_up is not None and v_dn is not None:
            greeks["Rho"] = (v_up - v_dn) / (2 * dr) * 0.01  # per 1% rate

    return greeks, base_val


def build_payoff_chart(product, params, option_type=None, extra_choices=None):
    """Create a payoff-at-expiry diagram."""
    # Determine spot key and strike key
    S_key = "S" if "S" in {p["key"] for p in product["params"]} else "F"
    S_key_alt = "S_f" if "S_f" in {p["key"] for p in product["params"]} else S_key
    if S_key_alt != S_key:
        S_key = S_key_alt

    K_key = "K"
    for p in product["params"]:
        if p["key"] in ("K", "K_f", "K1"):
            K_key = p["key"]
            break

    S_val = params.get(S_key, params.get("S", params.get("F", 100)))
    K_val = params.get(K_key, params.get("K", S_val))

    S_range = np.linspace(S_val * 0.5, S_val * 1.5, 200)
    payoff = np.zeros_like(S_range)

    # Determine if call or put
    is_call = True
    if option_type in ("put", "floor", "receiver"):
        is_call = False
    if extra_choices:
        bt = extra_choices.get("barrier_type", "")
        if "put" in bt:
            is_call = False
        ot = extra_choices.get("outer_type", "")
        it = extra_choices.get("inner_type", "")
        pt = extra_choices.get("payer_receiver", "")
        ot2 = extra_choices.get("option_type", "")
        if ot2 in ("put", "floor", "receiver"):
            is_call = False
        if pt == "receiver":
            is_call = False

    if is_call:
        payoff = np.maximum(S_range - K_val, 0)
    else:
        payoff = np.maximum(K_val - S_range, 0)

    # Try to compute model values across spot range
    model_values = None
    try:
        func = load_func(product["module"], product["func"])
        model_values = []
        for s in S_range:
            kw = dict(params)
            kw[S_key] = float(s)
            if option_type:
                kw["option_type"] = option_type
            if extra_choices:
                kw.update(extra_choices)
            v, e = safe_price(func, kw)
            model_values.append(v if v is not None else np.nan)
        model_values = np.array(model_values)
        if np.all(np.isnan(model_values)):
            model_values = None
    except Exception:
        model_values = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_range, y=payoff, mode="lines",
        name="Payoff at Expiry",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))
    if model_values is not None:
        fig.add_trace(go.Scatter(
            x=S_range, y=model_values, mode="lines",
            name="Present Value",
            line=dict(color="#2980b9", width=2.5),
            fill="tonexty", fillcolor="rgba(41,128,185,0.1)",
        ))
    # Mark current spot and strike with non-overlapping annotations
    # Use separate annotation objects positioned at different y-levels
    # to avoid text collision when Spot ≈ K
    spot_and_strike_close = abs(S_val - K_val) < S_val * 0.08

    fig.add_vline(x=S_val, line_dash="dot", line_color="green")
    fig.add_vline(x=K_val, line_dash="dot", line_color="gray")

    fig.add_annotation(
        x=S_val, y=1.0, yref="paper", showarrow=False,
        text=f"Spot={S_val:.2f}", font=dict(color="green", size=11),
        yanchor="bottom",
        xanchor="left" if spot_and_strike_close else "center",
        xshift=6 if spot_and_strike_close else 0,
    )
    fig.add_annotation(
        x=K_val, y=0.92 if spot_and_strike_close else 1.0,
        yref="paper", showarrow=False,
        text=f"K={K_val:.2f}", font=dict(color="gray", size=11),
        yanchor="bottom",
        xanchor="right" if spot_and_strike_close else "center",
        xshift=-6 if spot_and_strike_close else 0,
    )

    fig.update_layout(
        title="Payoff Diagram",
        xaxis_title="Underlying Price",
        yaxis_title="Option Value",
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_sensitivity_charts(func, base_kwargs, product,
                              option_type=None, extra_choices=None):
    """Create sensitivity analysis charts: value vs each key parameter."""
    # Identify which params to sweep
    sweep_params = []
    for p in product["params"]:
        key = p["key"]
        if key in ("n_paths", "n_steps", "N", "N_S", "N_T", "n", "notional"):
            continue
        if key in base_kwargs:
            sweep_params.append(p)

    if not sweep_params:
        return None

    n_params = len(sweep_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[p["label"] for p in sweep_params],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12",
              "#1abc9c", "#d35400", "#2c3e50", "#c0392b"]

    for idx, p in enumerate(sweep_params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        key = p["key"]
        val = base_kwargs[key]

        # Create sweep range
        lo = max(p["min"], val * 0.3) if val > 0 else p["min"]
        hi = min(p["max"], val * 2.0) if val > 0 else p["max"]
        if lo >= hi:
            lo, hi = p["min"], p["max"]
        sweep = np.linspace(lo, hi, 60)

        values = []
        for sv in sweep:
            kw = dict(base_kwargs)
            kw[key] = float(sv)
            if option_type:
                kw["option_type"] = option_type
            if extra_choices:
                kw.update(extra_choices)
            v, _ = safe_price(func, kw)
            values.append(v if v is not None else np.nan)

        fig.add_trace(
            go.Scatter(
                x=sweep, y=values, mode="lines",
                line=dict(color=colors[idx % len(colors)], width=2),
                showlegend=False,
            ),
            row=row, col=col,
        )
        # Mark current value
        fig.add_trace(
            go.Scatter(
                x=[val], y=[values[np.argmin(np.abs(sweep - val))]],
                mode="markers",
                marker=dict(color="red", size=8, symbol="diamond"),
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=300 * n_rows,
        title_text="Sensitivity Analysis — Option Value vs. Each Parameter",
        template="plotly_white",
        margin=dict(l=50, r=30, t=80, b=40),
    )
    return fig


def build_greek_surface(func, base_kwargs, product, x_key, y_key,
                         option_type=None, extra_choices=None):
    """Build a 3D surface of option value as function of two parameters."""
    x_param = next((p for p in product["params"] if p["key"] == x_key), None)
    y_param = next((p for p in product["params"] if p["key"] == y_key), None)
    if not x_param or not y_param:
        return None

    x_val = base_kwargs[x_key]
    y_val = base_kwargs[y_key]
    x_lo = max(x_param["min"], x_val * 0.5) if x_val > 0 else x_param["min"]
    x_hi = min(x_param["max"], x_val * 1.5) if x_val > 0 else x_param["max"]
    y_lo = max(y_param["min"], y_val * 0.5) if y_val > 0 else y_param["min"]
    y_hi = min(y_param["max"], y_val * 1.5) if y_val > 0 else y_param["max"]

    xs = np.linspace(x_lo, x_hi, 30)
    ys = np.linspace(y_lo, y_hi, 30)
    Z = np.full((len(ys), len(xs)), np.nan)

    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            kw = dict(base_kwargs)
            kw[x_key] = float(xv)
            kw[y_key] = float(yv)
            if option_type:
                kw["option_type"] = option_type
            if extra_choices:
                kw.update(extra_choices)
            v, _ = safe_price(func, kw)
            Z[i, j] = v if v is not None else np.nan

    fig = go.Figure(data=[go.Surface(
        x=xs, y=ys, z=Z,
        colorscale="Viridis",
        colorbar=dict(title="Value"),
    )])
    fig.update_layout(
        title=f"Option Value Surface: {x_param['label']} vs {y_param['label']}",
        scene=dict(
            xaxis_title=x_param["label"],
            yaxis_title=y_param["label"],
            zaxis_title="Option Value",
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# =====================================================================
# Sidebar: product selection
# =====================================================================
st.sidebar.markdown("## 📈 Derivatives Pricing Engine by Della 😊")
st.sidebar.markdown("---")

products_by_cat = get_products_by_category()

# Build flat list of product ids with category headers
product_options = []
product_labels = {}
for cat_id, items in products_by_cat.items():
    for pid, pdata in items:
        product_options.append(pid)
        product_labels[pid] = pdata["name"]

# Category filter
cat_ids = [c[0] for c in CATEGORIES if c[0] in products_by_cat]
cat_names = [get_category_name(c) for c in cat_ids]
selected_cat = st.sidebar.selectbox(
    "Category",
    cat_ids,
    format_func=lambda x: get_category_name(x),
    index=0,
)

# Product list for selected category
cat_products = products_by_cat.get(selected_cat, [])
if cat_products:
    selected_pid = st.sidebar.radio(
        "Select Product",
        [pid for pid, _ in cat_products],
        format_func=lambda x: PRODUCTS[x]["name"],
        index=0,
    )
else:
    selected_pid = None

st.sidebar.markdown("---")

# Methodology Handbook download
_handbook_path = os.path.join(ROOT, "期权定价公式完全指南_产品汇总分析.docx")
if os.path.isfile(_handbook_path):
    st.sidebar.markdown("### 📘 Methodology Handbook")
    with open(_handbook_path, "rb") as _hf:
        st.sidebar.download_button(
            label="Download Handbook",
            data=_hf,
            file_name="Derivatives_Pricing_Handbook.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    st.sidebar.markdown("---")

st.sidebar.markdown(
    "<small>Based on Haug (2007)<br/>"
    "<i>The Complete Guide to Option Pricing Formulas</i></small>",
    unsafe_allow_html=True,
)


# =====================================================================
# Main content
# =====================================================================
if selected_pid is None:
    st.info("Select a product from the sidebar.")
    st.stop()

product = PRODUCTS[selected_pid]

# Header
st.markdown(f'<div class="main-header">{product["name"]}</div>', unsafe_allow_html=True)
st.markdown(f"*{product['desc']}*")

# Tabs
tab_pricing, tab_analysis, tab_greeks = st.tabs([
    "💰 Pricing Calculator",
    "📖 Product Analysis",
    "📊 Greeks & Sensitivity",
])


# =====================================================================
# TAB 1: Pricing Calculator
# =====================================================================
with tab_pricing:
    col_params, col_results = st.columns([2, 3])

    with col_params:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)

        param_values = {}
        # Option type
        option_type = None
        if product.get("has_option_type"):
            choices = product.get("option_type_choices", ["call", "put"])
            option_type = st.radio("Option Type", choices, horizontal=True, key="ot_main")

        # Extra choices (barrier type, exercise style, etc.)
        extra_choice_values = {}
        if "extra_choices" in product:
            for ck, cv in product["extra_choices"].items():
                extra_choice_values[ck] = st.selectbox(
                    cv["label"], cv["options"],
                    index=cv["options"].index(cv["default"]) if cv["default"] in cv["options"] else 0,
                    key=f"ec_{ck}",
                )

        # Numeric parameters
        n_param_cols = 2
        param_list = product["params"]
        for i in range(0, len(param_list), n_param_cols):
            cols = st.columns(n_param_cols)
            for j, col in enumerate(cols):
                if i + j < len(param_list):
                    p = param_list[i + j]
                    with col:
                        val = st.number_input(
                            p["label"],
                            min_value=float(p["min"]),
                            max_value=float(p["max"]),
                            value=float(p["default"]),
                            step=float(p["step"]),
                            help=p.get("help", ""),
                            key=f"param_{p['key']}",
                            format="%.2f" if p["key"] == "notional" else "%.4f",
                        )
                        param_values[p["key"]] = val

        # Extra params (dividends, etc.)
        extra_param_values = {}
        if "extra_params" in product:
            for ek, ev in product["extra_params"].items():
                if ev.get("type") == "dividend_list":
                    st.markdown(f"**{ev['label']}**")
                    div_text = st.text_area(
                        "Enter as: time,amount (one per line)",
                        value="\n".join(f"{t},{d}" for t, d in ev["default"]),
                        height=80,
                        key=f"ep_{ek}",
                    )
                    divs = []
                    for line in div_text.strip().split("\n"):
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            try:
                                divs.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                pass
                    extra_param_values[ek] = divs

        # --- Run Valuation button ---
        st.markdown("")
        run_clicked = st.button(
            "▶  Run Valuation",
            key="run_valuation",
            type="primary",
            use_container_width=True,
        )

        # Persist the last-run results in session state so they survive re-renders
        result_key = f"_result_{selected_pid}"
        if run_clicked:
            st.session_state[result_key] = {
                "param_values": dict(param_values),
                "option_type": option_type,
                "extra_choice_values": dict(extra_choice_values),
                "extra_param_values": dict(extra_param_values),
            }

    with col_results:
        st.markdown('<div class="section-header">Pricing Results</div>', unsafe_allow_html=True)

        # Check if we have a run to display
        if result_key not in st.session_state:
            st.info("Set parameters on the left and click **Run Valuation**.")
            st.stop()

        # Retrieve the snapshotted inputs from the last run
        _snap = st.session_state[result_key]
        _snap_params = _snap["param_values"]
        _snap_ot = _snap["option_type"]
        _snap_ec = _snap["extra_choice_values"]
        _snap_ep = _snap["extra_param_values"]

        # Build kwargs
        func = load_func(product["module"], product["func"])
        base_kwargs = dict(_snap_params)
        if _snap_ot:
            base_kwargs["option_type"] = _snap_ot
        base_kwargs.update(_snap_ec)
        base_kwargs.update(_snap_ep)

        # Integer params
        int_keys = {"N", "N_S", "N_T", "n", "n_paths", "n_steps", "n_terms"}
        for k in int_keys:
            if k in base_kwargs:
                base_kwargs[k] = int(base_kwargs[k])

        # Compute price
        result, error = safe_price(func, base_kwargs)

        if error:
            st.error(f"Pricing error: {error}")
        else:
            # Display results
            is_dict = isinstance(result, dict)

            if is_dict:
                price = result.get("price", result.get("value", 0))
                std_err = result.get("std_error", None)
            else:
                price = result
                std_err = None

            # Single option value card
            label = "Call Value" if _snap_ot == "call" else ("Put Value" if _snap_ot == "put" else "Option Value")
            css_class = "metric-card-green" if _snap_ot == "call" else ("metric-card-red" if _snap_ot == "put" else "metric-card")
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <h3>{label}</h3>
                <div class="value">{price:.2f}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("")

            # MC-specific info (compact)
            if is_dict:
                mc_items = [(k, v) for k, v in result.items() if k not in ("price", "value")]
                if mc_items:
                    mc_parts = []
                    for k, v in mc_items:
                        name = k.replace("_", " ").title()
                        if isinstance(v, float):
                            mc_parts.append(f"**{name}:** {v:.2f}")
                        else:
                            mc_parts.append(f"**{name}:** {v:,}" if isinstance(v, int) else f"**{name}:** {v}")
                    st.markdown(
                        '<div style="font-size:0.82rem; color:#555; margin-top:-0.3rem">'
                        + " &nbsp;|&nbsp; ".join(mc_parts)
                        + '</div>',
                        unsafe_allow_html=True,
                    )

            # Payoff chart
            payoff_fig = build_payoff_chart(product, _snap_params, _snap_ot, _snap_ec)
            st.plotly_chart(payoff_fig, use_container_width=True)


# =====================================================================
# TAB 2: Product Analysis
# =====================================================================
with tab_analysis:
    # Methodology
    st.markdown('<div class="section-header">Valuation Methodology</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">{product["methodology"]}</div>', unsafe_allow_html=True)

    # Key Assumptions & Limitations
    st.markdown('<div class="section-header">Key Assumptions & Limitations</div>', unsafe_allow_html=True)
    a_col, l_col = st.columns(2)
    with a_col:
        st.markdown("**Assumptions**")
        for i, a in enumerate(product.get("assumptions", []), 1):
            st.markdown(f'<div class="assumption-item"><b>{i}.</b> {a}</div>', unsafe_allow_html=True)
    with l_col:
        st.markdown("**Limitations**")
        for i, l in enumerate(product.get("limitations", []), 1):
            st.markdown(f'<div class="limitation-item"><b>{i}.</b> {l}</div>', unsafe_allow_html=True)

    # Pricing Formula
    st.markdown('<div class="section-header">Pricing Formula</div>', unsafe_allow_html=True)
    st.latex(product["formula"])
    if product.get("formula_detail"):
        st.latex(product["formula_detail"])
    # Extended formula blocks (e.g. barrier A/B/C/D sub-formulas, 8-type table)
    for _extra_latex in product.get("formula_extra", []):
        st.latex(_extra_latex)

    # Parameter Definitions
    st.markdown('<div class="section-header">Parameter Definitions</div>', unsafe_allow_html=True)
    if product.get("param_definitions"):
        for sym, defn in product["param_definitions"].items():
            st.markdown(f"- **{sym}** — {defn}")
    else:
        # Auto-generate from params list + common definitions
        _common_defs = {
            "S": "Current spot price of the underlying asset",
            "K": "Strike (exercise) price of the option",
            "T": "Time to expiration in years",
            "r": "Annualized risk-free interest rate (continuously compounded)",
            "b": "Cost-of-carry rate: b = r (non-dividend stock), b = r − q (continuous div), b = r − r_f (FX), b = 0 (futures)",
            "sigma": "σ — Annualized volatility of the underlying returns",
            "q": "Continuous dividend yield",
            "r_d": "Domestic risk-free interest rate",
            "r_f": "Foreign risk-free interest rate",
            "F": "Forward / futures price",
            "H": "Barrier level — the price at which the option is knocked in or out",
            "L": "Lower barrier level",
            "U": "Upper barrier level",
            "rebate": "Cash payment received if the option is knocked out before expiry",
            "N(·)": "Standard normal cumulative distribution function (CDF)",
            "d_1": "d₁ = [ln(S/K) + (b + σ²/2)T] / (σ√T) — measures how far the option is in-the-money, adjusted for drift",
            "d_2": "d₂ = d₁ − σ√T — strike-adjusted moneyness under the risk-neutral measure",
            "n_paths": "Number of Monte Carlo simulation paths",
            "n_steps": "Number of time steps per simulation path",
            "S_min": "Minimum observed spot price (for lookback options)",
            "S_max": "Maximum observed spot price (for lookback options)",
            "K1": "Strike price of the first option (in spread / two-asset strategies)",
            "K2": "Strike price of the second option",
            "T1": "Time to expiry of the first option (in compound / chooser structures)",
            "T2": "Time to expiry of the second option",
            "rho": "ρ — Correlation between two underlying assets",
            "eta": "η — +1 for call-type, −1 for put-type (sign indicator)",
            "phi": "φ — +1 for down barriers, −1 for up barriers (direction indicator)",
            "alpha": "α — Strike-setting ratio (for forward-start options: K = α × S at start date)",
            "n_terms": "Number of terms in series expansion",
            "notional": "Notional / face value of the contract",
            "sigma1": "σ₁ — Volatility of the first underlying asset",
            "sigma2": "σ₂ — Volatility of the second underlying asset",
            "S1": "Spot price of the first underlying asset",
            "S2": "Spot price of the second underlying asset",
        }
        shown = False
        for p in product["params"]:
            key = p["key"]
            defn = _common_defs.get(key, p.get("help", ""))
            if defn:
                shown = True
                st.markdown(f"- **{key}** — {defn}")
        # Always show d1/d2 for BSM-family formulas
        if any(x in product.get("formula", "") for x in ["d_1", "d_2"]):
            if "d_1" in _common_defs:
                st.markdown(f"- **d₁** — {_common_defs['d_1']}")
                st.markdown(f"- **d₂** — {_common_defs['d_2']}")
            if "N(" in product.get("formula", ""):
                st.markdown(f"- **N(·)** — {_common_defs['N(·)']}")
        if not shown:
            st.markdown("*See the formula above for parameter descriptions.*")

    # Recommendations
    if product.get("recommendations"):
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        st.info(product["recommendations"])


# =====================================================================
# TAB 3: Greeks & Sensitivity
# =====================================================================
with tab_greeks:
    # Use snapshotted values if available
    _g_snap = st.session_state.get(result_key)
    if not _g_snap:
        st.info("Run a valuation first to see Greeks and sensitivity analysis.")
        st.stop()

    _g_params = _g_snap["param_values"]
    _g_ot = _g_snap["option_type"]
    _g_ec = _g_snap["extra_choice_values"]
    _g_ep = _g_snap["extra_param_values"]

    # Detect if this is a Monte Carlo product (returns dict with stochastic results)
    is_mc_product = product.get("returns_dict", False)

    st.markdown('<div class="section-header">Greeks (Numerical)</div>', unsafe_allow_html=True)

    # Prepare kwargs for greek computation
    int_keys = {"N", "N_S", "N_T", "n", "n_paths", "n_steps", "n_terms"}
    greek_func_kwargs = dict(_g_params)
    if _g_ot:
        greek_func_kwargs["option_type"] = _g_ot
    greek_func_kwargs.update(_g_ec)
    greek_func_kwargs.update(_g_ep)
    for k in int_keys:
        if k in greek_func_kwargs:
            greek_func_kwargs[k] = int(greek_func_kwargs[k])

    # Detect key names
    S_key = "S"
    for k in ("S", "F", "S_f", "F1", "S1"):
        if k in _g_params:
            S_key = k
            break
    sigma_key = "sigma"
    for k in ("sigma", "sigma_S", "sigma1", "sigma_swap", "sigma_abs"):
        if k in _g_params:
            sigma_key = k
            break
    T_key = "T"
    for k in ("T", "T_expiry"):
        if k in _g_params:
            T_key = k
            break
    r_key = "r"
    for k in ("r", "r_d"):
        if k in _g_params:
            r_key = k
            break

    func = load_func(product["module"], product["func"])

    if is_mc_product:
        # Monte Carlo products return stochastic results — bump-and-reprice Greeks
        # are too noisy and slow.  Show an info message instead.
        st.info(
            "Greeks via bump-and-reprice are not shown for Monte Carlo products "
            "because the stochastic noise makes finite-difference estimates unreliable. "
            "Use an analytical model (e.g., Generalized BSM) for Greeks, or increase "
            "path count substantially for rough estimates."
        )
        greeks = {}
        base_val = None
    else:
        # Compute Greeks using numerical differentiation
        def greek_pricer(**kw):
            all_kw = dict(kw)
            if _g_ot:
                all_kw["option_type"] = _g_ot
            all_kw.update(_g_ec)
            all_kw.update(_g_ep)
            for k in int_keys:
                if k in all_kw:
                    all_kw[k] = int(all_kw[k])
            return func(**all_kw)

        greeks, base_val = compute_numerical_greeks(
            greek_pricer, _g_params,
            S_key=S_key, sigma_key=sigma_key, T_key=T_key, r_key=r_key,
        )

    if greeks:
        gcols = st.columns(len(greeks))
        greek_colors = {
            "Delta": ("metric-card-green", "Δ"),
            "Gamma": ("metric-card-blue", "Γ"),
            "Vega": ("metric-card", "ν"),
            "Theta": ("metric-card-red", "Θ"),
            "Rho": ("metric-card-blue", "ρ"),
        }
        for i, (name, value) in enumerate(greeks.items()):
            css, symbol = greek_colors.get(name, ("metric-card", name[0]))
            with gcols[i]:
                st.markdown(f"""
                <div class="metric-card {css}">
                    <h3>{name} ({symbol})</h3>
                    <div class="value">{value:+.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Greek descriptions
        with st.expander("Greek Definitions", expanded=False):
            greek_defs = {
                "Delta": "Rate of change of option value with respect to underlying price. Δ ≈ probability of expiring ITM (for calls).",
                "Gamma": "Rate of change of Delta with respect to underlying price. High Gamma near ATM and expiry.",
                "Vega": "Sensitivity to a 1% change in implied volatility. Highest for ATM, long-dated options.",
                "Theta": "Daily time decay — value lost per calendar day. Accelerates near expiry for ATM options.",
                "Rho": "Sensitivity to a 1% change in the risk-free rate. More important for long-dated options.",
            }
            for name, defn in greek_defs.items():
                if name in greeks:
                    st.markdown(f"**{name}**: {defn}")
    else:
        st.warning("Could not compute Greeks for this product configuration.")

    # Sensitivity Analysis
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)

    if is_mc_product:
        st.info("Sensitivity analysis is not available for Monte Carlo products due to stochastic noise.")
    else:
        sens_fig = build_sensitivity_charts(
            func, greek_func_kwargs, product,
            option_type=None,  # already in greek_func_kwargs
            extra_choices=None,
        )
        if sens_fig:
            st.plotly_chart(sens_fig, use_container_width=True)

        # 3D Surface
        st.markdown('<div class="section-header">3D Value Surface</div>', unsafe_allow_html=True)

        # Pick two params for 3D surface
        surf_params = [p for p in product["params"]
                       if p["key"] not in ("n_paths", "n_steps", "N", "N_S", "N_T", "n", "notional")]
        if len(surf_params) >= 2:
            s3d_c1, s3d_c2 = st.columns(2)
            with s3d_c1:
                x_key = st.selectbox(
                    "X-axis parameter",
                    [p["key"] for p in surf_params],
                    format_func=lambda k: next(p["label"] for p in surf_params if p["key"] == k),
                    index=0,
                    key="surf_x",
                )
            with s3d_c2:
                y_options = [p["key"] for p in surf_params if p["key"] != x_key]
                y_key = st.selectbox(
                    "Y-axis parameter",
                    y_options,
                    format_func=lambda k: next(p["label"] for p in surf_params if p["key"] == k),
                    index=min(1, len(y_options) - 1) if y_options else 0,
                    key="surf_y",
                )

            if st.button("Generate 3D Surface", key="gen_surf"):
                with st.spinner("Computing surface (this may take a moment)..."):
                    surf_fig = build_greek_surface(
                        func, greek_func_kwargs, product,
                        x_key, y_key,
                        option_type=None,
                        extra_choices=None,
                    )
                    if surf_fig:
                        st.plotly_chart(surf_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate surface for selected parameters.")
        else:
            st.info("Not enough parameters for a 3D surface plot.")
