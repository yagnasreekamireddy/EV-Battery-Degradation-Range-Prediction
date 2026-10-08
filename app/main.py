import sys
import os
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Degradation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2e3a50;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #8899bb !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e0e8ff !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.85rem !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #0e1117);
        border-left: 4px solid #3b82f6;
        padding: 10px 18px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #cce0ff;
        letter-spacing: 0.02em;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #1a2740, #111827);
        border: 1px solid #2563eb44;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #b0c4de;
        font-size: 0.9rem;
    }
    .insight-box.warn {
        border-color: #f59e0b55;
        background: linear-gradient(135deg, #2d1f00, #1a1200);
    }
    .insight-box.good {
        border-color: #10b98155;
        background: linear-gradient(135deg, #001f14, #00110c);
    }
    .insight-box.danger {
        border-color: #ef444455;
        background: linear-gradient(135deg, #2d0a0a, #1a0606);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #111827 100%);
        border-right: 1px solid #1e2d45;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #93b4d4 !important;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6b7a99;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #60a5fa !important;
    }

    /* Gauge/KPI banner */
    .kpi-banner {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    /* Hero title */
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .hero-sub {
        color: #5a7299;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    /* Divider */
    hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS  (robust to running from any cwd)
# ─────────────────────────────────────────────
# app/main.py lives inside EV_PROJECT/app/
# data CSV  → EV_PROJECT/data/
# model pkl → EV_PROJECT/model/
APP_DIR      = os.path.dirname(os.path.abspath(__file__))   # …/EV_PROJECT/app
PROJECT_ROOT = os.path.dirname(APP_DIR)                     # …/EV_PROJECT
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ─────────────────────────────────────────────
# LOAD MODEL & DATA (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model.pkl"),    "rb") as f:
        model    = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),   "rb") as f:
        scaler   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_make.pkl"),  "rb") as f:
        le_make  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_drive.pkl"), "rb") as f:
        le_drive = pickle.load(f)
    return model, scaler, le_make, le_drive

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "electric_vehicle_analytics.csv")
    return pd.read_csv(path)

try:
    model, scaler, le_make, le_drive = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERR = str(e)

df_full = load_data()

# ─────────────────────────────────────────────
# PREDICT FUNCTION (inline fallback if no src/)
# ─────────────────────────────────────────────
def predict_ev(battery_capacity, battery_health, mileage, temperature, speed, cycles):
    if not MODEL_LOADED:
        # Heuristic fallback for demo
        base = battery_capacity * (battery_health / 100) * 5.5
        temp_factor = 1 - max(0, (abs(temperature - 22) - 5)) * 0.012
        speed_factor = 1 - max(0, (speed - 80)) * 0.003
        mileage_factor = 1 - (mileage / 1_000_000)
        return max(50, base * temp_factor * speed_factor * mileage_factor)

    try:
        sys.path.insert(0, PROJECT_ROOT)          # makes `from src.predict` work
        from src.predict import predict_ev as _predict
        return _predict(battery_capacity, battery_health, mileage, temperature, speed, cycles)
    except Exception:
        features = np.array([[battery_capacity, battery_health, mileage,
                               temperature, speed, cycles]])
        scaled = scaler.transform(features)
        return float(model.predict(scaled)[0])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Predictor")
    st.markdown("---")

    st.markdown("### 🔋 Battery Parameters")
    battery_capacity = st.slider("Battery Capacity (kWh)", 20, 150, 75, help="Total energy stored in battery pack")
    battery_health   = st.slider("Battery Health (%)", 50, 100, 90, help="State of Health — 100% is brand new")
    cycles           = st.number_input("Charge Cycles", 0, 5000, 500, step=50)

    st.markdown("### 🚗 Vehicle & Usage")
    mileage     = st.number_input("Mileage (km)", 0, 300_000, 50_000, step=5000)
    speed       = st.slider("Avg Speed (km/h)", 10, 150, 60)
    temperature = st.slider("Temperature (°C)", -20, 55, 25)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Range", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📂 Dataset Filters")
    makes   = ["All"] + sorted(df_full["Make"].unique().tolist())
    regions = ["All"] + sorted(df_full["Region"].unique().tolist())
    vtypes  = ["All"] + sorted(df_full["Vehicle_Type"].unique().tolist())

    sel_make   = st.selectbox("Make",         makes)
    sel_region = st.selectbox("Region",       regions)
    sel_vtype  = st.selectbox("Vehicle Type", vtypes)

# ─────────────────────────────────────────────
# FILTER DATASET
# ─────────────────────────────────────────────
df = df_full.copy()
if sel_make   != "All": df = df[df["Make"]         == sel_make]
if sel_region != "All": df = df[df["Region"]        == sel_region]
if sel_vtype  != "All": df = df[df["Vehicle_Type"]  == sel_vtype]

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">⚡ EV Battery Degradation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Final Year Project · Real-time range prediction + fleet analytics</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Fleet Analytics",
    "🔋 Degradation Analysis",
    "💡 Insights & Costs",
])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if predict_btn:
        result     = predict_ev(battery_capacity, battery_health, mileage, temperature, speed, cycles)
        efficiency = result / battery_capacity
        avg_range  = df_full["Range_km"].mean()
        pct_diff   = ((result - avg_range) / avg_range) * 100

        # ── KPI row ──────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔋 Predicted Range", f"{result:.1f} km",
                  delta=f"{pct_diff:+.1f}% vs fleet avg")
        k2.metric("⚡ Efficiency",       f"{efficiency:.2f} km/kWh",
                  delta="Good" if efficiency > 5 else "Low")
        k3.metric("🏁 Fleet Avg Range",  f"{avg_range:.1f} km")
        k4.metric("🔄 Charge Cycles",    f"{cycles:,}")

        st.markdown("---")
        col_left, col_right = st.columns([1.4, 1])

        # ── Gauge chart ──────────────────────────
        with col_left:
            st.markdown('<div class="section-header">🎯 Predicted Range Gauge</div>', unsafe_allow_html=True)
            max_range = 750
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result,
                delta={"reference": avg_range, "valueformat": ".0f",
                       "increasing": {"color": "#34d399"}, "decreasing": {"color": "#f87171"}},
                number={"suffix": " km", "font": {"size": 42, "color": "#e0e8ff"}},
                gauge={
                    "axis": {"range": [0, max_range], "tickcolor": "#4a5568",
                             "tickfont": {"color": "#6b7a99"}},
                    "bar":  {"color": "#3b82f6", "thickness": 0.25},
                    "bgcolor": "#1a1f2e",
                    "bordercolor": "#2e3a50",
                    "steps": [
                        {"range": [0,   250], "color": "#2d1515"},
                        {"range": [250, 450], "color": "#2d2515"},
                        {"range": [450, 750], "color": "#152d1e"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.8,
                        "value": avg_range,
                    },
                },
                title={"text": "Range (km) · Fleet avg line in amber",
                       "font": {"color": "#6b7a99", "size": 13}},
            ))
            fig_gauge.update_layout(
                height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=40, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Efficiency indicator ──────────────────
        with col_right:
            st.markdown('<div class="section-header">⚡ Efficiency Meter</div>', unsafe_allow_html=True)
            fig_eff = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(efficiency, 2),
                number={"suffix": " km/kWh", "font": {"size": 30, "color": "#e0e8ff"}},
                gauge={
                    "axis": {"range": [0, 10], "tickcolor": "#4a5568",
                             "tickfont": {"color": "#6b7a99"}},
                    "bar": {"color": "#a78bfa"},
                    "bgcolor": "#1a1f2e",
                    "steps": [
                        {"range": [0, 4],  "color": "#2d1515"},
                        {"range": [4, 6],  "color": "#2d2a15"},
                        {"range": [6, 10], "color": "#152d1e"},
                    ],
                },
                title={"text": "Efficiency", "font": {"color": "#6b7a99", "size": 13}},
            ))
            fig_eff.update_layout(
                height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        # ── Input profile radar ──────────────────
        st.markdown('<div class="section-header">📡 Input Profile Radar</div>', unsafe_allow_html=True)
        radar_labels  = ["Battery Cap", "Battery Health", "Speed", "Temperature", "Charge Cycles"]
        radar_max     = [150, 100, 150, 55, 5000]
        radar_vals    = [battery_capacity, battery_health, speed, max(temperature, 0), cycles]
        radar_norm    = [v / m for v, m in zip(radar_vals, radar_max)]
        radar_norm   += radar_norm[:1]
        radar_labels_c = radar_labels + [radar_labels[0]]
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]

        fig_radar, ax_r = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True),
                                       facecolor="#0e1117")
        ax_r.set_facecolor("#1a1f2e")
        ax_r.plot(angles, radar_norm, "o-", linewidth=2, color="#60a5fa")
        ax_r.fill(angles, radar_norm, alpha=0.25, color="#3b82f6")
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(radar_labels, color="#93b4d4", fontsize=9)
        ax_r.set_yticklabels([])
        ax_r.grid(color="#2e3a50", linestyle="--", alpha=0.6)
        ax_r.spines["polar"].set_color("#2e3a50")
        col_r, _ = st.columns([1, 1.5])
        with col_r:
            st.pyplot(fig_radar)

        # ── Smart insights ────────────────────────
        st.markdown('<div class="section-header">🧠 Smart Insights</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        with i1:
            if battery_health < 70:
                st.markdown('<div class="insight-box danger">⚠️ <b>Battery health below 70%</b> — significant range reduction expected. Consider battery replacement.</div>', unsafe_allow_html=True)
            elif battery_health < 85:
                st.markdown('<div class="insight-box warn">🔸 <b>Battery health moderate</b> — monitor degradation trend over the next 10,000 km.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box good">✅ <b>Battery health excellent</b> — optimal range performance.</div>', unsafe_allow_html=True)

            if mileage > 150_000:
                st.markdown('<div class="insight-box danger">⚠️ <b>Very high mileage</b> — accelerated degradation likely. Inspect cell voltage balance.</div>', unsafe_allow_html=True)
            elif mileage > 80_000:
                st.markdown('<div class="insight-box warn">🔸 <b>High mileage</b> — battery degradation expected. Cycles suggest ~{:.0f}% remaining life.</div>'.format(max(0, 100 - cycles / 50)), unsafe_allow_html=True)

        with i2:
            if temperature < 5:
                st.markdown('<div class="insight-box danger">🥶 <b>Very cold temperature</b> — lithium-ion efficiency drops sharply below 5°C. Pre-conditioning recommended.</div>', unsafe_allow_html=True)
            elif temperature > 40:
                st.markdown('<div class="insight-box danger">🔥 <b>High temperature</b> — thermal stress accelerates degradation. Check cooling system.</div>', unsafe_allow_html=True)
            elif temperature < 15 or temperature > 35:
                st.markdown('<div class="insight-box warn">🌡️ <b>Non-optimal temperature</b> — range reduced by ~5–12%.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box good">🌤️ <b>Ideal temperature range</b> — battery operating efficiently.</div>', unsafe_allow_html=True)

            if speed > 110:
                st.markdown('<div class="insight-box warn">🏎️ <b>High average speed</b> — aerodynamic drag significantly reduces range above 100 km/h.</div>', unsafe_allow_html=True)
            elif speed < 30:
                st.markdown('<div class="insight-box good">🐢 <b>Low average speed</b> — efficient driving pattern, maximises range.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box good">✅ <b>Optimal speed range</b> — good efficiency balance.</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Set parameters in the sidebar and click **Predict Range** to see results.")
        # Show a teaser from dataset
        st.markdown('<div class="section-header">📈 Fleet Overview (teaser)</div>', unsafe_allow_html=True)
        fig_t = px.histogram(df, x="Range_km", nbins=40, color_discrete_sequence=["#3b82f6"],
                             labels={"Range_km": "Range (km)"}, title="Distribution of EV Ranges in Dataset")
        fig_t.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                            font_color="#8899bb", title_font_color="#cce0ff", height=320)
        st.plotly_chart(fig_t, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — FLEET ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f"**Showing {len(df):,} vehicles** (filtered) out of {len(df_full):,} total")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Range",       f"{df['Range_km'].mean():.0f} km")
    k2.metric("Avg Battery Health", f"{df['Battery_Health_%'].mean():.1f}%")
    k3.metric("Avg Charge Cycles",  f"{df['Charge_Cycles'].mean():.0f}")
    k4.metric("Avg Mileage",     f"{df['Mileage_km'].mean()/1000:.1f}k km")
    k5.metric("Avg CO₂ Saved",   f"{df['CO2_Saved_tons'].mean():.1f} t")

    st.markdown("---")
    row1_l, row1_r = st.columns(2)

    # Range by make
    with row1_l:
        st.markdown('<div class="section-header">🚗 Avg Range by Make</div>', unsafe_allow_html=True)
        make_avg = df.groupby("Make")["Range_km"].mean().sort_values(ascending=True).reset_index()
        fig1 = px.bar(make_avg, x="Range_km", y="Make", orientation="h",
                      color="Range_km", color_continuous_scale="Blues",
                      labels={"Range_km": "Avg Range (km)", "Make": ""},
                      text_auto=".0f")
        fig1.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", showlegend=False, height=380,
                           coloraxis_showscale=False)
        fig1.update_traces(textfont_color="#e0e8ff")
        st.plotly_chart(fig1, use_container_width=True)

    # Range by vehicle type
    with row1_r:
        st.markdown('<div class="section-header">🏎️ Range by Vehicle Type</div>', unsafe_allow_html=True)
        fig2 = px.box(df, x="Vehicle_Type", y="Range_km",
                      color="Vehicle_Type",
                      color_discrete_sequence=["#3b82f6","#a78bfa","#34d399","#f59e0b"],
                      labels={"Range_km": "Range (km)", "Vehicle_Type": ""})
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    # Region pie
    with row2_l:
        st.markdown('<div class="section-header">🌍 Fleet by Region</div>', unsafe_allow_html=True)
        reg_cnt = df["Region"].value_counts().reset_index()
        fig3 = px.pie(reg_cnt, names="Region", values="count",
                      hole=0.55, color_discrete_sequence=["#3b82f6","#a78bfa","#34d399","#f59e0b"])
        fig3.update_traces(textinfo="percent+label", textfont_color="#e0e8ff")
        fig3.update_layout(paper_bgcolor="#0e1117", font_color="#8899bb",
                           showlegend=False, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # Scatter: mileage vs range
    with row2_r:
        st.markdown('<div class="section-header">📉 Mileage vs Range</div>', unsafe_allow_html=True)
        sample = df.sample(min(600, len(df)), random_state=42)
        fig4 = px.scatter(sample, x="Mileage_km", y="Range_km",
                          color="Battery_Health_%",
                          color_continuous_scale="RdYlGn",
                          hover_data=["Make","Vehicle_Type"],
                          labels={"Mileage_km": "Mileage (km)", "Range_km": "Range (km)",
                                  "Battery_Health_%": "Health %"},
                          opacity=0.7)
        fig4.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", height=320)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — DEGRADATION ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔋 Battery Health vs Charge Cycles</div>', unsafe_allow_html=True)

    sample2 = df.sample(min(800, len(df)), random_state=7)
    fig5 = px.scatter(sample2, x="Charge_Cycles", y="Battery_Health_%",
                      color="Range_km", size="Battery_Capacity_kWh",
                      color_continuous_scale="Turbo",
                      hover_data=["Make", "Mileage_km"],
                      trendline="lowess",
                      labels={"Charge_Cycles": "Charge Cycles",
                              "Battery_Health_%": "Battery Health (%)",
                              "Range_km": "Range (km)"})
    fig5.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                       font_color="#8899bb", height=400)
    st.plotly_chart(fig5, use_container_width=True)

    col_d1, col_d2 = st.columns(2)

    # Health distribution histogram
    with col_d1:
        st.markdown('<div class="section-header">📊 Battery Health Distribution</div>', unsafe_allow_html=True)
        fig6 = px.histogram(df, x="Battery_Health_%", nbins=30,
                            color_discrete_sequence=["#a78bfa"],
                            labels={"Battery_Health_%": "Battery Health (%)"})
        fig6.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", height=300)
        st.plotly_chart(fig6, use_container_width=True)

    # Avg health by make
    with col_d2:
        st.markdown('<div class="section-header">🏭 Avg Battery Health by Make</div>', unsafe_allow_html=True)
        health_make = df.groupby("Make")["Battery_Health_%"].mean().sort_values().reset_index()
        fig7 = px.bar(health_make, x="Make", y="Battery_Health_%",
                      color="Battery_Health_%", color_continuous_scale="RdYlGn",
                      labels={"Battery_Health_%": "Avg Health (%)", "Make": ""},
                      text_auto=".1f")
        fig7.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", showlegend=False,
                           coloraxis_showscale=False, height=300)
        fig7.update_traces(textfont_color="#e0e8ff")
        st.plotly_chart(fig7, use_container_width=True)

    # Energy consumption heatmap by make & vehicle type
    st.markdown('<div class="section-header">🌡️ Energy Consumption Heatmap (Make × Vehicle Type)</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(values="Energy_Consumption_kWh_per_100km",
                           index="Make", columns="Vehicle_Type", aggfunc="mean")
    fig8 = px.imshow(pivot, color_continuous_scale="RdYlGn_r", text_auto=".1f",
                     labels={"color": "kWh/100km"}, aspect="auto")
    fig8.update_layout(paper_bgcolor="#0e1117", font_color="#8899bb", height=350)
    st.plotly_chart(fig8, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — INSIGHTS & COSTS
# ══════════════════════════════════════════════
with tab4:
    col_c1, col_c2 = st.columns(2)

    # CO2 saved by region
    with col_c1:
        st.markdown('<div class="section-header">🌱 CO₂ Saved by Region</div>', unsafe_allow_html=True)
        co2_reg = df.groupby("Region")["CO2_Saved_tons"].mean().reset_index()
        fig9 = px.bar(co2_reg, x="Region", y="CO2_Saved_tons",
                      color="Region", color_discrete_sequence=["#34d399","#3b82f6","#f59e0b","#a78bfa"],
                      labels={"CO2_Saved_tons": "Avg CO₂ Saved (tons)"}, text_auto=".2f")
        fig9.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                           font_color="#8899bb", showlegend=False, height=320)
        fig9.update_traces(textfont_color="#e0e8ff")
        st.plotly_chart(fig9, use_container_width=True)

    # Monthly charging cost by usage type
    with col_c2:
        st.markdown('<div class="section-header">💰 Monthly Charging Cost by Usage Type</div>', unsafe_allow_html=True)
        fig10 = px.violin(df, x="Usage_Type", y="Monthly_Charging_Cost_USD",
                          color="Usage_Type", box=True,
                          color_discrete_sequence=["#3b82f6","#a78bfa","#34d399"],
                          labels={"Monthly_Charging_Cost_USD": "Monthly Cost (USD)", "Usage_Type": ""})
        fig10.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                            font_color="#8899bb", showlegend=False, height=320)
        st.plotly_chart(fig10, use_container_width=True)

    # Resale value vs mileage
    st.markdown('<div class="section-header">💵 Resale Value vs Mileage (coloured by Battery Health)</div>', unsafe_allow_html=True)
    sample3 = df.sample(min(700, len(df)), random_state=21)
    fig11 = px.scatter(sample3, x="Mileage_km", y="Resale_Value_USD",
                       color="Battery_Health_%", size="Battery_Capacity_kWh",
                       color_continuous_scale="RdYlGn",
                       hover_data=["Make","Vehicle_Type"],
                       trendline="ols",
                       labels={"Mileage_km": "Mileage (km)",
                               "Resale_Value_USD": "Resale Value (USD)",
                               "Battery_Health_%": "Health %"})
    fig11.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                        font_color="#8899bb", height=380)
    st.plotly_chart(fig11, use_container_width=True)

    # Cost breakdown table
    st.markdown('<div class="section-header">📋 Cost Comparison by Make</div>', unsafe_allow_html=True)
    cost_tbl = df.groupby("Make").agg(
        Avg_Maintenance   = ("Maintenance_Cost_USD",    "mean"),
        Avg_Insurance     = ("Insurance_Cost_USD",      "mean"),
        Avg_Charging_Mo   = ("Monthly_Charging_Cost_USD","mean"),
        Avg_Resale        = ("Resale_Value_USD",        "mean"),
        Count             = ("Vehicle_ID",              "count"),
    ).round(0).sort_values("Avg_Resale", ascending=False).reset_index()
    cost_tbl.columns = ["Make", "Maintenance ($)", "Insurance ($)",
                         "Charging/mo ($)", "Resale ($)", "Vehicles"]
    st.dataframe(cost_tbl, use_container_width=True, hide_index=True)


