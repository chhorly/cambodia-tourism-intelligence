from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="MoT Predictive Analytics & Early Warning Platform",
    page_icon="🇰🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Enterprise Government MLOps Theme (tourism.gov.kh Identity)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans', 'Kantumruy Pro', sans-serif; 
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .block-container { 
        /* Keep the first app row below Streamlit's fixed toolbar. */
        padding: 4.5rem 2rem 2.5rem 2rem; 
        max-width: 100%; 
    }

    /* Keep the wide header usable when the browser window is resized. */
    [data-testid="stHorizontalBlock"] {
        min-width: 0;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0;
    }

    [data-testid="stRadio"] > div {
        gap: 0.35rem;
        flex-wrap: wrap;
    }

    [data-testid="stRadio"] label {
        white-space: normal;
        line-height: 1.25;
    }

    @media (max-width: 1200px) {
        .block-container {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        [data-testid="stRadio"] label {
            font-size: 0.85rem;
        }
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 4rem;
        }

        .gov-navbar {
            align-items: flex-start;
        }

        .gov-title-khmer {
            font-size: 0.95rem;
        }

        .gov-title-en {
            font-size: 0.62rem;
            letter-spacing: 0.08em;
        }
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        .gov-navbar {
            padding: 0.5rem 0;
        }

        .gov-logo-img {
            width: 38px;
            height: 38px;
        }

        .gov-brand {
            gap: 0.55rem;
        }

        .gov-title-khmer {
            font-size: 0.78rem;
        }

        .gov-title-en {
            font-size: 0.5rem;
        }

        .mot-hero {
            padding: 2rem 1.25rem;
        }

        .mot-hero-title {
            font-size: 1.55rem;
        }
    }
    
    /* Government Header */
    .gov-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    
    .gov-brand {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    
    .gov-logo-img {
        width: 48px;
        height: 48px;
        object-fit: contain;
    }
    
    .gov-titles {
        display: flex;
        flex-direction: column;
    }
    
    .gov-title-khmer {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3a8a;
        line-height: 1.2;
    }
    
    .gov-title-en {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #64748b;
        text-transform: uppercase;
    }
    
    .mlops-badge {
        background: #0f172a;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #1e293b;
    }

    /* Hero Banner with Tourism Identity */
    .mot-hero {
        position: relative;
        border-radius: 18px;
        padding: 3.5rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.55) 0%, rgba(15, 23, 42, 0.88) 100%),
                    url('https://images.unsplash.com/photo-1569154941061-e231b4725ef1?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        box-shadow: 0 12px 30px -5px rgba(15, 23, 42, 0.25);
    }
    
    .mot-hero-sub {
        font-size: 0.95rem;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .mot-hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #facc15;
        line-height: 1.25;
        margin-bottom: 0.8rem;
    }
    
    .mot-hero-desc {
        font-size: 0.95rem;
        color: #e2e8f0;
        max-width: 850px;
        line-height: 1.6;
    }

    /* Section Headings with Gold Underline */
    .mot-section-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 1.5rem 0 1rem 0;
        display: inline-block;
        position: relative;
    }
    
    .mot-section-header::after {
        content: "";
        display: block;
        width: 42px;
        height: 3.5px;
        background: #d97706;
        border-radius: 2px;
        margin-top: 4px;
    }

    /* MLOps Card Containers */
    .ds-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border-top: 3.5px solid #1e3a8a;
    }
    
    .ds-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    
    .ds-card-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .ds-card-sub {
        font-size: 0.75rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Footer */
    .gov-footer {
        border-top: 1px solid #e2e8f0;
        padding: 2rem 0;
        margin-top: 3.5rem;
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 3. Data Ingestion & Fallback Resolution
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parents[1].name != ""
    else Path(".")
)
FORECAST_PATH = PROJECT_ROOT / "data" / "Cambodia_Tourism_Forecast_EWS.csv"
HISTORY_PATH = PROJECT_ROOT / "data" / "Cambodia_Tourism_Enriched.csv"

if not FORECAST_PATH.exists():
    FORECAST_PATH = Path("Cambodia_Tourism_Forecast_EWS.csv")
if not HISTORY_PATH.exists():
    HISTORY_PATH = Path("Cambodia_Tourism_Enriched.csv")


@st.cache_data
def load_datasets():
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
    forecast.columns = forecast.columns.str.strip()

    num_cols = [
        "Actual_Arrivals",
        "Tuned_Forecast",
        "Seasonal_Baseline",
        "Deviation_%",
        "APE_%",
    ]
    for col in num_cols:
        if col in forecast.columns:
            forecast[col] = pd.to_numeric(
                forecast[col].astype(str).str.replace("%", "").str.replace(",", ""),
                errors="coerce",
            )

    def clean_status(val):
        t = str(val)
        if "High" in t:
            return "High Demand (🔴)"
        if "Low" in t:
            return "Low Demand (🟡)"
        return "Normal Demand (🟢)"

    forecast["Clean_Status"] = forecast["Alert_Status"].apply(clean_status)

    def assign_action(status):
        if "High" in status:
            return "Surge border checkpoint staff at PNH/SAI; inspect hotel capacities; expand inter-provincial shuttles."
        if "Low" in status:
            return "Deploy targeted marketing campaigns; subsidize bundle air-hotel rates; offer promotional incentives."
        return "Maintain baseline resource allocations and scheduled off-peak maintenance schedules."

    forecast["Operational_Action"] = forecast["Clean_Status"].apply(assign_action)

    history = pd.DataFrame()
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
        history.columns = history.columns.str.strip()

    return forecast.sort_values("Date"), history.sort_values("Date")


try:
    forecast_df, history_df = load_datasets()
except Exception as e:
    st.error(f"Engine Load Exception: {e}")
    st.stop()

# 4. Check Microservice Health
API_URL = "http://127.0.0.1:8000"
api_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.2)
    if health_resp.status_code == 200:
        api_online = True
except Exception:
    api_online = False

# 5. Top Header Navigation
col_header, col_nav = st.columns([1.3, 2])
with col_header:
    st.markdown(
        f"""
        <div class="gov-navbar">
            <div class="gov-brand">
                <img class="gov-logo-img" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Emblem_of_Cambodia.svg/200px-Emblem_of_Cambodia.svg.png" alt="Cambodia Emblem">
                <div class="gov-titles">
                    <div class="gov-title-khmer">ក្រសួងទេសចរណ៍ • នាយកដ្ឋានស្ថិតិ និងផែនការ</div>
                    <div class="gov-title-en">DECISION SUPPORT SYSTEM • MACHINE LEARNING OPERATIONS</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav:
    current_tab = st.radio(
        "Navigation",
        options=[
            "📊 Executive KPIs & Model Health",
            "📈 Holdout Evaluation & Inferences",
            "🚨 Anomaly & Early Warning Matrix",
            "⚙️ Policy Simulation Sandbox",
            "📚 Data Architecture & Lineage",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

# -------------------------------------------------------------------
# TAB 1: EXECUTIVE KPIS & MODEL HEALTH
# -------------------------------------------------------------------
if current_tab == "📊 Executive KPIs & Model Health":
    st.markdown(
        """
        <div class="mot-hero">
            <div class="mot-hero-sub">Production Inferences & Predictive Intelligence</div>
            <div class="mot-hero-title">ព្រះរាជាណាចក្រអច្ឆរិយៈ • National Demand Forecast System</div>
            <div class="mot-hero-desc">
                Serving low-latency LightGBM regression inference to project national international tourist volumes. 
                Trained on 186 chronological monthly records across 2011–2026, fused with climate covariates and national holiday signals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='mot-section-header'>ម៉ូដែល និងដំណើរការប្រតិបត្តិការ • Champion Model Health & Lineage</div>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="ds-card">
                <div class="ds-card-title">Production Model</div>
                <div class="ds-card-val">LightGBM</div>
                <div class="ds-card-sub">Regressor • Leaf-wise GBDT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="ds-card">
                <div class="ds-card-title">Holdout MAPE</div>
                <div class="ds-card-val">32.9%</div>
                <div class="ds-card-sub">Outperforms SARIMAX (39.7%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="ds-card">
                <div class="ds-card-title">Test Horizon R²</div>
                <div class="ds-card-val">0.784</div>
                <div class="ds-card-sub">24-Month Temporal Holdout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        status_text = "LIVE (Port 8000)" if api_online else "OFFLINE (Fallback)"
        status_color = "#059669" if api_online else "#dc2626"
        st.markdown(
            f"""
            <div class="ds-card">
                <div class="ds-card-title">FastAPI Microservice</div>
                <div class="ds-card-val" style="color: {status_color}; font-size: 1.1rem; padding-top: 6px;">{status_text}</div>
                <div class="ds-card-sub">Dockerized PostgreSQL :5432</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='mot-section-header'>សូចនាករព្យាករណ៍ទូទាំងប្រទេស • 12-Month Projected Volume Summary</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cumulative Projected Inbound", f"{forecast_df['Tuned_Forecast'].sum():,.0f} Pax")
    k2.metric("Monthly Mean Projected Run-Rate", f"{forecast_df['Tuned_Forecast'].mean():,.0f} Pax/mo")
    k3.metric("Baseline Variance Delta (Mean)", f"{forecast_df['Deviation_%'].mean():+.2f}%")
    k4.metric(
        "Critical Alert Load (Anomaly)",
        f"{(forecast_df['Clean_Status'] != 'Normal Demand (🟢)').sum()} / {len(forecast_df)} Horizon Months",
    )

# -------------------------------------------------------------------
# TAB 2: HOLDOUT EVALUATION & INFERENCES
# -------------------------------------------------------------------
elif current_tab == "📈 Holdout Evaluation & Inferences":
    st.markdown("<div class='mot-section-header'>ការព្យាករណ៍តម្រូវការទេសចរណ៍ • Time-Series Forecast Curve vs. Actuals</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Seasonal_Baseline"],
            name="Seasonal Baseline (Norm Bt)",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Tuned_Forecast"],
            name="Tuned LightGBM Regressor (ŷt)",
            line=dict(color="#1d4ed8", width=3),
            mode="lines+markers",
        )
    )
    if "Actual_Arrivals" in forecast_df.columns and forecast_df["Actual_Arrivals"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Actual_Arrivals"],
                name="Empirical Ground Truth (yt)",
                line=dict(color="#059669", width=2.5),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        height=460,
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Inbound International Arrivals", tickformat=",.0f", showgrid=True, gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals & Error Breakdown
    if "Actual_Arrivals" in forecast_df.columns and forecast_df["Actual_Arrivals"].notna().any():
        st.markdown("<div class='mot-section-header'>ការវិភាគកំហុស • Residual Distribution & Error Metrics</div>", unsafe_allow_html=True)
        eval_df = forecast_df.dropna(subset=["Actual_Arrivals"]).copy()
        eval_df["Residual"] = eval_df["Actual_Arrivals"] - eval_df["Tuned_Forecast"]
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            fig_res = px.bar(
                eval_df,
                x="Date",
                y="Residual",
                title="Monthly Inbound Residual Error (Actuals - Forecast)",
                color="Residual",
                color_continuous_scale="Tealrose",
            )
            fig_res.update_layout(plot_bgcolor="white", height=320)
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c_res2:
            fig_ape = px.line(
                eval_df,
                x="Date",
                y="APE_%",
                title="Absolute Percentage Error (APE %) Over Evaluation Horizon",
                markers=True,
            )
            fig_ape.update_traces(line_color="#e11d48", line_width=2.5)
            fig_ape.update_layout(plot_bgcolor="white", height=320, yaxis_ticksuffix="%")
            st.plotly_chart(fig_ape, use_container_width=True)

# -------------------------------------------------------------------
# TAB 3: ANOMALY & EARLY WARNING MATRIX
# -------------------------------------------------------------------
elif current_tab == "🚨 Anomaly & Early Warning Matrix":
    st.markdown("<div class='mot-section-header'>ប្រព័ន្ធប្រកាសអាសន្ន និងវិធានការប្រតិបត្តិ • Early Warning Threshold Distribution</div>", unsafe_allow_html=True)

    col_dist, col_table = st.columns([1, 2.2])
    with col_dist:
        st.caption("Distribution of Seasonal Deviations ($\pm 15\%$)")
        order = ["High Demand (🔴)", "Normal Demand (🟢)", "Low Demand (🟡)"]
        palette = {
            "High Demand (🔴)": "#ef4444",
            "Normal Demand (🟢)": "#10b981",
            "Low Demand (🟡)": "#f59e0b",
        }
        counts = (
            forecast_df["Clean_Status"]
            .value_counts()
            .reindex(order)
            .fillna(0)
            .reset_index()
        )
        counts.columns = ["Status", "Count"]

        fig_bar = px.bar(
            counts,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map=palette,
            text="Count",
        )
        fig_bar.update_layout(
            height=380,
            showlegend=False,
            plot_bgcolor="white",
            xaxis_title="",
            yaxis_title="Observed Months",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.caption("Prescriptive Operations & Threshold Classification Matrix")
        display_df = forecast_df[[
            "Date",
            "Clean_Status",
            "Tuned_Forecast",
            "Seasonal_Baseline",
            "Deviation_%",
            "Operational_Action",
        ]].copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%b %Y")
        display_df["Tuned_Forecast"] = display_df["Tuned_Forecast"].apply(lambda x: f"{x:,.0f}")
        display_df["Seasonal_Baseline"] = display_df["Seasonal_Baseline"].apply(lambda x: f"{x:,.0f}")
        display_df["Deviation_%"] = display_df["Deviation_%"].apply(lambda x: f"{x:+.1f}%")
        display_df.columns = [
            "Month",
            "Alert Level",
            "Forecast (ŷt)",
            "Baseline (Bt)",
            "Deviation Δ%",
            "Operational Logistics Directive",
        ]

        def style_status(val):
            if "High" in str(val):
                return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
            if "Low" in str(val):
                return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
            return "background-color: #d1fae5; color: #065f46; font-weight: 700;"

        styler = display_df.style
        if hasattr(styler, "map"):
            styled_df = styler.map(style_status, subset=["Alert Level"])
        else:
            styled_df = styler.applymap(style_status, subset=["Alert Level"])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# TAB 4: POLICY SIMULATION SANDBOX (WHAT-IF INFERENCE)
# -------------------------------------------------------------------
elif current_tab == "⚙️ Policy Simulation Sandbox":
    st.markdown("<div class='mot-section-header'>ការពិសោធន៍គោលនយោបាយ • Real-Time Feature Attribution Sandbox</div>", unsafe_allow_html=True)
    st.markdown(
        "Directly evaluates sensitivity against live LightGBM inference endpoints (`/api/v1/predict`) to observe non-linear interactions across meteorological indices, origin market holidays, and seasonal trends."
    )

    with st.form("simulation_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_date = st.date_input("Target Evaluation Month", value=pd.to_datetime("2026-11-01"))
            sim_baseline = st.number_input("Seasonal Baseline Target (Bt)", value=364436.0, step=1000.0)
            sim_lag1 = st.number_input("Lag 1 Inbound Arrivals (t-1)", value=327239.0, step=1000.0)
            sim_lag3 = st.number_input("Lag 3 Inbound Arrivals (t-3)", value=347492.0, step=1000.0)
            sim_lag12 = st.number_input("Lag 12 Inbound Arrivals (t-12)", value=300000.0, step=1000.0)

        with col_s2:
            sim_roll3 = st.number_input("Rolling 3-Month Mean", value=337694.0, step=1000.0)
            sim_roll12 = st.number_input("Rolling 12-Month Mean", value=350000.0, step=1000.0)
            sim_temp = st.slider("Mean Temperature (°C)", min_value=20.0, max_value=38.0, value=27.5, step=0.1)
            sim_rain = st.slider("Total Precipitation (Rainfall mm)", min_value=0.0, max_value=600.0, value=110.0, step=5.0)

        with col_s3:
            sim_kh_holidays = st.number_input("Cambodia Statutory Holiday Days", min_value=0, max_value=15, value=3)
            sim_origin_holidays = st.number_input("Total Key Origin Holidays", min_value=0, max_value=30, value=12)
            sim_china_holidays = st.number_input("China Holidays (Golden Week / Spring Fest)", min_value=0, max_value=10, value=1)
            sim_covid = st.selectbox("Structural Shock (COVID Damping Parameter)", options=[0, 1], index=0)

        dispatch_btn = st.form_submit_button("⚡ Run Live LightGBM Inference", use_container_width=True)

    if dispatch_btn:
        payload = {
            "date": str(sim_date),
            "seasonal_baseline": float(sim_baseline),
            "Lag_1": float(sim_lag1),
            "Lag_3": float(sim_lag3),
            "Lag_12": float(sim_lag12),
            "Rolling_3_Mean": float(sim_roll3),
            "Rolling_12_Mean": float(sim_roll12),
            "Average_Temperature": float(sim_temp),
            "Rainfall": float(sim_rain),
            "Cambodia_Holiday_Days": int(sim_kh_holidays),
            "COVID_Indicator": int(sim_covid),
            "Month_Number": int(sim_date.month),
            "Total_Origin_Holidays": int(sim_origin_holidays),
            "China_Holidays": int(sim_china_holidays),
        }

        with st.spinner("Dispatching payload to FastAPI inference microservice..."):
            try:
                res = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=3.0)
                if res.status_code == 200:
                    out = res.json()
                    pred_val = out["predicted_arrivals"]
                    dev_val = out["deviation_pct"]
                    status_lbl = out["alert_status"]

                    st.success("✅ Real-Time Prediction Completed & Synchronized to PostgreSQL")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Predicted Arrival Volume (ŷt)", f"{pred_val:,.0f} Arrivals")
                    r2.metric("Variance Against Baseline (Bt)", f"{dev_val:+.2f}%")
                    r3.metric("Assigned EWS Operational Tier", status_lbl)
                else:
                    st.error(f"Inference Failure: HTTP {res.status_code} - {res.text}")
            except Exception as ex:
                st.warning(f"Connection Exception to {API_URL}: Microservice is offline. Local fallback calculation:")
                # Local estimate fallback
                pred_val = (sim_lag1 * 0.45) + (sim_roll3 * 0.55)
                dev_val = ((pred_val - sim_baseline) / sim_baseline) * 100
                st.info(f"Fallback Inferred Volume: **{pred_val:,.0f}** | Baseline Variance: **{dev_val:+.2f}%**")

# -------------------------------------------------------------------
# TAB 5: DATA ARCHITECTURE & LINEAGE
# -------------------------------------------------------------------
elif current_tab == "📚 Data Architecture & Lineage":
    st.markdown("<div class='mot-section-header'>ស្ថាបត្យកម្មទិន្នន័យ • Data Pipelines, Features & Specifications</div>", unsafe_allow_html=True)

    col_meta, col_feat = st.columns([1, 1.3])
    with col_meta:
        st.markdown("""
            ### Technical Specification
            * **Primary Target:** `International_Tourist_Arrivals` (Monthly aggregations)
            * **Historical Observations:** 186 monthly records (January 2011 – June 2026)
            * **Validation Strategy:** Chronological holdout validation (zero randomized shuffling to prevent temporal look-ahead leakage)
            * **Serving Layer:** Asynchronous FastAPI REST API mounted via Uvicorn
            * **Transactional Store:** PostgreSQL 15 (`tourism_db:5432`) running inside Docker
            * **Pipeline Orchestrator:** Apache Airflow DAG (`@monthly` schedule)
            """)
    
    with col_feat:
        st.markdown("""
            ### Engineered Feature Attributes & Signals
            * **Autoregressive Temporal Signals:** $Lag_1, Lag_3, Lag_{12}$ captures momentum, quarterly oscillations, and annual seasonality.
            * **Rolling Statistics:** $Rolling_3\text{ (Mean)}, Rolling_{12}\text{ (Mean)}$ constructed with explicit $t-1$ lag shift.
            * **Meteorological Covariates:** Surface Precipitation ($\text{Rainfall mm}$) and Mean Surface Temperature (°C) modulating monsoon impact.
            * **Calendar & Holiday Covariates:** Domestic Cambodia Holidays (Khmer New Year, Pchum Ben), China Statutory Holidays (Golden Week, Lunar New Year), and Aggregate Key Origin Market calendars.
            * **Structural Shock Damping:** Binary COVID regime parameter isolating non-repeating exogenous structural demand disruptions.
            """)

# 6. Official Government Footer
st.markdown(
    """
    <div class="gov-footer">
        <strong>ក្រសួងទេសចរណ៍ • MINISTRY OF TOURISM OF CAMBODIA</strong><br>
        Cambodia Tourism Intelligence Platform | Developed by RY Chhorly | Supervised by Ya Manon<br>
        Kingdom of Wonder • Applied Machine Learning Early Warning System (EWS)
    </div>
    """,
    unsafe_allow_html=True,
)