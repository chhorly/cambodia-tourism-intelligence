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

# 2. Enterprise Government MLOps Theme (Clean & Responsive)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans';
        
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .block-container { 
        padding: 1.5rem 1.5rem 2.5rem 1.5rem; 
        max-width: 100%; 
    }

    /* Government Header Bar */
    .gov-navbar {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .gov-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .gov-emblem-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        background: #1e3a8a;
        color: #facc15;
        font-size: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.25);
    }
    
    .gov-titles {
        display: flex;
        flex-direction: column;
    }
    
    .gov-title-khmer {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a8a;
        line-height: 1.25;
    }
    
    .gov-title-en {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #64748b;
        text-transform: uppercase;
    }
    
    .mlops-status-pill {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #334155;
    }

    /* Hero Banner with Angkor Backdrop */
    .mot-hero {
        position: relative;
        border-radius: 16px;
        padding: 2.5rem 1.8rem;
        color: white;
        margin-bottom: 1.5rem;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.6) 0%, rgba(15, 23, 42, 0.9) 100%),
                    url('https://images.unsplash.com/photo-1569154941061-e231b4725ef1?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.2);
    }
    
    .mot-hero-sub {
        font-size: 0.85rem;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .mot-hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #facc15;
        line-height: 1.3;
        margin-bottom: 0.6rem;
    }
    
    .mot-hero-desc {
        font-size: 0.88rem;
        color: #e2e8f0;
        max-width: 820px;
        line-height: 1.55;
    }

    /* Section Headings with Gold Underline */
    .mot-section-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 1.2rem 0 0.8rem 0;
        display: inline-block;
        position: relative;
    }
    
    .mot-section-header::after {
        content: "";
        display: block;
        width: 38px;
        height: 3px;
        background: #d97706;
        border-radius: 2px;
        margin-top: 4px;
    }

    /* MLOps Card Containers */
    .ds-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        border-top: 3.5px solid #1e3a8a;
        margin-bottom: 0.75rem;
    }
    
    .ds-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    
    .ds-card-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .ds-card-sub {
        font-size: 0.74rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem 0.8rem !important;
        }
        .gov-navbar {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .mot-hero {
            padding: 1.6rem 1.1rem !important;
            border-radius: 12px;
        }
        .mot-hero-title {
            font-size: 1.35rem !important;
        }
        .mot-hero-desc {
            font-size: 0.8rem !important;
        }
        .ds-card-val {
            font-size: 1.25rem !important;
        }
    }

    /* Footer */
    .gov-footer {
        border-top: 1px solid #e2e8f0;
        padding: 1.8rem 0;
        margin-top: 3rem;
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
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

# 5. Top Header (Stacked & Mobile-Clean)
api_tag = "API: LIVE (Port 8000)" if api_online else "API: FALLBACK MODE"
st.markdown(
    f"""
    <div class="gov-navbar">
        <div class="gov-brand">
            <div class="gov-emblem-badge">🇰🇭</div>
            <div class="gov-titles">
                <div class="gov-title-khmer">ក្រសួងទេសចរណ៍ • នាយកដ្ឋានស្ថិតិ និងផែនការ</div>
                <div class="gov-title-en">TOURISM INTELLIGENCE PLATFORM • APPLIED MLOps</div>
            </div>
        </div>
        <div class="mlops-status-pill">{api_tag}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Responsive Navigation Bar
nav_options = [
    "📊 Executive KPIs & Health",
    "📈 Forecast Curves & Residuals",
    "🚨 Early Warning & Action Matrix",
    "⚙️ Policy Simulation Sandbox",
    "📚 Data Lineage & Pipeline Architecture",
]

current_tab = st.selectbox(
    "Select Platform Module",
    options=nav_options,
    index=0,
    label_visibility="collapsed",
)

# -------------------------------------------------------------------
# TAB 1: EXECUTIVE KPIS & MODEL HEALTH
# -------------------------------------------------------------------
if current_tab == "📊 Executive KPIs & Health":
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
            """
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
        status_text = "LIVE (:8000)" if api_online else "FALLBACK (CSV)"
        status_color = "#059669" if api_online else "#d97706"
        st.markdown(
            f"""
            <div class="ds-card">
                <div class="ds-card-title">FastAPI Microservice</div>
                <div class="ds-card-val" style="color: {status_color}; font-size: 1.25rem;">{status_text}</div>
                <div class="ds-card-sub">PostgreSQL Container :5432</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='mot-section-header'>សូចនាករព្យាករណ៍ទូទាំងប្រទេស • 12-Month Projected Volume Summary</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cumulative Inbound", f"{forecast_df['Tuned_Forecast'].sum():,.0f} Pax")
    k2.metric("Monthly Mean Rate", f"{forecast_df['Tuned_Forecast'].mean():,.0f} Pax/mo")
    k3.metric("Baseline Variance (Mean)", f"{forecast_df['Deviation_%'].mean():+.2f}%")
    k4.metric(
        "Critical Alert Months",
        f"{(forecast_df['Clean_Status'] != 'Normal Demand (🟢)').sum()} / {len(forecast_df)} Months",
    )

# -------------------------------------------------------------------
# TAB 2: FORECAST CURVES & RESIDUALS
# -------------------------------------------------------------------
elif current_tab == "📈 Forecast Curves & Residuals":
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
        height=450,
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Inbound Arrivals", tickformat=",.0f", showgrid=True, gridcolor="#f1f5f9"),
        margin=dict(l=10, r=10, t=30, b=10),
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
                title="Monthly Inbound Residual (Actuals - Forecast)",
                color="Residual",
                color_continuous_scale="Tealrose",
            )
            fig_res.update_layout(plot_bgcolor="white", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c_res2:
            fig_ape = px.line(
                eval_df,
                x="Date",
                y="APE_%",
                title="Absolute Percentage Error (APE %) Horizon",
                markers=True,
            )
            fig_ape.update_traces(line_color="#e11d48", line_width=2.5)
            fig_ape.update_layout(plot_bgcolor="white", height=320, yaxis_ticksuffix="%", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ape, use_container_width=True)

# -------------------------------------------------------------------
# TAB 3: EARLY WARNING & ACTION MATRIX
# -------------------------------------------------------------------
elif current_tab == "🚨 Early Warning & Action Matrix":
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
            height=360,
            showlegend=False,
            plot_bgcolor="white",
            xaxis_title="",
            yaxis_title="Observed Months",
            margin=dict(l=10, r=10, t=20, b=10),
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
    st.caption("Evaluates sensitivity against LightGBM inference endpoints to observe climate & origin holiday interactions.")

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
            sim_kh_holidays = st.number_input("Cambodia Statutory Holidays", min_value=0, max_value=15, value=3)
            sim_origin_holidays = st.number_input("Total Key Origin Holidays", min_value=0, max_value=30, value=12)
            sim_china_holidays = st.number_input("China Holidays (Golden Week / Spring)", min_value=0, max_value=10, value=1)
            sim_covid = st.selectbox("Structural Shock (COVID Damping)", options=[0, 1], index=0)

        dispatch_btn = st.form_submit_button("⚡ Run LightGBM Inference", use_container_width=True)

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
                    r1.metric("Predicted Volume (ŷt)", f"{pred_val:,.0f} Arrivals")
                    r2.metric("Variance Delta (Bt)", f"{dev_val:+.2f}%")
                    r3.metric("Assigned EWS Tier", status_lbl)
                else:
                    st.error(f"Inference Failure: HTTP {res.status_code} - {res.text}")
            except Exception:
                st.warning(f"FastAPI microservice (:8000) offline. Executing model fallback formula:")
                pred_val = (sim_lag1 * 0.45) + (sim_roll3 * 0.55)
                dev_val = ((pred_val - sim_baseline) / sim_baseline) * 100
                st.info(f"Fallback Inferred Volume: **{pred_val:,.0f}** | Baseline Variance: **{dev_val:+.2f}%**")

# -------------------------------------------------------------------
# TAB 5: DATA LINEAGE & PIPELINE ARCHITECTURE
# -------------------------------------------------------------------
elif current_tab == "📚 Data Lineage & Pipeline Architecture":
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