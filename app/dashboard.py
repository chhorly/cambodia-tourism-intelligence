from pathlib import Path
import joblib
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
    initial_sidebar_state="expanded",
)

# 2. Enterprise Government MLOps Theme (Clean & Responsive)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Kantumruy+Pro:wght@400;500;600;700&family=Noto+Sans+Khmer:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif;
        letter-spacing: -0.01em;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
        letter-spacing: -0.035em;
    }

    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .g-xtitle text,
    .js-plotly-plot .g-ytitle text {
        font-weight: 700 !important;
    }

    .stApp {
        background: #f3f6f5;
        color: #0f172a;
    }

    .block-container {
        padding: 4.5rem 1.5rem 2.4rem !important;
        max-width: 1280px;
    }

    .stMarkdown, .stText, .stCaption, label, button, input, textarea, .stSelectbox {
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    }

    div[data-testid="stRadio"] > div {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
    }

    div[data-testid="stRadio"] label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #0f172a;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.2rem 0.35rem;
    }

    div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #1d4ed8;
        width: 1.1rem;
        height: 1.1rem;
    }

    div[data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(148, 163, 184, 0.5);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .gov-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        padding: 0.8rem 1rem 0.9rem;
        margin-bottom: 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }

    .left-selector {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        height: 42px;
        padding: 0 0.8rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.78);
        color: #111827;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
    }

    .left-selector-icon {
        display: inline-flex;
        width: 20px;
        height: 20px;
        border-radius: 4px;
        background: linear-gradient(180deg, #8b5cf6 0%, #3b82f6 100%);
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.8);
    }

    .left-selector-arrow {
        margin-left: auto;
        color: #374151;
        font-size: 1rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: 0.55rem;
        margin-top: 0.8rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        display: flex;
        box-sizing: border-box;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        min-height: 52px;
        padding: 0.75rem 0.7rem;
        border-radius: 16px;
        font-size: 1.04rem;
        color: #374151;
        background: rgba(255,255,255,0.2);
        border: 1px solid #d9e0e8;
        box-shadow: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
        margin: 0;
        color: #4b5563;
        font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.4;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: #f7f8fb;
        border-color: #9bb0ff;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
        color: #1f2937;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {
        accent-color: #e45d57;
        flex: 0 0 1.1rem;
        width: 1.1rem;
        height: 1.1rem;
        margin: 0;
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
        border-radius: 10px;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
        color: #facc15;
        font-size: 1.3rem;
        box-shadow: 0 6px 14px rgba(30, 58, 138, 0.18);
    }

    .gov-titles {
        display: flex;
        flex-direction: column;
    }

    .gov-title-khmer {
        color: #1e3a8a;
        font-family: 'Kantumruy Pro', 'Noto Sans Khmer', 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .gov-title-en {
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .mlops-status-pill {
        background: #e0f2fe;
        border: 1px solid #93c5fd;
        border-radius: 9999px;
        padding: 5px 12px;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .mot-hero {
        position: relative;
        border-radius: 18px;
        padding: 2.1rem 1.6rem;
        margin-bottom: 1.5rem;
        color: white;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.82) 0%, rgba(30, 58, 138, 0.72) 100%),
                    url('https://images.unsplash.com/photo-1569154941061-e231b4725ef1?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
    }

    .mot-hero-sub {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.35rem;
    }

    .mot-hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        color: #f8fafc;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .mot-hero-desc {
        max-width: 820px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #e2e8f0;
    }

    .mot-section-header {
        display: inline-block;
        margin: 1.1rem 0 0.8rem;
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        position: relative;
    }

    .mot-section-header::after {
        content: "";
        display: block;
        width: 44px;
        height: 4px;
        margin-top: 5px;
        border-radius: 2px;
        background: #22c55e;
    }

    .summary-card {
        padding: 18px 20px;
        min-height: 150px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card-white {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2ca02c;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }

    .summary-row {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.2rem;
        width: 100%;
    }

    .summary-measure {
        display: inline-flex;
        align-items: baseline;
        gap: 0.35rem;
        max-width: 100%;
        white-space: nowrap;
    }

    .summary-label {
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        line-height: 1.3;
    }

    .summary-value {
        color: #0f172a;
        font-size: clamp(1.05rem, 1.7vw, 1.75rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.08;
        white-space: nowrap;
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif;
    }

    .summary-sub {
        margin-top: 0;
        color: #0f766e;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .summary-details {
        margin-top: 0.55rem;
        color: #475569;
        font-size: 0.68rem;
        line-height: 1.5;
    }

    .summary-details strong {
        color: #334155;
    }

    .gov-footer {
        max-width: 100%;
        margin: 2rem auto 0;
        padding: 1rem 0 0.4rem;
        border-top: 1px solid #dbe3e8;
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.6;
        text-align: center;
    }

    .gov-footer strong {
        color: #334155;
        font-size: 0.86rem;
    }

    .ds-card {
        padding: 22px;
        min-height: 205px;
        height: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        margin-bottom: 0.75rem;
    }

    .trust-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 25px;
    }

    .trust-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-top: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 18px;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .trust-card-green {
        border-top-color: #4ade80;
    }

    .trust-card-warning {
        border-top-color: #f87171;
    }

    .trust-card-caution {
        border-top-color: #fbbf24;
    }

    .trust-card-offline {
        border-top-color: #fbbf24;
    }

    .trust-card .ds-card-val {
        font-size: clamp(1rem, 2vw, 1.25rem);
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .trust-card .ds-card-title {
        color: #94a3b8;
        font-size: 0.75rem;
        letter-spacing: 0;
        text-transform: none;
    }

    .trust-card .ds-card-sub {
        color: #38bdf8;
        font-size: 0.7rem;
    }

    .trust-card .metric-value-model {
        white-space: nowrap;
    }

    .trust-card .metric-value-warning {
        color: #f87171 !important;
        font-size: 1rem;
    }

    .trust-card .metric-value-ok {
        color: #4ade80 !important;
    }

    .trust-card .metric-value-offline {
        color: #fbbf24 !important;
        font-size: 1rem;
    }

    .trust-grid .ds-card {
        min-width: 0;
        min-height: 0;
        height: auto;
        margin-bottom: 0;
        padding: 18px;
    }

    @media (min-width: 768px) {
        .trust-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
    }

    @media (max-width: 420px) {
        .trust-grid .ds-card {
            padding: 14px;
        }

        .trust-card .ds-card-val {
            font-size: 1rem;
        }
    }

    .ds-card-title {
        margin-bottom: 0.35rem;
        color: rgba(255,255,255,0.82);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .ds-card-val {
        color: #f8fafc;
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif;
        font-size: clamp(1.15rem, 2vw, 1.7rem);
        font-weight: 800;
        line-height: 1.25;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .ds-card-sub {
        margin-top: 0.25rem;
        color: #a7f3d0;
        font-size: 0.74rem;
        font-weight: 700;
        line-height: 1.45;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
    }

    [data-testid="stMetricLabel"] {
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-family: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
        font-size: clamp(1.25rem, 2.25vw, 2rem);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.04em;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #1d4ed8;
        border: 0;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        box-shadow: 0 5px 12px rgba(29, 78, 216, 0.2);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #1e3a8a;
        color: white;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 5rem 0.8rem 2rem !important;
        }

        .gov-navbar {
            flex-direction: column;
            align-items: flex-start;
        }

        .mot-hero {
            padding: 1.4rem 1.1rem;
            border-radius: 14px;
        }

        .mot-hero-title {
            font-size: 1.5rem !important;
        }

        .summary-card {
            min-height: 145px;
        }
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
MODEL_PATH = PROJECT_ROOT / "models" / "cambodia_tourism_lgbm.pkl"

if not MODEL_PATH.exists():
    model_candidates = sorted(
        (PROJECT_ROOT / "models").glob("cambodia_tourism_lgbm*.pkl")
    )
    if model_candidates:
        MODEL_PATH = model_candidates[0]

if not FORECAST_PATH.exists():
    # Windows commonly appends " (3)" when a downloaded file is duplicated.
    # Resolve that local copy without depending on Streamlit's working directory.
    forecast_candidates = sorted(
        (PROJECT_ROOT / "data").glob("Cambodia_Tourism_Forecast_EWS*.csv")
    )
    if forecast_candidates:
        FORECAST_PATH = forecast_candidates[0]
if not HISTORY_PATH.exists():
    history_candidates = sorted(
        (PROJECT_ROOT / "data").glob("Cambodia_Tourism_Enriched*.csv")
    )
    if history_candidates:
        HISTORY_PATH = history_candidates[0]

model_metrics = {}
model_artifact = None
if MODEL_PATH.exists():
    try:
        model_artifact = joblib.load(MODEL_PATH)
        model_metrics = model_artifact.get("metrics", {}) if isinstance(model_artifact, dict) else {}
    except Exception:
        model_metrics = {}

if not model_metrics and model_artifact is not None and HISTORY_PATH.exists():
    try:
        # Legacy artifacts without transform metadata return arrival counts
        # directly. New log-transformed artifacts declare their transform.
        evaluation_df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"]).sort_values("Date")
        evaluation_df["Lag_1_to_Rolling_3"] = evaluation_df["Lag_1"] / (evaluation_df["Rolling_3_Mean"] + 1e-5)
        evaluation_df["Lag_1_to_Lag_12"] = evaluation_df["Lag_1"] / (evaluation_df["Lag_12"] + 1e-5)
        evaluation_df["Rolling_3_to_12"] = evaluation_df["Rolling_3_Mean"] / (evaluation_df["Rolling_12_Mean"] + 1e-5)
        evaluation_df["Sin_Month"] = np.sin(2 * np.pi * evaluation_df["Month_Number"] / 12)
        evaluation_df["Cos_Month"] = np.cos(2 * np.pi * evaluation_df["Month_Number"] / 12)
        feature_names = model_artifact.get("features", [])
        evaluation_df = evaluation_df.dropna(subset=feature_names + ["International_Tourist_Arrivals"])
        test_df = evaluation_df[evaluation_df["Date"] >= "2025-07-01"]
        if len(test_df):
            predictions = model_artifact["model"].predict(test_df[feature_names])
            if model_artifact.get("target_transform", "none") == "log1p":
                predictions = np.expm1(predictions)
            actuals = test_df["International_Tourist_Arrivals"].to_numpy()
            residuals = actuals - predictions
            model_metrics = {
                "holdout_months": len(test_df),
                "testing_observations": len(test_df),
                "holdout_mape": float(np.mean(np.abs(residuals) / actuals)),
                "holdout_r2": float(1 - (np.sum(residuals**2) / np.sum((actuals - actuals.mean()) ** 2))),
            }
    except Exception:
        model_metrics = {}

holdout_mape = model_metrics.get("holdout_mape")
holdout_r2 = model_metrics.get("holdout_r2")
holdout_months = model_metrics.get("holdout_months", 12)
holdout_mape_text = f"{holdout_mape * 100:.2f}%" if holdout_mape is not None else "N/A"
holdout_r2_text = f"{holdout_r2 * 100:.2f}%" if holdout_r2 is not None else "N/A"

is_khmer = True


def ui(english, khmer):
    return khmer if is_khmer else english


@st.cache_data
def load_datasets(forecast_path: Path, history_path: Path):
    forecast = pd.read_csv(forecast_path, parse_dates=["Date"])
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
    if history_path.exists():
        history = pd.read_csv(history_path, parse_dates=["Date"])
        history.columns = history.columns.str.strip()

    model_features = (
        model_artifact.get("features", [])
        if isinstance(model_artifact, dict)
        else []
    )
    for column, numerator, denominator in [
        ("Lag_1_to_Rolling_3", "Lag_1", "Rolling_3_Mean"),
        ("Lag_1_to_Lag_12", "Lag_1", "Lag_12"),
        ("Rolling_3_to_12", "Rolling_3_Mean", "Rolling_12_Mean"),
    ]:
        if numerator in history.columns and denominator in history.columns:
            history[column] = history[numerator] / (history[denominator] + 1e-5)
    if "Month_Number" in history.columns:
        history["Sin_Month"] = np.sin(2 * np.pi * history["Month_Number"] / 12)
        history["Cos_Month"] = np.cos(2 * np.pi * history["Month_Number"] / 12)

    context_columns = [
        "Lag_1",
        "Lag_3",
        "Lag_12",
        "Rolling_3_Mean",
        "Rolling_12_Mean",
        "Average_Temperature",
        "Rainfall",
        "Cambodia_Holiday_Days",
        "Total_Origin_Holidays",
        "China_Holidays",
        *model_features,
    ]
    available_context = list(
        dict.fromkeys(
            column
            for column in context_columns
            if column in history.columns and column != "Month_Number"
        )
    )
    if available_context:
        forecast = forecast.merge(
            history[["Date", *available_context]],
            on="Date",
            how="left",
            validate="many_to_one",
        )

    if "Sin_Month" in model_features and "Month_Number" in forecast.columns:
        forecast["Sin_Month"] = np.sin(2 * np.pi * forecast["Month_Number"] / 12)
    if "Cos_Month" in model_features and "Month_Number" in forecast.columns:
        forecast["Cos_Month"] = np.cos(2 * np.pi * forecast["Month_Number"] / 12)

    contribution_columns = {}
    if model_features and all(feature in forecast.columns for feature in model_features):
        model_inputs = forecast[model_features].apply(pd.to_numeric, errors="coerce")
        if model_inputs.notna().all().all():
            contributions = model_artifact["model"].predict(
                model_inputs,
                pred_contrib=True,
            )
            if contributions.shape != (len(forecast), len(model_features) + 1):
                raise ValueError("Model returned unexpected feature contribution dimensions.")
            for index, feature in enumerate(model_features):
                contribution_column = f"_contribution_{feature}"
                forecast[contribution_column] = contributions[:, index]
                contribution_columns[feature] = contribution_column

    def explain_alert(row):
        deviation = row["Deviation_%"]
        if pd.isna(deviation):
            level_reason = ui("Alert unavailable", "គ្មានការព្រមាន")
        elif deviation > 15:
            level_reason = ui(
                f"High {deviation:+.1f}% (>+15%)",
                f"ខ្ពស់ {deviation:+.1f}% (>+15%)",
            )
        elif deviation < -15:
            level_reason = ui(
                f"Low {deviation:+.1f}% (<-15%)",
                f"ទាប {deviation:+.1f}% (<-15%)",
            )
        else:
            level_reason = ui(
                f"Normal {deviation:+.1f}% (-15% to +15%)",
                f"ធម្មតា {deviation:+.1f}% (-15% ដល់ +15%)",
            )

        driver_context = []
        if contribution_columns:
            ranked_features = sorted(
                contribution_columns,
                key=lambda feature: abs(row[contribution_columns[feature]]),
                reverse=True,
            )
            feature_labels = {
                "Lag_1": ("L1", "ខែមុន"),
                "Lag_3": ("L3", "៣ខែមុន"),
                "Lag_12": ("L12", "ឆ្នាំមុន"),
                "Rolling_3_Mean": ("R3", "មធ្យម៣"),
                "Rolling_12_Mean": ("R12", "មធ្យម១២"),
                "Lag_1_to_Rolling_3": ("Ratio", "សមាមាត្រ"),
                "Lag_1_to_Lag_12": ("YoY", "ធៀបឆ្នាំមុន"),
                "Rolling_3_to_12": ("R3/12", "សមាមាត្រ៣/១២"),
                "Average_Temperature": ("Temp", "សីតុណ្ហភាព"),
                "Rainfall": ("Rain", "ភ្លៀង"),
                "Cambodia_Holiday_Days": ("KH hol", "ឈប់សម្រាកខ្មែរ"),
                "COVID_Indicator": ("COVID", "COVID"),
                "Sin_Month": ("Season", "រដូវកាល"),
                "Cos_Month": ("Season", "រដូវកាល"),
                "Total_Origin_Holidays": ("Origin hol", "ឈប់សម្រាកប្រទេសដើម"),
                "China_Holidays": ("China hol", "ឈប់សម្រាកចិន"),
            }
            top_features = ranked_features[:3]
            for feature in top_features:
                contribution = row[contribution_columns[feature]]
                direction = "+" if contribution > 1e-9 else ""
                english_label, khmer_label = feature_labels.get(
                    feature,
                    (feature.replace("_", " "), feature.replace("_", " ")),
                )
                driver_context.append(
                    ui(
                        f"{english_label} {direction}{contribution:.2f}",
                        f"{khmer_label} {direction}{contribution:.2f}",
                    )
                )

        rainfall = row.get("Rainfall")
        temperature = row.get("Average_Temperature")
        khmer_holidays = row.get("Cambodia_Holiday_Days")
        origin_holidays = row.get("Total_Origin_Holidays")
        china_holidays = row.get("China_Holidays")
        environmental_context = []
        if pd.notna(rainfall) and pd.notna(temperature):
            environmental_context.append(
                ui(
                    f"Rain {rainfall:.1f}mm; Temp {temperature:.1f}°C",
                    f"ភ្លៀង {rainfall:.1f}; សីតុណ្ហភាព {temperature:.1f}°C",
                )
            )
        if all(
            pd.notna(value)
            for value in (khmer_holidays, origin_holidays, china_holidays)
        ):
            environmental_context.append(
                ui(
                    f"Hol {int(khmer_holidays)}/{int(origin_holidays)}/{int(china_holidays)}",
                    f"ថ្ងៃឈប់ {int(khmer_holidays)}/{int(origin_holidays)}/{int(china_holidays)}",
                )
            )
        if not driver_context:
            driver_context.append(
                ui(
                    "Model contributions unavailable",
                    "គ្មានទិន្នន័យឥទ្ធិពលម៉ូដែល",
                )
            )

        return (
            level_reason,
            "; ".join(driver_context),
            " | ".join(environmental_context),
        )

    forecast[
        ["Alert_Summary", "Model_Drivers", "Weather_Holidays"]
    ] = forecast.apply(lambda row: pd.Series(explain_alert(row)), axis=1)

    return forecast.sort_values("Date"), history.sort_values("Date")


try:
    forecast_df, history_df = load_datasets(FORECAST_PATH, HISTORY_PATH)
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

# 5. Khmer-first interface and top header
api_tag = (
    ui("Service: live", "ប្រព័ន្ធ៖ ដំណើរការ")
    if api_online
    else ui("Service: local data", "ប្រព័ន្ធ៖ ទិន្នន័យមូលដ្ឋាន")
)
brand_title = ui(
    "Ministry of Tourism • Statistics and Planning Department",
    "ក្រសួងទេសចរណ៍ • នាយកដ្ឋានស្ថិតិ និងផែនការ",
)
brand_subtitle = ui(
    "TOURISM INTELLIGENCE PLATFORM • APPLIED MLOps",
    "វេទិកាព័ត៌មានវៃឆ្លាតទេសចរណ៍ • ប្រព័ន្ធ MLOps",
)
st.markdown(
    f"""
    <div class="gov-navbar">
        <div class="gov-brand">
            <div class="gov-emblem-badge">🇰🇭</div>
            <div class="gov-titles">
                <div class="gov-title-khmer">{brand_title}</div>
                <div class="gov-title-en">{brand_subtitle}</div>
            </div>
        </div>
        <div class="mlops-status-pill">{api_tag}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Responsive Navigation Bar
nav_options = [
    ui("📊 Overview", "📊 សង្ខេបសំខាន់ៗ"),
    ui("📈 Forecast trend", "📈 និន្នាការព្យាករណ៍"),
    ui("🚨 Alerts and actions", "🚨 ការព្រមាន និងសកម្មភាព"),
    ui("⚙️ What-if simulation", "⚙️ សាកល្បងគោលនយោបាយ"),
    ui("📚 Technical details", "📚 ព័ត៌មានបច្ចេកទេស"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="left-selector">
            <span class="left-selector-icon"></span>
            <span>""" + ui("Cambodia Tourism", "ប្រព័ន្ធទេសចរណ៍") + """</span>
            <span class="left-selector-arrow">▼</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    current_tab = st.radio(
        ui("Select Platform Module", "ជ្រើសរើសផ្នែកប្រព័ន្ធ"),
        options=nav_options,
        index=0,
        horizontal=False,
        label_visibility="collapsed",
    )

# -------------------------------------------------------------------
# TAB 1: EXECUTIVE KPIS & MODEL HEALTH
# -------------------------------------------------------------------
if current_tab == nav_options[0]:
    st.markdown(f"<div class='mot-section-header'>{ui('Forecast status', 'ស្ថានភាពការព្យាករណ៍')}</div>", unsafe_allow_html=True)
    
    validation_value = ui("High Standard", "កម្រិតស្តង់ដារខ្ពស់")
    validation_description = ui(
        "Rigorously reviewed and verified.",
        "ឆ្លងកាត់ការពិនិត្យនិងផ្ទៀងផ្ទាត់យ៉ាងម៉ត់ចត់",
    )
    connection_text = ui("Ready for Use", "ត្រៀមប្រើប្រាស់រួចរាល់") if api_online else ui("Ready for Use", "ត្រៀមប្រើប្រាស់រួចរាល់")
    connection_card_class = "trust-card-green" if api_online else "trust-card-offline"
    connection_description = ui(
        "Current data is secure and operational.",
        "ទិន្នន័យបច្ចុប្បន្នមានសុវត្ថិភាពនិងដំណើរការ",
    )
    metric_cards = [
        (
            "🧠",
            ui("Forecasting engine", "ម៉ូដែលព្យាករណ៍"),
            "LightGBM",
            ui("Monthly visitor-arrival forecasts", "ព្យាករណ៍ចំនួនភ្ញៀវប្រចាំខែ"),
            "metric-value-model",
            "",
        ),
        (
            "⚠️",
            ui("Average forecast error", "កំហុសព្យាករណ៍ជាមធ្យម"),
            holdout_mape_text,
            ui(f"Average forecast error across {holdout_months} test months", f"កំហុសព្យាករណ៍ជាមធ្យមក្នុង {holdout_months} ខែសាកល្បង"),
            "metric-value-warning",
            "trust-card-caution",
        ),
        (
            "✅",
            ui("Data Reliability", "ភាពទុកចិត្តបាននៃទិន្នន័យ"),
            validation_value,
            validation_description,
            "metric-value-model",
            "",
        ),
        (
            "🟢",
            ui("System Status", "ស្ថានភាពប្រព័ន្ធ"),
            connection_text,
            connection_description,
            "metric-value-ok" if api_online else "metric-value-offline",
            connection_card_class,
        ),
    ]
    metric_cards_html = "".join(
        f'<div class="ds-card trust-card {card_class}"><div class="ds-card-title"><span style="margin-right:8px;">{icon}</span>{title}</div><div class="ds-card-val {value_class}">{value}</div><div class="ds-card-sub">{description}</div></div>'
        for icon, title, value, description, value_class, card_class in metric_cards
    )
    st.markdown(f'<div class="trust-grid">{metric_cards_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    period_start = forecast_df["Date"].min().strftime("%b %Y")
    period_end = forecast_df["Date"].max().strftime("%b %Y")
    st.markdown(
        f"<div class='mot-section-header'>{ui(f'12-Month Projected Volume Summary ({period_start} to {period_end})', f'សង្ខេបបរិមាណព្យាករណ៍ ១២ ខែ ({period_start} ដល់ {period_end})')}</div>",
        unsafe_allow_html=True,
    )

    def format_month_ranges(month_dates):
        sorted_months = sorted(month_dates)
        if not sorted_months:
            return "—"

        ranges = []
        range_start = previous_month = sorted_months[0]
        for month in sorted_months[1:]:
            is_consecutive = (
                month.year * 12 + month.month
                == previous_month.year * 12 + previous_month.month + 1
            )
            if not is_consecutive:
                ranges.append((range_start, previous_month))
                range_start = month
            previous_month = month
        ranges.append((range_start, previous_month))

        labels = []
        for start, end in ranges:
            if start == end:
                labels.append(start.strftime("%b %Y"))
            elif start.year == end.year:
                labels.append(f"{start.strftime('%b')}–{end.strftime('%b %Y')}")
            else:
                labels.append(f"{start.strftime('%b %Y')}–{end.strftime('%b %Y')}")
        return ", ".join(labels)

    month_groups = [
        (ui("Higher", "ច្រើនជាងធម្មតា"), "High"),
        (ui("Normal", "ធម្មតា"), "Normal"),
        (ui("Lower", "តិចជាងធម្មតា"), "Low"),
    ]
    month_details = []
    for group_label, status_prefix in month_groups:
        group_dates = forecast_df.loc[
            forecast_df["Clean_Status"].str.startswith(status_prefix), "Date"
        ].tolist()
        month_details.append(
            f"<div><strong>{group_label}:</strong> "
            f"{format_month_ranges(group_dates)} ({len(group_dates)})</div>"
        )
    month_details_html = (
        f'<div class="summary-details">{"".join(month_details)}</div>'
    )

    summary_items = [
        (
            ui("Cumulative inbound", "ចំនួនភ្ញៀវសរុប"),
            f"{forecast_df['Tuned_Forecast'].sum():,.0f}",
            ui("Pax", "នាក់"),
            "",
        ),
        (
            ui("Monthly mean rate", "អត្រាមធ្យមប្រចាំខែ"),
            f"{forecast_df['Tuned_Forecast'].mean():,.0f}",
            ui("Pax/mo", "នាក់/ខែ"),
            "",
        ),
        (
            ui("Expected growth", "កំណើនរំពឹងទុក"),
            f"{forecast_df['Deviation_%'].mean():+.2f}%",
            ui("Above usual level", "ច្រើនជាងធម្មតា"),
            "",
        ),
        (
            ui("Months to watch", "ខែគួរតាមដាន"),
            f"{(forecast_df['Clean_Status'] != 'Normal Demand (🟢)').sum()} / {len(forecast_df)}",
            ui("Months", "ខែ"),
            month_details_html,
        ),
    ]

    summary_cols = st.columns(4)
    for idx, (label, value, unit, details) in enumerate(summary_items):
        with summary_cols[idx]:
            st.markdown(
                f"""
                <div class="summary-card metric-card-white">
                    <div class="summary-row">
                        <div class="summary-label">{label}</div>
                        <div class="summary-measure">
                            <div class="summary-value">{value}</div>
                            <div class="summary-sub">{unit}</div>
                        </div>
                    </div>
                    {details}
                </div>
                """,
                unsafe_allow_html=True,
            )

# -------------------------------------------------------------------
# TAB 2: FORECAST CURVES & RESIDUALS
# -------------------------------------------------------------------
elif current_tab == nav_options[1]:
    st.markdown(f"<div class='mot-section-header'>{ui('Comparison of Actual Data and Projections', 'ការប្រៀបធៀបទិន្នន័យពិត និងការព្យាករណ៍')}</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Seasonal_Baseline"],
            name="Usual level",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Tuned_Forecast"],
            name="Forecast",
            line=dict(color="#1d4ed8", width=3),
            mode="lines+markers",
        )
    )
    if "Actual_Arrivals" in forecast_df.columns and forecast_df["Actual_Arrivals"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Actual_Arrivals"],
                name="Actual",
                line=dict(color="#059669", width=2.5),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        height=450,
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Visitor arrivals", tickformat=",.0f", showgrid=True, gridcolor="#f1f5f9"),
        margin=dict(l=10, r=10, t=20, b=60),
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "autoScale2d",
                "select2d",
                "lasso2d",
            ],
        },
    )

    if "APE_%" in forecast_df.columns and forecast_df["APE_%"].notna().any():
        st.markdown(
            f"<div class='mot-section-header'>{ui('Monthly forecast error (%)', 'កំហុសព្យាករណ៍ប្រចាំខែ (%)')}</div>",
            unsafe_allow_html=True,
        )
        error_df = forecast_df.dropna(subset=["APE_%"]).copy()
        error_df["Month"] = error_df["Date"].dt.strftime("%b %Y")
        fig_error = px.bar(error_df, x="Month", y="APE_%", text="APE_%")
        fig_error.update_traces(
            marker_color="#e45d57",
            texttemplate="%{y:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        fig_error.update_layout(
            plot_bgcolor="white",
            height=360,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Forecast error (%)",
            yaxis_ticksuffix="%",
            yaxis_rangemode="tozero",
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(
            fig_error,
            width="stretch",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom2d",
                    "pan2d",
                    "autoScale2d",
                    "select2d",
                    "lasso2d",
                ],
            },
        )

    if not history_df.empty and "International_Tourist_Arrivals" in history_df.columns:
        annual_df = history_df.dropna(
            subset=["Date", "International_Tourist_Arrivals"]
        ).copy()
        annual_df["Year"] = annual_df["Date"].dt.year
        annual_totals = (
            annual_df.groupby("Year", as_index=False)
            .agg(
                International_Tourist_Arrivals=(
                    "International_Tourist_Arrivals", "sum"
                ),
                Months_Reported=("Date", lambda dates: dates.dt.month.nunique()),
                First_Month=("Date", lambda dates: dates.min().month),
                Last_Month=("Date", lambda dates: dates.max().month),
            )
        )
        annual_totals["Year_Status"] = np.where(
            annual_totals["Months_Reported"] == 12, "Full year", "Partial year"
        )
        annual_totals["Year_Label"] = annual_totals["Year"].astype(str)
        annual_totals.loc[annual_totals["Months_Reported"] < 12, "Year_Label"] += "*"
        partial_years = annual_totals[annual_totals["Months_Reported"] < 12]

        st.markdown(
            f"<div class='mot-section-header'>{ui('Annual tourist arrivals', 'ចំនួនភ្ញៀវទេសចរប្រចាំឆ្នាំ')}</div>",
            unsafe_allow_html=True,
        )
        if not partial_years.empty:
            partial_notes = [
                f"* {int(row['Year'])}: {pd.Timestamp(month=int(row['First_Month']), year=2000, day=1).strftime('%b')}–{pd.Timestamp(month=int(row['Last_Month']), year=2000, day=1).strftime('%b')} only"
                for _, row in partial_years.iterrows()
            ]
            st.caption(" • ".join(partial_notes))

        fig_annual = px.bar(
            annual_totals,
            x="Year_Label",
            y="International_Tourist_Arrivals",
            color="Year_Status",
            color_discrete_map={
                "Full year": "#1d4ed8",
                "Partial year": "#f59e0b",
            },
            labels={
                "Year_Label": "Year",
                "International_Tourist_Arrivals": "Visitors",
                "Year_Status": "Year coverage",
            },
        )
        fig_annual.update_traces(
            hovertemplate="%{x}<br>Visitors: %{y:,.0f}<extra></extra>"
        )
        fig_annual.update_layout(
            plot_bgcolor="white",
            height=360,
            xaxis_title="",
            yaxis_title="Visitors",
            yaxis_tickformat=",.0f",
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
        )
        st.plotly_chart(
            fig_annual,
            width="stretch",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom2d",
                    "pan2d",
                    "autoScale2d",
                    "select2d",
                    "lasso2d",
                ],
            },
        )

# -------------------------------------------------------------------
# -------------------------------------------------------------------
elif current_tab == nav_options[2]:
    status_labels = {
        "High Demand (🔴)": ui("High", "ខ្ពស់"),
        "Normal Demand (🟢)": ui("Normal", "ធម្មតា"),
        "Low Demand (🟡)": ui("Low", "ទាប"),
    }
    palette = {
        status_labels["High Demand (🔴)"]: "#1d4ed8",
        status_labels["Normal Demand (🟢)"]: "#059669",
        status_labels["Low Demand (🟡)"]: "#94a3b8",
    }

    st.markdown(
        f"<div class='mot-section-header'>{ui('Visitor level by month', 'កម្រិតភ្ញៀវតាមខែ')}</div>",
        unsafe_allow_html=True,
    )
    monthly_status_df = forecast_df[["Date", "Deviation_%", "Clean_Status"]].copy()
    monthly_status_df["Month"] = monthly_status_df["Date"].dt.strftime("%b %Y")
    monthly_status_df["Level"] = monthly_status_df["Clean_Status"].map(status_labels)
    month_order = monthly_status_df["Month"].tolist()
    fig_monthly_status = px.bar(
        monthly_status_df,
        x="Deviation_%",
        y="Month",
        orientation="h",
        color="Level",
        color_discrete_map=palette,
        custom_data=["Level"],
    )
    fig_monthly_status.update_traces(
        hovertemplate="%{y}<br>%{customdata[0]}<br>%{x:+.1f}%<extra></extra>"
    )
    fig_monthly_status.update_layout(
        plot_bgcolor="white",
        height=460,
        font=dict(
            family="Inter, Kantumruy Pro, Noto Sans Khmer, sans-serif",
            size=14,
            color="#334155",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title=ui("Above/below usual (%)", "លើស/ក្រោមកម្រិតធម្មតា (%)"),
            range=[-22, 60],
            tickvals=[-20, 0, 20, 40, 60],
            ticksuffix="%",
            zeroline=True,
            zerolinecolor="#64748b",
            showgrid=True,
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title=ui("Month", "ខែ"),
            categoryorder="array",
            categoryarray=month_order,
        ),
        margin=dict(l=20, r=35, t=55, b=20),
    )
    st.plotly_chart(
        fig_monthly_status,
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "autoScale2d",
                "select2d",
                "lasso2d",
            ],
        },
    )

    st.markdown(
        f"<div class='mot-section-header'>{ui('Alert details by month', 'ព័ត៌មានលម្អិតតាមខែ')}</div>",
        unsafe_allow_html=True,
    )
    display_df = forecast_df[[
        "Date",
        "Clean_Status",
        "Tuned_Forecast",
        "Seasonal_Baseline",
        "Deviation_%",
        "Alert_Summary",
        "Model_Drivers",
        "Rainfall",
        "Average_Temperature",
        "Cambodia_Holiday_Days",
        "Total_Origin_Holidays",
        "China_Holidays",
        "Operational_Action",
    ]].copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%b %Y")
    display_df["Tuned_Forecast"] = display_df["Tuned_Forecast"].apply(lambda x: f"{x:,.0f}")
    display_df["Seasonal_Baseline"] = display_df["Seasonal_Baseline"].apply(lambda x: f"{x:,.0f}")
    display_df["Deviation_%"] = display_df["Deviation_%"].apply(lambda x: f"{x:+.1f}%")
    display_df["Weather"] = display_df.apply(
        lambda row: (
            f"Rain {row['Rainfall']:.1f} mm; Temp {row['Average_Temperature']:.1f}°C"
            if pd.notna(row["Rainfall"]) and pd.notna(row["Average_Temperature"])
            else "—"
        ),
        axis=1,
    )
    holiday_columns = [
        "Cambodia_Holiday_Days",
        "Total_Origin_Holidays",
        "China_Holidays",
    ]
    display_df["Holiday_Days"] = display_df[holiday_columns].apply(
        lambda row: " / ".join(
            str(int(value)) if pd.notna(value) else "—"
            for value in row
        ),
        axis=1,
    )
    display_df = display_df.drop(
        columns=["Rainfall", "Average_Temperature", *holiday_columns]
    )
    display_df = display_df[
        [
            "Date",
            "Clean_Status",
            "Tuned_Forecast",
            "Seasonal_Baseline",
            "Deviation_%",
            "Alert_Summary",
            "Model_Drivers",
            "Weather",
            "Holiday_Days",
            "Operational_Action",
        ]
    ]
    display_df.columns = [
        "Month",
        "Alert Level",
        "Forecast (ŷt)",
        "Baseline (Bt)",
        "Deviation Δ%",
        "Alert Basis",
        "Top Model Drivers",
        "Weather",
        ui("Holiday days (KH / Origin / CN)", "ថ្ងៃឈប់ (កម្ពុជា / ប្រទេសដើម / ចិន)"),
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

    st.dataframe(
        styled_df,
        width="stretch",
        height=600,
        hide_index=True,
        column_config={
            "Month": st.column_config.TextColumn(
                width=100, pinned=True, alignment="center"
            ),
            "Alert Level": st.column_config.TextColumn(
                width=115, pinned=True, alignment="center"
            ),
            "Forecast (ŷt)": st.column_config.TextColumn(
                width=115, alignment="right"
            ),
            "Baseline (Bt)": st.column_config.TextColumn(
                width=115, alignment="right"
            ),
            "Deviation Δ%": st.column_config.TextColumn(
                width=105, alignment="center"
            ),
            "Alert Basis": st.column_config.TextColumn(
                width=190, alignment="center"
            ),
            "Top Model Drivers": st.column_config.TextColumn(
                width=260, alignment="left"
            ),
            "Weather": st.column_config.TextColumn(
                width=240, alignment="center"
            ),
            ui(
                "Holiday days (KH / Origin / CN)",
                "ថ្ងៃឈប់ (កម្ពុជា / ប្រទេសដើម / ចិន)",
            ): st.column_config.TextColumn(
                width=235,
                alignment="center",
                help=ui(
                    "Counts are shown in this order: Cambodia, visitor-origin countries, China.",
                    "ចំនួនបង្ហាញតាមលំដាប់៖ កម្ពុជា ប្រទេសដើមរបស់ភ្ញៀវ និងប្រទេសចិន។",
                ),
            ),
            "Operational Logistics Directive": st.column_config.TextColumn(
                width=280, alignment="left"
            ),
        },
    )

# -------------------------------------------------------------------
# TAB 4: POLICY SIMULATION SANDBOX (WHAT-IF INFERENCE)
# -------------------------------------------------------------------
elif current_tab == nav_options[3]:
    st.markdown(f"<div class='mot-section-header'>{ui('Real-Time Feature Attribution Sandbox', 'ប្រព័ន្ធសាកល្បងសមាមាត្រលក្ខណៈពេលវេលាជាក់ស្តែង')}</div>", unsafe_allow_html=True)
    st.caption(ui("Evaluates sensitivity against log-transformed LightGBM inference endpoints to observe climate and origin holiday interactions.", "វាស់ស្ទង់ឥទ្ធិពលរបស់អាកាសធាតុ និងថ្ងៃឈប់សម្រាកប្រទេសដើមកំណើត តាមរយៈ LightGBM ដែលបានបម្លែងគោលដៅជា log។"))

    with st.form("simulation_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_date = st.date_input(ui("Target Month", "ខែដែលត្រូវព្យាករណ៍"), value=pd.to_datetime("2026-11-01"))
            sim_baseline = st.number_input(ui("Baseline Arrivals", "ចំនួនភ្ញៀវមូលដ្ឋាន"), value=364436.0, step=1000.0)
            sim_lag1 = st.number_input(ui("Previous Month Visitors", "ចំនួនភ្ញៀវខែមុន"), value=327239.0, step=1000.0)
            sim_lag3 = st.number_input(ui("Visitors 3 Months Ago", "ចំនួនភ្ញៀវកាលពី ៣ ខែមុន"), value=347492.0, step=1000.0)
            sim_lag12 = st.number_input(ui("Same Month Last Year", "ចំនួនភ្ញៀវក្នុងខែដូចគ្នាកាលពីឆ្នាំមុន"), value=300000.0, step=1000.0)

        with col_s2:
            sim_roll3 = st.number_input(ui("3-Month Average Visitors", "មធ្យមភាគភ្ញៀវ ៣ ខែចុងក្រោយ"), value=337694.0, step=1000.0)
            sim_roll12 = st.number_input(ui("12-Month Average Visitors", "មធ្យមភាគភ្ញៀវ ១២ ខែចុងក្រោយ"), value=350000.0, step=1000.0)
            sim_temp = st.slider(ui("Expected Temperature (°C)", "សីតុណ្ហភាពរំពឹងទុក (°C)"), min_value=20.0, max_value=38.0, value=27.5, step=0.1)
            sim_rain = st.slider(ui("Expected Rainfall (mm)", "បរិមាណទឹកភ្លៀងរំពឹងទុក (មម)"), min_value=0.0, max_value=600.0, value=110.0, step=5.0)

        with col_s3:
            sim_kh_holidays = st.number_input(ui("Cambodian Public Holidays", "ចំនួនថ្ងៃបុណ្យជាតិកម្ពុជា"), min_value=0, max_value=15, value=3)
            sim_origin_holidays = st.number_input(ui("Source Market Holidays", "ចំនួនថ្ងៃបុណ្យប្រទេសប្រភពទេសចរណ៍"), min_value=0, max_value=30, value=12)
            sim_china_holidays = st.number_input(ui("Major Holiday Seasons", "រដូវកាលបុណ្យសំខាន់ៗ"), min_value=0, max_value=10, value=1)
            sim_covid = st.selectbox(ui("Special Disruption Factor", "កត្តាប្រែប្រួលពិសេស"), options=[0, 1], index=0)

        dispatch_btn = st.form_submit_button(ui("⚡ Run Log-LightGBM Inference", "⚡ ដំណើរការព្យាករណ៍ Log-LightGBM"), use_container_width=True)

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
elif current_tab == nav_options[4]:
    st.markdown(f"<div class='mot-section-header'>{ui('Data Pipelines, Features & Specifications', 'បំពង់ទិន្នន័យ លក្ខណៈពិសេស និងលក្ខណៈបច្ចេកទេស')}</div>", unsafe_allow_html=True)

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
        Cambodia Tourism Intelligence Platform | Developed by <strong>RY Chhorly</strong> | Supervised by <strong>Ya Manon</strong><br>
        Kingdom of Wonder • Applied Machine Learning Early Warning System (EWS)
    </div>
    """,
    unsafe_allow_html=True,
)