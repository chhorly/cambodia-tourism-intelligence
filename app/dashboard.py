from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.base import clone

from model_training import prepare_training_data
from app.scenario_transactions import (
    SCENARIO_TRANSACTIONS_PATH,
    append_scenario_transaction,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Cambodia Tourism Intelligence",
    page_icon=":material/travel_explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Enterprise Government MLOps Theme (Clean & Responsive)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Kantumruy+Pro:wght@400;500;600;700&family=Noto+Sans+Khmer:wght@400;500;600;700;800&display=swap');

    :root {
        --dashboard-background: #edf5ff;
        --dashboard-surface: #ffffff;
        --dashboard-text: #142b4a;
        --dashboard-heading: #142f56;
        --dashboard-muted: #667b94;
        --dashboard-border: #dce7f3;
        --dashboard-primary: #173b68;
        --dashboard-accent: #e8b33c;
        --dashboard-radius: 12px;
        --dashboard-shadow: 0 2px 10px rgba(31, 62, 98, 0.04);
        --dashboard-font: 'Inter', 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif;
    }

    * {
        font-family: var(--dashboard-font);
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

    .stApp {
        background: #edf5ff;
        color: #142b4a;
    }

    .block-container {
        max-width: 1480px;
        padding: 1.5rem 1.6rem 2.2rem !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dce7f3;
    }

    .gov-navbar {
        min-height: 62px;
        padding: 0.65rem 1rem;
        margin-bottom: 1.15rem;
        border-color: #dce7f3;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(31, 62, 98, 0.04);
    }

    .gov-emblem-badge {
        background: #ffffff;
        color: #17365e;
        border: 1px solid #dce7f3;
        box-shadow: none;
    }

    .gov-title-khmer {
        color: #17365e;
    }

    .gov-title-en {
        color: #6c7e93;
    }

    .mlops-status-pill {
        background: #eef6ff;
        border-color: #c9def6;
        color: #2563a9;
    }

    .left-selector {
        height: auto;
        min-height: 48px;
        background: #f7faff;
        border-color: #e2eaf4;
        box-shadow: none;
    }

    .left-selector-icon {
        background: linear-gradient(180deg, #2563eb 0%, #123e78 100%);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        min-height: 44px;
        padding: 0.55rem 0.7rem;
        background: #ffffff;
        border-color: transparent;
        border-radius: 9px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: #eaf3ff;
        border-color: #d5e6fb;
        box-shadow: inset 3px 0 #e8b33c;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
        color: #1c5aa6;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        accent-color: #2563eb;
    }

    .page-intro {
        margin: 0 0 1.15rem;
    }

    .page-intro-title {
        margin: 0;
        color: #142f56;
        font-family: 'Kantumruy Pro', 'Noto Sans Khmer', 'Inter', sans-serif;
        font-size: clamp(1.7rem, 3vw, 2.25rem);
        font-weight: 800;
        line-height: 1.25;
    }

    .page-intro-subtitle {
        margin-top: 0.15rem;
        color: #54708e;
        font-size: 1.02rem;
        font-weight: 600;
    }

    .mot-section-header::after {
        background: #e8b33c;
    }

    .trust-card {
        background: #ffffff;
        border: 1px solid #dce7f3;
        border-top: 3px solid #4d8fe8;
        border-radius: 12px;
        color: #142b4a;
        box-shadow: 0 2px 10px rgba(31, 62, 98, 0.04);
    }

    .trust-card-green {
        border-top-color: #26a69a;
    }

    .trust-card-warning,
    .trust-card-caution,
    .trust-card-offline {
        border-top-color: #e8b33c;
    }

    .trust-card .ds-card-title {
        color: #667b94;
    }

    .trust-card .ds-card-val {
        color: #142f56;
    }

    .trust-card .ds-card-sub {
        color: #667b94;
    }

    .trust-card .metric-value-warning,
    .trust-card .metric-value-offline {
        color: #a86b00 !important;
    }

    .trust-card .metric-value-ok {
        color: #147c72 !important;
    }

    .summary-card,
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7f3;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(31, 62, 98, 0.04);
    }

    .metric-card-white {
        border-top: 3px solid #e8b33c;
    }

    .summary-sub {
        color: #66809d;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #173b68;
        box-shadow: none;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #245b9b;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #dce7f3;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(31, 62, 98, 0.035);
    }

    .gov-footer {
        border-color: #dce7f3;
        color: #70839a;
    }

    section[data-testid="stSidebar"] {
        width: 16.5rem !important;
        min-width: 16.5rem !important;
        background: #ffffff;
        border-right: 1px solid #dce7f3;
    }

    [data-testid="stHeader"] {
        display: none;
    }

    .block-container {
        padding-top: 0.25rem !important;
    }

    [data-testid="stSidebarHeader"] {
        display: none;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }

    .sidebar-brand {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 0.2rem 0.25rem 1rem;
        margin-bottom: 0.7rem;
        border-bottom: 1px solid #e7edf5;
        color: #17365e;
    }

    .sidebar-logo {
        width: 76px;
        height: 54px;
        margin-bottom: 0.2rem;
        color: #17365e;
    }

    .sidebar-brand-name {
        color: #17365e;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.035em;
        line-height: 1.35;
    }

    .sidebar-brand-caption {
        color: #73849a;
        font-size: 0.76rem;
        line-height: 1.35;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.22rem;
        margin-top: 0.3rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        position: relative;
        min-height: 42px;
        padding: 0.55rem 0.65rem;
        border: 1px solid transparent;
        border-radius: 8px;
        color: #40536b;
        cursor: pointer;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: inherit;
        font-size: 0.92rem;
        line-height: 1.35;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        position: absolute;
        width: 1px;
        height: 1px;
        margin: 0;
        opacity: 0;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #eaf3ff;
        border-color: #d5e6fb;
        box-shadow: inset 3px 0 #e8b33c;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #1c5aa6;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:focus-visible) {
        outline: 2px solid #2563eb;
        outline-offset: 2px;
    }

    .sidebar-prototype {
        margin: 1rem 0.3rem 0;
        color: #71839a;
        font-size: 0.72rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) {
        align-items: center;
        min-height: 2.9rem;
        padding: 0.1rem 0.8rem 0;
        margin: -1.3rem -0.8rem 0.55rem;
        background: #ffffff;
        border-bottom: 1px solid #dce7f3;
    }

    .topbar-title {
        color: #17365e;
        font-size: 1.12rem;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [data-testid="stRadio"] {
        display: flex;
        justify-content: flex-end;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [role="radiogroup"] {
        justify-content: flex-end;
        gap: 0.25rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [role="radiogroup"] label {
        min-height: auto;
        padding: 0.2rem 0.45rem;
        border: 0;
        border-radius: 4px;
        color: #718096;
        font-size: 0.8rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [role="radiogroup"] label p {
        color: inherit;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [role="radiogroup"] label:has(input:checked) p {
        color: #17365e;
        font-weight: 700;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [role="radiogroup"] label + label {
        border-left: 1px solid #dce7f3;
        border-radius: 0;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: flex-end;
    }

    div[data-testid="stHorizontalBlock"]:has(.topbar-title) [data-testid="stSegmentedControl"] button {
        min-height: 2rem;
        padding: 0.2rem 0.45rem;
        color: #718096;
        font-size: 0.8rem;
    }

    .page-intro {
        margin-bottom: 0.85rem;
    }

    .page-intro-title {
        color: #142f56;
        font-size: clamp(1.8rem, 3.1vw, 2.35rem);
        line-height: 1.35;
    }

    .page-intro-subtitle {
        color: #243b5a;
        font-size: 1rem;
        font-weight: 500;
    }

    .page-intro > div:last-child {
        margin-top: 0.12rem;
        color: #58708a;
        font-size: 0.9rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(31, 62, 98, 0.035);
    }

    .st-key-scenario-input-card,
    .st-key-scenario-result-card {
        background: #ffffff !important;
        border: 1px solid #dce7f3;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(31, 62, 98, 0.035);
        padding: 0.75rem;
    }

    .st-key-scenario-input-card [data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    .st-key-scenario-result-card [data-testid="stMetric"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        padding: 0.35rem 0.2rem;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    .stFormSubmitButton > button {
        min-height: 2.65rem;
        border-radius: 7px;
        font-weight: 650;
    }

    .stFormSubmitButton button[kind="primary"] {
        background: #102f5e;
        color: #ffffff;
    }

    .stFormSubmitButton button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #9aacc1;
        color: #17365e;
        box-shadow: none;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: min(16.5rem, 85vw) !important;
            min-width: min(16.5rem, 85vw) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.topbar-title) {
            margin-right: -0.3rem;
            margin-left: -0.3rem;
            padding-right: 0.3rem;
            padding-left: 0.3rem;
        }
    }

    .stApp {
        background: var(--dashboard-background);
        color: var(--dashboard-text);
    }

    .block-container {
        color: var(--dashboard-text);
    }

    .page-intro,
    .mot-section-header {
        text-align: left;
    }

    .page-intro-title,
    h1, h2, h3, h4, h5, h6,
    .topbar-title {
        color: var(--dashboard-heading);
        font-family: var(--dashboard-font) !important;
    }

    .page-intro-subtitle,
    .page-intro > div:last-child,
    [data-testid="stCaptionContainer"],
    .gov-footer {
        color: var(--dashboard-muted);
    }

    .mot-section-header {
        display: block;
        margin: 1.1rem 0 0.8rem;
        color: var(--dashboard-heading);
        font-family: var(--dashboard-font);
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.4;
    }

    .mot-section-header::after {
        background: var(--dashboard-accent);
    }

    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stHeading"]),
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMetric"]),
    [data-testid="stMetric"],
    .summary-card,
    .st-key-scenario-input-card,
    .st-key-scenario-result-card,
    [data-testid="stExpander"] {
        background: var(--dashboard-surface);
        border: 1px solid var(--dashboard-border);
        border-radius: var(--dashboard-radius);
        box-shadow: var(--dashboard-shadow);
    }

    [data-testid="stMetric"] {
        padding: 0.85rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--dashboard-muted);
        font-family: var(--dashboard-font) !important;
        font-size: 0.78rem;
        letter-spacing: 0;
        text-transform: none;
    }

    [data-testid="stMetricValue"] {
        color: var(--dashboard-heading);
        font-family: var(--dashboard-font) !important;
    }

    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        background: var(--dashboard-surface);
        border-color: var(--dashboard-border);
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        font-family: var(--dashboard-font) !important;
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

model_artifact = None
model_load_error = None
if MODEL_PATH.exists():
    try:
        model_artifact = joblib.load(MODEL_PATH)
        if not isinstance(model_artifact, dict) or "model" not in model_artifact:
            raise ValueError("The model artifact does not contain a supported model.")
    except Exception as error:
        model_load_error = error
else:
    model_load_error = FileNotFoundError(f"Trained model artifact not found: {MODEL_PATH}")

if model_load_error is not None:
    st.error(f"Could not load the trained tourism model: {model_load_error}")
    st.stop()

holdout_mape = None
holdout_r2 = None
holdout_months = 0
holdout_training_count = 0
holdout_mape_text = "N/A"
holdout_r2_text = "N/A"

is_khmer = st.session_state.get("language", "ខ្មែរ") == "ខ្មែរ"


def ui(english, khmer):
    return khmer if is_khmer else english


CHART_COLORS = {
    "actual": "#173B68",
    "forecast": "#2563EB",
    "baseline": "#94A3B8",
    "high": "#C2414B",
    "normal": "#0F8B8D",
    "low": "#D97706",
    "error": "#D97706",
    "partial": "#94A3B8",
}
CHART_FONT_FAMILY = "Inter, Kantumruy Pro, Noto Sans Khmer, sans-serif"
CHART_TEXT_COLOR = "#26364A"
CHART_GRID_COLOR = "#E2E8F0"


def style_chart(figure):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family=CHART_FONT_FAMILY, size=13, color=CHART_TEXT_COLOR),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=CHART_GRID_COLOR,
            font=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
        ),
    )
    figure.update_xaxes(
        tickfont=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
        title_font=dict(family=CHART_FONT_FAMILY, size=13, color=CHART_TEXT_COLOR),
        linecolor=CHART_GRID_COLOR,
        automargin=True,
    )
    figure.update_yaxes(
        tickfont=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
        title_font=dict(family=CHART_FONT_FAMILY, size=13, color=CHART_TEXT_COLOR),
        linecolor=CHART_GRID_COLOR,
        gridcolor=CHART_GRID_COLOR,
        automargin=True,
    )
    return figure


def render_chart_insight(english, khmer):
    st.caption(
        ui(f"Key insight · {english}", f"ចំណុចសំខាន់ · {khmer}")
    )


def forecast_accuracy_insight(frame):
    error_rows = frame.dropna(subset=["APE_%"])
    if error_rows.empty:
        return (
            "Actual arrivals are unavailable, so forecast accuracy cannot be assessed.",
            "មិនមានចំនួនភ្ញៀវពិត ដូច្នេះមិនអាចវាយតម្លៃភាពត្រឹមត្រូវនៃការព្យាករណ៍បានទេ។",
        )

    mean_error = error_rows["APE_%"].mean()
    worst_error = error_rows.loc[error_rows["APE_%"].idxmax()]
    worst_month = worst_error["Date"].strftime("%b %Y")
    return (
        f"Mean absolute percentage error is {mean_error:.1f}%; the largest monthly error "
        f"was {worst_error['APE_%']:.1f}% in {worst_month}.",
        f"កំហុសភាគរយដាច់ខាតមធ្យមគឺ {mean_error:.1f}%; កំហុសប្រចាំខែខ្ពស់បំផុត "
        f"{worst_error['APE_%']:.1f}% នៅ {worst_month}។",
    )


def forecast_alignment_insight(frame):
    actual_rows = frame.dropna(
        subset=["Actual_Arrivals", "Tuned_Forecast", "APE_%"]
    )
    if actual_rows.empty:
        return (
            "Actual arrivals are unavailable, so forecast alignment cannot be assessed.",
            "មិនមានចំនួនភ្ញៀវពិត ដូច្នេះមិនអាចវាយតម្លៃភាពស្របគ្នានៃការព្យាករណ៍បានទេ។",
        )

    overestimated = int(
        (actual_rows["Tuned_Forecast"] > actual_rows["Actual_Arrivals"]).sum()
    )
    worst_error = actual_rows.loc[actual_rows["APE_%"].idxmax()]
    worst_month = worst_error["Date"].strftime("%b %Y")
    return (
        f"The model overestimated in {overestimated} of {len(actual_rows)} months; "
        f"its largest miss was {worst_error['APE_%']:.1f}% in {worst_month}.",
        f"ម៉ូដែលប៉ាន់ស្មានលើសចំនួនពិតក្នុង {overestimated} ក្នុងចំណោម "
        f"{len(actual_rows)} ខែ; កំហុសខ្ពស់បំផុត {worst_error['APE_%']:.1f}% នៅ {worst_month}។",
    )


def forecast_error_spread_insight(frame):
    error_rows = frame.dropna(subset=["APE_%"])
    if error_rows.empty:
        return forecast_accuracy_insight(frame)

    best_error = error_rows.loc[error_rows["APE_%"].idxmin()]
    worst_error = error_rows.loc[error_rows["APE_%"].idxmax()]
    median_error = error_rows["APE_%"].median()
    best_month = best_error["Date"].strftime("%b %Y")
    worst_month = worst_error["Date"].strftime("%b %Y")
    return (
        f"Median monthly error is {median_error:.1f}%; errors range from "
        f"{best_error['APE_%']:.1f}% in {best_month} to "
        f"{worst_error['APE_%']:.1f}% in {worst_month}.",
        f"កំហុសប្រចាំខែកណ្តាលគឺ {median_error:.1f}%; កំហុសចាប់ពី "
        f"{best_error['APE_%']:.1f}% នៅ {best_month} ដល់ "
        f"{worst_error['APE_%']:.1f}% នៅ {worst_month}។",
    )


@st.cache_data
def load_datasets(forecast_path: Path, history_path: Path):
    if not history_path.exists():
        raise FileNotFoundError(f"Training history CSV not found: {history_path}")
    if not isinstance(model_artifact, dict):
        raise ValueError("The dashboard requires a model artifact with feature metadata.")

    history = pd.read_csv(history_path, parse_dates=["Date"]).sort_values("Date")
    history.columns = history.columns.str.strip()
    required_history_columns = {"Date", "International_Tourist_Arrivals"}
    missing_history_columns = required_history_columns.difference(history.columns)
    if missing_history_columns:
        raise ValueError(
            "Training history is missing required columns: "
            + ", ".join(sorted(missing_history_columns))
        )

    model_features = model_artifact.get("features", [])
    if not model_features:
        raise ValueError("The trained model artifact does not declare its input features.")

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

    feature_frame, target, sample_weights = prepare_training_data(history_path)
    if list(feature_frame.columns) != list(model_features):
        raise ValueError(
            "The training feature contract does not match the bundled model. "
            "Retrain the model from the current training CSV."
        )
    holdout_count = min(
        int(model_artifact.get("metrics", {}).get("holdout_months", 12)),
        len(feature_frame) - 1,
    )
    if holdout_count < 1:
        raise ValueError("Not enough complete training rows to create a chronological holdout.")
    split_at = len(feature_frame) - holdout_count
    holdout_model = clone(model_artifact["model"])
    train_target = target.iloc[:split_at]
    target_transform = model_artifact.get("target_transform", "none")
    if target_transform == "log1p":
        train_target = np.log1p(train_target)
    holdout_model.fit(
        feature_frame.iloc[:split_at],
        train_target,
        sample_weight=sample_weights.iloc[:split_at],
    )
    predictions = holdout_model.predict(feature_frame.iloc[split_at:])
    if target_transform == "log1p":
        predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0.0)

    holdout_rows = history.loc[feature_frame.index].iloc[split_at:].copy()
    forecast = holdout_rows[
        ["Date", "Month_Number", "International_Tourist_Arrivals"]
    ].rename(columns={"International_Tourist_Arrivals": "Actual_Arrivals"})
    forecast["Tuned_Forecast"] = predictions

    if forecast_path.exists():
        ews = pd.read_csv(forecast_path, parse_dates=["Date"])
        ews.columns = ews.columns.str.strip()
        if "Seasonal_Baseline" in ews.columns:
            ews["Seasonal_Baseline"] = pd.to_numeric(
                ews["Seasonal_Baseline"].astype(str).str.replace(",", ""),
                errors="coerce",
            )
            forecast = forecast.merge(
                ews[["Date", "Seasonal_Baseline"]],
                on="Date",
                how="left",
                validate="one_to_one",
            )
    if "Seasonal_Baseline" not in forecast.columns:
        forecast["Seasonal_Baseline"] = np.nan
    if "Lag_12" in holdout_rows.columns:
        forecast["Seasonal_Baseline"] = forecast["Seasonal_Baseline"].fillna(
            pd.Series(holdout_rows["Lag_12"].to_numpy(), index=forecast.index)
        )
    if forecast["Seasonal_Baseline"].isna().any() or (forecast["Seasonal_Baseline"] <= 0).any():
        raise ValueError("No valid seasonal baseline is available for one or more holdout months.")

    forecast["Deviation_%"] = (
        (forecast["Tuned_Forecast"] - forecast["Seasonal_Baseline"])
        / forecast["Seasonal_Baseline"]
        * 100
    )
    forecast["Error"] = forecast["Actual_Arrivals"] - forecast["Tuned_Forecast"]
    forecast["APE_%"] = (
        forecast["Error"].abs() / forecast["Actual_Arrivals"].replace(0, np.nan) * 100
    )

    def assign_status(deviation):
        if deviation > 15:
            return "High Demand (🔴)"
        if deviation < -15:
            return "Low Demand (🟡)"
        return "Normal Demand (🟢)"

    forecast["Clean_Status"] = forecast["Deviation_%"].apply(assign_status)
    forecast["Alert_Status"] = forecast["Clean_Status"]

    def assign_action(status):
        if "High" in status:
            return "Surge border checkpoint staff at PNH/SAI; inspect hotel capacities; expand inter-provincial shuttles."
        if "Low" in status:
            return "Deploy targeted marketing campaigns; subsidize bundle air-hotel rates; offer promotional incentives."
        return "Maintain baseline resource allocations and scheduled off-peak maintenance schedules."

    forecast["Operational_Action"] = forecast["Clean_Status"].apply(assign_action)

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
            if column in holdout_rows.columns and column != "Month_Number"
        )
    )
    forecast = forecast.merge(
        holdout_rows[["Date", *available_context]],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    contribution_columns = {}
    if model_features and all(feature in forecast.columns for feature in model_features):
        model_inputs = forecast[model_features].apply(pd.to_numeric, errors="coerce")
        if model_inputs.notna().all().all():
            contributions = holdout_model.predict(
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

holdout_actuals = forecast_df["Actual_Arrivals"].to_numpy()
holdout_predictions = forecast_df["Tuned_Forecast"].to_numpy()
holdout_mape = float(
    np.mean(np.abs(holdout_actuals - holdout_predictions) / holdout_actuals)
)
holdout_r2 = float(
    1
    - np.sum((holdout_actuals - holdout_predictions) ** 2)
    / np.sum((holdout_actuals - holdout_actuals.mean()) ** 2)
)
holdout_months = len(forecast_df)
valid_training_rows = history_df.dropna(
    subset=model_artifact.get("features", [])
    + ["International_Tourist_Arrivals"]
)
holdout_training_count = len(valid_training_rows) - holdout_months
holdout_mape_text = f"{holdout_mape * 100:.2f}%"
holdout_r2_text = f"{holdout_r2 * 100:.2f}%"

# 4. Check Microservice Health
API_URL = "http://127.0.0.1:8000"
api_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.2)
    if health_resp.status_code == 200:
        api_online = True
except Exception:
    api_online = False

# 5. Shared application shell and navigation
api_tag = (
    ui("Service: live", "ប្រព័ន្ធ៖ ដំណើរការ")
    if api_online
    else ui("Service: local data", "ប្រព័ន្ធ៖ ទិន្នន័យមូលដ្ឋាន")
)
nav_options = [
    "overview",
    "forecasts",
    "alerts",
    "scenario",
    "technical",
]
nav_labels = {
    "overview": ui(":material/home: Overview", ":material/home: សង្ខេបសំខាន់ៗ"),
    "forecasts": ui(":material/bar_chart: Forecasts", ":material/bar_chart: និន្នាការព្យាករណ៍"),
    "alerts": ui(":material/notifications: Alerts & Actions", ":material/notifications: ការព្រមាន និងសកម្មភាព"),
    "scenario": ui(":material/show_chart: Scenario Explorer", ":material/show_chart: សាកល្បងគោលនយោបាយ"),
    "technical": ui(":material/settings: Technical", ":material/settings: ព័ត៌មានបច្ចេកទេស"),
}

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <svg class="sidebar-logo" viewBox="0 0 128 78" role="img" aria-label="Angkor Wat emblem">
                <g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                    <path stroke-width="2.5" d="M10 62h108M17 57h94M23 49h82M28 45h72M35 38h58M42 31h44M49 23h30M56 15h16M62 7h4"/>
                    <path stroke-width="2.5" d="M30 45V31l8-8 8 8v14M52 45V23l12-16 12 16v22M82 45V31l8-8 8 8v14"/>
                    <path stroke-width="1.7" d="M35 34h6m-6 5h6m15-11h6m5-7h6m5 7h6m-24 9h22m-22 7h22m13-10h6m-6 5h6"/>
                    <path stroke-width="2.5" d="M23 49v8m82-8v8M46 45v12m36-12v12M60 45v12m8-12v12"/>
                </g>
            </svg>
            <div class="sidebar-brand-name">CAMBODIA TOURISM</div>
            <div class="sidebar-brand-caption">Intelligence Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    current_tab = st.radio(
        ui("Select Platform Module", "ជ្រើសរើសផ្នែកប្រព័ន្ធ"),
        options=nav_options,
        index=0,
        format_func=lambda option: nav_labels[option],
        horizontal=False,
        label_visibility="collapsed",
        key="selected_module",
    )
    st.markdown(
        f'<div class="sidebar-prototype">{ui("Research prototype", "គំរូស្រាវជ្រាវ")}</div>',
        unsafe_allow_html=True,
    )

title_column, language_column = st.columns([5, 1])
with title_column:
    st.markdown('<div class="topbar-title">Tourism Intelligence</div>', unsafe_allow_html=True)
with language_column:
    language = st.segmented_control(
        "Language",
        options=["ខ្មែរ", "EN"],
        default="ខ្មែរ",
        label_visibility="collapsed",
        key="language",
        width="content",
    )
is_khmer = language == "ខ្មែរ"

page_copy = {
    "overview": (
        "ទិដ្ឋភាពទូទៅនៃវិស័យទេសចរណ៍",
        "Tourism overview",
        "Visitor demand, forecast quality and operational priorities.",
    ),
    "forecasts": (
        "និន្នាការព្យាករណ៍",
        "Forecast performance",
        "Monthly arrivals compared with actuals and the usual seasonal level.",
    ),
    "alerts": (
        "សញ្ញាតម្រូវការ និងសកម្មភាព",
        "Demand signals & recommended actions",
        "Review forecast deviations and the months that need attention.",
    ),
    "scenario": (
        "សាកល្បងសេណារីយ៉ូ",
        "Scenario explorer",
        "Explore model sensitivity to weather and calendar assumptions.",
    ),
    "technical": (
        "ព័ត៌មានបច្ចេកទេស",
        "Technical details",
        "Data sources, model inputs and platform health.",
    ),
}
page_title_kh, page_title_en, page_description = page_copy[current_tab]
st.markdown(
    f"""
    <div class="page-intro">
        <div class="page-intro-title">{page_title_kh}</div>
        <div class="page-intro-subtitle">{page_title_en}</div>
        <div>{page_description}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# TAB 1: EXECUTIVE KPIS & MODEL HEALTH
# -------------------------------------------------------------------
if current_tab == "overview":
    period_start = forecast_df["Date"].min().strftime("%b %Y")
    period_end = forecast_df["Date"].max().strftime("%b %Y")
    if forecast_df["Date"].max().date() < pd.Timestamp.now().date():
        st.warning(
            ui(
                f"Historical forecast window ({period_start} – {period_end}) — refresh data before operational use.",
                f"រយៈពេលព្យាករណ៍នេះជាទិន្នន័យប្រវត្តិសាស្ត្រ ({period_start} – {period_end}) — សូមធ្វើបច្ចុប្បន្នភាពទិន្នន័យមុនពេលប្រើប្រាស់។",
            )
        )
    period_column, sample_column = st.columns([4, 1])
    with period_column:
        st.caption(
            ui(
                f"Validation period · {period_start} – {period_end}",
                f"រយៈពេលផ្ទៀងផ្ទាត់ · {period_start} – {period_end}",
            )
        )
    with sample_column:
        st.caption(ui("Historical sample", "ទិន្នន័យប្រវត្តិសាស្ត្រ"))
    baseline_total = forecast_df["Seasonal_Baseline"].sum()
    above_baseline_pct = (
        (forecast_df["Tuned_Forecast"].sum() - baseline_total) / baseline_total * 100
        if baseline_total
        else 0.0
    )
    months_to_review = int(
        (forecast_df["Clean_Status"] != "Normal Demand (🟢)").sum()
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        ui("Forecast visitors", "ភ្ញៀវទេសចរព្យាករណ៍"),
        f"{forecast_df['Tuned_Forecast'].sum() / 1_000_000:.2f}M",
        f"{period_start} – {period_end}",
        border=True,
    )
    metric_columns[1].metric(
        ui("Monthly average", "មធ្យមភាគប្រចាំខែ"),
        f"{forecast_df['Tuned_Forecast'].mean() / 1_000:.0f}K",
        ui("Forecast visitors", "ភ្ញៀវទេសចរព្យាករណ៍"),
        border=True,
    )
    metric_columns[2].metric(
        ui("Above baseline", "លើសកម្រិតមូលដ្ឋាន"),
        f"{above_baseline_pct:+.2f}%",
        ui("Weighted across the forecast period", "គណនាតាមរយៈពេលព្យាករណ៍"),
        border=True,
    )
    metric_columns[3].metric(
        ui("Months to review", "ខែត្រូវពិនិត្យ"),
        f"{months_to_review} / {len(forecast_df)}",
        ui(
            f"{(forecast_df['Clean_Status'] == 'High Demand (🔴)').sum()} high · {(forecast_df['Clean_Status'] == 'Low Demand (🟡)').sum()} low",
            f"{(forecast_df['Clean_Status'] == 'High Demand (🔴)').sum()} ខ្ពស់ · {(forecast_df['Clean_Status'] == 'Low Demand (🟡)').sum()} ទាប",
        ),
        border=True,
    )

    latest_forecast_date = forecast_df["Date"].max()
    overview_figure = go.Figure()
    if forecast_df["Actual_Arrivals"].notna().any():
        actual_values_k = forecast_df["Actual_Arrivals"] / 1_000
        overview_figure.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=actual_values_k,
                name=ui("Actual", "ចំនួនពិត"),
                mode="lines+markers",
                line=dict(color=CHART_COLORS["actual"], width=2.5),
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
            )
        )
    forecast_values_k = forecast_df["Tuned_Forecast"] / 1_000
    overview_figure.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_values_k,
            name=ui("Forecast", "ការព្យាករណ៍"),
            mode="lines+markers",
            line=dict(color=CHART_COLORS["forecast"], width=3, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
        )
    )
    baseline_values_k = forecast_df["Seasonal_Baseline"] / 1_000
    overview_figure.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=baseline_values_k,
            name=ui("Usual level", "កម្រិតធម្មតា"),
            mode="lines",
            line=dict(color=CHART_COLORS["baseline"], width=2, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
        )
    )
    if forecast_df["Actual_Arrivals"].notna().any():
        last_actual = forecast_df.loc[
            forecast_df["Actual_Arrivals"].notna(), "Actual_Arrivals"
        ].iloc[-1] / 1_000
        overview_figure.add_annotation(
            x=latest_forecast_date,
            y=last_actual,
            text=ui("Actual", "ចំនួនពិត"),
            xshift=28,
            showarrow=False,
            font=dict(color=CHART_COLORS["actual"], size=11),
        )
    for series, label, color, shift in [
        (forecast_values_k, ui("Forecast", "ការព្យាករណ៍"), CHART_COLORS["forecast"], 34),
        (baseline_values_k, ui("Usual level", "កម្រិតធម្មតា"), CHART_COLORS["baseline"], 45),
    ]:
        overview_figure.add_annotation(
            x=latest_forecast_date,
            y=series.iloc[-1],
            text=label,
            xshift=shift,
            showarrow=False,
            font=dict(color=color, size=11),
        )
    overview_figure.update_layout(
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="",
            showgrid=False,
            range=[
                forecast_df["Date"].min(),
                latest_forecast_date + pd.DateOffset(days=18),
            ],
        ),
        yaxis=dict(
            title=ui("Visitor arrivals (thousands)", "ចំនួនភ្ញៀវ (ពាន់នាក់)"),
            tickformat=",.0f",
            showgrid=True,
            gridcolor=CHART_GRID_COLOR,
        ),
        margin=dict(l=10, r=65, t=25, b=35),
    )
    style_chart(overview_figure)

    chart_column, attention_column = st.columns([1.8, 1])
    with chart_column:
        with st.container(border=True):
            st.subheader(ui("Actual vs forecast", "ចំនួនពិត និងការព្យាករណ៍"))
            st.caption(
                f"{period_start} – {period_end} · "
                f"{ui('Monthly international visitor arrivals · thousands', 'ចំនួនភ្ញៀវអន្តរជាតិប្រចាំខែ · ពាន់នាក់')}"
            )
            st.plotly_chart(
                overview_figure,
                width="stretch",
                config={"displaylogo": False},
            )
            render_chart_insight(*forecast_alignment_insight(forecast_df))

    with attention_column:
        with st.container(border=True):
            st.subheader(ui("What needs attention", "ចំណុចត្រូវយកចិត្តទុកដាក់"))
            attention_df = (
                forecast_df.loc[
                    forecast_df["Clean_Status"] != "Normal Demand (🟢)"
                ]
                .assign(Absolute_Deviation=lambda frame: frame["Deviation_%"].abs())
                .nlargest(2, "Absolute_Deviation")
            )
            if attention_df.empty:
                st.success(ui("No demand alerts in this period.", "មិនមានការព្រមានតម្រូវការក្នុងរយៈពេលនេះទេ។"))
            else:
                for _, alert in attention_df.iterrows():
                    st.markdown(
                        f"**{alert['Date'].strftime('%b %Y')} · {alert['Clean_Status']}**"
                    )
                    st.caption(
                        f"{alert['Deviation_%']:+.1f}% · {alert['Operational_Action']}"
                    )
            st.markdown(
                f"**{ui('Forecast error needs review', 'កំហុសព្យាករណ៍ត្រូវពិនិត្យ')}**"
            )
            st.caption(
                ui(
                    f"Holdout MAPE: {holdout_mape_text} across {holdout_months} months.",
                    f"MAPE ទិន្នន័យផ្ទៀងផ្ទាត់៖ {holdout_mape_text} ក្នុង {holdout_months} ខែ។",
                )
            )
            if not api_online:
                st.markdown(
                    f"**{ui('Prediction service unavailable', 'សេវាព្យាករណ៍មិនដំណើរការ')}**"
                )
                st.caption(
                    ui(
                        "Scenario predictions use the bundled local model.",
                        "ការព្យាករណ៍សេណារីយ៉ូកំពុងប្រើម៉ូដែលក្នុងម៉ាស៊ីន។",
                    )
                )

    data_through = (
        history_df["Date"].max().strftime("%b %Y")
        if not history_df.empty and "Date" in history_df.columns
        else period_end
    )
    metadata_column, validation_column, action_column = st.columns([1, 1, 1])
    with metadata_column:
        st.caption(
            f"{ui('Model', 'ម៉ូដែល')}: {MODEL_PATH.name} · "
            f"{ui('Data through', 'ទិន្នន័យដល់')}: {data_through}"
        )
    with validation_column:
        st.caption(
            ui(
                f"Validation: {holdout_months} months · MAPE {holdout_mape_text}",
                f"ផ្ទៀងផ្ទាត់៖ {holdout_months} ខែ · MAPE {holdout_mape_text}",
            )
        )
    with action_column:
        st.button(
            ui("Explore forecasts", "មើលការព្យាករណ៍"),
            key="overview_explore_forecasts",
            icon=":material/arrow_forward:",
            width="stretch",
            on_click=lambda: st.session_state.update(
                {"selected_module": "forecasts"}
            ),
        )

# -------------------------------------------------------------------
# TAB 2: FORECAST CURVES & RESIDUALS
# -------------------------------------------------------------------
elif current_tab == "forecasts":
    period_start = forecast_df["Date"].min().strftime("%b %Y")
    period_end = forecast_df["Date"].max().strftime("%b %Y")
    st.caption(
        f"{period_start} – {period_end} · "
        f"{ui('Monthly international visitor arrivals', 'ចំនួនភ្ញៀវអន្តរជាតិប្រចាំខែ')} · "
        f"{ui('Generated by the trained model from your training CSV', 'បង្កើតដោយម៉ូដែលដែលបានបណ្តុះបណ្តាលពី CSV របស់អ្នក')}"
    )
    if forecast_df["Date"].max().date() < pd.Timestamp.now().date():
        st.info(
            ui(
                "Historical validation window — use refreshed forecasts for operational decisions.",
                "រយៈពេលផ្ទៀងផ្ទាត់ជាទិន្នន័យប្រវត្តិសាស្ត្រ — សូមប្រើការព្យាករណ៍ថ្មីសម្រាប់សេចក្តីសម្រេចប្រតិបត្តិការ។",
            )
        )

    fig = go.Figure()
    latest_forecast_date = forecast_df["Date"].max()
    if "Actual_Arrivals" in forecast_df.columns and forecast_df["Actual_Arrivals"].notna().any():
        actual_values_k = forecast_df["Actual_Arrivals"] / 1_000
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=actual_values_k,
                name=ui("Actual", "ចំនួនពិត"),
                line=dict(color=CHART_COLORS["actual"], width=2.5),
                mode="lines+markers",
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
            )
        )
        actual_last = actual_values_k.dropna().iloc[-1]
        fig.add_annotation(
            x=latest_forecast_date,
            y=actual_last,
            text=ui("Actual", "ចំនួនពិត"),
            xshift=30,
            showarrow=False,
            font=dict(color=CHART_COLORS["actual"], size=11),
        )
    forecast_values_k = forecast_df["Tuned_Forecast"] / 1_000
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_values_k,
            name=ui("Forecast", "ការព្យាករណ៍"),
            line=dict(color=CHART_COLORS["forecast"], width=3, dash="dash"),
            mode="lines+markers",
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
        )
    )
    fig.add_annotation(
        x=latest_forecast_date,
        y=forecast_values_k.iloc[-1],
        text=ui("Forecast", "ការព្យាករណ៍"),
        xshift=34,
        showarrow=False,
        font=dict(color=CHART_COLORS["forecast"], size=11),
    )
    baseline_values_k = forecast_df["Seasonal_Baseline"] / 1_000
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=baseline_values_k,
            name=ui("Usual level", "កម្រិតធម្មតា"),
            line=dict(color=CHART_COLORS["baseline"], width=2, dash="dot"),
            mode="lines",
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}K<extra></extra>",
        )
    )
    fig.add_annotation(
        x=latest_forecast_date,
        y=baseline_values_k.iloc[-1],
        text=ui("Usual level", "កម្រិតធម្មតា"),
        xshift=45,
        showarrow=False,
        font=dict(color=CHART_COLORS["baseline"], size=11),
    )

    fig.update_layout(
        template="plotly_white",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="",
            showgrid=False,
            range=[
                forecast_df["Date"].min(),
                latest_forecast_date + pd.DateOffset(days=18),
            ],
        ),
        yaxis=dict(
            title=ui("Visitor arrivals (thousands)", "ចំនួនភ្ញៀវ (ពាន់នាក់)"),
            tickformat=",.0f",
            showgrid=True,
            gridcolor=CHART_GRID_COLOR,
        ),
        margin=dict(l=10, r=70, t=25, b=35),
    )
    style_chart(fig)
    with st.container(border=True):
        st.subheader(ui("Actual vs forecast", "ចំនួនពិត និងការព្យាករណ៍"))
        st.caption(
            f"{period_start} – {period_end} · "
            f"{ui('Monthly international visitor arrivals · thousands', 'ចំនួនភ្ញៀវអន្តរជាតិប្រចាំខែ · ពាន់នាក់')}"
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False},
        )
        render_chart_insight(*forecast_alignment_insight(forecast_df))

    error_column, annual_column = st.columns(2)
    with error_column:
        with st.container(border=True):
            st.subheader(ui("Monthly forecast error", "កំហុសព្យាករណ៍ប្រចាំខែ"))
            st.caption(
                ui(
                    "Absolute percentage error on the chronological holdout.",
                    "កំហុសជាភាគរយដាច់ខាតលើទិន្នន័យផ្ទៀងផ្ទាត់តាមលំដាប់ពេលវេលា។",
                )
            )
            if "APE_%" in forecast_df.columns and forecast_df["APE_%"].notna().any():
                error_df = forecast_df.dropna(subset=["APE_%"]).copy()
                error_df["Month"] = error_df["Date"].dt.strftime("%b %Y")
                highest_error_index = error_df["APE_%"].idxmax()
                error_colors = [
                    CHART_COLORS["error"] if index == highest_error_index
                    else CHART_COLORS["partial"]
                    for index in error_df.index
                ]
                fig_error = px.bar(error_df, x="Month", y="APE_%", text="APE_%")
                fig_error.update_traces(
                    marker_color=error_colors,
                    texttemplate="%{y:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>Absolute error: %{y:.1f}%<extra></extra>",
                )
                fig_error.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title=ui("Month", "ខែ"),
                    yaxis_title=ui("Absolute error (%)", "កំហុសដាច់ខាត (%)"),
                    yaxis_ticksuffix="%",
                    yaxis_rangemode="tozero",
                    xaxis=dict(nticks=6, showgrid=False),
                    margin=dict(l=10, r=10, t=25, b=35),
                )
                style_chart(fig_error)
                st.plotly_chart(
                    fig_error,
                    width="stretch",
                    config={"displaylogo": False},
                )
                render_chart_insight(*forecast_error_spread_insight(forecast_df))
            else:
                st.info(ui("Monthly error values are not available.", "មិនមានទិន្នន័យកំហុសប្រចាំខែទេ។"))

    with annual_column:
        with st.container(border=True):
            st.subheader(ui("Annual visitor trend", "និន្នាការភ្ញៀវប្រចាំឆ្នាំ"))
            annual_view = st.segmented_control(
                ui("Comparison period", "រយៈពេលប្រៀបធៀប"),
                options=[
                    ui("Full years", "ឆ្នាំពេញ"),
                    ui("Jan–Jun comparison", "ប្រៀបធៀប មករា–មិថុនា"),
                ],
                default=ui("Full years", "ឆ្នាំពេញ"),
                key="annual_comparison_period",
                label_visibility="collapsed",
                width="stretch",
            )
            annual_download_column, annual_note_column = st.columns([1, 2])
            if not history_df.empty and "International_Tourist_Arrivals" in history_df.columns:
                annual_df = history_df.dropna(
                    subset=["Date", "International_Tourist_Arrivals"]
                ).copy()
                annual_df["Year"] = annual_df["Date"].dt.year
                monthly_coverage = annual_df.groupby("Year")["Month_Number"].nunique()
                if annual_view == ui("Full years", "ឆ្នាំពេញ"):
                    comparable_months = 12
                    period_label = ui("Full year", "ពេញឆ្នាំ")
                else:
                    comparable_months = 6
                    period_label = ui("Jan–Jun", "មករា–មិថុនា")

                comparison_df = annual_df.loc[
                    annual_df["Month_Number"] <= comparable_months
                ]
                comparison_totals = (
                    comparison_df.groupby("Year", as_index=False)
                    .agg(
                        International_Tourist_Arrivals=(
                            "International_Tourist_Arrivals", "sum"
                        ),
                        Months_In_Period=("Month_Number", "nunique"),
                    )
                )
                comparison_totals = comparison_totals.loc[
                    comparison_totals["Months_In_Period"] == comparable_months
                ].copy()
                comparison_totals["Year_Label"] = comparison_totals["Year"].astype(str)
                comparison_totals["Arrivals_Millions"] = (
                    comparison_totals["International_Tourist_Arrivals"] / 1_000_000
                )

                if comparison_totals.empty:
                    st.info(
                        ui(
                            f"No years have all {comparable_months} months of data.",
                            f"មិនមានឆ្នាំណាដែលមានទិន្នន័យគ្រប់ {comparable_months} ខែទេ។",
                        )
                    )
                else:
                    latest_comparison_year = comparison_totals["Year"].max()
                    comparison_totals["Year_Status"] = np.where(
                        comparison_totals["Year"] == latest_comparison_year,
                        ui("Latest year", "ឆ្នាំចុងក្រោយ"),
                        ui("Previous years", "ឆ្នាំមុនៗ"),
                    )
                    color_map = {
                        ui("Latest year", "ឆ្នាំចុងក្រោយ"): CHART_COLORS["error"],
                        ui("Previous years", "ឆ្នាំមុនៗ"): CHART_COLORS["actual"],
                    }
                    fig_annual = px.bar(
                        comparison_totals,
                        x="Year_Label",
                        y="Arrivals_Millions",
                        color="Year_Status",
                        color_discrete_map=color_map,
                        labels={
                            "Year_Label": "Year",
                            "Arrivals_Millions": "Visitors (millions)",
                            "Year_Status": "Year coverage",
                        },
                    )
                    fig_annual.update_traces(
                        hovertemplate="%{x}<br>Visitors: %{y:.2f}M<extra></extra>"
                    )
                    fig_annual.update_layout(
                        height=300,
                        xaxis_title="",
                        xaxis=dict(
                            tickangle=0,
                            tickmode="array",
                            tickvals=comparison_totals["Year_Label"].iloc[::2].tolist(),
                        ),
                        yaxis_title=ui("Visitors (millions)", "ភ្ញៀវ (លាននាក់)"),
                        yaxis_tickformat=".1f",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=20, b=10),
                    )
                    style_chart(fig_annual)
                    st.plotly_chart(
                        fig_annual,
                        width="stretch",
                        config={"displaylogo": False},
                    )
                    with annual_download_column:
                        st.download_button(
                            ui("Download data", "ទាញយកទិន្នន័យ"),
                            data=comparison_totals.to_csv(index=False).encode("utf-8"),
                            file_name=f"tourism_arrivals_{comparable_months}_month_comparison.csv",
                            mime="text/csv",
                            icon=":material/download:",
                            width="stretch",
                        )
                    years_not_comparable = sorted(
                        set(monthly_coverage.index)
                        - set(comparison_totals["Year"])
                    )
                    if annual_view == ui("Full years", "ឆ្នាំពេញ"):
                        comparison_note = ui(
                            f"Includes years with all 12 months; excludes incomplete years: "
                            f"{', '.join(map(str, years_not_comparable)) or 'none'}.",
                            f"រាប់បញ្ចូលតែឆ្នាំដែលមាន ១២ ខែពេញ; មិនរាប់បញ្ចូលឆ្នាំមិនពេញ៖ "
                            f"{', '.join(map(str, years_not_comparable)) or 'គ្មាន'}។",
                        )
                    else:
                        comparison_note = ui(
                            f"Matched Jan–Jun totals; years without all six months excluded: "
                            f"{', '.join(map(str, years_not_comparable)) or 'none'}.",
                            f"ប្រៀបធៀបចំនួនសរុប មករា–មិថុនា ដូចគ្នា; មិនរាប់បញ្ចូលឆ្នាំដែលគ្មាន ៦ ខែពេញ៖ "
                            f"{', '.join(map(str, years_not_comparable)) or 'គ្មាន'}។",
                        )
                    with annual_note_column:
                        st.caption(comparison_note)
                    if len(comparison_totals) >= 2:
                        previous_year = comparison_totals.iloc[-2]
                        latest_year = comparison_totals.iloc[-1]
                        previous_total = previous_year["International_Tourist_Arrivals"]
                        if previous_total > 0:
                            year_change = (
                                latest_year["International_Tourist_Arrivals"]
                                - previous_total
                            ) / previous_total * 100
                            render_chart_insight(
                                f"{period_label} arrivals in {int(latest_year['Year'])} "
                                f"changed {year_change:+.1f}% versus {int(previous_year['Year'])}.",
                                f"ចំនួនភ្ញៀវ {period_label} ឆ្នាំ {int(latest_year['Year'])} "
                                f"ប្រែប្រួល {year_change:+.1f}% ធៀបនឹងឆ្នាំ {int(previous_year['Year'])}។",
                            )
                        else:
                            st.caption(
                                ui(
                                    "Year-over-year change is unavailable because the comparison year has zero arrivals.",
                                    "មិនអាចគណនាការប្រែប្រួលពីមួយឆ្នាំទៅមួយឆ្នាំបានទេ ព្រោះឆ្នាំប្រៀបធៀបមានចំនួនភ្ញៀវសូន្យ។",
                                )
                            )
                    elif len(comparison_totals) == 1:
                        st.caption(
                            ui(
                                "Only one comparable year is available; a year-over-year change cannot be calculated.",
                                "មានតែឆ្នាំប្រៀបធៀបមួយប៉ុណ្ណោះ ដូច្នេះមិនអាចគណនាការប្រែប្រួលពីមួយឆ្នាំទៅមួយឆ្នាំបានទេ។",
                            )
                        )
            else:
                st.info(ui("Historical annual data is not available.", "មិនមានទិន្នន័យប្រចាំឆ្នាំទេ។"))

# -------------------------------------------------------------------
# -------------------------------------------------------------------
elif current_tab == "alerts":
    st.caption(
        ui(
            f"Alert levels use the trained model's chronological holdout predictions "
            f"({holdout_months} months) compared with the seasonal baseline.",
            f"កម្រិតព្រមានប្រើលទ្ធផលព្យាករណ៍ផ្ទៀងផ្ទាត់តាមលំដាប់ពេលវេលារបស់ម៉ូដែល "
            f"({holdout_months} ខែ) ធៀបនឹងកម្រិតមូលដ្ឋានតាមរដូវកាល។",
        )
    )
    status_labels = {
        "High Demand (🔴)": ui("High", "ខ្ពស់"),
        "Normal Demand (🟢)": ui("Normal", "ធម្មតា"),
        "Low Demand (🟡)": ui("Low", "ទាប"),
    }
    palette = {
        status_labels["High Demand (🔴)"]: CHART_COLORS["high"],
        status_labels["Normal Demand (🟢)"]: CHART_COLORS["normal"],
        status_labels["Low Demand (🟡)"]: CHART_COLORS["low"],
    }

    status_counts = forecast_df["Clean_Status"].value_counts()
    status_columns = st.columns(3)
    for column, raw_status, label in [
        (status_columns[0], "High Demand (🔴)", ui("High demand", "តម្រូវការខ្ពស់")),
        (status_columns[1], "Normal Demand (🟢)", ui("Normal range", "កម្រិតធម្មតា")),
        (status_columns[2], "Low Demand (🟡)", ui("Low demand", "តម្រូវការទាប")),
    ]:
        with column:
            with st.container(border=True):
                st.metric(
                    label,
                    int(status_counts.get(raw_status, 0)),
                    border=False,
                )
                st.caption(ui("months", "ខែ"))

    monthly_status_df = forecast_df[["Date", "Deviation_%", "Clean_Status"]].copy()
    monthly_status_df["Month"] = monthly_status_df["Date"].dt.strftime("%b %Y")
    monthly_status_df["Level"] = monthly_status_df["Clean_Status"].map(status_labels)
    month_order = monthly_status_df["Month"].tolist()
    palette = {
        status_labels["High Demand (🔴)"]: CHART_COLORS["high"],
        status_labels["Normal Demand (🟢)"]: CHART_COLORS["normal"],
        status_labels["Low Demand (🟡)"]: CHART_COLORS["low"],
    }
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
        texttemplate="%{x:+.1f}%",
        textposition="outside",
        textfont=dict(size=12, color=CHART_TEXT_COLOR),
        cliponaxis=False,
        hovertemplate="%{y}<br>%{customdata[0]}<br>%{x:+.1f}%<extra></extra>",
    )
    deviation_extent = max(
        60,
        float(np.ceil(forecast_df["Deviation_%"].abs().max() / 10) * 10),
    )
    alert_rows = forecast_df.loc[
        forecast_df["Clean_Status"] != "Normal Demand (🟢)"
    ].assign(Absolute_Deviation=lambda frame: frame["Deviation_%"].abs())
    fig_monthly_status.add_vline(
        x=-15,
        line_dash="dash",
        line_color=CHART_COLORS["low"],
        annotation_text="-15%",
        annotation_position="top",
    )
    fig_monthly_status.add_vline(
        x=15,
        line_dash="dash",
        line_color=CHART_COLORS["normal"],
        annotation_text="+15%",
        annotation_position="top",
    )
    chart_column, response_column = st.columns([1.7, 1])
    with chart_column:
        with st.container(border=True):
            st.subheader(ui("Deviation from usual level", "គម្លាតពីកម្រិតធម្មតា"))
            st.caption(
                ui(
                    "Monthly forecast compared with the seasonal baseline.",
                    "ការព្យាករណ៍ប្រចាំខែធៀបនឹងកម្រិតមូលដ្ឋានតាមរដូវកាល។",
                )
            )
            fig_monthly_status.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                height=440,
                font=dict(
                    family="Inter, Kantumruy Pro, Noto Sans Khmer, sans-serif",
                    size=14,
                    color="#334155",
                ),
                showlegend=False,
                xaxis=dict(
                    title=ui("Above/below usual (%)", "លើស/ក្រោមកម្រិតធម្មតា (%)"),
                    range=[-deviation_extent, deviation_extent],
                    tickvals=np.linspace(-deviation_extent, deviation_extent, 7),
                    ticksuffix="%",
                    zeroline=True,
                    zerolinecolor="#64748b",
                    showgrid=True,
                    gridcolor=CHART_GRID_COLOR,
                ),
                yaxis=dict(
                    title="",
                    categoryorder="array",
                    categoryarray=month_order,
                    autorange="reversed",
                ),
                margin=dict(l=20, r=25, t=15, b=20),
            )
            style_chart(fig_monthly_status)
            st.plotly_chart(
                fig_monthly_status,
                width="stretch",
                config={"displaylogo": False},
            )
            if alert_rows.empty:
                render_chart_insight(
                    f"All {len(forecast_df)} months are within the ±15% usual-demand range.",
                    f"ខែទាំង {len(forecast_df)} ស្ថិតក្នុងចន្លោះតម្រូវការធម្មតា ±15%។",
                )
            else:
                largest_alert = alert_rows.nlargest(
                    1, "Absolute_Deviation"
                ).iloc[0]
                largest_direction = (
                    "above" if largest_alert["Deviation_%"] > 0 else "below"
                )
                largest_direction_kh = (
                    "លើស" if largest_alert["Deviation_%"] > 0 else "ក្រោម"
                )
                render_chart_insight(
                    f"{len(alert_rows)} of {len(forecast_df)} months exceed ±15%; "
                    f"the largest signal is {largest_alert['Deviation_%']:+.1f}% "
                    f"{largest_direction} usual in {largest_alert['Date']:%b %Y}.",
                    f"{len(alert_rows)} ក្នុង {len(forecast_df)} ខែ លើសកម្រិត ±15%; "
                    f"សញ្ញាខ្លាំងបំផុតគឺ {largest_alert['Deviation_%']:+.1f}% "
                    f"{largest_direction_kh}កម្រិតធម្មតា នៅ {largest_alert['Date']:%b %Y}។",
                )

    with response_column:
        with st.container(border=True):
            st.subheader(ui("Suggested response", "សកម្មភាពដែលបានណែនាំ"))
            if alert_rows.empty:
                st.success(
                    ui(
                        "No demand alerts need action in this period.",
                        "មិនមានការព្រមានតម្រូវការដែលត្រូវចាត់វិធានការក្នុងរយៈពេលនេះទេ។",
                    )
                )
            else:
                prioritized_alerts = alert_rows.nlargest(3, "Absolute_Deviation")
                priority_alert = prioritized_alerts.iloc[0]
                selected_alert_label = st.selectbox(
                    ui("Review signal", "ជ្រើសសញ្ញាដើម្បីពិនិត្យ"),
                    options=[
                        f"{row['Date']:%b %Y} · {status_labels[row['Clean_Status']]}"
                        for _, row in prioritized_alerts.iterrows()
                    ],
                    key="selected_alert_signal",
                )
                selected_alert_index = next(
                    index
                    for index, (_, row) in enumerate(prioritized_alerts.iterrows())
                    if f"{row['Date']:%b %Y} · {status_labels[row['Clean_Status']]}"
                    == selected_alert_label
                )
                priority_alert = prioritized_alerts.iloc[selected_alert_index]
                st.markdown(
                    f"**{priority_alert['Date'].strftime('%b %Y')} · "
                    f"{status_labels[priority_alert['Clean_Status']]}**"
                )
                st.metric(
                    ui("Deviation from baseline", "គម្លាតពីកម្រិតមូលដ្ឋាន"),
                    f"{priority_alert['Deviation_%']:+.1f}%",
                )
                action_text = (
                    ui(
                        "Review visitor-market demand and campaign options.",
                        "ពិនិត្យតម្រូវការទីផ្សារភ្ញៀវ និងជម្រើសយុទ្ធនាការផ្សព្វផ្សាយ។",
                    )
                    if "Low" in priority_alert["Clean_Status"]
                    else ui(
                        "Review capacity and visitor services for the demand increase.",
                        "ពិនិត្យសមត្ថភាព និងសេវាកម្មភ្ញៀវសម្រាប់ការកើនឡើងនៃតម្រូវការ។",
                    )
                )
                st.write(action_text)
                st.caption(
                    f"{ui('Owner', 'អ្នកទទួលខុសត្រូវ')}: "
                    f"{ui('Unassigned', 'មិនទាន់កំណត់')} · "
                    f"{ui('Status', 'ស្ថានភាព')}: {ui('To review', 'ត្រូវពិនិត្យ')}"
                )
                st.button(
                    ui("Review signal", "ពិនិត្យសញ្ញា"),
                    key="review_selected_signal",
                    icon=":material/search:",
                    width="stretch",
                    on_click=lambda: st.session_state.update(
                        {"selected_module": "forecasts"}
                    ),
                )
                high_alerts = alert_rows.loc[
                    alert_rows["Clean_Status"] == "High Demand (🔴)"
                ].sort_values("Date")
                if not high_alerts.empty:
                    with st.expander(ui("High-demand months", "ខែដែលមានតម្រូវការខ្ពស់")):
                        for _, high_alert in high_alerts.iterrows():
                            st.markdown(
                                f"**{high_alert['Date']:%b %Y}** · "
                                f"{high_alert['Deviation_%']:+.1f}%"
                            )
                            st.caption(
                                ui(
                                    "Review capacity and visitor services.",
                                    "ពិនិត្យសមត្ថភាព និងសេវាកម្មភ្ញៀវ។",
                                )
                            )

    st.subheader(
        ui(
            "Monthly demand signals (selected months)",
            "សញ្ញាតម្រូវការប្រចាំខែ (ខែដែលបានជ្រើស)",
        )
    )
    filter_column, row_count_column = st.columns([3, 1])
    available_statuses = [
        status_labels["High Demand (🔴)"],
        status_labels["Normal Demand (🟢)"],
        status_labels["Low Demand (🟡)"],
    ]
    with filter_column:
        selected_statuses = st.multiselect(
            ui("Signals", "សញ្ញា"),
            options=available_statuses,
            default=available_statuses,
            key="alert_signal_filter",
        )
    with row_count_column:
        selected_month_count = st.selectbox(
            ui("Months to show", "ចំនួនខែដែលត្រូវបង្ហាញ"),
            options=[3, 6, 12],
            index=0,
            key="alert_month_count",
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

    filtered_alerts = forecast_df.loc[
        forecast_df["Clean_Status"].map(status_labels).isin(selected_statuses)
    ].copy()
    ranked_filtered_alerts = filtered_alerts.assign(
        Absolute_Deviation=lambda frame: frame["Deviation_%"].abs()
    ).sort_values("Absolute_Deviation", ascending=False)
    representative_indices = []
    for raw_status in [
        "High Demand (🔴)",
        "Normal Demand (🟢)",
        "Low Demand (🟡)",
    ]:
        matching_status = ranked_filtered_alerts.loc[
            ranked_filtered_alerts["Clean_Status"] == raw_status
        ]
        if not matching_status.empty:
            representative_indices.append(matching_status.index[0])
    remaining_indices = [
        index
        for index in ranked_filtered_alerts.index
        if index not in representative_indices
    ]
    selected_indices = (
        representative_indices + remaining_indices
    )[:selected_month_count]
    selected_signal_df = filtered_alerts.loc[selected_indices].sort_values("Date")
    compact_signals = selected_signal_df[
        ["Date", "Clean_Status", "Tuned_Forecast", "Seasonal_Baseline", "Deviation_%"]
    ].copy()
    compact_signals["Date"] = compact_signals["Date"].dt.strftime("%b %Y")
    compact_signals["Clean_Status"] = compact_signals["Clean_Status"].map(status_labels)
    compact_signals["Tuned_Forecast"] = compact_signals["Tuned_Forecast"].map(
        lambda value: f"{value:,.0f}"
    )
    compact_signals["Seasonal_Baseline"] = compact_signals["Seasonal_Baseline"].map(
        lambda value: f"{value:,.0f}"
    )
    compact_signals["Deviation_%"] = compact_signals["Deviation_%"].map(
        lambda value: f"{value:+.1f}%"
    )
    compact_signals.columns = [
        ui("Month", "ខែ"),
        ui("Signal", "សញ្ញា"),
        ui("Forecast", "ការព្យាករណ៍"),
        ui("Baseline", "កម្រិតមូលដ្ឋាន"),
        ui("Deviation", "គម្លាត"),
    ]
    if compact_signals.empty:
        st.info(ui("No months match the selected signals.", "គ្មានខែដែលត្រូវនឹងសញ្ញាដែលបានជ្រើសទេ។"))
    else:
        st.dataframe(
            compact_signals,
            width="stretch",
            hide_index=True,
            height=min(220, 42 + 35 * len(compact_signals)),
        )

    def style_status(val):
        if "High" in str(val):
            return "background-color: #dbeafe; color: #173b68; font-weight: 700;"
        if "Low" in str(val):
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        return "background-color: #ccfbf1; color: #0f766e; font-weight: 700;"

    styler = display_df.style
    if hasattr(styler, "map"):
        styled_df = styler.map(style_status, subset=["Alert Level"])
    else:
        styled_df = styler.applymap(style_status, subset=["Alert Level"])

    with st.expander(ui("Detailed alert context", "ព័ត៌មានលម្អិតអំពីសញ្ញា")):
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
elif current_tab == "scenario":
    target_column = "International_Tourist_Arrivals"
    latest_history = history_df.dropna(subset=[target_column]).iloc[-1]
    latest_history_date = latest_history["Date"]
    default_scenario_date = (
        latest_history_date + pd.offsets.MonthBegin(1)
    ).date()
    historical_month_data = history_df.loc[
        (history_df["Month_Number"] == default_scenario_date.month)
        & (history_df["Date"] < pd.Timestamp(default_scenario_date))
    ].copy()
    if "COVID_Indicator" in historical_month_data:
        non_covid_months = historical_month_data.loc[
            pd.to_numeric(historical_month_data["COVID_Indicator"], errors="coerce").fillna(0) == 0
        ]
        if not non_covid_months.empty:
            historical_month_data = non_covid_months
    baseline_sample = historical_month_data.tail(5)
    if baseline_sample.empty:
        raise ValueError("Training data does not contain a historical seasonal baseline.")
    default_scenario_baseline = float(baseline_sample[target_column].median())

    weather_reference = historical_month_data.iloc[-1]
    recent_arrivals = history_df.loc[
        history_df["Date"] <= latest_history_date, target_column
    ].dropna()
    if len(recent_arrivals) < 12:
        raise ValueError("At least 12 monthly arrival observations are required for scenarios.")

    scenario_defaults = {
        "scenario_date": default_scenario_date,
        "scenario_temperature": float(weather_reference["Average_Temperature"]),
        "scenario_rainfall": float(weather_reference["Rainfall"]),
        "scenario_cambodia_holidays": int(weather_reference["Cambodia_Holiday_Days"]),
        "scenario_source_holidays": int(weather_reference["Total_Origin_Holidays"]),
        "scenario_china_holidays": int(weather_reference["China_Holidays"]),
        "scenario_disruption": 0,
        "scenario_baseline": default_scenario_baseline,
        "scenario_lag_1": float(recent_arrivals.iloc[-1]),
        "scenario_lag_3": float(recent_arrivals.iloc[-3]),
        "scenario_lag_12": float(recent_arrivals.iloc[-12]),
        "scenario_rolling_3": float(recent_arrivals.tail(3).mean()),
        "scenario_rolling_12": float(recent_arrivals.tail(12).mean()),
    }
    if st.session_state.get("scenario_defaults_version") != 3:
        st.session_state.update(scenario_defaults)
        st.session_state["scenario_defaults_version"] = 3
    else:
        for key, default_value in scenario_defaults.items():
            st.session_state.setdefault(key, default_value)

    def populate_scenario_features():
        selected_date = st.session_state.get("scenario_date")
        if selected_date is None:
            return

        selected_month = pd.Timestamp(selected_date).to_period("M")
        normalized_date = selected_month.to_timestamp().date()
        if normalized_date != selected_date:
            st.session_state["scenario_date"] = normalized_date

        historical_rows = history_df.loc[
            history_df["Date"].dt.to_period("M") == selected_month
        ]
        exact_match = not historical_rows.empty
        historical_row = historical_rows.iloc[0] if exact_match else None

        if exact_match:
            st.session_state["scenario_autofill_source"] = ui(
                "Features loaded from this exact training month; prediction runs automatically using only earlier months for a historical backtest.",
                "បានផ្ទុកលក្ខណៈពីខែនេះក្នុងទិន្នន័យបណ្តុះបណ្តាល ហើយដំណើរការព្យាករណ៍ដោយស្វ័យប្រវត្តិ ដោយប្រើតែខែមុនសម្រាប់ការផ្ទៀងផ្ទាត់ប្រវត្តិសាស្ត្រ។",
            )
        else:
            st.session_state["scenario_autofill_source"] = ui(
                "No exact training row: lag features use preceding observed arrivals; weather and calendar use the nearest matching month. Prediction runs automatically when required history is available.",
                "មិនមានជួរទិន្នន័យខែនេះទេ៖ លក្ខណៈ lag ប្រើចំនួនភ្ញៀវពីខែមុន ហើយអាកាសធាតុនិងប្រតិទិនប្រើខែដែលត្រូវគ្នាជិតបំផុត។ ប្រព័ន្ធព្យាករណ៍ដោយស្វ័យប្រវត្តិនៅពេលមានទិន្នន័យប្រវត្តិសាស្ត្រគ្រប់គ្រាន់។",
            )

        def numeric_value(value):
            return float(value) if pd.notna(value) else None

        values = {}
        if exact_match:
            values.update(
                {
                    "scenario_lag_1": numeric_value(historical_row["Lag_1"]),
                    "scenario_lag_3": numeric_value(historical_row["Lag_3"]),
                    "scenario_lag_12": numeric_value(historical_row["Lag_12"]),
                    "scenario_rolling_3": numeric_value(historical_row["Rolling_3_Mean"]),
                    "scenario_rolling_12": numeric_value(historical_row["Rolling_12_Mean"]),
                }
            )
        else:
            arrivals_by_month = history_df.set_index(
                history_df["Date"].dt.to_period("M")
            )[target_column]

            def arrivals_for_lag(months_ago):
                prior_month = selected_month - months_ago
                return numeric_value(arrivals_by_month.get(prior_month, np.nan))

            def rolling_average(months):
                arrivals = [arrivals_for_lag(offset) for offset in months]
                if any(value is None for value in arrivals):
                    return None
                return float(np.mean(arrivals))

            values.update(
                {
                    "scenario_lag_1": arrivals_for_lag(1),
                    "scenario_lag_3": arrivals_for_lag(3),
                    "scenario_lag_12": arrivals_for_lag(12),
                    "scenario_rolling_3": rolling_average((1, 2, 3)),
                    "scenario_rolling_12": rolling_average(range(1, 13)),
                }
            )

        same_month = history_df.loc[
            (history_df["Month_Number"] == selected_month.month)
            & (history_df["Date"] < selected_month.to_timestamp())
        ].copy()
        if "COVID_Indicator" in same_month:
            non_covid_months = same_month.loc[
                pd.to_numeric(same_month["COVID_Indicator"], errors="coerce").fillna(0) == 0
            ]
            if not non_covid_months.empty:
                same_month = non_covid_months
        baseline_value = (
            float(same_month.tail(5)[target_column].median())
            if not same_month.empty
            else (
                numeric_value(historical_row["Lag_12"])
                if exact_match
                else arrivals_for_lag(12)
            )
        )

        month_weather = history_df.loc[
            history_df["Month_Number"] == selected_month.month
        ].copy()
        if not month_weather.empty:
            month_weather["_month_distance"] = (
                month_weather["Date"].dt.to_period("M").astype(int)
                - selected_month.ordinal
            ).abs()
            weather_row = (
                historical_row
                if exact_match
                else month_weather.sort_values("_month_distance").iloc[0]
            )
            values.update(
                {
                    "scenario_temperature": float(weather_row["Average_Temperature"]),
                    "scenario_rainfall": float(weather_row["Rainfall"]),
                    "scenario_cambodia_holidays": int(weather_row["Cambodia_Holiday_Days"]),
                    "scenario_source_holidays": int(weather_row["Total_Origin_Holidays"]),
                    "scenario_china_holidays": int(weather_row["China_Holidays"]),
                }
            )
        else:
            for key in [
                "scenario_temperature",
                "scenario_rainfall",
                "scenario_cambodia_holidays",
                "scenario_source_holidays",
                "scenario_china_holidays",
            ]:
                values[key] = None

        values.update(
            {
                "scenario_disruption": (
                    int(historical_row["COVID_Indicator"])
                    if exact_match
                    else 0
                ),
                "scenario_baseline": baseline_value,
            }
        )
        st.session_state.update(values)
        st.session_state.pop("scenario_result", None)
        st.session_state["scenario_auto_predict"] = True

    if not api_online:
        st.warning(
            ui(
                "Prediction service offline — submitted features will be run through the bundled local model.",
                "សេវាព្យាករណ៍មិនដំណើរការ — ធាតុបញ្ចូលដែលបានដាក់ស្នើនឹងត្រូវព្យាករណ៍ដោយម៉ូដែលក្នុងម៉ាស៊ីន។",
            )
        )

    input_column, result_column = st.columns([1, 1.12])
    with input_column:
        with st.container(border=True, key="scenario-input-card"):
            st.subheader(ui("Scenario assumptions", "ការកំណត់សេណារីយ៉ូ"))
            st.date_input(
                ui("Target month", "ខែដែលត្រូវព្យាករណ៍"),
                value=None,
                min_value=history_df["Date"].min().date(),
                key="scenario_date",
                on_change=populate_scenario_features,
                help=ui(
                    "Changing the month loads matching features from the training data.",
                    "ការផ្លាស់ប្តូរខែនឹងផ្ទុកលក្ខណៈដែលត្រូវគ្នាពីទិន្នន័យបណ្តុះបណ្តាល។",
                ),
            )
            st.caption(
                st.session_state.get(
                    "scenario_autofill_source",
                    ui(
                        f"Defaults use your training data through {latest_history_date:%b %Y}.",
                        f"តម្លៃដើមប្រើទិន្នន័យបណ្តុះបណ្តាលរហូតដល់ {latest_history_date:%b %Y}។",
                    ),
                )
            )
            if not st.session_state.get("scenario_initialized"):
                populate_scenario_features()
                st.session_state["scenario_initialized"] = True
            with st.form("simulation_form", border=False):
                sim_date = st.session_state["scenario_date"]

                st.markdown(f"**{ui('Weather', 'អាកាសធាតុ')}**")
                sim_temp = st.slider(
                    ui("Temperature (°C)", "សីតុណ្ហភាព (°C)"),
                    min_value=20.0,
                    max_value=35.0,
                    value=None,
                    step=0.1,
                    key="scenario_temperature",
                )
                sim_rain = st.slider(
                    ui("Rainfall (mm)", "បរិមាណទឹកភ្លៀង (មម)"),
                    min_value=0.0,
                    max_value=300.0,
                    value=None,
                    step=5.0,
                    key="scenario_rainfall",
                )

                st.markdown(f"**{ui('Calendar', 'ប្រតិទិន')}**")
                holiday_col1, holiday_col2 = st.columns(2)
                with holiday_col1:
                    sim_kh_holidays = st.number_input(
                        ui("Cambodia holidays (days)", "ថ្ងៃឈប់សម្រាកកម្ពុជា"),
                        min_value=0,
                        max_value=15,
                        value=None,
                        key="scenario_cambodia_holidays",
                    )
                    sim_origin_holidays = st.number_input(
                        ui("Source-market holidays", "ថ្ងៃឈប់សម្រាកទីផ្សារដើម"),
                        min_value=0,
                        max_value=30,
                        value=None,
                        key="scenario_source_holidays",
                    )
                with holiday_col2:
                    sim_china_holidays = st.number_input(
                        ui("China holidays", "ថ្ងៃឈប់សម្រាកចិន"),
                        min_value=0,
                        max_value=10,
                        value=None,
                        key="scenario_china_holidays",
                    )
                    sim_covid = st.selectbox(
                        ui("Special disruption", "ការរំខានពិសេស"),
                        options=[0, 1],
                        index=None,
                        format_func=lambda value: (
                            ui("None", "គ្មាន") if value == 0 else ui("Active", "សកម្ម")
                        ),
                        key="scenario_disruption",
                    )

                with st.expander(ui("Advanced historical inputs", "ទិន្នន័យប្រវត្តិសាស្ត្រកម្រិតខ្ពស់")):
                    sim_baseline = st.number_input(
                        ui("Seasonal baseline visitors", "ចំនួនភ្ញៀវកម្រិតមូលដ្ឋាន"),
                        min_value=1.0,
                        value=None,
                        step=1000.0,
                        key="scenario_baseline",
                    )
                    history_col1, history_col2 = st.columns(2)
                    with history_col1:
                        sim_lag1 = st.number_input(
                            ui("Previous month visitors", "ចំនួនភ្ញៀវខែមុន"),
                            value=None,
                            step=1000.0,
                            key="scenario_lag_1",
                        )
                        sim_lag3 = st.number_input(
                            ui("Visitors 3 months ago", "ចំនួនភ្ញៀវ ៣ ខែមុន"),
                            value=None,
                            step=1000.0,
                            key="scenario_lag_3",
                        )
                        sim_lag12 = st.number_input(
                            ui("Same month last year", "ចំនួនភ្ញៀវខែដូចគ្នាឆ្នាំមុន"),
                            value=None,
                            step=1000.0,
                            key="scenario_lag_12",
                        )
                    with history_col2:
                        sim_roll3 = st.number_input(
                            ui("3-month average", "មធ្យមភាគ ៣ ខែ"),
                            value=None,
                            step=1000.0,
                            key="scenario_rolling_3",
                        )
                        sim_roll12 = st.number_input(
                            ui("12-month average", "មធ្យមភាគ ១២ ខែ"),
                            value=None,
                            step=1000.0,
                            key="scenario_rolling_12",
                        )

                def reset_scenario_inputs():
                    st.session_state.update(scenario_defaults)
                    st.session_state["scenario_autofill_source"] = ui(
                        f"Defaults use your training data through {latest_history_date:%b %Y}.",
                        f"តម្លៃដើមប្រើទិន្នន័យបណ្តុះបណ្តាលរហូតដល់ {latest_history_date:%b %Y}។",
                    )
                    st.session_state.pop("scenario_result", None)
                    st.session_state.pop("scenario_auto_predict", None)

                run_column, reset_column = st.columns(2)
                with run_column:
                    dispatch_btn = st.form_submit_button(
                        ui("Run scenario", "ដំណើរការសេណារីយ៉ូ"),
                        type="primary",
                        icon=":material/play_arrow:",
                        width="stretch",
                    )
                with reset_column:
                    st.form_submit_button(
                        ui("Reset", "កំណត់ឡើងវិញ"),
                        icon=":material/restart_alt:",
                        on_click=reset_scenario_inputs,
                        width="stretch",
                    )

    auto_predict = st.session_state.pop("scenario_auto_predict", False)
    should_predict = dispatch_btn or auto_predict
    scenario_result = st.session_state.get("scenario_result")

    if should_predict:
        required_inputs = {
            "Seasonal baseline": sim_baseline,
            "Previous month visitors": sim_lag1,
            "Visitors 3 months ago": sim_lag3,
            "Same month last year": sim_lag12,
            "3-month average": sim_roll3,
            "12-month average": sim_roll12,
            "Temperature": sim_temp,
            "Rainfall": sim_rain,
            "Cambodia holidays": sim_kh_holidays,
            "Source-market holidays": sim_origin_holidays,
            "China holidays": sim_china_holidays,
            "Special disruption": sim_covid,
        }
        missing_inputs = [
            name
            for name, value in required_inputs.items()
            if value is None or pd.isna(value)
        ]
        if missing_inputs:
            st.error(
                ui(
                    "Cannot run this month: the training data has no value for "
                    + ", ".join(missing_inputs)
                    + ". Enter those inputs manually if you have reliable values.",
                    "មិនអាចដំណើរការខែនេះបានទេ៖ ទិន្នន័យបណ្តុះបណ្តាលគ្មានតម្លៃសម្រាប់ "
                    + ", ".join(missing_inputs)
                    + "។ សូមបញ្ចូលតម្លៃដោយដៃ ប្រសិនបើមានទិន្នន័យដែលអាចទុកចិត្តបាន។",
                )
            )
            should_predict = False

    if should_predict:
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

        def local_model_result(inputs):
            if not isinstance(model_artifact, dict):
                return None

            feature_names = model_artifact.get("features", [])
            model = model_artifact.get("model")
            if not feature_names or model is None:
                return None

            month_number = inputs["Month_Number"]
            feature_values = {
                "Lag_1": inputs["Lag_1"],
                "Lag_3": inputs["Lag_3"],
                "Lag_12": inputs["Lag_12"],
                "Rolling_3_Mean": inputs["Rolling_3_Mean"],
                "Rolling_12_Mean": inputs["Rolling_12_Mean"],
                "Lag_1_to_Rolling_3": inputs["Lag_1"] / (inputs["Rolling_3_Mean"] + 1e-5),
                "Lag_1_to_Lag_12": inputs["Lag_1"] / (inputs["Lag_12"] + 1e-5),
                "Rolling_3_to_12": inputs["Rolling_3_Mean"] / (inputs["Rolling_12_Mean"] + 1e-5),
                "Average_Temperature": inputs["Average_Temperature"],
                "Rainfall": inputs["Rainfall"],
                "Cambodia_Holiday_Days": inputs["Cambodia_Holiday_Days"],
                "COVID_Indicator": inputs["COVID_Indicator"],
                "Sin_Month": np.sin(2 * np.pi * month_number / 12),
                "Cos_Month": np.cos(2 * np.pi * month_number / 12),
                "Total_Origin_Holidays": inputs["Total_Origin_Holidays"],
                "China_Holidays": inputs["China_Holidays"],
            }
            missing_features = [name for name in feature_names if name not in feature_values]
            if missing_features:
                raise ValueError(
                    f"The local model requires unsupported features: {', '.join(missing_features)}"
                )

            feature_frame = pd.DataFrame(
                [{name: feature_values[name] for name in feature_names}],
                columns=feature_names,
            )

            selected_month = pd.Timestamp(inputs["date"]).to_period("M")
            training_features, training_target, training_weights = (
                prepare_training_data(HISTORY_PATH)
            )
            training_dates = history_df.loc[
                training_features.index, "Date"
            ].dt.to_period("M")
            historical_indices = training_dates.index[
                training_dates == selected_month
            ]
            is_historical_backtest = len(historical_indices) > 0
            prediction_model = model
            actual_arrivals = None
            if is_historical_backtest:
                training_mask = training_dates < selected_month
                if not training_mask.any():
                    raise ValueError(
                        "The selected month has no earlier training rows for a leakage-free backtest."
                    )
                prediction_model = clone(model)
                fit_target = training_target.loc[training_mask]
                if model_artifact.get("target_transform", "none") == "log1p":
                    fit_target = np.log1p(fit_target)
                prediction_model.fit(
                    training_features.loc[training_mask],
                    fit_target,
                    sample_weight=training_weights.loc[training_mask],
                )
                actual_arrivals = float(
                    history_df.loc[historical_indices[0], "International_Tourist_Arrivals"]
                )

            prediction = float(prediction_model.predict(feature_frame)[0])
            if model_artifact.get("target_transform", "none") == "log1p":
                prediction = float(np.expm1(prediction))
            prediction = max(0.0, prediction)
            deviation = (prediction - inputs["seasonal_baseline"]) / inputs["seasonal_baseline"] * 100
            if deviation > 15:
                status = "high"
            elif deviation < -15:
                status = "low"
            else:
                status = "normal"
            return {
                "prediction": round(prediction),
                "deviation": deviation,
                "status": status,
                "source": (
                    "historical_backtest"
                    if is_historical_backtest
                    else "local_model"
                ),
                "baseline": inputs["seasonal_baseline"],
                "actual_arrivals": actual_arrivals,
                "actual_ape": (
                    abs(prediction - actual_arrivals) / actual_arrivals * 100
                    if actual_arrivals
                    else None
                ),
            }

        def use_local_prediction(inputs):
            local_result = local_model_result(inputs)
            if local_result is None:
                raise ValueError("The bundled trained model cannot run this scenario.")
            st.info(
                ui(
                    "Result calculated by the bundled model using the selected date's features.",
                    "លទ្ធផលគណនាដោយម៉ូដែលក្នុងម៉ាស៊ីន ដោយប្រើលក្ខណៈរបស់កាលបរិច្ឆេទដែលបានជ្រើស។",
                )
            )
            st.session_state["scenario_result"] = local_result
            return local_result

        historical_month_exists = bool(
            (history_df["Date"].dt.to_period("M") == pd.Timestamp(sim_date).to_period("M")).any()
        )
        if auto_predict or historical_month_exists:
            with st.spinner(ui("Running date-matched model prediction...", "កំពុងគណនាការព្យាករណ៍តាមកាលបរិច្ឆេទ...")):
                scenario_result = use_local_prediction(payload)
        else:
            with st.spinner(ui("Running scenario...", "កំពុងដំណើរការសេណារីយ៉ូ...")):
                try:
                    res = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=3.0)
                    if res.status_code == 200:
                        out = res.json()
                        alert_status = out["alert_status"]
                        if "High" in alert_status:
                            status = "high"
                        elif "Low" in alert_status:
                            status = "low"
                        else:
                            status = "normal"
                        scenario_result = {
                            "prediction": out["predicted_arrivals"],
                            "deviation": out["deviation_pct"],
                            "status": status,
                            "source": "model",
                            "baseline": sim_baseline,
                            "actual_arrivals": None,
                            "actual_ape": None,
                        }
                        st.session_state["scenario_result"] = scenario_result
                    else:
                        st.error(
                            ui(
                                f"Prediction service returned HTTP {res.status_code}; using the bundled local model.",
                                f"សេវាព្យាករណ៍បានត្រឡប់ HTTP {res.status_code}; កំពុងប្រើម៉ូដែលក្នុងម៉ាស៊ីន។",
                            )
                        )
                        scenario_result = use_local_prediction(payload)
                except requests.RequestException:
                    st.info(
                        ui(
                            "The prediction service could not be reached; calculating with the bundled local model.",
                            "មិនអាចភ្ជាប់ទៅសេវាព្យាករណ៍បានទេ កំពុងគណនាដោយម៉ូដែលក្នុងម៉ាស៊ីន។",
                        )
                    )
                    scenario_result = use_local_prediction(payload)

    saved_transaction_id = None
    if dispatch_btn and scenario_result is not None:
        try:
            saved_transaction_id = append_scenario_transaction(
                payload,
                scenario_result,
                SCENARIO_TRANSACTIONS_PATH,
            )
        except (OSError, ValueError) as error:
            st.error(
                ui(
                    f"The scenario result is available, but its transaction could not be saved: {error}",
                    f"លទ្ធផលសេណារីយ៉ូមានរួចហើយ ប៉ុន្តែមិនអាចរក្សាទុកប្រតិបត្តិការបានទេ៖ {error}",
                )
            )

    with result_column:
        with st.container(border=True, key="scenario-result-card"):
            source_labels = {
                "model": ui("Model estimate", "ការប៉ាន់ស្មានដោយម៉ូដែល"),
                "local_model": ui("Local model estimate", "ការប៉ាន់ស្មានដោយម៉ូដែលក្នុងម៉ាស៊ីន"),
                "historical_backtest": ui(
                    "Historical backtest · trained on earlier months",
                    "ការផ្ទៀងផ្ទាត់ប្រវត្តិសាស្ត្រ · បណ្តុះបណ្តាលលើខែមុន",
                ),
            }
            status_labels = {
                "high": ui("High demand", "តម្រូវការខ្ពស់"),
                "low": ui("Low demand", "តម្រូវការទាប"),
                "normal": ui("Normal range", "កម្រិតធម្មតា"),
            }
            st.caption(
                source_labels[scenario_result["source"]]
                if scenario_result
                else ui("Scenario result", "លទ្ធផលសេណារីយ៉ូ")
            )
            if saved_transaction_id:
                st.success(
                    ui(
                        f"Scenario inputs and results saved (ID: {saved_transaction_id}).",
                        f"បានរក្សាទុកធាតុបញ្ចូល និងលទ្ធផលសេណារីយ៉ូ (លេខសម្គាល់៖ {saved_transaction_id})។",
                    )
                )
            if scenario_result is None:
                st.info(
                    ui(
                        "Run a scenario to see the visitor estimate and baseline comparison.",
                        "ដំណើរការសេណារីយ៉ូ ដើម្បីមើលការប៉ាន់ស្មានភ្ញៀវ និងការប្រៀបធៀបកម្រិតមូលដ្ឋាន។",
                    )
                )
            else:
                st.metric(
                    ui("Estimated visitors", "ចំនួនភ្ញៀវប៉ាន់ស្មាន"),
                    f"{scenario_result['prediction']:,.0f}",
                    border=False,
                )
                has_actual = scenario_result.get("actual_arrivals") is not None
                result_metrics = st.columns(3 if has_actual else 2)
                result_metrics[0].metric(
                    ui("Baseline", "កម្រិតមូលដ្ឋាន"),
                    f"{scenario_result['baseline']:,.0f}",
                )
                result_metrics[1].metric(
                    ui("Difference", "គម្លាត"),
                    f"{scenario_result['deviation']:+.2f}%",
                    delta=f"{scenario_result['deviation']:+.2f}%",
                )
                if has_actual:
                    result_metrics[2].metric(
                        ui("Actual in training data", "ចំនួនពិតក្នុងទិន្នន័យបណ្តុះបណ្តាល"),
                        f"{scenario_result['actual_arrivals']:,.0f}",
                    )
                st.caption(
                    f"{ui('Demand signal', 'សញ្ញាតម្រូវការ')}: "
                    f"{status_labels[scenario_result['status']]}"
                )
                chart_labels = [
                    ui("Baseline", "កម្រិតមូលដ្ឋាន"),
                    (
                        ui("Historical prediction", "ការព្យាករណ៍ប្រវត្តិសាស្ត្រ")
                        if has_actual
                        else ui("Scenario", "សេណារីយ៉ូ")
                    ),
                ]
                chart_values = [
                    scenario_result["baseline"],
                    scenario_result["prediction"],
                ]
                chart_colors = [
                    CHART_COLORS["baseline"],
                    CHART_COLORS["forecast"],
                ]
                if has_actual:
                    chart_labels.append(ui("Actual", "ចំនួនពិត"))
                    chart_values.append(scenario_result["actual_arrivals"])
                    chart_colors.append(CHART_COLORS["normal"])
                scenario_figure = go.Figure(
                    go.Bar(
                        x=chart_labels,
                        y=[value / 1000 for value in chart_values],
                        marker_color=chart_colors,
                        text=[f"{value:,.0f}" for value in chart_values],
                        textposition="outside",
                        textfont=dict(color=CHART_TEXT_COLOR, size=12),
                    )
                )
                chart_max = max(
                    chart_values,
                ) / 1000
                chart_upper_bound = max(100, int(np.ceil(chart_max * 1.25 / 100) * 100))
                scenario_figure.update_layout(
                    height=250,
                    showlegend=False,
                    yaxis=dict(
                        title=ui("Visitors (K)", "ភ្ញៀវ (ពាន់)"),
                        range=[0, chart_upper_bound],
                        dtick=100,
                        tickformat=",.0f",
                        gridcolor="#e7edf5",
                        zeroline=False,
                    ),
                    margin=dict(l=44, r=12, t=48, b=30),
                )
                style_chart(scenario_figure)
                st.plotly_chart(
                    scenario_figure,
                    width="stretch",
                    config={"displaylogo": False},
                )
                deviation = scenario_result["deviation"]
                direction = (
                    "above" if deviation > 0 else "below" if deviation < 0 else "at"
                )
                direction_kh = (
                    "លើស" if deviation > 0 else "ក្រោម" if deviation < 0 else "ស្មើនឹង"
                )
                insight_en = (
                    f"The estimate is {abs(deviation):.1f}% {direction} the seasonal "
                    f"baseline ({scenario_result['prediction'] - scenario_result['baseline']:+,.0f} visitors)."
                )
                insight_kh = (
                    f"ការប៉ាន់ស្មាន {abs(deviation):.1f}% {direction_kh}កម្រិតមូលដ្ឋានតាមរដូវ "
                    f"({scenario_result['prediction'] - scenario_result['baseline']:+,.0f} នាក់)។"
                )
                if scenario_result.get("actual_ape") is not None:
                    insight_en += (
                        f" Historical absolute percentage error: "
                        f"{scenario_result['actual_ape']:.1f}%."
                    )
                    insight_kh += (
                        f" កំហុសភាគរយដាច់ខាតធៀបចំនួនពិត៖ "
                        f"{scenario_result['actual_ape']:.1f}%。"
                    )
                render_chart_insight(insight_en, insight_kh)
                st.info(
                    ui(
                        (
                            "This historical backtest trained only on earlier months. The observed count is shown for comparison; a prediction is an estimate and may differ."
                            if has_actual
                            else "Exploratory estimate only; this is not a measured policy impact."
                        ),
                        (
                            "ការផ្ទៀងផ្ទាត់នេះបណ្តុះបណ្តាលតែលើខែមុនប៉ុណ្ណោះ។ បង្ហាញចំនួនពិតសម្រាប់ប្រៀបធៀប ហើយលទ្ធផលព្យាករណ៍អាចខុសពីចំនួនពិត។"
                            if has_actual
                            else "ជាការប៉ាន់ស្មានសម្រាប់ស្វែងយល់ មិនមែនជាផលប៉ះពាល់គោលនយោបាយដែលបានវាស់វែងទេ។"
                        ),
                    )
                )
                if scenario_result.get("actual_ape") is not None:
                    st.caption(
                        ui(
                            f"Historical absolute percentage error: {scenario_result['actual_ape']:.2f}%",
                            f"កំហុសភាគរយដាច់ខាតធៀបចំនួនពិត: {scenario_result['actual_ape']:.2f}%",
                        )
                    )
            with st.expander(ui("Inputs and assumptions", "ធាតុបញ្ចូល និងការសន្មត")):
                st.write(
                    f"{ui('Weather', 'អាកាសធាតុ')}: {sim_temp:.1f} °C · {sim_rain:.0f} mm"
                )
                st.write(
                    f"{ui('Calendar', 'ប្រតិទិន')}: "
                    f"{sim_kh_holidays} Cambodia · {sim_origin_holidays} source-market · "
                    f"{sim_china_holidays} China holidays"
                )

    if SCENARIO_TRANSACTIONS_PATH.exists():
        with st.container(border=True):
            st.subheader(
                ui("Saved scenario transactions", "ប្រវត្តិប្រតិបត្តិការសេណារីយ៉ូ")
            )
            transactions_df = pd.read_csv(SCENARIO_TRANSACTIONS_PATH)
            st.caption(
                ui(
                    f"{len(transactions_df)} scenario run(s) saved in "
                    f"{SCENARIO_TRANSACTIONS_PATH.relative_to(PROJECT_ROOT)}.",
                    f"បានរក្សាទុកសេណារីយ៉ូចំនួន {len(transactions_df)} ក្នុង "
                    f"{SCENARIO_TRANSACTIONS_PATH.relative_to(PROJECT_ROOT)}។",
                )
            )
            rows_to_show = st.selectbox(
                ui("Rows to display", "ចំនួនជួរដេកដែលត្រូវបង្ហាញ"),
                options=[5, 10, 25, 50, 100],
                index=4,
                key="scenario_transaction_rows",
            )
            st.dataframe(
                transactions_df.tail(rows_to_show),
                width="stretch",
                height=420,
            )
            st.download_button(
                ui("Download transaction CSV", "ទាញយក CSV ប្រតិបត្តិការ"),
                data=SCENARIO_TRANSACTIONS_PATH.read_bytes(),
                file_name=SCENARIO_TRANSACTIONS_PATH.name,
                mime="text/csv",
                icon=":material/download:",
            )

# -------------------------------------------------------------------
# TAB 5: DATA LINEAGE & PIPELINE ARCHITECTURE
# -------------------------------------------------------------------
elif current_tab == "technical":
    st.markdown(f"<div class='mot-section-header'>{ui('Data Pipelines, Features & Specifications', 'បំពង់ទិន្នន័យ លក្ខណៈពិសេស និងលក្ខណៈបច្ចេកទេស')}</div>", unsafe_allow_html=True)

    history_start = history_df["Date"].min().strftime("%B %Y")
    history_end = history_df["Date"].max().strftime("%B %Y")
    model_features = model_artifact.get("features", [])
    col_meta, col_feat = st.columns([1, 1.3])
    with col_meta:
        st.markdown(
            f"""
            ### Technical Specification
            * **Training CSV:** `{HISTORY_PATH.name}`
            * **Primary target:** `International_Tourist_Arrivals` (monthly)
            * **Training observations:** {len(history_df)} ({history_start} – {history_end})
            * **Model artifact:** `{MODEL_PATH.name}`
            * **Model input features:** {len(model_features)}
            * **Validation:** chronological holdout; {holdout_training_count} training rows and {holdout_months} later test months
            * **Holdout MAPE / R²:** {holdout_mape_text} / {holdout_r2_text}
            * **Prediction source:** saved LightGBM model configuration retrained on the training CSV's pre-holdout rows
            * **Baseline source:** `Seasonal_Baseline` from the EWS CSV; missing values use training-data `Lag_12`
            * **Ignored EWS fields:** precomputed forecast, error, and alert columns; these are recalculated from the trained model
            * **Serving API:** FastAPI REST endpoint (used when available; local model used otherwise)
            """
        )
    
    with col_feat:
        st.markdown(
            "### Model feature contract\n"
            + "\n".join(f"* `{feature}`" for feature in model_features)
        )

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