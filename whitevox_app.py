"""
Whitevox Revenue Forecast — Streamlit app
------------------------------------------
Loads the pre-trained artifacts produced by `whitevox_forecasting.ipynb`
(does NOT retrain here) and lets the user explore a revenue forecast for the
whole company or a single service line, with optional spend/leads scenarios.

Run locally:
    streamlit run app.py

Keep in sync: if you change the feature engineering or retrain the models in
the notebook, just re-run the notebook to refresh the files under artifacts/ —
this app picks them up automatically on next launch (no code changes needed
unless you change feature *names*, in which case update forecasting_lib.py,
which both the notebook and this app import).
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting_lib as flib

st.set_page_config(page_title="Whitevox Revenue Forecast", page_icon="\U0001F4C8", layout="wide")

ARTIFACT_DIR = "artifacts"
REQUIRED_ARTIFACTS = [
    "company_model.pkl", "company_trend.pkl",
    "service_model.pkl", "service_trends.pkl", "metrics.json",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_ARTIFACTS if not os.path.exists(f"{ARTIFACT_DIR}/{f}")]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s): {missing}. Run whitevox_forecasting.ipynb first "
            f"to generate them into '{ARTIFACT_DIR}/'."
        )
    return {
        "company_model": joblib.load(f"{ARTIFACT_DIR}/company_model.pkl"),
        "company_trend": joblib.load(f"{ARTIFACT_DIR}/company_trend.pkl"),
        "service_model": joblib.load(f"{ARTIFACT_DIR}/service_model.pkl"),
        "service_trends": joblib.load(f"{ARTIFACT_DIR}/service_trends.pkl"),
        "metrics": flib.load_json(f"{ARTIFACT_DIR}/metrics.json"),
    }


@st.cache_data
def load_data():
    return flib.load_company_df(), flib.load_service_df()


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Load everything up front; show a clear error if artifacts are missing
# ---------------------------------------------------------------------------
st.title("Whitevox Revenue Forecast")
st.caption(
    "Forecast built from a trend + XGBoost-residual model trained in "
    "`whitevox_forecasting.ipynb`. See the sidebar for controls."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df_company, df_service = load_data()
service_lines = flib.get_service_lines(df_service)
metrics = artifacts["metrics"]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    view = st.selectbox("View", ["Company-wide total"] + service_lines)
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)

    st.subheader("Scenario (optional)")
    st.caption("Adjust projected future spend/leads vs. their trailing-12-month average.")
    spend_pct = st.slider("Marketing spend change (%)", -50, 100, 0, step=5)
    leads_pct = st.slider("Lead volume change (%)", -50, 100, 0, step=5)

    st.divider()
    with st.expander("Model performance (from notebook holdout)"):
        st.write("12-month holdout comparison (company level):")
        st.dataframe(pd.DataFrame(metrics["model_comparison"]).T, width='stretch')
        st.caption(
            f"Deployed model: {metrics.get('deployed_model', 'XGBoost (trend+residual)')}. "
            f"Per-service-line model MAPE: {metrics['service_model_metrics']['MAPE_%']}%. "
            f"Trained through {metrics['trained_through_date']}."
        )

# ---------------------------------------------------------------------------
# Build forecast for the selected view
# ---------------------------------------------------------------------------
if view == "Company-wide total":
    hist = df_company[["date", "total_revenue_usd"]].rename(columns={"total_revenue_usd": "revenue"})
    forecast = flib.recursive_forecast_company(
        artifacts["company_model"], artifacts["company_trend"], df_company,
        horizon=horizon, spend_pct_change=spend_pct, leads_pct_change=leads_pct,
    )
    resid_std = metrics.get("holdout_residual_std_usd", 0.0)
else:
    line_df = df_service[df_service["service_line"] == view]
    hist = line_df[line_df["revenue_usd"] > 0][["date", "revenue_usd"]].rename(columns={"revenue_usd": "revenue"})
    forecast = flib.recursive_forecast_service_line(
        artifacts["service_model"], artifacts["service_trends"], df_service, view,
        horizon=horizon, spend_pct_change=spend_pct, leads_pct_change=leads_pct,
    )
    resid_std = None  # per-line interval omitted; only company-level interval is calibrated in the notebook

forecast = forecast.rename(columns={"forecast_revenue_usd": "revenue"})

if resid_std:
    forecast["lower_80"] = (forecast["revenue"] - 1.28 * resid_std).clip(lower=0)
    forecast["upper_80"] = forecast["revenue"] + 1.28 * resid_std

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
next_month_val = forecast["revenue"].iloc[0]
cum_val = forecast["revenue"].sum()
last_actual = hist["revenue"].iloc[-1]
same_month_last_year = hist["revenue"].iloc[-12] if len(hist) >= 12 else np.nan
yoy_growth = (next_month_val / same_month_last_year - 1) * 100 if same_month_last_year and not np.isnan(same_month_last_year) else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last actual month", fmt_usd(last_actual))
c2.metric(f"Next month forecast ({forecast['date'].iloc[0]:%b %Y})", fmt_usd(next_month_val))
c3.metric(f"{horizon}-month cumulative forecast", fmt_usd(cum_val))
c4.metric("YoY growth (next month vs. same month last year)", f"{yoy_growth:+.1f}%" if not np.isnan(yoy_growth) else "n/a")

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["revenue"], name="Actual", mode="lines", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["revenue"], name="Forecast", mode="lines+markers",
                          line=dict(color="#d62728", dash="dash")))
if "upper_80" in forecast.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["date"], forecast["date"][::-1]]),
        y=pd.concat([forecast["upper_80"], forecast["lower_80"][::-1]]),
        fill="toself", fillcolor="rgba(214,39,40,0.15)", line=dict(color="rgba(255,255,255,0)"),
        name="80% interval", showlegend=True,
    ))
fig.update_layout(
    title=f"{view} — revenue history and {horizon}-month forecast",
    yaxis_title="USD", xaxis_title=None, height=460, margin=dict(t=50, b=20),
)
st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Table + download
# ---------------------------------------------------------------------------
st.subheader("Forecast table")
display_df = forecast.copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
for col in ["revenue", "lower_80", "upper_80"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(0)
st.dataframe(display_df, width='stretch', hide_index=True)

csv_bytes = forecast.assign(date=forecast["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download forecast as CSV",
    data=csv_bytes,
    file_name=f"whitevox_forecast_{view.replace(' ', '_').replace('/', '-')}.csv",
    mime="text/csv",
)

st.caption(
    "Synthetic demo data. Future leads/spend are projected from the trailing "
    "12-month average and adjusted by the scenario sliders above; they are not "
    "real forward-looking commitments. Note: in this synthetic dataset, leads "
    "and spend were generated as *consequences* of revenue rather than causes "
    "of it, so scenario adjustments here move the forecast only modestly. On "
    "real operating data, where spend/leads genuinely lead revenue, the same "
    "scenario sliders should show a stronger effect."
)
