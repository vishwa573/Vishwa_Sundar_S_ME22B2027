import io
import json
import math
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Real-Time Quant Dashboard")

st.sidebar.header("Controls")
symbol_a = st.sidebar.text_input("Symbol A", value="btcusdt").lower()
symbol_b = st.sidebar.text_input("Symbol B", value="ethusdt").lower()
timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"], index=0)
rolling_window = st.sidebar.slider("Rolling Window", min_value=10, max_value=200, value=50)
z_alert = st.sidebar.number_input("Z-Score Alert Threshold", value=2.0)
run_adf = st.sidebar.button("Run ADF Test on Spread")

st_autorefresh(interval=5000)

API_BASE = "http://localhost:8000"

# Sidebar: ADF test under the button
adf_box = st.sidebar.container()
with adf_box:
    if run_adf:
        try:
            r = requests.get(
                f"{API_BASE}/api/adf_test",
                params={"symbol_a": symbol_a, "symbol_b": symbol_b, "timeframe": timeframe},
                timeout=10,
            )
            r.raise_for_status()
            res = r.json()
            st.sidebar.info(f"ADF Statistic: {res.get('statistic')}, p-value: {res.get('pvalue')}")
        except Exception as e:
            st.sidebar.error(f"ADF test failed: {e}")

@st.cache_data(ttl=2)
def fetch_analytics(symbol_a: str, symbol_b: str, timeframe: str, rolling_window: int):
    params = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "timeframe": timeframe,
        "rolling_window": rolling_window,
    }
    r = requests.get(f"{API_BASE}/api/analytics", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def to_dataframe(data: dict) -> pd.DataFrame:
    pd_idx = pd.to_datetime(data["price_data"].get("index", []))
    sa = data["price_data"].get(symbol_a, [])
    sb = data["price_data"].get(symbol_b, [])
    df = pd.DataFrame(
        {
            symbol_a: sa,
            symbol_b: sb,
            "spread": data.get("spread", []),
            "zscore": data.get("zscore", []),
            "rolling_corr": data.get("rolling_corr", []),
        },
        index=pd_idx,
    )
    return df


st.title("Real-Time Quantitative Analytics Dashboard")

try:
    data = fetch_analytics(symbol_a, symbol_b, timeframe, rolling_window)
except Exception as e:
    st.error(f"Failed to fetch analytics: {e}")
    st.stop()

latest_z = data.get("latest_zscore")
hr_value = data.get("hedge_ratio")
hr_display = None if hr_value is None else f"{hr_value:.4f}"

# Build DataFrame
frame = to_dataframe(data)
if frame.empty:
    st.info("Waiting for data...")
    st.stop()

# Charts
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=frame.index, y=frame[symbol_a], name=symbol_a.upper()))
fig_price.add_trace(go.Scatter(x=frame.index, y=frame[symbol_b], name=symbol_b.upper()))
fig_price.update_layout(title="Prices", xaxis_title="Time", yaxis_title="Price", legend_title="Symbols")

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=frame.index, y=frame["spread"], name="Spread"))
fig_spread.update_layout(title="Spread", xaxis_title="Time", yaxis_title="Spread")

fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=frame.index, y=frame["zscore"], name="Z-Score"))
fig_z.update_layout(title="Z-Score", xaxis_title="Time", yaxis_title="Z-Score")

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=frame.index, y=frame["rolling_corr"], name="Rolling Corr"))
fig_corr.update_layout(title="Rolling Correlation", xaxis_title="Time", yaxis_title="Correlation")

# Tabs layout
tab_dash, tab_charts, tab_data = st.tabs(["Dashboard", "Detailed Charts", "Data View"])

with tab_dash:
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Latest Z-Score", None if latest_z is None else f"{latest_z:.2f}")
    with m2:
        st.metric("Hedge Ratio", hr_display)
    if latest_z is not None and abs(latest_z) >= z_alert:
        st.warning(f"Z-Score {latest_z:.2f} exceeds threshold {z_alert}")
    st.plotly_chart(fig_price, width='stretch')
    st.plotly_chart(fig_z, width='stretch')

with tab_charts:
    st.plotly_chart(fig_spread, width='stretch')
    st.plotly_chart(fig_corr, width='stretch')

with tab_data:
    st.dataframe(frame)
    csv_bytes = frame.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="analytics.csv",
        mime="text/csv",
    )
