import io
import json
import math
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Real-Time Quant Dashboard")

# Custom styling
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div { background: #0b1220; }
    .sidebar-title { font-weight: 700; font-size: 1rem; margin: 0.25rem 0 0.5rem 0; }
    .badge-live { display:inline-block; padding:2px 8px; border-radius:12px; background:#1f6feb; color:#fff; font-size:0.75rem; margin-left:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### ⚙️ Control Panel")

# Pair & Symbols
with st.sidebar.expander("🎯 Pair & Symbols", expanded=True):
    # Bind to session state for swap/presets
    if "sym_a" not in st.session_state:
        st.session_state.sym_a = "btcusdt"
    if "sym_b" not in st.session_state:
        st.session_state.sym_b = "ethusdt"
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sym_a = st.text_input("Symbol A", value=st.session_state.sym_a).lower()
    with c2:
        st.session_state.sym_b = st.text_input("Symbol B", value=st.session_state.sym_b).lower()
    colx, coly = st.columns([1,2])
    if colx.button("Swap A/B"):
        st.session_state.sym_a, st.session_state.sym_b = st.session_state.sym_b, st.session_state.sym_a
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    preset = coly.selectbox("Preset", ["—", "BTC/ETH", "BTC/BNB", "ETH/BNB"], index=0)
    if preset != "—" and st.button("Apply preset"):
        pa, pb = ("btcusdt","ethusdt") if preset=="BTC/ETH" else ("btcusdt","bnbusdt") if preset=="BTC/BNB" else ("ethusdt","bnbusdt")
        st.session_state.sym_a, st.session_state.sym_b = pa, pb
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    symbol_a = st.session_state.sym_a
    symbol_b = st.session_state.sym_b

# Time & Model
with st.sidebar.expander("⏱️ Time & Model", expanded=True):
    timeframe = st.radio("Timeframe", ["1s", "1m", "5m"], index=0, horizontal=True)
    regression_type = st.selectbox("Regression", ["OLS", "Huber", "Kalman"], index=0)
    rolling_window = st.slider("Rolling Window", min_value=10, max_value=200, value=50)

# Refresh
with st.sidebar.expander("🔄 Refresh", expanded=True):
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_ms = st.slider("Refresh interval (ms)", min_value=500, max_value=5000, value=1500, step=100)

# Analytics
with st.sidebar.expander("📊 Analytics", expanded=False):
    use_ohlc = st.checkbox("Use uploaded OHLC data", value=False)
    show_volume = st.checkbox("Show Volume/VWAP (slower)", value=False)
    max_ticks = st.number_input("Max ticks per symbol", min_value=500, max_value=200000, value=20000, step=500)
    z_alert = st.number_input("Z-Score Alert Threshold", value=2.0)
    min_corr = st.slider("Min Rolling Corr (alerts)", min_value=-1.0, max_value=1.0, value=-0.2, step=0.05)
    hyst = st.slider("Alert Hysteresis", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    min_vol = st.number_input("Min Volume per bar (alerts)", value=0.0, step=1.0)
    lookback_hours = st.number_input("Lookback Hours", min_value=1, max_value=168, value=6)
    adf_win = st.slider("Rolling ADF Window (0=off)", min_value=0, max_value=300, value=0, step=10)
    run_adf = st.button("Run ADF Test on Spread")

# Auto-refresh (toggleable)
if auto_refresh:
    st_autorefresh(interval=int(refresh_ms))

API_BASE = "http://localhost:8000"

# Sidebar: Upload OHLC CSV with preview
uploaded = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"], key="ohlc_upload")
if uploaded is not None:
    try:
        tmp_df = pd.read_csv(uploaded)
        st.sidebar.write("Preview:")
        st.sidebar.dataframe(tmp_df.head())
        if "symbol" in tmp_df.columns:
            syms = sorted(tmp_df["symbol"].astype(str).str.lower().unique().tolist())
            sel = st.sidebar.multiselect("Select symbols to ingest", options=syms, default=syms)
            if st.sidebar.button("Ingest Selected"):
                filt = tmp_df[tmp_df["symbol"].astype(str).str.lower().isin(sel)]
                csv_bytes = filt.to_csv(index=False).encode("utf-8")
                r = requests.post(f"{API_BASE}/api/upload_ohlc", files={"file": csv_bytes})
                res = r.json()
                if res.get("ok"):
                    st.sidebar.success(f"Uploaded {res.get('rows')} rows to OHLC store")
                else:
                    st.sidebar.error(f"Upload failed: {res.get('error')}")
        else:
            st.sidebar.warning("CSV must include a 'symbol' column")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# Sidebar: Alerts
st.sidebar.markdown("---")
st.sidebar.subheader("Alerts")
with st.sidebar.form("alert_form", clear_on_submit=True):
    a_name = st.text_input("Name", value="rule1")
    a_enable = st.checkbox("Enabled", value=True)
    a_webhook = st.text_input("Webhook URL (optional)", value="")
    submitted = st.form_submit_button("Add alert for current pair")
    if submitted:
        try:
            payload = {
                "name": a_name, "enabled": a_enable,
                "symbol_a": symbol_a, "symbol_b": symbol_b, "timeframe": timeframe,
                "z_threshold": float(z_alert), "corr_min": float(min_corr),
                "min_vol": float(min_vol), "hysteresis": float(hyst),
                "webhook_url": (a_webhook or None),
            }
            r = requests.post(f"{API_BASE}/api/alerts", json=payload, timeout=120)
            st.success("Alert added") if r.ok else st.error("Failed")
        except Exception as e:
            st.error(f"Add failed: {e}")

load_alerts = st.sidebar.checkbox("Show alerts list", value=False)
if load_alerts:
    try:
        al = requests.get(f"{API_BASE}/api/alerts", timeout=120).json().get("alerts", [])
        for rule in al:
            cols = st.sidebar.columns([3,1,1])
            cols[0].write(f"{rule['name']} ({rule['symbol_a']}-{rule['symbol_b']} {rule['timeframe']})")
            if cols[1].button("Toggle", key=f"tgl_{rule['id']}"):
                try:
                    requests.patch(f"{API_BASE}/api/alerts/{rule['id']}", json={"enabled": not rule['enabled']}, timeout=10)
                    st.sidebar.success("Toggled")
                except Exception as e:
                    st.sidebar.error(f"Toggle failed: {e}")
            if cols[2].button("Delete", key=f"del_{rule['id']}"):
                try:
                    requests.delete(f"{API_BASE}/api/alerts/{rule['id']}", timeout=120)
                    st.sidebar.success("Deleted")
                except Exception as e:
                    st.sidebar.error(f"Delete failed: {e}")
    except Exception:
        pass

# Sidebar: ADF test under the button
adf_box = st.sidebar.container()
with adf_box:
    if run_adf:
        try:
            r = requests.get(
                f"{API_BASE}/api/adf_test",
                params={"symbol_a": symbol_a, "symbol_b": symbol_b, "timeframe": timeframe, "use_ohlc": use_ohlc},
                timeout=10,
            )
            r.raise_for_status()
            res = r.json()
            st.sidebar.info(f"ADF Statistic: {res.get('statistic')}, p-value: {res.get('pvalue')}")
        except Exception as e:
            st.sidebar.error(f"ADF test failed: {e}")

# Sidebar: Dynamic subscriptions
st.sidebar.markdown("---")
st.sidebar.subheader("Live Subscriptions")

@st.cache_data(ttl=10)
def get_subscriptions_cached():
    try:
        return requests.get(f"{API_BASE}/api/subscriptions", timeout=5).json().get("symbols", [])
    except Exception:
        return []

sub = get_subscriptions_cached()
all_syms = sorted(list(set(sub + ["btcusdt","ethusdt","bnbusdt","xrpusdt","adausdt","solusdt","ltcusdt","dogeusdt"])) )
sel_subs = st.sidebar.multiselect("Subscribe symbols", options=all_syms, default=sub)
colA, colB = st.sidebar.columns([1,1])
if colA.button("Apply Subscriptions"):
    try:
        requests.post(f"{API_BASE}/api/subscriptions", params={"symbols": ",".join(sel_subs)}, timeout=120)
        st.sidebar.success("Updated subscriptions (ingester will adjust within ~5s)")
    except Exception as e:
        st.sidebar.error(f"Failed to update subscriptions: {e}")
if colB.button("Refresh List"):
    get_subscriptions_cached.clear()

@st.cache_data(ttl=1)
def fetch_analytics(symbol_a: str, symbol_b: str, timeframe: str, rolling_window: int, regression_type: str, use_ohlc: bool, adf_window: int, lookback_hours: int, include_volume: bool, max_ticks: int):
    params = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "timeframe": timeframe,
        "rolling_window": rolling_window,
        "regression_type": regression_type.lower(),
        "use_ohlc": json.dumps(use_ohlc),
        "adf_window": adf_window,
        "lookback_hours": int(lookback_hours),
        "include_volume": json.dumps(include_volume),
        "max_ticks": int(max_ticks),
    }
    r = requests.get(f"{API_BASE}/api/analytics", params=params, timeout=120)
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

# Quick live panel (lighter payload)
with st.expander("Live quick panel"):
    enable_quick = st.checkbox("Enable quick panel", value=False)
    if enable_quick:
        try:
            q = requests.get(f"{API_BASE}/api/analytics_quick", params={"symbol_a": symbol_a, "symbol_b": symbol_b, "timeframe": timeframe, "lookback_hours": 1}, timeout=45)
            q.raise_for_status()
            qd = q.json()
            st.metric("Latest Z (quick)", None if qd.get("latest_zscore") is None else f"{qd['latest_zscore']:.2f}")
        except Exception as e:
            st.write(f"Quick panel error: {e}")

try:
    t0 = time.perf_counter()
    data = fetch_analytics(symbol_a, symbol_b, timeframe, rolling_window, regression_type, use_ohlc, adf_win, lookback_hours, show_volume, max_ticks)
    api_ms = (time.perf_counter() - t0) * 1000
    st.sidebar.subheader("Diagnostics")
    st.sidebar.text(f"API {api_ms:.0f} ms | points={len(data.get('price_data',{}).get('index',[]))}")
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
# VWAP overlays if available
vwap_map = data.get("vwap", {})
if vwap_map.get(symbol_a):
    fig_price.add_trace(go.Scatter(x=frame.index, y=vwap_map[symbol_a], name=f"{symbol_a.upper()} VWAP", line=dict(dash="dot")))
if vwap_map.get(symbol_b):
    fig_price.add_trace(go.Scatter(x=frame.index, y=vwap_map[symbol_b], name=f"{symbol_b.upper()} VWAP", line=dict(dash="dot")))
fig_price.update_layout(title="Prices", xaxis_title="Time", yaxis_title="Price", legend_title="Symbols", template="plotly_dark")

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=frame.index, y=frame["spread"], name="Spread"))
fig_spread.update_layout(title="Spread", xaxis_title="Time", yaxis_title="Spread", template="plotly_dark")

fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=frame.index, y=frame["zscore"], name="Z-Score"))
# Rolling ADF p-values sparkline (secondary)
pvals = data.get("adf_pvalues")
if pvals:
    fig_z.add_trace(go.Scatter(x=frame.index, y=pvals, name="ADF p-value", yaxis="y2", line=dict(color="orange")))
    fig_z.update_layout(yaxis2=dict(overlaying='y', side='right', title='p-value', range=[0,1]))
fig_z.update_layout(title="Z-Score", xaxis_title="Time", yaxis_title="Z-Score", template="plotly_dark")

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=frame.index, y=frame["rolling_corr"], name="Rolling Corr"))
fig_corr.update_layout(title="Rolling Correlation", xaxis_title="Time", yaxis_title="Correlation", template="plotly_dark")

# Volume chart (optional)
fig_vol = None
if show_volume:
    vol_map = data.get("volume", {})
    fig_vol = go.Figure()
    if vol_map.get(symbol_a):
        fig_vol.add_trace(go.Bar(x=frame.index, y=vol_map[symbol_a], name=f"{symbol_a.upper()} Vol", opacity=1.0))
    if vol_map.get(symbol_b):
        fig_vol.add_trace(go.Bar(x=frame.index, y=vol_map[symbol_b], name=f"{symbol_b.upper()} Vol", opacity=1.0))
    fig_vol.update_layout(title="Resampled Volume", xaxis_title="Time", yaxis_title="Volume", template="plotly_dark")

# Tabs layout
tab_dash, tab_charts, tab_heat, tab_data, tab_bt = st.tabs(["Dashboard", "Detailed Charts", "Heatmap", "Data View", "Backtest"])

with tab_dash:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Latest Z-Score", None if latest_z is None else f"{latest_z:.2f}")
    with m2:
        st.metric("Hedge Ratio", hr_display)
    with m3:
        st.metric("Rolling Corr (last)", None if frame["rolling_corr"].dropna().empty else f"{frame['rolling_corr'].dropna().iloc[-1]:.2f}")

    # Summary price stats (baseline requirement)
    try:
        stats_idx = [symbol_a.upper(), symbol_b.upper()]
        last_vals = [frame[symbol_a].iloc[-1], frame[symbol_b].iloc[-1]]
        mean_vals = [frame[symbol_a].mean(), frame[symbol_b].mean()]
        std_vals = [frame[symbol_a].std(), frame[symbol_b].std()]
        # 1m/5m pct changes via resample to 1min/5min
        re_1m = frame[[symbol_a, symbol_b]].resample("1min").last().pct_change().iloc[-1].fillna(0)
        re_5m = frame[[symbol_a, symbol_b]].resample("5min").last().pct_change().iloc[-1].fillna(0)
        stats_df = pd.DataFrame(
            {
                "Last": last_vals,
                "Mean": mean_vals,
                "Std": std_vals,
                "1m %": [re_1m.get(symbol_a, 0.0), re_1m.get(symbol_b, 0.0)],
                "5m %": [re_5m.get(symbol_a, 0.0), re_5m.get(symbol_b, 0.0)],
            },
            index=stats_idx,
        )
        st.dataframe(stats_df)
    except Exception:
        pass

    # Rule-based alerts with hysteresis
    if "alert_state" not in st.session_state:
        st.session_state.alert_state = {"z": False, "corr": False}
    z_now = latest_z if latest_z is not None else 0
    corr_now = frame["rolling_corr"].dropna().iloc[-1] if not frame["rolling_corr"].dropna().empty else 0
    # Latest volumes for gating
    last_vol_a = (data.get("volume", {}).get(symbol_a) or [None])[-1]
    last_vol_b = (data.get("volume", {}).get(symbol_b) or [None])[-1]
    liquid = (last_vol_a is not None and last_vol_b is not None and last_vol_a >= min_vol and last_vol_b >= min_vol)

    # Z-score rule (gated by liquidity)
    if liquid and abs(z_now) >= z_alert and not st.session_state.alert_state["z"]:
        st.session_state.alert_state["z"] = True
        st.warning(f"Z-Score {z_now:.2f} exceeds threshold {z_alert}")
        st.toast(f"Alert: |z| ≥ {z_alert}")
    elif st.session_state.alert_state["z"] and abs(z_now) <= max(0.0, z_alert - hyst):
        st.session_state.alert_state["z"] = False
    # Corr rule (low correlation, gated)
    if liquid and corr_now <= min_corr and not st.session_state.alert_state["corr"]:
        st.session_state.alert_state["corr"] = True
        st.warning(f"Rolling corr {corr_now:.2f} fell below {min_corr}")
    elif st.session_state.alert_state["corr"] and corr_now >= min_corr + hyst:
        st.session_state.alert_state["corr"] = False

    st.plotly_chart(fig_price, width='stretch')
    st.plotly_chart(fig_z, width='stretch')
    if fig_vol is not None:
        st.plotly_chart(fig_vol, width='stretch')

with tab_charts:
    st.plotly_chart(fig_spread, width='stretch')
    st.plotly_chart(fig_corr, width='stretch')

with tab_heat:
    st.subheader("Cross-correlation Heatmap")
    heat_syms = st.multiselect("Symbols", options=sel_subs or all_syms, default=(sel_subs[:6] if sel_subs else all_syms[:6]))
    heat_win = st.slider("Window (bars)", min_value=20, max_value=300, value=60, step=10)
    compute_hm = st.button("Compute heatmap")
    if compute_hm and heat_syms:
        try:
            r = requests.get(f"{API_BASE}/api/heatmap", params={"symbols": ",".join(heat_syms), "timeframe": timeframe, "window": heat_win, "use_ohlc": json.dumps(use_ohlc)}, timeout=120)
            mat = r.json()
            import plotly.express as px
            z = mat.get("matrix", [])
            x = mat.get("symbols", [])
            if z and x:
                fig_hm = px.imshow(z, x=x, y=x, color_continuous_scale="RdBu", origin="lower", zmin=-1, zmax=1)
                fig_hm.update_layout(title="Correlation Heatmap", template="plotly_dark")
                st.plotly_chart(fig_hm, width='stretch')
            else:
                st.info("Heatmap waiting for data…")
        except Exception as e:
            st.error(f"Heatmap failed: {e}")

with tab_data:
    st.dataframe(frame)
    csv_bytes = frame.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="analytics.csv",
        mime="text/csv",
    )
    
# Signals table for a few pairs
    st.subheader("Signals (latest)")
    compute_signals = st.checkbox("Compute signals (may be slow)", value=False)
    if compute_signals:
        import itertools
        pairs = list(itertools.combinations(sel_subs if sel_subs else all_syms[:4], 2))[:6]
        rows = []
        for a,b in pairs:
            try:
                x = fetch_analytics(a, b, timeframe, rolling_window, regression_type, use_ohlc, 0, lookback_hours)
                zlast = x.get("latest_zscore")
                corr_series = x.get("rolling_corr", [])
                corr_last = corr_series[-1] if corr_series else None
                rows.append({"pair": f"{a.upper()}-{b.upper()}", "z": zlast, "corr": corr_last, "hr": x.get("hedge_ratio")})
            except Exception:
                pass
        if rows:
            sig_df = pd.DataFrame(rows)
            st.dataframe(sig_df)
            st.download_button("Download Signals CSV", data=sig_df.to_csv(index=False).encode("utf-8"), file_name="signals.csv", mime="text/csv")

@st.cache_data(ttl=1)
def fetch_backtest(symbol_a: str, symbol_b: str, timeframe: str, entry_z: float, exit_z: float, regression_type: str, use_ohlc: bool):
    params = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "timeframe": timeframe,
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "regression_type": regression_type.lower(),
        "use_ohlc": json.dumps(use_ohlc),
    }
    r = requests.get(f"{API_BASE}/api/backtest", params=params, timeout=120)
    r.raise_for_status()
    return r.json()

with tab_bt:
    try:
        bt = fetch_backtest(symbol_a, symbol_b, timeframe, z_alert, 0.0, regression_type, use_ohlc)
        if bt.get("index"):
            bt_df = pd.DataFrame({"pnl": bt["pnl"]}, index=pd.to_datetime(bt["index"]))
            st.line_chart(bt_df["pnl"])
            m = bt.get("metrics", {})
            st.write({k: (None if v is None else round(v, 3)) for k, v in m.items()})
        else:
            st.info("Backtest waiting for data…")
    except Exception as e:
        st.error(f"Backtest failed: {e}")

    st.subheader("Parameter sweep")
    c1, c2, c3 = st.columns(3)
    e_min = c1.number_input("Entry min", value=1.0)
    e_max = c2.number_input("Entry max", value=3.0)
    e_step = c3.number_input("Entry step", value=0.5)
    exits = st.text_input("Exit z list", value="0.0,0.5")
    if st.button("Run grid"):
        try:
            r = requests.get(f"{API_BASE}/api/backtest_grid", params={
                "symbol_a": symbol_a, "symbol_b": symbol_b, "timeframe": timeframe,
                "entry_min": e_min, "entry_max": e_max, "entry_step": e_step, "exit_levels": exits,
                "regression_type": regression_type.lower(), "use_ohlc": json.dumps(use_ohlc)
            }, timeout=50)
            r.raise_for_status()
            res = r.json()
            st.write("Best:", res.get("best"))
            if res.get("grid"):
                st.dataframe(pd.DataFrame(res["grid"]))
        except Exception as e:
            st.error(f"Grid failed: {e}")
