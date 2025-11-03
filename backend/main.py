import logging
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.analytics import (
    load_data_and_resample,
    load_volume_and_vwap,
    rolling_adf_pvalues,
    calculate_ols_hedge_ratio,
    calculate_huber_hedge_ratio,
    calculate_kalman_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    calculate_rolling_correlation,
    run_adf_test,
)
from backend.backtest import run_simple_backtest
from backend.database import Subscription, SYNC_DB_URL
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Real-Time Quant Analytics API")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/analytics")
async def analytics(
    symbol_a: str = Query(..., description="Symbol A, e.g., btcusdt"),
    symbol_b: str = Query(..., description="Symbol B, e.g., ethusdt"),
    timeframe: str = Query("1s", description="Resample timeframe: 1s, 1m, 5m"),
    rolling_window: int = Query(50, ge=2, description="Rolling window for correlation and z-score context"),
    regression_type: str = Query("ols", description="Regression type: ols | huber | kalman"),
    use_ohlc: bool = Query(False, description="Use uploaded OHLC data instead of live ticks"),
    adf_window: int = Query(0, ge=0, description="If >0, return rolling ADF p-values on spread with this window"),
    lookback_hours: int = Query(6, ge=1, le=168, description="Limit DB scan to recent N hours"),
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc, lookback_hours)

    # Cold start guard
    if df.empty or len(df) < rolling_window:
        return {
            "price_data": {"index": [], symbol_a.lower(): [], symbol_b.lower(): []},
            "spread": [],
            "zscore": [],
            "rolling_corr": [],
            "hedge_ratio": None,
            "latest_zscore": None,
        }

    rtype = regression_type.lower()
    if rtype == "huber":
        hr = calculate_huber_hedge_ratio(df, symbols[0], symbols[1])
    elif rtype == "kalman":
        hr = calculate_kalman_hedge_ratio(df, symbols[0], symbols[1])
    else:
        hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1])

    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    z = calculate_zscore(spread)
    rc = calculate_rolling_correlation(df, symbols[0], symbols[1], rolling_window)

    # Volume/VWAP
    metrics = load_volume_and_vwap(symbols, timeframe, use_ohlc, lookback_hours)

    # Rolling ADF p-values if requested
    adf_vals = None
    if adf_window and adf_window > 0:
        adf_vals = rolling_adf_pvalues(spread, window=adf_window).reindex(df.index).tolist()

    idx = [i.isoformat() for i in df.index]
    result = {
        "price_data": {
            "index": idx,
            symbols[0]: df[symbols[0]].tolist(),
            symbols[1]: df[symbols[1]].tolist(),
        },
        "spread": spread.tolist(),
        "zscore": z.tolist(),
        "rolling_corr": rc.tolist(),
        "hedge_ratio": hr,
        "latest_zscore": float(z.iloc[-1]) if len(z) else None,
        "volume": {s: (metrics.get(s, {}).get("volume", pd.Series(index=df.index)).reindex(df.index).fillna(0).tolist()) for s in symbols},
        "vwap": {s: (metrics.get(s, {}).get("vwap", pd.Series(index=df.index)).reindex(df.index).tolist()) for s in symbols},
        "adf_pvalues": adf_vals,
    }
    logger.info("API: Returning analytics data.")
    return result


@app.get("/api/adf_test")
async def adf_test(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query("1s"),
    use_ohlc: bool = Query(False),
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc)
    if df.empty:
        return {"pvalue": None, "statistic": None}
    hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1])
    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    res = run_adf_test(spread)
    return res


@app.get("/api/backtest")
async def backtest(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query("1s"),
    entry_z: float = Query(2.0),
    exit_z: float = Query(0.0),
    regression_type: str = Query("ols"),
    use_ohlc: bool = Query(False),
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc)
    if df.empty:
        return {"index": [], "pnl": []}

    rtype = regression_type.lower()
    if rtype == "huber":
        hr = calculate_huber_hedge_ratio(df, symbols[0], symbols[1])
    elif rtype == "kalman":
        hr = calculate_kalman_hedge_ratio(df, symbols[0], symbols[1])
    else:
        hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1])

    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    z = calculate_zscore(spread)
    pnl = run_simple_backtest(z, entry_z=entry_z, exit_z=exit_z)
    return {"index": [i.isoformat() for i in pnl.index], "pnl": pnl.tolist()}


@app.post("/api/upload_ohlc")
async def upload_ohlc(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload OHLC CSV with columns: timestamp, symbol, open, high, low, close, volume."""
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"timestamp", "symbol", "close"}
        if not required.issubset(set(df.columns)):
            return {"ok": False, "error": f"CSV must include at least {sorted(required)}"}
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        # Keep only known columns
        keep = [c for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        # Write to SQLite (append)
        from sqlalchemy import create_engine
        from backend.database import SYNC_DB_URL

        engine = create_engine(SYNC_DB_URL)
        df.to_sql("ohlc", con=engine, if_exists="append", index=False)
        return {"ok": True, "rows": int(len(df))}
    except Exception as e:
        logger.exception("Upload failed")
        return {"ok": False, "error": str(e)}


@app.get("/api/heatmap")
async def heatmap(
    symbols: str = Query(..., description="Comma-separated symbols"),
    timeframe: str = Query("1m"),
    window: int = Query(60, ge=3),
    use_ohlc: bool = Query(False),
) -> Dict[str, Any]:
    syms = [s.strip().lower() for s in symbols.split(",") if s.strip()]
    df = load_data_and_resample(syms, timeframe, use_ohlc)
    if df.empty:
        return {"symbols": [], "matrix": []}
    rets = df.pct_change().dropna()
    if len(rets) > window:
        rets = rets.tail(window)
    corr = rets.corr()
    return {"symbols": corr.columns.tolist(), "matrix": corr.values.tolist()}


@app.get("/api/subscriptions")
async def get_subscriptions() -> Dict[str, Any]:
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        rows = s.query(Subscription).all()
        syms = [r.symbol for r in rows]
    return {"symbols": syms}


@app.post("/api/subscriptions")
async def set_subscriptions(symbols: str = Query(..., description="Comma-separated symbols, e.g., btcusdt,ethusdt")) -> Dict[str, Any]:
    syms = sorted(set([s.strip().lower() for s in symbols.split(",") if s.strip()]))
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        s.execute(delete(Subscription))
        for sym in syms:
            s.add(Subscription(symbol=sym))
        s.commit()
    return {"ok": True, "symbols": syms}
