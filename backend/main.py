import logging
import asyncio
from typing import Dict, Any, List

import pandas as pd
import httpx
from fastapi import FastAPI, Query, UploadFile, File, Body, HTTPException
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
from backend.backtest import run_simple_backtest, compute_metrics
from backend.database import Subscription, AlertRule, SYNC_DB_URL
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
    include_volume: bool = Query(False, description="Include volume/VWAP (extra DB read)"),
    max_ticks: int = Query(0, ge=0, description="Per-symbol tick cap (0=unbounded)"),
    evaluate_alerts: bool = Query(False, description="Evaluate alert rules (slower)"),
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc, lookback_hours, max_ticks)

    # Cold start guard: return whatever price data we have, defer analytics until enough points
    if df.empty:
        return {
            "price_data": {"index": [], symbol_a.lower(): [], symbol_b.lower(): []},
            "spread": [],
            "zscore": [],
            "rolling_corr": [],
            "hedge_ratio": None,
            "latest_zscore": None,
        }
    if len(df) < rolling_window:
        idx = [i.isoformat() for i in df.index]
        return {
            "price_data": {
                "index": idx,
                symbols[0]: df[symbols[0]].tolist(),
                symbols[1]: df[symbols[1]].tolist(),
            },
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

    # Volume/VWAP (optional)
    metrics = load_volume_and_vwap(symbols, timeframe, use_ohlc, lookback_hours, max_ticks) if include_volume else {}

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
        "volume": {s: (metrics.get(s, {}).get("volume", pd.Series(index=df.index)).reindex(df.index).fillna(0).tolist()) for s in symbols} if metrics else {},
        "vwap": {s: (metrics.get(s, {}).get("vwap", pd.Series(index=df.index)).reindex(df.index).tolist()) for s in symbols} if metrics else {},
        "adf_pvalues": adf_vals,
        "alerts_triggered": [],
    }

    # Evaluate alert rules (non-blocking webhook)
    if evaluate_alerts:
        try:
            eng = create_engine(SYNC_DB_URL)
            with Session(eng) as s:
                rules: List[AlertRule] = (
                    s.query(AlertRule)
                    .filter(AlertRule.enabled == True)
                    .filter(AlertRule.symbol_a == symbols[0])
                    .filter(AlertRule.symbol_b == symbols[1])
                    .filter(AlertRule.timeframe == timeframe)
                    .all()
                )
            latest_corr = float(rc.dropna().iloc[-1]) if len(rc.dropna()) else None
            latest_vol_a = (metrics.get(symbols[0], {}).get("volume", pd.Series(index=df.index)).reindex(df.index).iloc[-1]) if len(df) else None
            latest_vol_b = (metrics.get(symbols[1], {}).get("volume", pd.Series(index=df.index)).reindex(df.index).iloc[-1]) if len(df) else None
            liquid = (latest_vol_a is not None and latest_vol_b is not None)
            for r in rules:
                cond_z = result["latest_zscore"] is not None and abs(result["latest_zscore"]) >= r.z_threshold
                cond_corr = (latest_corr is not None) and (latest_corr <= r.corr_min)
                cond_vol = (not liquid) or (latest_vol_a >= r.min_vol and latest_vol_b >= r.min_vol)
                if cond_z and cond_corr and cond_vol:
                    result["alerts_triggered"].append({"id": r.id, "name": r.name})
                    if r.webhook_url:
                        async def _fire(url: str, payload: dict):
                            try:
                                async with httpx.AsyncClient(timeout=2.0) as client:
                                    await client.post(url, json=payload)
                            except Exception:
                                pass
                        asyncio.create_task(_fire(r.webhook_url, {"rule": r.name, "latest_z": result["latest_zscore"], "corr": latest_corr}))
        except Exception as e:
            logger.warning("Alert evaluation failed: %s", e)

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
    metrics = compute_metrics(pnl)
    return {"index": [i.isoformat() for i in pnl.index], "pnl": pnl.tolist(), "metrics": metrics}


@app.get("/api/backtest_grid")
async def backtest_grid(
    symbol_a: str = Query(...), symbol_b: str = Query(...), timeframe: str = Query("1s"),
    entry_min: float = Query(1.0), entry_max: float = Query(3.0), entry_step: float = Query(0.5),
    exit_levels: str = Query("0.0,0.5"), regression_type: str = Query("ols"), use_ohlc: bool = Query(False)
) -> Dict[str, Any]:
    import numpy as np
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc)
    if df.empty:
        return {"grid": [], "best": None}
    hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1]) if regression_type=="ols" else calculate_huber_hedge_ratio(df, symbols[0], symbols[1])
    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    z = calculate_zscore(spread)
    best = None
    grid = []
    exits = [float(x) for x in exit_levels.split(",") if x.strip()]
    e = entry_min
    while e <= entry_max + 1e-9:
        for ex in exits:
            pnl = run_simple_backtest(z, entry_z=float(e), exit_z=ex)
            m = compute_metrics(pnl)
            row = {"entry_z": float(e), "exit_z": ex, **m}
            grid.append(row)
            score = (m["sharpe"] or 0) + (-(m["max_dd"] or 0))
            if best is None or score > ((best.get("sharpe") or 0) + (-(best.get("max_dd") or 0))):
                best = row
        e = round(e + entry_step, 3)
    return {"grid": grid, "best": best}


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


@app.get("/api/analytics_quick")
async def analytics_quick(
    symbol_a: str = Query(...), symbol_b: str = Query(...), timeframe: str = Query("1s"), lookback_hours: int = Query(1)
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe, use_ohlc=False, lookback_hours=lookback_hours)
    if df.empty:
        return {"latest_zscore": None, "last": {symbols[0]: None, symbols[1]: None}}
    hr = calculate_ols_hedge_ratio(df.tail(200), symbols[0], symbols[1])
    spread = calculate_spread(df.tail(200), symbols[0], symbols[1], hr)
    z = calculate_zscore(spread)
    return {"latest_zscore": float(z.iloc[-1]) if len(z) else None, "last": {symbols[0]: float(df[symbols[0]].iloc[-1]), symbols[1]: float(df[symbols[1]].iloc[-1])}}


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


# Alerts CRUD
@app.get("/api/alerts")
async def list_alerts() -> Dict[str, Any]:
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        rows = s.query(AlertRule).all()
        out = [
            {"id": r.id, "name": r.name, "enabled": r.enabled, "symbol_a": r.symbol_a, "symbol_b": r.symbol_b,
             "timeframe": r.timeframe, "z_threshold": r.z_threshold, "corr_min": r.corr_min,
             "min_vol": r.min_vol, "hysteresis": r.hysteresis, "webhook_url": r.webhook_url}
            for r in rows
        ]
    return {"alerts": out}


@app.post("/api/alerts")
async def create_alert(
    name: str = Body(...), enabled: bool = Body(True), symbol_a: str = Body(...), symbol_b: str = Body(...), timeframe: str = Body("1s"),
    z_threshold: float = Body(2.0), corr_min: float = Body(-1.0), min_vol: float = Body(0.0), hysteresis: float = Body(0.2), webhook_url: str | None = Body(None)
) -> Dict[str, Any]:
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        r = AlertRule(name=name, enabled=enabled, symbol_a=symbol_a.lower(), symbol_b=symbol_b.lower(), timeframe=timeframe,
                      z_threshold=z_threshold, corr_min=corr_min, min_vol=min_vol, hysteresis=hysteresis, webhook_url=webhook_url)
        s.add(r)
        s.commit()
        s.refresh(r)
        return {"ok": True, "id": r.id}


@app.patch("/api/alerts/{alert_id}")
async def update_alert(alert_id: int, enabled: bool | None = Body(None)) -> Dict[str, Any]:
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        r = s.get(AlertRule, alert_id)
        if not r:
            raise HTTPException(status_code=404, detail="Not found")
        if enabled is not None:
            r.enabled = bool(enabled)
            s.commit()
        return {"ok": True}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int) -> Dict[str, Any]:
    eng = create_engine(SYNC_DB_URL)
    with Session(eng) as s:
        r = s.get(AlertRule, alert_id)
        if not r:
            raise HTTPException(status_code=404, detail="Not found")
        s.delete(r)
        s.commit()
        return {"ok": True}
