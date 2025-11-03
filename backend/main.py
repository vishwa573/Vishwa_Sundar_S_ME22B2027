import logging
from typing import Dict, Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.analytics import (
    load_data_and_resample,
    calculate_ols_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    calculate_rolling_correlation,
    run_adf_test,
)

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
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe)

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

    hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1])
    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    z = calculate_zscore(spread)
    rc = calculate_rolling_correlation(df, symbols[0], symbols[1], rolling_window)

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
    }
    logger.info("API: Returning analytics data.")
    return result


@app.get("/api/adf_test")
async def adf_test(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query("1s"),
) -> Dict[str, Any]:
    symbols = [symbol_a.lower(), symbol_b.lower()]
    df = load_data_and_resample(symbols, timeframe)
    if df.empty:
        return {"pvalue": None, "statistic": None}
    hr = calculate_ols_hedge_ratio(df, symbols[0], symbols[1])
    spread = calculate_spread(df, symbols[0], symbols[1], hr)
    res = run_adf_test(spread)
    return res
