from __future__ import annotations

from typing import List, Dict, Any

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sqlalchemy import create_engine

from backend.database import SYNC_DB_URL

logger = logging.getLogger("analytics")


def _timeframe_to_rule(tf: str) -> str:
    mapping = {"1s": "1S", "1m": "1T", "5m": "5T"}
    return mapping.get(tf, "1S")


def load_data_and_resample(symbols: List[str], timeframe: str) -> pd.DataFrame:
    symbols = [s.lower() for s in symbols]
    engine = create_engine(SYNC_DB_URL)
    placeholders = ",".join([f"'{s}'" for s in symbols])
    query = f"""
        SELECT timestamp, lower(symbol) AS symbol, price, size
        FROM ticks
        WHERE lower(symbol) IN ({placeholders})
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, con=engine)
    logger.info("Loaded %d raw ticks from database at %s", len(df), engine.url)
    if df.empty:
        logger.warning("Analytics: No data loaded from DB. Check if DB path is correct.")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    rule = _timeframe_to_rule(timeframe)

    price_frames = []
    for sym in symbols:
        d = df[df["symbol"] == sym]
        if d.empty:
            continue
        pr = d["price"].resample(rule).last().rename(sym)
        price_frames.append(pr)

    if not price_frames:
        return pd.DataFrame()

    prices = pd.concat(price_frames, axis=1)
    prices = prices.ffill().dropna(how="all")
    return prices


def calculate_ols_hedge_ratio(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> float:
    df2 = df[[symbol_a, symbol_b]].dropna()
    if df2.shape[0] < 2:
        return float("nan")
    X = sm.add_constant(df2[symbol_b])
    model = sm.OLS(df2[symbol_a], X).fit()
    return float(model.params[symbol_b])


def calculate_spread(df: pd.DataFrame, symbol_a: str, symbol_b: str, hedge_ratio: float) -> pd.Series:
    return df[symbol_a] - hedge_ratio * df[symbol_b]


def calculate_zscore(spread_series: pd.Series) -> pd.Series:
    if spread_series.empty:
        return spread_series
    mean = spread_series.mean()
    std = spread_series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return spread_series * np.nan
    return (spread_series - mean) / std


def calculate_rolling_correlation(df: pd.DataFrame, symbol_a: str, symbol_b: str, window: int) -> pd.Series:
    return df[symbol_a].rolling(window).corr(df[symbol_b])


def run_adf_test(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {"pvalue": None, "statistic": None}
    stat, pvalue, *_ = adfuller(s, autolag="AIC")
    return {"pvalue": float(pvalue), "statistic": float(stat)}
