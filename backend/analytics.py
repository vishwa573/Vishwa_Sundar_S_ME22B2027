from __future__ import annotations

from typing import List, Dict, Any

import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sqlalchemy import create_engine
from sklearn.linear_model import HuberRegressor

from backend.database import SYNC_DB_URL, TickData, OhlcData

logger = logging.getLogger("analytics")


def _timeframe_to_rule(tf: str) -> str:
    # Use lowercase offset aliases to avoid FutureWarning and ensure fast resample
    mapping = {"1s": "1s", "1m": "1min", "5m": "5min"}
    return mapping.get(tf, "1s")


def load_data_and_resample(symbols: List[str], timeframe: str, use_ohlc: bool = False, lookback_hours: int = 6, max_ticks: int = 0) -> pd.DataFrame:
    symbols = [s.lower() for s in symbols]
    engine = create_engine(SYNC_DB_URL)
    start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    # Single query path (fast and simple). max_ticks is ignored here intentionally for stability.
    sym_params = {f"s{i}": s for i, s in enumerate(symbols)}
    placeholders = ",".join([f":s{i}" for i in range(len(symbols))])
    if use_ohlc:
        query = f"""
            SELECT timestamp, symbol, close AS price
            FROM {OhlcData.__tablename__}
            WHERE symbol IN ({placeholders}) AND timestamp >= :start_time
            ORDER BY timestamp ASC
        """
    else:
        query = f"""
            SELECT timestamp, symbol, price, size
            FROM {TickData.__tablename__}
            WHERE symbol IN ({placeholders}) AND timestamp >= :start_time
            ORDER BY timestamp ASC
        """
    params = {"start_time": start_time, **sym_params}
    df = pd.read_sql(query, con=engine, params=params)
    logger.info("Loaded %d rows from %s (lookback=%dh)", len(df), "ohlc" if use_ohlc else "ticks", lookback_hours)
    if df.empty:
        logger.warning("Analytics: No data loaded from DB.")
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


def load_volume_and_vwap(symbols: List[str], timeframe: str, use_ohlc: bool = False, lookback_hours: int = 6, max_ticks: int = 0) -> Dict[str, Dict[str, pd.Series]]:
    """Return per-symbol resampled volume and VWAP series.

    For ticks: volume = sum(size), vwap = sum(price*size)/sum(size).
    For ohlc: volume from table if present; vwap approximated by close (fallback).
    """
    symbols = [s.lower() for s in symbols]
    engine = create_engine(SYNC_DB_URL)
    start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    if use_ohlc:
        sym_params = {f"s{i}": s for i, s in enumerate(symbols)}
        placeholders = ",".join([f":s{i}" for i in range(len(symbols))])
        query = f"""
            SELECT timestamp, symbol, close, volume
            FROM {OhlcData.__tablename__}
            WHERE symbol IN ({placeholders}) AND timestamp >= :start_time
            ORDER BY timestamp ASC
        """
    else:
        sym_params = {f"s{i}": s for i, s in enumerate(symbols)}
        placeholders = ",".join([f":s{i}" for i in range(len(symbols))])
        query = f"""
            SELECT timestamp, symbol, price, size
            FROM {TickData.__tablename__}
            WHERE symbol IN ({placeholders}) AND timestamp >= :start_time
            ORDER BY timestamp ASC
        """
    params = {"start_time": start_time, **sym_params}
    df = pd.read_sql(query, con=engine, params=params)
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    rule = _timeframe_to_rule(timeframe)

    out: Dict[str, Dict[str, pd.Series]] = {}
    for sym in symbols:
        d = df[df["symbol"] == sym]
        if d.empty:
            continue
        if use_ohlc:
            vol = d.get("volume", pd.Series(index=d.index, dtype=float)).resample(rule).sum(min_count=1)
            vwap = d["close"].resample(rule).last()
        else:
            vol = d["size"].resample(rule).sum(min_count=1)
            pv = (d["price"] * d["size"]).resample(rule).sum(min_count=1)
            vwap = pv / vol.replace({0.0: np.nan})
        out[sym] = {"volume": vol, "vwap": vwap}
    return out


def rolling_adf_pvalues(series: pd.Series, window: int, step: int = 5) -> pd.Series:
    s = series.dropna()
    if window <= 0 or len(s) < window:
        return pd.Series([], dtype=float, index=series.index)
    # Cap total ADF evaluations to avoid timeouts
    max_evals = 500
    total_windows = max(1, len(s) - window + 1)
    if total_windows > max_evals:
        step = max(step, total_windows // max_evals)
    idx = []
    vals = []
    arr = s.values
    times = s.index
    for start in range(0, len(arr) - window + 1, step):
        end = start + window
        try:
            stat, p, *_ = adfuller(arr[start:end], autolag="AIC")
        except Exception:
            p = np.nan
        idx.append(times[end - 1])
        vals.append(float(p) if p is not None else np.nan)
    return pd.Series(vals, index=pd.to_datetime(idx))


def calculate_ols_hedge_ratio(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> float:
    df2 = df[[symbol_a, symbol_b]].dropna()
    if df2.shape[0] < 2:
        return float("nan")
    X = sm.add_constant(df2[symbol_b])
    model = sm.OLS(df2[symbol_a], X).fit()
    return float(model.params[symbol_b])


def calculate_huber_hedge_ratio(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> float:
    df2 = df[[symbol_a, symbol_b]].dropna()
    if df2.shape[0] < 2:
        return float("nan")
    X = df2[[symbol_b]].values
    y = df2[symbol_a].values
    reg = HuberRegressor()
    reg.fit(X, y)
    return float(reg.coef_[0])


def calculate_kalman_hedge_ratio(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> float:
    """Estimate a time-varying hedge ratio beta_t using a simple 1D Kalman Filter.

    Model: y_t = beta_t * x_t + e_t,   beta_t = beta_{t-1} + w_t
    Returns the latest beta_t.
    """
    try:
        from pykalman import KalmanFilter  # type: ignore
    except Exception:
        return float("nan")
    df2 = df[[symbol_a, symbol_b]].dropna()
    if df2.shape[0] < 5:
        return float("nan")
    y = df2[symbol_a].values
    x = df2[symbol_b].values
    # Time-varying observation matrix (N x 1 x 1)
    obs_mats = x.reshape(-1, 1, 1)
    kf = KalmanFilter(
        transition_matrices=[1.0],
        observation_matrices=obs_mats,
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        transition_covariance=1e-4,   # process noise
        observation_covariance=1e-2,  # measurement noise
    )
    state_means, _ = kf.filter(y)
    return float(state_means[-1][0])


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
