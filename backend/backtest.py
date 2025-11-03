import pandas as pd
import numpy as np


def run_simple_backtest(zscore_series: pd.Series, entry_z: float = 2.0, exit_z: float = 0.0) -> pd.Series:
    """
    Simple z-score based backtest strategy.
    
    Rules:
    - Go short (position = -1) when zscore > entry_z
    - Go long (position = 1) when zscore < -entry_z  
    - Exit (position = 0) when abs(zscore) < exit_z
    
    Returns cumulative P&L series.
    """
    if zscore_series.empty:
        return pd.Series([], dtype=float, index=zscore_series.index)
    
    positions = pd.Series(0, index=zscore_series.index, dtype=int)
    current_pos = 0
    
    for i, z in enumerate(zscore_series):
        if pd.isna(z):
            positions.iloc[i] = current_pos
            continue
            
        # Entry signals
        if current_pos == 0:
            if z > entry_z:
                current_pos = -1  # Short
            elif z < -entry_z:
                current_pos = 1   # Long
        
        # Exit signals
        elif abs(z) < exit_z:
            current_pos = 0
            
        positions.iloc[i] = current_pos
    
    # Calculate P&L: position * (-change in z-score)
    # Negative because we profit when z-score reverts (moves opposite to our position)
    zscore_changes = zscore_series.diff().fillna(0)
    pnl = -positions.shift(1).fillna(0) * zscore_changes
    cumulative_pnl = pnl.cumsum()
    
    return cumulative_pnl


def compute_metrics(pnl: pd.Series) -> dict:
    if pnl.empty:
        return {"sharpe": None, "max_dd": None, "win_rate": None}
    ret = pnl.diff().fillna(0)
    std = ret.std(ddof=0)
    sharpe = float(ret.mean() / std) if std and std != 0 else None
    # Max drawdown
    cummax = pnl.cummax()
    dd = (pnl - cummax)
    max_dd = float(dd.min()) if not dd.empty else None
    # Win-rate (positive return steps)
    wins = (ret > 0).sum()
    total = (ret != 0).sum()
    win_rate = float(wins / total) if total > 0 else None
    return {"sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate}
