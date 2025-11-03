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