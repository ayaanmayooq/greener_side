# backtest/strategies.py
from __future__ import annotations
import pandas as pd
from backtest.core import Strategy

class BreakoutHold(Strategy):
    """
    Long when adj_close > highest(adj_close, lookback) AND adj_close > sma_200.
    Enter next open; exit after HOLD_DAYS (time stop).
    """
    def __init__(self, lookback: int = 20, hold_days: int = 5, cost_bps: float = 2.0):
        self.lookback = lookback
        self.hold_days = hold_days
        self._cost_bps = cost_bps

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        roll_high = fe["adj_close"].shift(1).rolling(self.lookback).max()
        breakout = fe["adj_close"] > roll_high
        regime   = fe["adj_close"] > fe["sma_200"]
        return (breakout & regime)

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        # Enter at i_signal+1, then count hold_days forward
        return min(i_signal + 1 + self.hold_days, len(fe) - 1)


class TripleBarrierATR(Strategy):
    """
    Enter when 'trigger' is True (e.g., momentum > 0).
    Exit when: target (G * ATR) or stop (L * ATR) is hit (on CLOSE),
               or after horizon days (time stop). Execution at next open for simplicity.
    NOTE: This is close-based barrier detection for clarity; you can upgrade to intraday highs/lows later.
    """
    def __init__(self, trigger_col: str, atr_col: str = "atr_14",
                 G_mult: float = 2.0, L_mult: float = 1.25, horizon: int = 5, cost_bps: float = 2.0):
        self.trigger_col = trigger_col
        self.atr_col = atr_col
        self.G_mult = G_mult
        self.L_mult = L_mult
        self.horizon = horizon
        self._cost_bps = cost_bps

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        # Example trigger: positive 20D momentum AND above 200DMA
        if self.trigger_col in fe.columns:
            trigger = fe[self.trigger_col] > 0
        else:
            # default to simple momentum
            trigger = fe["mom_20"] > 0
        regime = fe["adj_close"] > fe["sma_200"]
        return (trigger & regime)

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        # Entry at i_signal+1
        entry_i = i_signal + 1
        entry_px = fe["_open"].iloc[entry_i]
        atr = fe[self.atr_col].iloc[entry_i]
        tgt = entry_px + self.G_mult * atr
        stp = entry_px - self.L_mult * atr
        end_i = min(entry_i + self.horizon, len(fe) - 1)

        # Walk forward on CLOSE to find first hit (close-based approximation)
        for j in range(entry_i + 1, end_i + 1):
            c = fe["adj_close"].iloc[j]
            if c >= tgt or c <= stp:
                return j
        return end_i
    
    
class BuyAndHold(Strategy):
    """
    Enter once near the start and hold to the end.
    We enter at the NEXT day's open after the first feature row (engine convention).
    """
    def __init__(self, cost_bps: float = 0.0):
        self._cost_bps = cost_bps

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        s = pd.Series(False, index=fe.index)
        if len(fe) >= 2:
            s.iloc[0] = True  # signal on first feature row → enter next day's open
        return s

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        return len(fe) - 1  # exit at the last available bar (open)