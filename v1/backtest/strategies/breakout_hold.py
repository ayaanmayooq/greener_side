from __future__ import annotations
import pandas as pd
from backtest.core import Strategy

class BreakoutHold(Strategy):
    def __init__(self, lookback=20, hold_days=5, cost_bps=2.0):
        self.lookback, self.hold_days, self._cost_bps = lookback, hold_days, cost_bps

    def required_feature_keys(self) -> list[str]:
        return ["swing_core"]

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        # Breakout over prior N-day high, with a 200DMA trend filter
        roll_high = fe["adj_close"].shift(1).rolling(self.lookback).max()
        return (fe["adj_close"] > roll_high) & (fe["adj_close"] > fe["sma_200"])

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        # Time-based exit: hold for fixed number of days, then exit at open
        return min(i_signal + 1 + self.hold_days, len(fe) - 1)