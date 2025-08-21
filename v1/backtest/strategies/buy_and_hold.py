from __future__ import annotations
import pandas as pd
from backtest.core import Strategy

class BuyAndHold(Strategy):
    def __init__(self, cost_bps: float = 0.0):
        self._cost_bps = cost_bps

    def required_feature_keys(self) -> list[str]:
        # Doesn’t truly need features, but using swing_core keeps flow consistent
        return ["swing_core"]

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        s = pd.Series(False, index=fe.index)
        if len(fe) >= 2:
            s.iloc[0] = True  # enter next open after first feature row
        return s

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        # Hold to the last available bar (exit at that day's open)
        return len(fe) - 1