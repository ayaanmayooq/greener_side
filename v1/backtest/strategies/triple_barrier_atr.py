from __future__ import annotations
import pandas as pd
from backtest.core import Strategy

class TripleBarrierATR(Strategy):
    def __init__(self, trigger_col="mom_20", atr_col="atr_14",
                 G_mult=3.0, L_mult=1.0, horizon=10, cost_bps=2.0):
        self.trigger_col, self.atr_col = trigger_col, atr_col
        self.G_mult, self.L_mult, self.horizon, self._cost_bps = G_mult, L_mult, horizon, cost_bps

    def required_feature_keys(self) -> list[str]:
        return ["swing_core"]

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        trigger = fe[self.trigger_col] > 0 if self.trigger_col in fe else fe["mom_20"] > 0
        # Simple trend filter
        return trigger & (fe["adj_close"] > fe["sma_200"])

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        # Enter at next day's open
        entry_i = i_signal + 1
        end_i   = min(entry_i + self.horizon, len(fe) - 1)

        entry_px = fe["_open"].iloc[entry_i]
        atr0 = fe[self.atr_col].iloc[entry_i]
        tgt = entry_px + self.G_mult * atr0
        stp = entry_px - self.L_mult * atr0

        # First touch via canonical execution highs/lows
        for j in range(entry_i + 1, end_i + 1):
            if fe["_high"].iloc[j] >= tgt or fe["_low"].iloc[j] <= stp:
                return j
        return end_i