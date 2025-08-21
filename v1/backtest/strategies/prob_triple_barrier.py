from __future__ import annotations
import pandas as pd
from backtest.core import Strategy

class ProbTripleBarrier(Strategy):
    """
    Trade only when calibrated p(tp before sl) > breakeven + margin.
    Exits on first touch of target/stop using daily highs/lows, or time cap h.
    """
    def __init__(self, model_id: str = "swing_xgb_prob_tp_v1", p_col: str = "p_tp",
                 atr_col: str = "atr_14", G_mult: float = 3.0, L_mult: float = 1.0,
                 horizon: int = 10, margin: float = 0.02, cost_bps: float = 2.0):
        self.model_id = model_id
        self.p_col = p_col
        self.atr_col = atr_col
        self.G_mult = G_mult
        self.L_mult = L_mult
        self.horizon = horizon
        self.margin = margin
        self._cost_bps = cost_bps

    def required_feature_keys(self) -> list[str]:
        # Needs base indicators + the p_tp predictions from the specified model
        return ["swing_core", f"prob_tp:{self.model_id}"]

    def cost_bps(self) -> float:
        return self._cost_bps

    def generate_entries(self, fe: pd.DataFrame) -> pd.Series:
        C = self._cost_bps / 10000.0
        p_be = (self.L_mult + C) / (self.L_mult + self.G_mult)

        p = fe[self.p_col].fillna(0.0)

        q = p.quantile(0.95)
        gate = p > max(p_be + self.margin, q)

        regime = fe["adj_close"] > fe["sma_200"]
        return gate & regime

    def pick_exit_index(self, i_signal: int, fe: pd.DataFrame) -> int:
        entry_i = i_signal + 1
        end_i   = min(entry_i + self.horizon, len(fe) - 1)

        entry_px = fe["_open"].iloc[entry_i]
        atr0 = fe[self.atr_col].iloc[entry_i]
        tgt = entry_px + self.G_mult * atr0
        stp = entry_px - self.L_mult * atr0

        for j in range(entry_i + 1, end_i + 1):
            if fe["_high"].iloc[j] >= tgt or fe["_low"].iloc[j] <= stp:
                return j
        return end_i