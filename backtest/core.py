# backtest/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import math
import numpy as np
import pandas as pd

# ---- Types ----
FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]

@dataclass
class Trade:
    entry: pd.Timestamp
    exit: pd.Timestamp
    px_in: float
    px_out: float
    gross: float   # (px_out/px_in - 1) for longs
    cost_bps: float
    @property
    def net(self) -> float:
        return self.gross - self.cost_bps/10000.0


class Strategy:
    """
    Minimal interface:
    - generate_entries(fe): Boolean Series (True on the day you want to ENTER at next open)
    - pick_exit_index(i_entry, fe): returns the row index (int) where we EXIT (at that day's open)
    - cost_bps(): round-trip cost assumption
    """
    def generate_entries(self, fe: pd.DataFrame) -> pd.Series: ...
    def pick_exit_index(self, i_entry: int, fe: pd.DataFrame) -> int: ...
    def cost_bps(self) -> float: return 2.0  # default 2 bps round-trip


class BacktestEngine:
    def __init__(self, feature_fn: FeatureFn):
        self.feature_fn = feature_fn

    def run_symbol(self, bars: pd.DataFrame, strategy: Strategy,
                   initial_equity: float = 100_000.0, alloc: float = 1.0) -> pd.DataFrame:
        """
        bars: DataFrame with columns open, high, low, close, volume, adj_close (Date index).
        Returns trades DataFrame with entry/exit/gross/net and cash columns (notional_$, pnl_$, equity_$).
        Assumes non-overlapping trades (true for the provided strategies).
        """
        fe = self.feature_fn(bars).copy()
        fe["_open"] = bars.loc[fe.index, "open"]

        entries_mask = strategy.generate_entries(fe).astype(bool)
        entries_idx = fe.index[entries_mask]
        trades: list[Trade] = []

        last_exit_pos = -1
        for entry_day in entries_idx:
            i = fe.index.get_loc(entry_day)
            if i + 1 >= len(fe): continue
            if i + 1 <= last_exit_pos: continue

            px_in  = float(fe["_open"].iloc[i + 1])
            entry_ts = fe.index[i + 1]
            j_exit = min(max(strategy.pick_exit_index(i, fe), i + 1), len(fe) - 1)
            px_out = float(fe["_open"].iloc[j_exit])
            exit_ts = fe.index[j_exit]

            gross = (px_out / px_in) - 1.0
            trades.append(Trade(entry=entry_ts, exit=exit_ts, px_in=px_in, px_out=px_out,
                                gross=gross, cost_bps=strategy.cost_bps()))
            last_exit_pos = j_exit

        out = pd.DataFrame([{
            "entry": t.entry, "exit": t.exit, "px_in": t.px_in, "px_out": t.px_out,
            "gross": t.gross, "net": t.net
        } for t in trades]).astype({"px_in":"float64","px_out":"float64","gross":"float64","net":"float64"})

        # >>> add cash columns here <<<
        out = self._attach_cash(out, initial_equity=initial_equity, alloc=alloc)
        return out

    def _attach_cash(self, tr: pd.DataFrame, initial_equity: float, alloc: float) -> pd.DataFrame:
        """Add notional_$, pnl_$, equity_$ assuming non-overlapping trades and compounding."""
        if tr.empty:
            tr["notional_$"] = tr["pnl_$"] = tr["equity_$"] = pd.Series(dtype="float64")
            return tr
        eq = initial_equity
        notionals = []
        pnls = []
        equities = []
        for net in tr["net"]:
            notional = eq * alloc
            pnl = notional * net
            eq = eq + pnl  # compound after each trade
            notionals.append(notional)
            pnls.append(pnl)
            equities.append(eq)
        tr["notional_$"] = notionals
        tr["pnl_$"] = pnls
        tr["equity_$"] = equities
        return tr

    @staticmethod
    def cash_summary(tr: pd.DataFrame, initial_equity: float = 100_000.0) -> dict:
        if tr.empty or "equity_$" not in tr:
            return {"final_equity": initial_equity, "pnl_$": 0.0}
        final_eq = float(tr["equity_$"].iloc[-1])
        return {"final_equity": final_eq, "pnl_$": final_eq - initial_equity}

    @staticmethod
    def metrics(tr: pd.DataFrame) -> dict:
        if tr.empty:
            return dict(trades=0, winrate=np.nan, avg_net=np.nan, sharpe=np.nan,
                        max_dd=np.nan, cagr=np.nan)
        eq = (1.0 + tr["net"]).cumprod()
        s = tr["net"].std(ddof=1)
        sharpe = (tr["net"].mean() / s * math.sqrt(252)) if s > 0 else np.nan
        roll_max = eq.cummax()
        dd = eq/roll_max - 1.0
        max_dd = dd.min()
        n_years = (tr["exit"].iloc[-1] - tr["entry"].iloc[0]).days / 365.25
        cagr = (eq.iloc[-1] ** (1/n_years) - 1.0) if n_years > 0 else np.nan
        return dict(trades=len(tr), winrate=(tr["net"] > 0).mean(),
                    avg_net=tr["net"].mean(), sharpe=sharpe, max_dd=max_dd, cagr=cagr)