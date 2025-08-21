# backtest/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
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
        Execution model: signal on day t -> ENTER at next day's OPEN; EXIT at chosen day's OPEN.
        Sizing: use 'alloc' * current equity per trade (non-overlapping per symbol).
        Returns trades DataFrame with: entry, exit, px_in, px_out, gross, net, notional_$, pnl_$, equity_$.
        """
        fe = self.feature_fn(bars).copy()
        # inject raw OHLC with unique names to avoid any join/dup confusion
        fe["_open"] = bars.loc[fe.index, "open"].astype(float)
        fe["_high"] = bars.loc[fe.index, "high"].astype(float)
        fe["_low"]  = bars.loc[fe.index, "low"].astype(float)

        entries_mask = strategy.generate_entries(fe).astype(bool)
        entries_idx = fe.index[entries_mask]
        trades: list[Trade] = []

        # Prevent overlapping positions for this symbol
        last_exit_pos = -1
        for entry_day in entries_idx:
            i = fe.index.get_loc(entry_day)
            if i + 1 >= len(fe): 
                continue
            if i + 1 <= last_exit_pos: 
                continue  # still in prior trade

            px_in  = float(fe["_open"].iloc[i + 1])
            entry_ts = fe.index[i + 1]
            j_exit = min(max(strategy.pick_exit_index(i, fe), i + 1), len(fe) - 1)
            px_out = float(fe["_open"].iloc[j_exit])
            exit_ts = fe.index[j_exit]

            gross = (px_out / px_in) - 1.0
            trades.append(Trade(entry=entry_ts, exit=exit_ts, px_in=px_in, px_out=px_out,
                                gross=gross, cost_bps=strategy.cost_bps()))
            last_exit_pos = j_exit

        rows = [{
            "entry": t.entry, "exit": t.exit, "px_in": t.px_in, "px_out": t.px_out,
            "gross": t.gross, "net": t.net
        } for t in trades]

        cols = ["entry", "exit", "px_in", "px_out", "gross", "net"]
        out = pd.DataFrame(rows, columns=cols)

        out = out.astype({
            "entry":  "datetime64[ns]",
            "exit":   "datetime64[ns]",
            "px_in":  "float64",
            "px_out": "float64",
            "gross":  "float64",
            "net":    "float64",
        }, errors="ignore")

        return self._attach_cash(out, initial_equity=initial_equity, alloc=alloc)

    # ---------- Cash & equity helpers ----------
    def _attach_cash(self, tr: pd.DataFrame, initial_equity: float, alloc: float) -> pd.DataFrame:
        """Add notional_$, pnl_$, equity_$ assuming non-overlapping trades and compounding."""
        if tr.empty:
            tr["notional_$"] = tr["pnl_$"] = tr["equity_$"] = pd.Series(dtype="float64")
            return tr
        eq = initial_equity
        notionals, pnls, equities = [], [], []
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

    # ---------- Trade-level summary (coarse) ----------
    @staticmethod
    def metrics(tr: pd.DataFrame) -> dict:
        """Per-trade metrics (good for sanity; prefer daily metrics for truth)."""
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

    # ---------- Daily equity & robust metrics ----------
    @staticmethod
    def daily_equity_from_trades(bars: pd.DataFrame, tr: pd.DataFrame,
                                 initial_equity: float = 100_000.0, alloc: float = 1.0) -> pd.Series:
        """
        Build an OPEN->OPEN daily equity curve from trades.
        Costs are applied half on entry day, half on exit day.
        """
        eq = pd.Series(index=bars.index, data=initial_equity, dtype="float64")
        if tr.empty:
            return eq

        o = bars["open"].astype(float)
        oret = o.pct_change().fillna(0.0)

        in_pos = pd.Series(False, index=bars.index)
        cost_adj = pd.Series(0.0, index=bars.index)

        for _, r in tr.iterrows():
            mask = (bars.index >= r["entry"]) & (bars.index <= r["exit"])
            in_pos.loc[mask] = True
            half_cost = (r["net"] - r["gross"]) / 2.0  # negative return “bump”
            cost_adj.loc[r["entry"]] += half_cost
            cost_adj.loc[r["exit"]]  += half_cost

        ret = (oret * in_pos.astype(float) * alloc) + cost_adj
        return (1.0 + ret).cumprod() * initial_equity

    @staticmethod
    def buy_and_hold_equity_open(bars: pd.DataFrame, start_date, initial_equity: float = 100_000.0) -> pd.Series:
        """Benchmark: buy at the first OPEN on start_date, hold. Open-to-open equity to match execution model."""
        o = bars["open"].astype(float)
        o = o.loc[o.index >= start_date]
        if o.empty: 
            return pd.Series(dtype="float64")
        return initial_equity * (o / o.iloc[0])

    @staticmethod
    def perf_daily(eq: pd.Series) -> dict:
        """Robust metrics from a **daily equity** curve."""
        rets = eq.pct_change().dropna()
        if rets.empty:
            return {k: np.nan for k in ["sharpe","sortino","vol","max_dd","dd_dur","cagr","mar","ulcer","gain_to_pain"]}
        mu, sd = rets.mean(), rets.std(ddof=1)
        sharpe  = (mu / sd) * np.sqrt(252) if sd > 0 else np.nan
        downside = rets[rets < 0]
        dsd = downside.std(ddof=1)
        sortino = (mu / dsd) * np.sqrt(252) if dsd and dsd > 0 else np.nan

        roll_max = eq.cummax()
        dd = eq/roll_max - 1.0
        max_dd = float(dd.min())
        # longest drawdown duration (days)
        dd_dur = int((dd != 0).astype(int).groupby((dd == 0).astype(int).cumsum()).cumcount().max())

        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = float((eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1) if yrs > 0 else np.nan
        mar  = float(cagr / abs(max_dd)) if (cagr == cagr and max_dd < 0) else np.nan

        ulcer = float(np.sqrt(np.mean((dd*100.0)**2)))  # Ulcer Index
        gains = rets[rets > 0].sum()
        pain  = -rets[rets < 0].sum()
        gtp   = float(gains / pain) if pain > 0 else np.nan

        return {"sharpe": float(sharpe), "sortino": float(sortino), "vol": float(sd*np.sqrt(252)),
                "max_dd": max_dd, "dd_dur": dd_dur, "cagr": cagr, "mar": mar,
                "ulcer": ulcer, "gain_to_pain": gtp}

    @staticmethod
    def perf_trades(tr: pd.DataFrame) -> dict:
        """Trade-level stats for intuition (expectancy, PF, exposure, etc.)."""
        if tr.empty:
            return {k: np.nan for k in ["trades","winrate","expectancy","avg_win","avg_loss","pf",
                                        "avg_hold_days","exposure_pct","turnover_pa"]}
        wins = tr[tr["net"] > 0]["net"]; losses = tr[tr["net"] <= 0]["net"]
        winrate = len(wins) / len(tr)
        avg_win = wins.mean() if len(wins) else np.nan
        avg_loss = losses.mean() if len(losses) else np.nan
        expectancy = (wins.mean() if len(wins) else 0.0) * winrate \
                   + (losses.mean() if len(losses) else 0.0) * (1 - winrate)
        pf = (wins.sum() / -losses.sum()) if len(losses) and -losses.sum() > 0 else np.nan

        hold_days = (pd.to_datetime(tr["exit"]) - pd.to_datetime(tr["entry"])).dt.days
        avg_hold = float(hold_days.mean()) if len(hold_days) else np.nan
        total_days = (pd.to_datetime(tr["exit"]).max() - pd.to_datetime(tr["entry"]).min()).days or 1
        exposure = float(hold_days.sum() / total_days * 100.0)
        yrs = total_days/365.25
        turnover_pa = float(len(tr) / yrs) if yrs > 0 else np.nan

        return {"trades": int(len(tr)), "winrate": float(winrate),
                "expectancy": float(expectancy),
                "avg_win": float(avg_win) if avg_win == avg_win else np.nan,
                "avg_loss": float(avg_loss) if avg_loss == avg_loss else np.nan,
                "pf": float(pf) if pf == pf else np.nan,
                "avg_hold_days": avg_hold, "exposure_pct": exposure, "turnover_pa": turnover_pa}