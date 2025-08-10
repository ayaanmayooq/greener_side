# backtest/swing_breakout.py
from pathlib import Path
import math
import pandas as pd
import numpy as np

from utils.config import DATA_ROOT, CFG
from research.features import swing_features

# ---- knobs you can tweak ----
LOOKBACK   = 20     # breakout lookback (days)
HOLD_DAYS  = 5      # time stop
COST_BPS   = 2.0    # round-trip cost in basis points (0.02%)
# -----------------------------

def load_symbol(symbol: str) -> pd.DataFrame:
    p = DATA_ROOT / f"processed/daily/symbol={symbol}/bars.parquet"
    df = pd.read_parquet(p).sort_index()
    df.index.name = "date"
    return df

def generate_signals(fe: pd.DataFrame) -> pd.Series:
    # breakout on adjusted close
    roll_high = fe["adj_close"].shift(1).rolling(LOOKBACK).max()
    breakout  = fe["adj_close"] > roll_high
    regime    = fe["adj_close"] > fe["sma_200"]
    signal = (breakout & regime).astype(int)  # 1 when we will enter next open
    return signal

def backtest_symbol(df: pd.DataFrame) -> pd.DataFrame:
    fe  = swing_features(df)
    sig = generate_signals(fe)

    # Entries at next day's open when signal==1 today
    entries = sig[sig == 1].index
    trades = []
    for entry_day in entries:
        # find next trading day
        try:
            entry_idx = fe.index.get_loc(entry_day)
        except KeyError:
            continue
        if entry_idx + 1 >= len(fe):
            continue
        px_in  = fe["open"].iloc[entry_idx + 1]   # next open
        entry_dt = fe.index[entry_idx + 1]

        exit_idx = min(entry_idx + 1 + HOLD_DAYS, len(fe) - 1)
        px_out = fe["open"].iloc[exit_idx]        # exit at open
        exit_dt = fe.index[exit_idx]

        # gross return; subtract costs (round-trip)
        gross = (px_out / px_in) - 1.0
        net   = gross - (COST_BPS / 10000.0)
        trades.append(dict(entry=entry_dt, exit=exit_dt,
                           px_in=float(px_in), px_out=float(px_out),
                           gross=float(gross), net=float(net)))
    return pd.DataFrame(trades)

def metrics_from_trades(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(trades=0, winrate=np.nan, avg_net=np.nan, sharpe=np.nan,
                    max_dd=np.nan, cagr=np.nan)
    # build equity curve assuming 100% notional each trade, non-overlapping by design (time exit)
    eq = (1.0 + tr["net"]).cumprod()
    ret_series = tr["net"].rename("ret")
    wins = (ret_series > 0).mean()

    # Sharpe on trade returns (rough); for daily you’d use daily returns
    sharpe = np.nan
    if ret_series.std(ddof=1) > 0:
        sharpe = ret_series.mean() / ret_series.std(ddof=1) * math.sqrt(252 / HOLD_DAYS)

    # max drawdown on trade equity
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = dd.min()

    # CAGR estimate using average holding length
    n_years = (tr["exit"].iloc[-1] - tr["entry"].iloc[0]).days / 365.25
    cagr = (eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

    return dict(trades=len(tr), winrate=wins, avg_net=ret_series.mean(),
                sharpe=sharpe, max_dd=max_dd, cagr=cagr)

def run_for_symbol(symbol: str):
    df = load_symbol(symbol)
    tr = backtest_symbol(df)
    m  = metrics_from_trades(tr)
    print(f"\n=== {symbol} ({len(tr)} trades) ===")
    if len(tr):
        print(tr.tail(5))
    print({k: (round(v,4) if isinstance(v, (float, np.floating)) else v) for k,v in m.items()})

if __name__ == "__main__":
    for sym in CFG["symbols"]["swing"]:
        run_for_symbol(sym)