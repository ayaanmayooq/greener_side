import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import os
from options.suggest_options import bs_price_greeks, realized_vol_from_close, suggest_verticals_from_df


def load_yfinance_data(symbol, start=None, end=None):
    symbol = symbol.lower()
    data_dir = "data"
    file_path = f"{data_dir}/{symbol}.csv"

    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=3,
        names=['Date','Close','High','Low','Open','Volume'],
        usecols=[0,1,2,3,4,5]
    )
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors="coerce")
    df = df.sort_values('Date').set_index('Date')
    df_bt = df[['Open','High','Low','Close','Volume']].astype(float)
    if start is not None: df_bt = df_bt.loc[pd.to_datetime(start):]
    if end   is not None: df_bt = df_bt.loc[:pd.to_datetime(end)]
    return df_bt

def pull_and_save_symbols(symbols, start="2003-01-01", data_dir="data"):
    today = datetime.now().strftime("%Y-%m-%d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        print(f"Downloading {symbol} …")
        df = yf.download(symbol, start=start, end=today, progress=False, threads=False, auto_adjust=True)
        out = Path(data_dir) / f"{symbol.lower()}.csv"
        df.to_csv(out)
        print(f"  saved -> {out}")


def vertical_price_given_strikes(S, T_years, sigma, K_long, K_short, r=0.0, q=0.0):
    longC  = bs_price_greeks(S, K_long,  T_years, r, q, sigma, 'call')
    shortC = bs_price_greeks(S, K_short, T_years, r, q, sigma, 'call')
    return max(1e-10, longC["price"] - shortC["price"])

LOG_NAME = "signals_log.csv"

def _append_to_log(rows_df: pd.DataFrame, data_dir="data"):
    path = Path(data_dir); path.mkdir(parents=True, exist_ok=True)
    logp = path / LOG_NAME
    rows_df = rows_df.copy()
    rows_df["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    if logp.exists():
        old = pd.read_csv(logp)
        both = pd.concat([old, rows_df], ignore_index=True)
        # dedupe by symbol + signal_date
        both["signal_date"] = pd.to_datetime(both["signal_date"])
        both = both.sort_values(["symbol","signal_date","created_at"]).drop_duplicates(
            subset=["symbol","signal_date"], keep="last"
        )
        both.to_csv(logp, index=False)
    else:
        rows_df.to_csv(logp, index=False)

def _load_log(data_dir="data") -> pd.DataFrame:
    logp = Path(data_dir) / LOG_NAME
    if not logp.exists():
        return pd.DataFrame(columns=[
            "symbol","signal_date","exec_timing","DTE","S",
            "K_long","K_short","width","debit","cost_frac","net_delta",
            "TP_on_prem","SL_on_prem","notes","created_at"
        ])
    df = pd.read_csv(logp, parse_dates=["signal_date"])
    return df


def show_recent_and_reprice(
    symbols,
    *,
    lookback_days=10,
    actionable_window_bars=3,
    min_remaining_dte=7,
    data_dir="data",
    start_date="2003-01-01",
    end_date=None,
    # for fresh re-pick now:
    N_fresh=20,
    strike_inc=1.0,
    target_cost_max=0.40
):
    """
    Show signals fired in the last `lookback_days` and evaluate them *today*:
      - bars_elapsed since planned execution (signal_date+1)
      - remaining DTE
      - repriced value of the original strikes today
      - a fresh spread suggestion today
      - status: fresh_today | late_but_ok | stale
    """
    logdf = _load_log(data_dir=data_dir)
    if logdf.empty:
        print("No signals yet in the log.")
        return

    cutoff = pd.Timestamp(datetime.now().date() - timedelta(days=lookback_days))
    recent = logdf[logdf["signal_date"] >= cutoff].copy()
    if recent.empty:
        print(f"No signals in the last {lookback_days} days.")
        return

    rows = []
    for sym in symbols:
        df_bt = load_yfinance_data(sym, start_date, end_date)
        if df_bt.empty:
            continue
        close = df_bt["Close"].astype(float)
        idx = close.index
        sigma_series = realized_vol_from_close(close, 20).bfill().fillna(0.30)

        subset = recent[recent["symbol"] == sym.upper()].sort_values("signal_date")
        for _, r in subset.iterrows():
            sig_dt = pd.to_datetime(r["signal_date"])
            if sig_dt not in idx:
                # align to next available date in data
                i_sig = int(idx.searchsorted(sig_dt))
                if i_sig >= len(idx):
                    continue
            else:
                i_sig = idx.get_loc(sig_dt)

            i_exec = min(i_sig + 1, len(idx)-1)  # plan: next open
            i_now  = len(idx) - 1
            bars_elapsed = max(0, i_now - i_exec)

            # Remaining DTE for original plan
            DTE0 = int(r["DTE"])
            rem_dte = max(1, DTE0 - bars_elapsed)

            # Reprice original strikes *today*
            S_now = float(close.iloc[i_now])
            sigma_now = float(sigma_series.iloc[i_now])
            T_rem = rem_dte / 252.0
            K_low, K_high = float(r["K_long"]), float(r["K_short"])
            debit_now = vertical_price_given_strikes(S_now, T_rem, sigma_now, K_low, K_high)
            width = float(r["width"])
            cost_frac_now = debit_now / max(width, 1e-9)
            roi_max_now = max((width - debit_now) / max(debit_now, 1e-12), 0.0)

            # Make a fresh suggestion *today* (same DTE target as N_fresh)
            # Re-use your existing generator:
            # use suggest_verticals_from_df, then pick the last row if it exists; else None
            recs_today = suggest_verticals_from_df(df_bt, N=N_fresh, strike_inc=strike_inc)
            fresh = recs_today.iloc[-1].to_dict() if not recs_today.empty else None

            # status labeling
            if i_now == i_sig:
                status = "fresh_today"
            elif bars_elapsed <= actionable_window_bars and rem_dte >= min_remaining_dte:
                status = "late_but_ok"
            else:
                status = "stale"

            rows.append({
                "symbol": sym.upper(),
                "signal_date": sig_dt,
                "planned_DTE": DTE0,
                "bars_elapsed": bars_elapsed,
                "remaining_DTE": rem_dte,
                # repriced original
                "S_now": S_now,
                "K_long": K_low, "K_short": K_high, "width": width,
                "debit_now": debit_now,
                "cost_frac_now": cost_frac_now,
                "roi_max_now": roi_max_now,
                # fresh pick now (summarized)
                "fresh_K_long": (fresh["K_long"] if fresh else None),
                "fresh_K_short": (fresh["K_short"] if fresh else None),
                "fresh_width": (fresh["width"] if fresh else None),
                "fresh_debit": (fresh["debit"] if fresh else None),
                "fresh_cost_frac": (fresh["cost_frac"] if fresh else None),
                "fresh_DTE": (fresh["DTE"] if fresh else None),
                "status": status
            })

    out = pd.DataFrame(rows).sort_values(["symbol","signal_date"])
    if out.empty:
        print("Nothing to show after repricing.")
        return

    print(f"\n=== CATCH-UP (last {lookback_days} days) ===")
    cols = [
        "symbol","signal_date","status","planned_DTE","bars_elapsed","remaining_DTE",
        "S_now","K_long","K_short","debit_now","cost_frac_now","roi_max_now",
        "fresh_K_long","fresh_K_short","fresh_debit","fresh_cost_frac","fresh_DTE"
    ]
    print(out[cols].to_string(index=False))

    # Optional: save a copy
    catch_path = Path(data_dir) / f"catchup_{datetime.now():%Y-%m-%d}.csv"
    out.to_csv(catch_path, index=False)
    print(f"\nSaved catch-up: {catch_path}")