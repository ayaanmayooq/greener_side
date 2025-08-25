import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

from options.suggest_options import recommend_vertical_now, size_spread  # <= added size_spread
from utils import load_yfinance_data, pull_and_save_symbols, show_recent_and_reprice, _append_to_log

def run_daily(
    symbols,
    start_date="2003-01-01",
    end_date=None,
    N=20,
    strike_inc=1.0,
    data_dir="data",
    # sizing knobs
    start_equity=10_000,
    budget_frac=0.10,
    multiplier=100
):
    # 1) pull/update CSVs
    pull_and_save_symbols(symbols, start=start_date, data_dir=data_dir)

    # 2) build fresh "today" signals
    rows = []
    for sym in symbols:
        df_bt = load_yfinance_data(sym, start_date, end_date)
        if df_bt.empty:
            continue
        rec = recommend_vertical_now(df_bt, N=N, strike_inc=strike_inc, force=False)
        if rec is None:
            continue

        # size the spread
        sz = size_spread(rec, equity=start_equity, budget_frac=budget_frac, multiplier=multiplier)

        rows.append({
            "symbol": sym.upper(),
            "signal_date": rec["signal_date"],
            "exec_timing": rec["exec_timing"],
            "DTE": rec["DTE"],
            "S": rec["S"],
            "K_long": rec["K_long"],
            "K_short": rec["K_short"],
            "width": rec["width"],
            "debit": rec["debit"],
            "cost_frac": rec["cost_frac"],
            "net_delta": rec["net_delta"],
            "TP_on_prem": rec["TP_on_prem"],
            "SL_on_prem": rec["SL_on_prem"],
            "contracts": sz["contracts"],
            "premium_used": sz["premium_used"],
            "planned_risk_pct_of_equity": sz["planned_risk_pct_of_equity"],
            "worst_case_loss_pct_of_equity": sz["worst_case_loss_pct_of_equity"],
            "notes": rec.get("notes","")
        })

    if rows:
        today_df = pd.DataFrame(rows).sort_values(["signal_date","symbol"])
        print("\n=== ACTIONABLE SIGNALS (enter at next open) ===")
        cols = [
            "symbol","signal_date","DTE","S",
            "K_long","K_short","width","debit","cost_frac","net_delta",
            "contracts","premium_used","TP_on_prem","SL_on_prem",
            "planned_risk_pct_of_equity","worst_case_loss_pct_of_equity"
        ]
        print(today_df[cols].to_string(index=False))

        # persist to *master* signals log
        _append_to_log(today_df.drop(columns=[
            "contracts","premium_used","planned_risk_pct_of_equity","worst_case_loss_pct_of_equity"
        ]), data_dir=data_dir)

        # optional: also save a dated snapshot
        out_path = Path(data_dir) / f"signals_{datetime.now():%Y-%m-%d}.csv"
        today_df.to_csv(out_path, index=False)
        print(f"\nSaved snapshot: {out_path}")
        print(f"Assumptions: start_equity={start_equity:,}  budget_frac={budget_frac:.1%}  multiplier={multiplier}")
    else:
        print(f"No new entries today ({datetime.now():%Y-%m-%d}).")


if __name__ == "__main__":
    SYMBOLS = ["AAPL","MSFT","GOOGL","NVDA","META","SPY"]

    # 1) Run your normal daily step (updates data, logs fresh signals)
    run_daily(
        SYMBOLS,
        start_date="2003-01-01",
        end_date=None,
        N=20,
        data_dir="data",
        start_equity=5_000,
        budget_frac=0.10,
        multiplier=100
    )

    # 2) Then show missed/recent signals and reprice them *today*
    show_recent_and_reprice(
        SYMBOLS,
        lookback_days=10,           # last 10 calendar days of signals
        actionable_window_bars=3,   # within 3 trading bars after planned exec = “late_but_ok”
        min_remaining_dte=7,        # if too little DTE left, label as stale
        data_dir="data",
        start_date="2003-01-01",
        end_date=None,
        N_fresh=20,                 # fresh re-pick uses this DTE
        strike_inc=1.0,
        target_cost_max=0.40
    )