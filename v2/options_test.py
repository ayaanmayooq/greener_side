import pandas as pd
import backtrader as bt
from datetime import datetime
import pprint

from strats.sma_cross import SmaCross
from strats.monthly_trend import MonthlyTrend, MonthlyTrendLongShort, MonthlyTrendRiskManaged
from utils import load_yfinance_data
from metrics.directionality import (
    monthly_ls_signal, sma_longonly_signal,
    directionality_from_signal, estimate_options_overlay
)
from metrics.options_estimate import estimate_options_from_trade_analyzer
from options.suggest_options import suggest_verticals_from_df, recommend_vertical_now, size_spread
from options.simulate_options import simulate_from_recs


symbol = "AAPL"
start_date = "2015-01-01"
today = datetime.now().strftime("%Y-%m-%d")
end_date = "2025-08-25"

plot = False

strategy = MonthlyTrend
# strategy = MonthlyTrend

df_bt = load_yfinance_data(symbol, start_date, end_date)

start_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")

df_bt = df_bt[(df_bt.index >= start_date) & (df_bt.index <= end_date)]

recs = suggest_verticals_from_df(df_bt, N=20, strike_inc=1.0)
print(recs.tail(5)[['S','sigma','K_long','K_short','width','debit','cost_frac','TP_level','SL_level','notes']])

trades, summary = simulate_from_recs(
    df_bt, recs,
    N=None,                 # use each rec’s DTE
    tp_on_prem=0.85,
    tp_mode='frac_of_max', tp_target=0.85,
    sl_on_prem=0.3,
    exec_timing='next_open',
    # adaptive sauce:
    be_after_roi=0.20,
    trail_after_roi=0.50,
    trail_frac=0.40,
    min_progress_midtime=0.10,
    ema_guard=True,         # flip on if you like
    # sizing:
    start_equity=5_000,
    budget_frac=0.1,
    compound_on_close=True,
    reserve_premium=True
)

print("=== Options overlay summary ===")
# for k, v in summary.items():
#     print(f"{k}: {v}")

total_pnl = float(trades['pnl_dollars'].sum())
total_premium = float(trades['premium_used'].sum())
end_equity = summary['end_equity_fixed_budget']   # start + sum of pnl (fixed-budget policy)

print(f"Trades: {len(trades)}")
print(f"Total premium used: ${total_premium:,.2f}")
print(f"Total PnL:           ${total_pnl:,.2f}")
print(f"End equity:          ${end_equity:,.2f} (start ${summary['start_equity']:,.2f})")
print(f"Return on start:     {100* (end_equity/summary['start_equity'] - 1):.2f}%")

print("\nSample trades:")
print(trades.tail(10))

def quick_grid(df, recs, *, tps=(0.5, 0.6,0.8, 0.85), sls=(0.3,0.5), trails=(0.3,0.4),
               be=(0.2,0.3), trail_after=(0.5,0.6), budget_frac=0.1):
    rows=[]
    for tp in tps:
        for sl in sls:
            for tr in trails:
                for be_a in be:
                    for tr_a in trail_after:
                        trd,summ = simulate_from_recs(
                            df, recs,
                            tp_on_prem=tp, 
                            tp_mode='frac_of_max', tp_target=tp,
                            sl_on_prem=sl,
                            be_after_roi=be_a, trail_after_roi=tr_a, trail_frac=tr,
                            min_progress_midtime=0.10,
                            exec_timing='next_open',
                            start_equity=5_000, budget_frac=budget_frac,
                            N=20, ema_guard=True,
                            compound_on_close=True,
                            reserve_premium=True
                        )
                        rows.append({
                            "tp":tp, "sl":sl, "trail":tr, "be_after":be_a, "trail_after":tr_a,
                            "trades":summ["trades"],
                            "win_rate": (trd['roi']>0).mean() if len(trd) else 0.0,
                            "mean_roi": trd['roi'].mean() if len(trd) else 0.0,
                            "pnl": float(trd['pnl_dollars'].sum()),
                            "end_equity": summ.get("end_equity_fixed_budget", None)
                        })
    return pd.DataFrame(rows).sort_values(["end_equity","pnl","mean_roi","win_rate"], ascending=False)

# Example:
grid = quick_grid(df_bt, recs) 
print(grid.head(10))