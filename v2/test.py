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


symbol = "GOOGL"
start_date = "2024-01-01"
today = datetime.now().strftime("%Y-%m-%d")
end_date = "2025-08-20"

plot = False

strategy = MonthlyTrend
# strategy = MonthlyTrend

df_bt = load_yfinance_data(symbol, start_date, end_date)

start_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")

data = bt.feeds.PandasData(dataname=df_bt,
                           fromdate=start_date,
                           todate=end_date,)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(strategy)
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.0001)
cerebro.broker.set_slippage_perc(0.0001)

cerebro.addsizer(bt.sizers.PercentSizer, percents=99)

cerebro.addanalyzer(bt.analyzers.AnnualReturn,   _name='annual')
cerebro.addanalyzer(bt.analyzers.DrawDown,       _name='dd')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer,  _name='trades')
cerebro.addanalyzer(bt.analyzers.SharpeRatio_A,  _name='sharpeA', riskfreerate=0.0)
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyf')

res = cerebro.run()
st = res[0]

start = 100_000
end = cerebro.broker.getvalue()
pnl = end - start
ret = 100*(end/start - 1)
dd  = st.analyzers.dd.get_analysis().max.drawdown
sharp = st.analyzers.sharpeA.get_analysis().get('sharperatio', None)

ta = st.analyzers.trades.get_analysis()
# import pprint; pprint.pprint(ta)

def get(dic, *keys, default=0):
    for k in keys:
        if not isinstance(dic, dict) or k not in dic: return default
        dic = dic[k]
    return dic

trades = get(ta, 'total', 'total', default=0)
won    = get(ta, 'won', 'total', default=0)
lost   = get(ta, 'lost', 'total', default=0)
winpct = 100*won/trades if trades else 0

gprofit= get(ta, 'won','pnl','total', default=0.0)
gloss  = get(ta, 'lost','pnl','total',   default=0.0)
pf     = (gprofit/abs(gloss)) if gloss else float('inf')

avgwin = get(ta, 'won','pnl','average', default=0.0)
avgloss= get(ta, 'lost','pnl','average', default=0.0)
annual = st.analyzers.annual.get_analysis()

print(f"Start Value: ${start:,.2f}")
print(f"End Value:   ${end:,.2f}  (P/L: ${pnl:,.2f}, {ret:.2f}%)")
print(f"Sharpe(A):   {sharp}")
print(f"Max DD:      {dd:.2f} %")
print(f"Trades:      {trades} | Win%: {winpct:.1f}% | PF: {pf:.2f}")
print(f"Avg Win:     {avgwin:.2f} | Avg Loss: {avgloss:.2f}")
# print("Annual Returns:", annual)

# _ = estimate_options_from_trade_analyzer(
#     trades=trades, won=won, avgwin=avgwin, avgloss=avgloss,
#     start_equity=start,
#     backtest_sizer=0.10,     # your PercentSizer(10%)
#     estimate_sizer=0.01,     # model 1% per idea
#     spread_width_pct=0.03,   # 3% wide vertical
#     premium_cost_frac=0.30,  # costs ~30% of width
#     realism_beta=0.50,
#     win_capture_frac=0.60,
#     loser_stop_frac=0.50,    # cut losers at -50% premium
#     per_trade_budget_frac=0.1,   # spend 1% per trade
#     pot_budget_frac=0.001,         # 1% pot compounding
#     equity_reinvest_frac=0.1,    # reinvest 1% of equity each trade
#     verbose=True
# )

if plot:
    cerebro.plot()
