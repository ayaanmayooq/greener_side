from utils.config import CFG, DATA_ROOT
import pandas as pd
from backtest.core import BacktestEngine
from backtest.strategies import BreakoutHold, TripleBarrierATR, BuyAndHold
from research.features import swing_features

def load_bars(symbol: str) -> pd.DataFrame:
    p = DATA_ROOT / f"processed/daily/symbol={symbol}/bars.parquet"
    return pd.read_parquet(p).sort_index()

if __name__ == "__main__":
    engine = BacktestEngine(feature_fn=swing_features)
    initial_cap = 100_000.0  # change this to see $ numbers
    alloc = 1.0             # fraction of equity per trade (non-overlapping long → 1.0 is fine)

    for sym in CFG["symbols"]["swing"]:
        bars = load_bars(sym)

        # --- Strategy 1: 20D breakout, hold 5 days ---
        s1 = BreakoutHold(lookback=20, hold_days=5, cost_bps=2.0)
        tr1 = engine.run_symbol(bars, s1, initial_equity=initial_cap, alloc=alloc)
        m1  = engine.metrics(tr1)
        c1  = engine.cash_summary(tr1, initial_equity=initial_cap)

        print(f"\n=== {sym} Breakout(20)/Hold(5) ===")
        # print(tr1.tail(5)[["entry","exit","px_in","px_out","net","pnl_$","equity_$"]])
        print("Metrics :", {k: round(v,4) if isinstance(v, float) else v for k,v in m1.items()})
        print("Cash    :", {k: round(v,2)   if isinstance(v, float) else v for k,v in c1.items()})

        # --- Strategy 2: Triple-barrier with ATR ---
        s2 = TripleBarrierATR(trigger_col="mom_20", G_mult=2.0, L_mult=1.25, horizon=5, cost_bps=2.0)
        tr2 = engine.run_symbol(bars, s2, initial_equity=initial_cap, alloc=alloc)
        m2  = engine.metrics(tr2)
        c2  = engine.cash_summary(tr2, initial_equity=initial_cap)

        print(f"\n=== {sym} TripleBarrier ATR (G=2.0, L=1.25, h=5) ===")
        # print(tr2.tail(5)[["entry","exit","px_in","px_out","net","pnl_$","equity_$"]])
        print("Metrics :", {k: round(v,4) if isinstance(v, float) else v for k,v in m2.items()})
        print("Cash    :", {k: round(v,2)   if isinstance(v, float) else v for k,v in c2.items()})

        # --- Strategy 3: Buy & Hold (as a strategy) ---
        s3 = BuyAndHold(cost_bps=0.0)  # set cost if you want to model ETF fees/slippage
        tr3 = engine.run_symbol(bars, s3, initial_equity=initial_cap, alloc=1.0)
        m3  = engine.metrics(tr3)
        c3  = engine.cash_summary(tr3, initial_equity=initial_cap)

        print(f"\n=== {sym} Buy & Hold ===")
        # print(tr3.tail(1)[["entry","exit","px_in","px_out","net","pnl_$","equity_$"]])  # single long trade
        print("Metrics :", {k: round(v,4) if isinstance(v, float) else v for k,v in m3.items()})
        print("Cash    :", {k: round(v,2)   if isinstance(v, float) else v for k,v in c3.items()})

        # --- Quick head-to-head on final equity (same start/end as B&H) ---
        if not tr3.empty:
            bh_final = c3["final_equity"]
            b1_final = c1["final_equity"] if len(tr1) else initial_cap
            b2_final = c2["final_equity"] if len(tr2) else initial_cap
            edge1 = (b1_final / bh_final - 1.0) * 100.0
            edge2 = (b2_final / bh_final - 1.0) * 100.0
            print(f"\n{sym} Edge vs B&H — Breakout: {edge1:+.2f}% | TripleBarrier: {edge2:+.2f}%")