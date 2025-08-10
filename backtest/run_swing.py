# backtest/run_swing.py
from utils.config import CFG, DATA_ROOT
import pandas as pd
from backtest.core import BacktestEngine
from backtest.strategies import BreakoutHold, TripleBarrierATR
from research.features import swing_features

def load_bars(symbol: str) -> pd.DataFrame:
    p = DATA_ROOT / f"processed/daily/symbol={symbol}/bars.parquet"
    return pd.read_parquet(p).sort_index()

if __name__ == "__main__":
    engine = BacktestEngine(feature_fn=swing_features)

    for sym in CFG["symbols"]["swing"]:
        bars = load_bars(sym)

        # Strategy 1: 20D breakout, hold 5 days
        s1 = BreakoutHold(lookback=20, hold_days=5, cost_bps=2.0)
        tr1 = engine.run_symbol(bars, s1)
        m1  = engine.metrics(tr1)
        print(f"\n=== {sym} Breakout(20)/Hold(5) ===")
        print(tr1.tail(5))
        print({k: round(v,4) if isinstance(v, float) else v for k,v in m1.items()})

        # Strategy 2: Triple-barrier with ATR
        s2 = TripleBarrierATR(trigger_col="mom_20", G_mult=2.0, L_mult=1.25, horizon=5, cost_bps=2.0)
        tr2 = engine.run_symbol(bars, s2)
        m2  = engine.metrics(tr2)
        print(f"\n=== {sym} TripleBarrier ATR (G=2.0, L=1.25, h=5) ===")
        print(tr2.tail(5))
        print({k: round(v,4) if isinstance(v, float) else v for k,v in m2.items()})