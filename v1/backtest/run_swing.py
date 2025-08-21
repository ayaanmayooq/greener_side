# backtest/run_swing.py
from utils.config import CFG, DATA_ROOT
import pandas as pd

from backtest.core import BacktestEngine
from backtest.strategies.buy_and_hold import BuyAndHold
from backtest.strategies.breakout_hold import BreakoutHold
from backtest.strategies.triple_barrier_atr import TripleBarrierATR
from backtest.strategies.prob_triple_barrier import ProbTripleBarrier

from research.features.registry import build_features

INITIAL_CAP = 50_000.0
ALLOC = 1.0

# edit this list to add/remove strategies you want to test
STRATEGIES = [
    ("Buy&Hold",              BuyAndHold(0.0)),
    ("Breakout(20)/Hold(5)",  BreakoutHold(lookback=20, hold_days=5, cost_bps=2.0)),
    ("TripleBarrier ATR",     TripleBarrierATR(trigger_col="mom_20", G_mult=3.0, L_mult=1.0, horizon=10, cost_bps=2.0)),
    ("Prob-Gated TB (v1)",    ProbTripleBarrier(model_id="swing_xgb_prob_tp_v1", G_mult=3.0, L_mult=1.0, horizon=10, margin=0.05, cost_bps=2.0)),
]

def load_bars(symbol: str) -> pd.DataFrame:
    p = DATA_ROOT / f"processed/daily/symbol={symbol}/bars.parquet"
    return pd.read_parquet(p).sort_index()

if __name__ == "__main__":
    for sym in CFG["symbols"]["swing"]:
        bars = load_bars(sym)

        for label, strat in STRATEGIES:
            # ask the strategy what features it needs
            feature_keys = strat.required_feature_keys() if hasattr(strat, "required_feature_keys") else ["swing_core"]

            # feature function using the registry
            def feature_fn(df, _sym=sym, _keys=feature_keys):
                return build_features(_sym, df, _keys, DATA_ROOT)

            engine = BacktestEngine(feature_fn=feature_fn)

            try:
                tr = engine.run_symbol(bars, strat, initial_equity=INITIAL_CAP, alloc=ALLOC)
            except FileNotFoundError as e:
                print(f"\n=== {sym} {label} ===")
                print(f"Skipping: {e}")
                continue

            m  = engine.metrics(tr)
            c  = engine.cash_summary(tr, initial_equity=INITIAL_CAP)

            print(f"\n=== {sym} {label} ===")
            print("Metrics :", {k: round(float(v),4) if isinstance(v,(int,float)) else v for k,v in m.items()})
            print("Cash    :", {k: round(float(v),2)   if isinstance(v,(int,float)) else v for k,v in c.items()})