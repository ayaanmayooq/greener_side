import pandas as pd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().rename(f"atr{n}")

def swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: open, high, low, close, volume, adj_close
    Returns a new DataFrame with added feature columns.
    """
    out = df.copy()
    out["ret_1"]   = out["adj_close"].pct_change()
    out["mom_10"]  = out["adj_close"].pct_change(10)
    out["mom_20"]  = out["adj_close"].pct_change(20)
    out["sma_200"] = out["adj_close"].rolling(200, min_periods=200).mean()
    out["atr_14"]  = atr(out["high"], out["low"], out["adj_close"], 14)
    out.dropna(inplace=True)
    return out