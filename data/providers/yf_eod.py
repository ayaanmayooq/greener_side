from datetime import date
import pandas as pd
import yfinance as yf
from .base_eod import EODEquitiesProvider

class YFEODEquitiesProvider(EODEquitiesProvider):
    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        df = yf.download(
            symbol,
            auto_adjust=True,  # returns adjusted OHLC
            start=None if start is None else start.strftime("%Y-%m-%d"),
            end=None if end is None else end.strftime("%Y-%m-%d"),
            progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize columns
        out = df.rename(columns=str.lower)[["open","high","low","close","volume"]].copy()
        out.index = out.index.tz_localize(None) if getattr(out.index, "tz", None) else out.index
        # yfinance with auto_adjust=True doesn't give adj close separately; compute from original if present
        if "adj close" in df.columns:
            out["adj_close"] = df["Adj Close"].values
        else:
            out["adj_close"] = out["close"]
        return out