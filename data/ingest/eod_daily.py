from datetime import date
from pathlib import Path
import pandas as pd
from loguru import logger

from utils.config import CFG, DATA_ROOT
from data.providers.yf_eod import YFEODEquitiesProvider

PROVIDER = YFEODEquitiesProvider()

def out_path(symbol: str) -> Path:
    return DATA_ROOT / f"processed/daily/symbol={symbol}/bars.parquet"

def get_history_window():
    sh = CFG.get("swing_history", {})
    start = sh.get("start")
    end = sh.get("end")
    start_d = date.fromisoformat(start) if start else None
    end_d   = date.fromisoformat(end) if end else None
    return start_d, end_d

if __name__ == "__main__":
    start_d, end_d = get_history_window()
    symbols = CFG["symbols"]["swing"]

    for sym in symbols:

        logger.info(f"Fetching EOD for {sym} via yfinance ({start_d} → {end_d or 'today'}) …")
        df = PROVIDER.fetch_history(sym, start=start_d, end=end_d)
        if df.empty:
            logger.warning(f"No data for {sym}")
            continue
        p = out_path(sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)

        logger.success(f"Wrote {p} rows={len(df)} | {df.index.min().date()} → {df.index.max().date()}")
        
        # tiny preview
        print(sym, df.index.min().date(), "→", df.index.max().date(), "| cols:", list(df.columns))