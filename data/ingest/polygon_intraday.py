from __future__ import annotations
from datetime import datetime, timedelta, timezone
import asyncio
import httpx
import pandas as pd
from loguru import logger

from utils.env import POLYGON_API_KEY
from utils.config import CFG, DATA_ROOT

POLY_BASE = "https://api.polygon.io"

def _out_path(symbol: str, day: datetime):
    d = day.strftime("%Y-%m-%d")
    return DATA_ROOT / f"processed/intraday/symbol={symbol}/date={d}/bars.parquet"

def _day_str(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")  # <- required format for from/to

def _is_weekend(day: datetime) -> bool:
    return day.weekday() >= 5  # 5=Sat, 6=Sun

async def fetch_day(client: httpx.AsyncClient, symbol: str, day: datetime) -> pd.DataFrame:
    # Use date strings (Polygon will "snap/stretch" internally)
    # See: aggs docs + limit behavior notes
    from_d = _day_str(day)
    to_d   = _day_str(day)  # same day is fine for 1m bars
    url = f"{POLY_BASE}/v2/aggs/ticker/{symbol}/range/1/minute/{from_d}/{to_d}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    r = await client.get(url, params=params, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        logger.info(f"No results for {symbol} on {from_d} (status={data.get('status')})")
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("ts").rename(columns={
        "o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap","n":"trades"
    })[["open","high","low","close","volume","vwap","trades"]]
    return df

async def ingest_intraday(symbol: str, days_back: int = 7):
    async with httpx.AsyncClient() as client:
        start_day = datetime.now(timezone.utc).date() - timedelta(days=days_back)
        for i in range(days_back + 1):
            d = datetime.combine(start_day + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc)
            if _is_weekend(d):  # skip Sat/Sun
                continue
            out = _out_path(symbol, d)
            if out.exists():
                continue
            try:
                df = await fetch_day(client, symbol, d)
            except httpx.HTTPStatusError as e:
                # 400s can happen on holidays/invalid date formats
                logger.error(f"{symbol} {d.date()} HTTP {e.response.status_code}: {e}")
                continue
            if not df.empty:
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out)
                logger.info(f"Wrote {out} rows={len(df)}")

if __name__ == "__main__":
    symbol = CFG["symbols"]["intraday"][0]
    asyncio.run(ingest_intraday(symbol, days_back=7))