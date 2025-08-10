# Algo

## Quickstart
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (we'll add later)
- Run ingesters from `data/ingest/`

## Notes / TODO
- Data source: Polygon (intraday US equities) — REST aggregates, chunked per day.
- Secrets in `.env` (POLYGON_API_KEY).

### Swing data (EOD OHLCV)
- Standardized columns: `open, high, low, close, volume, adj_close` (adjusted for splits/divs).
- File layout (one file per symbol):
  `data/processed/daily/symbol={SYMBOL}/bars.parquet`
- Current provider: **yfinance** (prototype). Later: **Tiingo** (recommended).
- Provider interface: `EODEquitiesProvider.fetch_history(symbol, start, end)` so we can swap sources without touching downstream code.
- Known behavior:
  - yfinance `auto_adjust=True` returns adjusted OHLC directly.
  - Tiingo provides `adjOpen/adjHigh/adjLow/adjClose/adjVolume` → we map to our standard columns.
- Next:
  - Add integrity checks (no missing days, monotonic dates).
  - Append/incremental updates (download last 30 days, merge).
  - Add dividends/splits columns (optional) for auditability.