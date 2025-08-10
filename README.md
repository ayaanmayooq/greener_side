# Algo

## Quickstart
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (we'll add later)
- Run ingesters from `data/ingest/`

## Notes / TODO
- Data source: Polygon (intraday US equities) — REST aggregates, chunked per day.
- Secrets in `.env` (POLYGON_API_KEY).