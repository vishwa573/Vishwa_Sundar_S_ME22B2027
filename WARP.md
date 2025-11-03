# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: Real-Time Quantitative Analytics Dashboard (Python)

Commands
- Setup
  - pip install -r requirements.txt
- Run services (use three terminals, run from repo root to keep ./ticks.db in the right place)
  - Backend (FastAPI): uvicorn backend.main:app --reload
  - Ingestion (Binance WS → SQLite): python backend/websocket_client.py
  - Frontend (Streamlit): streamlit run frontend/app.py
  - Unix helper (spawns all three): bash run.sh
- API smoke tests
  - Prices/analytics: curl "http://localhost:8000/api/analytics?symbol_a=btcusdt&symbol_b=ethusdt&timeframe=1s&rolling_window=50"
  - ADF test: curl "http://localhost:8000/api/adf_test?symbol_a=btcusdt&symbol_b=ethusdt&timeframe=1s"
- Database
  - SQLite file: ./ticks.db (auto-created by websocket client)
  - Reset DB: delete ./ticks.db then restart the websocket client
- Linting/formatting
  - No linter/formatter is configured in this repo
- Testing
  - No test suite is present

Architecture and code structure (big picture)
- End-to-end flow
  - Binance Futures trade streams → websocket ingester → SQLite (ticks table) → analytics loader/computation → FastAPI JSON API → Streamlit UI
- Storage (backend/database.py)
  - SQLAlchemy model TickData(id, timestamp, symbol, price, size)
  - Async engine: sqlite+aiosqlite:///./ticks.db; schema created via init_db() when websocket_client.py starts
- Ingestion (backend/websocket_client.py)
  - Connects to two streams: wss://fstream.binance.com/ws/btcusdt@trade and .../ethusdt@trade
  - Parses trade messages into rows, buffers in-memory with asyncio.Lock, batch-inserts using SQLAlchemy insert(TickData)
  - Flush policy: BATCH_SIZE=100 or FLUSH_INTERVAL_SEC=1.0; resilient reconnect loop on errors
- Analytics (backend/analytics.py)
  - Loads raw ticks via pandas.read_sql against the sync SQLite URL; resamples last trade price per timeframe (1s/1m/5m)
  - Computes: OLS hedge ratio (statsmodels OLS of A on B), spread = A − hr·B, z-score (global mean/std), rolling correlation(window)
  - Optional ADF test on spread (adfuller) for stationarity
- API (backend/main.py)
  - GET /api/analytics: params symbol_a, symbol_b, timeframe(1s|1m|5m), rolling_window; returns prices, spread, zscore, rolling_corr, hedge_ratio, latest_zscore
  - GET /api/adf_test: params symbol_a, symbol_b, timeframe; returns statistic and pvalue
  - Cold-start handling: returns empty arrays/None until sufficient data exists
  - CORS open to all origins for local dev
- Frontend (frontend/app.py)
  - Streamlit app polling the FastAPI at API_BASE=http://localhost:8000; auto-refresh every 5s
  - Controls: symbols, timeframe, rolling window, z-threshold; renders Plotly charts (prices, spread, z-score, rolling corr)
  - On-demand ADF test call; CSV export of current analytics frame

Operational notes
- Ensure port 8000 is free for the API; first seconds may render empty until ticks accumulate and resampling fills
- Run commands from the repository root so the relative SQLite path (./ticks.db) resolves correctly
