# Real-Time Quantitative Analytics Dashboard

## Overview
A modular Python project that streams real-time trades from Binance Futures, stores them in SQLite, computes quantitative analytics (OLS/Huber/Kalman hedge ratio, spread, z-score, rolling correlation, ADF test), and visualizes results with Streamlit + Plotly. The backend is powered by FastAPI.

## Architecture
- WebSocket ingestion (binance futures `btcusdt@trade` and `ethusdt@trade`) via `websockets`
- SQLite storage with SQLAlchemy (async engine) and batch inserts
- Analytics layer that loads data via pandas and computes:
  - OLS / Huber / Kalman hedge ratio
  - Spread and z-score
  - Rolling correlation
  - ADF test for spread stationarity
- Optional OHLC CSV upload path feeding an `ohlc` table for offline analytics
- FastAPI exposes analytics over HTTP for the frontend
- Streamlit frontend polls backend and renders Plotly charts

```
Binance WS --> SQLite (ticks.db) --> FastAPI (/api) --> Streamlit + Plotly
                 ^                                 |
                 |------ OHLC CSV upload (optional) |
```

## Features
- Real-time streaming from Binance Futures
- OLS, Huber and Kalman hedge ratio options
- Spread z-score, rolling correlation and ADF test
- OHLC CSV upload and toggle to use uploaded data
- CSV export of latest analytics data
- Near-real-time dashboard refresh (~500ms)

## Directory Structure
```
project_root/
├── requirements.txt
├── run.py
├── README.md
├── CHATGPT_USAGE.md
├── docs/
│   ├── architecture.drawio
│   └── architecture.svg
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── websocket_client.py
│   ├── analytics.py
│   └── main.py
└── frontend/
    ├── __init__.py
    └── app.py
```

## How to Run

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Single-command run (spawns API + ingestion + UI)
```bash
python run.py
```

Alternatively, run services manually in separate terminals:
```bash
# API
uvicorn backend.main:app --reload
# Ingestion
python -m backend.websocket_client
# Frontend
streamlit run frontend/app.py
```

## Notes
- Ensure port 8000 is available for the FastAPI server.
- The first few seconds may show empty charts until enough ticks have been stored and resampled.
- Use the sidebar to upload OHLC CSVs (columns: timestamp, symbol, open, high, low, close[, volume]) and toggle "Use uploaded OHLC data".

## Deliverables & Docs
- Architecture Diagram: see `docs/architecture.drawio` and `docs/architecture.svg`.
- ChatGPT usage transparency: see `CHATGPT_USAGE.md`.

## Architecture, Design, & Scalability

### Core Design & Trade-offs
This app follows a modular, API-driven pipeline: WebSocket Ingester → SQLite → FastAPI → Streamlit, so ingestion, storage, analytics, and UI remain loosely coupled and can evolve independently [cite: 57]. The explicit API boundary allows us to run and scale each part separately, swap data sources (ticks, OHLC, CSV), and keep front-end rendering decoupled from back-end compute.

We chose SQLite for the prototype because it is zero-setup, file-based, and extremely fast to get running locally, which aligns with the assignment’s “one-day” deliverable focus [cite: 51, 60]. Its simplicity makes iteration and testing easy, and it avoids operational overhead during rapid development.

### Identified Scaling Challenges
The key bottlenecks we observed are intentional to surface real-world trade-offs [cite: 64].
- Database Contention: as the single `ticks.db` grows, SQLite’s single-file design introduces write-locking contention when the high-frequency WebSocket ingester flushes batches while the analytics API performs large reads, causing I/O stalls and increased latency.
- API Latency: CPU-heavy analytics (e.g., Kalman filters, rolling ADF, backtest grid) executed inline with live requests can lead to slow responses or timeouts under load, especially with longer lookbacks or multiple concurrent users.

## Future Mitigation (Production-Scale Path)
Database: migrate from SQLite to a client-server store like PostgreSQL (row-store with robust concurrency) or a purpose-built time-series database (e.g., InfluxDB) to sustain high-concurrent reads/writes and large histories. Partitioning, proper indexes, and WAL/retention policies would further reduce contention.

API/Analytics: move heavy analytics to asynchronous workers (e.g., Celery) and cache results in Redis so the API only serves cached summaries for near-instant responses. Long-running jobs (rolling ADF, Kalman, grid search) would be scheduled, materialized, and streamed to the UI as they complete rather than computed in-request.

## Tried Solutions & Key Decisions
- Ingestion: evaluated high-level client libraries and chose bare `websockets` for precise control over stream handling and retry behavior.
- Persistence: started with per-tick inserts; switched to batch writes via async SQLAlchemy to reduce lock contention and improve throughput.
- Analytics: compared OLS, Huber, and Kalman hedge ratios; kept all to let users trade off robustness vs. responsiveness. Tuned lookbacks to balance stability and latency.
- Frontend: considered Plotly Dash; chose Streamlit to deliver faster within the assignment timeline while retaining Plotly for charts.
- Data options: added optional OHLC CSV upload to validate algorithms offline and support reproducibility.

## Challenges Faced and What We Learned
- SQLite write locks under concurrent reads/writes: batching significantly reduced contention; a server DB (Postgres/TSDB) is the long-term fix.
- Timestamp alignment between BTCUSDT and ETHUSDT: normalized to UTC and resampled to a common cadence before computing spread/correlation to avoid bias.
- ADF sensitivity on short samples: very short windows can yield unstable p-values; prefer longer windows or cached/materialized stats for stability.
- UI refresh vs. API compute: aggressive refresh intervals can collide with heavy analytics; caching or background workers mitigate this.

## Additional (Out-of-scope) Items Worth Calling Out
These were considered or partially explored because they strengthen a production version, though not required by the assignment:
- Enable SQLite WAL mode and indices on time/symbol to improve concurrent access.
- Background jobs + Redis cache for heavy analytics with streaming partial results to the UI.
- Postgres migration with partitioning and retention policy for long histories.
- Robust reconnect/backoff for WebSocket ingestion with jitter and health probes.

