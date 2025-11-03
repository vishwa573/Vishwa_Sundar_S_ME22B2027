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

