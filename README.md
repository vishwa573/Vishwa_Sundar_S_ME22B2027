# Real-Time Quantitative Analytics Dashboard

## Overview
A modular Python project that streams real-time trades from Binance Futures, stores them in SQLite, computes quantitative analytics (OLS hedge ratio, spread, z-score, rolling correlation, ADF test), and visualizes results with Streamlit + Plotly. The backend is powered by FastAPI.

## Architecture
- WebSocket ingestion (binance futures `btcusdt@trade` and `ethusdt@trade`) via `websockets`
- SQLite storage with SQLAlchemy (async engine) and batch inserts
- Analytics layer that loads data via pandas and computes:
  - OLS hedge ratio (statsmodels)
  - Spread and z-score
  - Rolling correlation
  - ADF test for spread stationarity
- FastAPI exposes analytics over HTTP for the frontend
- Streamlit frontend polls backend and renders Plotly charts

```
Binance WS --> SQLite (ticks.db) --> FastAPI (/api) --> Streamlit + Plotly
```

## Features
- Real-time streaming from Binance Futures
- OLS hedge ratio and spread z-score
- Rolling correlation visualization
- ADF test on the spread
- CSV export of latest analytics data
- Auto-refreshing dashboard (5s)

## Directory Structure
```
project_root/
├── requirements.txt
├── run.sh
├── README.md
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

2) Start the FastAPI backend (Terminal 1)
```bash
uvicorn backend.main:app --reload
```

3) Start the WebSocket client (Terminal 2)
```bash
python -m backend.websocket_client
```

4) Start the Streamlit frontend (Terminal 3)
```bash
streamlit run frontend/app.py
```

Or on Unix shells, you can use the helper script:
```bash
bash run.sh
```

## Notes
- Ensure port 8000 is available for the FastAPI server.
- The first few seconds may show empty charts until enough ticks have been stored and resampled.

