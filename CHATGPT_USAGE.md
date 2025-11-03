# ChatGPT Usage Transparency

- Used ChatGPT (auto model) to:
  - Draft initial project structure and FastAPI/Streamlit boilerplate
  - Implement analytics functions (OLS, Huber, Kalman) and backtest helper
  - Add OHLC upload endpoint and Streamlit uploader UI
  - Create single-command runner and update README
- Prompts/themes:
  - “Design a modular FastAPI + Streamlit app for real-time Binance ticks with SQLite storage.”
  - “Implement OLS and robust regression (Huber) for hedge ratio, plus z-score and ADF.”
  - “Add Kalman-filter-based dynamic hedge estimation.”
  - “Implement OHLC CSV upload endpoint and UI; add single-command launcher.”
- Human oversight:
  - Verified API responses, adjusted schemas and defaults
  - Tuned refresh rates and DB paths; validated CSV parsing and error handling
