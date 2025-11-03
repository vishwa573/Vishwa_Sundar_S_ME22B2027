#!/bin/bash
set -euo pipefail

echo "Starting backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting websocket client..."
python backend/websocket_client.py &
WS_PID=$!

echo "Starting frontend..."
streamlit run frontend/app.py &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $WS_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait