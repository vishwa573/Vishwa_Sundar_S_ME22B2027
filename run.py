import os
import sys
import subprocess
import time
import signal

PROCESSES = []


def spawn(cmd, env=None):
    print(f"Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env or os.environ.copy())


def main():
    try:
        # Backend API (FastAPI + Uvicorn)
        api = spawn([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"])  # localhost:8000
        PROCESSES.append(api)

        # Ingestion (Binance WS -> SQLite)
        ing = spawn([sys.executable, "-m", "backend.websocket_client"])  # writes ./ticks.db
        PROCESSES.append(ing)

        # Frontend (Streamlit)
        fe = spawn([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])  # opens in browser
        PROCESSES.append(fe)

        # Wait for children
        while True:
            time.sleep(1)
            # Exit if any process exits with error
            for p in PROCESSES:
                ret = p.poll()
                if ret is not None and ret != 0:
                    raise SystemExit(f"Process exited with code {ret}: {p.args}")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in PROCESSES:
            if p.poll() is None:
                try:
                    if os.name == "nt":
                        p.terminate()
                    else:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
        # Give them a moment to exit
        time.sleep(1)
        for p in PROCESSES:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
