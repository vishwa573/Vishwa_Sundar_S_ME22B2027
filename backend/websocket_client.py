import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set

import websockets
from sqlalchemy import insert, select

from backend.database import TickData, Subscription, get_session, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ws-client")

DEFAULT_SYMBOLS: List[str] = [
    "btcusdt",
    "ethusdt",
    "bnbusdt",
    "xrpusdt",
    "adausdt",
    "solusdt",
]

BUFFER: List[dict] = []
BUFFER_LOCK = asyncio.Lock()
BATCH_SIZE = 100
FLUSH_INTERVAL_SEC = 0.5

# Live tasks per URL
TASKS: Dict[str, asyncio.Task] = {}
TASKS_LOCK = asyncio.Lock()


def _url_for(sym: str) -> str:
    return f"wss://fstream.binance.com/ws/{sym.lower()}@trade"


def _parse_msg(msg: dict) -> dict | None:
    try:
        if msg.get("e") != "trade":
            return None
        ts = datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc).replace(tzinfo=None)
        return {
            "timestamp": ts,
            "symbol": msg["s"].lower(),
            "price": float(msg["p"]),
            "size": float(msg["q"]),
        }
    except Exception as e:
        logger.error("Failed to parse message: %s", e)
        return None


async def _flush_buffer(reason: str) -> None:
    global BUFFER
    async with BUFFER_LOCK:
        if not BUFFER:
            return
        batch = BUFFER
        BUFFER = []
    try:
        async with get_session() as session:
            await session.execute(insert(TickData), batch)
        logger.info("Batch insert: %d rows (%s)", len(batch), reason)
    except Exception as e:
        logger.exception("Batch insert failed: %s", e)


async def _periodic_flusher() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SEC)
        await _flush_buffer("interval")


async def _consume(url: str) -> None:
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                logger.info("Connected: %s", url)
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    row = _parse_msg(msg)
                    if row is None:
                        continue
                    async with BUFFER_LOCK:
                        BUFFER.append(row)
                        if len(BUFFER) >= BATCH_SIZE:
                            asyncio.create_task(_flush_buffer("size"))
        except asyncio.CancelledError:
            logger.info("Cancelled consumer: %s", url)
            break
        except Exception as e:
            logger.error("WebSocket error (%s): %s", url, e)
            await asyncio.sleep(3)


async def _ensure_default_subscriptions():
    # Seed defaults if empty
    async with get_session() as s:
        res = await s.execute(select(Subscription))
        rows = res.scalars().all()
        if not rows:
            s.add_all([Subscription(symbol=sym) for sym in DEFAULT_SYMBOLS])


async def _subscription_manager():
    while True:
        try:
            async with get_session() as s:
                res = await s.execute(select(Subscription))
                rows = res.scalars().all()
                syms = {r.symbol.lower() for r in rows}
            desired_urls: Set[str] = {_url_for(sym) for sym in syms}
            async with TASKS_LOCK:
                # Start new tasks
                for url in desired_urls - set(TASKS.keys()):
                    TASKS[url] = asyncio.create_task(_consume(url))
                # Cancel removed tasks
                for url in set(TASKS.keys()) - desired_urls:
                    TASKS[url].cancel()
                    del TASKS[url]
        except Exception as e:
            logger.error("Subscription manager error: %s", e)
        await asyncio.sleep(5)


async def main() -> None:
    await init_db()
    await _ensure_default_subscriptions()
    mgr = asyncio.create_task(_subscription_manager())
    flusher = asyncio.create_task(_periodic_flusher())
    await asyncio.gather(mgr, flusher)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
