import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List

import websockets
from sqlalchemy import insert

from backend.database import TickData, get_session, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ws-client")

WS_URLS: List[str] = [
    "wss://fstream.binance.com/ws/btcusdt@trade",
    "wss://fstream.binance.com/ws/ethusdt@trade",
]

BUFFER: List[dict] = []
BUFFER_LOCK = asyncio.Lock()
BATCH_SIZE = 100
FLUSH_INTERVAL_SEC = 1.0


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
                        # print(f"GOT A TICK: {msg}")
                    except json.JSONDecodeError:
                        continue
                    row = _parse_msg(msg)
                    if row is None:
                        continue
                    async with BUFFER_LOCK:
                        BUFFER.append(row)
                        if len(BUFFER) >= BATCH_SIZE:
                            asyncio.create_task(_flush_buffer("size"))
        except Exception as e:
            logger.error("WebSocket error (%s): %s", url, e)
            await asyncio.sleep(3)


async def main() -> None:
    await init_db()
    tasks = [asyncio.create_task(_consume(u)) for u in WS_URLS]
    tasks.append(asyncio.create_task(_periodic_flusher()))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
