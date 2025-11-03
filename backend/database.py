from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, Index, Boolean, Text
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Database URLs (absolute path to ensure writer/reader use the same DB file)
TICKS_DB_PATH = (Path(__file__).resolve().parent.parent / "ticks.db")
ASYNC_DB_URL = f"sqlite+aiosqlite:///{TICKS_DB_PATH.as_posix()}"
SYNC_DB_URL = f"sqlite:///{TICKS_DB_PATH.as_posix()}"

# SQLAlchemy base and engine/session factories
Base = declarative_base()
engine = create_async_engine(ASYNC_DB_URL, echo=False, future=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class TickData(Base):
    __tablename__ = "ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)


class OhlcData(Base):
    __tablename__ = "ohlc"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=True)
    high: Mapped[float] = mapped_column(Float, nullable=True)
    low: Mapped[float] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=True)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    symbol_a: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    symbol_b: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), default="1s", nullable=False)
    z_threshold: Mapped[float] = mapped_column(Float, default=2.0)
    corr_min: Mapped[float] = mapped_column(Float, default=-1.0)
    min_vol: Mapped[float] = mapped_column(Float, default=0.0)
    hysteresis: Mapped[float] = mapped_column(Float, default=0.2)
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)


# Composite indexes to accelerate symbol + timestamp filters
Index("idx_ticks_symbol_ts", TickData.symbol, TickData.timestamp)
Index("idx_ohlc_symbol_ts", OhlcData.symbol, OhlcData.timestamp)
Index("idx_alerts_pair_tf", AlertRule.symbol_a, AlertRule.symbol_b, AlertRule.timeframe)

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding an AsyncSession with commit/rollback handling."""
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create database tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
