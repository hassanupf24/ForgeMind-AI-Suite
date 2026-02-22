"""
ForgeMind AI Suite — Database Engine & Session Management
Async PostgreSQL (relational) + TimescaleDB (time-series).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings

settings = get_settings()

# ── Naming convention for Alembic auto-generation ──
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    metadata = metadata


# ── Primary PostgreSQL Engine ──
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── TimescaleDB Engine ──
timescale_engine = create_async_engine(
    settings.timescale_url,
    echo=settings.debug,
    pool_size=30,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)

timescale_session_factory = async_sessionmaker(
    bind=timescale_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope for relational database operations."""
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_timescale_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope for time-series database operations."""
    session = timescale_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_databases() -> None:
    """Initialize database schemas and TimescaleDB extensions."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with timescale_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        await conn.run_sync(Base.metadata.create_all)


async def close_databases() -> None:
    """Dispose of all database connection pools."""
    await engine.dispose()
    await timescale_engine.dispose()
