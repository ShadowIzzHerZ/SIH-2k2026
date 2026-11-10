# 10Hz Dead Reckoning Backend Engine - SIH 2026
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    # SQLite compatibility for testing:
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()

async def init_db():
    """Create database tables and TimescaleDB hypertables if available."""
    # Register the ORM models on Base.metadata, otherwise create_all is a no-op
    # when init_db runs before anything else has imported app.db.models.
    from app.db import models  # noqa: F401
    from sqlalchemy import inspect, text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Schema migration for movement_state & step_count on tables created by
        # older versions. Only ALTER when the column is actually missing: a failed
        # ALTER aborts the whole transaction on PostgreSQL, which rolls back the
        # create_all above and leaves the database without any tables.
        existing_cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("device_trajectories")}
        )
        if "movement_state" not in existing_cols:
            await conn.execute(text("ALTER TABLE device_trajectories ADD COLUMN movement_state VARCHAR(32) DEFAULT 'REST'"))
        if "step_count" not in existing_cols:
            await conn.execute(text("ALTER TABLE device_trajectories ADD COLUMN step_count INTEGER DEFAULT 0"))

        # If PostgreSQL/TimescaleDB is used, convert trajectory table to hypertable:
        if "postgresql" in settings.DATABASE_URL:
            try:
                # Savepoint: a failure here (e.g. plain PostgreSQL without the
                # extension) must not abort the outer transaction holding the tables.
                async with conn.begin_nested():
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                    # TimescaleDB requires primary key to include the partition column 'time':
                    await conn.execute(text("""
                        DO $$
                        BEGIN
                            IF EXISTS (
                                SELECT 1 FROM pg_constraint 
                                WHERE conrelid = 'device_trajectories'::regclass 
                                AND contype = 'p'
                            ) THEN
                                ALTER TABLE device_trajectories DROP CONSTRAINT device_trajectories_pkey CASCADE;
                                ALTER TABLE device_trajectories ADD PRIMARY KEY (id, time);
                            END IF;
                        EXCEPTION WHEN OTHERS THEN
                            NULL;
                        END $$;
                    """))
                    await conn.execute(
                        text("SELECT create_hypertable('device_trajectories', 'time', if_not_exists => TRUE, migrate_data => TRUE);")
                    )
                    logger.info("TimescaleDB extension and hypertable confirmed.")
            except Exception as e:
                logger.warning(f"Could not initialize TimescaleDB hypertable extension: {e}")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining async DB sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

