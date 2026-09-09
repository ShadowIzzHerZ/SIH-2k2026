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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Safe schema migration for movement_state & step_count
        from sqlalchemy import text
        try:
            await conn.execute(text("ALTER TABLE device_trajectories ADD COLUMN movement_state VARCHAR(32) DEFAULT 'REST'"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE device_trajectories ADD COLUMN step_count INTEGER DEFAULT 0"))
        except Exception:
            pass
        
        # If PostgreSQL/TimescaleDB is used, convert trajectory table to hypertable:
        if "postgresql" in settings.DATABASE_URL:
            try:
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

