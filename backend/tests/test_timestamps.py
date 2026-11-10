# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
Trajectory timestamps must round-trip through the database unchanged, regardless
of the server's local timezone, and inserts must work on PostgreSQL/asyncpg.

Set TEST_POSTGRES_URL (e.g. postgresql+asyncpg://postgres@127.0.0.1:5432/sih_test)
to also run the PostgreSQL check; it is skipped otherwise.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Runs in a fresh interpreter so DATABASE_URL / TZ apply from the start.
ROUND_TRIP = """
import asyncio
from app.db.database import engine, Base, AsyncSessionLocal
from app.db import models
from app.db.crud import save_trajectory_point, get_last_known_position, get_trajectory_history, db_time_to_epoch

TS = 1767225600.25  # 2026-01-01T00:00:00.25Z

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as s:
        await save_trajectory_point(s, {"device_id": "tz_dev", "timestamp": TS, "x": 1.0})
        last = await get_last_known_position(s, "tz_dev")
        hist = await get_trajectory_history(s, "tz_dev")
    print(repr(last["timestamp"]), repr(db_time_to_epoch(hist[-1].time)))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

asyncio.run(main())
"""

TS = 1767225600.25


def _round_trip(database_url: str):
    env = dict(os.environ, DATABASE_URL=database_url, TZ="Asia/Kolkata")
    env.setdefault("SECRET_KEY", "test-secret")
    out = subprocess.run(
        [sys.executable, "-c", ROUND_TRIP],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return [float(v) for v in out.stdout.strip().splitlines()[-1].split()]


class TestTimestampRoundTrip(unittest.TestCase):
    def test_round_trip_sqlite_non_utc_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = _round_trip(f"sqlite+aiosqlite:///{tmp}/ts_test.db")
        for v in got:
            self.assertAlmostEqual(v, TS, places=3)

    @unittest.skipUnless(os.environ.get("TEST_POSTGRES_URL"), "TEST_POSTGRES_URL not set")
    def test_round_trip_postgres(self):
        got = _round_trip(os.environ["TEST_POSTGRES_URL"])
        for v in got:
            self.assertAlmostEqual(v, TS, places=3)


if __name__ == "__main__":
    unittest.main()
