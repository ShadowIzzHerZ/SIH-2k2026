# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
init_db must create the schema on its own (without app.db.models having been
imported first) and must leave the tables in place on PostgreSQL even when the
TimescaleDB extension is unavailable.

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

# Runs in a fresh interpreter so no earlier import has registered the models.
INIT_AND_LIST_TABLES = """
import asyncio
from sqlalchemy import inspect
from app.db.database import init_db, engine

async def main():
    await init_db()
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda c: sorted(inspect(c).get_table_names()))
    print(",".join(names))
    await engine.dispose()

asyncio.run(main())
"""


def _run_init(database_url: str) -> str:
    env = dict(os.environ, DATABASE_URL=database_url)
    env.setdefault("SECRET_KEY", "test-secret")
    out = subprocess.run(
        [sys.executable, "-c", INIT_AND_LIST_TABLES],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return out.stdout.strip().splitlines()[-1]


class TestInitDb(unittest.TestCase):
    def test_init_db_creates_tables_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            url = f"sqlite+aiosqlite:///{tmp}/init_test.db"
            tables = _run_init(url).split(",")
            self.assertIn("device_trajectories", tables)
            self.assertIn("device_sessions", tables)
            # Second run against an existing schema must also succeed
            self.assertIn("device_trajectories", _run_init(url).split(","))

    @unittest.skipUnless(os.environ.get("TEST_POSTGRES_URL"), "TEST_POSTGRES_URL not set")
    def test_init_db_creates_tables_postgres(self):
        url = os.environ["TEST_POSTGRES_URL"]
        tables = _run_init(url).split(",")
        self.assertIn("device_trajectories", tables)
        self.assertIn("device_trajectories", _run_init(url).split(","))


if __name__ == "__main__":
    unittest.main()
