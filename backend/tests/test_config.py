# 10Hz Dead Reckoning Backend Engine - SIH 2026
import unittest

from app.config import Settings


class TestDatabaseUrl(unittest.TestCase):
    def _url(self, value: str) -> str:
        return Settings(SECRET_KEY="test", DATABASE_URL=value).DATABASE_URL

    def test_plain_postgres_urls_use_asyncpg(self):
        self.assertEqual(self._url("postgres://u:p@db:5432/sih_db"), "postgresql+asyncpg://u:p@db:5432/sih_db")
        self.assertEqual(self._url("postgresql://u:p@db/sih_db"), "postgresql+asyncpg://u:p@db/sih_db")

    def test_explicit_drivers_untouched(self):
        for url in ("postgresql+asyncpg://u:p@db/sih_db", "sqlite+aiosqlite:///./sih_trajectories.db"):
            self.assertEqual(self._url(url), url)


if __name__ == "__main__":
    unittest.main()
