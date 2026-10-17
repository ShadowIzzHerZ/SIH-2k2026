# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""A short or placeholder SECRET_KEY is logged as a warning (startup continues)."""
import os
import unittest

os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

from app.main import warn_if_weak_secret_key


class TestWeakSecretWarning(unittest.TestCase):
    def test_short_key_warns(self):
        with self.assertLogs("sih_backend", level="WARNING") as logs:
            self.assertTrue(warn_if_weak_secret_key("short-secret"))
        self.assertIn("12 bytes", logs.output[0])

    def test_placeholder_key_warns(self):
        with self.assertLogs("sih_backend", level="WARNING"):
            self.assertTrue(warn_if_weak_secret_key("change-me-generate-with-openssl-rand-hex-32"))

    def test_strong_key_is_silent(self):
        with self.assertNoLogs("sih_backend", level="WARNING"):
            self.assertFalse(warn_if_weak_secret_key("a" * 64))


if __name__ == "__main__":
    unittest.main()
