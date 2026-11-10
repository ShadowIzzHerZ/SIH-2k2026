# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Production-safe defaults when no .env / environment overrides are present."""
import os
import unittest
from unittest import mock

from app.config import Settings


class TestSettingsDefaults(unittest.TestCase):
    def test_debug_off_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != "DEBUG"}
        env["SECRET_KEY"] = "test-only-secret-key"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(Settings(_env_file=None).DEBUG)
            with mock.patch.dict(os.environ, {"DEBUG": "True"}):
                self.assertTrue(Settings(_env_file=None).DEBUG)


if __name__ == "__main__":
    unittest.main()
