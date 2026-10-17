# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Secrets the client scripts send: from the environment or backend/.env, nothing when auth is off."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import client_auth


class TestClientAuth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env_file = Path(self.tmp.name) / ".env"
        patcher = mock.patch.object(client_auth, "_ENV_FILE", self.env_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _env(self, **values):
        return mock.patch.dict("os.environ", values, clear=True)

    def test_auth_off_sends_nothing_extra(self):
        with self._env():
            self.assertEqual(client_auth.token_request_body("dev", 60),
                             {"device_id": "dev", "expires_minutes": 60})
        self.assertEqual(client_auth.bearer_headers(""), {})
        self.assertEqual(client_auth.bearer_headers("tok"), {"Authorization": "Bearer tok"})

    def test_secret_from_environment(self):
        with self._env(DEVICE_ENROLL_SECRET="enroll", ADMIN_SECRET="adm"):
            self.assertEqual(client_auth.token_request_body("dev")["secret"], "enroll")
        with self._env(ADMIN_SECRET="adm"):
            self.assertEqual(client_auth.token_request_body("dev")["secret"], "adm")

    def test_secret_from_env_file(self):
        self.env_file.write_text("SECRET_KEY=x\n# DEVICE_ENROLL_SECRET=old\nDEVICE_ENROLL_SECRET=\"fromfile\"\n")
        with self._env():
            self.assertEqual(client_auth.enroll_secret(), "fromfile")
        with self._env(DEVICE_ENROLL_SECRET="fromenv"):
            self.assertEqual(client_auth.enroll_secret(), "fromenv")


if __name__ == "__main__":
    unittest.main()
