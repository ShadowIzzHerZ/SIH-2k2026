# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Token minting secrets and REST token checks (off until DEVICE_ENROLL_SECRET is set)."""
import contextlib
import unittest
from unittest import mock

import httpx

from app.core import auth
from app.main import app


class TestAuthSecrets(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    @contextlib.contextmanager
    def _secrets(self, enroll="", admin=""):
        with mock.patch.object(auth.settings, "DEVICE_ENROLL_SECRET", enroll), \
             mock.patch.object(auth.settings, "ADMIN_SECRET", admin):
            yield

    async def _mint(self, device_id, secret=None):
        body = {"device_id": device_id, "expires_minutes": 5}
        if secret is not None:
            body["secret"] = secret
        return await self.client.post("/auth/token", json=body)

    async def test_defaults_keep_open_behaviour_and_warn(self):
        with self._secrets(), mock.patch.object(auth, "_warned_open_enrollment", False):
            with self.assertLogs("app.core.auth", level="WARNING"):
                res = await self._mint("open_dev")
            self.assertEqual(res.status_code, 200)
            self.assertEqual((await self._mint("admin")).status_code, 200)
            # The token check itself is skipped, so no Bearer header is needed
            self.assertIsNone(await auth.get_rest_subject(None))
            auth.ensure_device_access(None, "any_device")

    async def test_enroll_secret_required_for_device_tokens(self):
        with self._secrets(enroll="enroll-s"):
            self.assertEqual((await self._mint("dev_a")).status_code, 403)
            self.assertEqual((await self._mint("dev_a", "wrong")).status_code, 403)
            self.assertEqual((await self._mint("dev_a", "enroll-s")).status_code, 200)
            # No ADMIN_SECRET configured: admin cannot be minted at all, not even with the enroll secret
            self.assertEqual((await self._mint("admin", "enroll-s")).status_code, 403)
            self.assertEqual((await self._mint("admin")).status_code, 403)

    async def test_admin_token_needs_admin_secret(self):
        with self._secrets(admin="admin-s"):
            self.assertEqual((await self._mint("admin")).status_code, 403)
            self.assertEqual((await self._mint("admin", "nope")).status_code, 403)
            self.assertEqual((await self._mint("admin", "admin-s")).status_code, 200)
            # Device minting stays open while DEVICE_ENROLL_SECRET is unset
            self.assertEqual((await self._mint("dev_b")).status_code, 200)

    async def test_rest_endpoints_need_matching_token_when_enabled(self):
        with self._secrets(enroll="enroll-s", admin="admin-s"):
            tok_a = (await self._mint("dev_a", "enroll-s")).json()["access_token"]
            tok_b = (await self._mint("dev_b", "enroll-s")).json()["access_token"]
            tok_admin = (await self._mint("admin", "admin-s")).json()["access_token"]
            hdr = lambda t: {"Authorization": f"Bearer {t}"}

            for method, url, body in [
                ("GET", "/device/dev_a/state", None),
                ("GET", "/device/dev_a/trajectory", None),
                ("POST", "/device/dev_a/mode", {"mode": "adaptive"}),
                ("POST", "/sensor-data", {"device_id": "dev_a", "packets": []}),
            ]:
                with self.subTest(url=url):
                    res = await self.client.request(method, url, json=body)
                    self.assertEqual(res.status_code, 401)
                    res = await self.client.request(method, url, json=body, headers=hdr("not-a-jwt"))
                    self.assertEqual(res.status_code, 401)
                    res = await self.client.request(method, url, json=body, headers=hdr(tok_b))
                    self.assertEqual(res.status_code, 403)
                    # Auth passes for the device itself and for admin (the handler may still
                    # fail later on a missing table, which is not what this test checks)
                    for tok in (tok_a, tok_admin):
                        await auth.require_device_access("dev_a", await auth.get_rest_subject(
                            auth.HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)))

            res = await self.client.post("/device/dev_a/mode", json={"mode": "adaptive"}, headers=hdr(tok_a))
            self.assertEqual(res.status_code, 200)
            res = await self.client.post("/sensor-data", json={"device_id": "dev_a", "packets": []},
                                         headers=hdr(tok_admin))
            self.assertEqual(res.status_code, 200)
            self.assertEqual((await self.client.get("/health")).status_code, 200)


if __name__ == "__main__":
    unittest.main()
