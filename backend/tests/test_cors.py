# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""CORS: explicit origins from CORS_ORIGINS, and no credentials with "*"."""
import unittest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.main import app, cors_settings


def _client(origins_setting: str) -> TestClient:
    probe = FastAPI()
    probe.add_middleware(CORSMiddleware, **cors_settings(origins_setting),
                         allow_methods=["*"], allow_headers=["*"])

    @probe.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(probe)


class TestCorsSettings(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(cors_settings("*"), {"allow_origins": ["*"], "allow_credentials": False})
        self.assertEqual(cors_settings(" https://a.example , https://b.example ,"),
                         {"allow_origins": ["https://a.example", "https://b.example"], "allow_credentials": True})

    def test_wildcard_does_not_echo_origin_with_credentials(self):
        res = _client("*").get("/ping", headers={"Origin": "https://evil.example", "Cookie": "a=b"})
        self.assertEqual(res.headers.get("access-control-allow-origin"), "*")
        self.assertNotIn("access-control-allow-credentials", res.headers)

    def test_explicit_origins(self):
        client = _client("https://dash.example")
        res = client.get("/ping", headers={"Origin": "https://dash.example"})
        self.assertEqual(res.headers.get("access-control-allow-origin"), "https://dash.example")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")
        res = client.get("/ping", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", res.headers)

    def test_app_default_is_wildcard_without_credentials(self):
        cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
        self.assertEqual(len(cors), 1)
        self.assertEqual(cors[0].kwargs["allow_origins"], ["*"])
        self.assertFalse(cors[0].kwargs["allow_credentials"])


if __name__ == "__main__":
    unittest.main()
