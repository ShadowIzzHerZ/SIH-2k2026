# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""A new socket for a device keeps its in-memory filter and restores only when there is none."""
import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.api import websocket as ws_mod
from app.fusion.kalman_filter import IMUKalmanFilter
from app.main import app


class TestKeepLiveFilter(unittest.TestCase):
    # The ingestion queue and DB lock are module-level asyncio objects bound to the
    # first loop that uses them, so this client's loop gets private copies.
    @classmethod
    def setUpClass(cls):
        cls._patches = [
            mock.patch.object(ws_mod, "queue_mgr", ws_mod.IngestionQueueManager(maxsize=100)),
            mock.patch.object(ws_mod, "_db_write_lock", asyncio.Lock()),
        ]
        for p in cls._patches:
            p.start()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)
        for p in cls._patches:
            p.stop()

    def _connect(self, device_id):
        token = self.client.post("/auth/token", json={"device_id": device_id}).json()["access_token"]
        return self.client.websocket_connect(f"/ws/track/{device_id}?token={token}")

    def _sync(self, ws):
        ws.send_json({"type": "time_sync"})
        ws.receive_json()

    def test_reconnect_and_duplicate_keep_filter(self):
        dev = "keep_filter_dev"
        live = IMUKalmanFilter()
        live.step_count = 7
        live._last_step_time = 123.0
        ws_mod.device_kalman_filters[dev] = live
        restore = mock.AsyncMock(return_value=IMUKalmanFilter())
        try:
            with mock.patch.object(ws_mod, "restore_or_init_filter", restore):
                with self._connect(dev) as first:
                    self._sync(first)
                    with self._connect(dev) as duplicate:
                        self._sync(duplicate)
                        self.assertIs(ws_mod.device_kalman_filters[dev], live)
                with self._connect(dev) as again:
                    self._sync(again)
                    self.assertIs(ws_mod.device_kalman_filters[dev], live)
            restore.assert_not_awaited()
            self.assertEqual(live._last_step_time, 123.0)
        finally:
            ws_mod.device_kalman_filters.pop(dev, None)

    def test_restores_when_no_filter_in_memory(self):
        dev = "restore_filter_dev"
        ws_mod.device_kalman_filters.pop(dev, None)
        cached = IMUKalmanFilter()
        restore = mock.AsyncMock(return_value=cached)
        try:
            with mock.patch.object(ws_mod, "restore_or_init_filter", restore):
                with self._connect(dev) as ws:
                    self._sync(ws)
                    self.assertIs(ws_mod.device_kalman_filters[dev], cached)
            restore.assert_awaited_once_with(dev)
        finally:
            ws_mod.device_kalman_filters.pop(dev, None)


if __name__ == "__main__":
    unittest.main()
