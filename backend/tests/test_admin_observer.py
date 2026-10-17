# 10Hz Dead Reckoning Backend Engine - SIH 2026
import asyncio
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.main import app
from app.api import websocket as ws_mod
from app.api.websocket import active_connections, device_observers


class TestAdminObserverSocket(unittest.TestCase):
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

    def _token(self, device_id: str) -> str:
        res = self.client.post("/auth/token", json={"device_id": device_id, "expires_minutes": 5})
        return res.json()["access_token"]

    def test_admin_socket_observes_without_replacing_device(self):
        dev_token = self._token("obs_dev")
        admin_token = self._token("admin")
        with self.client.websocket_connect(f"/ws/track/obs_dev?token={dev_token}") as dev_ws:
            dev_ws.send_json({"type": "time_sync"})
            dev_ws.receive_json()
            with self.client.websocket_connect(f"/ws/track/obs_dev?token={admin_token}") as admin_ws:
                admin_ws.send_json({"type": "time_sync"})
                self.assertEqual(admin_ws.receive_json()["type"], "time_sync_ack")
                # The device keeps its slot; the admin socket is only an observer
                self.assertEqual(len(device_observers.get("obs_dev", ())), 1)
                self.assertNotIn(active_connections["obs_dev"], device_observers["obs_dev"])

                # Frames sent by the observer are ignored, not fused as the device's data
                admin_ws.send_json({"type": "live", "seq": 1, "timestamp": time.time(),
                                    "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0})
                dev_ws.send_json({"type": "live", "seq": 2, "timestamp": time.time(),
                                  "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                                  "mode": "stationary"})
                dev_msg = dev_ws.receive_json()
                admin_msg = admin_ws.receive_json()
                self.assertEqual(dev_msg["type"], "position_update")
                self.assertEqual(dev_msg["seq"], 2)
                self.assertEqual(admin_msg, dev_msg)

            time.sleep(0.3)  # let the server run the observer's cleanup
            self.assertNotIn("obs_dev", device_observers)
            self.assertIn("obs_dev", active_connections)
            dev_ws.send_json({"type": "time_sync"})
            self.assertEqual(dev_ws.receive_json()["type"], "time_sync_ack")


if __name__ == "__main__":
    unittest.main()
