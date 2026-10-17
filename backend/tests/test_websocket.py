# 10Hz Dead Reckoning Backend Engine - SIH 2026
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.api.websocket import active_connections


class TestWebSocketTrack(unittest.TestCase):
    # One client (one event loop) for the class: the ingestion queue and DB lock
    # are module-level asyncio objects bound to the first loop that uses them.
    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _token(self, client: TestClient, device_id: str) -> str:
        res = client.post("/auth/token", json={"device_id": device_id, "expires_minutes": 5})
        return res.json()["access_token"]

    def test_non_object_json_does_not_drop_connection(self):
        client = self.client
        token = self._token(client, "ws_json_dev")
        with client.websocket_connect(f"/ws/track/ws_json_dev?token={token}") as ws:
            ws.send_text("[1, 2, 3]")
            ws.send_text("42")
            time.sleep(0.3)
            # Checked before receiving: a dropped handler would leave receive_json() waiting forever
            self.assertIn("ws_json_dev", active_connections)
            ws.send_json({"type": "time_sync", "t_client_send": 1.0})
            self.assertEqual(ws.receive_json()["type"], "time_sync_ack")

    def test_old_socket_closing_keeps_newer_connection_registered(self):
        client = self.client
        token = self._token(client, "ws_re_dev")
        url = f"/ws/track/ws_re_dev?token={token}"
        with client.websocket_connect(url) as old_ws:
            old_ws.send_json({"type": "time_sync"})
            old_ws.receive_json()
            with client.websocket_connect(url) as new_ws:
                new_ws.send_json({"type": "time_sync"})
                new_ws.receive_json()
                old_ws.close()
                time.sleep(0.3)  # let the server run the old handler's cleanup
                self.assertIn("ws_re_dev", active_connections)
                new_ws.send_json({"type": "time_sync"})
                self.assertEqual(new_ws.receive_json()["type"], "time_sync_ack")


if __name__ == "__main__":
    unittest.main()
