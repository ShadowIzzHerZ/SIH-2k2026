# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Malformed timestamp/seq values must not drop the WebSocket or break the ingestion queue."""
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.api.websocket import active_connections, _has_valid_queue_keys


def _frame(seq, ts):
    return {"type": "live", "seq": seq, "timestamp": ts, "mode": "stationary",
            "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}


class TestPacketKeyValidation(unittest.TestCase):
    def test_validator(self):
        self.assertTrue(_has_valid_queue_keys({"seq": 1, "timestamp": 1.5}))
        self.assertTrue(_has_valid_queue_keys({}))
        for bad in ({"timestamp": "abc"}, {"timestamp": None}, {"timestamp": float("nan")},
                    {"seq": "3"}, {"seq": None}, {"timestamp": True}, "not-a-dict"):
            self.assertFalse(_has_valid_queue_keys(bad), bad)


class TestWebSocketBadPackets(unittest.TestCase):
    def test_bad_timestamps_do_not_drop_connection(self):
        with TestClient(app) as client:
            token = client.post("/auth/token", json={"device_id": "ws_bad_ts", "expires_minutes": 5}).json()["access_token"]
            with client.websocket_connect(f"/ws/track/ws_bad_ts?token={token}") as ws:
                t = time.time()
                ws.send_json(_frame(1, t))
                ws.send_json(_frame(2, "abc"))
                ws.send_json(_frame(3, None))
                ws.send_json(_frame("x", t + 0.1))
                ws.send_json({"type": "backlog", "chunk": [_frame(4, t - 5), _frame(5, "later"), _frame(6, None)]})
                ws.send_json({"type": "backlog", "chunk": "nope"})
                ws.send_json(_frame(9, t + 0.2))
                time.sleep(0.5)
                self.assertIn("ws_bad_ts", active_connections)

                seqs = set()
                deadline = time.time() + 5
                while seqs != {1, 4, 9} and time.time() < deadline:
                    msg = ws.receive_json()
                    if msg.get("type") == "position_update":
                        seqs.add(msg["seq"])
                self.assertEqual(seqs, {1, 4, 9})


if __name__ == "__main__":
    unittest.main()
