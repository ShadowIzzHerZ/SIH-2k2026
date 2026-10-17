# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""WebSocket token from the Authorization header or Sec-WebSocket-Protocol, ?token= still accepted."""
import json
import unittest
from unittest import mock

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import websocket as ws_module
from app.core.auth import create_access_token
from app.main import app


async def _fresh_filter(device_id):
    return ws_module.IMUKalmanFilter()


class TestWsTokenSources(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(ws_module, "restore_or_init_filter", _fresh_filter)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)  # no lifespan: no DB or queue worker needed
        self.token = create_access_token({"sub": "hdr_dev", "device_id": "hdr_dev"})

    def _assert_time_sync_works(self, ws):
        ws.send_text(json.dumps({"type": "time_sync", "t_client_send": 1.0}))
        self.assertIn("t_server_recv", json.loads(ws.receive_text()))

    def test_authorization_header(self):
        with self.client.websocket_connect(
            "/ws/track/hdr_dev", headers={"Authorization": f"Bearer {self.token}"}
        ) as ws:
            self._assert_time_sync_works(ws)

    def test_subprotocol_is_accepted_and_echoed(self):
        with self.client.websocket_connect(
            "/ws/track/hdr_dev", subprotocols=["bearer", self.token]
        ) as ws:
            self.assertEqual(ws.accepted_subprotocol, "bearer")
            self._assert_time_sync_works(ws)

    def test_query_param_still_works(self):
        with self.client.websocket_connect(f"/ws/track/hdr_dev?token={self.token}") as ws:
            self.assertIsNone(ws.accepted_subprotocol)
            self._assert_time_sync_works(ws)

    def test_header_preferred_over_query_param(self):
        with self.client.websocket_connect(
            "/ws/track/hdr_dev?token=garbage", headers={"Authorization": f"Bearer {self.token}"}
        ) as ws:
            self._assert_time_sync_works(ws)

    def test_missing_or_bad_token_rejected(self):
        for kwargs in ({}, {"subprotocols": ["bearer"]}, {"subprotocols": ["bearer", "garbage"]},
                       {"headers": {"Authorization": "Bearer garbage"}}):
            with self.subTest(**kwargs):
                with self.assertRaises(WebSocketDisconnect) as ctx:
                    with self.client.websocket_connect("/ws/track/hdr_dev", **kwargs):
                        pass
                self.assertEqual(ctx.exception.code, 1008)

    def test_token_for_other_device_rejected(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                "/ws/track/someone_else", headers={"Authorization": f"Bearer {self.token}"}
            ):
                pass


if __name__ == "__main__":
    unittest.main()
