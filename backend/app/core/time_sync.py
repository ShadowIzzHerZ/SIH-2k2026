# 10Hz Dead Reckoning Backend Engine - SIH 2026
import time
from typing import Dict, Any

class TimeSyncManager:
    """
    Manages lightweight NTP-style 3-way time synchronization over WebSocket.
    Calculates clock drift between mobile client and backend server.
    """

    @staticmethod
    def process_sync_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to client's time sync request with server reception and transmission timestamps.
        Client payload: {"t_client_send": float}
        """
        t_server_recv = time.time()
        t_client_send = data.get("t_client_send", t_server_recv)
        t_server_send = time.time()

        return {
            "type": "time_sync_ack",
            "t_client_send": t_client_send,
            "t_server_recv": t_server_recv,
            "t_server_send": t_server_send
        }

    @staticmethod
    def calculate_offset_and_rtt(
        t_client_send: float,
        t_server_recv: float,
        t_server_send: float,
        t_client_recv: float
    ) -> Dict[str, float]:
        """
        Standard NTP formula:
        Round-trip delay (RTT) = (t_client_recv - t_client_send) - (t_server_send - t_server_recv)
        Clock Offset (drift)  = ((t_server_recv - t_client_send) + (t_server_send - t_client_recv)) / 2
        """
        rtt = (t_client_recv - t_client_send) - (t_server_send - t_server_recv)
        offset = ((t_server_recv - t_client_send) + (t_server_send - t_client_recv)) / 2.0
        return {
            "rtt": max(0.0, rtt),
            "offset": offset
        }

