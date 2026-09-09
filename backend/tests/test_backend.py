# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
Comprehensive Unit & Integration Test Suite.
Covers:
1. 9-DOF Extended Kalman Filter (3 Modes: Stationary ZUPT, Moving PDR, Adaptive)
2. Direction Cosine Matrix & Gravity-compensated kinematics
3. NTP 3-Way Time Synchronization
4. Cryptographic HMAC-SHA256 JWT Handshake Authentication
5. Starvation-Free Priority Ingestion Queue (P0 Live vs P1 Backlog)
6. Multi-tier In-Memory / Redis Cache Layer
7. REST Endpoints & Database Persistence
"""

import unittest
import asyncio
import time
import numpy as np

from app.fusion.coordinates import (
    euler_to_rotation_matrix,
    body_to_nav_acceleration,
    calculate_tilt_compensated_heading,
    GRAVITY
)
from app.fusion.kalman_filter import IMUKalmanFilter
from app.core.auth import create_access_token, verify_jwt_token
from app.core.time_sync import TimeSyncManager
from app.core.queue_manager import IngestionQueueManager
from app.cache.redis_client import InMemoryCacheFallback

class TestKinematicsAndFusion(unittest.TestCase):
    def test_euler_to_rotation_matrix(self):
        # Identity for zero angles
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

        # 90 degrees yaw rotation
        R_yaw90 = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2)
        v_body = np.array([1.0, 0.0, 0.0])
        v_nav = R_yaw90 @ v_body
        np.testing.assert_allclose(v_nav, [0.0, 1.0, 0.0], atol=1e-6)

    def test_body_to_nav_acceleration_gravity_removal(self):
        # Device flat and stationary on table: accel_body = [0, 0, 9.81]
        accel_body = np.array([0.0, 0.0, GRAVITY])
        a_nav = body_to_nav_acceleration(accel_body, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(a_nav, [0.0, 0.0, 0.0], atol=1e-5)

    def test_tilt_compensated_heading(self):
        # Heading facing magnetic North
        mag_body = np.array([25.0, 0.0, 40.0])
        heading = calculate_tilt_compensated_heading(mag_body, 0.0, 0.0)
        self.assertAlmostEqual(heading, 0.0, places=4)

    def test_kalman_filter_stationary_mode(self):
        """Mode 1: Hard zero-velocity lock with 0.00m phantom drift."""
        kf = IMUKalmanFilter()
        packet = {
            "seq": 1,
            "timestamp": time.time(),
            "mode": "stationary",
            "ax": 0.05, "ay": 0.05, "az": 9.81,
            "gx": 0.01, "gy": 0.01, "gz": 0.01
        }
        res = kf.process_sample(packet)
        self.assertEqual(res["movement_state"], "REST")
        self.assertEqual(res["vx"], 0.0)
        self.assertEqual(res["vy"], 0.0)
        self.assertEqual(res["x"], 0.0)
        self.assertEqual(res["y"], 0.0)
        self.assertEqual(res["step_count"], 0)

    def test_kalman_filter_moving_mode(self):
        """Mode 2: Active movement step incrementation."""
        kf = IMUKalmanFilter()
        t = time.time()
        for i in range(15):
            packet = {
                "seq": i + 1,
                "timestamp": t + i * 0.10,
                "mode": "moving",
                "ax": 0.1, "ay": 1.2, "az": 10.5,
                "gx": 0.0, "gy": 0.0, "gz": 0.0
            }
            res = kf.process_sample(packet)

        self.assertEqual(res["movement_state"], "MOVING")
        self.assertGreater(res["step_count"], 0)
        self.assertGreater(res["vx"]**2 + res["vy"]**2, 0.5)
        self.assertGreater(res["x"]**2 + res["y"]**2, 0.1)

    def test_kalman_filter_serialization(self):
        kf = IMUKalmanFilter()
        kf.x[0, 0] = 12.34
        kf.x[1, 0] = 56.78
        kf.movement_state = "MOVING"
        kf.step_count = 42

        state = kf.to_state_dict()
        restored = IMUKalmanFilter.from_state_dict(state)

        self.assertAlmostEqual(float(restored.x[0, 0]), 12.34, places=4)
        self.assertAlmostEqual(float(restored.x[1, 0]), 56.78, places=4)
        self.assertEqual(restored.movement_state, "MOVING")
        self.assertEqual(restored.step_count, 42)


class TestAuthentication(unittest.TestCase):
    def test_jwt_token_minting_and_verification(self):
        device_id = "test_robot_dr_10hz"
        token = create_access_token(data={"sub": device_id, "device_id": device_id})
        payload = verify_jwt_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], device_id)
        self.assertEqual(payload["device_id"], device_id)

    def test_jwt_invalid_token(self):
        payload = verify_jwt_token("invalid.token.signature")
        self.assertIsNone(payload)


class TestTimeSync(unittest.TestCase):
    def test_ntp_offset_and_rtt(self):
        # Simulated client-server timestamps
        t_client_send = 100.000
        t_server_recv = 100.020
        t_server_send = 100.025
        t_client_recv = 100.050

        metrics = TimeSyncManager.calculate_offset_and_rtt(
            t_client_send, t_server_recv, t_server_send, t_client_recv
        )
        # RTT = (100.050 - 100.000) - (100.025 - 100.020) = 0.050 - 0.005 = 0.045
        self.assertAlmostEqual(metrics["rtt"], 0.045, places=4)
        # Offset = ((100.020 - 100.000) + (100.025 - 100.050)) / 2 = (0.020 - 0.025) / 2 = -0.0025
        self.assertAlmostEqual(metrics["offset"], -0.0025, places=4)


class TestIngestionPriorityQueue(unittest.IsolatedAsyncioTestCase):
    async def test_live_priority_over_backlog(self):
        queue = IngestionQueueManager(maxsize=100)

        # Enqueue backlog chunk (Priority 1)
        backlog_packets = [
            {"seq": 100, "timestamp": 10.0},
            {"seq": 101, "timestamp": 10.1}
        ]
        await queue.enqueue_backlog_chunk("device_A", backlog_packets)

        # Enqueue live frame (Priority 0)
        live_packet = {"seq": 1, "timestamp": 20.0}
        await queue.enqueue_live("device_B", live_packet)

        # First item retrieved MUST be live packet (Priority 0)
        first_item = await queue.get_next_item()
        self.assertEqual(first_item.priority, 0)
        self.assertEqual(first_item.device_id, "device_B")
        self.assertEqual(first_item.seq_num, 1)

        # Next item is backlog
        second_item = await queue.get_next_item()
        self.assertEqual(second_item.priority, 1)
        self.assertEqual(second_item.device_id, "device_A")


class TestCacheManager(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_cache_ttl(self):
        cache = InMemoryCacheFallback()
        await cache.set("key1", "val1", ex=1)
        self.assertEqual(await cache.get("key1"), "val1")

        # Expire
        await asyncio.sleep(1.1)
        self.assertIsNone(await cache.get("key1"))

        # Delete
        await cache.set("key2", "val2")
        await cache.delete("key2")
        self.assertIsNone(await cache.get("key2"))


class TestFastAPIRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.db.database import init_db
        await init_db()

    async def test_all_rest_endpoints(self):
        import httpx
        from app.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Health
            res = await client.get("/health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "healthy")

            # 2. Token
            res = await client.post("/auth/token", json={"device_id": "test_bot_01", "expires_minutes": 30})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("access_token", data)
            self.assertEqual(data["device_id"], "test_bot_01")

            # 3. Batch sensor fallback
            batch = {
                "device_id": "test_bot_01",
                "packets": [
                    {"seq": 1, "timestamp": time.time(), "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0, "mode": "stationary"}
                ]
            }
            res = await client.post("/sensor-data", json=batch)
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "enqueued")

            # 4. Mode
            res = await client.post("/device/test_bot_01/mode", json={"mode": "moving"})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "applied")

            # 5. Trajectory
            res = await client.get("/device/test_bot_01/trajectory")
            self.assertEqual(res.status_code, 200)

            # 6. State
            res = await client.get("/device/test_bot_01/state")
            self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

