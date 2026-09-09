# ==============================================================================
# SIH 2026: End-to-End Production Deployment Verification Suite
# Usage:
#   python tests/verify_deployment.py
#   python tests/verify_deployment.py --url https://your-domain.com
# ==============================================================================
import argparse
import asyncio
import json
import random
import sys
import time
import httpx
import websockets

def print_banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)

async def test_health(client: httpx.AsyncClient, base_url: str) -> bool:
    print("\n[STEP 1/5] Testing GET /health...")
    try:
        res = await client.get(f"{base_url}/health", timeout=5.0)
        data = res.json()
        print(f"  Status Code: {res.status_code}")
        print(f"  System Status: {data.get('status')}")
        print(f"  Redis Connected: {data.get('redis_connected')}")
        print(f"  Queue Depth: {data.get('queue_depth')}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert data.get("status") == "healthy", f"Expected healthy status"
        print("  --> Health Check: PASSED")
        return True
    except Exception as e:
        print(f"  --> Health Check: FAILED ({e})")
        return False

async def test_auth(client: httpx.AsyncClient, base_url: str, device_id: str) -> str:
    print(f"\n[STEP 2/5] Testing POST /auth/token for '{device_id}'...")
    try:
        res = await client.post(
            f"{base_url}/auth/token",
            json={"device_id": device_id, "expires_minutes": 60},
            timeout=5.0
        )
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        token = res.json().get("access_token")
        assert token, "No access_token returned"
        print(f"  JWT Token Acquired: {token[:20]}...{token[-10:]}")
        print("  --> Auth Token: PASSED")
        return token
    except Exception as e:
        print(f"  --> Auth Token: FAILED ({e})")
        return ""

async def test_websocket_stream(ws_base: str, device_id: str, token: str) -> bool:
    print(f"\n[STEP 3/5] Testing WebSocket 10Hz Dead Reckoning Stream...")
    ws_url = f"{ws_base}/ws/track/{device_id}?token={token}"
    print(f"  Connecting to {ws_url}...")
    try:
        async with websockets.connect(ws_url, close_timeout=3.0) as ws:
            print("  WebSocket Handshake: OK")

            # 1. NTP Time Sync
            t_send = time.time()
            await ws.send(json.dumps({"type": "time_sync", "t_client_send": t_send}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            t_recv = time.time()
            rtt_ms = round((t_recv - t_send) * 1000, 2)
            print(f"  NTP Handshake: Round-Trip Time = {rtt_ms}ms")
            assert resp.get("type") == "time_sync_ack"

            # 2. 10Hz Live Telemetry Packets
            received_feedback = 0
            for seq in range(1, 6):
                packet = {
                    "type": "live",
                    "mode": "moving",
                    "seq": seq,
                    "timestamp": time.time(),
                    "ax": 0.05 + random.uniform(-0.02, 0.02),
                    "ay": 0.95 + random.uniform(-0.02, 0.02),
                    "az": 9.81 + random.uniform(-0.1, 0.1),
                    "gx": 0.01, "gy": 0.01, "gz": 0.02,
                    "mx": 22.5, "my": -5.1, "mz": 41.2
                }
                await ws.send(json.dumps(packet))
                try:
                    fb = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                    if fb.get("type") == "position_update":
                        received_feedback += 1
                        print(f"    [10Hz ACK] Seq={fb.get('seq')} Pos=({fb.get('x')}, {fb.get('y')}) State={fb.get('movement_state')} Steps={fb.get('step_count')}")
                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(0.10)  # 10Hz rate

            print(f"  Received {received_feedback}/5 verified 10Hz position feedback frames.")
            assert received_feedback > 0, "No Kalman position feedback received"
            print("  --> WebSocket 10Hz Fusion: PASSED")
            return True
    except Exception as e:
        print(f"  --> WebSocket 10Hz Fusion: FAILED ({e})")
        return False

async def test_device_state(client: httpx.AsyncClient, base_url: str, device_id: str) -> bool:
    print(f"\n[STEP 4/5] Testing GET /device/{device_id}/state (Persistence & Cache)...")
    try:
        # Give queue worker a brief moment to commit to cache/db
        await asyncio.sleep(0.5)
        res = await client.get(f"{base_url}/device/{device_id}/state", timeout=5.0)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print(f"  Persisted Position: ({data.get('x')}, {data.get('y')}, {data.get('z')})")
        print(f"  Movement State: {data.get('movement_state')} | Steps: {data.get('step_count')}")
        print("  --> Persistence & Cache Check: PASSED")
        return True
    except Exception as e:
        print(f"  --> Persistence & Cache Check: FAILED ({e})")
        return False

async def test_trajectory_history(client: httpx.AsyncClient, base_url: str, device_id: str) -> bool:
    print(f"\n[STEP 5/5] Testing GET /device/{device_id}/trajectory...")
    try:
        res = await client.get(f"{base_url}/device/{device_id}/trajectory?limit=10", timeout=5.0)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        history = res.json()
        print(f"  Retrieved {len(history)} historical trajectory records.")
        assert len(history) > 0, "Expected at least 1 trajectory record"
        print("  --> Trajectory History: PASSED")
        return True
    except Exception as e:
        print(f"  --> Trajectory History: FAILED ({e})")
        return False

async def main():
    parser = argparse.ArgumentParser(description="SIH 2026 Production Deployment Verifier")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base HTTP URL of backend")
    args = parser.parse_args()

    http_base = args.url.rstrip("/")
    ws_base = http_base.replace("https://", "wss://").replace("http://", "ws://")
    device_id = f"verifier_{int(time.time())}"

    print_banner("SIH 2026: DEAD RECKONING PRODUCTION VERIFICATION")
    print(f"  Target Endpoint: {http_base}")
    print(f"  WebSocket URL:   {ws_base}")
    print(f"  Test Device ID:  {device_id}")

    results = {}
    async with httpx.AsyncClient() as client:
        results["Health"] = await test_health(client, http_base)
        token = await test_auth(client, http_base, device_id)
        results["Authentication"] = bool(token)

        if token:
            results["WebSocket 10Hz"] = await test_websocket_stream(ws_base, device_id, token)
            results["State Persistence"] = await test_device_state(client, http_base, device_id)
            results["Trajectory History"] = await test_trajectory_history(client, http_base, device_id)
        else:
            results["WebSocket 10Hz"] = False
            results["State Persistence"] = False
            results["Trajectory History"] = False

    print_banner("VERIFICATION SUMMARY REPORT")
    all_passed = True
    for test, passed in results.items():
        tag = "PASSED [OK]" if passed else "FAILED [X]"
        print(f"  {test:<25} : {tag}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL PRODUCTION SYSTEMS OPERATIONAL AND VERIFIED! [OK]")
        sys.exit(0)
    else:
        print("\nONE OR MORE CHECKS FAILED. Check logs above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
