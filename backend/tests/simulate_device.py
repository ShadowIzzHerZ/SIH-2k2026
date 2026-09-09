# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
Backend Integration Test Simulator (10Hz IMU).
Simulates incoming 10Hz client sensor streams, network outages, backlog chunking,
NTP time synchronization, and verified coordinate streaming.
"""

import asyncio
import json
import random
import time
import httpx
import websockets

BACKEND_HTTP = "http://127.0.0.1:8000"
BACKEND_WS = "ws://127.0.0.1:8000"
DEVICE_ID = "phone_user_sih_01"

async def get_token() -> str:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BACKEND_HTTP}/auth/token",
            json={"device_id": DEVICE_ID, "expires_minutes": 60}
        )
        data = res.json()
        token = data["access_token"]
        print(f"[AUTH] Obtained JWT Token: {token[:25]}...")
        return token

def generate_10hz_packet(seq: int, timestamp: float, walking: bool = True):
    """Generates synthetic 10Hz IMU packet (100ms interval)."""
    ax = 0.2 * random.uniform(-0.1, 0.1)
    ay = 0.5 * random.uniform(0.9, 1.1) if walking else 0.0
    az = 9.81 + (0.8 * random.uniform(-1, 1) if walking else 0.0)

    gx = 0.03 * random.uniform(-1, 1)
    gy = 0.03 * random.uniform(-1, 1)
    gz = 0.02 * random.uniform(-1, 1)

    return {
        "seq": seq,
        "timestamp": timestamp,
        "ax": round(ax, 4),
        "ay": round(ay, 4),
        "az": round(az, 4),
        "gx": round(gx, 4),
        "gy": round(gy, 4),
        "gz": round(gz, 4),
        "mx": 22.5,
        "my": -5.1,
        "mz": 41.2
    }

async def run_simulation():
    print("=" * 65)
    print(" Starting Backend Integration Test (10Hz Ingestion & Prioritization)")
    print("=" * 65)

    try:
        token = await get_token()
    except Exception as e:
        print(f"Cannot reach backend at {BACKEND_HTTP}. Make sure server is running!")
        print(f"Error: {e}")
        return

    ws_url = f"{BACKEND_WS}/ws/track/{DEVICE_ID}?token={token}"
    print(f"[WS] Connecting to {ws_url}...")

    async with websockets.connect(ws_url) as ws:
        print("[WS] Connected successfully!")

        # 1. NTP 3-Way Handshake
        t_send = time.time()
        await ws.send(json.dumps({"type": "time_sync", "t_client_send": t_send}))
        resp = json.loads(await ws.recv())
        t_recv = time.time()
        print(f"[TIME_SYNC] NTP Handshake: Round-Trip = {round((t_recv - t_send)*1000, 2)}ms")

        seq = 1

        # 2. Stream 10Hz Live Real-Time Data (Priority 0)
        print("\n--- PHASE 1: Real-Time 10Hz Streaming (100ms packets) ---")
        for _ in range(10):
            packet = generate_10hz_packet(seq, time.time())
            packet["type"] = "live"
            await ws.send(json.dumps(packet))
            seq += 1
            await asyncio.sleep(0.10)  # Exact 10Hz rate

            try:
                server_msg = await asyncio.wait_for(ws.recv(), timeout=0.15)
                parsed = json.loads(server_msg)
                print(f"  [10Hz FEEDBACK] Seq {parsed['seq']} -> Pos: ({parsed['x']}, {parsed['y']}, {parsed['z']}) | Verified: {parsed.get('is_verified')}")
            except asyncio.TimeoutError:
                pass

        # 3. Simulate Network Outage
        print("\n--- PHASE 2: Simulating 2-Second Network Drop ---")
        await ws.close()

    print("  * Disconnected: Accumulating 10Hz offline backlog (20 packets = 2 sec)...")
    offline_buffer = []
    for _ in range(20):
        offline_buffer.append(generate_10hz_packet(seq, time.time()))
        seq += 1
        await asyncio.sleep(0.10)

    # 4. Reconnect with Jitter
    print("\n--- PHASE 3: Reconnecting with Jitter ---")
    await asyncio.sleep(random.uniform(0.1, 0.3))

    async with websockets.connect(ws_url) as ws:
        print("  * Reconnected successfully!")

        # 5. Flush Backlog Chunk (Priority 1)
        print(f"\n--- PHASE 4: Ingesting Backlog Chunk ({len(offline_buffer)} items) ---")
        backlog_msg = {
            "type": "backlog",
            "chunk": offline_buffer
        }
        await ws.send(json.dumps(backlog_msg))

        for _ in range(10):
            try:
                server_msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                parsed = json.loads(server_msg)
                print(f"  [BACKLOG RECONCILED] Seq {parsed['seq']} -> Pos: ({parsed['x']}, {parsed['y']}, {parsed['z']})")
            except asyncio.TimeoutError:
                break

    # 6. Check Health & Metrics
    print("\n--- PHASE 5: Checking Backend Health ---")
    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{BACKEND_HTTP}/health")).json()
        print(f"Health Status: {json.dumps(health, indent=2)}")

    print("\n" + "=" * 65)
    print(" 10Hz Pure Dead Reckoning Backend Verification Complete!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_simulation())

