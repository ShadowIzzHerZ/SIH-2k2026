# Dead Reckoning Backend Engine (10Hz)
**High-Concurrency FastAPI Ingestion, Priority Queues, Redis Caching & TimescaleDB Persistence**  
*Latest Sync: September 8, 2026 | 10Hz Pure Dead Reckoning Service*

---

## 1. Overview & Scope

This repository contains the **pure backend service** for the Smart India Hackathon (SIH) Dead Reckoning navigation system. It focuses exclusively on:
- 10Hz IMU sensor fusion using a 9-DOF Extended Kalman Filter (`filterpy` / NumPy).
- Starvation-free priority queue processing (Live `P0` vs Backlog `P1`).
- WebSocket streaming and NTP-based clock drift synchronization.
- Redis multi-tier state caching (1-hour TTL) with TimescaleDB/PostgreSQL permanent storage fallback.
- Cryptographic JWT handshake security.

---

## 2. Technical Specifications (10Hz Standard)

- **Sampling Frequency:** **10 Hz** (100ms packet interval, $\Delta t = 0.1\text{s}$).
- **Sensor Fusion:** 9-DOF IMU Extended Kalman Filter (Accelerometer, Gyroscope, Magnetometer with Zero Velocity Updates).
- **Transport Protocols:** Full-duplex WebSocket (`/ws/track/{device_id}`) with REST fallback (`/sensor-data`).
- **Security:** HMAC-SHA256 JWT handshake authentication.
- **Clock Drift Compensation:** 3-way NTP handshake calculating $\Delta t_{\text{offset}}$ and RTT.
- **Starvation-Free Queue:** `asyncio.PriorityQueue` prioritizing Live 10Hz frames (`P0`) over Backlog recovery dumps (`P1`).
- **State Caching:** Redis 7+ for live positions and calibration constants with 1-hour TTL.
- **Persistence:** PostgreSQL 16 + TimescaleDB hypertables for trajectory history.

---

## 3. Integration Contract for App Dev Team (Mobile Client)

### 1. WebSocket Connection
```text
ws://<server_host>:8000/ws/track/{device_id}?token=<jwt_token>
```
*Token can be obtained via `POST /auth/token` with body `{"device_id": "...", "expires_minutes": 60}`.*

Prefer sending the token in a header rather than the URL (URLs end up in proxy access logs): `Authorization: Bearer <jwt_token>`, or, from a browser, the subprotocol pair `new WebSocket(url, ["bearer", token])`. The `?token=` query parameter still works for existing clients.

### 2. Handshake & NTP Clock Sync
Immediately after connecting, the client sends:
```json
{
  "type": "time_sync",
  "t_client_send": 1725780000.123
}
```
The backend immediately replies with `t_server_recv` and `t_server_send` to compute client clock skew.

### 3. Real-Time Live Streaming (10Hz, Priority 0)
Send continuous 100ms interval frames:
```json
{
  "type": "live",
  "seq": 1,
  "timestamp": 1725780000.223,
  "ax": 0.05, "ay": 0.98, "az": 9.81,
  "gx": 0.01, "gy": 0.02, "gz": -0.01,
  "mx": 22.5, "my": -5.1, "mz": 41.2
}
```

### 4. Offline Backlog Ingestion (Priority 1)
When reconnecting after an outage, send buffered data in manageable chunks (max 20–50 packets per chunk):
```json
{
  "type": "backlog",
  "chunk": [
    {"seq": 25, "timestamp": 1725780002.500, "ax": 0.04, "ay": 0.95, "az": 9.80, "gx": 0.0, "gy": 0.0, "gz": 0.0, "mx": 22.0, "my": -5.0, "mz": 41.0},
    {"seq": 26, "timestamp": 1725780002.600, "ax": 0.06, "ay": 0.96, "az": 9.82, "gx": 0.0, "gy": 0.0, "gz": 0.0, "mx": 22.1, "my": -5.1, "mz": 41.1}
  ]
}
```
 
### 5. Position Feedback Stream (Server $\to$ Client)
The backend pushes back fused Kalman coordinates:
```json
{
  "type": "position_update",
  "device_id": "phone_user_sih_01",
  "seq": 26,
  "timestamp": 1725780002.600,
  "x": 4.521,
  "y": 12.834,
  "z": 0.120,
  "vx": 1.25,
  "vy": 0.11,
  "vz": 0.01,
  "roll": 0.012,
  "pitch": -0.034,
  "yaw": 0.354,
  "movement_state": "MOVING",
  "step_count": 14,
  "is_backlog": false,
  "is_verified": true
}
```

---

## 4. 3 Operational Modes & Zero Velocity Updates (ZUPT)

The backend provides three operational modes to eliminate drift and preserve coordinates:

1. **Mode 1: Stationary (Rest / ZUPT):**
   - Hard Zero-Velocity Update (`ZUPT`).
   - Velocity vector is forced to strictly `0.00 m/s`.
   - Cartesian coordinates $(x, y)$ are completely frozen, guaranteeing **0.00m phantom drift** when the device is stationary.
   - `movement_state` is set to `REST`.

2. **Mode 2: Moving (Active Pedestrian Dead Reckoning):**
   - Human step cadence detection (~1.8 Hz) integrated along heading ($\psi$).
   - Computes displacement along heading angle with attitude filtering (Roll, Pitch, Yaw).
   - Increments persistent step count and reports `movement_state` as `MOVING`.

3. **Mode 3: Adaptive Storage:**
   - Evaluates IMU acceleration variance dynamically: below threshold ($0.15 \text{ m/s}^2$) it auto-locks into `REST`, above threshold it activates `MOVING`.
   - Persists every update into SQLite / TimescaleDB and Redis cache.
   - Queryable via `GET /device/{device_id}/trajectory` or verified directly on the dashboard.

---

## 5. Key Architectural Mechanisms

1. **Starvation-Free Backlog Ingestion:** Live 10Hz packets enter the priority queue at `Priority 0`, while backlog chunks enter at `Priority 1` and cooperatively yield to the event loop (`await asyncio.sleep(0)`). Live streams maintain $<50\text{ms}$ latency even during large backlog dumps.
2. **Multi-Tier State Recovery:** If a device stays offline longer than the 1-hour Redis TTL, the backend queries TimescaleDB/PostgreSQL for the last known position and covariance matrix to restore the 9-DOF Kalman filter seamlessly.
3. **Sequence Reordering & Clock Sync:** Protects against out-of-order packet delivery using strict sequence numbering, while NTP calculations prevent timestamp corruption from phone clock skew.

---

## 6. Running the Backend

### Windows (1-Click Run):
Double-click `run_windows.bat` or run:
```cmd
run_windows.bat
```

### Local Python Run (Cross-Platform):
```powershell
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*(Automatically falls back to SQLite and in-memory LRU cache if Redis/Postgres are not installed).*

### Run Test Suite:
```powershell
python -m unittest tests/test_backend.py
```

### Production VPS / Cloud VM Deployment (AWS, DigitalOcean, Azure):
On your remote Ubuntu/Debian server, run:
```bash
git clone https://github.com/manishpathak2407-bot/Sih-Backend.git
cd Sih-Backend
chmod +x deploy.sh
./deploy.sh
```
*This automatically configures Docker, launches the FastAPI backend, Redis cache, and TimescaleDB containers, and verifies the health endpoint.*

### Live Simulation Integration Test:
```powershell
python tests/simulate_device.py
```


---

## 7. Authentication Settings

Token checks are **off by default** so the dashboard and local demos keep working. Turn them on by setting these environment variables (in `.env` or the service environment):

| Variable | Default | Effect when set |
| :--- | :--- | :--- |
| `DEVICE_ENROLL_SECRET` | empty | `POST /auth/token` needs `"secret": "<DEVICE_ENROLL_SECRET>"` in the body to mint a device token. `POST /sensor-data`, `GET /device/{id}/state`, `GET /device/{id}/trajectory` and `POST /device/{id}/mode` then need `Authorization: Bearer <token>` for that device (or an admin token). |
| `ADMIN_SECRET` | empty | The only way to mint an `admin` token: `{"device_id": "admin", "secret": "<ADMIN_SECRET>"}`. It also works as the secret for device tokens. While `DEVICE_ENROLL_SECRET` is set and `ADMIN_SECRET` is not, admin tokens cannot be minted at all. |

With `DEVICE_ENROLL_SECRET` unset, any caller can mint a token for any device id and the REST endpoints need no token; the server logs a warning the first time a token is minted that way. `/health` and the dashboard page are always open. The dashboard asks for the enrollment secret when the server requires one.

Generate secrets with `openssl rand -hex 32`. `SECRET_KEY` (the JWT signing key) is separate and always required.
