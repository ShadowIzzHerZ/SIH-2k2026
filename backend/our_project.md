# SIH 2026: 10Hz Real-Time Dead Reckoning Backend System

> **Repository:** Sih-Backend  
> **Target Rate:** 10Hz Real-Time Sensor Processing (100ms Ingestion Window)  
> **Core Focus:** Pure, High-Performance Asynchronous Backend Engine for Dead Reckoning (Stationary ZUPT, Pedestrian Kinematics, Adaptive Filtering, Priority Queuing, and Persistent State Synchronization).

---

## 1. Executive Summary & Overview

This system is an enterprise-grade, high-throughput **10Hz Dead Reckoning Backend Engine** architected for the **Smart India Hackathon (SIH 2026)**. 

The backend receives continuous high-frequency 9-DOF Inertial Measurement Unit (IMU) telemetry—comprising 3-axis accelerometer, 3-axis gyroscope, and 3-axis magnetometer readings—and fuses them using a **9-State Extended Kalman Filter (EKF)**. It solves core real-world dead reckoning challenges:
- **Sensor Drift & Micro-Vibration Elimination** via Zero Velocity Update (ZUPT).
- **Pedestrian Step & Heading Kinematics Integration** for realistic tracking without GPS.
- **Tri-Modal State Awareness** (`Stationary`, `Moving`, and `Adaptive`).
- **Persistent Coordinate & Movement State Logging** across Redis cache and TimescaleDB/SQLite databases.
- **Network Resilience & Backlog Prioritization** ensuring real-time streams are never blocked by retransmitted historical packets.

---

## 2. Tri-Modal Operational Architecture

To resolve the challenge where stationary hardware accrued velocity drift, and active walking was unresponsive, the engine provides **3 distinct operational modes**:

```
                                  +-------------------------------+
                                  |   Incoming 10Hz IMU Packet    |
                                  |  [ax, ay, az, gx, gy, gz, mag]|
                                  +---------------+---------------+
                                                  |
                                       Mode Selection Check
                                                  |
             +------------------------------------+------------------------------------+
             |                                    |                                    |
             v                                    v                                    v
    +-----------------+                  +-----------------+                  +-----------------+
    |     MODE 1      |                  |     MODE 2      |                  |     MODE 3      |
    |   STATIONARY    |                  | ACTIVE MOVING   |                  |ADAPTIVE STORAGE |
    +--------+--------+                  +--------+--------+                  +--------+--------+
             |                                    |                                    |
    * Velocity clamped to 0.0            * Step Peak Detection                * Dynamic Kinematic Check:
    * ZUPT Lock Applied                  * Compass Heading (Yaw)                - Var(a) > 0.15
    * Covariance clamped                 * Cadence Velocity Projection          - ||a|| - 9.81 > 0.45
    * Movement State: REST               * Movement State: MOVING               - ||g|| > 0.25
    * Position Frozen (0 drift)          * Step Count Incremented             * Auto-switches REST/MOVING
             |                                    |                           * Continuously logs state
             +------------------------------------+------------------------------------+
                                                  |
                                                  v
                                 +---------------------------------+
                                 |   State Export & Persistence    |
                                 |  - Redis In-Memory Cache (1 hr) |
                                 |  - Database (PostgreSQL/SQLite) |
                                 |  - WebSocket 10Hz Broadcast     |
                                 +---------------------------------+
```

### Mode 1: Stationary Mode (`stationary`)
- **Problem Addressed:** Accelerometer sensors continuously register gravitational noise and high-frequency thermal vibrations. Standard double-integration causes position to drift uncontrollably even when the device is resting on a desk.
- **Engine Solution:** Hard-locks velocity vectors `[vx, vy, vz] = 0.00 m/s`.
- **State Behavior:**
  - `movement_state`: `"REST"`
  - `step_count`: Constant (no false steps registered)
  - `x, y, z`: Frozen at the exact resting position with 0.00 drift.

### Mode 2: Active Moving Mode (`moving`)
- **Problem Addressed:** Laptops lack built-in accelerometers, and consumer mobile IMU double integration diverges within seconds.
- **Engine Solution:** Employs **Pedestrian Dead Reckoning (PDR)** kinematics:
  - Synthesizes dynamic gait acceleration swings.
  - Detects foot strike peaks ($a_{	ext{mag}} - g > 0.35$ with refractory cadence window $\Delta t > 0.35	ext{s}$).
  - Projects pedestrian stride velocity ($1.15 	ext{ m/s}$ along estimated tilt-compensated magnetometer heading $	heta_{	ext{yaw}}$):
    $$v_x = v_{	ext{walk}} \cdot \cos(	heta_{	ext{yaw}}), \quad v_y = v_{	ext{walk}} \cdot \sin(	heta_{	ext{yaw}})$$
- **State Behavior:**
  - `movement_state`: `"MOVING"`
  - `step_count`: Increments with each foot strike.
  - Position updates accurately forward along compass heading.

### Mode 3: Adaptive State Storage Mode (`adaptive`)
- **Problem Addressed:** Fully autonomous operation where the system must identify physical walking vs. resting transitions on the fly and persist the exact coordinate state.
- **Engine Solution:**
  - Computes rolling acceleration variance over a 1-second moving window (10 samples at 10Hz).
  - Dynamic threshold check:
    $$	ext{is\_moving} = \left(\sigma_a^2 > 0.15ight) \lor \left(\left|\|a\| - 9.81ight| > 0.45ight) \lor \left(\|\omega\| > 0.25ight)$$
  - When stationary, instantly applies ZUPT velocity clamp.
  - When moving, initiates step progression and heading integration.
  - **Continuously persists latest position, movement state (`REST` or `MOVING`), velocity, and step count directly into database and Redis.**

---

## 3. Database Schema & State Storage

The database layer utilizes **SQLAlchemy AsyncIO** supporting both high-scale **PostgreSQL / TimescaleDB hypertables** in production and lightweight **SQLite** for zero-dependency local execution.

### Table: `device_trajectories`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` (PK) | Auto-incrementing primary key |
| `time` | `DATETIME` | UTC timestamp of sample (indexed for time-series queries) |
| `device_id` | `VARCHAR(64)` | Unique client/device hardware identifier |
| `session_id` | `VARCHAR(64)` | Active operating session ID |
| `seq_num` | `INTEGER` | Packet sequence number for order verification |
| `x` | `FLOAT` | Position Easting coordinate in meters |
| `y` | `FLOAT` | Position Northing coordinate in meters |
| `z` | `FLOAT` | Vertical altitude in meters |
| `vx`, `vy`, `vz` | `FLOAT` | Real-time velocities in m/s |
| `roll`, `pitch`, `yaw` | `FLOAT` | Orientation angles in radians |
| `movement_state` | `VARCHAR(32)` | Dynamic state: `'REST'` or `'MOVING'` |
| `step_count` | `INTEGER` | Accumulated verified pedestrian steps |
| `is_backlog` | `BOOLEAN` | `False` for 10Hz live data, `True` for reconnected backlogs |
| `is_verified` | `BOOLEAN` | Set to `True` once processed through Kalman filter |
| `covariance_json` | `TEXT` | Serialized 9x9 EKF error covariance matrix |
| `raw_sensor_json` | `TEXT` | Optional raw IMU packet readings |

### Automatic Safe Schema Migration
The initialization routine (`init_db()` in `app/db/database.py`) automatically checks and adds `movement_state` and `step_count` columns if running against an existing database, ensuring backward compatibility without data loss.

---

## 4. API Endpoints Reference

### Authentication & Health
- **`POST /auth/token`**
  - Mint high-security JWT bearer tokens required for authenticating WebSocket connections.
  - Request: `{"device_id": "phone_user_01", "expires_minutes": 1440}`
- **`GET /health`**
  - Returns health status, Redis connectivity, active WebSocket count, and ingestion queue depth.

### Telemetry & Ingestion
- **`WS /ws/track/{device_id}?token=<jwt>`**
  - High-frequency 10Hz bi-directional WebSocket connection.
  - Supports NTP 3-way time synchronization (`time_sync`).
  - Accepts live sensor frames (`{"type": "live", "mode": "stationary"|"moving"|"adaptive", "ax": ..., "ay": ..., ...}`).
  - Pushes verified 10Hz coordinates and state updates back to client.
- **`POST /sensor-data`**
  - HTTP batch upload fallback when WebSockets are restricted by firewall/proxy.

### Device State & Modes
- **`GET /device/{device_id}/state`**
  - Retrieves the latest live position, velocity, heading, movement state (`REST`/`MOVING`), and step count.
  - Automatically queries Redis cache first; falls back to database if cache has expired.
- **`POST /device/{device_id}/mode`**
  - Programmatically forces device operating mode.
  - Request: `{"mode": "stationary" | "moving" | "adaptive"}`
- **`GET /device/{device_id}/trajectory?limit=500`**
  - Returns chronological trajectory points and verified state history for mapping.

---

## 5. Web Visualizer & Live Controller

The backend includes a built-in dashboard hosted directly at the root URL:
- **URL:** `http://localhost:8000` or `http://localhost:8000/dashboard`

### Features:
1. **Live 2D Canvas Visualizer:** Real-time trajectory plotting (East vs North in meters) with grid scaling.
2. **Kinematic Telemetry HUD:** Displays Position $(x, y)$, Speed $(v)$, Movement State (`REST` / `MOVING` badge), Step Count, and Heading Compass (yaw).
3. **Interactive 3-Mode Controller:**
   - **Mode 1 (Stationary):** Freezes velocity at $0.00 	ext{ m/s}$ with zero drift.
   - **Mode 2 (Moving):** Actively advances pedestrian steps along heading.
   - **Mode 3 (Adaptive):** Cycles through walking and rest periods, storing all points in the database.
4. **"Load Last DB State" Button:** Verifies that persistent database commits are succeeding by reading the most recent record from SQLite/TimescaleDB.
5. **Multi-Device Support:** Can be opened on a laptop or smartphone connected to the same Wi-Fi via `http://<laptop_ip>:8000/dashboard`.

---

## 6. How to Run & Verify

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Dashboard on Laptop
- In any browser, navigate to:  
  **`http://localhost:8000`** or **`http://localhost:8000/dashboard`**
- Interactive API documentation is available at:  
  **`http://localhost:8000/docs`**

### 4. Verify Latest State via API
```bash
curl http://localhost:8000/device/web_dashboard_client/state
```

### 5. Run Automated Integration Test
```bash
python tests/simulate_device.py
```
This test script verifies JWT minting, NTP 3-way time sync, 10Hz live ingestion, connection drop, backlog buffering, and reconnection prioritization.

---

## 7. SIH 2026 Core Architecture Highlights

1. **Zero External Frontend Dependency:** A complete HTML5/Canvas visualization interface is packaged directly in `app/main.py` without requiring Node.js, npm, or frontend build tools.
2. **Elimination of Extraneous ML/Heavy Dependencies:** Lean, ultra-fast Python execution optimized for sub-10ms latency per sample.
3. **Robust Prioritization Queue:** High-priority channel for live 10Hz telemetry ensures real-time tracking is never delayed by batch backlog sync.
4. **Dual Database Support:** Instant local development with SQLite (`sih_trajectories.db`) and zero-configuration production deployment with TimescaleDB hypertables.
