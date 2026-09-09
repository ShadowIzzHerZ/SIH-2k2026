# SIH 2026: Dead Reckoning System - Master Links Directory

> **Repository:** [manishpathak2407-bot/Sih-Backend](https://github.com/manishpathak2407-bot/Sih-Backend)  
> **Service:** 10Hz Real-Time Dead Reckoning Backend & Kinematic Visualizer  
> **Last Synchronized:** Wednesday, September 9, 2026

---

## 1. Primary Testing & Visualizer Links

| Target | Description | URL Link |
| :--- | :--- | :--- |
| **Direct Browser Testing Page** | Main interactive 10Hz visualizer with 3-Mode Controller (Stationary, Moving, Adaptive) | **[http://localhost:8000/](http://localhost:8000/)** |
| **Alternative Localhost IP** | Fallback loopback address for the testing dashboard | **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** |
| **Dashboard Explicit Route** | Direct route alias to the HTML5 canvas visualizer | **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)** |
| **Local Network / Mobile Stream** | Open on mobile or collaborator laptop on same network (replace `<HOST_IP>` via `ipconfig`) | `http://<HOST_IP>:8000/` |

---

## 2. API & Documentation Links

| Resource | Purpose | URL Link |
| :--- | :--- | :--- |
| **Swagger Interactive Docs** | Live OpenAPI UI to test endpoints, authenticate JWT, and execute requests | **[http://localhost:8000/docs](http://localhost:8000/docs)** |
| **ReDoc API Reference** | Clean, formatted API schema specifications | **[http://localhost:8000/redoc](http://localhost:8000/redoc)** |
| **Health & Queue Status** | Real-time JSON health, Redis cache connectivity, and ingestion queue depth | **[http://localhost:8000/health](http://localhost:8000/health)** |

---

## 3. Data & State Inspection Endpoints

| Endpoint | Method | Purpose | Example Link |
| :--- | :---: | :--- | :--- |
| **Latest Kinematic State** | `GET` | Fetches live coordinates $(x,y,z)$, speed, movement state (`REST`/`MOVING`), and step count | **[http://localhost:8000/device/web_dashboard_client/state](http://localhost:8000/device/web_dashboard_client/state)** |
| **Trajectory History** | `GET` | Returns persistent chronological trajectory points from TimescaleDB/SQLite | **[http://localhost:8000/device/web_dashboard_client/trajectory?limit=50](http://localhost:8000/device/web_dashboard_client/trajectory?limit=50)** |
| **Mint Auth JWT Token** | `POST` | Mints cryptographically signed bearer tokens for WebSocket connection handshake | `http://localhost:8000/auth/token` |
| **Batch Sensor Fallback** | `POST` | REST fallback ingestion when WebSockets are restricted by firewall/proxy | `http://localhost:8000/sensor-data` |
| **Set Operational Mode** | `POST` | Programmatically toggles mode (`stationary`, `moving`, `adaptive`) | `http://localhost:8000/device/{device_id}/mode` |

---

## 4. WebSocket Protocols

| Protocol | Path | Usage |
| :--- | :--- | :--- |
| **Local WebSocket** | `ws://localhost:8000/ws/track/{device_id}?token=<jwt>` | 10Hz live sensor streaming & coordinate feedback |
| **Local Network / Wi-Fi WebSocket** | `ws://<HOST_IP>:8000/ws/track/{device_id}?token=<jwt>` | Connect smartphone apps / Flutter to your host machine |
| **Cloud Production WSS** | `wss://<YOUR_DOMAIN>/ws/track/{device_id}?token=<jwt>` | Encrypted production WebSocket over SSL / Let's Encrypt |

---

## 5. Cloud Deployment & GitHub Links

| Destination | Purpose | URL Link |
| :--- | :--- | :--- |
| **GitHub Repository** | Source code, models, documentation, and configuration | **[https://github.com/manishpathak2407-bot/Sih-Backend](https://github.com/manishpathak2407-bot/Sih-Backend)** |
| **GitHub Actions / CI-CD** | Automated test verification and Docker build workflows | **[https://github.com/manishpathak2407-bot/Sih-Backend/actions](https://github.com/manishpathak2407-bot/Sih-Backend/actions)** |
| **1-Click Render Deploy** | Deploy stack to Render cloud via Blueprint | **[Deploy to Render](https://render.com/deploy?repo=https://github.com/manishpathak2407-bot/Sih-Backend)** |

---

## 6. Testing Instructions

### To Test Modes in the Browser:
1. Open **[http://localhost:8000/](http://localhost:8000/)**.
2. Click **`🛑 Mode 1: Stationary Mode (Rest)`** -> Observe velocity hard-lock to $0.00\text{ m/s}$, zero position drift, state = `REST`.
3. Click **`🚶 Mode 2: Active Moving Mode (Walking)`** -> Observe step counter incrementing, forward 2D trajectory plotting, state = `MOVING`.
4. Click **`💾 Mode 3: Adaptive State Storage Mode`** -> Observe automatic alternation between rest and walking, with permanent database recording.
5. Click **`📂 Load Last DB State`** -> Confirms persistent database commits.
