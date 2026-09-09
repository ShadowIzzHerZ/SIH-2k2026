# SIH 2026: Production Deployment & Cloud Setup Guide

> **10Hz Real-Time Dead Reckoning Backend Engine**  
> High-concurrency FastAPI ingestion, 9-State Extended Kalman Filter (EKF), Zero Velocity Updates (ZUPT), Redis State Caching, TimescaleDB Persistence, and Starvation-Free Priority Queuing.

---

## 1. Quick Architecture Map

```
                     +---------------------------------------+
                     |    Mobile App / Hardware IMU / Web    |
                     +-------------------+-------------------+
                                         |
                       WSS / HTTPS (Port 443 / 80)
                                         v
                     +---------------------------------------+
                     |          Nginx Reverse Proxy          |
                     |  - SSL Termination (Let's Encrypt)    |
                     |  - WebSocket Upgrades (No Buffering)  |
                     +-------------------+-------------------+
                                         |
                        Internal Proxy (Port 8000)
                                         v
                     +---------------------------------------+
                     |        FastAPI Backend Container      |
                     |  - 10Hz Async Priority Ingestion      |
                     |  - 9-DOF Extended Kalman Filter (EKF) |
                     |  - 3-Mode Controller (ZUPT / Cadence) |
                     +---------+-------------------+---------+
                               |                   |
               Internal Redis  |                   | AsyncPG Connection
               (Port 6379)     v                   v (Port 5432)
                     +-------------------+   +--------------------+
                     |    Redis Cache    |   | TimescaleDB PG 16  |
                     |  - 1-Hr Device TTL|   | - Hypertables      |
                     |  - Calibration Mem|   | - Permanent Logs   |
                     +-------------------+   +--------------------+
```

---

## 2. Option A: Local / Server Multi-Container Stack (Docker Compose)

The fastest and most reliable deployment method for production Linux servers or local development.

### Prerequisites:
- Docker Engine 24+ and Docker Compose v2.

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure database and redis settings point to the container services:
```env
HOST=0.0.0.0
PORT=8000
DEBUG=False
SECRET_KEY=your_generated_jwt_secret_key_here
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql+asyncpg://postgres:postgres@timescaledb:5432/sih_db
IMU_SAMPLING_RATE_HZ=10.0
BACKLOG_CHUNK_SIZE=20
MAX_QUEUE_SIZE=50000
```

### Step 2: Launch the Stack
```bash
docker compose up -d --build
```

### Step 3: Verify Running Services
```bash
docker compose ps
```
Both `redis` and `timescaledb` have built-in healthchecks; `backend` starts only after both are fully healthy.

### Step 4: View Logs
```bash
docker compose logs -f backend
```

---

## 3. Option B: Automated Cloud VPS Deployment (`deploy.sh`)

For **AWS EC2, DigitalOcean Droplets, Linode, Hetzner, or any Ubuntu/Debian VPS**.

### Step 1: SSH into your VPS
```bash
ssh ubuntu@<YOUR_VPS_IP>
```

### Step 2: Clone the Repository
```bash
git clone <YOUR_REPO_URL> sih-backend
cd sih-backend
```

### Step 3: Execute the Automated Deploy Script
```bash
chmod +x deploy.sh
./deploy.sh
```
The script will automatically:
1. Update system packages non-interactively.
2. Install Docker & Docker Compose if missing.
3. Configure UFW firewall rules for ports `80`, `443`, and `8000`.
4. Generate a unique cryptographic JWT secret and create `.env`.
5. Build and launch `FastAPI`, `Redis`, and `TimescaleDB` containers.
6. Check health status and print public URLs.

---

## 4. Option C: Nginx Reverse Proxy with Free SSL (Let's Encrypt)

To serve the backend securely over `https://` and `wss://`:

### Step 1: Install Nginx & Certbot
```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

### Step 2: Deploy Nginx Configuration
```bash
sudo cp nginx.conf.example /etc/nginx/sites-available/sih-backend
sudo sed -i 's/your-domain.com/YOUR_ACTUAL_DOMAIN.com/g' /etc/nginx/sites-available/sih-backend
sudo ln -sf /etc/nginx/sites-available/sih-backend /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Step 3: Obtain SSL Certificate
```bash
sudo certbot --nginx -d YOUR_ACTUAL_DOMAIN.com
```
*Note: The Nginx configuration in `nginx.conf.example` explicitly sets `proxy_buffering off;` which is required to prevent 100ms packet buffering delays during 10Hz WebSocket streaming.*

---

## 5. Option D: 1-Click Cloud Hosting (Render / PaaS)

This repository includes a native [`render.yaml`](file:///C:/Users/manis/OneDrive/Desktop/Sih-Backend/render.yaml) blueprint.

### Deploying to Render:
1. Push this repository to GitHub or GitLab.
2. Open the [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Blueprint**.
4. Connect your repository.
5. Render will automatically provision:
   - Web Service (`uvicorn app.main:app --host 0.0.0.0 --port $PORT --ws websockets`)
   - Managed PostgreSQL database for TimescaleDB / trajectory persistence.
   - High-throughput WebSocket support with zero config.

---

## 6. Verification Suite

Run the automated deployment verifier to test any local or cloud URL:

### Local Test:
```bash
python tests/verify_deployment.py --url http://127.0.0.1:8000
```

### Cloud Production Test:
```bash
python tests/verify_deployment.py --url https://YOUR_DOMAIN.com
```

### What this test verifies:
- `GET /health`: API health status, Redis availability, queue worker activity.
- `POST /auth/token`: Cryptographic JWT minting.
- `WS /ws/track/{device_id}`: Bi-directional WebSocket handshake, NTP 3-way time sync, and real-time 10Hz EKF streaming with position feedback.
- `GET /device/{device_id}/state`: Redis caching and database state persistence.
- `GET /device/{device_id}/trajectory`: Historical trajectory retrieval from TimescaleDB/SQLite.

---

## 7. Connecting Mobile / Flutter / Hardware Clients to Cloud

Update the connection endpoints in your mobile client or hardware sender:

```dart
// Production URLs
const String serverHost = "YOUR_DOMAIN.com"; // or VPS IP
final String wsUrl = "wss://$serverHost/ws/track/$deviceId?token=$jwtToken";
final String authUrl = "https://$serverHost/auth/token";
```

### Recommended Client Rates:
- **IMU Rate:** 10Hz ($\Delta t = 100\text{ms}$)
- **Backlog Chunk Size:** Max 20–50 packets per chunk during network recovery.
