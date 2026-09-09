# 10Hz Dead Reckoning Backend Engine - SIH 2026
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.cache.redis_client import cache_manager
from app.db.database import init_db
from app.api.websocket import router as ws_router, queue_worker
from app.api.rest_routes import router as rest_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sih_backend")
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SIH Dead Reckoning Backend...")
    await cache_manager.connect()
    await init_db()
    
    worker_task = asyncio.create_task(queue_worker())
    logger.info("System ready. Ingestion queue worker active.")

    yield

    logger.info("Shutting down backend...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await cache_manager.disconnect()
    logger.info("Shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(rest_router)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH Dead Reckoning 10Hz 3-Mode Controller</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-bright: #ffffff;
            --accent: #58a6ff;
            --green: #2ea043;
            --orange: #f0883e;
            --red: #da3633;
            --purple: #8957e5;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
        h1 { font-size: 1.4rem; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }
        .badge { background: #238636; color: white; font-size: 0.75rem; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .badge-offline { background: var(--orange); }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
        .card-title { font-size: 0.95rem; font-weight: 600; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        canvas { width: 100%; height: 400px; background: #010409; border: 1px solid var(--border); border-radius: 6px; display: block; }
        .telemetry { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
        .stat-box { background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid var(--border); }
        .stat-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
        .stat-val { font-size: 1.25rem; font-weight: bold; color: var(--text-bright); font-family: monospace; }
        .btn-group { display: flex; flex-direction: column; gap: 12px; }
        .mode-btn { background: #21262d; border: 2px solid var(--border); color: var(--text-bright); padding: 12px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s; font-size: 0.88rem; text-align: left; }
        .mode-btn:hover { background: #30363d; }
        .mode-btn.active-stationary { border-color: var(--orange); background: rgba(240, 136, 62, 0.15); }
        .mode-btn.active-moving { border-color: var(--green); background: rgba(46, 160, 67, 0.15); }
        .mode-btn.active-adaptive { border-color: var(--purple); background: rgba(137, 87, 229, 0.15); }
        .action-btn { background: #21262d; border: 1px solid var(--border); color: var(--text-bright); padding: 10px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }
        .action-btn:hover { background: #30363d; }
        .logs { height: 160px; overflow-y: auto; background: #010409; border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-family: monospace; font-size: 0.75rem; line-height: 1.4; color: #8b949e; margin-top: 14px; }
        .log-entry { margin-bottom: 4px; }
        .log-ok { color: #56d364; }
        .log-warn { color: var(--orange); }
        .log-info { color: var(--accent); }
        .state-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
        .state-rest { background: #6e40c9; color: white; }
        .state-moving { background: #238636; color: white; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Dead Reckoning 10Hz Monitor <span id="conn-badge" class="badge badge-offline">DISCONNECTED</span></h1>
            <p style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">Stationary Locking (ZUPT) • Active Walking Kinematics • Persistent Database State</p>
        </div>
        <div>
            <a href="/docs" target="_blank" style="color:var(--accent); text-decoration:none; margin-right:15px; font-size:0.85rem;">API Docs</a>
            <a href="/health" target="_blank" style="color:var(--accent); text-decoration:none; font-size:0.85rem;">Health JSON</a>
        </div>
    </header>

    <div class="grid">
        <div>
            <div class="card">
                <div class="card-title" style="display:flex; justify-content:space-between;">
                    <span>2D Trajectory Canvas (East vs North Meters)</span>
                    <span id="coord-hud" style="color:var(--accent); font-family:monospace; font-size:0.85rem;">(0.00m, 0.00m)</span>
                </div>
                <canvas id="mapCanvas" width="700" height="400"></canvas>
                <div class="logs" id="logBox">
                    <div class="log-entry log-info">[SYSTEM] Ready. Select one of the 3 modes below to test.</div>
                </div>
            </div>

            <div class="card" style="background:#0d1117; border-color:#388bfd;">
                <div class="card-title" style="color:#58a6ff; font-size:0.85rem;">💡 Note on Physical Laptop Motion vs Smartphone Sensors</div>
                <p style="font-size:0.8rem; color:#8b949e; line-height:1.4;">
                    Most desktop/laptop computers <b>do not have hardware accelerometers</b>. Therefore, holding a laptop will not provide real IMU forces. For live physical walking, you can open this dashboard on your smartphone connected to the same Wi-Fi: 
                    <span style="color:#58a6ff; font-family:monospace;">http://&lt;your_laptop_ip&gt;:8000/dashboard</span>.
                </p>
            </div>
        </div>

        <div>
            <div class="card">
                <div class="card-title">Live Kinematic Telemetry</div>
                <div class="telemetry">
                    <div class="stat-box">
                        <div class="stat-label">Position X (East)</div>
                        <div class="stat-val" id="stat-x">0.000 m</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Position Y (North)</div>
                        <div class="stat-val" id="stat-y">0.000 m</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Velocity</div>
                        <div class="stat-val" id="stat-v">0.00 m/s</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Movement State</div>
                        <div class="stat-val" id="stat-state"><span class="state-tag state-rest">REST</span></div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Step Count</div>
                        <div class="stat-val" id="stat-steps">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Heading (Yaw)</div>
                        <div class="stat-val" id="stat-yaw">0.0°</div>
                    </div>
                </div>

                <div class="card-title">Select Operational Mode (3 Modes)</div>
                <div class="btn-group">
                    <!-- MODE 1: STATIONARY -->
                    <button id="btn-mode-1" class="mode-btn" onclick="setMode('stationary')">
                        <div style="font-size:0.95rem; color:#f0883e;">🛑 Mode 1: Stationary Mode (Rest)</div>
                        <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">Hard-locks velocity to 0.00 m/s. Position is frozen at current spot with zero drift.</div>
                    </button>

                    <!-- MODE 2: MOVING -->
                    <button id="btn-mode-2" class="mode-btn" onclick="setMode('moving')">
                        <div style="font-size:0.95rem; color:#56d364;">🚶 Mode 2: Active Moving Mode (Walking)</div>
                        <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">Simulates forward pedestrian steps. State = MOVING, steps advance along heading.</div>
                    </button>

                    <!-- MODE 3: ADAPTIVE STORAGE -->
                    <button id="btn-mode-3" class="mode-btn" onclick="setMode('adaptive')">
                        <div style="font-size:0.95rem; color:#bc8cff;">💾 Mode 3: Adaptive State Storage Mode</div>
                        <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">Auto-detects REST vs MOVING & continuously persists latest position & state in DB.</div>
                    </button>

                    <!-- LIVE PHYSICAL SENSORS (PHONE / LAPTOP) -->
                    <button id="btn-mode-phone" class="mode-btn" onclick="togglePhoneSensors()">
                        <div style="font-size:0.95rem; color:#58a6ff;">📱 Live Hardware IMU Stream (Phone / Laptop)</div>
                        <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">Reads actual physical accelerometer & gyro in real-time (walk around holding your phone).</div>
                    </button>
                </div>

                <div style="display:flex; gap:10px; margin-top:14px;">
                    <button class="action-btn" style="flex:1;" onclick="fetchLatestDbState()">📂 Load Last DB State</button>
                    <button class="action-btn" style="flex:1;" onclick="clearCanvas()">🧹 Reset View</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById("mapCanvas");
        const ctx = canvas.getContext("2d");
        const logBox = document.getElementById("logBox");
        const badge = document.getElementById("conn-badge");
        
        let ws = null;
        let currentMode = null; // 'stationary', 'moving', 'adaptive'
        let streamInterval = null;
        let seq = 1;
        let currentPos = { x: 0, y: 0 };
        let pathHistory = [{ x: 0, y: 0 }];
        const deviceId = "web_dashboard_client";

        function log(msg, cls = "") {
            const d = document.createElement("div");
            d.className = "log-entry " + cls;
            d.textContent = `[${new Date().toLocaleTimeString()}] ` + msg;
            logBox.appendChild(d);
            logBox.scrollTop = logBox.scrollHeight;
        }

        function drawCanvas() {
            ctx.fillStyle = "#010409";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = "#161b22";
            ctx.lineWidth = 1;
            const step = 40;
            for (let x = 0; x < canvas.width; x += step) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += step) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
            }

            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const scale = 25;

            ctx.strokeStyle = "#30363d";
            ctx.beginPath();
            ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
            ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
            ctx.stroke();

            if (pathHistory.length > 1) {
                ctx.strokeStyle = "#58a6ff";
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(cx + pathHistory[0].x * scale, cy - pathHistory[0].y * scale);
                for (let i = 1; i < pathHistory.length; i++) {
                    ctx.lineTo(cx + pathHistory[i].x * scale, cy - pathHistory[i].y * scale);
                }
                ctx.stroke();
            }

            const px = cx + currentPos.x * scale;
            const py = cy - currentPos.y * scale;
            ctx.fillStyle = "#2ea043";
            ctx.beginPath();
            ctx.arc(px, py, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        async function connectWebSocket() {
            if (ws && ws.readyState === WebSocket.OPEN) return;

            const res = await fetch("/auth/token", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ device_id: deviceId, expires_minutes: 60 })
            });
            const authData = await res.json();
            const token = authData.access_token;

            const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${proto}//${window.location.host}/ws/track/${deviceId}?token=${token}`;
            
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                badge.textContent = "CONNECTED (10Hz)";
                badge.className = "badge";
                log("WebSocket connected with JWT handshake.", "log-ok");
                ws.send(JSON.stringify({ type: "time_sync", t_client_send: Date.now() / 1000 }));
            };

            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === "position_update") {
                    currentPos.x = data.x;
                    currentPos.y = data.y;
                    pathHistory.push({ x: data.x, y: data.y });
                    if (pathHistory.length > 500) pathHistory.shift();

                    document.getElementById("stat-x").textContent = data.x.toFixed(3) + " m";
                    document.getElementById("stat-y").textContent = data.y.toFixed(3) + " m";
                    const speed = Math.sqrt(data.vx*data.vx + data.vy*data.vy);
                    document.getElementById("stat-v").textContent = speed.toFixed(2) + " m/s";

                    const stateEl = document.getElementById("stat-state");
                    if (data.movement_state === "MOVING") {
                        stateEl.innerHTML = '<span class="state-tag state-moving">MOVING</span>';
                    } else {
                        stateEl.innerHTML = '<span class="state-tag state-rest">REST</span>';
                    }

                    document.getElementById("stat-steps").textContent = data.step_count || 0;
                    document.getElementById("stat-yaw").textContent = (data.yaw * (180/Math.PI)).toFixed(1) + "°";
                    document.getElementById("coord-hud").textContent = `(${data.x.toFixed(2)}m, ${data.y.toFixed(2)}m)`;

                    drawCanvas();
                }
            };

            ws.onclose = () => {
                badge.textContent = "DISCONNECTED";
                badge.className = "badge badge-offline";
                log("WebSocket disconnected.", "log-warn");
            };
        }

        let isPhoneStreaming = false;
        let sensorHandler = null;
        let orientationHandler = null;
        let lastSensorData = { ax: 0, ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0, mx: 22.0, my: -5.0, mz: 41.0 };

        function updateButtonStyles(active) {
            document.getElementById("btn-mode-1").className = "mode-btn" + (active === "stationary" ? " active-stationary" : "");
            document.getElementById("btn-mode-2").className = "mode-btn" + (active === "moving" ? " active-moving" : "");
            document.getElementById("btn-mode-3").className = "mode-btn" + (active === "adaptive" ? " active-adaptive" : "");
            const phoneBtn = document.getElementById("btn-mode-phone");
            if (phoneBtn) {
                phoneBtn.className = "mode-btn" + (isPhoneStreaming ? " active-moving" : "");
            }
        }

        async function setMode(mode) {
            if (isPhoneStreaming) {
                togglePhoneSensors(); // stop phone sensors if active
            }
            if (currentMode === mode) {
                // Toggle off
                if (streamInterval) clearInterval(streamInterval);
                streamInterval = null;
                currentMode = null;
                updateButtonStyles(null);
                log(`Stopped ${mode} mode.`);
                return;
            }

            if (streamInterval) clearInterval(streamInterval);
            await connectWebSocket();

            currentMode = mode;
            updateButtonStyles(mode);

            if (mode === "stationary") {
                log(">>> MODE 1 ACTIVE: STATIONARY (REST). Hard-locking velocity to 0.0 m/s.", "log-warn");
                streamInterval = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: "live",
                            mode: "stationary",
                            seq: seq++,
                            timestamp: Date.now() / 1000,
                            ax: 0.0, ay: 0.0, az: 9.81,
                            gx: 0.0, gy: 0.0, gz: 0.0,
                            mx: 22.0, my: -5.0, mz: 41.0
                        }));
                    }
                }, 100);
            } 
            else if (mode === "moving") {
                log(">>> MODE 2 ACTIVE: MOVING (WALKING). Actively integrating forward steps.", "log-ok");
                let stepAngle = 0;
                streamInterval = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        stepAngle += 0.05;
                        const swingAcc = 0.9 + 0.4 * Math.sin(stepAngle * 6);
                        ws.send(JSON.stringify({
                            type: "live",
                            mode: "moving",
                            seq: seq++,
                            timestamp: Date.now() / 1000,
                            ax: Math.cos(stepAngle) * 0.1,
                            ay: swingAcc,
                            az: 9.81 + 0.5 * Math.sin(stepAngle * 6),
                            gx: 0.01, gy: 0.01, gz: 0.03,
                            mx: 22.0, my: -5.0, mz: 41.0
                        }));
                    }
                }, 100);
            }
            else if (mode === "adaptive") {
                log(">>> MODE 3 ACTIVE: ADAPTIVE STORAGE. Dynamically storing position & rest/moving state in DB.", "log-info");
                let cycle = 0;
                streamInterval = setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        cycle++;
                        // Alternate between 4 seconds walking and 3 seconds resting
                        const isRestCycle = (cycle % 70) > 40; // 40 samples (4s) moving, 30 samples (3s) rest
                        const ax = isRestCycle ? 0.0 : 0.2 * Math.sin(cycle * 0.2);
                        const ay = isRestCycle ? 0.0 : 0.85 + 0.4 * Math.sin(cycle * 0.5);
                        const az = isRestCycle ? 9.81 : 9.81 + 0.5 * Math.sin(cycle * 0.5);
                        const gx = isRestCycle ? 0.0 : 0.02;
                        const gz = isRestCycle ? 0.0 : 0.05;

                        ws.send(JSON.stringify({
                            type: "live",
                            mode: "adaptive",
                            seq: seq++,
                            timestamp: Date.now() / 1000,
                            ax: ax, ay: ay, az: az,
                            gx: gx, gy: 0.0, gz: gz,
                            mx: 22.0, my: -5.0, mz: 41.0
                        }));
                    }
                }, 100);
            }
        }

        async function fetchLatestDbState() {
            log("Querying database for latest stored trajectory point & state...", "log-info");
            const res = await fetch(`/device/${deviceId}/trajectory?limit=5`);
            const history = await res.json();
            if (history && history.length > 0) {
                const latest = history[history.length - 1];
                log(`[DB CONFIRMED] Last Stored Point -> X: ${latest.x.toFixed(3)}m, Y: ${latest.y.toFixed(3)}m, Velocity: ${latest.vx.toFixed(2)}m/s, State: ${latest.movement_state}`, "log-ok");
                alert(`Database Record Confirmed!\\n\\nPosition: (${latest.x.toFixed(3)}m, ${latest.y.toFixed(3)}m)\\nMovement State: ${latest.movement_state}\\nSteps: ${latest.step_count || 0}\\nTime: ${new Date(latest.timestamp * 1000).toLocaleTimeString()}`);
            } else {
                log("No points found in database yet. Run Mode 2 or 3 first.", "log-warn");
            }
        }

        function clearCanvas() {
            pathHistory = [{ x: 0, y: 0 }];
            currentPos = { x: 0, y: 0 };
            document.getElementById("stat-x").textContent = "0.000 m";
            document.getElementById("stat-y").textContent = "0.000 m";
            document.getElementById("stat-v").textContent = "0.00 m/s";
            document.getElementById("stat-state").innerHTML = '<span class="state-tag state-rest">REST</span>';
            document.getElementById("stat-steps").textContent = "0";
            document.getElementById("coord-hud").textContent = "(0.00m, 0.00m)";
            drawCanvas();
            log("Canvas reset.");
        }

        async function togglePhoneSensors() {
            const btn = document.getElementById("btn-mode-phone");
            if (isPhoneStreaming) {
                isPhoneStreaming = false;
                if (streamInterval) clearInterval(streamInterval);
                streamInterval = null;
                if (sensorHandler) window.removeEventListener("devicemotion", sensorHandler);
                if (orientationHandler) window.removeEventListener("deviceorientation", orientationHandler);
                btn.className = "mode-btn";
                log("Stopped Live Phone Hardware IMU Stream.", "log-warn");
                return;
            }

            if (currentMode) {
                currentMode = null;
                updateButtonStyles(null);
            }
            if (streamInterval) clearInterval(streamInterval);

            if (!window.DeviceMotionEvent) {
                alert("DeviceMotionEvent is not supported on this device/browser.");
                log("DeviceMotionEvent not supported on this browser.", "log-warn");
                return;
            }

            if (typeof DeviceMotionEvent.requestPermission === "function") {
                try {
                    const permission = await DeviceMotionEvent.requestPermission();
                    if (permission !== "granted") {
                        alert("Sensor motion permission denied.");
                        log("Sensor motion permission denied by user.", "log-warn");
                        return;
                    }
                } catch (err) {
                    log("Permission notice: " + err.message, "log-info");
                }
            }

            await connectWebSocket();
            isPhoneStreaming = true;
            btn.className = "mode-btn active-adaptive";
            log(">>> LIVE HARDWARE IMU ACTIVE: Streaming physical accelerometer & gyro at 10Hz...", "log-ok");

            sensorHandler = (event) => {
                const acc = event.accelerationIncludingGravity || event.acceleration;
                if (acc) {
                    lastSensorData.ax = acc.x || 0;
                    lastSensorData.ay = acc.y || 0;
                    lastSensorData.az = acc.z !== null && acc.z !== undefined ? acc.z : 9.81;
                }
                const rot = event.rotationRate;
                if (rot) {
                    lastSensorData.gx = (rot.beta || 0) * (Math.PI / 180);
                    lastSensorData.gy = (rot.gamma || 0) * (Math.PI / 180);
                    lastSensorData.gz = (rot.alpha || 0) * (Math.PI / 180);
                }
            };

            orientationHandler = (event) => {
                if (event.alpha !== null && event.alpha !== undefined) {
                    const headingRad = (event.alpha || 0) * (Math.PI / 180);
                    lastSensorData.mx = 25.0 * Math.cos(headingRad);
                    lastSensorData.my = 25.0 * Math.sin(headingRad);
                }
            };

            window.addEventListener("devicemotion", sensorHandler);
            window.addEventListener("deviceorientation", orientationHandler);

            streamInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: "live",
                        mode: "adaptive",
                        seq: seq++,
                        timestamp: Date.now() / 1000,
                        ax: Number(lastSensorData.ax.toFixed(4)),
                        ay: Number(lastSensorData.ay.toFixed(4)),
                        az: Number(lastSensorData.az.toFixed(4)),
                        gx: Number(lastSensorData.gx.toFixed(4)),
                        gy: Number(lastSensorData.gy.toFixed(4)),
                        gz: Number(lastSensorData.gz.toFixed(4)),
                        mx: Number(lastSensorData.mx.toFixed(2)),
                        my: Number(lastSensorData.my.toFixed(2)),
                        mz: Number(lastSensorData.mz.toFixed(2))
                    }));
                }
            }, 100);
        }

        drawCanvas();
        connectWebSocket();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
