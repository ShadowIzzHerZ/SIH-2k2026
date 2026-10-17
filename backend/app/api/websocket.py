# 10Hz Dead Reckoning Backend Engine - SIH 2026
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from app.core.auth import authenticate_ws
from app.core.time_sync import TimeSyncManager
from app.core.queue_manager import IngestionQueueManager
from app.fusion.kalman_filter import IMUKalmanFilter
from app.config import get_settings
from app.cache.redis_client import cache_manager
from app.db.database import AsyncSessionLocal
from app.db.crud import get_last_known_position, save_trajectory_point

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Global Queue and Connection Pool
queue_mgr = IngestionQueueManager()
active_connections: Dict[str, WebSocket] = {}
device_kalman_filters: Dict[str, IMUKalmanFilter] = {}

# Per-device in-memory state is freed after DEVICE_IDLE_TTL_SEC with no traffic and
# no open socket. The filter state is also cached (Redis / memory fallback) after
# every sample, so a device returning later is restored by restore_or_init_filter.
# An idle TTL rather than cleanup-on-disconnect never races a reconnecting socket,
# lets frames still queued after a disconnect use the live filter, and also covers
# devices that only ever post to the REST /sensor-data fallback.
_device_last_seen: Dict[str, float] = {}
_IDLE_SWEEP_INTERVAL_SEC = 60.0
_last_idle_sweep = 0.0

def _touch_device(device_id: str, now: Optional[float] = None):
    _device_last_seen[device_id] = time.monotonic() if now is None else now

def sweep_idle_devices(now: Optional[float] = None) -> int:
    """Free filters and queue bookkeeping of devices idle longer than the TTL."""
    now = time.monotonic() if now is None else now
    cutoff = now - settings.DEVICE_IDLE_TTL_SEC
    known = set(device_kalman_filters) | set(_device_last_seen)
    freed = 0
    for device_id in known:
        if device_id in active_connections:
            continue
        if _device_last_seen.get(device_id, 0.0) > cutoff:
            continue
        device_kalman_filters.pop(device_id, None)
        _device_last_seen.pop(device_id, None)
        queue_mgr.forget_device(device_id)
        freed += 1
    if freed:
        logger.info(f"Freed in-memory state for {freed} idle device(s).")
    return freed

def _maybe_sweep_idle_devices():
    global _last_idle_sweep
    now = time.monotonic()
    if now - _last_idle_sweep >= _IDLE_SWEEP_INTERVAL_SEC:
        _last_idle_sweep = now
        sweep_idle_devices(now)

async def restore_or_init_filter(device_id: str) -> IMUKalmanFilter:
    """
    Restores Kalman filter state from Redis, or falls back to TimescaleDB
    if Redis key expired (after >1hr), or creates new filter.
    """
    # 1. Try Redis
    cached_state = await cache_manager.get_live_device_state(device_id)
    if cached_state and "x" in cached_state and "P" in cached_state:
        logger.info(f"Restored Kalman filter for {device_id} from Redis.")
        return IMUKalmanFilter.from_state_dict(cached_state)

    # 2. Redis miss / expired: Query TimescaleDB
    async with AsyncSessionLocal() as session:
        last_pos = await get_last_known_position(session, device_id)
        if last_pos:
            logger.info(f"Restored Kalman filter for {device_id} from TimescaleDB (Redis TTL had expired).")
            kf = IMUKalmanFilter()
            kf.x[0, 0] = last_pos["x"]
            kf.x[1, 0] = last_pos["y"]
            kf.x[2, 0] = last_pos["z"]
            kf.x[3, 0] = last_pos["vx"]
            kf.x[4, 0] = last_pos["vy"]
            kf.x[5, 0] = last_pos["vz"]
            kf.x[6, 0] = last_pos["roll"]
            kf.x[7, 0] = last_pos["pitch"]
            kf.x[8, 0] = last_pos["yaw"]
            kf.movement_state = last_pos.get("movement_state", "REST")
            kf.step_count = last_pos.get("step_count", 0)
            kf.last_timestamp = last_pos["timestamp"]
            return kf

    # 3. Fresh device session
    logger.info(f"Initialized brand-new Kalman filter state for {device_id}.")
    return IMUKalmanFilter()

@router.websocket("/ws/track/{device_id}")
async def websocket_track_endpoint(websocket: WebSocket, device_id: str):
    # Step 1: JWT Authentication during handshake
    authenticated_id = await authenticate_ws(websocket)
    if not authenticated_id:
        return

    # Check device_id match
    if authenticated_id != device_id and authenticated_id != "admin":
        logger.warning(f"Device ID mismatch: token sub={authenticated_id}, path={device_id}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Device ID mismatch")
        return

    await websocket.accept()
    active_connections[device_id] = websocket
    _touch_device(device_id)
    logger.info(f"WebSocket client connected: {device_id}")

    # Restore or create Kalman filter
    device_kalman_filters[device_id] = await restore_or_init_filter(device_id)

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "live")

            # 1. NTP Time Sync Request
            if msg_type == "time_sync":
                response = TimeSyncManager.process_sync_request(data)
                await websocket.send_text(json.dumps(response))
                continue

            # 2. Real-Time Live Frame (10Hz, Priority 0)
            elif msg_type == "live":
                await queue_mgr.enqueue_live(device_id, data)

            # 3. Backlog Recovery Dump (Priority 1 with chunking)
            elif msg_type == "backlog":
                chunk = data.get("chunk", [])
                enqueued = await queue_mgr.enqueue_backlog_chunk(device_id, chunk)
                logger.info(f"Enqueued {enqueued} backlog frames for {device_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {device_id}")
    except Exception as e:
        logger.error(f"Error on WebSocket for {device_id}: {e}")
    finally:
        active_connections.pop(device_id, None)

async def queue_worker():
    """
    Background worker that continuously pulls from the Priority Ingestion Queue.
    Processes Live (P0) immediately, and Backlog (P1) cooperatively without blocking.
    """
    logger.info("Starting background Ingestion Queue Worker...")
    while True:
        item = await queue_mgr.get_next_item()
        try:
            device_id = item.device_id
            packet = item.data
            seq = item.seq_num
            is_backlog = item.is_backlog

            # 1. Retrieve or recover Kalman filter
            kf = device_kalman_filters.get(device_id)
            if not kf:
                kf = await restore_or_init_filter(device_id)
                device_kalman_filters[device_id] = kf
            _touch_device(device_id)

            # 2. 9-DOF IMU Kalman Filter Statistical Sensor Fusion
            fused_state = kf.process_sample(packet)

            payload_out = {
                "type": "position_update",
                "device_id": device_id,
                "seq": seq,
                "timestamp": item.timestamp,
                "x": round(fused_state["x"], 4),
                "y": round(fused_state["y"], 4),
                "z": round(fused_state["z"], 4),
                "vx": round(fused_state["vx"], 4),
                "vy": round(fused_state["vy"], 4),
                "vz": round(fused_state["vz"], 4),
                "roll": round(fused_state["roll"], 4),
                "pitch": round(fused_state["pitch"], 4),
                "yaw": round(fused_state["yaw"], 4),
                "movement_state": fused_state.get("movement_state", "REST"),
                "step_count": fused_state.get("step_count", 0),
                "is_backlog": is_backlog,
                "is_verified": True
            }

            # 3. Push back to client if currently connected
            ws = active_connections.get(device_id)
            if ws:
                try:
                    await ws.send_text(json.dumps(payload_out))
                except Exception:
                    pass

            # 4. Update Redis Live Cache (1-hour TTL)
            state_dict = kf.to_state_dict()
            await cache_manager.set_live_device_state(device_id, state_dict)

            # 5. Non-blocking asynchronous persistence to database (prevents I/O lag on 10Hz stream)
            asyncio.create_task(_persist_point(payload_out))

        except Exception as e:
            logger.error(f"Error processing item in queue worker: {e}", exc_info=True)
        finally:
            queue_mgr.task_done()
            _maybe_sweep_idle_devices()

_db_write_lock = asyncio.Lock()

async def _persist_point(payload: Dict[str, Any]):
    """Background task for async persistence without stalling real-time 10Hz stream."""
    try:
        async with _db_write_lock:
            async with AsyncSessionLocal() as session:
                await save_trajectory_point(session, payload)
    except Exception as e:
        logger.error(f"Error persisting point for {payload.get('device_id')}: {e}")


