# 10Hz Dead Reckoning Backend Engine - SIH 2026
from typing import List, Optional
from datetime import timedelta
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db_session
from app.core.auth import create_access_token
from app.cache.redis_client import cache_manager
from app.api.websocket import queue_mgr, active_connections, device_kalman_filters
from app.db.crud import get_trajectory_history, get_last_known_position

router = APIRouter()

class TokenRequest(BaseModel):
    device_id: str = Field(..., description="Unique hardware or app UUID")
    expires_minutes: Optional[int] = Field(default=60 * 24 * 7, description="Token lifetime in minutes")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    device_id: str
    expires_in_seconds: int

class SensorPacketSchema(BaseModel):
    seq: int
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: Optional[float] = None
    my: Optional[float] = None
    mz: Optional[float] = None
    mode: Optional[str] = Field(default="adaptive", description="'stationary', 'moving', or 'adaptive'")

class SetModeRequest(BaseModel):
    mode: str = Field(..., description="Operating mode: 'stationary', 'moving', or 'adaptive'")

class BatchUploadRequest(BaseModel):
    device_id: str
    packets: List[SensorPacketSchema]

@router.post("/auth/token", response_model=TokenResponse, summary="Mint JWT token for a device")
async def generate_device_token(req: TokenRequest):
    """Generates an authentication token for WebSocket handshake."""
    token = create_access_token(
        data={"sub": req.device_id, "device_id": req.device_id},
        expires_delta=timedelta(minutes=req.expires_minutes)
    )
    return TokenResponse(
        access_token=token,
        device_id=req.device_id,
        expires_in_seconds=req.expires_minutes * 60
    )

@router.post("/sensor-data", summary="Fallback REST batch upload endpoint")
async def upload_batch_sensor_data(req: BatchUploadRequest):
    """
    REST fallback when WebSockets are blocked by proxies or firewalls.
    Enqueues batch to the priority queue as backlog.
    """
    packets_dict = [p.model_dump() for p in req.packets]
    enqueued = await queue_mgr.enqueue_backlog_chunk(req.device_id, packets_dict)
    return {
        "status": "enqueued",
        "device_id": req.device_id,
        "count": enqueued,
        "queue_size": queue_mgr.queue_size
    }

@router.get("/health", summary="System healthcheck and live status")
async def health_check():
    """Returns status of cache layer, queues, and active connections."""
    return {
        "status": "healthy",
        "redis_connected": cache_manager.is_redis_connected,
        "active_websockets": len(active_connections),
        "queue_depth": queue_mgr.queue_size
    }

@router.get("/device/{device_id}/trajectory", summary="Fetch stored trajectory history")
async def get_device_trajectory(
    device_id: str,
    limit: int = 500,
    db: AsyncSession = Depends(get_db_session)
):
    """Returns chronological trajectory points from TimescaleDB/SQLite."""
    points = await get_trajectory_history(db, device_id, limit=limit)
    return [
        {
            "seq": p.seq_num,
            "timestamp": p.time.timestamp(),
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "vx": p.vx,
            "vy": p.vy,
            "vz": p.vz,
            "roll": p.roll,
            "pitch": p.pitch,
            "yaw": p.yaw,
            "movement_state": p.movement_state or "REST",
            "step_count": p.step_count or 0,
            "is_backlog": p.is_backlog,
            "is_verified": p.is_verified
        }
        for p in points
    ]

@router.get("/device/{device_id}/state", summary="Fetch latest live position & movement state")
async def get_device_latest_state(
    device_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Returns the current position, velocity, orientation, movement state, and step count.
    First checks Redis cache, then falls back to the database.
    """
    cached = await cache_manager.get_live_device_state(device_id)
    if cached and "x" in cached:
        x_vec = cached["x"]
        def _get_val(idx: int) -> float:
            try:
                elem = x_vec[idx]
                if isinstance(elem, (list, tuple)):
                    return float(elem[0])
                return float(elem)
            except (IndexError, TypeError, ValueError):
                return 0.0

        return {
            "device_id": device_id,
            "source": "redis_cache",
            "x": round(_get_val(0), 4),
            "y": round(_get_val(1), 4),
            "z": round(_get_val(2), 4),
            "vx": round(_get_val(3), 4),
            "vy": round(_get_val(4), 4),
            "vz": round(_get_val(5), 4),
            "roll": round(_get_val(6), 4),
            "pitch": round(_get_val(7), 4),
            "yaw": round(_get_val(8), 4),
            "movement_state": cached.get("movement_state", "REST"),
            "step_count": cached.get("step_count", 0),
            "timestamp": cached.get("last_timestamp")
        }

    last_db = await get_last_known_position(db, device_id)
    if last_db:
        return {
            "device_id": device_id,
            "source": "database",
            **last_db
        }

    return {
        "device_id": device_id,
        "source": "none",
        "movement_state": "REST",
        "step_count": 0,
        "x": 0.0, "y": 0.0, "z": 0.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "message": "No trajectory points recorded yet for this device."
    }

@router.post("/device/{device_id}/mode", summary="Set operational mode for a device")
async def set_device_mode(device_id: str, req: SetModeRequest):
    """
    Switches operational mode between 'stationary' (rest ZUPT), 'moving', and 'adaptive'.
    """
    valid_modes = {"stationary", "moving", "adaptive"}
    mode_clean = req.mode.lower().strip()
    if mode_clean not in valid_modes:
        return {"error": f"Invalid mode '{req.mode}'. Choose from: {list(valid_modes)}"}
    
    kf = device_kalman_filters.get(device_id)
    if kf:
        if mode_clean == "stationary":
            kf.movement_state = "REST"
            kf.x[3:6, 0] = 0.0
        elif mode_clean == "moving":
            kf.movement_state = "MOVING"
    
    return {
        "device_id": device_id,
        "mode": mode_clean,
        "status": "applied"
    }


