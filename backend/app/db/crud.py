# 10Hz Dead Reckoning Backend Engine - SIH 2026
from datetime import datetime, timezone
import json
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.models import TrajectoryPoint, DeviceSession

async def get_last_known_position(db: AsyncSession, device_id: str) -> Optional[Dict[str, Any]]:
    """
    Fallback for Redis TTL expiration (Problem 6):
    Queries permanent database (TimescaleDB/PostgreSQL) to retrieve the last verified
    trajectory point and state matrix when a device reconnects after >1 hour.
    """
    stmt = (
        select(TrajectoryPoint)
        .where(TrajectoryPoint.device_id == device_id)
        .order_by(desc(TrajectoryPoint.time))
        .limit(1)
    )
    result = await db.execute(stmt)
    point = result.scalar_one_or_none()
    if not point:
        return None

    return {
        "device_id": point.device_id,
        "seq": point.seq_num,
        "timestamp": point.time.timestamp(),
        "x": point.x,
        "y": point.y,
        "z": point.z,
        "vx": point.vx,
        "vy": point.vy,
        "vz": point.vz,
        "roll": point.roll,
        "pitch": point.pitch,
        "yaw": point.yaw,
        "movement_state": point.movement_state or "REST",
        "step_count": point.step_count or 0,
        "is_verified": point.is_verified,
        "covariance": json.loads(point.covariance_json) if point.covariance_json else None
    }

async def save_trajectory_point(db: AsyncSession, point_data: Dict[str, Any]) -> TrajectoryPoint:
    """Insert a single verified trajectory point."""
    ts = point_data.get("timestamp")
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)

    point = TrajectoryPoint(
        time=dt,
        device_id=point_data["device_id"],
        session_id=point_data.get("session_id", "default"),
        seq_num=point_data.get("seq", 0),
        x=point_data.get("x", 0.0),
        y=point_data.get("y", 0.0),
        z=point_data.get("z", 0.0),
        vx=point_data.get("vx", 0.0),
        vy=point_data.get("vy", 0.0),
        vz=point_data.get("vz", 0.0),
        roll=point_data.get("roll", 0.0),
        pitch=point_data.get("pitch", 0.0),
        yaw=point_data.get("yaw", 0.0),
        movement_state=point_data.get("movement_state", "REST"),
        step_count=point_data.get("step_count", 0),
        is_backlog=point_data.get("is_backlog", False),
        is_verified=point_data.get("is_verified", True),
        covariance_json=json.dumps(point_data.get("covariance")) if point_data.get("covariance") else None,
        raw_sensor_json=json.dumps(point_data.get("raw_sensor")) if point_data.get("raw_sensor") else None,
    )
    db.add(point)
    await db.commit()
    return point

async def save_trajectory_batch(db: AsyncSession, points_data: List[Dict[str, Any]]) -> int:
    """Bulk-insert backlog chunk into permanent database."""
    if not points_data:
        return 0

    points = []
    for p in points_data:
        ts = p.get("timestamp")
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
        points.append(
            TrajectoryPoint(
                time=dt,
                device_id=p["device_id"],
                session_id=p.get("session_id", "default"),
                seq_num=p.get("seq", 0),
                x=p.get("x", 0.0),
                y=p.get("y", 0.0),
                z=p.get("z", 0.0),
                vx=p.get("vx", 0.0),
                vy=p.get("vy", 0.0),
                vz=p.get("vz", 0.0),
                roll=p.get("roll", 0.0),
                pitch=p.get("pitch", 0.0),
                yaw=p.get("yaw", 0.0),
                movement_state=p.get("movement_state", "REST"),
                step_count=p.get("step_count", 0),
                is_backlog=p.get("is_backlog", False),
                is_verified=p.get("is_verified", True),
                covariance_json=json.dumps(p.get("covariance")) if p.get("covariance") else None,
            )
        )
    db.add_all(points)
    await db.commit()
    return len(points)

async def get_trajectory_history(db: AsyncSession, device_id: str, limit: int = 500) -> List[TrajectoryPoint]:
    """Fetch recent trajectory history for a device."""
    stmt = (
        select(TrajectoryPoint)
        .where(TrajectoryPoint.device_id == device_id)
        .order_by(TrajectoryPoint.time.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())

