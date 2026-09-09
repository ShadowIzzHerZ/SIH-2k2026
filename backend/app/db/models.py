# 10Hz Dead Reckoning Backend Engine - SIH 2026
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, Index
from app.db.database import Base

class DeviceSession(Base):
    __tablename__ = "device_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_active_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    calibration_data = Column(Text, nullable=True)

class TrajectoryPoint(Base):
    __tablename__ = "device_trajectories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    device_id = Column(String(64), index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False, default="default")
    seq_num = Column(Integer, nullable=False, default=0)
    
    # 3D Position Coordinates (meters)
    x = Column(Float, nullable=False, default=0.0)
    y = Column(Float, nullable=False, default=0.0)
    z = Column(Float, nullable=False, default=0.0)
    
    # Velocities (m/s)
    vx = Column(Float, default=0.0)
    vy = Column(Float, default=0.0)
    vz = Column(Float, default=0.0)
    
    # Orientation Euler angles (radians)
    roll = Column(Float, default=0.0)
    pitch = Column(Float, default=0.0)
    yaw = Column(Float, default=0.0)
    
    # Metadata & Quality
    is_backlog = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=True)
    movement_state = Column(String(20), default="REST")  # "REST" or "MOVING"
    step_count = Column(Integer, default=0)
    covariance_json = Column(Text, nullable=True)
    raw_sensor_json = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_device_time", "device_id", "time"),
        Index("idx_device_seq", "device_id", "seq_num"),
    )

