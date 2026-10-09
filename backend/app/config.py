# 10Hz Dead Reckoning Backend Engine - SIH 2026
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "SIH Dead Reckoning Backend"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Security & JWT
    SECRET_KEY: str  # required, set in .env (see .env.example)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Caching Layer (Redis with in-memory fallback)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_LIVE_SEC: int = 3600             # 1 hour for device live position
    REDIS_TTL_CALIBRATION_SEC: int = 86400     # 24 hours for calibration constants

    # Storage Layer (PostgreSQL/TimescaleDB or SQLite fallback)
    DATABASE_URL: str = "sqlite+aiosqlite:///./sih_trajectories.db"

    # IMU & Queue Configuration (10Hz Standard)
    IMU_SAMPLING_RATE_HZ: float = 10.0         # 10 Hz sampling frequency (100ms intervals)
    BACKLOG_CHUNK_SIZE: int = 20               # Max packets to yield per backlog chunk
    MAX_QUEUE_SIZE: int = 50000

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

