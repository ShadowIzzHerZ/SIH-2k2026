# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
Token secrets for the client scripts (simulate_device.py, verify_deployment.py).
Reads the same variables as the server (DEVICE_ENROLL_SECRET, falling back to
ADMIN_SECRET, which the server also accepts for device tokens) from the
environment or backend/.env. Empty means the server runs with auth off.
"""
import os
from pathlib import Path
from typing import Dict, Optional

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

def _from_env_file(name: str) -> str:
    try:
        lines = _ENV_FILE.read_text().splitlines()
    except OSError:
        return ""
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"")
    return ""

def _setting(name: str) -> str:
    return os.environ.get(name) or _from_env_file(name)

def enroll_secret() -> str:
    return _setting("DEVICE_ENROLL_SECRET") or _setting("ADMIN_SECRET")

def token_request_body(device_id: str, expires_minutes: int = 60) -> Dict:
    """POST /auth/token body, with the secret only when one is configured."""
    body = {"device_id": device_id, "expires_minutes": expires_minutes}
    secret = enroll_secret()
    if secret:
        body["secret"] = secret
    return body

def bearer_headers(token: Optional[str]) -> Dict[str, str]:
    """Authorization header for the REST data endpoints (ignored by a server with auth off)."""
    return {"Authorization": f"Bearer {token}"} if token else {}
