# 10Hz Dead Reckoning Backend Engine - SIH 2026
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import hmac
import logging
import jwt
from fastapi import Depends, WebSocket, status, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

ADMIN_SUBJECT = "admin"
_bearer = HTTPBearer(auto_error=False)
_warned_open_enrollment = False

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JSON Web Token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify HMAC-SHA256 JWT."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

# Browsers cannot set headers on a WebSocket, so they offer the token as a
# subprotocol pair: new WebSocket(url, ["bearer", token]).
WS_BEARER_SUBPROTOCOL = "bearer"

def ws_token_from_handshake(websocket: WebSocket) -> Optional[str]:
    """
    JWT from the WebSocket handshake, in order of preference:
    'Authorization: Bearer <jwt>', 'Sec-WebSocket-Protocol: bearer, <jwt>',
    then the legacy ?token=<jwt> query param (kept for existing clients, but
    URLs end up in proxy access logs).
    """
    auth_header = websocket.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token:
            return token
    offered = websocket.scope.get("subprotocols") or []
    if WS_BEARER_SUBPROTOCOL in offered:
        i = offered.index(WS_BEARER_SUBPROTOCOL)
        if i + 1 < len(offered) and offered[i + 1]:
            return offered[i + 1]
    return websocket.query_params.get("token") or None

def ws_accept_subprotocol(websocket: WebSocket) -> Optional[str]:
    """Subprotocol to echo on accept; browsers drop the socket if an offered one is not echoed."""
    offered = websocket.scope.get("subprotocols") or []
    return WS_BEARER_SUBPROTOCOL if WS_BEARER_SUBPROTOCOL in offered else None

async def authenticate_ws(websocket: WebSocket) -> Optional[str]:
    """
    Authenticate WebSocket connection during initial handshake.
    Token source: see ws_token_from_handshake.
    Returns device_id if authenticated, else None.
    """
    token = ws_token_from_handshake(websocket)

    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing auth token")
        return None

    payload = verify_jwt_token(token)
    if not payload:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or expired token")
        return None

    device_id = payload.get("sub") or payload.get("device_id")
    if not device_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token missing device_id")
        return None

    return str(device_id)


def auth_enabled() -> bool:
    """Token checks on minting and on the REST data endpoints are on once DEVICE_ENROLL_SECRET is set."""
    return bool(settings.DEVICE_ENROLL_SECRET)

def _secret_matches(given: Optional[str], expected: str) -> bool:
    if not expected or not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))

def check_token_mint_allowed(device_id: str, secret: Optional[str]) -> None:
    """
    Raise 403 unless the caller may mint a token for device_id.
    "admin" needs ADMIN_SECRET (refused when auth is enabled without one); a device
    token needs DEVICE_ENROLL_SECRET (or ADMIN_SECRET) once DEVICE_ENROLL_SECRET is set.
    """
    global _warned_open_enrollment
    if device_id == ADMIN_SUBJECT:
        if settings.ADMIN_SECRET:
            if not _secret_matches(secret, settings.ADMIN_SECRET):
                raise HTTPException(status_code=403, detail="Invalid admin secret")
            return
        if auth_enabled():
            raise HTTPException(status_code=403, detail="Admin tokens are disabled; set ADMIN_SECRET")
    elif auth_enabled():
        if not (_secret_matches(secret, settings.DEVICE_ENROLL_SECRET)
                or _secret_matches(secret, settings.ADMIN_SECRET)):
            raise HTTPException(status_code=403, detail="Invalid enrollment secret")
        return
    if not _warned_open_enrollment:
        _warned_open_enrollment = True
        logger.warning("DEVICE_ENROLL_SECRET is not set: /auth/token mints tokens (including 'admin' "
                       "without ADMIN_SECRET) for anyone and REST data endpoints need no token.")

async def get_rest_subject(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """
    Device id of the Bearer token on a REST request. None when auth is disabled;
    401 when auth is enabled and the token is missing or invalid.
    """
    if not auth_enabled():
        return None
    payload = verify_jwt_token(creds.credentials) if creds else None
    subject = (payload.get("sub") or payload.get("device_id")) if payload else None
    if not subject:
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    return str(subject)

def ensure_device_access(subject: Optional[str], device_id: str) -> None:
    """403 when an authenticated caller is neither the device itself nor admin."""
    if subject is not None and subject not in (device_id, ADMIN_SUBJECT):
        raise HTTPException(status_code=403, detail="Token is not valid for this device")

async def require_device_access(device_id: str, subject: Optional[str] = Depends(get_rest_subject)) -> None:
    """Route dependency for /device/{device_id}/... endpoints."""
    ensure_device_access(subject, device_id)
