"""
ForgeMind AI Suite — Authentication & Authorization
OAuth2 + JWT (RS256) with role-based access control.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.core.config import get_settings

settings = get_settings()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AgentRole(str, Enum):
    """Role-based access control for agent API endpoints."""

    ADMIN = "admin"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    AGENT = "agent"  # inter-agent communication
    SYSTEM = "system"


class TokenPayload(BaseModel):
    """JWT token payload schema."""

    sub: str
    role: AgentRole
    agent_id: Optional[str] = None
    exp: datetime
    iat: datetime
    jti: Optional[str] = None
    scopes: list[str] = []


class AuthUser(BaseModel):
    """Authenticated user/agent context."""

    user_id: str
    role: AgentRole
    agent_id: Optional[str] = None
    scopes: list[str] = []


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    role: AgentRole,
    agent_id: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes))
    payload = {
        "sub": subject,
        "role": role.value,
        "agent_id": agent_id,
        "scopes": scopes or [],
        "exp": expire,
        "iat": now,
        "iss": settings.app_name,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def create_refresh_token(subject: str, role: AgentRole) -> str:
    """Create a JWT refresh token with longer expiry."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload = {
        "sub": subject,
        "role": role.value,
        "exp": expire,
        "iat": now,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        return TokenPayload(
            sub=payload["sub"],
            role=AgentRole(payload["role"]),
            agent_id=payload.get("agent_id"),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            scopes=payload.get("scopes", []),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthUser:
    """Extract and validate the current user from the JWT token."""
    token_data = decode_token(credentials.credentials)
    return AuthUser(
        user_id=token_data.sub,
        role=token_data.role,
        agent_id=token_data.agent_id,
        scopes=token_data.scopes,
    )


def require_role(*roles: AgentRole):
    """Dependency that requires one of the specified roles."""

    async def role_checker(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {current_user.role} not authorized. Required: {[r.value for r in roles]}",
            )
        return current_user

    return role_checker


def require_scope(scope: str):
    """Dependency that requires a specific scope."""

    async def scope_checker(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if scope not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {scope}",
            )
        return current_user

    return scope_checker
