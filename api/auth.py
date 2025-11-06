"""Authentication utilities for JWT token management."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from opentelemetry import trace

from .database import get_db

# Initialize logger
logger = structlog.get_logger(__name__)


# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", "30"))

security = HTTPBearer()


def create_access_token(user_id: str, email: str) -> str:
    """
    Create a JWT access token for a user.

    Args:
        user_id: The user's ID
        email: The user's email

    Returns:
        Encoded JWT token
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("create_access_token") as span:
        span.set_attribute("user.id", user_id)

        logger.debug("jwt_token_creating", user_id=user_id)

        expire = datetime.utcnow() + timedelta(days=EXPIRATION_DAYS)
        to_encode = {
            "sub": user_id,  # Subject (user_id)
            "email": email,
            "exp": expire,
        }
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        logger.debug("jwt_token_created", user_id=user_id)

        return encoded_jwt


def decode_access_token(token: str) -> dict | None:
    """
    Decode and verify a JWT token.

    Args:
        token: The JWT token to decode

    Returns:
        Decoded token payload or None if invalid
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("decode_access_token") as span:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            span.set_attribute("user.id", user_id)
            logger.debug("jwt_token_decoded", user_id=user_id)
            return payload
        except JWTError as e:
            logger.warning("jwt_token_decode_failed", error=str(e), error_type=type(e).__name__)
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get the current authenticated user.

    Validates JWT token and retrieves user from database.
    Raises 401 if token is invalid or user not found.
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("get_current_user") as span:
        token = credentials.credentials

        logger.debug("auth_validating_credentials")

        # Decode token
        payload = decode_access_token(token)
        if payload is None:
            logger.warning("auth_failed_invalid_token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("auth_failed_missing_user_id")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        span.set_attribute("user.id", user_id)

        # Get user from database
        db = get_db()
        from bson import ObjectId

        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
        except Exception as e:
            logger.error("auth_db_error", user_id=user_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if user is None:
            logger.warning("auth_failed_user_not_found", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is active
        if user.get("status") != "active":
            logger.warning("auth_failed_account_disabled", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled"
            )

        logger.debug("auth_user_authenticated", user_id=user_id)

        return user
