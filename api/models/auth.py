"""Authentication-related Pydantic models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """Request model for user registration."""

    email: EmailStr
    name: str


class UserResponse(BaseModel):
    """Response model for user data."""

    id: str
    email: str
    name: str
    status: Literal["active", "disabled"]
    created_at: datetime


class AuthResponse(BaseModel):
    """Response model for authentication endpoints."""

    access_token: str
    token_type: str
    user: UserResponse


class LoginRequest(BaseModel):
    """Request model for login."""

    email: EmailStr
