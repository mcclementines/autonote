"""Authentication endpoints."""

from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException

from ..auth import create_access_token
from ..database import get_db
from ..models import AuthResponse, LoginRequest, UserCreate, UserResponse
from ..observability import get_app_metrics, get_tracer

# Initialize logger
logger = structlog.get_logger(__name__)

# Get tracer and metrics
tracer = get_tracer(__name__)
metrics = get_app_metrics()

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register_user(user: UserCreate):
    """
    Register a new user and return JWT token.

    Creates a new user with the provided email and name.
    Email must be unique. Returns JWT token for immediate authentication.
    """
    with tracer.start_as_current_span("register_user") as span:
        span.set_attribute("user.email", user.email)
        span.set_attribute("user.name", user.name)

        logger.info("user_registration_attempt", email=user.email, name=user.name)

        db = get_db()

        # Check if user with this email already exists
        existing_user = await db.users.find_one({"email": user.email})
        if existing_user:
            logger.warning("registration_failed_duplicate_email", email=user.email)
            metrics.auth_failures.add(1, {"reason": "duplicate_email"})
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user document
        user_doc = {
            "email": user.email,
            "name": user.name,
            "created_at": datetime.utcnow(),
            "status": "active",
        }

        # Insert into database
        result = await db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)

        span.set_attribute("user.id", user_id)

        # Create JWT token
        access_token = create_access_token(user_id=user_id, email=user.email)

        # Return auth response
        user_response = UserResponse(
            id=user_id,
            email=user_doc["email"],
            name=user_doc["name"],
            status=user_doc["status"],
            created_at=user_doc["created_at"],
        )

        logger.info("user_registered_successfully", user_id=user_id, email=user.email)
        metrics.user_registrations.add(1)

        return AuthResponse(access_token=access_token, token_type="bearer", user=user_response)


@router.post("/login", response_model=AuthResponse)
async def login_user(login: LoginRequest):
    """
    Login with email and return JWT token.

    Validates email exists and user is active.
    Returns JWT token for authentication.
    """
    with tracer.start_as_current_span("login_user") as span:
        span.set_attribute("user.email", login.email)

        logger.info("user_login_attempt", email=login.email)

        db = get_db()

        # Find user by email
        user = await db.users.find_one({"email": login.email})
        if not user:
            logger.warning("login_failed_user_not_found", email=login.email)
            metrics.auth_failures.add(1, {"reason": "user_not_found"})
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if user is active
        if user.get("status") != "active":
            logger.warning("login_failed_account_disabled", email=login.email)
            metrics.auth_failures.add(1, {"reason": "account_disabled"})
            raise HTTPException(status_code=403, detail="User account is disabled")

        user_id = str(user["_id"])
        span.set_attribute("user.id", user_id)

        # Create JWT token
        access_token = create_access_token(user_id=user_id, email=user["email"])

        # Return auth response
        user_response = UserResponse(
            id=user_id,
            email=user["email"],
            name=user["name"],
            status=user["status"],
            created_at=user["created_at"],
        )

        logger.info("user_logged_in_successfully", user_id=user_id, email=login.email)
        metrics.user_logins.add(1)

        return AuthResponse(access_token=access_token, token_type="bearer", user=user_response)
