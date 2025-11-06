"""Health check and root endpoints."""

import structlog
from fastapi import APIRouter

# Initialize logger
logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint."""
    logger.info("root_endpoint_accessed")
    return {"message": "Welcome to Autonote API"}


@router.get("/health")
async def health():
    """Health check endpoint."""
    logger.debug("health_check_requested")
    return {"status": "healthy", "service": "autonote-api"}
