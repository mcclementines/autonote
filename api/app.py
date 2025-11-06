"""FastAPI application for Autonote."""
import structlog
from fastapi import FastAPI
from contextlib import asynccontextmanager

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .database import Database
from .observability import initialize_observability
from .routes import health_router, auth_router, chat_router, notes_router

# Initialize logger
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("api_starting")

    # Initialize OpenTelemetry
    initialize_observability()

    # Connect to database
    await Database.connect()
    logger.info("api_started")

    yield

    # Shutdown
    logger.info("api_shutting_down")
    await Database.disconnect()
    logger.info("api_shutdown_complete")


app = FastAPI(
    title="Autonote API",
    description="API for processing user input and generating responses",
    version="0.1.0",
    lifespan=lifespan
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(notes_router)
