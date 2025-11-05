"""FastAPI application for Autonote."""
import structlog
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional
from bson import ObjectId

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .database import Database, get_db
from .auth import create_access_token, get_current_user
from .observability import initialize_observability, get_tracer, get_app_metrics

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

# Get tracer and metrics
tracer = get_tracer(__name__)
metrics = get_app_metrics()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    session_id: str
    message_id: str


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


class LinkOut(BaseModel):
    """Model for note links (backlinks support)."""
    note_id: str
    type: str = "wiki"


class NoteCreate(BaseModel):
    """Request model for creating a note."""
    notebook_id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=500)
    content_md: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False


class NoteResponse(BaseModel):
    """Response model for note data."""
    id: str
    notebook_id: Optional[str] = None
    author_id: str
    title: str
    content_md: str
    tags: list[str]
    status: Literal["active", "archived", "trashed"]
    pinned: bool
    created_at: datetime
    updated_at: datetime
    version: int
    word_count: int
    links_out: list[LinkOut] = Field(default_factory=list)


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session data."""
    id: str
    user_id: str
    title: str
    created_at: datetime
    last_active_at: datetime


class Citation(BaseModel):
    """Model for message citations."""
    note_id: str
    chunk_id: Optional[str] = None
    span: dict[str, int] = Field(default_factory=dict)  # {"start": 0, "end": 120}


class ChatMessageResponse(BaseModel):
    """Response model for chat message data."""
    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("root_endpoint_accessed")
    return {"message": "Welcome to Autonote API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    logger.debug("health_check_requested")
    return {"status": "healthy", "service": "autonote-api"}


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
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
            "status": "active"
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
            created_at=user_doc["created_at"]
        )

        logger.info("user_registered_successfully", user_id=user_id, email=user.email)
        metrics.user_registrations.add(1)

        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )


@app.post("/auth/login", response_model=AuthResponse)
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
            created_at=user["created_at"]
        )

        logger.info("user_logged_in_successfully", user_id=user_id, email=login.email)
        metrics.user_logins.add(1)

        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Process user input and return a response with conversation history.

    Requires authentication. If session_id is not provided, creates a new session.
    Stores both user message and assistant response in chat_messages collection.
    """
    with tracer.start_as_current_span("process_chat") as span:
        user_id = str(current_user.get("_id"))
        user_name = current_user.get("name", "User")

        span.set_attribute("user.id", user_id)
        span.set_attribute("user.name", user_name)
        span.set_attribute("message.length", len(request.message))

        logger.info("chat_message_received", user_id=user_id, message_length=len(request.message))

        db = get_db()
        now = datetime.utcnow()

        # Get or create session
        if request.session_id:
            try:
                session_obj_id = ObjectId(request.session_id)
                session = await db.chat_sessions.find_one({
                    "_id": session_obj_id,
                    "user_id": ObjectId(user_id)
                })

                if not session:
                    logger.warning("chat_session_not_found", user_id=user_id, session_id=request.session_id)
                    raise HTTPException(status_code=404, detail="Chat session not found")

                session_id = request.session_id

                # Update last_active_at
                await db.chat_sessions.update_one(
                    {"_id": session_obj_id},
                    {"$set": {"last_active_at": now}}
                )

            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                logger.error("invalid_session_id", user_id=user_id, session_id=request.session_id, error=str(e))
                raise HTTPException(status_code=400, detail="Invalid session ID format")
        else:
            # Create new session
            session_doc = {
                "user_id": ObjectId(user_id),
                "title": f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
                "created_at": now,
                "last_active_at": now
            }
            result = await db.chat_sessions.insert_one(session_doc)
            session_id = str(result.inserted_id)
            session_obj_id = result.inserted_id
            logger.info("chat_session_auto_created", session_id=session_id, user_id=user_id)

        span.set_attribute("session.id", session_id)

        # Store user message
        user_message_doc = {
            "session_id": session_obj_id,
            "role": "user",
            "content": request.message,
            "citations": [],
            "created_at": now
        }
        user_msg_result = await db.chat_messages.insert_one(user_message_doc)

        # TODO: Replace with actual chat/processing logic (RAG, LLM, etc.)
        response_text = f"Hello {user_name}! You said: {request.message}"

        # Store assistant message
        assistant_message_doc = {
            "session_id": session_obj_id,
            "role": "assistant",
            "content": response_text,
            "citations": [],  # TODO: Add citations when implementing RAG
            "created_at": datetime.utcnow()
        }
        assistant_msg_result = await db.chat_messages.insert_one(assistant_message_doc)
        assistant_msg_id = str(assistant_msg_result.inserted_id)

        metrics.chat_messages.add(1)

        logger.info("chat_message_processed", user_id=user_id, session_id=session_id,
                   message_id=assistant_msg_id)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=assistant_msg_id
        )


@app.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(
    note: NoteCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new note with markdown content.

    Requires authentication. Supports multi-line markdown content,
    tags, and optional notebook assignment.
    """
    with tracer.start_as_current_span("create_note") as span:
        user_id = str(current_user.get("_id"))
        user_name = current_user.get("name", "User")

        span.set_attribute("user.id", user_id)
        span.set_attribute("note.title", note.title)
        span.set_attribute("note.tags_count", len(note.tags))

        logger.info("note_creation_attempt", user_id=user_id, title=note.title)

        db = get_db()

        # Calculate word count
        word_count = len(note.content_md.split())

        # Validate notebook_id if provided
        notebook_obj_id = None
        if note.notebook_id:
            try:
                notebook_obj_id = ObjectId(note.notebook_id)
                # Verify notebook exists and belongs to user
                notebook = await db.notebooks.find_one({
                    "_id": notebook_obj_id,
                    "author_id": ObjectId(user_id)
                })
                if not notebook:
                    logger.warning("note_creation_failed_invalid_notebook",
                                 user_id=user_id, notebook_id=note.notebook_id)
                    raise HTTPException(status_code=404, detail="Notebook not found")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                logger.error("note_creation_failed_invalid_notebook_id",
                           user_id=user_id, notebook_id=note.notebook_id, error=str(e))
                raise HTTPException(status_code=400, detail="Invalid notebook ID format")

        # Create note document
        now = datetime.utcnow()
        note_doc = {
            "notebook_id": notebook_obj_id,
            "author_id": ObjectId(user_id),
            "title": note.title,
            "content_md": note.content_md,
            "tags": note.tags,
            "status": "active",
            "pinned": note.pinned,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "word_count": word_count,
            "links_out": []
        }

        # Insert into database
        result = await db.notes.insert_one(note_doc)
        note_id = str(result.inserted_id)

        span.set_attribute("note.id", note_id)
        span.set_attribute("note.word_count", word_count)

        # Return response
        note_response = NoteResponse(
            id=note_id,
            notebook_id=note.notebook_id,
            author_id=user_id,
            title=note_doc["title"],
            content_md=note_doc["content_md"],
            tags=note_doc["tags"],
            status=note_doc["status"],
            pinned=note_doc["pinned"],
            created_at=note_doc["created_at"],
            updated_at=note_doc["updated_at"],
            version=note_doc["version"],
            word_count=note_doc["word_count"],
            links_out=[]
        )

        logger.info("note_created_successfully", note_id=note_id, user_id=user_id,
                   word_count=word_count)

        return note_response


@app.post("/chat/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new chat session.

    Requires authentication. Auto-generates title if not provided.
    """
    with tracer.start_as_current_span("create_chat_session") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)

        logger.info("chat_session_creation_attempt", user_id=user_id)

        db = get_db()

        # Create session document
        now = datetime.utcnow()
        session_doc = {
            "user_id": ObjectId(user_id),
            "title": session_data.title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
            "created_at": now,
            "last_active_at": now
        }

        # Insert into database
        result = await db.chat_sessions.insert_one(session_doc)
        session_id = str(result.inserted_id)

        span.set_attribute("session.id", session_id)

        logger.info("chat_session_created", session_id=session_id, user_id=user_id)

        return ChatSessionResponse(
            id=session_id,
            user_id=user_id,
            title=session_doc["title"],
            created_at=session_doc["created_at"],
            last_active_at=session_doc["last_active_at"]
        )


@app.get("/chat/sessions", response_model=list[ChatSessionResponse])
async def list_chat_sessions(
    current_user: dict = Depends(get_current_user),
    limit: int = 50,
    skip: int = 0
):
    """
    List all chat sessions for the current user.

    Returns sessions ordered by last_active_at (most recent first).
    """
    with tracer.start_as_current_span("list_chat_sessions") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("query.limit", limit)
        span.set_attribute("query.skip", skip)

        logger.info("list_chat_sessions_attempt", user_id=user_id, limit=limit, skip=skip)

        db = get_db()

        # Query sessions
        cursor = db.chat_sessions.find(
            {"user_id": ObjectId(user_id)}
        ).sort("last_active_at", -1).skip(skip).limit(limit)

        sessions = []
        async for session in cursor:
            sessions.append(ChatSessionResponse(
                id=str(session["_id"]),
                user_id=user_id,
                title=session["title"],
                created_at=session["created_at"],
                last_active_at=session["last_active_at"]
            ))

        span.set_attribute("result.count", len(sessions))

        logger.info("chat_sessions_listed", user_id=user_id, count=len(sessions))

        return sessions


@app.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    limit: int = 100,
    skip: int = 0
):
    """
    Get all messages for a specific chat session.

    Returns messages ordered by created_at (oldest first).
    """
    with tracer.start_as_current_span("get_session_messages") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("session.id", session_id)

        logger.info("get_session_messages_attempt", user_id=user_id, session_id=session_id)

        db = get_db()

        # Verify session exists and belongs to user
        try:
            session_obj_id = ObjectId(session_id)
            session = await db.chat_sessions.find_one({
                "_id": session_obj_id,
                "user_id": ObjectId(user_id)
            })

            if not session:
                logger.warning("session_not_found", user_id=user_id, session_id=session_id)
                raise HTTPException(status_code=404, detail="Chat session not found")

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error("invalid_session_id", user_id=user_id, session_id=session_id, error=str(e))
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Query messages
        cursor = db.chat_messages.find(
            {"session_id": session_obj_id}
        ).sort("created_at", 1).skip(skip).limit(limit)

        messages = []
        async for msg in cursor:
            citations = [
                Citation(
                    note_id=str(cit["note_id"]),
                    chunk_id=str(cit["chunk_id"]) if cit.get("chunk_id") else None,
                    span=cit.get("span", {})
                )
                for cit in msg.get("citations", [])
            ]

            messages.append(ChatMessageResponse(
                id=str(msg["_id"]),
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
                citations=citations,
                created_at=msg["created_at"]
            ))

        span.set_attribute("result.count", len(messages))

        logger.info("session_messages_retrieved", user_id=user_id, session_id=session_id, count=len(messages))

        return messages
