"""Chat and chat session endpoints."""

import structlog
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from bson import ObjectId

from ..database import get_db
from ..auth import get_current_user
from ..observability import get_tracer, get_app_metrics
from ..models import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageResponse,
    Citation,
)

# Initialize logger
logger = structlog.get_logger(__name__)

# Get tracer and metrics
tracer = get_tracer(__name__)
metrics = get_app_metrics()

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
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


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
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


@router.get("/sessions", response_model=list[ChatSessionResponse])
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


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
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
