"""Chat and chat session endpoints."""

import json
import os
from datetime import datetime

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from connectors.openai import OpenAIConnector, OpenAIModel

from ..auth import get_current_user
from ..database import get_db
from ..models import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    Citation,
)
from ..observability import get_app_metrics, get_tracer
from ..prompts import get_default_system_prompt, get_rag_system_prompt
from ..services import NoteRetrieval

# Initialize logger
logger = structlog.get_logger(__name__)

# Get tracer and metrics
tracer = get_tracer(__name__)
metrics = get_app_metrics()

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
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
                session = await db.chat_sessions.find_one(
                    {"_id": session_obj_id, "user_id": ObjectId(user_id)}
                )

                if not session:
                    logger.warning(
                        "chat_session_not_found", user_id=user_id, session_id=request.session_id
                    )
                    raise HTTPException(status_code=404, detail="Chat session not found")

                session_id = request.session_id

                # Update last_active_at
                await db.chat_sessions.update_one(
                    {"_id": session_obj_id}, {"$set": {"last_active_at": now}}
                )

            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                logger.error(
                    "invalid_session_id",
                    user_id=user_id,
                    session_id=request.session_id,
                    error=str(e),
                )
                raise HTTPException(status_code=400, detail="Invalid session ID format")
        else:
            # Create new session
            session_doc = {
                "user_id": ObjectId(user_id),
                "title": f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
                "created_at": now,
                "last_active_at": now,
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
            "created_at": now,
        }
        _user_msg_result = await db.chat_messages.insert_one(user_message_doc)

        # Retrieve relevant notes for RAG using hybrid search
        # keyword_weight=0.3, vector_weight=0.7 favors semantic understanding
        retrieval = NoteRetrieval(
            top_k=3, max_tokens_per_note=500, keyword_weight=0.3, vector_weight=0.7
        )
        relevant_notes = await retrieval.hybrid_retrieve(user_id=user_id, query=request.message)

        span.set_attribute("rag.notes_retrieved", len(relevant_notes))
        logger.info("notes_retrieved_for_rag", user_id=user_id, count=len(relevant_notes))

        # Get conversation history from this session (last 10 messages)
        history_cursor = (
            db.chat_messages.find({"session_id": session_obj_id}).sort("created_at", -1).limit(10)
        )
        history_messages = []
        async for msg in history_cursor:
            history_messages.append(msg)

        # Reverse to get chronological order
        history_messages.reverse()

        # Build messages for OpenAI
        messages = []

        # System message with RAG context
        if relevant_notes:
            context = retrieval.format_notes_for_context(relevant_notes)
            system_prompt = get_rag_system_prompt(context)
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": get_default_system_prompt()})

        # Add conversation history (exclude the message we just added)
        for msg in history_messages[:-1]:  # Skip the last one (current message)
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("openai_api_key_missing", user_id=user_id)
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        # Call OpenAI API
        try:
            async with OpenAIConnector(api_key=api_key) as connector:
                # Send messages with RAG context and conversation history
                completion = await connector.chat_completion(
                    messages=messages,
                    model=OpenAIModel.GPT_4O_MINI,
                    temperature=0.7,
                )

                response_text = completion.choices[0].message.content

                # Log token usage and cost
                if completion.usage:
                    span.set_attribute("openai.prompt_tokens", completion.usage.prompt_tokens)
                    span.set_attribute(
                        "openai.completion_tokens", completion.usage.completion_tokens
                    )
                    span.set_attribute("openai.total_tokens", completion.usage.total_tokens)

                    cost = connector.estimate_cost(
                        model=OpenAIModel.GPT_4O_MINI,
                        prompt_tokens=completion.usage.prompt_tokens,
                        completion_tokens=completion.usage.completion_tokens,
                    )
                    span.set_attribute("openai.estimated_cost_usd", cost)
                    logger.info(
                        "openai_completion_success",
                        user_id=user_id,
                        session_id=session_id,
                        prompt_tokens=completion.usage.prompt_tokens,
                        completion_tokens=completion.usage.completion_tokens,
                        cost_usd=cost,
                    )

        except Exception as e:
            logger.error("openai_api_error", user_id=user_id, session_id=session_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {e!s}")

        # Extract citations from response
        citations = retrieval.extract_citations_from_response(response_text, relevant_notes)

        span.set_attribute("rag.citations_found", len(citations))
        logger.info("citations_extracted", user_id=user_id, count=len(citations))

        # Store assistant message with citations
        assistant_message_doc = {
            "session_id": session_obj_id,
            "role": "assistant",
            "content": response_text,
            "citations": citations,
            "created_at": datetime.utcnow(),
        }
        assistant_msg_result = await db.chat_messages.insert_one(assistant_message_doc)
        assistant_msg_id = str(assistant_msg_result.inserted_id)

        metrics.chat_messages.add(1)

        logger.info(
            "chat_message_processed",
            user_id=user_id,
            session_id=session_id,
            message_id=assistant_msg_id,
        )

        # Convert citations to response format
        citation_responses = [
            Citation(
                note_id=str(cit["note_id"]),
                chunk_id=str(cit["chunk_id"]) if cit.get("chunk_id") else None,
                span=cit.get("span", {}),
            )
            for cit in citations
        ]

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=assistant_msg_id,
            citations=citation_responses,
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Process user input and stream the response in real-time.

    Requires authentication. If session_id is not provided, creates a new session.
    Streams response chunks as Server-Sent Events (SSE).
    Stores both user message and complete assistant response in chat_messages collection.
    """

    async def generate_stream():
        """Generate SSE stream for chat response."""
        with tracer.start_as_current_span("process_chat_stream") as span:
            user_id = str(current_user.get("_id"))
            user_name = current_user.get("name", "User")

            span.set_attribute("user.id", user_id)
            span.set_attribute("user.name", user_name)
            span.set_attribute("message.length", len(request.message))

            logger.info(
                "chat_stream_message_received", user_id=user_id, message_length=len(request.message)
            )

            db = get_db()
            now = datetime.utcnow()

            try:
                # Get or create session
                if request.session_id:
                    try:
                        session_obj_id = ObjectId(request.session_id)
                        session = await db.chat_sessions.find_one(
                            {"_id": session_obj_id, "user_id": ObjectId(user_id)}
                        )

                        if not session:
                            logger.warning(
                                "chat_session_not_found",
                                user_id=user_id,
                                session_id=request.session_id,
                            )
                            error_data = json.dumps(
                                {"type": "error", "error": "Chat session not found"}
                            )
                            yield f"data: {error_data}\n\n"
                            return

                        session_id = request.session_id

                        # Update last_active_at
                        await db.chat_sessions.update_one(
                            {"_id": session_obj_id}, {"$set": {"last_active_at": now}}
                        )

                    except Exception as e:
                        logger.error(
                            "invalid_session_id",
                            user_id=user_id,
                            session_id=request.session_id,
                            error=str(e),
                        )
                        error_data = json.dumps(
                            {"type": "error", "error": "Invalid session ID format"}
                        )
                        yield f"data: {error_data}\n\n"
                        return
                else:
                    # Create new session
                    session_doc = {
                        "user_id": ObjectId(user_id),
                        "title": f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
                        "created_at": now,
                        "last_active_at": now,
                    }
                    result = await db.chat_sessions.insert_one(session_doc)
                    session_id = str(result.inserted_id)
                    session_obj_id = result.inserted_id
                    logger.info("chat_session_auto_created", session_id=session_id, user_id=user_id)

                    # Send session info to client
                    session_data = json.dumps({"type": "session", "session_id": session_id})
                    yield f"data: {session_data}\n\n"

                span.set_attribute("session.id", session_id)

                # Store user message
                user_message_doc = {
                    "session_id": session_obj_id,
                    "role": "user",
                    "content": request.message,
                    "citations": [],
                    "created_at": now,
                }
                _user_msg_result = await db.chat_messages.insert_one(user_message_doc)

                # Retrieve relevant notes for RAG using hybrid search
                retrieval = NoteRetrieval(
                    top_k=3, max_tokens_per_note=500, keyword_weight=0.3, vector_weight=0.7
                )
                relevant_notes = await retrieval.hybrid_retrieve(
                    user_id=user_id, query=request.message
                )

                span.set_attribute("rag.notes_retrieved", len(relevant_notes))
                logger.info("notes_retrieved_for_rag", user_id=user_id, count=len(relevant_notes))

                # Get conversation history from this session (last 10 messages)
                history_cursor = (
                    db.chat_messages.find({"session_id": session_obj_id})
                    .sort("created_at", -1)
                    .limit(10)
                )
                history_messages = []
                async for msg in history_cursor:
                    history_messages.append(msg)

                # Reverse to get chronological order
                history_messages.reverse()

                # Build messages for OpenAI
                messages = []

                # System message with RAG context
                if relevant_notes:
                    context = retrieval.format_notes_for_context(relevant_notes)
                    system_prompt = get_rag_system_prompt(context)
                    messages.append({"role": "system", "content": system_prompt})
                else:
                    messages.append({"role": "system", "content": get_default_system_prompt()})

                # Add conversation history (exclude the message we just added)
                for msg in history_messages[:-1]:  # Skip the last one (current message)
                    messages.append({"role": msg["role"], "content": msg["content"]})

                # Add current user message
                messages.append({"role": "user", "content": request.message})

                # Get OpenAI API key from environment
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("openai_api_key_missing", user_id=user_id)
                    error_data = json.dumps(
                        {"type": "error", "error": "OpenAI API key not configured"}
                    )
                    yield f"data: {error_data}\n\n"
                    return

                # Call OpenAI API with streaming
                full_response = ""
                try:
                    async with OpenAIConnector(api_key=api_key) as connector:
                        # Send messages with streaming enabled
                        stream = await connector.chat_completion(
                            messages=messages,
                            model=OpenAIModel.GPT_4O_MINI,
                            temperature=0.7,
                            stream=True,
                        )

                        async for chunk in stream:
                            # Extract content from chunk
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    full_response += delta.content
                                    # Send chunk to client
                                    chunk_data = json.dumps(
                                        {"type": "content", "content": delta.content}
                                    )
                                    yield f"data: {chunk_data}\n\n"

                        # Log completion (note: streaming doesn't provide usage stats)
                        logger.info(
                            "openai_stream_completion_success",
                            user_id=user_id,
                            session_id=session_id,
                            response_length=len(full_response),
                        )

                except Exception as e:
                    logger.error(
                        "openai_api_error", user_id=user_id, session_id=session_id, error=str(e)
                    )
                    error_data = json.dumps(
                        {"type": "error", "error": f"Failed to generate response: {e!s}"}
                    )
                    yield f"data: {error_data}\n\n"
                    return

                # Extract citations from response
                citations = retrieval.extract_citations_from_response(full_response, relevant_notes)

                span.set_attribute("rag.citations_found", len(citations))
                logger.info("citations_extracted", user_id=user_id, count=len(citations))

                # Store assistant message with citations
                assistant_message_doc = {
                    "session_id": session_obj_id,
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations,
                    "created_at": datetime.utcnow(),
                }
                assistant_msg_result = await db.chat_messages.insert_one(assistant_message_doc)
                assistant_msg_id = str(assistant_msg_result.inserted_id)

                metrics.chat_messages.add(1)

                logger.info(
                    "chat_stream_message_processed",
                    user_id=user_id,
                    session_id=session_id,
                    message_id=assistant_msg_id,
                )

                # Send citations if any
                if citations:
                    citation_responses = [
                        {
                            "note_id": str(cit["note_id"]),
                            "chunk_id": str(cit["chunk_id"]) if cit.get("chunk_id") else None,
                            "span": cit.get("span", {}),
                        }
                        for cit in citations
                    ]
                    citations_data = json.dumps(
                        {"type": "citations", "citations": citation_responses}
                    )
                    yield f"data: {citations_data}\n\n"

                # Send completion event
                done_data = json.dumps(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "message_id": assistant_msg_id,
                    }
                )
                yield f"data: {done_data}\n\n"

            except Exception as e:
                logger.error("chat_stream_error", user_id=user_id, error=str(e))
                error_data = json.dumps({"type": "error", "error": str(e)})
                yield f"data: {error_data}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_chat_session(
    session_data: ChatSessionCreate, current_user: dict = Depends(get_current_user)
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
            "last_active_at": now,
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
            last_active_at=session_doc["last_active_at"],
        )


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_chat_sessions(
    current_user: dict = Depends(get_current_user), limit: int = 50, skip: int = 0
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
        cursor = (
            db.chat_sessions.find({"user_id": ObjectId(user_id)})
            .sort("last_active_at", -1)
            .skip(skip)
            .limit(limit)
        )

        sessions = []
        async for session in cursor:
            sessions.append(
                ChatSessionResponse(
                    id=str(session["_id"]),
                    user_id=user_id,
                    title=session["title"],
                    created_at=session["created_at"],
                    last_active_at=session["last_active_at"],
                )
            )

        span.set_attribute("result.count", len(sessions))

        logger.info("chat_sessions_listed", user_id=user_id, count=len(sessions))

        return sessions


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: str, current_user: dict = Depends(get_current_user), limit: int = 100, skip: int = 0
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
            session = await db.chat_sessions.find_one(
                {"_id": session_obj_id, "user_id": ObjectId(user_id)}
            )

            if not session:
                logger.warning("session_not_found", user_id=user_id, session_id=session_id)
                raise HTTPException(status_code=404, detail="Chat session not found")

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error("invalid_session_id", user_id=user_id, session_id=session_id, error=str(e))
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Query messages
        cursor = (
            db.chat_messages.find({"session_id": session_obj_id})
            .sort("created_at", 1)
            .skip(skip)
            .limit(limit)
        )

        messages = []
        async for msg in cursor:
            citations = [
                Citation(
                    note_id=str(cit["note_id"]),
                    chunk_id=str(cit["chunk_id"]) if cit.get("chunk_id") else None,
                    span=cit.get("span", {}),
                )
                for cit in msg.get("citations", [])
            ]

            messages.append(
                ChatMessageResponse(
                    id=str(msg["_id"]),
                    session_id=session_id,
                    role=msg["role"],
                    content=msg["content"],
                    citations=citations,
                    created_at=msg["created_at"],
                )
            )

        span.set_attribute("result.count", len(messages))

        logger.info(
            "session_messages_retrieved",
            user_id=user_id,
            session_id=session_id,
            count=len(messages),
        )

        return messages
