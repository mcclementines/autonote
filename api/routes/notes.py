"""Notes endpoints."""

import os
from datetime import UTC, datetime

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from connectors.openai import OpenAIConnector, OpenAIModel

from ..auth import get_current_user
from ..database import get_db
from ..models import NoteCreate, NoteListResponse, NoteResponse, NoteUpdate
from ..observability import get_tracer

# Initialize logger
logger = structlog.get_logger(__name__)

# Get tracer
tracer = get_tracer(__name__)

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", response_model=NoteResponse, status_code=201)
async def create_note(note: NoteCreate, current_user: dict = Depends(get_current_user)):
    """
    Create a new note with markdown content.

    Requires authentication. Supports multi-line markdown content,
    tags, and optional notebook assignment.
    """
    with tracer.start_as_current_span("create_note") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("note.title", note.title)
        span.set_attribute("note.tags_count", len(note.tags))

        logger.info("note_creation_attempt", user_id=user_id, title=note.title)

        db = get_db()

        # Calculate word count
        word_count = len(note.content_md.split())

        # Generate embedding for semantic search
        # Combine title and content for richer semantic representation
        embedding_text = f"{note.title}\n\n{note.content_md}"
        embedding_vector = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                async with OpenAIConnector(api_key=api_key) as connector:
                    embedding_response = await connector.embeddings(
                        input_text=embedding_text,
                        model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
                        dimensions=1536,  # Standard dimension for text-embedding-3-small
                    )
                    embedding_vector = embedding_response.data[0].embedding
                    span.set_attribute("embedding.generated", True)
                    logger.info("embedding_generated", user_id=user_id, title=note.title)
            except Exception as e:
                # Don't fail note creation if embedding fails
                logger.warning(
                    "embedding_generation_failed", user_id=user_id, title=note.title, error=str(e)
                )
                span.set_attribute("embedding.generated", False)
        else:
            logger.warning("openai_api_key_missing_for_embedding", user_id=user_id)
            span.set_attribute("embedding.generated", False)

        # Validate notebook_id if provided
        notebook_obj_id = None
        if note.notebook_id:
            try:
                notebook_obj_id = ObjectId(note.notebook_id)
                # Verify notebook exists and belongs to user
                notebook = await db.notebooks.find_one(
                    {"_id": notebook_obj_id, "author_id": ObjectId(user_id)}
                )
                if not notebook:
                    logger.warning(
                        "note_creation_failed_invalid_notebook",
                        user_id=user_id,
                        notebook_id=note.notebook_id,
                    )
                    raise HTTPException(status_code=404, detail="Notebook not found")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                logger.error(
                    "note_creation_failed_invalid_notebook_id",
                    user_id=user_id,
                    notebook_id=note.notebook_id,
                    error=str(e),
                )
                raise HTTPException(status_code=400, detail="Invalid notebook ID format")

        # Create note document
        now = datetime.now(UTC)
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
            "links_out": [],
            "embedding": embedding_vector,  # Store embedding for vector search
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
            links_out=[],
        )

        logger.info(
            "note_created_successfully", note_id=note_id, user_id=user_id, word_count=word_count
        )

        return note_response


@router.get("", response_model=NoteListResponse)
async def list_notes(
    status: str | None = None,
    limit: int = 100,
    skip: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """
    List all notes for the authenticated user.

    By default, only shows active notes. Optionally filter by status (active, archived, trashed).
    Supports pagination with limit and skip parameters.
    """
    with tracer.start_as_current_span("list_notes") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("query.limit", limit)
        span.set_attribute("query.skip", skip)

        db = get_db()

        # Build query - default to "active" notes only
        query = {"author_id": ObjectId(user_id)}
        if status:
            if status not in ["active", "archived", "trashed"]:
                raise HTTPException(status_code=400, detail="Invalid status value")
            query["status"] = status
            span.set_attribute("query.status", status)
        else:
            # Default to showing only active notes
            query["status"] = "active"
            span.set_attribute("query.status", "active")

        # Get total count
        total = await db.notes.count_documents(query)

        # Fetch notes with pagination, sorted by updated_at descending
        cursor = db.notes.find(query).sort("updated_at", -1).skip(skip).limit(limit)
        notes_docs = await cursor.to_list(length=limit)

        # Convert to response models
        notes = []
        for doc in notes_docs:
            notes.append(
                NoteResponse(
                    id=str(doc["_id"]),
                    notebook_id=str(doc["notebook_id"]) if doc.get("notebook_id") else None,
                    author_id=str(doc["author_id"]),
                    title=doc["title"],
                    content_md=doc["content_md"],
                    tags=doc.get("tags", []),
                    status=doc["status"],
                    pinned=doc.get("pinned", False),
                    created_at=doc["created_at"],
                    updated_at=doc["updated_at"],
                    version=doc.get("version", 1),
                    word_count=doc.get("word_count", 0),
                    links_out=doc.get("links_out", []),
                )
            )

        span.set_attribute("notes.count", len(notes))
        span.set_attribute("notes.total", total)

        logger.info("notes_listed", user_id=user_id, count=len(notes), total=total)

        return NoteListResponse(notes=notes, total=total)


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(note_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieve a specific note by ID.

    Requires authentication and ownership of the note.
    """
    with tracer.start_as_current_span("get_note") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("note.id", note_id)

        db = get_db()

        # Validate note_id format
        try:
            note_obj_id = ObjectId(note_id)
        except Exception:
            logger.warning("get_note_invalid_id", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=400, detail="Invalid note ID format")

        # Fetch note
        note_doc = await db.notes.find_one({"_id": note_obj_id, "author_id": ObjectId(user_id)})

        if not note_doc:
            logger.warning("get_note_not_found", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=404, detail="Note not found")

        # Convert to response model
        note_response = NoteResponse(
            id=str(note_doc["_id"]),
            notebook_id=str(note_doc["notebook_id"]) if note_doc.get("notebook_id") else None,
            author_id=str(note_doc["author_id"]),
            title=note_doc["title"],
            content_md=note_doc["content_md"],
            tags=note_doc.get("tags", []),
            status=note_doc["status"],
            pinned=note_doc.get("pinned", False),
            created_at=note_doc["created_at"],
            updated_at=note_doc["updated_at"],
            version=note_doc.get("version", 1),
            word_count=note_doc.get("word_count", 0),
            links_out=note_doc.get("links_out", []),
        )

        logger.info("note_retrieved", user_id=user_id, note_id=note_id)

        return note_response


@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str, note_update: NoteUpdate, current_user: dict = Depends(get_current_user)
):
    """
    Update a note.

    Supports partial updates - only provided fields will be updated.
    Increments version number on each update.
    """
    with tracer.start_as_current_span("update_note") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("note.id", note_id)

        db = get_db()

        # Validate note_id format
        try:
            note_obj_id = ObjectId(note_id)
        except Exception:
            logger.warning("update_note_invalid_id", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=400, detail="Invalid note ID format")

        # Check if note exists and belongs to user
        existing_note = await db.notes.find_one(
            {"_id": note_obj_id, "author_id": ObjectId(user_id)}
        )

        if not existing_note:
            logger.warning("update_note_not_found", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=404, detail="Note not found")

        # Build update document with only provided fields
        update_doc = {}
        if note_update.title is not None:
            update_doc["title"] = note_update.title
            span.set_attribute("note.title_updated", True)

        if note_update.content_md is not None:
            update_doc["content_md"] = note_update.content_md
            # Recalculate word count
            update_doc["word_count"] = len(note_update.content_md.split())
            span.set_attribute("note.content_updated", True)

        if note_update.tags is not None:
            update_doc["tags"] = note_update.tags
            span.set_attribute("note.tags_updated", True)

        if note_update.pinned is not None:
            update_doc["pinned"] = note_update.pinned
            span.set_attribute("note.pinned_updated", True)

        if note_update.status is not None:
            update_doc["status"] = note_update.status
            span.set_attribute("note.status_updated", True)

        # If nothing to update, return current note
        if not update_doc:
            logger.info("update_note_no_changes", user_id=user_id, note_id=note_id)
            return NoteResponse(
                id=str(existing_note["_id"]),
                notebook_id=(
                    str(existing_note["notebook_id"]) if existing_note.get("notebook_id") else None
                ),
                author_id=str(existing_note["author_id"]),
                title=existing_note["title"],
                content_md=existing_note["content_md"],
                tags=existing_note.get("tags", []),
                status=existing_note["status"],
                pinned=existing_note.get("pinned", False),
                created_at=existing_note["created_at"],
                updated_at=existing_note["updated_at"],
                version=existing_note.get("version", 1),
                word_count=existing_note.get("word_count", 0),
                links_out=existing_note.get("links_out", []),
            )

        # Update timestamp and increment version
        update_doc["updated_at"] = datetime.now(UTC)
        update_doc["version"] = existing_note.get("version", 1) + 1

        # Perform update
        await db.notes.update_one({"_id": note_obj_id}, {"$set": update_doc})

        # Fetch updated note
        updated_note = await db.notes.find_one({"_id": note_obj_id})

        # Convert to response model
        note_response = NoteResponse(
            id=str(updated_note["_id"]),
            notebook_id=str(updated_note["notebook_id"])
            if updated_note.get("notebook_id")
            else None,
            author_id=str(updated_note["author_id"]),
            title=updated_note["title"],
            content_md=updated_note["content_md"],
            tags=updated_note.get("tags", []),
            status=updated_note["status"],
            pinned=updated_note.get("pinned", False),
            created_at=updated_note["created_at"],
            updated_at=updated_note["updated_at"],
            version=updated_note.get("version", 1),
            word_count=updated_note.get("word_count", 0),
            links_out=updated_note.get("links_out", []),
        )

        logger.info("note_updated_successfully", user_id=user_id, note_id=note_id)

        return note_response


@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: str, current_user: dict = Depends(get_current_user)):
    """
    Delete a note (soft delete by setting status to 'trashed').

    Requires authentication and ownership of the note.
    The note is not permanently deleted - it's moved to trash status.
    """
    with tracer.start_as_current_span("delete_note") as span:
        user_id = str(current_user.get("_id"))

        span.set_attribute("user.id", user_id)
        span.set_attribute("note.id", note_id)

        db = get_db()

        # Validate note_id format
        try:
            note_obj_id = ObjectId(note_id)
        except Exception:
            logger.warning("delete_note_invalid_id", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=400, detail="Invalid note ID format")

        # Check if note exists and belongs to user
        existing_note = await db.notes.find_one(
            {"_id": note_obj_id, "author_id": ObjectId(user_id)}
        )

        if not existing_note:
            logger.warning("delete_note_not_found", user_id=user_id, note_id=note_id)
            raise HTTPException(status_code=404, detail="Note not found")

        # Soft delete by setting status to 'trashed'
        update_doc = {"status": "trashed", "updated_at": datetime.now(UTC)}

        await db.notes.update_one({"_id": note_obj_id}, {"$set": update_doc})

        span.set_attribute("note.deleted", True)
        logger.info("note_deleted_successfully", user_id=user_id, note_id=note_id)

        # 204 No Content - no response body needed
        return
