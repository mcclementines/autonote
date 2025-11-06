"""Notes endpoints."""

from datetime import datetime

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import get_db
from ..models import NoteCreate, NoteResponse
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
            "links_out": [],
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
