"""Notes-related Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LinkOut(BaseModel):
    """Model for note links (backlinks support)."""

    note_id: str
    type: str = "wiki"


class NoteCreate(BaseModel):
    """Request model for creating a note."""

    notebook_id: str | None = None
    title: str = Field(..., min_length=1, max_length=500)
    content_md: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False


class NoteResponse(BaseModel):
    """Response model for note data."""

    id: str
    notebook_id: str | None = None
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
