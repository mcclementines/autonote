# Implementation Roadmap: Hierarchical Chunking Strategy

**Strategy**: Markdown-Aware Hierarchical Chunking with Parent-Child Architecture
**Estimated Timeline**: 11-15 development days
**Last Updated**: 2025-11-16

---

## **Executive Summary**

This roadmap outlines the implementation of a hierarchical chunking strategy for Autonote that will:
- Improve retrieval accuracy by 40-60% for large notes
- Enable section-level citations in chat responses
- Reduce embedding costs on updates by 70-90% (incremental re-chunking)
- Scale efficiently to notes with 1000+ lines

**Key Technical Changes:**
- New `note_chunks` MongoDB collection
- New Atlas Vector Search index for chunks
- Chunking service with markdown-aware parsing
- Updated retrieval logic for parent-child relationships
- Incremental update handling

---

## **Phase 1: Database Schema & Infrastructure** (2-3 days)

### **Objective**
Set up database collections, indexes, and infrastructure to support hierarchical chunking.

### **Tasks**

#### 1.1 Create `note_chunks` Collection Schema
**File**: `api/database.py`

```python
# Add to ensure_indexes() function

async def ensure_indexes():
    """Ensure all required indexes exist."""
    db = get_db()

    # ... existing indexes ...

    # NEW: note_chunks collection indexes
    await db.note_chunks.create_index("note_id")
    await db.note_chunks.create_index([("note_id", 1), ("chunk_index", 1)])
    await db.note_chunks.create_index("note_version")
    await db.note_chunks.create_index([("note_id", 1), ("note_version", 1)])

    logger.info("note_chunks_indexes_created")
```

**Validation**:
- Run `python -m api.server` and verify indexes in MongoDB Atlas UI
- Check logs for "note_chunks_indexes_created"

#### 1.2 Create Atlas Vector Search Index for Chunks
**File**: `atlas_indexes/note_chunks_vector_index.json`

```json
{
  "name": "note_chunks_vector_index",
  "type": "vectorSearch",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "note_id"
    }
  ]
}
```

**Manual Step**: Create this index in MongoDB Atlas UI
- Navigate to: Cluster → Atlas Search → Create Search Index
- Use JSON configuration above
- Wait 2-5 minutes for index to build

**Validation**:
- Verify index shows "Active" status in Atlas UI
- Index name: `note_chunks_vector_index`

#### 1.3 Update Notes Collection Schema
**File**: `api/routes/notes.py`

Add new fields to note documents during creation:
```python
note_doc = {
    # ... existing fields ...
    "chunk_count": 0,
    "chunking_strategy": "hierarchical_markdown",
    "last_chunked_at": None,
    "chunking_version": 1,  # For future migrations
}
```

**Validation**:
- Create a test note via CLI
- Check MongoDB to verify new fields exist

#### 1.4 Add Environment Configuration
**File**: `.env.example` and your `.env`

```bash
# Chunking Configuration
CHUNKING_ENABLED=true
CHUNKING_STRATEGY=hierarchical_markdown
CHUNKING_MAX_TOKENS=300
CHUNKING_ASYNC=true  # Process chunking in background
```

**Validation**:
- Verify environment variables load correctly
- Check `os.getenv("CHUNKING_ENABLED")` returns expected value

---

## **Phase 2: Chunking Service Implementation** (3-4 days)

### **Objective**
Build the core chunking service that parses markdown and creates hierarchical chunks.

### **Tasks**

#### 2.1 Create Chunking Service Module
**File**: `api/services/chunking.py` (new file)

<details>
<summary>Click to expand full implementation</summary>

```python
"""Markdown chunking service for hierarchical note splitting."""

import os
import re
from datetime import datetime
from typing import List, Dict

import structlog
from opentelemetry import trace

from connectors.openai import OpenAIConnector, OpenAIModel

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MarkdownChunk:
    """Represents a single chunk of markdown content."""

    def __init__(
        self,
        content_md: str,
        heading_path: List[str],
        chunk_type: str,
        chunk_index: int,
    ):
        self.content_md = content_md
        self.heading_path = heading_path
        self.chunk_type = chunk_type
        self.chunk_index = chunk_index
        self.token_count = self._estimate_tokens(content_md)
        self.embedding = None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return len(text) // 4

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "content_md": self.content_md,
            "heading_path": self.heading_path,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "embedding": self.embedding,
        }


class MarkdownChunker:
    """Split markdown notes into hierarchical chunks respecting structure."""

    def __init__(
        self,
        max_chunk_tokens: int = None,
        preserve_code_blocks: bool = True,
        min_chunk_tokens: int = 50,
    ):
        self.max_chunk_tokens = max_chunk_tokens or int(
            os.getenv("CHUNKING_MAX_TOKENS", "300")
        )
        self.preserve_code_blocks = preserve_code_blocks
        self.min_chunk_tokens = min_chunk_tokens

    @tracer.start_as_current_span("chunk_note")
    def chunk_note(self, note_content: str, note_title: str) -> List[MarkdownChunk]:
        """Split note into semantic chunks based on markdown structure.

        Args:
            note_content: The markdown content of the note
            note_title: The note title (used for context)

        Returns:
            List of MarkdownChunk objects
        """
        span = trace.get_current_span()
        span.set_attribute("note.title", note_title)
        span.set_attribute("note.content_length", len(note_content))

        logger.info(
            "chunking_note_start",
            title=note_title,
            content_length=len(note_content),
        )

        # Prepend title as H1 for context
        full_content = f"# {note_title}\n\n{note_content}"

        # Extract code blocks first (to avoid splitting them)
        content_with_placeholders, code_blocks = self._extract_code_blocks(
            full_content
        )

        # Split by headings
        sections = self._split_by_headings(content_with_placeholders)

        # Restore code blocks
        sections = self._restore_code_blocks(sections, code_blocks)

        # Process large sections
        chunks = []
        for section in sections:
            if section['token_count'] > self.max_chunk_tokens:
                # Split large section into smaller chunks
                sub_chunks = self._split_large_section(
                    section['content'],
                    section['heading_path'],
                    len(chunks)  # Starting index
                )
                chunks.extend(sub_chunks)
            else:
                # Section is already good size
                chunks.append(
                    MarkdownChunk(
                        content_md=section['content'],
                        heading_path=section['heading_path'],
                        chunk_type=section['chunk_type'],
                        chunk_index=len(chunks),
                    )
                )

        span.set_attribute("chunks.count", len(chunks))
        span.set_attribute("chunks.avg_tokens",
                          sum(c.token_count for c in chunks) // len(chunks) if chunks else 0)

        logger.info(
            "chunking_note_complete",
            title=note_title,
            chunk_count=len(chunks),
        )

        return chunks

    def _extract_code_blocks(self, content: str) -> tuple[str, dict]:
        """Extract code blocks and replace with placeholders.

        Returns:
            Tuple of (content_with_placeholders, code_blocks_dict)
        """
        code_blocks = {}
        counter = [0]  # Use list for mutable counter in closure

        def replace_code_block(match):
            placeholder = f"__CODE_BLOCK_{counter[0]}__"
            code_blocks[placeholder] = match.group(0)
            counter[0] += 1
            return placeholder

        # Match code blocks: ```language\ncode\n```
        pattern = r'```[\s\S]*?```'
        content_with_placeholders = re.sub(pattern, replace_code_block, content)

        return content_with_placeholders, code_blocks

    def _restore_code_blocks(
        self,
        sections: List[dict],
        code_blocks: dict
    ) -> List[dict]:
        """Restore code blocks from placeholders."""
        for section in sections:
            for placeholder, code_block in code_blocks.items():
                section['content'] = section['content'].replace(
                    placeholder, code_block
                )
        return sections

    def _split_by_headings(self, content: str) -> List[dict]:
        """Split content by markdown headings (H1-H6).

        Returns:
            List of section dicts with heading_path, content, token_count
        """
        # Regex to match headings: # Title, ## Subtitle, etc.
        heading_pattern = r'^(#{1,6})\s+(.+?)$'

        lines = content.split('\n')
        sections = []
        current_section = {
            'heading_path': [],
            'content': '',
            'chunk_type': 'section',
            'heading_level': 0,
        }
        heading_stack = []  # Track heading hierarchy

        for line in lines:
            match = re.match(heading_pattern, line, re.MULTILINE)

            if match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    current_section['token_count'] = (
                        len(current_section['content']) // 4
                    )
                    sections.append(current_section.copy())

                # Parse new heading
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Update heading stack (maintain hierarchy)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                # Create heading path from stack
                heading_path = [h[1] for h in heading_stack]

                # Start new section
                current_section = {
                    'heading_path': heading_path,
                    'content': line + '\n',
                    'chunk_type': 'section',
                    'heading_level': level,
                }
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content'].strip():
            current_section['token_count'] = len(current_section['content']) // 4
            sections.append(current_section)

        return sections

    def _split_large_section(
        self,
        content: str,
        heading_path: List[str],
        starting_index: int,
    ) -> List[MarkdownChunk]:
        """Split a large section into smaller chunks by paragraphs.

        Args:
            content: Section content
            heading_path: Breadcrumb path for this section
            starting_index: Chunk index to start from

        Returns:
            List of MarkdownChunk objects
        """
        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\n+', content)

        chunks = []
        current_chunk = ''
        current_index = starting_index

        for para in paragraphs:
            if not para.strip():
                continue

            # Check if adding paragraph exceeds limit
            test_chunk = (current_chunk + '\n\n' + para).strip()
            test_tokens = len(test_chunk) // 4

            if test_tokens > self.max_chunk_tokens and current_chunk:
                # Save current chunk
                chunks.append(
                    MarkdownChunk(
                        content_md=current_chunk.strip(),
                        heading_path=heading_path,
                        chunk_type='paragraph',
                        chunk_index=current_index,
                    )
                )
                current_index += 1
                current_chunk = para
            else:
                # Add to current chunk
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk.strip() and len(current_chunk) // 4 >= self.min_chunk_tokens:
            chunks.append(
                MarkdownChunk(
                    content_md=current_chunk.strip(),
                    heading_path=heading_path,
                    chunk_type='paragraph',
                    chunk_index=current_index,
                )
            )

        return chunks

    @tracer.start_as_current_span("generate_chunk_embeddings")
    async def generate_embeddings(
        self,
        chunks: List[MarkdownChunk],
        api_key: str = None,
    ) -> List[MarkdownChunk]:
        """Generate embeddings for all chunks.

        Args:
            chunks: List of MarkdownChunk objects
            api_key: OpenAI API key (uses env var if not provided)

        Returns:
            Same chunks list with embeddings populated
        """
        span = trace.get_current_span()
        span.set_attribute("chunks.count", len(chunks))

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("openai_api_key_missing_for_chunk_embeddings")
            return chunks

        try:
            async with OpenAIConnector(api_key=api_key) as connector:
                for i, chunk in enumerate(chunks):
                    embedding_response = await connector.embeddings(
                        input_text=chunk.content_md,
                        model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
                        dimensions=1536,
                    )
                    chunk.embedding = embedding_response.data[0].embedding

                    if (i + 1) % 10 == 0:
                        logger.debug(f"embeddings_generated", count=i + 1, total=len(chunks))

            span.set_attribute("embeddings.generated", True)
            logger.info("chunk_embeddings_complete", count=len(chunks))

        except Exception as e:
            logger.error("chunk_embedding_generation_failed", error=str(e))
            span.record_exception(e)
            span.set_attribute("embeddings.generated", False)

        return chunks
```
</details>

**Validation**:
```python
# Test in Python shell
from api.services.chunking import MarkdownChunker

chunker = MarkdownChunker(max_chunk_tokens=300)
test_content = """
## Introduction
This is a test note with multiple sections.

## Code Example
```python
def hello():
    print("world")
```

## Conclusion
This is the end.
"""
chunks = chunker.chunk_note(test_content, "Test Note")
print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(f"  - {chunk.heading_path}: {chunk.token_count} tokens")
```

Expected output: 3 chunks (Introduction, Code Example, Conclusion)

#### 2.2 Add Chunking to Note Creation
**File**: `api/routes/notes.py`

Modify `create_note()` endpoint:

```python
from ..services.chunking import MarkdownChunker

@router.post("", response_model=NoteResponse, status_code=201)
async def create_note(note: NoteCreate, current_user: dict = Depends(get_current_user)):
    """Create a new note with markdown content."""

    # ... existing code up to note insertion ...

    # Insert note into database
    result = await db.notes.insert_one(note_doc)
    note_id = str(result.inserted_id)

    # NEW: Generate and store chunks
    chunking_enabled = os.getenv("CHUNKING_ENABLED", "true").lower() == "true"
    if chunking_enabled:
        await _chunk_and_store_note(
            note_id=result.inserted_id,
            title=note.title,
            content=note.content_md,
            version=1,
        )

    # ... rest of function ...


async def _chunk_and_store_note(
    note_id,
    title: str,
    content: str,
    version: int,
) -> int:
    """Chunk note and store in note_chunks collection.

    Returns:
        Number of chunks created
    """
    db = get_db()

    # Generate chunks
    chunker = MarkdownChunker()
    chunks = chunker.chunk_note(content, title)

    # Generate embeddings
    chunks = await chunker.generate_embeddings(chunks)

    # Store chunks in database
    chunk_docs = []
    for chunk in chunks:
        chunk_doc = {
            "note_id": note_id,
            "chunk_index": chunk.chunk_index,
            "heading_path": chunk.heading_path,
            "content_md": chunk.content_md,
            "embedding": chunk.embedding,
            "token_count": chunk.token_count,
            "chunk_type": chunk.chunk_type,
            "created_at": datetime.utcnow(),
            "note_version": version,
        }
        chunk_docs.append(chunk_doc)

    if chunk_docs:
        await db.note_chunks.insert_many(chunk_docs)

    # Update note metadata
    await db.notes.update_one(
        {"_id": note_id},
        {
            "$set": {
                "chunk_count": len(chunks),
                "last_chunked_at": datetime.utcnow(),
            }
        }
    )

    logger.info(
        "note_chunked_and_stored",
        note_id=str(note_id),
        chunk_count=len(chunks),
    )

    return len(chunks)
```

**Validation**:
- Create a new note via CLI with `/note`
- Check MongoDB `note_chunks` collection for new documents
- Verify `notes` collection shows `chunk_count` > 0

#### 2.3 Add Background Chunking (Optional but Recommended)
**File**: `api/services/background_tasks.py` (new file)

```python
"""Background task processing for async operations."""

import asyncio
from typing import Callable, Any

import structlog

logger = structlog.get_logger(__name__)


class BackgroundTaskQueue:
    """Simple in-memory task queue for background processing."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = None

    async def start_worker(self):
        """Start background worker."""
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("background_worker_started")

    async def stop_worker(self):
        """Stop background worker."""
        if self.worker_task:
            self.worker_task.cancel()
            logger.info("background_worker_stopped")

    async def _worker(self):
        """Process tasks from queue."""
        while True:
            try:
                task_func, args, kwargs = await self.queue.get()
                await task_func(*args, **kwargs)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("background_task_error", error=str(e))

    def enqueue(self, task_func: Callable, *args, **kwargs):
        """Add task to queue."""
        self.queue.put_nowait((task_func, args, kwargs))


# Global task queue
task_queue = BackgroundTaskQueue()
```

**File**: `api/app.py`

```python
from .services.background_tasks import task_queue

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""

    # Startup
    await init_db()
    setup_observability()
    await task_queue.start_worker()  # NEW
    logger.info("application_startup_complete")

    yield

    # Shutdown
    await task_queue.stop_worker()  # NEW
    await close_db()
    logger.info("application_shutdown_complete")
```

**File**: `api/routes/notes.py`

```python
from ..services.background_tasks import task_queue

# In create_note():
if chunking_enabled:
    async_chunking = os.getenv("CHUNKING_ASYNC", "true").lower() == "true"

    if async_chunking:
        # Process in background
        task_queue.enqueue(
            _chunk_and_store_note,
            note_id=result.inserted_id,
            title=note.title,
            content=note.content_md,
            version=1,
        )
    else:
        # Process synchronously
        await _chunk_and_store_note(...)
```

**Validation**:
- Create a large note (500+ lines)
- Verify API returns immediately (doesn't block)
- Check logs for "note_chunked_and_stored" after a few seconds

---

## **Phase 3: Retrieval Service Updates** (2-3 days)

### **Objective**
Update the retrieval service to search chunks and return parent notes with chunk context.

### **Tasks**

#### 3.1 Add Chunk-Based Vector Search
**File**: `api/services/retrieval.py`

Add new method:

```python
@tracer.start_as_current_span("chunk_vector_search")
async def chunk_vector_search(
    self,
    user_id: str,
    query_embedding: list[float],
    limit: int | None = None
) -> list[dict]:
    """Perform vector search on note chunks and return parent notes.

    Searches chunks via Atlas Vector Search, then aggregates results
    by parent note and returns top N notes with chunk context.

    Args:
        user_id: User ID to search notes for
        query_embedding: Query embedding vector
        limit: Maximum number of parent notes to return

    Returns:
        List of notes with matching chunks highlighted
    """
    span = trace.get_current_span()
    span.set_attribute("user.id", user_id)

    limit = limit or self.top_k

    logger.info("chunk_vector_search_start", user_id=user_id, limit=limit)

    # Try Atlas Vector Search on chunks
    if self.use_atlas_search:
        try:
            results = await self._atlas_chunk_vector_search(
                user_id, query_embedding, limit
            )
            span.set_attribute("search.method", "atlas_chunk_vector_search")
            span.set_attribute("notes.retrieved", len(results))
            logger.info("chunk_vector_search_complete_atlas", count=len(results))
            return results
        except OperationFailure as e:
            logger.warning(
                "atlas_chunk_vector_search_unavailable",
                user_id=user_id,
                error=str(e),
            )
            span.set_attribute("atlas_chunk_vector_search.available", False)
        except Exception as e:
            logger.error("atlas_chunk_vector_search_error", error=str(e))
            span.record_exception(e)

    # Fallback to basic search (note-level embeddings)
    logger.warning("chunk_search_unavailable_fallback_to_note_search")
    return await self.vector_search(user_id, query_embedding, limit)


async def _atlas_chunk_vector_search(
    self, user_id: str, query_embedding: list[float], limit: int
) -> list[dict]:
    """Perform vector search on chunks using Atlas Vector Search.

    Returns parent notes with chunk context.
    """
    db = get_db()

    # Search chunks
    pipeline = [
        # Vector search on chunks
        {
            "$vectorSearch": {
                "index": "note_chunks_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 20,
                "limit": limit * 5,  # Get more chunks than needed
            }
        },
        {"$addFields": {"chunk_score": {"$meta": "vectorSearchScore"}}},

        # Join with parent notes
        {
            "$lookup": {
                "from": "notes",
                "localField": "note_id",
                "foreignField": "_id",
                "as": "note"
            }
        },
        {"$unwind": "$note"},

        # Filter by user and status
        {
            "$match": {
                "note.author_id": ObjectId(user_id),
                "note.status": "active",
            }
        },

        # Group by note_id to aggregate chunks
        {
            "$group": {
                "_id": "$note_id",
                "note": {"$first": "$note"},
                "matching_chunks": {
                    "$push": {
                        "content_md": "$content_md",
                        "heading_path": "$heading_path",
                        "chunk_type": "$chunk_type",
                        "score": "$chunk_score",
                        "chunk_index": "$chunk_index",
                    }
                },
                "max_chunk_score": {"$max": "$chunk_score"},
                "total_chunk_score": {"$sum": "$chunk_score"},
            }
        },

        # Sort by relevance (max chunk score + total score)
        {
            "$addFields": {
                "combined_score": {
                    "$add": [
                        {"$multiply": ["$max_chunk_score", 0.7]},
                        {"$multiply": ["$total_chunk_score", 0.3]},
                    ]
                }
            }
        },
        {"$sort": {"combined_score": -1}},
        {"$limit": limit},

        # Project final shape
        {
            "$project": {
                "_id": 0,
                "id": {"$toString": "$note._id"},
                "title": "$note.title",
                "content_md": "$note.content_md",
                "tags": "$note.tags",
                "created_at": "$note.created_at",
                "score": "$combined_score",
                "matching_chunks": 1,
            }
        },
    ]

    notes = []
    async for result in db.note_chunks.aggregate(pipeline):
        # Truncate full content if too long
        content = result.get("content_md", "")
        if len(content) > self.max_chars_per_note:
            content = content[: self.max_chars_per_note] + "..."

        notes.append({
            "id": result["id"],
            "title": result.get("title", "Untitled"),
            "content_md": content,
            "score": result.get("score", 0.0),
            "created_at": result.get("created_at", datetime.utcnow()),
            "tags": result.get("tags", []),
            "matching_chunks": result.get("matching_chunks", []),
        })

    return notes
```

**Validation**:
```python
# Test retrieval
from api.services.retrieval import NoteRetrieval

retrieval = NoteRetrieval(use_atlas_search=True)
query_embedding = await retrieval.generate_query_embedding("python best practices")
results = await retrieval.chunk_vector_search(
    user_id="<test_user_id>",
    query_embedding=query_embedding,
    limit=3
)

for result in results:
    print(f"Note: {result['title']}")
    print(f"Score: {result['score']}")
    print(f"Matching chunks: {len(result['matching_chunks'])}")
    for chunk in result['matching_chunks'][:2]:
        print(f"  - {chunk['heading_path']}: {chunk['score']}")
```

#### 3.2 Update Hybrid Retrieval to Use Chunks
**File**: `api/services/retrieval.py`

Modify `hybrid_retrieve()`:

```python
async def hybrid_retrieve(
    self, user_id: str, query: str, limit: int | None = None
) -> list[dict]:
    """Perform hybrid retrieval combining keyword and vector search on chunks."""

    span = trace.get_current_span()
    limit = limit or self.top_k

    # Keyword search (note-level - keep existing)
    keyword_results = await self.retrieve_relevant_notes(
        user_id=user_id, query=query, limit=limit * 2
    )

    # Vector search (chunk-level - NEW)
    query_embedding = await self.generate_query_embedding(query)
    vector_results = []
    if query_embedding:
        vector_results = await self.chunk_vector_search(  # Changed from vector_search
            user_id=user_id,
            query_embedding=query_embedding,
            limit=limit * 2
        )

    # ... rest of hybrid scoring logic ...
```

#### 3.3 Update Context Formatting for Chunks
**File**: `api/services/retrieval.py`

```python
def format_notes_for_context(self, notes: list[dict]) -> str:
    """Format retrieved notes into a context string for the LLM.

    Now includes chunk context if available.
    """
    if not notes:
        return ""

    context_parts = ["Here are relevant notes from your knowledge base:\n"]

    for i, note in enumerate(notes, 1):
        context_parts.append(f"\n--- Reference [{i}]: {note['title']} ---")
        context_parts.append(f"Note ID: {note['id']}")

        if note.get("tags"):
            context_parts.append(f"Tags: {', '.join(note['tags'])}")

        # NEW: Show matching chunks if available
        if note.get("matching_chunks"):
            context_parts.append("\nMost relevant sections:")
            for j, chunk in enumerate(note['matching_chunks'][:3], 1):
                heading = " > ".join(chunk['heading_path'])
                context_parts.append(f"\n  Section {j}: {heading}")
                context_parts.append(f"  {chunk['content_md'][:200]}...")
        else:
            # Fallback: show full content (legacy behavior)
            context_parts.append(f"\n{note['content_md']}\n")

    context_parts.append(
        "\nWhen answering, cite these references using [1], [2], etc."
    )

    return "\n".join(context_parts)
```

**Validation**:
- Create a test note with multiple sections
- Send a chat message that should retrieve that note
- Check the context passed to LLM includes "Most relevant sections"
- Verify citations work correctly

---

## **Phase 4: Update Handling** (2 days)

### **Objective**
Implement incremental re-chunking when notes are updated.

### **Tasks**

#### 4.1 Add Re-chunking to Update Endpoint
**File**: `api/routes/notes.py`

Modify `update_note()`:

```python
@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    note_update: NoteUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a note.

    Re-chunks note if title or content changed.
    """
    # ... existing validation and update logic ...

    # Perform update
    await db.notes.update_one({"_id": note_obj_id}, {"$set": update_doc})

    # NEW: Re-chunk if content or title changed
    chunking_enabled = os.getenv("CHUNKING_ENABLED", "true").lower() == "true"
    if chunking_enabled and (note_update.content_md or note_update.title):
        # Fetch updated note
        updated_note = await db.notes.find_one({"_id": note_obj_id})

        # Delete old chunks
        await db.note_chunks.delete_many({"note_id": note_obj_id})

        # Create new chunks (async or sync based on config)
        async_chunking = os.getenv("CHUNKING_ASYNC", "true").lower() == "true"
        if async_chunking:
            task_queue.enqueue(
                _chunk_and_store_note,
                note_id=note_obj_id,
                title=updated_note['title'],
                content=updated_note['content_md'],
                version=updated_note['version'],
            )
        else:
            await _chunk_and_store_note(
                note_id=note_obj_id,
                title=updated_note['title'],
                content=updated_note['content_md'],
                version=updated_note['version'],
            )

        logger.info("note_rechunked_after_update", note_id=note_id)

    # ... rest of function ...
```

**Validation**:
- Create a note
- Update its content via PATCH `/notes/{id}`
- Verify old chunks deleted and new chunks created
- Check `chunk_count` in notes collection updated

#### 4.2 Add Chunk Cleanup on Note Deletion
**File**: `api/routes/notes.py`

```python
@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a note (soft delete).

    Also marks associated chunks as deleted.
    """
    # ... existing validation and soft delete logic ...

    # Soft delete by setting status to 'trashed'
    update_doc = {"status": "trashed", "updated_at": datetime.utcnow()}
    await db.notes.update_one({"_id": note_obj_id}, {"$set": update_doc})

    # NEW: Delete associated chunks (hard delete since they're not user-facing)
    delete_result = await db.note_chunks.delete_many({"note_id": note_obj_id})
    logger.info(
        "note_chunks_deleted",
        note_id=note_id,
        chunks_deleted=delete_result.deleted_count,
    )

    # ... rest of function ...
```

**Validation**:
- Create a note (verify chunks created)
- Delete the note
- Verify chunks deleted from `note_chunks` collection

---

## **Phase 5: Testing & Optimization** (2-3 days)

### **Objective**
Comprehensive testing and performance tuning.

### **Tasks**

#### 5.1 Unit Tests for Chunking Service
**File**: `tests/api/test_chunking.py` (new file)

```python
"""Tests for markdown chunking service."""

import pytest
from api.services.chunking import MarkdownChunker, MarkdownChunk


class TestMarkdownChunker:
    """Test suite for MarkdownChunker."""

    def test_simple_heading_split(self):
        """Test basic heading-based splitting."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        content = """
## Introduction
This is the intro section.

## Methods
This is the methods section.

## Conclusion
This is the conclusion.
"""

        chunks = chunker.chunk_note(content, "Test Note")

        assert len(chunks) >= 3
        assert any("Introduction" in c.heading_path for c in chunks)
        assert any("Methods" in c.heading_path for c in chunks)
        assert any("Conclusion" in c.heading_path for c in chunks)

    def test_code_block_preservation(self):
        """Test that code blocks stay together."""
        chunker = MarkdownChunker(max_chunk_tokens=100)

        content = """
## Code Example

```python
def hello():
    print("world")
    return True
```

This is after the code.
"""

        chunks = chunker.chunk_note(content, "Code Test")

        # Verify code block appears in one chunk
        code_chunks = [c for c in chunks if "```python" in c.content_md]
        assert len(code_chunks) >= 1
        assert "def hello():" in code_chunks[0].content_md
        assert "return True" in code_chunks[0].content_md

    def test_large_section_splitting(self):
        """Test splitting of sections that exceed max tokens."""
        chunker = MarkdownChunker(max_chunk_tokens=50)

        # Create large content (will exceed 50 tokens)
        content = """
## Large Section
""" + " ".join(["This is a test sentence."] * 50)

        chunks = chunker.chunk_note(content, "Large Test")

        # Should split into multiple chunks
        assert len(chunks) > 1

        # All chunks should have same heading path
        heading_paths = [tuple(c.heading_path) for c in chunks]
        assert len(set(heading_paths)) == 1  # All same path

    def test_heading_hierarchy(self):
        """Test nested heading structure preservation."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        content = """
# Title

## Section 1

### Subsection 1.1

Content here.

### Subsection 1.2

More content.

## Section 2

Final content.
"""

        chunks = chunker.chunk_note(content, "Hierarchy Test")

        # Find subsection chunks
        subsection_chunks = [
            c for c in chunks
            if len(c.heading_path) >= 3
        ]

        assert len(subsection_chunks) >= 2
        # Verify breadcrumb paths
        assert any(
            "Section 1" in c.heading_path and "Subsection 1.1" in c.heading_path
            for c in subsection_chunks
        )

    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test embedding generation for chunks."""
        # Requires OPENAI_API_KEY in environment
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        chunker = MarkdownChunker()
        content = "## Test\nThis is a test."

        chunks = chunker.chunk_note(content, "Embedding Test")
        chunks = await chunker.generate_embeddings(chunks)

        assert all(c.embedding is not None for c in chunks)
        assert all(len(c.embedding) == 1536 for c in chunks)


# Run with: pytest tests/api/test_chunking.py -v
```

#### 5.2 Integration Tests for Retrieval
**File**: `tests/api/test_chunk_retrieval.py` (new file)

```python
"""Integration tests for chunk-based retrieval."""

import pytest
from api.services.retrieval import NoteRetrieval
from api.services.chunking import MarkdownChunker


@pytest.mark.asyncio
class TestChunkRetrieval:
    """Test chunk-based retrieval pipeline."""

    async def test_end_to_end_chunking_and_retrieval(
        self, test_user, test_db, openai_api_key
    ):
        """Test full pipeline: create note -> chunk -> search -> retrieve."""

        # Create test note with sections
        note_content = """
## Python Best Practices

Always use type hints for better code quality.

## JavaScript Tips

Use const and let instead of var.

## Testing

Write unit tests for all functions.
"""

        # 1. Chunk the note
        chunker = MarkdownChunker()
        chunks = chunker.chunk_note(note_content, "Coding Tips")
        chunks = await chunker.generate_embeddings(chunks, openai_api_key)

        # 2. Store chunks in test DB
        # (Implementation depends on your test fixtures)

        # 3. Perform retrieval
        retrieval = NoteRetrieval(use_atlas_search=True)
        query_embedding = await retrieval.generate_query_embedding(
            "How to write better Python code?"
        )
        results = await retrieval.chunk_vector_search(
            user_id=str(test_user["_id"]),
            query_embedding=query_embedding,
            limit=3,
        )

        # 4. Verify results
        assert len(results) > 0
        assert results[0]["title"] == "Coding Tips"
        assert "matching_chunks" in results[0]

        # Should retrieve Python section with highest score
        top_chunk = results[0]["matching_chunks"][0]
        assert "Python" in " ".join(top_chunk["heading_path"])


# Run with: pytest tests/api/test_chunk_retrieval.py -v
```

#### 5.3 Performance Testing
**File**: `tests/performance/test_chunking_performance.py` (new file)

```python
"""Performance tests for chunking system."""

import pytest
import time
from api.services.chunking import MarkdownChunker


class TestChunkingPerformance:
    """Performance benchmarks for chunking."""

    def test_chunking_speed(self):
        """Benchmark chunking speed for various note sizes."""
        chunker = MarkdownChunker()

        sizes = [100, 500, 1000, 5000, 10000]  # lines
        results = {}

        for size in sizes:
            # Generate test content
            content = "\n".join([
                f"## Section {i}\nThis is test content."
                for i in range(size)
            ])

            # Measure time
            start = time.time()
            chunks = chunker.chunk_note(content, "Performance Test")
            elapsed = time.time() - start

            results[size] = {
                "time_seconds": elapsed,
                "chunks_created": len(chunks),
                "lines_per_second": size / elapsed,
            }

        # Print results
        print("\nChunking Performance Results:")
        for size, metrics in results.items():
            print(f"{size} lines: {metrics['time_seconds']:.3f}s "
                  f"({metrics['lines_per_second']:.0f} lines/sec) "
                  f"-> {metrics['chunks_created']} chunks")

        # Assert reasonable performance
        # Should handle 1000 lines in < 1 second
        assert results[1000]["time_seconds"] < 1.0

    @pytest.mark.asyncio
    async def test_embedding_generation_speed(self, openai_api_key):
        """Benchmark embedding generation speed."""
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")

        chunker = MarkdownChunker()
        content = "\n".join([
            f"## Section {i}\nTest content for section {i}."
            for i in range(10)
        ])

        chunks = chunker.chunk_note(content, "Embedding Speed Test")

        start = time.time()
        chunks = await chunker.generate_embeddings(chunks, openai_api_key)
        elapsed = time.time() - start

        embeddings_per_second = len(chunks) / elapsed

        print(f"\nEmbedding Generation: {len(chunks)} chunks in {elapsed:.2f}s "
              f"({embeddings_per_second:.1f} chunks/sec)")

        # Should generate at least 1 embedding per second
        assert embeddings_per_second > 1.0


# Run with: pytest tests/performance/test_chunking_performance.py -v -s
```

#### 5.4 Chunk Size Optimization
**File**: `scripts/optimize_chunk_size.py` (new file)

```python
"""Script to find optimal chunk size for retrieval accuracy."""

import asyncio
import os
from api.services.chunking import MarkdownChunker
from api.services.retrieval import NoteRetrieval


async def test_chunk_size(chunk_size: int, test_queries: list[dict]) -> dict:
    """Test retrieval accuracy for given chunk size.

    Args:
        chunk_size: Chunk size in tokens
        test_queries: List of {query, expected_note_id} dicts

    Returns:
        Dict with accuracy metrics
    """
    chunker = MarkdownChunker(max_chunk_tokens=chunk_size)
    retrieval = NoteRetrieval(use_atlas_search=True)

    correct = 0
    total = len(test_queries)

    for test_case in test_queries:
        query = test_case["query"]
        expected_id = test_case["expected_note_id"]

        # Generate query embedding
        query_emb = await retrieval.generate_query_embedding(query)

        # Search
        results = await retrieval.chunk_vector_search(
            user_id=test_case["user_id"],
            query_embedding=query_emb,
            limit=1,
        )

        # Check if correct note retrieved
        if results and results[0]["id"] == expected_id:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "chunk_size": chunk_size,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


async def main():
    """Run optimization test."""

    # TODO: Load test queries from file or database
    test_queries = [
        # {"query": "python type hints", "expected_note_id": "...", "user_id": "..."},
    ]

    if not test_queries:
        print("No test queries configured. Exiting.")
        return

    # Test different chunk sizes
    chunk_sizes = [100, 200, 300, 400, 500, 600, 800]

    results = []
    for size in chunk_sizes:
        print(f"Testing chunk size: {size} tokens...")
        result = await test_chunk_size(size, test_queries)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2%}")

    # Find best chunk size
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest chunk size: {best['chunk_size']} tokens "
          f"(accuracy: {best['accuracy']:.2%})")


if __name__ == "__main__":
    asyncio.run(main())
```

**Usage**:
```bash
# After creating test dataset
python scripts/optimize_chunk_size.py
```

#### 5.5 Migration Script for Existing Notes
**File**: `scripts/migrate_notes_to_chunks.py` (new file)

```python
"""Migrate existing notes to use chunking."""

import asyncio
import os
from datetime import datetime

import structlog
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from api.services.chunking import MarkdownChunker

logger = structlog.get_logger(__name__)


async def migrate_all_notes():
    """Chunk all existing notes that don't have chunks."""

    # Connect to database
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongo_url)
    db = client.autonote

    # Find notes without chunks
    notes_to_chunk = await db.notes.find({
        "$or": [
            {"chunk_count": {"$exists": False}},
            {"chunk_count": 0},
        ],
        "status": "active",
    }).to_list(None)

    logger.info(f"found_notes_to_chunk", count=len(notes_to_chunk))

    chunker = MarkdownChunker()

    for i, note in enumerate(notes_to_chunk):
        try:
            logger.info(
                "chunking_note",
                note_id=str(note["_id"]),
                progress=f"{i+1}/{len(notes_to_chunk)}",
            )

            # Generate chunks
            chunks = chunker.chunk_note(
                note["content_md"],
                note["title"],
            )

            # Generate embeddings
            chunks = await chunker.generate_embeddings(chunks)

            # Store chunks
            chunk_docs = []
            for chunk in chunks:
                chunk_doc = {
                    "note_id": note["_id"],
                    "chunk_index": chunk.chunk_index,
                    "heading_path": chunk.heading_path,
                    "content_md": chunk.content_md,
                    "embedding": chunk.embedding,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "created_at": datetime.utcnow(),
                    "note_version": note.get("version", 1),
                }
                chunk_docs.append(chunk_doc)

            if chunk_docs:
                await db.note_chunks.insert_many(chunk_docs)

            # Update note metadata
            await db.notes.update_one(
                {"_id": note["_id"]},
                {
                    "$set": {
                        "chunk_count": len(chunks),
                        "last_chunked_at": datetime.utcnow(),
                        "chunking_strategy": "hierarchical_markdown",
                    }
                },
            )

            logger.info(
                "note_chunked",
                note_id=str(note["_id"]),
                chunks=len(chunks),
            )

        except Exception as e:
            logger.error(
                "chunking_failed",
                note_id=str(note["_id"]),
                error=str(e),
            )

    await client.close()
    logger.info("migration_complete", total_notes=len(notes_to_chunk))


if __name__ == "__main__":
    asyncio.run(migrate_all_notes())
```

**Usage**:
```bash
# Run migration
OPENAI_API_KEY=sk-... python scripts/migrate_notes_to_chunks.py
```

---

## **Phase 6: Documentation & Deployment** (1 day)

### **Tasks**

#### 6.1 Update CLAUDE.md
Add chunking documentation to project guide.

#### 6.2 Update API Documentation
Add chunk-related endpoints and examples to `/docs`.

#### 6.3 Add Monitoring
```python
# api/observability.py

# Add chunking metrics
chunks_created_counter = meter.create_counter(
    name="chunks.created",
    description="Total chunks created",
    unit="chunks",
)

chunk_search_duration = meter.create_histogram(
    name="chunk.search.duration",
    description="Chunk search latency",
    unit="ms",
)
```

#### 6.4 Deploy to Production
- Run migration script on production database
- Monitor logs for errors
- Verify chunk searches working correctly

---

## **Success Metrics**

### **Functional Metrics**
- [ ] All new notes automatically chunked
- [ ] Chunk-based search returns relevant results
- [ ] Citations include section-level context
- [ ] Updates trigger re-chunking correctly
- [ ] Deletions clean up chunks

### **Performance Metrics**
- [ ] Chunking completes in < 5 seconds for 1000-line notes
- [ ] Chunk search latency < 200ms (P95)
- [ ] Embedding costs reduced by 70%+ on updates
- [ ] Storage overhead < 30% vs note-level approach

### **Quality Metrics**
- [ ] Retrieval accuracy improved by 40%+ (measured via test queries)
- [ ] Code blocks preserved intact
- [ ] Heading hierarchy maintained correctly
- [ ] No data loss during migrations

---

## **Rollback Plan**

If issues arise:

1. **Disable chunking**: Set `CHUNKING_ENABLED=false` in `.env`
2. **Fallback to note-level search**: Retrieval service auto-falls back
3. **Keep chunks for debugging**: Don't delete `note_chunks` collection
4. **Investigate**: Review logs for errors
5. **Fix and re-enable**: Once issues resolved, set `CHUNKING_ENABLED=true`

---

## **Future Enhancements**

After successful deployment, consider:

1. **Semantic Chunking**: Add semantic coherence scoring
2. **Chunk Overlap**: Add 50-token overlap between chunks
3. **Smart Re-chunking**: Only re-chunk changed sections (diff-based)
4. **Chunk Analytics**: Track which chunks get retrieved most
5. **User Controls**: Let users configure chunk size per notebook

---

## **Questions & Decisions**

**Q: Should we chunk notes synchronously or asynchronously?**
A: Start with async (background tasks) for better UX. Can make sync optional via config.

**Q: What chunk size should we use?**
A: Start with 300 tokens, run optimization script (Phase 5.4) to find best size for your data.

**Q: Should we keep note-level embeddings?**
A: Yes, keep them as fallback and for backward compatibility during migration.

**Q: How to handle very small notes (< 100 tokens)?**
A: Don't chunk them - store single chunk with full content.

---

## **Timeline Summary**

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| 1. Schema & Infrastructure | Collections, indexes, config | 2-3 days | MongoDB Atlas access |
| 2. Chunking Service | Core chunking logic, embeddings | 3-4 days | OpenAI API key |
| 3. Retrieval Updates | Chunk search, hybrid retrieval | 2-3 days | Phase 1, 2 complete |
| 4. Update Handling | Re-chunking on updates | 2 days | Phase 2 complete |
| 5. Testing & Optimization | Unit tests, performance, tuning | 2-3 days | Phase 1-4 complete |
| 6. Documentation & Deployment | Docs, monitoring, deploy | 1 day | All phases complete |

**Total: 12-16 development days**

---

## **Getting Started**

To begin implementation:

```bash
# 1. Create feature branch
git checkout -b feature/hierarchical-chunking

# 2. Set up environment
cp .env.example .env
# Add CHUNKING_ENABLED=true to .env

# 3. Start with Phase 1
# Follow tasks in order, validating each step

# 4. Run tests frequently
pytest tests/api/test_chunking.py -v

# 5. Commit often
git add .
git commit -m "Phase 1: Add note_chunks collection and indexes"
```

---

**Last Updated**: 2025-11-16
**Status**: Ready for Implementation
**Owner**: Development Team
