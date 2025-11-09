"""Note retrieval service for RAG (Retrieval-Augmented Generation).

This module provides hybrid retrieval of user notes from MongoDB using:
1. Keyword-based full-text search for exact matches (via Atlas Search or $text)
2. Vector similarity search for semantic understanding (via Atlas Vector Search)
3. Hybrid scoring to combine and rank results

Retrieved notes are used to augment LLM responses with relevant context
from the user's personal knowledge base.

Atlas Search Features (when available):
- Native vector indexing with HNSW algorithm (100-1000x faster)
- Advanced text search with better relevance scoring
- Fuzzy matching and autocomplete
- Single-pipeline hybrid search
- Scalable to millions of documents

Falls back to basic MongoDB operations when Atlas Search is not available.
"""

import os
from datetime import datetime

import structlog
from bson import ObjectId
from opentelemetry import trace
from pymongo.errors import OperationFailure

from connectors.openai import OpenAIConnector, OpenAIModel

from ..database import get_db

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class NoteRetrieval:
    """Service for retrieving relevant notes for RAG using hybrid search."""

    def __init__(
        self,
        top_k: int = 3,
        max_tokens_per_note: int = 500,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        use_atlas_search: bool | None = None,
    ):
        """Initialize retrieval service.

        Args:
            top_k: Number of top relevant notes to retrieve
            max_tokens_per_note: Maximum token estimate per note (rough: ~4 chars per token)
            keyword_weight: Weight for keyword search scores (0-1)
            vector_weight: Weight for vector similarity scores (0-1)
            use_atlas_search: Force Atlas Search on/off. If None, auto-detect.
        """
        self.top_k = top_k
        self.max_tokens_per_note = max_tokens_per_note
        self.max_chars_per_note = max_tokens_per_note * 4  # Rough estimate
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight

        # Determine if Atlas Search should be used
        if use_atlas_search is None:
            # Auto-detect from environment
            self.use_atlas_search = os.getenv("MONGODB_ATLAS_SEARCH", "false").lower() == "true"
        else:
            self.use_atlas_search = use_atlas_search

        # Track whether we've detected Atlas availability
        self._atlas_available_checked = False
        self._atlas_vector_available = False
        self._atlas_text_available = False

    @tracer.start_as_current_span("retrieve_notes")
    async def retrieve_relevant_notes(
        self, user_id: str, query: str, limit: int | None = None
    ) -> list[dict]:
        """Retrieve relevant notes using keyword search.

        Uses Atlas Search (if available) or MongoDB's $text search as fallback.
        Returns notes with metadata for citation tracking.

        Args:
            user_id: User ID to search notes for
            query: Search query (typically the user's message)
            limit: Override default top_k limit

        Returns:
            List of note dicts with id, title, content_md, score, created_at
        """
        span = trace.get_current_span()
        span.set_attribute("user.id", user_id)
        span.set_attribute("query.length", len(query))

        limit = limit or self.top_k

        logger.info("retrieving_notes", user_id=user_id, query_length=len(query), limit=limit)

        # Try Atlas Search first if enabled
        if self.use_atlas_search:
            try:
                notes = await self._atlas_text_search(user_id, query, limit)
                span.set_attribute("search.method", "atlas_search")
                span.set_attribute("notes.retrieved", len(notes))
                logger.info(
                    "notes_retrieved_atlas",
                    user_id=user_id,
                    count=len(notes),
                    query_length=len(query),
                )
                return notes
            except OperationFailure as e:
                # Atlas Search not available, fall back
                logger.warning(
                    "atlas_search_unavailable_fallback",
                    user_id=user_id,
                    error=str(e),
                )
                span.set_attribute("atlas_search.available", False)
            except Exception as e:
                logger.error("atlas_search_error", user_id=user_id, error=str(e))
                span.record_exception(e)

        # Fallback to basic $text search
        try:
            notes = await self._basic_text_search(user_id, query, limit)
            span.set_attribute("search.method", "text_search")
            span.set_attribute("notes.retrieved", len(notes))
            logger.info(
                "notes_retrieved_basic", user_id=user_id, count=len(notes), query_length=len(query)
            )
            return notes

        except Exception as e:
            logger.error("note_retrieval_error", user_id=user_id, error=str(e))
            span.record_exception(e)
            # Don't fail the whole request if retrieval fails
            # Just return empty results and let chat continue
            return []

    async def _atlas_text_search(self, user_id: str, query: str, limit: int) -> list[dict]:
        """Perform text search using Atlas Search.

        Args:
            user_id: User ID to search notes for
            query: Search query
            limit: Maximum number of results

        Returns:
            List of notes with scores

        Raises:
            OperationFailure: If Atlas Search index doesn't exist
        """
        db = get_db()

        pipeline = [
            {
                "$search": {
                    "index": "notes_search_index",
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": query,
                                    "path": ["title", "content_md"],
                                    "score": {"boost": {"path": "title", "value": 2.0}},
                                }
                            }
                        ],
                        "filter": [
                            {"equals": {"path": "author_id", "value": ObjectId(user_id)}},
                            {"equals": {"path": "status", "value": "active"}},
                        ],
                    },
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "content_md": 1,
                    "tags": 1,
                    "created_at": 1,
                    "score": 1,
                }
            },
        ]

        notes = []
        async for note in db.notes.aggregate(pipeline):
            content = note.get("content_md", "")
            if len(content) > self.max_chars_per_note:
                content = content[: self.max_chars_per_note] + "..."

            notes.append(
                {
                    "id": str(note["_id"]),
                    "title": note.get("title", "Untitled"),
                    "content_md": content,
                    "score": note.get("score", 0.0),
                    "created_at": note.get("created_at", datetime.utcnow()),
                    "tags": note.get("tags", []),
                }
            )

        return notes

    async def _basic_text_search(self, user_id: str, query: str, limit: int) -> list[dict]:
        """Perform text search using MongoDB $text operator.

        Args:
            user_id: User ID to search notes for
            query: Search query
            limit: Maximum number of results

        Returns:
            List of notes with scores
        """
        db = get_db()

        cursor = (
            db.notes.find(
                {
                    "$text": {"$search": query},
                    "author_id": ObjectId(user_id),
                    "status": "active",
                },
                {"score": {"$meta": "textScore"}},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

        notes = []
        async for note in cursor:
            content = note.get("content_md", "")
            if len(content) > self.max_chars_per_note:
                content = content[: self.max_chars_per_note] + "..."

            notes.append(
                {
                    "id": str(note["_id"]),
                    "title": note.get("title", "Untitled"),
                    "content_md": content,
                    "score": note.get("score", 0.0),
                    "created_at": note.get("created_at", datetime.utcnow()),
                    "tags": note.get("tags", []),
                }
            )

        return notes

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))

        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    @tracer.start_as_current_span("generate_query_embedding")
    async def generate_query_embedding(self, query: str) -> list[float] | None:
        """Generate embedding vector for search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector or None if generation fails
        """
        span = trace.get_current_span()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.warning("openai_api_key_missing_for_query_embedding")
            return None

        try:
            async with OpenAIConnector(api_key=api_key) as connector:
                embedding_response = await connector.embeddings(
                    input_text=query,
                    model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
                    dimensions=1536,
                )
                embedding = embedding_response.data[0].embedding
                span.set_attribute("query_embedding.generated", True)
                return embedding
        except Exception as e:
            logger.error("query_embedding_generation_failed", error=str(e))
            span.record_exception(e)
            span.set_attribute("query_embedding.generated", False)
            return None

    @tracer.start_as_current_span("vector_search")
    async def vector_search(
        self, user_id: str, query_embedding: list[float], limit: int | None = None
    ) -> list[dict]:
        """Perform vector similarity search on notes.

        Uses Atlas Vector Search (if available) or in-memory cosine similarity as fallback.

        Args:
            user_id: User ID to search notes for
            query_embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of notes with similarity scores
        """
        span = trace.get_current_span()
        span.set_attribute("user.id", user_id)

        limit = limit or self.top_k * 2  # Get more for hybrid ranking

        logger.info("vector_search_start", user_id=user_id, limit=limit)

        # Try Atlas Vector Search first if enabled
        if self.use_atlas_search:
            try:
                results = await self._atlas_vector_search(user_id, query_embedding, limit)
                span.set_attribute("search.method", "atlas_vector_search")
                span.set_attribute("vector_search.results", len(results))
                logger.info("vector_search_complete_atlas", user_id=user_id, count=len(results))
                return results
            except OperationFailure as e:
                # Atlas Vector Search not available, fall back
                logger.warning(
                    "atlas_vector_search_unavailable_fallback",
                    user_id=user_id,
                    error=str(e),
                )
                span.set_attribute("atlas_vector_search.available", False)
            except Exception as e:
                logger.error("atlas_vector_search_error", user_id=user_id, error=str(e))
                span.record_exception(e)

        # Fallback to in-memory cosine similarity
        try:
            results = await self._basic_vector_search(user_id, query_embedding, limit)
            span.set_attribute("search.method", "cosine_similarity")
            span.set_attribute("vector_search.results", len(results))
            logger.info("vector_search_complete_basic", user_id=user_id, count=len(results))
            return results

        except Exception as e:
            logger.error("vector_search_error", user_id=user_id, error=str(e))
            span.record_exception(e)
            return []

    async def _atlas_vector_search(
        self, user_id: str, query_embedding: list[float], limit: int
    ) -> list[dict]:
        """Perform vector search using Atlas Vector Search.

        Args:
            user_id: User ID to search notes for
            query_embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of notes with similarity scores

        Raises:
            OperationFailure: If Atlas Vector Search index doesn't exist
        """
        db = get_db()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "notes_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,  # Overrequest for better recall
                    "limit": limit,
                    "filter": {
                        "author_id": ObjectId(user_id),
                        "status": "active",
                    },
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "content_md": 1,
                    "tags": 1,
                    "created_at": 1,
                    "score": 1,
                }
            },
        ]

        notes_with_similarity = []
        async for note in db.notes.aggregate(pipeline):
            content = note.get("content_md", "")
            if len(content) > self.max_chars_per_note:
                content = content[: self.max_chars_per_note] + "..."

            notes_with_similarity.append(
                {
                    "id": str(note["_id"]),
                    "title": note.get("title", "Untitled"),
                    "content_md": content,
                    "score": note.get("score", 0.0),
                    "created_at": note.get("created_at", datetime.utcnow()),
                    "tags": note.get("tags", []),
                }
            )

        return notes_with_similarity

    async def _basic_vector_search(
        self, user_id: str, query_embedding: list[float], limit: int
    ) -> list[dict]:
        """Perform vector search using in-memory cosine similarity.

        WARNING: This method does not scale well. Use Atlas Vector Search for production.

        Args:
            user_id: User ID to search notes for
            query_embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of notes with similarity scores
        """
        db = get_db()

        # Find all active notes with embeddings for this user
        cursor = db.notes.find(
            {
                "author_id": ObjectId(user_id),
                "status": "active",
                "embedding": {"$exists": True, "$ne": None},
            }
        )

        notes_with_similarity = []
        async for note in cursor:
            note_embedding = note.get("embedding")
            if not note_embedding:
                continue

            # Calculate cosine similarity
            similarity = self.cosine_similarity(query_embedding, note_embedding)

            # Truncate content if too long
            content = note.get("content_md", "")
            if len(content) > self.max_chars_per_note:
                content = content[: self.max_chars_per_note] + "..."

            notes_with_similarity.append(
                {
                    "id": str(note["_id"]),
                    "title": note.get("title", "Untitled"),
                    "content_md": content,
                    "score": similarity,
                    "created_at": note.get("created_at", datetime.utcnow()),
                    "tags": note.get("tags", []),
                }
            )

        # Sort by similarity score (highest first)
        notes_with_similarity.sort(key=lambda x: x["score"], reverse=True)

        # Take top N and return
        return notes_with_similarity[:limit]

    @tracer.start_as_current_span("hybrid_retrieve")
    async def hybrid_retrieve(
        self, user_id: str, query: str, limit: int | None = None
    ) -> list[dict]:
        """Perform hybrid retrieval combining keyword and vector search.

        Args:
            user_id: User ID to search notes for
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of deduplicated and re-ranked notes
        """
        span = trace.get_current_span()
        span.set_attribute("user.id", user_id)

        limit = limit or self.top_k

        # Perform keyword search
        keyword_results = await self.retrieve_relevant_notes(
            user_id=user_id, query=query, limit=limit * 2
        )

        # Generate query embedding and perform vector search
        query_embedding = await self.generate_query_embedding(query)
        vector_results = []
        if query_embedding:
            vector_results = await self.vector_search(
                user_id=user_id, query_embedding=query_embedding, limit=limit * 2
            )
        else:
            logger.warning("hybrid_retrieve_fallback_to_keyword", user_id=user_id)
            # Fallback to keyword-only if embedding fails
            return keyword_results[:limit]

        # Combine results with hybrid scoring
        note_scores = {}

        # Normalize and add keyword scores
        max_keyword_score = max([n["score"] for n in keyword_results], default=1.0)
        for note in keyword_results:
            note_id = note["id"]
            normalized_score = note["score"] / max_keyword_score if max_keyword_score > 0 else 0
            note_scores[note_id] = {
                "note": note,
                "keyword_score": normalized_score,
                "vector_score": 0.0,
            }

        # Normalize and add vector scores
        max_vector_score = max([n["score"] for n in vector_results], default=1.0)
        for note in vector_results:
            note_id = note["id"]
            normalized_score = note["score"] / max_vector_score if max_vector_score > 0 else 0

            if note_id in note_scores:
                note_scores[note_id]["vector_score"] = normalized_score
            else:
                note_scores[note_id] = {
                    "note": note,
                    "keyword_score": 0.0,
                    "vector_score": normalized_score,
                }

        # Calculate hybrid scores
        ranked_notes = []
        for _note_id, data in note_scores.items():
            hybrid_score = (
                self.keyword_weight * data["keyword_score"]
                + self.vector_weight * data["vector_score"]
            )
            note_data = data["note"].copy()
            note_data["score"] = hybrid_score
            note_data["keyword_score"] = data["keyword_score"]
            note_data["vector_score"] = data["vector_score"]
            ranked_notes.append(note_data)

        # Sort by hybrid score and return top N
        ranked_notes.sort(key=lambda x: x["score"], reverse=True)
        results = ranked_notes[:limit]

        span.set_attribute("hybrid.keyword_results", len(keyword_results))
        span.set_attribute("hybrid.vector_results", len(vector_results))
        span.set_attribute("hybrid.final_results", len(results))

        logger.info(
            "hybrid_retrieve_complete",
            user_id=user_id,
            keyword_count=len(keyword_results),
            vector_count=len(vector_results),
            final_count=len(results),
        )

        return results

    def format_notes_for_context(self, notes: list[dict]) -> str:
        """Format retrieved notes into a context string for the LLM.

        Args:
            notes: List of note dicts from retrieve_relevant_notes

        Returns:
            Formatted string to include in system/user context
        """
        if not notes:
            return ""

        context_parts = ["Here are relevant notes from your knowledge base:\n"]

        for i, note in enumerate(notes, 1):
            context_parts.append(f"\n--- Reference [{i}]: {note['title']} ---")
            context_parts.append(f"Note ID: {note['id']}")
            if note.get("tags"):
                context_parts.append(f"Tags: {', '.join(note['tags'])}")
            context_parts.append(f"\n{note['content_md']}\n")

        context_parts.append(
            "\nWhen answering, cite these references when relevant using [1], [2], etc."
        )

        return "\n".join(context_parts)

    def extract_citations_from_response(self, response_text: str, notes: list[dict]) -> list[dict]:
        """Extract citations from LLM response.

        Simple extraction: looks for [1], [2], etc. in the response and maps
        them to note IDs.

        Args:
            response_text: The LLM's response text
            notes: The notes that were provided as context (in order)

        Returns:
            List of citation dicts with note_id
        """
        citations = []
        seen_note_ids = set()

        # Look for citation markers [1], [2], etc.
        for i, note in enumerate(notes, 1):
            citation_marker = f"[{i}]"
            if citation_marker in response_text and note["id"] not in seen_note_ids:
                citations.append({"note_id": ObjectId(note["id"]), "chunk_id": None, "span": {}})
                seen_note_ids.add(note["id"])

        return citations
