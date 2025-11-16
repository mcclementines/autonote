"""MongoDB database connection management."""

from __future__ import annotations

import os
from time import time

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from opentelemetry import trace

# Initialize logger
logger = structlog.get_logger(__name__)


class Database:
    """MongoDB database connection manager."""

    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

    @classmethod
    async def connect(cls) -> None:
        """Establish connection to MongoDB."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("db.connect") as span:
            mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB_NAME", "autonote")
            init_db = os.getenv("INIT_DB", "true").lower() == "true"

            span.set_attribute("db.system", "mongodb")
            span.set_attribute("db.name", db_name)

            # Mask password in connection string for logging
            def mask_connection_string(url: str) -> str:
                """Mask password in MongoDB connection string."""
                import re

                # Pattern matches: mongodb://user:password@host or mongodb+srv://user:password@host
                pattern = r"(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@.+)"
                return re.sub(pattern, r"\1****\3", url)

            safe_url = mask_connection_string(mongo_url)
            logger.info("mongodb_connecting", url=safe_url, database=db_name)

            cls.client = AsyncIOMotorClient(mongo_url)
            cls.db = cls.client[db_name]

            # Verify connection
            start_time = time()
            await cls.client.admin.command("ping")
            duration = (time() - start_time) * 1000

            logger.info("mongodb_connected", ping_ms=round(duration, 2))

            # Initialize collections and indexes if enabled
            if init_db:
                await cls._initialize_collections()
                logger.info("database_initialization_completed")

    @classmethod
    async def _initialize_collections(cls) -> None:
        """
        Initialize collections and indexes.
        Only creates collections/indexes if they don't already exist.
        """
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("db.initialize_collections"):
            if cls.db is None:
                raise RuntimeError("Database not connected")

            logger.info("db_initializing_collections")

            # Get list of existing collections
            existing_collections = await cls.db.list_collection_names()

            # Create 'users' collection if it doesn't exist
            if "users" not in existing_collections:
                await cls.db.create_collection("users")
                logger.info("collection_created", collection="users")
            else:
                logger.debug("collection_exists", collection="users")

            # Create unique index on email for users
            await cls.db.users.create_index("email", unique=True)
            logger.debug("index_ensured", collection="users", field="email", unique=True)

            # Create 'notebooks' collection if it doesn't exist
            if "notebooks" not in existing_collections:
                await cls.db.create_collection("notebooks")
                logger.info("collection_created", collection="notebooks")
            else:
                logger.debug("collection_exists", collection="notebooks")

            # Create 'notes' collection if it doesn't exist
            if "notes" not in existing_collections:
                await cls.db.create_collection("notes")
                logger.info("collection_created", collection="notes")
            else:
                logger.debug("collection_exists", collection="notes")

            # Create indexes for notes collection
            # This is idempotent - MongoDB won't recreate if they exist
            await cls.db.notes.create_index("created_at")
            logger.debug("index_ensured", collection="notes", field="created_at")

            # Index for filtering by author and status
            await cls.db.notes.create_index([("author_id", 1), ("status", 1)])
            logger.debug(
                "compound_index_ensured", collection="notes", fields=["author_id", "status"]
            )

            # Basic text search index on title and content_md (fallback for non-Atlas deployments)
            # Note: If using Atlas Search, this index is not needed but harmless to have
            await cls.db.notes.create_index(
                [("title", "text"), ("content_md", "text")], name="notes_basic_text_index"
            )
            logger.debug("text_index_ensured", collection="notes", fields=["title", "content_md"])

            # Create 'chat_sessions' collection if it doesn't exist
            if "chat_sessions" not in existing_collections:
                await cls.db.create_collection("chat_sessions")
                logger.info("collection_created", collection="chat_sessions")
            else:
                logger.debug("collection_exists", collection="chat_sessions")

            # Create 'chat_messages' collection if it doesn't exist
            if "chat_messages" not in existing_collections:
                await cls.db.create_collection("chat_messages")
                logger.info("collection_created", collection="chat_messages")
            else:
                logger.debug("collection_exists", collection="chat_messages")

            # Create 'note_chunks' collection if it doesn't exist
            if "note_chunks" not in existing_collections:
                await cls.db.create_collection("note_chunks")
                logger.info("collection_created", collection="note_chunks")
            else:
                logger.debug("collection_exists", collection="note_chunks")

            # Create indexes for note_chunks collection
            await cls.db.note_chunks.create_index("note_id")
            logger.debug("index_ensured", collection="note_chunks", field="note_id")

            await cls.db.note_chunks.create_index([("note_id", 1), ("chunk_index", 1)])
            logger.debug(
                "compound_index_ensured",
                collection="note_chunks",
                fields=["note_id", "chunk_index"],
            )

            await cls.db.note_chunks.create_index("note_version")
            logger.debug("index_ensured", collection="note_chunks", field="note_version")

            await cls.db.note_chunks.create_index([("note_id", 1), ("note_version", 1)])
            logger.debug(
                "compound_index_ensured",
                collection="note_chunks",
                fields=["note_id", "note_version"],
            )

            # Atlas Search Indexes (must be created via Atlas UI or Admin API):
            # - notes_vector_index: Vector search on 'embedding' field (note-level)
            # - notes_search_index: Full-text search on 'title' and 'content_md'
            # - note_chunks_vector_index: Vector search on chunks 'embedding' field
            # See: python -m scripts.setup_atlas_indexes for setup instructions
            logger.info(
                "atlas_indexes_note",
                message="Atlas Search indexes must be created separately via Atlas UI",
            )

    @classmethod
    async def disconnect(cls) -> None:
        """Close MongoDB connection."""
        if cls.client:
            logger.info("mongodb_disconnecting")
            cls.client.close()
            logger.info("mongodb_disconnected")

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if cls.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls.db


# Convenience function to get database
def get_db() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    return Database.get_database()
