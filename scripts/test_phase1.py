#!/usr/bin/env python3
"""Test script to validate Phase 1: Database Schema & Infrastructure."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import Database


async def test_database_schema():
    """Test that all collections and indexes are created correctly."""

    print("🔍 Testing Phase 1: Database Schema & Infrastructure\n")

    # Connect to database
    print("1. Connecting to MongoDB...")
    await Database.connect()
    db = Database.get_database()
    print("   ✅ Connected successfully\n")

    # Check collections
    print("2. Checking collections...")
    collections = await db.list_collection_names()

    expected_collections = ["users", "notebooks", "notes", "chat_sessions", "chat_messages", "note_chunks"]

    for collection in expected_collections:
        if collection in collections:
            print(f"   ✅ {collection}")
        else:
            print(f"   ❌ {collection} (missing)")
    print()

    # Check note_chunks indexes
    print("3. Checking note_chunks indexes...")
    if "note_chunks" in collections:
        indexes = await db.note_chunks.index_information()

        expected_indexes = [
            "_id_",
            "note_id_1",
            "note_id_1_chunk_index_1",
            "note_version_1",
            "note_id_1_note_version_1"
        ]

        for index_name in expected_indexes:
            if index_name in indexes:
                print(f"   ✅ {index_name}")
            else:
                print(f"   ❌ {index_name} (missing)")
    else:
        print("   ❌ note_chunks collection doesn't exist")
    print()

    # Check notes collection has new fields structure
    print("4. Checking notes collection schema...")
    notes_indexes = await db.notes.index_information()
    print(f"   ℹ️  Notes has {len(notes_indexes)} indexes")
    print()

    # Check environment configuration
    print("5. Checking environment configuration...")
    chunking_enabled = os.getenv("CHUNKING_ENABLED", "false")
    chunking_strategy = os.getenv("CHUNKING_STRATEGY", "none")
    chunking_max_tokens = os.getenv("CHUNKING_MAX_TOKENS", "0")
    chunking_async = os.getenv("CHUNKING_ASYNC", "false")

    print(f"   CHUNKING_ENABLED: {chunking_enabled}")
    print(f"   CHUNKING_STRATEGY: {chunking_strategy}")
    print(f"   CHUNKING_MAX_TOKENS: {chunking_max_tokens}")
    print(f"   CHUNKING_ASYNC: {chunking_async}")
    print()

    # Disconnect
    await Database.disconnect()
    print("6. Disconnected from MongoDB")
    print("\n✅ Phase 1 validation complete!")


if __name__ == "__main__":
    asyncio.run(test_database_schema())
