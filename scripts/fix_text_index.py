"""Fix text index conflict by dropping old index."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


async def fix_text_index():
    """Drop old text index that conflicts with new schema."""
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Load MongoDB URL from environment
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "autonote")

    print(f"Connecting to MongoDB: {db_name}")
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

    # Check existing indexes
    print("\nExisting indexes on 'notes' collection:")
    indexes = await db.notes.list_indexes().to_list(length=None)
    for idx in indexes:
        print(f"  - {idx['name']}: {idx.get('key', {})}")

    # Drop the old conflicting text index
    old_index_name = "title_text_content_text"
    try:
        print(f"\nDropping old text index: {old_index_name}")
        await db.notes.drop_index(old_index_name)
        print(f"✓ Successfully dropped {old_index_name}")
    except Exception as e:
        print(f"✗ Failed to drop index: {e}")

    # Close connection
    client.close()
    print("\nDone! You can now start the server.")


if __name__ == "__main__":
    asyncio.run(fix_text_index())
