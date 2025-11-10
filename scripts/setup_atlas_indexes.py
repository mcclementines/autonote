"""Script to set up MongoDB Atlas Search and Vector Search indexes.

This script helps create the necessary indexes for Atlas Search and Vector Search.

Usage:
    python -m scripts.setup_atlas_indexes

Requirements:
    - MongoDB Atlas cluster (not local MongoDB)
    - pymongo with Atlas connectivity
    - Appropriate permissions to create search indexes

Note: Some index creation requires Atlas UI or Admin API. This script provides
the configurations you need to create indexes manually if needed.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables from .env file
load_dotenv()

# Atlas Vector Search Index Configuration
VECTOR_SEARCH_INDEX = {
    "name": "notes_vector_index",
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "author_id",
            },
            {
                "type": "filter",
                "path": "status",
            },
        ]
    },
}

# Atlas Search Index Configuration
ATLAS_SEARCH_INDEX = {
    "name": "notes_search_index",
    "type": "search",
    "definition": {
        "mappings": {
            "dynamic": False,
            "fields": {
                "title": {
                    "type": "string",
                    "analyzer": "lucene.standard",
                },
                "content_md": {
                    "type": "string",
                    "analyzer": "lucene.standard",
                },
                "tags": {
                    "type": "string",
                    "analyzer": "lucene.keyword",
                },
                "status": {
                    "type": "string",
                    "analyzer": "lucene.keyword",
                },
                "author_id": {
                    "type": "objectId",
                },
                "created_at": {
                    "type": "date",
                },
            },
        }
    },
}


def print_separator():
    """Print a visual separator."""
    print("=" * 80)


def print_index_config(index_name: str, config: dict):
    """Print index configuration in a readable format."""
    print(f"\n📋 {index_name} Configuration:")
    print("-" * 80)
    print(json.dumps(config, indent=2))
    print("-" * 80)


async def check_atlas_connection(client: AsyncIOMotorClient) -> tuple[bool, str]:
    """Check if connected to Atlas (not local MongoDB).

    Returns:
        Tuple of (is_atlas, message)
    """
    try:
        # Get server status for host info
        server_status = await client.admin.command("serverStatus")

        # Atlas has specific characteristics
        host_info = server_status.get("host", "")
        is_atlas = ".mongodb.net" in host_info or "atlas" in host_info.lower()

        if is_atlas:
            return True, f"✅ Connected to MongoDB Atlas: {host_info}"
        return (
            False,
            f"⚠️  Connected to non-Atlas MongoDB: {host_info}\n"
            "   Atlas Search and Vector Search require MongoDB Atlas.",
        )
    except Exception as e:
        return False, f"❌ Error checking connection: {e}"


async def list_existing_indexes(db, collection_name: str):
    """List existing search indexes on a collection."""
    try:
        collection = db[collection_name]

        print(f"\n📚 Existing indexes on '{collection_name}' collection:")
        print("-" * 80)

        # List regular indexes
        indexes = await collection.list_indexes().to_list(None)
        print(f"\nRegular indexes ({len(indexes)}):")
        for idx in indexes:
            print(f"  - {idx.get('name')}: {list(idx.get('key', {}).keys())}")

        # Note: Search indexes require Atlas Admin API to list programmatically
        print("\n⚠️  Atlas Search and Vector Search indexes must be viewed in Atlas UI:")
        print("   Data Services > Database > Atlas Search")

    except Exception as e:
        print(f"❌ Error listing indexes: {e}")


def print_manual_instructions():
    """Print instructions for manual index creation."""
    print_separator()
    print("\n📖 MANUAL INDEX CREATION INSTRUCTIONS")
    print_separator()

    print(
        """
Since Atlas Search and Vector Search indexes cannot be created programmatically
via the standard MongoDB driver, you need to create them via the Atlas UI or Admin API.

🔧 OPTION 1: Atlas UI (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Log in to MongoDB Atlas (https://cloud.mongodb.com)
2. Navigate to your cluster
3. Click "Search" tab (or "Atlas Search" in the left sidebar)
4. Click "Create Search Index"
5. Choose "JSON Editor"
6. Create TWO indexes with the configurations below:

"""
    )

    print_index_config("VECTOR SEARCH INDEX", VECTOR_SEARCH_INDEX)
    print(
        """
Steps for Vector Search Index:
  a) Collection: notes
  b) Index Name: notes_vector_index
  c) Paste the configuration above
  d) Click "Create Search Index"

"""
    )

    print_index_config("ATLAS SEARCH INDEX", ATLAS_SEARCH_INDEX)
    print(
        """
Steps for Atlas Search Index:
  a) Collection: notes
  b) Index Name: notes_search_index
  c) Paste the configuration above
  d) Click "Create Search Index"

⏱️  Index creation typically takes 1-5 minutes depending on data size.

"""
    )

    print(
        """
🔧 OPTION 2: Atlas Admin API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You can also use the Atlas Admin API to create indexes programmatically:
https://www.mongodb.com/docs/atlas/reference/api-resources-spec/#tag/Atlas-Search

You'll need:
  - Atlas API Public Key
  - Atlas API Private Key
  - Project ID
  - Cluster Name

"""
    )

    print(
        """
🔧 OPTION 3: MongoDB CLI (mongocli/atlas cli)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install: https://www.mongodb.com/docs/atlas/cli/stable/

Example commands:
  atlas auth login
  atlas clusters search indexes create --clusterName <name> --file vector_index.json
  atlas clusters search indexes create --clusterName <name> --file search_index.json

"""
    )


def save_index_configs():
    """Save index configurations to JSON files for easy import."""
    try:
        atlas_indexes_dir = Path("atlas_indexes")
        atlas_indexes_dir.mkdir(parents=True, exist_ok=True)

        # Save vector search index
        vector_index_path = atlas_indexes_dir / "vector_search_index.json"
        with vector_index_path.open("w") as f:
            json.dump(VECTOR_SEARCH_INDEX, f, indent=2)

        # Save atlas search index
        search_index_path = atlas_indexes_dir / "atlas_search_index.json"
        with search_index_path.open("w") as f:
            json.dump(ATLAS_SEARCH_INDEX, f, indent=2)

        print("\n💾 Index configurations saved to:")
        print("   - atlas_indexes/vector_search_index.json")
        print("   - atlas_indexes/atlas_search_index.json")
        print("\n   You can use these files with MongoDB CLI or copy-paste into Atlas UI.")

    except Exception as e:
        print(f"⚠️  Could not save index configs to files: {e}")


async def main():
    """Main setup function."""
    print_separator()
    print("🔍 MONGODB ATLAS SEARCH INDEX SETUP")
    print_separator()

    # Load MongoDB connection from environment
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "autonote")

    # Mask password for display
    def mask_url(url: str) -> str:
        pattern = r"(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@.+)"
        return re.sub(pattern, r"\1****\3", url)

    print(f"\n🔗 Connection URL: {mask_url(mongo_url)}")
    print(f"🗄️  Database: {db_name}\n")

    # Connect to MongoDB
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

    try:
        # Test connection
        await client.admin.command("ping")
        print("✅ Connection successful\n")

        # Check if Atlas
        is_atlas, message = await check_atlas_connection(client)
        print(message)
        print()

        if not is_atlas:
            print("⚠️  WARNING: Atlas Search and Vector Search are only available on MongoDB Atlas.")
            print("   If you're using local MongoDB, these features will not work.")
            print("   The code will fall back to basic search, but performance will be poor.\n")

        # List existing indexes
        await list_existing_indexes(db, "notes")

        # Save index configs to files
        save_index_configs()

        # Print manual instructions
        print_manual_instructions()

        print_separator()
        print("✅ Setup information generated successfully!")
        print_separator()
        print("\n📌 NEXT STEPS:")
        print("   1. Create the indexes in Atlas UI using the configurations above")
        print("   2. Wait for indexes to build (1-5 minutes)")
        print("   3. Verify indexes are active in Atlas UI")
        print("   4. Set MONGODB_ATLAS_SEARCH=true in your .env file to enable Atlas Search")
        print("   5. Restart your application")
        print(
            "\n   The application will automatically detect and use Atlas indexes when available.\n"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure your MongoDB connection string is correct in .env file:\n")
        print("   MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net\n")
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
