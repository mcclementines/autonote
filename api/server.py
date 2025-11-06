"""Server entry point for running the FastAPI application."""

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("api.app:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run_server(reload=True)
