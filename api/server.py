"""Server entry point for running the FastAPI application."""

import asyncio
import signal

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Server:
    """Custom server wrapper with proper signal handling."""

    def __init__(self, config: uvicorn.Config):
        self.server = uvicorn.Server(config)
        self.should_exit = False

    def handle_exit(self, _sig, _frame):
        """Handle exit signals."""
        print("\n[INFO] Received shutdown signal, stopping server...")
        self.should_exit = True

    async def serve(self):
        """Run the server with proper signal handling."""
        # Install signal handlers
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

        await self.server.serve()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    if reload:
        # For reload mode, use the simple uvicorn.run() approach
        # Note: Ctrl-C handling may be degraded in reload mode due to subprocess
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=False,
        )
    else:
        # For production, use Server class with proper async handling
        config = uvicorn.Config(
            "api.app:app",
            host=host,
            port=port,
            log_level="info",
            access_log=False,
        )
        server = Server(config)
        asyncio.run(server.serve())


if __name__ == "__main__":
    run_server(reload=True)
