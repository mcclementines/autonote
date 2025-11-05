# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonote is a Python-based CLI chat application with FastAPI backend. Features JWT authentication, MongoDB storage, and full OpenTelemetry observability for production-grade monitoring.

## Development Environment

- **Python Version**: 3.13 (managed via `.python-version`)
- **Package Manager**: uv (modern Python package manager)
- **Virtual Environment**: `.venv` directory

## Common Commands

### Environment Setup
```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Running the Application
```bash
# Start the FastAPI server (in one terminal)
python -m api.server

# Run the CLI client (in another terminal)
python main.py

# Or start server with custom settings
python -m api.server  # defaults to localhost:8000 with auto-reload
```

### Development Workflow
```bash
# Sync dependencies from pyproject.toml
uv pip sync

# Add new dependencies
uv add <package-name>

# Install dev dependencies (includes pytest)
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=api --cov=cli

# Access API documentation (with server running)
# Open http://localhost:8000/docs for Swagger UI
# Open http://localhost:8000/redoc for ReDoc
```

## Code Architecture

### Architecture Overview

Autonote follows a client-server architecture with a FastAPI REST backend and a CLI frontend:

```
autonote/
├── api/                    # FastAPI backend
│   ├── __init__.py
│   ├── app.py             # FastAPI application initialization
│   ├── server.py          # Server entry point with uvicorn
│   ├── database.py        # MongoDB connection management
│   ├── auth.py            # JWT authentication utilities
│   ├── observability.py   # OpenTelemetry configuration
│   ├── models/            # Pydantic models
│   │   ├── __init__.py
│   │   ├── auth.py        # Auth models (UserCreate, UserResponse, etc.)
│   │   ├── chat.py        # Chat models (ChatRequest, ChatResponse, etc.)
│   │   └── notes.py       # Notes models (NoteCreate, NoteResponse, etc.)
│   └── routes/            # Route handlers by domain
│       ├── __init__.py
│       ├── health.py      # Health check endpoints
│       ├── auth.py        # Authentication endpoints
│       ├── chat.py        # Chat endpoints
│       └── notes.py       # Notes endpoints
├── cli/                   # CLI client (frontend)
│   ├── __init__.py
│   ├── client.py          # Main REPL loop
│   ├── config.py          # Configuration and storage utilities
│   └── commands/          # Command handlers
│       ├── __init__.py
│       ├── auth.py        # Auth commands (register, login, logout)
│       ├── chat.py        # Chat commands (new, sessions, switch, history)
│       └── notes.py       # Notes commands (create)
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── api/               # API tests
│   │   ├── test_auth.py
│   │   ├── test_chat.py
│   │   └── test_notes.py
│   └── cli/               # CLI tests
│       └── test_commands.py
└── main.py                # Entry point (thin wrapper)
```

### Components

**FastAPI Backend (`api/`)**
- `api/app.py`: FastAPI application initialization (~51 lines)
  - Lifespan management (startup/shutdown)
  - OpenTelemetry instrumentation
  - Router registration
  - Lightweight and focused on app setup only

- `api/models/`: Pydantic models for request/response validation
  - `auth.py`: UserCreate, UserResponse, AuthResponse, LoginRequest
  - `chat.py`: ChatRequest, ChatResponse, ChatSessionCreate, ChatSessionResponse, Citation, ChatMessageResponse
  - `notes.py`: NoteCreate, NoteResponse, LinkOut
  - Clean separation of data models by domain

- `api/routes/`: Route handlers organized by domain
  - `health.py`: Root (/) and health check endpoints
  - `auth.py`: Authentication endpoints
    - `POST /auth/register`: User registration with JWT token
    - `POST /auth/login`: User login with JWT token
  - `chat.py`: Chat and session endpoints
    - `POST /chat`: Authenticated chat endpoint with session support
    - `POST /chat/sessions`: Create new chat session
    - `GET /chat/sessions`: List all user's chat sessions
    - `GET /chat/sessions/{session_id}/messages`: Get conversation history
  - `notes.py`: Notes endpoints
    - `POST /notes`: Create note with markdown, tags, and optional notebook
  - All routes fully instrumented with OpenTelemetry spans and metrics

- `api/server.py`: Server runner using uvicorn
  - Configurable host, port, and reload settings
  - Loads environment variables from `.env` file
  - Defaults to `0.0.0.0:8000` with auto-reload for development

- `api/database.py`: MongoDB connection manager
  - Async Motor client for MongoDB operations
  - Auto-initialization of collections and indexes
  - Connection lifecycle management (startup/shutdown)
  - Instrumented with OpenTelemetry tracing

- `api/auth.py`: JWT authentication utilities
  - Token creation and validation
  - User authentication dependency for protected routes
  - Secure password-less email-based authentication
  - Instrumented with OpenTelemetry tracing

- `api/observability.py`: OpenTelemetry configuration
  - Centralized setup for traces, metrics, and logs
  - Configurable exporters (console for dev, OTLP for prod)
  - Custom application metrics (registrations, logins, chat messages, auth failures)
  - Automatic log correlation with trace context

**CLI Client (`cli/`)**
- `cli/client.py`: Main REPL loop
  - Interactive terminal interface
  - Command routing and chat message handling
  - Error handling and user feedback

- `cli/config.py`: Configuration and storage utilities
  - API URL configuration
  - Token and session file management
  - Save/load/delete helper functions

- `cli/commands/`: Command handlers by domain
  - `auth.py`: register_user, login_user, logout_user
  - `chat.py`: new_session, list_sessions, switch_session, view_history
  - `notes.py`: create_note (multi-line markdown input)
  - Each command handles its own user interaction and API communication

- Uses `httpx` for HTTP requests with JWT authentication
- Token storage in `~/.autonote/token`
- Session storage in `~/.autonote/session`
- Auto-creates chat sessions on first message if no active session
- Handles connection errors and authentication failures gracefully

**Tests (`tests/`)**
- `conftest.py`: Pytest fixtures and test configuration
  - Database fixtures for testing
  - API client fixtures
  - Sample data fixtures
- `api/`: API endpoint tests
  - `test_auth.py`: Authentication endpoint tests
  - `test_chat.py`: Chat and session endpoint tests
  - `test_notes.py`: Notes endpoint tests
- `cli/`: CLI command tests
  - `test_commands.py`: CLI command function tests
- Run with: `pytest` (after installing dev dependencies)

### Request Flow

**Authentication Flow:**
1. User registers or logs in via CLI (`/register` or `/login` commands)
2. JWT token is stored locally in `~/.autonote/token`

**Chat Flow:**
1. User enters chat message in CLI
2. CLI sends authenticated POST request to `/chat` with JWT and optional session_id
3. FastAPI validates JWT, retrieves user from MongoDB
4. If no session_id provided, auto-creates a new chat session
5. User message is stored in `chat_messages` collection
6. Request is processed (currently echoes back) with full observability tracing
7. Assistant response is stored in `chat_messages` collection
8. Response is returned to CLI and displayed to user
9. Session's `last_active_at` is updated
10. All operations are logged and traced with OpenTelemetry

**Note Creation Flow:**
1. User enters `/note` command in CLI
2. Multi-line markdown input collected (type `END` or Ctrl+D to finish)
3. CLI sends authenticated POST request to `/notes` with title, content, tags, and optional notebook_id
4. FastAPI validates JWT, calculates word count, and stores note in MongoDB
5. Note document created with version 1, status "active"
6. Response with note details returned to CLI

### Observability & Monitoring

**OpenTelemetry Integration:**
- **Traces**: End-to-end request tracing across all endpoints
- **Metrics**: Custom application metrics (user activity, auth events)
- **Logs**: Structured logging with automatic trace correlation
- **Exporters**: Console (development) or OTLP (production)

**Instrumentation:**
- Automatic instrumentation of FastAPI endpoints
- Custom spans for business operations (registration, login, chat)
- Database operation tracing
- Authentication flow tracing
- Request/response logging with trace context

**Available Metrics:**
- `user.registrations`: Total user registrations
- `user.logins`: Total successful logins
- `chat.messages`: Total chat messages processed
- `auth.failures`: Authentication failures by reason
- `http.server.request.duration`: HTTP request latency
- `db.query.duration`: Database query performance

**Configuration:**
See `.env.example` for all OpenTelemetry configuration options. Key settings:
- `OTEL_EXPORTER_TYPE`: Set to `console` (dev) or `otlp` (prod)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Your observability backend endpoint
- `OTEL_LOG_LEVEL`: Log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Supported Backends:**
- Jaeger (open source, local development)
- Grafana Cloud / Tempo + Loki
- Datadog
- New Relic
- AWS X-Ray / CloudWatch
- Google Cloud Trace
- Any OTLP-compatible backend

### Database Collections

**users:**
- Fields: `_id`, `email`, `name`, `created_at`, `status` ("active" | "disabled")
- Index: `unique(email)`

**chat_sessions:**
- Fields: `_id`, `user_id`, `title`, `created_at`, `last_active_at`
- Index: `user_id`, `last_active_at`
- Stores conversation metadata, auto-created on first message

**chat_messages:**
- Fields: `_id`, `session_id`, `role` ("user" | "assistant" | "system"), `content`, `citations[]`, `created_at`
- Index: `session_id`, `created_at`
- Stores complete conversation history with citation support

**notes:**
- Fields: `_id`, `notebook_id`, `author_id`, `title`, `content_md`, `tags[]`, `status` ("active" | "archived" | "trashed"), `pinned`, `created_at`, `updated_at`, `version`, `word_count`, `links_out[]`
- Index: `created_at`, `author_id + status`, `text(title, content_md)`
- Full markdown support with versioning and backlinks

**notebooks:** (placeholder for future use)

### Extension Points

The `/chat` endpoint in `api/app.py` is designed to be extended with:
- AI/LLM integration for intelligent responses
- RAG (Retrieval-Augmented Generation) with citations from notes
- Vector search for semantic note retrieval
- Additional processing logic
- All extensions automatically inherit observability instrumentation

## Project Configuration

- `pyproject.toml`: Defines project metadata and dependencies
  - **Core Dependencies**:
    - `fastapi`: Web framework for building the REST API
    - `uvicorn[standard]`: ASGI server for running FastAPI
    - `httpx`: HTTP client for CLI-to-API communication
    - `motor`: Async MongoDB driver
    - `python-jose[cryptography]`: JWT token handling
    - `email-validator`: Email validation for authentication
  - **Observability Dependencies**:
    - `opentelemetry-api`: OpenTelemetry API
    - `opentelemetry-sdk`: OpenTelemetry SDK
    - `opentelemetry-instrumentation-fastapi`: FastAPI auto-instrumentation
    - `opentelemetry-instrumentation-httpx`: httpx auto-instrumentation
    - `opentelemetry-instrumentation-logging`: Logging integration
    - `opentelemetry-exporter-otlp`: OTLP exporter for production
  - Requires Python >=3.13

- `.env.example`: Template for environment configuration
  - MongoDB connection settings
  - JWT authentication configuration
  - OpenTelemetry observability settings
  - Copy to `.env` and customize for your environment

- `.gitignore`: Standard Python exclusions for `__pycache__`, virtual environments, and build artifacts

## Prerequisites

- **Python 3.13+**
- **MongoDB**: Local instance or MongoDB Atlas
  - Default: `mongodb://localhost:27017`
  - Database name: `autonote`
- **uv package manager**: For dependency management

## Getting Started

1. **Clone and setup:**
   ```bash
   cd autonote
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB URL and JWT secret
   ```

3. **Start MongoDB:**
   ```bash
   # If using Docker:
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   ```

4. **Run the server:**
   ```bash
   python -m api.server
   # Server starts at http://localhost:8000
   # API docs at http://localhost:8000/docs
   ```

5. **Run the CLI client:**
   ```bash
   python main.py
   # Use /register to create an account
   # Use /login to authenticate
   # Use /note to create notes
   # Use /new to start a new chat session
   # Start chatting!
   ```

## Development Tips

- **Logs**: By default, console logging is enabled for development
- **Traces**: Set `OTEL_EXPORTER_TYPE=console` to see traces in terminal
- **API Docs**: Visit `http://localhost:8000/docs` for interactive API documentation
- **Database**: Collections (`users`, `chat_sessions`, `chat_messages`, `notes`, `notebooks`) are auto-created on first run
- **Authentication**: JWT tokens are valid for 30 days by default
- **Sessions**: Chat sessions are auto-created on first message if not explicitly created
- **Notes**: Support multi-line markdown with tags and optional notebook assignment
