# Claude Code Subagents

This directory contains specialized Claude Code subagent configurations for the Autonote project.

## Available Subagents

### 1. Test Writer (`test-writer`)
**Purpose**: Write comprehensive unit tests for Python code using pytest.

**When to use**:
- Adding tests for new features
- Improving test coverage
- Writing tests for existing code
- Creating test fixtures and utilities

**How to invoke**:
```
Use the test-writer subagent to add tests for the new authentication endpoint
```

**What it does**:
- Creates pytest-based unit tests
- Follows AAA pattern (Arrange-Act-Assert)
- Uses fixtures from `tests/conftest.py`
- Covers happy paths, edge cases, and error conditions
- Ensures tests pass CI requirements
- Verifies code formatting with ruff

---

### 2. Code Formatter (`code-formatter`)
**Purpose**: Format and lint Python code using ruff to meet CI standards.

**When to use**:
- Before committing code
- Fixing CI formatting failures
- Cleaning up code style
- Organizing imports

**How to invoke**:
```
Use the code-formatter subagent to format all Python files and fix linting issues
```

**What it does**:
- Runs `ruff format` to auto-format code
- Runs `ruff check --fix` to fix linting issues
- Organizes imports (stdlib → third-party → local)
- Ensures code passes CI checks
- Reports formatting and linting results

---

## CI Integration

Both subagents are designed to ensure code passes the GitHub Actions CI pipeline defined in `.github/workflows/ci.yml`:

### Required Checks
1. **Linting**: `ruff format --check .` and `ruff check .`
2. **Testing**: `pytest -v --tb=short`
3. **Coverage**: `pytest --cov=api --cov=cli --cov-report=xml`

### Configuration
All formatting and linting rules are defined in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.13
- Quote style: Double quotes
- Import sorting: Enabled (isort)
- Multiple rule sets enabled (see `pyproject.toml` for details)

---

## Usage Examples

### Example 1: Add Tests for New Feature
```
I just added a new endpoint in api/routes/users.py for user profile updates.
Use the test-writer subagent to create comprehensive tests for this endpoint.
```

### Example 2: Fix Formatting Before Commit
```
Use the code-formatter subagent to format all files and ensure they pass CI checks.
```

### Example 3: Combined Workflow
```
1. Use the code-formatter subagent to format and lint the code
2. Use the test-writer subagent to add tests for the new chat export feature
```

---

## Development Workflow

### Recommended Flow
1. **Write Code**: Implement your feature or fix
2. **Format**: Use `code-formatter` subagent
3. **Test**: Use `test-writer` subagent to add tests
4. **Verify**: Run local checks before pushing

### Local Verification Commands
```bash
# Formatting check
ruff format --check .

# Linting check
ruff check .

# Run tests
pytest -v --tb=short

# Coverage report
pytest --cov=api --cov=cli --cov-report=term
```

---

## Tips for Best Results

### For Test Writer
- Be specific about what to test: "Add tests for the /chat endpoint"
- Mention edge cases if you know them: "Make sure to test empty session_id"
- Reference existing test files for context: "Follow the pattern in test_auth.py"

### For Code Formatter
- Run before committing: "Format all modified files"
- Fix specific issues: "Fix import ordering in api/routes/chat.py"
- Verify CI readiness: "Ensure all files pass ruff checks"

---

## Subagent Development

To add new subagents:

1. Create a new markdown file in this directory: `.claude/subagents/your-agent.md`
2. Follow the existing structure with clear sections:
   - Role description
   - Standards and requirements
   - Examples and patterns
   - Checklist for completion
   - Expected output format
3. Update this README with the new subagent info

---

## Related Files

- **CI Configuration**: `.github/workflows/ci.yml`
- **Linting Config**: `pyproject.toml` (tool.ruff section)
- **Test Config**: `pyproject.toml` (dev dependencies)
- **Test Fixtures**: `tests/conftest.py`
- **Project Guide**: `CLAUDE.md`
