# GitHub Actions CI/CD Pipeline

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Current Workflows

### `ci.yml` - Continuous Integration

Main CI pipeline that runs on every push and pull request. The pipeline is structured into modular jobs for easy extension.

**Triggers:**
- Push to: `main`, `develop`, `claude/**` branches
- Pull requests to: `main`, `develop` branches

**Jobs:**

1. **lint** - Code quality checks
   - ✅ Runs ruff for fast, comprehensive Python linting
   - ✅ Checks code formatting with `ruff format --check`
   - ✅ Lints code with `ruff check`
   - Enforces consistent code style and catches common bugs

2. **test** - Test suite execution
   - Runs pytest on Python 3.13 (configurable for multiple versions)
   - Generates coverage reports
   - Uploads coverage artifacts
   - Matrix strategy ready for multi-version testing

3. **security** - Security scanning
   - Future: Add safety or pip-audit for dependency vulnerability scanning
   - Placeholder included for easy integration

4. **build** - Build validation
   - Validates package installation
   - Prepared for Docker builds
   - Gateway job for CD pipeline

5. **ci-success** - Summary job
   - Checks all previous jobs passed
   - Useful for branch protection rules
   - Single status check for PR requirements

## Extension Points

### Adding Container Builds

The `build` job has a placeholder for Docker builds. To enable:

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: false
    tags: autonote:${{ github.sha }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Adding Deployment (CD)

Create a new `cd.yml` workflow or extend `ci.yml` with deployment jobs:

```yaml
deploy:
  name: Deploy to Production
  needs: [ci-success]
  if: github.ref == 'refs/heads/main'
  runs-on: ubuntu-latest
  steps:
    # Deployment steps here
```

### Linting (Active)

✅ **Linting is now fully configured and active!**

The `lint` job uses ruff for comprehensive Python linting:
- Configured in `pyproject.toml` with sensible defaults
- Runs `ruff format --check` to verify code formatting
- Runs `ruff check` to catch bugs and style issues
- See `pyproject.toml` for full configuration details

### Multi-Version Testing

Update the matrix in the `test` job:

```yaml
matrix:
  python-version: ["3.12", "3.13", "3.14"]
```

## Environment Variables

- `PYTHON_VERSION`: Default Python version (currently 3.13)
- `UV_CACHE_DIR`: UV package cache directory

## Best Practices

1. **Job Dependencies**: Use `needs:` to create dependencies between jobs
2. **Caching**: UV caching is enabled via `astral-sh/setup-uv@v3`
3. **Artifacts**: Coverage reports are uploaded for 7 days
4. **Concurrency**: Only one workflow runs per branch at a time
5. **Matrix**: Designed for easy multi-version and multi-platform testing

## Local Testing

Test workflow syntax before pushing:

```bash
# Install yamllint
pip install yamllint

# Validate YAML
yamllint .github/workflows/ci.yml

# Or use Python
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

## Future Enhancements

- [x] ~~Add code formatting checks (ruff/black)~~ ✅ **Done** - Using ruff
- [x] ~~Add linting (ruff/flake8)~~ ✅ **Done** - Using ruff
- [ ] Add dependency vulnerability scanning (safety/pip-audit)
- [ ] Add Docker image builds
- [ ] Add container scanning (trivy/snyk)
- [ ] Add deployment workflows
- [ ] Add performance benchmarks
- [ ] Add documentation building
- [ ] Add release automation
