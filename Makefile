.PHONY: help install install-dev test lint format type-check security clean build pre-commit all

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security     Run security scans"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  pre-commit   Install and run pre-commit hooks"
	@echo "  all          Run all quality checks"

# Installation
install:
	uv sync

install-dev:
	uv sync --dev

# Testing
test:
	uv run pytest --cov=lib --cov-report=term-missing --cov-report=html

test-fast:
	uv run pytest -x -v

# Code quality
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run black .
	uv run ruff format .
	uv run isort .

type-check:
	uv run mypy lib/

security:
	uv run safety check
	uv run bandit -r lib/

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Pre-commit
pre-commit:
	uv run pre-commit install
	uv run pre-commit run --all-files

# All checks
all: lint type-check test security
	@echo "All quality checks passed!"

# Development workflow
dev-setup: install-dev pre-commit
	@echo "Development environment setup complete!"

# CI workflow
ci: lint type-check test security
	@echo "CI checks completed successfully!"

# Release workflow
release: clean build
	@echo "Package built successfully!"
	@echo "Run 'uv run twine upload dist/*' to publish"