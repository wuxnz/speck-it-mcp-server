# Contributing Guide

Thank you for your interest in improving the Speck-It MCP Server! This document outlines how to get set up, propose changes, and contribute effectively.

## Getting Started

1. **Clone the repository** and create a new branch for your work.
2. **Install dependencies** using `uv sync` (or `pip install -r requirements.txt` if you manage dependencies manually).
3. **Activate tooling** by running the MCP server with `uv run python main.py` to verify your environment works end-to-end.

## Development Workflow

- **Coding style**: Follow the existing patterns in `lib/speckit.py` and `main.py`. Use descriptive docstrings and type hints.
- **Formatting & linting**: Run `uv run ruff check .` and `uv run ruff format .` (or your preferred formatter/linter) before submitting a change.
- **Testing**: Execute `uv run pytest` to validate the test suite. Add new tests in the `test/` directory to cover bug fixes or features.
- **Artifacts**: If your change impacts generated markdown templates or the feature workflow, regenerate specs/plans/tasks in a sample project and ensure the outputs look correct.

## Submitting Changes

1. **Commit messages**: Use clear, descriptive messages that explain the motivation for the change.
2. **Pull requests** should include:
   - A summary of the change and motivation.
   - Testing steps/results.
   - Any follow-up work or known issues.
3. **Review process**: Maintainers will review PRs for correctness, style, and alignment with project goals. Please respond promptly to feedback.

## Reporting Issues

- Use the issue tracker to report bugs, request features, or ask questions.
- Provide reproduction steps, logs, and environment details to help diagnose problems.

## License and Attribution

- Contributions are accepted under the project’s **BSD 3-Clause License**.
- Remember that upstream GitHub Spec Kit assets are MIT licensed; do not remove or alter their attribution.

We appreciate your contributions—thank you for helping make Spec-Driven development more accessible to MCP agents!
