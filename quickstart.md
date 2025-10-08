# Speck-It MCP Server - Quick Start Guide

This guide provides comprehensive instructions for setting up and running the Speck-It MCP Server in your development environment.

## Prerequisites

### System Requirements

- **Python**: 3.13 or higher
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: `uv` (recommended) or `pip`

### Required Tools

- **uv**: Modern Python package installer and resolver
- **MCP-compatible agent**: Windsurf, Roo Code, Cline, or similar

## Installation & Setup

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):

   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone or download the project**:

   ```bash
   git clone <repository-url>
   cd speck-it
   ```

3. **Install dependencies**:

   ```bash
   uv sync
   ```

4. **Verify installation**:
   ```bash
   uv run python --version
   uv run python main.py --help
   ```

### Option 2: Using pip

1. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the MCP Server

### Basic Execution

```bash
# Using uv
uv run python main.py

# Using pip/venv
python main.py
```

The server communicates over stdio by default and is ready to accept MCP protocol commands.

### Environment Configuration

Optional environment variables:

```bash
# Set custom project root directory
export SPECKIT_PROJECT_ROOT="/path/to/your/project"

# Set custom storage directory name (default: .speck-it)
export SPECKIT_STORAGE_DIR=".my-speck-it"

# Run with custom configuration
uv run python main.py
```

## MCP Client Configuration

### Windsurf Configuration

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "speck-it": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "/path/to/speck-it/main.py"
      ],
      "disabledTools": []
    }
  }
}
```

### Roo Code Configuration

```json
{
  "mcpServers": {
    "speck-it": {
      "command": "uv",
      "args": ["run", "python", "/path/to/speck-it/main.py"],
      "env": {
        "SPECKIT_PROJECT_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

### Cline Configuration

```json
{
  "mcpServers": {
    "speck-it": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "/path/to/speck-it"
    }
  }
}
```

## First-Time Workflow

### 1. Initialize Your Project

Start by setting a project constitution:

```python
# Call via your MCP agent
set_constitution(content="""
# My Project Constitution

## Core Principles
- Quality first approach
- Test-driven development
- Clean code standards
- Regular documentation

## Development Standards
- All code must be tested
- Follow PEP 8 style guide
- Use type hints where applicable
""")
```

### 2. Register Feature Root

```python
set_feature_root(feature_id="my-first-feature")
```

### 3. Create Feature Specification

```python
generate_spec(
    feature_name="User Authentication",
    description="Implement user login and registration functionality with JWT tokens",
    feature_id="auth-feature"
)
```

### 4. Generate Implementation Plan

```python
generate_plan(feature_id="auth-feature")
```

### 5. Create Task List

```python
generate_tasks(feature_id="auth-feature")
```

### 6. Execute Tasks

```python
# Get next task
next_task(feature_id="auth-feature")

# Complete task
complete_task(feature_id="auth-feature", task_id="T001")

# Update with notes
update_task(feature_id="auth-feature", task_id="T002", note="Implemented user model")
```

## Project Structure

After initialization, your project will have this structure:

```
your-project/
├── .speck-it/
│   ├── memory/
│   │   └── constitution.md
│   ├── specs/
│   │   └── feature-id/
│   │       ├── spec.md
│   │       ├── plan.md
│   │       ├── tasks.md
│   │       └── analysis.json
│   └── state/
│       └── feature-id_status.json
├── src/
├── tests/
└── docs/
```

## Troubleshooting

### Common Issues

1. **"No workspace found" error**

   - Ensure you've called `set_constitution` first
   - Check that you're in the correct directory

2. **"Feature root not found" error**

   - Call `set_feature_root` with your feature ID
   - Verify the directory exists and is writable

3. **Python version compatibility**

   - Ensure Python 3.13+ is installed
   - Check with `python --version`

4. **MCP connection issues**
   - Verify your MCP client configuration
   - Check that the server path is correct
   - Ensure all required dependencies are installed

### Debug Mode

Enable debug output by setting environment variable:

```bash
export SPECKIT_DEBUG=1
uv run python main.py
```

### Getting Help

- Check the [README.md](README.md) for detailed documentation
- Use `get_workflow_guide()` for step-by-step guidance
- Review generated artifacts in `.speck-it/` directory

## Manual Validation Steps

This section provides comprehensive validation procedures to ensure your Speck-It workflow is functioning correctly.

### Validation Overview

The Speck-It workflow follows a strict order: **Constitution → Feature Root → Spec → Plan → Tasks → Execution**. Each step must be validated before proceeding to the next.

### Step-by-Step Validation Checklist

#### 1. Constitution Validation

**Objective**: Verify project constitution is properly saved and accessible.

**Validation Steps**:

```python
# Check if constitution exists
constitution = load_constitution()
if constitution is None:
    raise ValueError("Constitution not found. Run set_constitution() first.")

# Verify constitution content
if len(constitution.strip()) < 10:
    raise ValueError("Constitution appears to be empty or too short.")

print("✅ Constitution validation passed")
```

**Expected Outcome**: Constitution content is returned without errors.

#### 2. Feature Root Registration Validation

**Objective**: Verify feature root is properly registered and accessible.

**Validation Steps**:

```python
# Check if feature root is registered
root = lookup_feature_root("your-feature-id")
if root is None:
    raise ValueError("Feature root not found. Run set_feature_root() first.")

# Verify directory exists and is writable
if not root.exists():
    raise ValueError(f"Feature root directory does not exist: {root}")

if not os.access(root, os.W_OK):
    raise ValueError(f"Feature root directory is not writable: {root}")

print("✅ Feature root validation passed")
```

**Expected Outcome**: Feature root directory is found and accessible.

#### 3. Specification Generation Validation

**Objective**: Verify specification is properly generated with all required sections.

**Validation Steps**:

```python
# Generate specification
artifacts, analysis, content = generate_spec(
    feature_name="Test Feature",
    description="A test feature for validation"
)

# Verify artifacts
if artifacts.spec_path is None:
    raise ValueError("Specification path is None")

if not artifacts.spec_path.exists():
    raise ValueError("Specification file was not created")

# Verify analysis
if analysis is None:
    raise ValueError("Feature analysis is None")

validation_issues = analysis.validate()
if validation_issues:
    print(f"⚠️  Specification validation warnings: {validation_issues}")

# Verify content
required_sections = [
    "Feature Specification:",
    "Quick Summary",
    "User Scenarios & Testing",
    "Requirements",
    "Review & Acceptance Checklist"
]

for section in required_sections:
    if section not in content:
        raise ValueError(f"Required section missing: {section}")

print("✅ Specification generation validation passed")
```

**Expected Outcome**: Specification file is created with all required sections.

#### 4. Plan Generation Validation

**Objective**: Verify implementation plan is properly generated based on specification.

**Validation Steps**:

```python
# Generate plan
plan_artifacts, plan_analysis, plan_content = generate_plan(feature_id="your-feature-id")

# Verify artifacts
if plan_artifacts.plan_path is None:
    raise ValueError("Plan path is None")

if not plan_artifacts.plan_path.exists():
    raise ValueError("Plan file was not created")

# Verify content
required_plan_sections = [
    "Implementation Plan:",
    "Technical Context",
    "Project Structure Recommendation",
    "Research Agenda",
    "Design Deliverables",
    "Task Generation Strategy"
]

for section in required_plan_sections:
    if section not in plan_content:
        raise ValueError(f"Required plan section missing: {section}")

# Verify plan references specification
if "spec.md" not in plan_content:
    raise ValueError("Plan does not reference specification")

print("✅ Plan generation validation passed")
```

**Expected Outcome**: Plan file is created with all required sections and references specification.

#### 5. Task Generation Validation

**Objective**: Verify task list is properly generated based on plan.

**Validation Steps**:

```python
# Generate tasks
task_artifacts, task_analysis, task_content = generate_tasks(feature_id="your-feature-id")

# Verify artifacts
if task_artifacts.tasks_path is None:
    raise ValueError("Tasks path is None")

if not task_artifacts.tasks_path.exists():
    raise ValueError("Tasks file was not created")

# Verify content
if "Tasks:" not in task_content:
    raise ValueError("Tasks header missing")

if "## Task List" not in task_content:
    raise ValueError("Task list section missing")

# Verify task format
import re
task_pattern = r"- \[ \] T\d{3}"
if not re.search(task_pattern, task_content):
    raise ValueError("No properly formatted tasks found")

# Count tasks
task_count = len(re.findall(task_pattern, task_content))
if task_count < 3:
    raise ValueError(f"Too few tasks generated: {task_count}")

print(f"✅ Task generation validation passed ({task_count} tasks)")
```

**Expected Outcome**: Tasks file is created with properly formatted tasks.

#### 6. Task Execution Validation

**Objective**: Verify tasks can be properly tracked and completed.

**Validation Steps**:

```python
# Get task list
tasks = list_tasks(feature_id="your-feature-id")
if not tasks:
    raise ValueError("No tasks found")

# Get first task
first_task = next_open_task(feature_id="your-feature-id")
if first_task is None:
    raise ValueError("No open tasks found")

# Update task
result = update_task(
    feature_id="your-feature-id",
    task_id=first_task["task_id"],
    note="Validation test note"
)

if "task" not in result:
    raise ValueError("Task update failed")

# Complete task
complete_result = complete_task(
    feature_id="your-feature-id",
    task_id=first_task["task_id"]
)

if not complete_result["task"]["completed"]:
    raise ValueError("Task completion failed")

print("✅ Task execution validation passed")
```

**Expected Outcome**: Tasks can be retrieved, updated, and completed successfully.

#### 7. Feature Finalization Validation

**Objective**: Verify feature can be properly finalized after all tasks are complete.

**Validation Steps**:

```python
# Complete all remaining tasks
tasks = list_tasks(feature_id="your-feature-id")
for task in tasks:
    if not task["completed"]:
        complete_task(feature_id="your-feature-id", task_id=task["task_id"])

# Finalize feature
status = finalize_feature(feature_id="your-feature-id")

if "finalized_at" not in status:
    raise ValueError("Feature was not properly finalized")

# Verify status file
status_path = Path(status["status_path"])
if not status_path.exists():
    raise ValueError("Status file was not created")

print("✅ Feature finalization validation passed")
```

**Expected Outcome**: Feature is properly finalized with status file created.

### End-to-End Workflow Validation

**Objective**: Validate the complete workflow from start to finish.

**Validation Script**:

```python
def validate_complete_workflow():
    """Validate the complete Speck-It workflow."""
    print("🔍 Starting complete workflow validation...")

    try:
        # Step 1: Set constitution
        set_constitution(content="# Test Constitution\n\nThis is a test constitution.")
        print("✅ Step 1: Constitution set")

        # Step 2: Register feature root
        set_feature_root(feature_id="validation-test")
        print("✅ Step 2: Feature root registered")

        # Step 3: Generate spec
        artifacts, analysis, content = generate_spec(
            feature_name="Validation Test Feature",
            description="A feature for testing the complete workflow"
        )
        feature_id = artifacts.feature_id
        print("✅ Step 3: Specification generated")

        # Step 4: Generate plan
        generate_plan(feature_id=feature_id)
        print("✅ Step 4: Implementation plan generated")

        # Step 5: Generate tasks
        generate_tasks(feature_id=feature_id)
        print("✅ Step 5: Tasks generated")

        # Step 6: Complete all tasks
        tasks = list_tasks(feature_id=feature_id)
        for task in tasks:
            complete_task(feature_id=feature_id, task_id=task["task_id"])
        print(f"✅ Step 6: All {len(tasks)} tasks completed")

        # Step 7: Finalize feature
        finalize_feature(feature_id=feature_id)
        print("✅ Step 7: Feature finalized")

        print("🎉 Complete workflow validation passed!")
        return True

    except Exception as e:
        print(f"❌ Workflow validation failed: {e}")
        return False

# Run validation
validate_complete_workflow()
```

### Automated Validation Script

Two validation scripts are provided:

#### Option 1: Direct Module Import (validate_workflow.py)

For direct module import usage (may have circular import issues):

```bash
python validate_workflow.py --detailed
```

#### Option 2: MCP Server Communication (validate_workflow_mcp.py)

For reliable validation using MCP server communication:

```bash
# Basic validation
python validate_workflow_mcp.py

# Detailed step-by-step validation
python validate_workflow_mcp.py --detailed
```

The MCP version is recommended as it avoids circular import issues and provides more reliable validation.

#### Validation Script Features

Both validation scripts provide:

- **Complete workflow validation**: Tests all 7 steps of the Speck-It workflow
- **Step-by-step validation**: Detailed validation with checks for each step
- **Error reporting**: Clear error messages when validation fails
- **Progress tracking**: Shows progress through each validation step
- **File verification**: Checks that all required files are created properly

#### Running Validation

```bash
# Run basic validation
python validate_workflow_mcp.py

# Run detailed validation
python validate_workflow_mcp.py --detailed

# Check validation results
echo $?
```

**Expected Output**:

- ✅ All steps should pass if the system is working correctly
- ❌ Any failures will be clearly reported with error details

### Troubleshooting Validation Issues

#### Common Validation Failures

1. **"Constitution not found"**

   - Ensure `set_constitution()` is called first
   - Check that the `.speck-it/memory/` directory exists
   - Verify file permissions

2. **"Specification file was not created"**

   - Check that the feature directory exists
   - Verify write permissions
   - Ensure feature ID is properly formatted

3. **"No properly formatted tasks found"**

   - Ensure plan generation completed successfully
   - Check that the task template is not corrupted
   - Verify the task generation logic

4. **"Task update failed"**

   - Ensure tasks file exists and is readable
   - Check task ID format (should match T001, T002, etc.)
   - Verify file permissions

5. **"Feature was not properly finalized"**
   - Ensure all tasks are completed
   - Check that spec, plan, and tasks files exist
   - Verify write permissions for status directory

### Validation Best Practices

1. **Run validation after each major change**
2. **Keep validation logs for troubleshooting**
3. **Test with different feature names and descriptions**
4. **Validate in clean environment**
5. **Check file system permissions before starting**

## Development Workflow

### For Developers

1. **Run tests**:

   ```bash
   uv run pytest
   ```

2. **Code formatting**:

   ```bash
   uv run black .
   uv run ruff check .
   ```

3. **Run validation**:

   ```bash
   # Recommended: MCP validation
   python validate_workflow_mcp.py

   # Alternative: Direct import validation
   python validate_workflow.py
   ```

4. **Regenerate specs** (when modifying templates):
   ```bash
   # After changing lib/speckit.py, regenerate existing specs
   # to keep them up to date with new templates
   ```

### Advanced Configuration

#### Custom Storage Location

```bash
export SPECKIT_STORAGE_DIR=".custom-speck-it"
```

#### Multiple Project Support

The server supports multiple projects simultaneously. Each project maintains its own `.speck-it/` directory and feature isolation.

## Next Steps

1. **Explore the tools**: Use `list_features()` to see available tools
2. **Create your first feature**: Follow the 7-step workflow
3. **Review examples**: Check `screenshots/example-outputs/` for sample projects
4. **Extend functionality**: Modify templates in `lib/speckit.py` for custom workflows

## Support

- **Documentation**: [README.md](README.md)
- **Issues**: Report via your project's issue tracker
- **Community**: Join discussions in your development community

---

_This quickstart guide covers the essential setup and first-use scenarios. For detailed API documentation and advanced usage, refer to the main README.md file._
