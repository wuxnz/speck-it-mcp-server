"""MCP server exposing Spec Kit-inspired workflow tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, List

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import TextResource

from lib import (
    FeatureAnalysis,
    FeatureArtifacts,
    SpecKitWorkspace,
    ProjectTask,
    ProjectStatus,
    register_feature_root as _register_feature_root,
    lookup_feature_root as _lookup_feature_root,
)

mcp = FastMCP("speck-it")


PROJECT_MARKER_DIRECTORIES = (".speck-it")
SERVER_ROOT = Path(__file__).resolve().parent


def _candidate_bases() -> List[Path]:
    cwd = Path.cwd().resolve()
    bases: List[Path] = [cwd]
    bases.extend(cwd.parents)
    if SERVER_ROOT not in bases:
        bases.append(SERVER_ROOT)
    for parent in SERVER_ROOT.parents:
        if parent not in bases:
            bases.append(parent)
    seen: set[Path] = set()
    ordered: List[Path] = []
    for base in bases:
        if base not in seen:
            seen.add(base)
            ordered.append(base)
    return ordered


def _locate_workspace_root() -> Optional[Path]:
    for base in _candidate_bases():
        for marker in PROJECT_MARKER_DIRECTORIES:
            if (base / marker).exists():
                return base
    return None


def _locate_existing_storage(feature_id: str) -> Optional[Path]:
    registered = _lookup_feature_root(feature_id)
    if registered:
        for marker in PROJECT_MARKER_DIRECTORIES:
            if (registered / marker / "specs" / feature_id).exists():
                return registered

    for base in _candidate_bases():
        for marker in PROJECT_MARKER_DIRECTORIES:
            if (base / marker / "specs" / feature_id).exists():
                return _register_feature_root(feature_id, base)
    return None


def _resolve_root(root: Optional[str], *, feature_id: Optional[str] = None) -> Path:
    if root:
        resolved = Path(root).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Provided root '{root}' does not exist.")
        return resolved

    env_root = os.getenv("SPECKIT_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if not env_path.exists():
            raise ValueError(
                f"Environment variable SPECKIT_PROJECT_ROOT points to '{env_root}', which does not exist."
            )
        return env_path

    if feature_id:
        spec_root = _locate_existing_storage(feature_id)
        if spec_root:
            return spec_root

    detected_root = _locate_workspace_root()
    if detected_root:
        return detected_root

    raise ValueError(
        "Unable to determine project root automatically. Provide the 'root' argument when calling the tool "
        "or set the SPECKIT_PROJECT_ROOT environment variable."
    )


def _serialize_artifacts(artifacts: FeatureArtifacts) -> Dict[str, Any]:
    return artifacts.to_dict()


def _serialize_analysis(analysis: FeatureAnalysis) -> Dict[str, Any]:
    return analysis.to_dict()


def _workspace(root: Optional[str], *, feature_id: Optional[str] = None) -> SpecKitWorkspace:
    resolved = _resolve_root(root, feature_id=feature_id)
    workspace = SpecKitWorkspace(resolved)
    if feature_id:
        _register_feature_root(feature_id, resolved)
    return workspace


def _workspace_optional(root: Optional[str], *, feature_id: Optional[str] = None) -> Optional[SpecKitWorkspace]:
    try:
        return _workspace(root, feature_id=feature_id)
    except ValueError:
        return None


@mcp.tool()
def set_constitution(content: str, mode: str = "replace", root: Optional[str] = None) -> Dict[str, str]:
    """STEP 1: Create or update the project constitution used for downstream planning.
    This establishes the foundational principles and guidelines for the entire project.
    Should be called first before any feature development."""

    workspace = _workspace(root)
    path = workspace.save_constitution(content, mode=mode)

    # Auto-update project tasks using manage_project_tasks function
    task_update_result = manage_project_tasks(
        action="auto_update",
        feature_id="global",
        root=root
    )
    updated_tasks = task_update_result.get("updated_tasks", [])

    return {
        "constitution_path": str(path),
        "next_suggested_step": "set_feature_root",
        "workflow_tip": "Next: Register your project root with set_feature_root to establish the workspace",
        "auto_updated_tasks": updated_tasks,
        "message": f"Constitution saved. Auto-updated {len(updated_tasks)} project tasks."
    }


@mcp.tool()
def get_constitution(root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Retrieve the current constitution contents, if any."""

    workspace = _workspace(root)
    content = workspace.load_constitution()
    return {"constitution_path": str(workspace.constitution_path), "content": content}


@mcp.tool()
def list_features(root: Optional[str] = None) -> Dict[str, Any]:
    """Enumerate features that have been generated in the workspace."""

    workspace = _workspace_optional(root)
    if not workspace:
        raise ValueError(
            "Unable to determine project root. Provide the 'root' argument or set SPECKIT_PROJECT_ROOT."
        )

    return {"features": workspace.list_features()}


@mcp.resource("speck-it://features")
def resource_features():
    """Resource view exposing generated feature metadata for discovery."""

    workspace = _workspace_optional(None)
    if not workspace:
        return TextResource(
            "No project root detected. Launch tools with a 'root' argument or set SPECKIT_PROJECT_ROOT."
        )

    features = workspace.list_features()
    if not features:
        return TextResource("No features have been generated yet.")

    lines = ["Speck-It Features"]
    for feature in features:
        lines.append("")
        lines.append(f"- {feature['feature_id']}: {feature['feature_name']}")
        if feature.get("spec_path"):
            lines.append(f"  Spec: {feature['spec_path']}")
        if feature.get("plan_path"):
            lines.append(f"  Plan: {feature['plan_path']}")
        if feature.get("tasks_path"):
            lines.append(f"  Tasks: {feature['tasks_path']}")

    return TextResource("\n".join(lines))


@mcp.tool()
def set_feature_root(feature_id: str, root: str) -> Dict[str, str]:
    """STEP 2: Register the project root that owns the feature's `.speck-it/` artifacts.
    This establishes where feature specifications and artifacts will be stored.
    Prerequisites: Project should have a constitution set via set_constitution."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Provided root '{root}' does not exist.")

    for marker in PROJECT_MARKER_DIRECTORIES:
        specs_dir = resolved / marker / "specs"
        if specs_dir.exists():
            _register_feature_root(feature_id, resolved)

            # Auto-update project tasks using manage_project_tasks function
            workspace = _workspace(str(resolved), feature_id=feature_id)
            task_update_result = manage_project_tasks(
                action="auto_update",
                feature_id=feature_id,
                root=str(resolved)
            )
            updated_tasks = task_update_result.get("updated_tasks", [])

            return {
                "feature_id": feature_id,
                "root": str(resolved),
                "marker": marker,
                "next_suggested_step": "generate_spec",
                "workflow_tip": f"Next: Create a specification for feature '{feature_id}' using generate_spec",
                "auto_updated_tasks": updated_tasks,
                "message": f"Feature root registered. Auto-updated {len(updated_tasks)} project tasks."
            }

    raise ValueError(
        f"Root '{resolved}' does not contain any recognized storage directories: {PROJECT_MARKER_DIRECTORIES}."
    )


@mcp.tool()
def generate_spec(
    feature_name: str,
    description: str,
    feature_id: Optional[str] = None,
    root: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """STEP 3: Create a specification artifact for the provided feature description.
    This generates detailed requirements and analysis for the feature.
    Prerequisites: Project root should be registered via set_feature_root."""

    workspace = _workspace(root, feature_id=feature_id)
    artifacts, analysis, content = workspace.generate_spec(
        feature_name,
        description,
        feature_id=feature_id,
        save=save,
    )
    # Auto-update project tasks using manage_project_tasks function
    task_update_result = manage_project_tasks(
        action="auto_update",
        feature_id=feature_id,
        root=root
    )
    updated_tasks = task_update_result.get("updated_tasks", [])

    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
        "next_suggested_step": "generate_plan",
        "workflow_tip": "Next: Generate an implementation plan using generate_plan with the feature_id",
        "auto_updated_tasks": updated_tasks,
        "message": f"Specification generated. Auto-updated {len(updated_tasks)} project tasks."
    }


@mcp.tool()
def generate_plan(
    feature_id: str,
    tech_context: Optional[str] = None,
    root: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """STEP 4: Render an implementation plan using the previously generated spec.
    This creates a detailed technical plan based on the feature specification.
    Prerequisites: Feature spec must exist (created via generate_spec)."""

    workspace = _workspace(root, feature_id=feature_id)

    # Check if spec exists
    if not workspace.spec_exists(feature_id):
        return {
            "error": "No specification found",
            "suggestion": f"Call generate_spec first to create a specification for feature '{feature_id}'",
            "next_suggested_step": "generate_spec"
        }

    artifacts, analysis, content = workspace.generate_plan(
        feature_id,
        tech_context=tech_context,
        save=save,
    )
    # Auto-update project tasks using manage_project_tasks function
    task_update_result = manage_project_tasks(
        action="auto_update",
        feature_id=feature_id,
        root=root
    )
    updated_tasks = task_update_result.get("updated_tasks", [])

    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
        "next_suggested_step": "generate_tasks",
        "workflow_tip": "Next: Generate task list using generate_tasks to break down the plan",
        "auto_updated_tasks": updated_tasks,
        "message": f"Implementation plan generated. Auto-updated {len(updated_tasks)} project tasks."
    }


@mcp.tool()
def generate_tasks(
    feature_id: str,
    root: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """STEP 5: Create a TDD-oriented task list from the existing plan.
    This breaks down the implementation plan into actionable tasks.
    Prerequisites: Implementation plan must exist (created via generate_plan)."""

    workspace = _workspace(root, feature_id=feature_id)

    # Check if plan exists
    if not workspace.plan_exists(feature_id):
        return {
            "error": "No implementation plan found",
            "suggestion": f"Call generate_plan first to create an implementation plan for feature '{feature_id}'",
            "next_suggested_step": "generate_plan"
        }

    artifacts, analysis, content = workspace.generate_tasks(
        feature_id,
        save=save,
    )
    # Auto-update project tasks using manage_project_tasks function
    task_update_result = manage_project_tasks(
        action="auto_update",
        feature_id=feature_id,
        root=root
    )
    updated_tasks = task_update_result.get("updated_tasks", [])

    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
        "next_suggested_step": "list_tasks",
        "workflow_tip": "Next: Use list_tasks to see the generated tasks, then use next_task to start execution",
        "auto_updated_tasks": updated_tasks,
        "message": f"Task list generated. Auto-updated {len(updated_tasks)} project tasks."
    }


@mcp.tool()
def manage_project_tasks(
    action: str,
    feature_id: Optional[str] = None,
    task_id: Optional[str] = None,
    description: Optional[str] = None,
    task_type: str = "custom",
    priority: int = 5,
    dependencies: Optional[List[str]] = None,
    prerequisites: Optional[List[str]] = None,
    estimated_hours: Optional[float] = None,
    status: Optional[str] = None,
    actual_hours: Optional[float] = None,
    add_note: Optional[str] = None,
    add_tag: Optional[str] = None,
    remove_tag: Optional[str] = None,
    tags: Optional[List[str]] = None,
    filter_feature_id: Optional[str] = None,
    filter_status: Optional[str] = None,
    filter_task_type: Optional[str] = None,
    priority_range: Optional[List[int]] = None,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """COMPREHENSIVE TASK MANAGEMENT: Advanced project task management with dependencies, prerequisites, and progress tracking.

    Actions:
    - 'create': Create a new project task
    - 'list': List tasks with optional filtering
    - 'update': Update an existing task
    - 'validate': Validate task prerequisites
    - 'get_next': Get next executable tasks
    - 'get_status': Get comprehensive project status
    - 'auto_update': Automatically update task statuses based on workflow actions

    Prerequisites are validated conditions like:
    - 'constitution_exists': Project constitution is set
    - 'feature_root_registered': Feature root is registered
    - 'spec_exists': Feature specification exists
    - 'plan_exists': Implementation plan exists
    - 'tasks_exist': Task list is generated

    Task Types:
    - 'workflow': Constitution, feature root, spec, plan, tasks generation
    - 'spec': Specification-related tasks
    - 'plan': Planning and design tasks
    - 'implementation': Development and coding tasks
    - 'validation': Testing and quality assurance tasks
    - 'custom': User-defined tasks

    Examples:
    - Create workflow task: action='create', feature_id='feat-001', description='Set project constitution', task_type='workflow', prerequisites=['constitution_exists']
    - List high priority tasks: action='list', priority_range=[1,3]
    - Update task status: action='update', task_id='PROJ-001', status='in_progress'
    - Validate prerequisites: action='validate', task_id='PROJ-001'
    - Get next tasks: action='get_next'
    - Get project status: action='get_status'
    """

    workspace = _workspace(root)

    if action == "create":
        if not feature_id or not description:
            return {"error": "feature_id and description are required for task creation"}

        task = workspace.create_project_task(
            feature_id=feature_id,
            description=description,
            task_type=task_type,
            priority=priority,
            dependencies=dependencies,
            prerequisites=prerequisites,
            estimated_hours=estimated_hours,
            tags=tags,
        )

        return {
            "success": True,
            "task": task.to_dict(),
            "message": f"Created project task {task.task_id}",
            "next_suggested_action": "validate",
            "workflow_tip": f"Validate prerequisites for task {task.task_id} before starting"
        }

    elif action == "list":
        if priority_range and len(priority_range) == 2:
            priority_range_tuple = (priority_range[0], priority_range[1])
        else:
            priority_range_tuple = None

        tasks = workspace.get_project_tasks(
            feature_id=filter_feature_id,
            status=filter_status,
            task_type=filter_task_type,
            priority_range=priority_range_tuple,
        )

        return {
            "tasks": [task.to_dict() for task in tasks],
            "total_count": len(tasks),
            "filters_applied": {
                "feature_id": filter_feature_id,
                "status": filter_status,
                "task_type": filter_task_type,
                "priority_range": priority_range,
            }
        }

    elif action == "update":
        if not task_id:
            return {"error": "task_id is required for task updates"}

        updated_task = workspace.update_project_task(
            task_id=task_id,
            status=status,
            priority=priority,
            actual_hours=actual_hours,
            add_note=add_note,
            add_tag=add_tag,
            remove_tag=remove_tag,
        )

        if not updated_task:
            return {"error": f"Task '{task_id}' not found"}

        return {
            "success": True,
            "task": updated_task.to_dict(),
            "message": f"Updated task {task_id}",
            "next_suggested_action": "get_next" if status == "completed" else "validate",
            "workflow_tip": "Check for newly available tasks" if status == "completed" else f"Validate prerequisites for task {task_id}"
        }

    elif action == "validate":
        if not task_id:
            return {"error": "task_id is required for validation"}

        validation = workspace.validate_task_prerequisites(task_id)

        return {
            "task_id": task_id,
            "validation": validation,
            "can_proceed": validation["can_proceed"],
            "next_suggested_action": "update" if validation["can_proceed"] else "list",
            "workflow_tip": f"Task {task_id} is ready to start" if validation["can_proceed"] else "Resolve validation issues before proceeding"
        }

    elif action == "get_next":
        next_tasks = workspace.get_next_executable_tasks()

        return {
            "executable_tasks": [task.to_dict() for task in next_tasks],
            "count": len(next_tasks),
            "next_suggested_action": "update" if next_tasks else "create",
            "workflow_tip": f"{len(next_tasks)} tasks ready for execution" if next_tasks else "No tasks ready - create new tasks or resolve blockers"
        }

    elif action == "get_status":
        project_status = workspace.get_project_status()

        # Get detailed feature breakdown
        features = workspace.list_features()
        feature_details = []

        for feature in features:
            feature_tasks = workspace.get_project_tasks(feature_id=feature["feature_id"])
            feature_status = workspace.feature_status(feature["feature_id"])

            feature_details.append({
                "feature_id": feature["feature_id"],
                "feature_name": feature.get("feature_name", feature["feature_id"]),
                "project_tasks_count": len(feature_tasks),
                "completed_project_tasks": sum(1 for t in feature_tasks if t.status == "completed"),
                "traditional_tasks": feature_status["tasks"],
            })

        return {
            "project_status": project_status.to_dict(),
            "feature_breakdown": feature_details,
            "next_suggested_action": "get_next",
            "workflow_tip": "Use 'get_next' to find tasks ready for execution"
        }

    elif action == "auto_update":
        if not feature_id:
            return {"error": "feature_id is required for auto-update"}

        # Map common actions to auto-update triggers
        action_map = {
            "set_constitution": "constitution_set",
            "set_feature_root": "feature_root_set",
            "generate_spec": "spec_generated",
            "generate_plan": "plan_generated",
            "generate_tasks": "tasks_generated",
        }

        trigger_action = action_map.get(action, action)
        # Use workspace method directly to avoid circular dependency
        updated_task_ids = workspace.auto_update_task_status(feature_id, trigger_action)

        return {
            "success": True,
            "updated_tasks": updated_task_ids,
            "count": len(updated_task_ids),
            "message": f"Auto-updated {len(updated_task_ids)} tasks based on {action}",
            "next_suggested_action": "get_next",
            "workflow_tip": "Check for newly unlocked tasks"
        }

    else:
        return {
            "error": f"Unknown action '{action}'",
            "available_actions": ["create", "list", "update", "validate", "get_next", "get_status", "auto_update"],
            "next_suggested_action": "get_status",
            "workflow_tip": "Use 'get_status' to understand current project state"
        }


@mcp.tool()
def get_workflow_guide() -> Dict[str, Any]:
    """Get comprehensive guidance on the recommended Speck-It feature development workflow."""
    return {
        "workflow_overview": "Complete feature development workflow in recommended order",
        "steps": [
            {
                "step": 1,
                "tool": "set_constitution",
                "description": "Establish project constitution and foundational principles",
                "purpose": "Define project guidelines and standards for all development"
            },
            {
                "step": 2,
                "tool": "set_feature_root",
                "description": "Register the project root directory for feature artifacts",
                "purpose": "Establish where specifications and plans will be stored"
            },
            {
                "step": 3,
                "tool": "generate_spec",
                "description": "Create detailed feature specification from description",
                "purpose": "Generate comprehensive requirements and analysis"
            },
            {
                "step": 4,
                "tool": "generate_plan",
                "description": "Create implementation plan from specification",
                "purpose": "Develop technical approach and architecture"
            },
            {
                "step": 5,
                "tool": "generate_tasks",
                "description": "Break down plan into actionable TDD-oriented tasks",
                "purpose": "Create executable checklist for implementation"
            },
            {
                "step": 6,
                "tools": ["list_tasks", "next_task", "update_task", "complete_task"],
                "description": "Execute tasks sequentially with progress tracking",
                "purpose": "Implement feature following TDD methodology"
            },
            {
                "step": 7,
                "tool": "finalize_feature",
                "description": "Mark feature complete after all tasks are done",
                "purpose": "Validate completion and archive feature artifacts"
            }
        ],
        "tips": [
            "Always follow steps in order - each step depends on the previous",
            "Use next_task to get guidance on what to work on next",
            "Update tasks with notes as you implement for traceability",
            "Complete tasks as you finish them to track progress",
            "Only finalize when ALL tasks are complete"
        ]
    }


@mcp.tool()
def feature_status(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Return milestone readiness details for the feature."""

    workspace = _workspace(root, feature_id=feature_id)
    return workspace.feature_status(feature_id)


@mcp.tool()
def finalize_feature(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """STEP 7 (FINAL): Validate all artifacts and mark the feature complete only when every task is done.
    This is the final step - only call after ALL tasks are completed.
    Prerequisites: All feature tasks must be completed via complete_task."""

    workspace = _workspace(root, feature_id=feature_id)

    # Check if all tasks are complete
    status = workspace.feature_status(feature_id)
    if not status["tasks"]["all_completed"]:
        incomplete_count = len(status["tasks"]["incomplete"])
        return {
            "error": "Cannot finalize - tasks still incomplete",
            "incomplete_tasks": incomplete_count,
            "suggestion": f"Complete all {incomplete_count} remaining tasks before finalizing",
            "next_suggested_step": "next_task",
            "workflow_tip": "Use next_task to continue working through remaining tasks"
        }

    return workspace.finalize_feature(feature_id)


@mcp.tool()
def list_tasks(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Return the task checklist for a feature, including completion state."""

    workspace = _workspace(root, feature_id=feature_id)
    tasks = workspace.list_tasks(feature_id)
    return {"feature_id": feature_id, "tasks": tasks}


@mcp.tool()
def update_task(
    feature_id: str,
    task_id: str,
    completed: Optional[bool] = None,
    note: Optional[str] = None,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Update task completion or append notes for execution traceability."""

    workspace = _workspace(root, feature_id=feature_id)
    result = workspace.update_task(
        feature_id,
        task_id,
        completed=completed,
        note=note,
    )
    return {
        "feature_id": feature_id,
        "task": result["task"],
        "tasks_path": result["tasks_path"],
        "remaining": result.get("remaining"),
        "all_completed": result.get("all_completed"),
        "next_task": result.get("next_task"),
    }


@mcp.tool()
def next_task(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve the next incomplete task, if any, to guide sequential execution."""

    workspace = _workspace(root, feature_id=feature_id)
    task = workspace.next_open_task(feature_id)
    status = workspace.feature_status(feature_id)
    return {
        "feature_id": feature_id,
        "task": task,
        "remaining": status["tasks"]["remaining"],
    }


@mcp.tool()
def complete_task(
    feature_id: str,
    task_id: Optional[str] = None,
    note: Optional[str] = None,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark a task complete and report remaining work; defaults to the next open task when task_id omitted."""

    workspace = _workspace(root, feature_id=feature_id)
    chosen_task_id = task_id
    if chosen_task_id is None:
        next_item = workspace.next_open_task(feature_id)
        if not next_item:
            raise ValueError(f"Feature '{feature_id}' has no remaining tasks to complete.")
        chosen_task_id = next_item["task_id"]

    result = workspace.complete_task(
        feature_id,
        chosen_task_id,
        note=note,
    )
    status = workspace.feature_status(feature_id)
    return {
        "feature_id": feature_id,
        "task": result["task"],
        "tasks_path": result["tasks_path"],
        "remaining": status["tasks"]["remaining"],
        "all_completed": status["tasks"]["all_completed"],
        "next_task": status["tasks"]["incomplete"][0] if status["tasks"]["incomplete"] else None,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
