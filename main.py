"""MCP server exposing Spec Kit-inspired workflow tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import TextResource

from lib import (
    FeatureAnalysis,
    FeatureArtifacts,
    SpecKitWorkspace,
    register_feature_root as _register_feature_root,
    lookup_feature_root as _lookup_feature_root,
)

mcp = FastMCP("speck-it")


PROJECT_MARKER_DIRECTORIES = (".speck-it", ".spec-kit")
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
    """Create or update the project constitution used for downstream planning."""

    workspace = _workspace(root)
    path = workspace.save_constitution(content, mode=mode)
    return {"constitution_path": str(path)}


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
    """Explicitly register the project root that owns the feature's `.speck-it/` artifacts."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Provided root '{root}' does not exist.")

    for marker in PROJECT_MARKER_DIRECTORIES:
        specs_dir = resolved / marker / "specs"
        if specs_dir.exists():
            _register_feature_root(feature_id, resolved)
            return {
                "feature_id": feature_id,
                "root": str(resolved),
                "marker": marker,
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
    """Create a specification artifact for the provided feature description."""

    workspace = _workspace(root, feature_id=feature_id)
    artifacts, analysis, content = workspace.generate_spec(
        feature_name,
        description,
        feature_id=feature_id,
        save=save,
    )
    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
    }


@mcp.tool()
def generate_plan(
    feature_id: str,
    tech_context: Optional[str] = None,
    root: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Render an implementation plan using the previously generated spec."""

    workspace = _workspace(root, feature_id=feature_id)
    artifacts, analysis, content = workspace.generate_plan(
        feature_id,
        tech_context=tech_context,
        save=save,
    )
    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
    }


@mcp.tool()
def generate_tasks(
    feature_id: str,
    root: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Create a TDD-oriented task list from the existing plan."""

    workspace = _workspace(root, feature_id=feature_id)
    artifacts, analysis, content = workspace.generate_tasks(
        feature_id,
        save=save,
    )
    return {
        "artifacts": _serialize_artifacts(artifacts),
        "analysis": _serialize_analysis(analysis),
        "content": content,
    }


@mcp.tool()
def feature_status(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Return milestone readiness details for the feature."""

    workspace = _workspace(root, feature_id=feature_id)
    return workspace.feature_status(feature_id)


@mcp.tool()
def finalize_feature(feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Validate all artifacts and mark the feature complete only when every task is done."""

    workspace = _workspace(root, feature_id=feature_id)
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
