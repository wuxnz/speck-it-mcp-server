"""Workflow management for Speck-It.

This module provides the main workflow orchestration, ensuring that
steps are executed in the correct order and managing the overall
feature development process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    FeatureAnalysis,
    FeatureArtifacts,
    ProjectTask,
    ProjectStatus,
    WorkflowStep,
    WORKFLOW_STEPS,
)
from .workspace import Workspace
from .speckit_logging import (
    log_operation,
    log_performance,
    log_workflow_step,
    log_error_with_context,
    observability_hooks,
)


# Global registry for feature roots
_FEATURE_ROOT_REGISTRY: Dict[str, Path] = {}


def register_feature_root(feature_id: str, root: Path | str) -> Path:
    """Record the canonical project root for a feature's artifacts."""
    resolved = Path(root).resolve()
    _FEATURE_ROOT_REGISTRY[feature_id.lower()] = resolved
    return resolved


def lookup_feature_root(feature_id: str) -> Optional[Path]:
    """Return the registered project root for the feature, if any."""
    return _FEATURE_ROOT_REGISTRY.get(feature_id.lower())


class WorkflowManager:
    """Manages the complete Speck-It workflow for feature development."""

    def __init__(self, root: Path | str):
        """Initialize workflow manager with workspace root."""
        self.workspace = Workspace(root)

    # ------------------------------------------------------------------
    # Constitution management
    # ------------------------------------------------------------------

    @log_performance("set_constitution")
    def set_constitution(self, content: str, mode: str = "replace") -> Dict[str, Any]:
        """Set the project constitution."""
        import logging
        logger = logging.getLogger("speckit.workflow")
        
        try:
            # Validate inputs
            if not content or not content.strip():
                raise ValueError("Constitution content cannot be empty")
            if mode not in {"replace", "append"}:
                raise ValueError("Mode must be 'replace' or 'append'")
            
            with log_operation("set_constitution", mode=mode, content_length=len(content)):
                # Save constitution
                path = self.workspace.save_constitution(content, mode=mode)
                
                # Auto-update project tasks
                updated_tasks = self._auto_update_tasks("global", "constitution_set")
                
                # Log successful operation
                log_workflow_step("constitution_set", content_length=len(content))
                logger.info(f"Constitution saved to {path}")
                
                # Trigger observability hooks
                observability_hooks.log_workflow_event(
                    "constitution_set",
                    mode=mode,
                    path=str(path),
                    content_length=len(content),
                    auto_updated_tasks=len(updated_tasks)
                )
                
                return {
                    "constitution_path": str(path),
                    "next_suggested_step": "set_feature_root",
                    "workflow_tip": "Next: Register your project root with set_feature_root to establish the workspace",
                    "auto_updated_tasks": updated_tasks,
                    "message": f"Constitution saved successfully to {path}. Auto-updated {len(updated_tasks)} project tasks."
                }
                
        except Exception as e:
            logger.error(f"Failed to set constitution: {e}")
            log_error_with_context(e, {
                "operation": "set_constitution",
                "mode": mode,
                "content_length": len(content) if content else 0
            })
            
            return {
                "error": f"Failed to save constitution: {e}",
                "suggestion": "Check that the project root exists and is writable",
                "next_suggested_step": "set_constitution",
                "workflow_tip": "Ensure you have write permissions to the project directory",
                "constitution_path": None,
                "auto_updated_tasks": [],
                "message": f"Error: {e}"
            }

    def get_constitution(self) -> Dict[str, Any]:
        """Get the current constitution."""
        try:
            content = self.workspace.load_constitution()
            return {
                "constitution_path": str(self.workspace.constitution_path),
                "content": content,
                "exists": content is not None,
                "message": "Constitution found" if content else "No constitution set yet"
            }
        except Exception as e:
            return {
                "constitution_path": None,
                "content": None,
                "exists": False,
                "error": str(e),
                "message": "No workspace found. Use set_constitution to create a project constitution first.",
                "next_suggested_step": "set_constitution"
            }

    # ------------------------------------------------------------------
    # Feature management
    # ------------------------------------------------------------------

    def list_features(self) -> Dict[str, Any]:
        """List all features in the workspace."""
        features = self.workspace.list_features()
        return {
            "features": features,
            "count": len(features),
            "message": f"Found {len(features)} features" if features else "No features generated yet. Use generate_spec to create your first feature."
        }

    def set_feature_root(self, feature_id: str, root: Optional[str] = None) -> Dict[str, Any]:
        """Set the feature root directory."""
        if root:
            resolved = Path(root).expanduser().resolve()
            if not resolved.exists():
                raise ValueError(f"Provided root '{root}' does not exist.")
        else:
            resolved = self.workspace.root

        # Check if .speck-it directory exists, create if it doesn't
        speck_it_dir = resolved / ".speck-it"
        if not speck_it_dir.exists():
            speck_it_dir.mkdir(parents=True, exist_ok=True)

        register_feature_root(feature_id, resolved)

        # Auto-update project tasks
        updated_tasks = self._auto_update_tasks(feature_id, "feature_root_set")

        return {
            "feature_id": feature_id,
            "root": str(resolved),
            "marker": ".speck-it",
            "next_suggested_step": "generate_spec",
            "workflow_tip": f"Next: Create a specification for feature '{feature_id}' using generate_spec",
            "auto_updated_tasks": updated_tasks,
            "message": f"Feature root registered. Auto-updated {len(updated_tasks)} project tasks."
        }

    # ------------------------------------------------------------------
    # Artifact generation
    # ------------------------------------------------------------------

    def generate_spec(
        self,
        feature_name: str,
        description: str,
        feature_id: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Generate a feature specification."""
        try:
            artifacts, analysis, content = self.workspace.generate_spec(
                feature_name,
                description,
                feature_id=feature_id,
                save=save,
            )
            
            # Auto-update project tasks
            updated_tasks = self._auto_update_tasks(artifacts.feature_id, "spec_generated")
            
            return {
                "artifacts": artifacts.to_dict(),
                "analysis": analysis.to_dict(),
                "content": content,
                "next_suggested_step": "generate_plan",
                "workflow_tip": "Next: Generate an implementation plan using generate_plan with the feature_id",
                "auto_updated_tasks": updated_tasks,
                "message": f"Specification generated. Auto-updated {len(updated_tasks)} project tasks."
            }
        except Exception as e:
            return {
                "error": f"Failed to generate specification: {e}",
                "suggestion": "Check your feature name and description",
                "next_suggested_step": "generate_spec",
                "auto_updated_tasks": [],
                "message": f"Error: {e}"
            }

    def generate_plan(
        self,
        feature_id: str,
        tech_context: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Generate an implementation plan."""
        try:
            # Check if spec exists
            if not self.workspace.spec_exists(feature_id):
                return {
                    "error": "No specification found",
                    "suggestion": f"Call generate_spec first to create a specification for feature '{feature_id}'",
                    "next_suggested_step": "generate_spec"
                }

            artifacts, analysis, content = self.workspace.generate_plan(
                feature_id,
                tech_context=tech_context,
                save=save,
            )
            
            # Auto-update project tasks
            updated_tasks = self._auto_update_tasks(feature_id, "plan_generated")
            
            return {
                "artifacts": artifacts.to_dict(),
                "analysis": analysis.to_dict(),
                "content": content,
                "next_suggested_step": "generate_tasks",
                "workflow_tip": "Next: Generate task list using generate_tasks to break down the plan",
                "auto_updated_tasks": updated_tasks,
                "message": f"Implementation plan generated. Auto-updated {len(updated_tasks)} project tasks."
            }
        except Exception as e:
            return {
                "error": f"Failed to generate plan: {e}",
                "suggestion": f"Ensure specification exists for feature '{feature_id}'",
                "next_suggested_step": "generate_spec",
                "auto_updated_tasks": [],
                "message": f"Error: {e}"
            }

    def generate_tasks(
        self,
        feature_id: str,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Generate tasks for a feature."""
        try:
            # Check if plan exists
            if not self.workspace.plan_exists(feature_id):
                return {
                    "error": "No implementation plan found",
                    "suggestion": f"Call generate_plan first to create an implementation plan for feature '{feature_id}'",
                    "next_suggested_step": "generate_plan"
                }

            artifacts, analysis, content = self.workspace.generate_tasks(
                feature_id,
                save=save,
            )
            
            # Auto-update project tasks
            updated_tasks = self._auto_update_tasks(feature_id, "tasks_generated")
            
            return {
                "artifacts": artifacts.to_dict(),
                "analysis": analysis.to_dict(),
                "content": content,
                "next_suggested_step": "list_tasks",
                "workflow_tip": "Next: Use list_tasks to see the generated tasks, then use next_task to start execution",
                "auto_updated_tasks": updated_tasks,
                "message": f"Task list generated. Auto-updated {len(updated_tasks)} project tasks."
            }
        except Exception as e:
            return {
                "error": f"Failed to generate tasks: {e}",
                "suggestion": f"Ensure implementation plan exists for feature '{feature_id}'",
                "next_suggested_step": "generate_plan",
                "auto_updated_tasks": [],
                "message": f"Error: {e}"
            }

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def list_tasks(self, feature_id: str) -> Dict[str, Any]:
        """List tasks for a feature."""
        try:
            tasks = self.workspace.list_tasks(feature_id)
            return {"feature_id": feature_id, "tasks": tasks}
        except Exception as e:
            return {
                "error": f"Failed to list tasks: {e}",
                "suggestion": f"Generate tasks for feature '{feature_id}' first",
                "next_suggested_step": "generate_tasks"
            }

    def next_task(self, feature_id: str) -> Dict[str, Any]:
        """Get the next task for a feature."""
        try:
            task = self.workspace.next_open_task(feature_id)
            status = self.workspace.feature_status(feature_id)
            return {
                "feature_id": feature_id,
                "task": task,
                "remaining": status["tasks"]["remaining"],
            }
        except Exception as e:
            return {
                "error": f"Failed to get next task: {e}",
                "suggestion": f"Generate tasks for feature '{feature_id}' first",
                "next_suggested_step": "generate_tasks"
            }

    def update_task(
        self,
        feature_id: str,
        task_id: str,
        completed: Optional[bool] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a task."""
        try:
            result = self.workspace.update_task(
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
        except Exception as e:
            return {
                "error": f"Failed to update task: {e}",
                "suggestion": f"Check that task '{task_id}' exists for feature '{feature_id}'",
                "next_suggested_step": "list_tasks"
            }

    def complete_task(
        self,
        feature_id: str,
        task_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete a task."""
        try:
            result = self.workspace.complete_task(
                feature_id,
                task_id or "next",  # Will be handled by workspace
                note=note,
            )
            status = self.workspace.feature_status(feature_id)
            return {
                "feature_id": feature_id,
                "task": result["task"],
                "tasks_path": result["tasks_path"],
                "remaining": status["tasks"]["remaining"],
                "all_completed": status["tasks"]["all_completed"],
                "next_task": status["tasks"]["incomplete"][0] if status["tasks"]["incomplete"] else None,
            }
        except Exception as e:
            return {
                "error": f"Failed to complete task: {e}",
                "suggestion": f"Check that tasks exist for feature '{feature_id}'",
                "next_suggested_step": "list_tasks"
            }

    # ------------------------------------------------------------------
    # Feature status and finalization
    # ------------------------------------------------------------------

    def feature_status(self, feature_id: str) -> Dict[str, Any]:
        """Get the status of a feature."""
        try:
            return self.workspace.feature_status(feature_id)
        except Exception as e:
            return {
                "error": f"Failed to get feature status: {e}",
                "suggestion": f"Check that feature '{feature_id}' exists",
                "next_suggested_step": "list_features"
            }

    def finalize_feature(self, feature_id: str) -> Dict[str, Any]:
        """Finalize a feature."""
        try:
            return self.workspace.finalize_feature(feature_id)
        except Exception as e:
            return {
                "error": f"Failed to finalize feature: {e}",
                "suggestion": f"Ensure all tasks are completed for feature '{feature_id}'",
                "next_suggested_step": "next_task"
            }

    # ------------------------------------------------------------------
    # Workflow guidance
    # ------------------------------------------------------------------

    def get_workflow_guide(self) -> Dict[str, Any]:
        """Get comprehensive workflow guidance."""
        return {
            "workflow_overview": "Complete feature development workflow in recommended order",
            "steps": [
                {
                    "step": step.step_number,
                    "tool": step.tool_name,
                    "description": step.description,
                    "purpose": step.purpose
                }
                for step in WORKFLOW_STEPS
            ],
            "tips": [
                "Always follow steps in order - each step depends on the previous",
                "Use next_task to get guidance on what to work on next",
                "Update tasks with notes as you implement for traceability",
                "Complete tasks as you finish them to track progress",
                "Only finalize when ALL tasks are complete"
            ]
        }

    # ------------------------------------------------------------------
    # Project task management
    # ------------------------------------------------------------------

    def create_project_task(
        self,
        feature_id: str,
        description: str,
        task_type: str = "custom",
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        estimated_hours: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a project-level task."""
        try:
            task = self.workspace.create_project_task(
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
        except Exception as e:
            return {
                "error": f"Failed to create project task: {e}",
                "suggestion": "Check your input parameters",
                "next_suggested_action": "create"
            }

    def get_project_tasks(
        self,
        feature_id: Optional[str] = None,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        priority_range: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Get project tasks with optional filtering."""
        try:
            if priority_range and len(priority_range) == 2:
                priority_range_tuple = (priority_range[0], priority_range[1])
            else:
                priority_range_tuple = None

            tasks = self.workspace.get_project_tasks(
                feature_id=feature_id,
                status=status,
                task_type=task_type,
                priority_range=priority_range_tuple,
            )

            return {
                "tasks": [task.to_dict() for task in tasks],
                "total_count": len(tasks),
                "filters_applied": {
                    "feature_id": feature_id,
                    "status": status,
                    "task_type": task_type,
                    "priority_range": priority_range,
                }
            }
        except Exception as e:
            return {
                "error": f"Failed to get project tasks: {e}",
                "suggestion": "Check your filter parameters",
                "next_suggested_action": "list"
            }

    def update_project_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        actual_hours: Optional[float] = None,
        add_note: Optional[str] = None,
        add_tag: Optional[str] = None,
        remove_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a project task."""
        try:
            updated_task = self.workspace.update_project_task(
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
        except Exception as e:
            return {
                "error": f"Failed to update project task: {e}",
                "suggestion": f"Check that task '{task_id}' exists",
                "next_suggested_action": "list"
            }

    def validate_task_prerequisites(self, task_id: str) -> Dict[str, Any]:
        """Validate task prerequisites."""
        try:
            validation = self.workspace.validate_task_prerequisites(task_id)

            return {
                "task_id": task_id,
                "validation": validation,
                "can_proceed": validation["can_proceed"],
                "next_suggested_action": "update" if validation["can_proceed"] else "list",
                "workflow_tip": f"Task {task_id} is ready to start" if validation["can_proceed"] else "Resolve validation issues before proceeding"
            }
        except Exception as e:
            return {
                "error": f"Failed to validate task prerequisites: {e}",
                "suggestion": f"Check that task '{task_id}' exists",
                "next_suggested_action": "list"
            }

    def get_next_executable_tasks(self) -> Dict[str, Any]:
        """Get tasks that are ready to be executed."""
        try:
            next_tasks = self.workspace.get_next_executable_tasks()

            return {
                "executable_tasks": [task.to_dict() for task in next_tasks],
                "count": len(next_tasks),
                "next_suggested_action": "update" if next_tasks else "create",
                "workflow_tip": f"{len(next_tasks)} tasks ready for execution" if next_tasks else "No tasks ready - create new tasks or resolve blockers"
            }
        except Exception as e:
            return {
                "error": f"Failed to get next executable tasks: {e}",
                "suggestion": "Check your project setup",
                "next_suggested_action": "get_status"
            }

    def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status."""
        try:
            project_status = self.workspace.get_project_status()

            # Get detailed feature breakdown
            features = self.workspace.list_features()
            feature_details = []

            for feature in features:
                feature_tasks = self.workspace.get_project_tasks(feature_id=feature["feature_id"])
                feature_status = self.workspace.feature_status(feature["feature_id"])

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
        except Exception as e:
            return {
                "error": f"Failed to get project status: {e}",
                "suggestion": "Check your workspace setup",
                "next_suggested_action": "list_features"
            }

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _auto_update_tasks(self, feature_id: str, action: str) -> List[str]:
        """Automatically update task statuses based on workflow actions."""
        try:
            return self.workspace.auto_update_task_status(feature_id, action)
        except Exception:
            # If auto-update fails, don't fail the whole operation
            return []