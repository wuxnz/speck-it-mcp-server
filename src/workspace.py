"""Workspace management for Speck-It workflow.

This module provides the core workspace functionality for managing
specifications, plans, tasks, and project artifacts.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    FeatureAnalysis,
    FeatureArtifacts,
    ProjectTask,
    ProjectStatus,
    TaskItem,
    WORKFLOW_STEPS,
)
from .speckit_logging import (
    log_operation,
    log_performance,
    log_workflow_step,
    log_spec_generation,
    log_plan_generation,
    log_task_generation,
    log_task_update,
    log_feature_finalization,
    log_error_with_context,
    observability_hooks,
)


class Workspace:
    """Manage Speck-It-inspired artifacts within a repository."""

    STORAGE_DIR_ENV = "SPECKIT_STORAGE_DIR"
    STORAGE_DIR_CANDIDATES = (".speck-it")

    def __init__(self, root: Path | str):
        """Initialize workspace with given root directory."""
        import logging
        
        logger = logging.getLogger("speckit.workspace")
        
        try:
            self.root = Path(root).resolve()
            preferred_name = os.getenv(self.STORAGE_DIR_ENV)
            candidates: List[str] = []
            if preferred_name:
                candidates.append(preferred_name)
            candidates.extend(self.STORAGE_DIR_CANDIDATES)

            base_dir = self.root / ".speck-it"

            self.base_dir = base_dir
            self.memory_dir = self.base_dir / "memory"
            self.specs_dir = self.base_dir / "specs"
            self.data_dir = self.base_dir / "state"
            self.tasks_dir = self.base_dir / "project_tasks"

            # Create directories with error handling
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                self.memory_dir.mkdir(parents=True, exist_ok=True)
                self.specs_dir.mkdir(parents=True, exist_ok=True)
                self.data_dir.mkdir(parents=True, exist_ok=True)
                self.tasks_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create workspace directories: {e}")
                raise RuntimeError(f"Could not initialize workspace at {self.root}: {e}")
            
            logger.info(f"Workspace initialized at {self.root}")
            observability_hooks.log_workflow_event(
                "workspace_initialized",
                root=str(self.root)
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize workspace: {e}")
            log_error_with_context(e, {"operation": "workspace_init", "root": str(root)})
            raise

    # ------------------------------------------------------------------
    # Constitution management
    # ------------------------------------------------------------------

    @property
    def constitution_path(self) -> Path:
        """Get path to constitution file."""
        return self.memory_dir / "constitution.md"

    @log_performance("save_constitution")
    def save_constitution(self, content: str, *, mode: str = "replace") -> Path:
        """Save constitution content to file."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            if mode not in {"replace", "append"}:
                raise ValueError("mode must be 'replace' or 'append'")
            
            path = self.constitution_path
            
            with log_operation("save_constitution", mode=mode, path=str(path)):
                if mode == "replace" or not path.exists():
                    path.write_text(content.strip() + "\n", encoding="utf-8")
                else:
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write("\n" + content.strip() + "\n")
            
            logger.info(f"Constitution saved to {path}")
            observability_hooks.log_workflow_event(
                "constitution_saved",
                mode=mode,
                path=str(path),
                content_length=len(content)
            )
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to save constitution: {e}")
            log_error_with_context(e, {
                "operation": "save_constitution",
                "mode": mode,
                "path": str(self.constitution_path)
            })
            raise

    def load_constitution(self) -> Optional[str]:
        """Load constitution content from file."""
        if not self.constitution_path.exists():
            return None
        return self.constitution_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def list_features(self) -> List[Dict[str, str]]:
        """List all features in the workspace."""
        features: List[Dict[str, str]] = []
        for path in sorted(self.specs_dir.glob("*/")):
            feature_id = path.name.rstrip("/")
            analysis = self._load_analysis(feature_id)
            features.append(
                {
                    "feature_id": feature_id,
                    "feature_name": analysis.feature_name if analysis else feature_id,
                    "spec_path": str((path / "spec.md").resolve()) if (path / "spec.md").exists() else None,
                    "plan_path": str((path / "plan.md").resolve()) if (path / "plan.md").exists() else None,
                    "tasks_path": str((path / "tasks.md").resolve()) if (path / "tasks.md").exists() else None,
                }
            )
        return features

    def _next_feature_number(self) -> int:
        """Get the next available feature number."""
        highest = 0
        for directory in self.specs_dir.glob("*/"):
            name = directory.name.rstrip("/")
            match = re.match(r"(\d{3})-", name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _feature_identifier(self, feature_name: str, feature_id: Optional[str]) -> str:
        """Generate a unique feature identifier."""
        if feature_id:
            slug = self._slugify(feature_id)
            if not re.match(r"^\d{3}-", slug):
                slug = f"{self._next_feature_number():03d}-{slug}"
            return slug
        slug = self._slugify(feature_name)
        number = self._next_feature_number()
        candidate = f"{number:03d}-{slug}"
        while (self.specs_dir / candidate).exists():
            number += 1
            candidate = f"{number:03d}-{slug}"
        return candidate

    def _feature_dir(self, feature_id: str) -> Path:
        """Get the directory for a feature."""
        path = self.specs_dir / feature_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _analysis_path(self, feature_dir: Path) -> Path:
        """Get the analysis file path for a feature."""
        return feature_dir / "analysis.json"

    def _save_analysis(self, feature_dir: Path, analysis: FeatureAnalysis) -> None:
        """Save feature analysis to file."""
        path = self._analysis_path(feature_dir)
        path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")

    def _load_analysis(self, feature_id: str) -> Optional[FeatureAnalysis]:
        """Load feature analysis from file."""
        feature_dir = self.specs_dir / feature_id
        path = self._analysis_path(feature_dir)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return FeatureAnalysis.from_dict(data)

    # ------------------------------------------------------------------
    # Artifact existence checks
    # ------------------------------------------------------------------

    def spec_exists(self, feature_id: str) -> bool:
        """Check if a specification exists for the given feature."""
        feature_dir = self._feature_dir(feature_id)
        return (feature_dir / "spec.md").exists()

    def plan_exists(self, feature_id: str) -> bool:
        """Check if an implementation plan exists for the given feature."""
        feature_dir = self._feature_dir(feature_id)
        return (feature_dir / "plan.md").exists()

    def tasks_exists(self, feature_id: str) -> bool:
        """Check if tasks exist for the given feature."""
        feature_dir = self._feature_dir(feature_id)
        return (feature_dir / "tasks.md").exists()

    # ------------------------------------------------------------------
    # Artifact generation
    # ------------------------------------------------------------------

    @log_performance("generate_spec")
    def generate_spec(
        self,
        feature_name: str,
        description: str,
        *,
        feature_id: Optional[str] = None,
        save: bool = True,
    ) -> tuple[FeatureArtifacts, FeatureAnalysis, str]:
        """Generate a feature specification."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Validate inputs
            if not feature_name or not feature_name.strip():
                raise ValueError("Feature name cannot be empty")
            if not description or not description.strip():
                raise ValueError("Description cannot be empty")
            
            feature_identifier = self._feature_identifier(feature_name, feature_id)
            
            with log_operation("generate_spec",
                            feature_id=feature_identifier,
                            feature_name=feature_name,
                            save=save):
                
                # Extract and analyze the description
                actors = self._extract_actors(description)
                sentences = self._split_sentences(description)
                actions = self._extract_actions(sentences)
                goals = self._extract_goals(actions)
                primary_story = self._build_primary_story(actors, goals, feature_name, description)
                clarifications = self._identify_clarifications(description, actions)
                edge_cases = self._build_edge_cases(actions)
                summary = self._build_summary(feature_name, actors, goals)
                keywords = sorted({kw for sentence in sentences for kw in self._sentence_keywords(sentence)})

                # Create analysis object
                analysis = FeatureAnalysis(
                    feature_id=feature_identifier,
                    feature_name=feature_name,
                    description=description,
                    primary_story=primary_story,
                    actors=actors,
                    actions=actions,
                    goals=goals,
                    clarifications=clarifications,
                    edge_cases=edge_cases,
                    summary=summary,
                    keywords=keywords,
                )

                # Validate analysis
                validation_issues = analysis.validate()
                if validation_issues:
                    logger.warning(f"Specification validation issues: {validation_issues}")
                    observability_hooks.log_workflow_event(
                        "spec_validation_issues",
                        feature_id=feature_identifier,
                        issues=validation_issues
                    )

                # Render specification content
                spec_content = self._render_spec(analysis)

                # Save artifacts if requested
                feature_dir = self._feature_dir(feature_identifier) if save else self.specs_dir / feature_identifier
                spec_path = feature_dir / "spec.md" if save else None
                
                if save:
                    try:
                        feature_dir.mkdir(parents=True, exist_ok=True)
                        spec_path.write_text(spec_content, encoding="utf-8")
                        self._save_analysis(feature_dir, analysis)
                        
                        logger.info(f"Specification saved to {spec_path}")
                        
                    except OSError as e:
                        logger.error(f"Failed to save specification files: {e}")
                        raise RuntimeError(f"Could not save specification for feature '{feature_identifier}': {e}")

                # Create artifacts object
                artifacts = FeatureArtifacts(
                    feature_id=feature_identifier,
                    feature_dir=feature_dir,
                    spec_path=spec_path,
                )
                
                # Log successful generation
                log_spec_generation(feature_identifier, feature_name,
                                  actors_count=len(actors),
                                  actions_count=len(actions),
                                  clarifications_count=len(clarifications))
                
                logger.info(f"Generated specification for feature '{feature_identifier}'")
                
                return artifacts, analysis, spec_content
                
        except Exception as e:
            logger.error(f"Failed to generate specification for feature '{feature_name}': {e}")
            log_error_with_context(e, {
                "operation": "generate_spec",
                "feature_name": feature_name,
                "feature_id": feature_id,
                "save": save
            })
            raise

    def generate_plan(
        self,
        feature_id: str,
        *,
        tech_context: Optional[str] = None,
        save: bool = True,
    ) -> tuple[FeatureArtifacts, FeatureAnalysis, str]:
        """Generate an implementation plan for a feature."""
        analysis = self._load_analysis(feature_id)
        if not analysis:
            raise FileNotFoundError(
                f"Feature '{feature_id}' does not have analysis data. Generate a spec first."
            )

        feature_dir = self._feature_dir(feature_id)
        spec_path = feature_dir / "spec.md"
        if not spec_path.exists():
            raise FileNotFoundError(f"Expected spec at {spec_path}. Generate it before planning.")

        constitution_excerpt = self._constitution_excerpt()
        context_map = self._parse_context(tech_context)
        plan_content = self._render_plan(analysis, spec_path, constitution_excerpt, context_map)

        plan_path = feature_dir / "plan.md" if save else None
        if save:
            plan_path.write_text(plan_content, encoding="utf-8")

        artifacts = FeatureArtifacts(
            feature_id=feature_id,
            feature_dir=feature_dir,
            spec_path=spec_path,
            plan_path=plan_path,
        )
        return artifacts, analysis, plan_content

    def generate_tasks(
        self,
        feature_id: str,
        *,
        save: bool = True,
    ) -> tuple[FeatureArtifacts, FeatureAnalysis, str]:
        """Generate tasks for a feature."""
        analysis = self._load_analysis(feature_id)
        if not analysis:
            raise FileNotFoundError(
                f"Feature '{feature_id}' does not have analysis data. Generate a spec first."
            )
        feature_dir = self._feature_dir(feature_id)
        plan_path = feature_dir / "plan.md"
        if not plan_path.exists():
            raise FileNotFoundError(
                f"Expected plan at {plan_path}. Generate it before creating tasks."
            )
        plan_content = plan_path.read_text(encoding="utf-8")
        tasks_content = self._render_tasks(analysis, plan_content)

        tasks_path = feature_dir / "tasks.md" if save else None
        if save:
            tasks_path.write_text(tasks_content, encoding="utf-8")

        artifacts = FeatureArtifacts(
            feature_id=feature_id,
            feature_dir=feature_dir,
            spec_path=feature_dir / "spec.md" if (feature_dir / "spec.md").exists() else None,
            plan_path=plan_path,
            tasks_path=tasks_path,
        )
        return artifacts, analysis, tasks_content

    # ------------------------------------------------------------------
    # Task progress management
    # ------------------------------------------------------------------

    def _tasks_path(self, feature_id: str) -> Path:
        """Get the tasks file path for a feature."""
        feature_dir = self._feature_dir(feature_id)
        return feature_dir / "tasks.md"

    def list_tasks(self, feature_id: str) -> List[Dict[str, Any]]:
        """List all tasks for a feature."""
        tasks = self._read_tasks(feature_id)
        return [task.to_dict() for task in tasks]

    @log_performance("update_task")
    def update_task(
        self,
        feature_id: str,
        task_id: str,
        *,
        completed: Optional[bool] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a task's completion status or add notes."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Validate inputs
            if not feature_id or not feature_id.strip():
                raise ValueError("Feature ID cannot be empty")
            if not task_id or not task_id.strip():
                raise ValueError("Task ID cannot be empty")
            
            with log_operation("update_task",
                            feature_id=feature_id,
                            task_id=task_id,
                            completed=completed,
                            has_note=note is not None):
                
                # Check if tasks file exists
                tasks_path = self._tasks_path(feature_id)
                if not tasks_path.exists():
                    raise FileNotFoundError(
                        f"No tasks.md found for feature '{feature_id}'. Generate tasks before updating."
                    )

                # Read and find the task
                tasks = self._read_tasks(feature_id)
                task_map = {task.task_id.upper(): task for task in tasks}
                lookup = task_map.get(task_id.upper())
                if not lookup:
                    raise ValueError(f"Task '{task_id}' not found for feature '{feature_id}'.")

                # Parse the task line
                lines = tasks_path.read_text(encoding="utf-8").splitlines()
                line = lines[lookup.line_index]
                match = self._TASK_LINE_PATTERN.match(line)
                if not match:
                    raise RuntimeError(f"Unable to parse stored task line for '{task_id}'.")

                # Update task status
                mark = match.group("mark")
                rest = match.group("rest")
                previously_completed = lookup.completed
                status_changed = False

                if completed is not None:
                    new_mark = "x" if completed else " "
                    if mark != new_mark:
                        mark = new_mark
                        lookup.completed = completed
                        status_changed = True

                # Add note if needed
                note_to_add = note
                if note_to_add is None and completed and not previously_completed:
                    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    note_to_add = f"Completed via API {timestamp}"
                elif note_to_add is None and completed and previously_completed:
                    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    note_to_add = f"Completion reconfirmed {timestamp}"

                if note_to_add:
                    note_line = f"{lookup.indent}  - Note: {note_to_add}"
                    insert_index = lookup.line_index + 1
                    while insert_index < len(lines) and lines[insert_index].startswith(lookup.indent + "  - "):
                        insert_index += 1
                    lines.insert(insert_index, note_line)
                    lookup.notes.append(note_to_add)

                # Write updated content
                lines[lookup.line_index] = f"{lookup.indent}- [{mark}] {rest}"
                tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                # Get updated status
                status = self.feature_status(feature_id)
                next_task = status["tasks"]["incomplete"][0] if status["tasks"]["incomplete"] else None

                # Log the update
                log_task_update(feature_id, task_id, lookup.completed,
                              status_changed=status_changed,
                              note_added=note_to_add is not None)
                
                logger.info(f"Updated task '{task_id}' for feature '{feature_id}'")

                return {
                    "task": lookup.to_dict(),
                    "tasks_path": str(tasks_path),
                    "remaining": status["tasks"]["remaining"],
                    "all_completed": status["tasks"]["all_completed"],
                    "next_task": next_task,
                }
                
        except Exception as e:
            logger.error(f"Failed to update task '{task_id}' for feature '{feature_id}': {e}")
            log_error_with_context(e, {
                "operation": "update_task",
                "feature_id": feature_id,
                "task_id": task_id,
                "completed": completed,
                "has_note": note is not None
            })
            raise

    def next_open_task(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get the next open task for a feature."""
        tasks = self._read_tasks(feature_id)
        for task in tasks:
            if not task.completed:
                return task.to_dict()
        return None

    def complete_task(
        self,
        feature_id: str,
        task_id: str,
        *,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a task as completed."""
        note_to_use = note or None
        result = self.update_task(
            feature_id,
            task_id,
            completed=True,
            note=note_to_use,
        )
        if result["task"]["completed"] is False:
            raise RuntimeError(f"Failed to mark task '{task_id}' as completed.")
        return result

    def feature_status(self, feature_id: str) -> Dict[str, Any]:
        """Get the status of a feature."""
        feature_dir = self._feature_dir(feature_id)
        spec_path = feature_dir / "spec.md"
        plan_path = feature_dir / "plan.md"
        tasks_path = self._tasks_path(feature_id)

        try:
            tasks = self._read_tasks(feature_id)
        except FileNotFoundError:
            tasks = []

        total = len(tasks)
        completed = sum(1 for task in tasks if task.completed)
        incomplete_items = [task.to_dict() for task in tasks if not task.completed]

        return {
            "feature_id": feature_id,
            "spec_path": str(spec_path) if spec_path.exists() else None,
            "plan_path": str(plan_path) if plan_path.exists() else None,
            "tasks_path": str(tasks_path) if tasks_path.exists() else None,
            "tasks": {
                "total": total,
                "completed": completed,
                "remaining": total - completed,
                "all_completed": total > 0 and completed == total,
                "items": [task.to_dict() for task in tasks],
                "incomplete": incomplete_items,
            },
        }

    @log_performance("finalize_feature")
    def finalize_feature(self, feature_id: str) -> Dict[str, Any]:
        """Finalize a feature after all tasks are completed."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Validate input
            if not feature_id or not feature_id.strip():
                raise ValueError("Feature ID cannot be empty")
            
            with log_operation("finalize_feature", feature_id=feature_id):
                
                # Get current status
                status = self.feature_status(feature_id)

                # Validate prerequisites
                if not status["spec_path"]:
                    raise ValueError(
                        f"Cannot finalize feature '{feature_id}': spec.md not found. Generate the specification first."
                    )
                if not status["plan_path"]:
                    raise ValueError(
                        f"Cannot finalize feature '{feature_id}': plan.md not found. Generate the plan before finalizing."
                    )
                if not status["tasks_path"]:
                    raise ValueError(
                        f"Cannot finalize feature '{feature_id}': tasks.md not found. Generate tasks before finalizing."
                    )

                tasks_info = status["tasks"]
                if tasks_info["total"] == 0:
                    raise ValueError(
                        f"Cannot finalize feature '{feature_id}': no tasks were recorded in tasks.md."
                    )
                if not tasks_info["all_completed"]:
                    pending_ids = [item["task_id"] for item in tasks_info["incomplete"]]
                    raise ValueError(
                        f"Cannot finalize feature '{feature_id}': pending tasks remain -> "
                        + ", ".join(pending_ids)
                    )

                # Mark as finalized
                status["finalized_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                status_path = self.data_dir / f"{feature_id}_status.json"
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                status["status_path"] = str(status_path)

                # Log the finalization
                log_feature_finalization(feature_id,
                                      total_tasks=tasks_info["total"],
                                      completed_tasks=tasks_info["completed"])
                
                logger.info(f"Finalized feature '{feature_id}' with {tasks_info['total']} tasks")
                
                # Trigger observability hooks
                observability_hooks.log_workflow_event(
                    "feature_finalized",
                    feature_id=feature_id,
                    total_tasks=tasks_info["total"],
                    completed_tasks=tasks_info["completed"],
                    finalized_at=status["finalized_at"]
                )
                
                return status
                
        except Exception as e:
            logger.error(f"Failed to finalize feature '{feature_id}': {e}")
            log_error_with_context(e, {
                "operation": "finalize_feature",
                "feature_id": feature_id
            })
            raise
    
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
    ) -> ProjectTask:
        """Create a new project task."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Validate inputs
            if not feature_id or not feature_id.strip():
                raise ValueError("Feature ID cannot be empty")
            if not description or not description.strip():
                raise ValueError("Description cannot be empty")
            
            # Generate task ID
            task_id = _generate_task_id()
            
            # Create task
            task = ProjectTask(
                task_id=task_id,
                feature_id=feature_id,
                description=description,
                task_type=task_type,
                priority=priority,
                dependencies=dependencies or [],
                prerequisites=prerequisites or [],
                estimated_hours=estimated_hours,
                tags=tags or []
            )
            
            # Validate task
            validation_issues = task.validate()
            if validation_issues:
                logger.warning(f"Task validation issues: {validation_issues}")
            
            # Load existing tasks
            tasks = _load_project_tasks(feature_id)
            
            # Add new task
            tasks.append(task)
            
            # Save tasks
            _save_project_tasks(feature_id, tasks)
            
            logger.info(f"Created project task {task_id} for feature {feature_id}")
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to create project task: {e}")
            log_error_with_context(e, {
                "operation": "create_project_task",
                "feature_id": feature_id,
                "description": description
            })
            raise
    
    def get_project_tasks(
        self,
        feature_id: Optional[str] = None,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        priority_range: Optional[tuple[int, int]] = None,
    ) -> List[ProjectTask]:
        """Get project tasks with optional filtering."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Load all tasks
            all_tasks = []
            if feature_id:
                all_tasks = _load_project_tasks(feature_id)
            else:
                # Load tasks from all features
                for tasks_dir in self.tasks_dir.glob("*.json"):
                    task_feature_id = tasks_dir.stem[:-6]  # Remove "_tasks" suffix
                    all_tasks.extend(_load_project_tasks(task_feature_id))
            
            # Apply filters
            filtered_tasks = []
            for task in all_tasks:
                # Filter by status
                if status and task.status != status:
                    continue
                
                # Filter by task type
                if task_type and task.task_type != task_type:
                    continue
                
                # Filter by priority range
                if priority_range and not (priority_range[0] <= task.priority <= priority_range[1]):
                    continue
                
                filtered_tasks.append(task)
            
            # Sort by priority (lower number = higher priority)
            filtered_tasks.sort(key=lambda t: t.priority)
            
            return filtered_tasks
            
        except Exception as e:
            logger.error(f"Failed to get project tasks: {e}")
            log_error_with_context(e, {
                "operation": "get_project_tasks",
                "feature_id": feature_id,
                "status": status,
                "task_type": task_type,
                "priority_range": priority_range
            })
            return []
    
    def update_project_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        actual_hours: Optional[float] = None,
        add_note: Optional[str] = None,
        add_tag: Optional[str] = None,
        remove_tag: Optional[str] = None,
    ) -> Optional[ProjectTask]:
        """Update an existing project task."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Find the task
            task = None
            feature_id = None
            
            # Search in all feature task files
            for tasks_file in self.tasks_dir.glob("*.json"):
                task_feature_id = tasks_file.stem[:-6]  # Remove "_tasks" suffix
                tasks = _load_project_tasks(task_feature_id)
                
                for t in tasks:
                    if t.task_id == task_id:
                        task = t
                        feature_id = task_feature_id
                        break
                
                if task:
                    break
            
            if not task:
                logger.error(f"Task '{task_id}' not found")
                return None
            
            # Update task properties
            if status is not None:
                task.update_status(status, add_note)
            
            if priority is not None:
                task.priority = priority
                task.updated_at = datetime.utcnow().isoformat() + "Z"
            
            if actual_hours is not None:
                task.actual_hours = actual_hours
                task.updated_at = datetime.utcnow().isoformat() + "Z"
            
            if add_note:
                task.add_note(add_note)
            
            if add_tag:
                if add_tag not in task.tags:
                    task.tags.append(add_tag)
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
            
            if remove_tag and remove_tag in task.tags:
                task.tags.remove(remove_tag)
                task.updated_at = datetime.utcnow().isoformat() + "Z"
            
            # Save updated tasks
            tasks = _load_project_tasks(feature_id)
            for i, t in enumerate(tasks):
                if t.task_id == task_id:
                    tasks[i] = task
                    break
            
            _save_project_tasks(feature_id, tasks)
            
            logger.info(f"Updated project task {task_id}")
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to update project task '{task_id}': {e}")
            log_error_with_context(e, {
                "operation": "update_project_task",
                "task_id": task_id,
                "status": status,
                "priority": priority,
                "actual_hours": actual_hours,
                "add_note": add_note,
                "add_tag": add_tag,
                "remove_tag": remove_tag
            })
            return None
    
    def validate_task_prerequisites(self, task_id: str) -> Dict[str, Any]:
        """Validate task prerequisites."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Find the task
            task = None
            feature_id = None
            
            # Search in all feature task files
            for tasks_file in self.tasks_dir.glob("*.json"):
                task_feature_id = tasks_file.stem[:-6]  # Remove "_tasks" suffix
                tasks = _load_project_tasks(task_feature_id)
                
                for t in tasks:
                    if t.task_id == task_id:
                        task = t
                        feature_id = task_feature_id
                        break
                
                if task:
                    break
            
            if not task:
                return {
                    "task_id": task_id,
                    "can_proceed": False,
                    "error": "Task not found"
                }
            
            # Check prerequisites
            unmet_prerequisites = []
            for prerequisite in task.prerequisites:
                if prerequisite == "constitution_exists":
                    if not self.constitution_path.exists():
                        unmet_prerequisites.append("Constitution must be set first")
                elif prerequisite == "feature_root_registered":
                    # Check if feature root is registered
                    from .workflow import lookup_feature_root
                    if not lookup_feature_root(feature_id):
                        unmet_prerequisites.append("Feature root must be registered first")
                elif prerequisite == "spec_exists":
                    if not self.spec_exists(feature_id):
                        unmet_prerequisites.append("Specification must be generated first")
                elif prerequisite == "plan_exists":
                    if not self.plan_exists(feature_id):
                        unmet_prerequisites.append("Implementation plan must be generated first")
                elif prerequisite == "tasks_exist":
                    if not self.tasks_exists(feature_id):
                        unmet_prerequisites.append("Tasks must be generated first")
                else:
                    # Check if it's a task dependency
                    dep_task = self.get_project_tasks(feature_id, status="completed")
                    if not any(t.task_id == prerequisite for t in dep_task):
                        unmet_prerequisites.append(f"Task '{prerequisite}' must be completed first")
            
            # Check dependencies
            unmet_dependencies = []
            for dependency in task.dependencies:
                dep_task = self.get_project_tasks(feature_id, status="completed")
                if not any(t.task_id == dependency for t in dep_task):
                    unmet_dependencies.append(f"Task '{dependency}' must be completed first")
            
            can_proceed = not unmet_prerequisites and not unmet_dependencies
            
            return {
                "task_id": task_id,
                "can_proceed": can_proceed,
                "unmet_prerequisites": unmet_prerequisites,
                "unmet_dependencies": unmet_dependencies,
                "all_blockers": unmet_prerequisites + unmet_dependencies
            }
            
        except Exception as e:
            logger.error(f"Failed to validate task prerequisites for '{task_id}': {e}")
            log_error_with_context(e, {
                "operation": "validate_task_prerequisites",
                "task_id": task_id
            })
            return {
                "task_id": task_id,
                "can_proceed": False,
                "error": str(e)
            }
    
    def get_next_executable_tasks(self) -> List[ProjectTask]:
        """Get the next executable tasks."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Get all pending tasks
            all_pending_tasks = self.get_project_tasks(status="pending")
            
            # Filter tasks that can proceed
            executable_tasks = []
            for task in all_pending_tasks:
                validation = self.validate_task_prerequisites(task.task_id)
                if validation["can_proceed"]:
                    executable_tasks.append(task)
            
            # Sort by priority (lower number = higher priority)
            executable_tasks.sort(key=lambda t: t.priority)
            
            return executable_tasks
            
        except Exception as e:
            logger.error(f"Failed to get next executable tasks: {e}")
            log_error_with_context(e, {
                "operation": "get_next_executable_tasks"
            })
            return []
    
    def get_project_status(self) -> ProjectStatus:
        """Get the overall project status."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            # Get all tasks
            all_tasks = []
            for tasks_file in self.tasks_dir.glob("*.json"):
                task_feature_id = tasks_file.stem[:-6]  # Remove "_tasks" suffix
                all_tasks.extend(_load_project_tasks(task_feature_id))
            
            # Calculate status metrics
            total_tasks = len(all_tasks)
            completed_tasks = sum(1 for task in all_tasks if task.status == "completed")
            in_progress_tasks = sum(1 for task in all_tasks if task.status == "in_progress")
            blocked_tasks = sum(1 for task in all_tasks if task.status == "blocked")
            
            high_priority_tasks = sum(1 for task in all_tasks if task.is_high_priority())
            
            # Calculate hours
            estimated_hours_remaining = sum(
                task.estimated_hours or 0 for task in all_tasks
                if task.status in ["pending", "in_progress"]
            )
            actual_hours_spent = sum(task.actual_hours or 0 for task in all_tasks)
            
            # Get project name from constitution
            project_name = "Speck-It Project"
            constitution = self.load_constitution()
            if constitution:
                lines = constitution.strip().split('\n')
                if lines and lines[0].startswith("# "):
                    project_name = lines[0][2:].strip()
            
            return ProjectStatus(
                project_name=project_name,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                in_progress_tasks=in_progress_tasks,
                blocked_tasks=blocked_tasks,
                high_priority_tasks=high_priority_tasks,
                estimated_hours_remaining=estimated_hours_remaining,
                actual_hours_spent=actual_hours_spent
            )
            
        except Exception as e:
            logger.error(f"Failed to get project status: {e}")
            log_error_with_context(e, {
                "operation": "get_project_status"
            })
            return ProjectStatus(project_name="Speck-It Project")
    
    def auto_update_task_status(self, feature_id: str, action: str) -> List[str]:
        """Automatically update task statuses based on workflow actions."""
        import logging
        logger = logging.getLogger("speckit.workspace")
        
        try:
            updated_task_ids = []
            
            # Get all tasks for the feature
            tasks = self.get_project_tasks(feature_id)
            
            # Map actions to task updates
            if action == "constitution_set":
                # Mark any "constitution_exists" prerequisites as completed
                for task in tasks:
                    if "constitution_exists" in task.prerequisites:
                        # No need to update, just log
                        logger.debug(f"Task {task.task_id} prerequisite 'constitution_exists' satisfied")
            
            elif action == "feature_root_set":
                # Mark any "feature_root_registered" prerequisites as completed
                for task in tasks:
                    if "feature_root_registered" in task.prerequisites:
                        # No need to update, just log
                        logger.debug(f"Task {task.task_id} prerequisite 'feature_root_registered' satisfied")
            
            elif action == "spec_generated":
                # Mark any "spec_exists" prerequisites as completed
                for task in tasks:
                    if "spec_exists" in task.prerequisites:
                        # No need to update, just log
                        logger.debug(f"Task {task.task_id} prerequisite 'spec_exists' satisfied")
            
            elif action == "plan_generated":
                # Mark any "plan_exists" prerequisites as completed
                for task in tasks:
                    if "plan_exists" in task.prerequisites:
                        # No need to update, just log
                        logger.debug(f"Task {task.task_id} prerequisite 'plan_exists' satisfied")
            
            elif action == "tasks_generated":
                # Mark any "tasks_exists" prerequisites as completed
                for task in tasks:
                    if "tasks_exists" in task.prerequisites:
                        # No need to update, just log
                        logger.debug(f"Task {task.task_id} prerequisite 'tasks_exists' satisfied")
            
            # Check if any tasks can now be executed
            for task in tasks:
                if task.status == "pending":
                    validation = self.validate_task_prerequisites(task.task_id)
                    if validation["can_proceed"]:
                        # Task can now be executed, but we don't change the status
                        logger.debug(f"Task {task.task_id} is now executable")
            
            return updated_task_ids
            
        except Exception as e:
            logger.error(f"Failed to auto-update task status for feature '{feature_id}': {e}")
            log_error_with_context(e, {
                "operation": "auto_update_task_status",
                "feature_id": feature_id,
                "action": action
            })
            return []
    
    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _read_tasks(self, feature_id: str) -> List[TaskItem]:
        """Read tasks from the tasks file."""
        tasks_path = self._tasks_path(feature_id)
        if not tasks_path.exists():
            raise FileNotFoundError(
                f"No tasks.md found for feature '{feature_id}'. Generate tasks before listing."
            )
        lines = tasks_path.read_text(encoding="utf-8").splitlines()
        tasks: List[TaskItem] = []
        current_task: Optional[TaskItem] = None
        for idx, line in enumerate(lines):
            task_match = self._TASK_LINE_PATTERN.match(line)
            if task_match:
                rest = task_match.group("rest").strip()
                task_id_match = self._TASK_ID_PATTERN.search(rest)
                task_id = task_id_match.group(0).upper() if task_id_match else f"LINE-{idx+1}"
                completed = task_match.group("mark").lower() == "x"
                current_task = TaskItem(
                    task_id=task_id,
                    description=rest,
                    completed=completed,
                    line_index=idx,
                    indent=task_match.group("prefix"),
                )
                tasks.append(current_task)
                continue

            note_match = self._NOTE_LINE_PATTERN.match(line)
            if note_match and current_task:
                current_task.notes.append(note_match.group("note").strip())

        return tasks

    # ------------------------------------------------------------------
    # Text processing utilities
    # ------------------------------------------------------------------

    _TASK_LINE_PATTERN = re.compile(r"^(?P<prefix>\s*)-\s*\[(?P<mark> |x|X)\]\s*(?P<rest>.+)$")
    _TASK_ID_PATTERN = re.compile(r"\bT\d{3,}\b", re.IGNORECASE)
    _NOTE_LINE_PATTERN = re.compile(r"^\s{2,}-\s*Note:\s*(?P<note>.+)$", re.IGNORECASE)

    def _slugify(self, value: str) -> str:
        """Convert a string to a slug."""
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "feature"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def _sentence_keywords(self, sentence: str) -> List[str]:
        """Extract keywords from a sentence."""
        return [w for w in re.split(r"[^a-z0-9]+", sentence.lower()) if len(w) > 3]

    def _extract_actors(self, description: str) -> List[str]:
        """Extract actors from a description."""
        actor_keywords = {
            "administrators": ("admin", "administrator", "admins"),
            "operators": ("operator", "operators"),
            "customers": ("customer", "customers"),
            "team members": ("team", "teams", "teammate", "teammates"),
            "managers": ("manager", "managers"),
            "developers": ("developer", "developers", "engineer", "engineers"),
            "analysts": ("analyst", "analysts"),
            "moderators": ("moderator", "moderators"),
        }
        
        detected: List[str] = []
        lower = description.lower()
        for canonical, variants in actor_keywords.items():
            if any(term in lower for term in variants):
                detected.append(canonical)
        if "user" in lower and "users" not in detected:
            detected.append("users")
        if not detected:
            detected.append("users")
        return detected

    def _extract_actions(self, sentences: Iterable[str]) -> List[str]:
        """Extract actions from sentences."""
        actions: List[str] = []
        for sentence in sentences:
            normalized = sentence.strip().rstrip(".!?")
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered.startswith(("allow", "enable", "support", "provide")):
                action = normalized[0].upper() + normalized[1:]
            else:
                action = f"Enable users to {lowered}"
                action = action[0].upper() + action[1:]
            if action not in actions:
                actions.append(action)
        return actions

    def _extract_goals(self, actions: Iterable[str]) -> List[str]:
        """Extract goals from actions."""
        goals: List[str] = []
        for action in actions:
            goal = action
            if goal.lower().startswith("enable users to "):
                goal = goal[len("Enable users to ") :]
            elif goal.lower().startswith("allow users to "):
                goal = goal[len("Allow users to ") :]
            goals.append(goal.rstrip("."))
        return goals

    def _build_primary_story(self, actors: List[str], goals: List[str], feature_name: str, description: str) -> str:
        """Build a primary user story."""
        actor_text = ", ".join(actors)
        if goals:
            main_goal = goals[0]
        else:
            main_goal = description.strip().rstrip(".")
        return f"As {actor_text}, I want to {main_goal} so that {feature_name} delivers value."

    def _identify_clarifications(self, description: str, actions: Iterable[str]) -> List[str]:
        """Identify areas that need clarification."""
        ambiguous_hints = {
            "fast": "Specify measurable performance expectations (e.g., latency, throughput).",
            "quick": "Clarify concrete speed targets or user-facing latency requirements.",
            "responsive": "Define acceptable response thresholds across target devices.",
            "secure": "Describe required security controls (authN/Z, encryption, compliance).",
            "intuitive": "Document usability heuristics or UX patterns that define 'intuitive'.",
            "simple": "Clarify what simplicity means (workflow steps, learning time, UI density).",
            "scalable": "Provide scale targets (users, data volume, concurrency) and constraints.",
            "efficient": "Describe efficiency metrics (resource usage, time savings).",
            "robust": "Identify expected failure conditions and resilience requirements.",
            "reliable": "Specify uptime/SLA expectations and error budgets.",
            "modern": "Clarify UI/UX expectations or technology constraints for 'modern'.",
            "accessible": "List accessibility standards (e.g., WCAG level) that must be satisfied.",
            "compliant": "Identify the compliance regimes that apply (e.g., GDPR, SOC2).",
        }
        
        clarifications: List[str] = []
        lower = description.lower()
        for term, hint in ambiguous_hints.items():
            if term in lower:
                clarifications.append(f"{term.upper()} requirement: {hint}")
        
        # Additional checks
        if "authentication" in lower or "login" in lower:
            clarifications.append("Authentication method: Clarify auth flow (SSO, email/password, OAuth?).")
        if "drag" in lower and "drop" in lower:
            clarifications.append("Drag-and-drop: Specify desktop vs mobile behaviour and accessibility requirements.")
        if "notifications" in lower:
            clarifications.append("Notifications: Identify channels (email, push, in-app) and delivery guarantees.")
        if "report" in lower:
            clarifications.append("Reporting: Define format, cadence, and aggregation level for reports.")
        if not any("performance" in a.lower() for a in actions) and "fast" in lower:
            clarifications.append("Performance metrics: Define quantitative performance targets.")
        
        return clarifications

    def _build_edge_cases(self, actions: Iterable[str]) -> List[str]:
        """Build edge case questions."""
        edge_cases: List[str] = []
        for action in actions:
            base = action.rstrip(".")
            edge_cases.append(f"What happens when {base.lower()} fails due to invalid input or system errors?")
            edge_cases.append(f"How should {base.lower()} behave when the user lacks permissions?")
        return sorted(set(edge_cases))

    def _build_summary(self, feature_name: str, actors: Iterable[str], goals: Iterable[str]) -> str:
        """Build a feature summary."""
        actor_text = ", ".join(actors)
        goals_text = "; ".join(goals)
        return f"{feature_name} empowers {actor_text} by enabling: {goals_text}."

    def _render_spec(self, analysis: FeatureAnalysis) -> str:
        """Render a specification document."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        clarifications_block = "\n".join(
            f"- [NEEDS CLARIFICATION] {item}" for item in analysis.clarifications
        ) or "- None identified"
        acceptance = []
        for idx, action in enumerate(analysis.actions, start=1):
            acceptance.append(
                f"{idx}. **Given** the feature is available, **When** users {action.lower()}, **Then** the system fulfils the requirement."
            )
        acceptance_block = "\n".join(acceptance) or "1. **Given** the feature is available, **When** users engage with it, **Then** the system delivers the described value."
        edge_cases_block = "\n".join(f"- {item}" for item in analysis.edge_cases) or "- Define edge-case behaviour for invalid input, concurrency, and failure states."
        functional_requirements = []
        for idx, action in enumerate(analysis.actions, start=1):
            functional_requirements.append(f"- **FR-{idx:03d}**: System MUST {action}.")
        if analysis.clarifications:
            for idx, item in enumerate(analysis.clarifications, start=len(functional_requirements) + 1):
                functional_requirements.append(f"- **FR-{idx:03d}**: [NEEDS CLARIFICATION] {item}")
        functional_block = "\n".join(functional_requirements) or "- **FR-001**: System MUST deliver the described user value."
        entities = []
        for keyword in analysis.keywords[:5]:
            entities.append(f"- **{keyword.title()}**: Define structure and relationships relevant to the feature.")
        entity_block = "\n".join(entities) or "- Identify domain entities once design explorations begin."

        spec_template = f"""# Feature Specification: {analysis.feature_name}

**Feature Branch**: `{analysis.feature_id}`  
**Created**: {today}  
**Status**: Draft  
**Input**: User description: "{analysis.description}"\

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Summary
{analysis.summary}

## User Scenarios & Testing *(mandatory)*

### Primary User Story
{analysis.primary_story}

### Acceptance Scenarios
{acceptance_block}

### Edge Cases
{edge_cases_block}

## Requirements *(mandatory)*

### Functional Requirements
{functional_block}

### Key Entities *(include if feature involves data)*
{entity_block}

---

## Review & Acceptance Checklist
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---

## Ambiguities & Clarifications
{clarifications_block}

## Execution Status
- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed
"""
        return textwrap.dedent(spec_template).strip() + "\n"

    def _render_plan(
        self,
        analysis: FeatureAnalysis,
        spec_path: Path,
        constitution_excerpt: str,
        context_map: Dict[str, str],
    ) -> str:
        """Render an implementation plan."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        context_lines = [
            f"**Language/Version**: {context_map.get('language', 'NEEDS CLARIFICATION')}",
            f"**Primary Dependencies**: {context_map.get('primary_dependencies', 'NEEDS CLARIFICATION')}",
            f"**Storage**: {context_map.get('storage', 'NEEDS CLARIFICATION')}",
            f"**Testing**: {context_map.get('testing', 'NEEDS CLARIFICATION')}",
            f"**Target Platform**: {context_map.get('target_platform', 'NEEDS CLARIFICATION')}",
            f"**Performance Goals**: {context_map.get('performance_goals', 'Define explicit targets')}",
            f"**Constraints**: {context_map.get('constraints', 'Capture regulatory, compliance, or legacy constraints')}",
            f"**Scale/Scope**: {context_map.get('scale_scope', 'Document expected load, data volume, or usage cadence')}",
        ]
        technical_context_block = "\n".join(context_lines)

        research_items = "\n".join(f"- {item}" for item in analysis.clarifications) or "- No open questions. Confirm scope before proceeding."

        phase_one = []
        for idx, goal in enumerate(analysis.goals, start=1):
            phase_one.append(f"- Map requirement {idx} to API/data structures. Document contracts covering `{goal}`.")
        phase_one_block = "\n".join(phase_one) or "- Identify core domain entities and draft API surface."

        tasks_preview = []
        for idx, goal in enumerate(analysis.goals, start=1):
            tasks_preview.append(f"- Generate implementation and test tasks for requirement {idx}.")
        tasks_preview_block = "\n".join(tasks_preview) or "- Produce detailed tasks aligned with TDD workflow."

        relative_spec = spec_path.relative_to(self.root) if spec_path.is_relative_to(self.root) else spec_path

        plan_template = f"""# Implementation Plan: {analysis.feature_name}

**Branch**: `{analysis.feature_id}` | **Date**: {today} | **Spec**: {relative_spec}

## Summary
{analysis.summary}

## Technical Context
{technical_context_block}

## Constitution Check
{constitution_excerpt}

## Project Structure Recommendation
- `docs/` — specifications (`spec.md`, `plan.md`, `tasks.md`, research outputs)
- `src/` — primary implementation (models, services, presentation)
- `tests/contract/` — contract tests derived from API contracts
- `tests/integration/` — end-to-end scenarios for {analysis.feature_name}
- `tests/unit/` — focused unit coverage per module

## Research Agenda (Phase 0)
{research_items}

## Design Deliverables (Phase 1)
{phase_one_block}

Deliverables:
- `research.md` — Answers to Phase 0 questions
- `data-model.md` — Entities, relationships, validation rules
- `contracts/` — API schemas or protocol definitions
- `quickstart.md` — Step-by-step validation checklist

## Task Generation Strategy (Phase 2)
{tasks_preview_block}

- Order tasks via TDD: tests before implementation
- Mark independent tasks with `[P]` for parallel execution
- Reference exact file paths for every task

## Risks & Mitigations
- Capture dependencies on third-party services or teams
- Flag performance or compliance constraints early
- Document fallback strategies for critical unknowns

## Progress Tracking
- [ ] Phase 0 research complete
- [ ] Phase 1 design artifacts ready
- [ ] Phase 2 task list generated
- [ ] Phase 3 implementation underway
- [ ] Phase 4 validation complete

## Next Steps
1. Resolve clarifications and update constitution alignment
2. Complete Phase 0/1 deliverables
3. Invoke task generator once design assets are ready
4. Execute implementation guided by tasks and constitution
"""
        return textwrap.dedent(plan_template).strip() + "\n"

    def _render_tasks(self, analysis: FeatureAnalysis, plan_content: str) -> str:
        """Render a task list."""
        header = f"# Tasks: {analysis.feature_name}\n\n**Input**: Design artifacts for `{analysis.feature_id}`\n**Prerequisites**: plan.md, research.md, data-model.md, contracts/\n"

        task_items = []
        counter = 1

        def add_task(description: str, parallel: bool = False) -> None:
            nonlocal counter
            marker = "[P] " if parallel else ""
            task_items.append(f"- [ ] T{counter:03d} {marker}{description}")
            counter += 1

        add_task("Establish project scaffolding per plan structure (src/, tests/, docs/).")
        add_task("Document environment bootstrap instructions in quickstart.md.")
        add_task("Configure linting, formatting, and CI guards.", parallel=True)

        for idx, goal in enumerate(analysis.goals, start=1):
            add_task(
                f"Author contract test for requirement {idx} covering `{goal}` in tests/contract/.",
                parallel=True,
            )
            add_task(
                f"Create integration test for requirement {idx} in tests/integration/.",
                parallel=True,
            )
            add_task(
                f"Implement functionality for requirement {idx} in src/ with corresponding unit tests.")

        add_task("Wire integration paths end-to-end and ensure contract tests fail prior to implementation.")
        add_task("Harden error handling, logging, and observability hooks.")
        add_task("Document manual validation steps in quickstart.md.")
        add_task("Perform regression run and collect metrics for performance goals.")

        dependencies = textwrap.dedent(
            """
            - Contract and integration tests precede implementation tasks.
            - Implementation tasks unblock polish and hardening activities.
            - Documentation updates require completed functionality.
            """
        ).strip()

        parallel_batches = textwrap.dedent(
            """
            - Initial setup tasks (T001-T003) can run in parallel across different files.
            - Contract tests for each requirement (paired T00x) are parallelizable.
            - Implementation tasks should follow once associated tests exist and fail.
            """
        ).strip()

        qa = textwrap.dedent(
            """
            - Verify all contract tests fail before implementation, then pass afterward.
            - Ensure [NEEDS CLARIFICATION] items have explicit resolutions.
            - Update tasks.md with completion notes or links to commits/issues.
            """
        ).strip()

        content = (
            header
            + "\n## Task List\n"
            + "\n".join(task_items)
            + "\n\n## Dependencies\n"
            + dependencies
            + "\n\n## Parallel Execution Examples\n"
            + parallel_batches
            + "\n\n## Quality Checks\n"
            + qa
            + "\n"
        )
        return content

    def _constitution_excerpt(self) -> str:
        """Get an excerpt from the constitution."""
        constitution = self.load_constitution()
        if not constitution:
            return "No constitution recorded. Use set_constitution to define project principles."
        lines = [line.strip() for line in constitution.splitlines() if line.strip()]
        excerpt = "\n".join(f"- {line}" for line in lines[:8])
        return excerpt or "Constitution exists but is empty."

    def _parse_context(self, tech_context: Optional[str]) -> Dict[str, str]:
        """Parse technical context into a dictionary."""
        if not tech_context:
            return {}
        context: Dict[str, str] = {}
        aliases = {
            "lang": "language",
            "language": "language",
            "framework": "primary_dependencies",
            "libraries": "primary_dependencies",
            "deps": "primary_dependencies",
            "dependencies": "primary_dependencies",
            "storage": "storage",
            "database": "storage",
            "db": "storage",
            "test": "testing",
            "testing": "testing",
            "platform": "target_platform",
            "target": "target_platform",
            "performance": "performance_goals",
            "perf": "performance_goals",
            "constraint": "constraints",
            "constraints": "constraints",
            "scale": "scale_scope",
            "scope": "scale_scope",
        }
        for line in tech_context.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if ":" in cleaned:
                key, value = cleaned.split(":", 1)
            elif "=" in cleaned:
                key, value = cleaned.split("=", 1)
            else:
                continue
            key = key.strip().lower()
            value = value.strip()
            canonical = aliases.get(key, key)
            context[canonical] = value
        return context


# Import required modules
from datetime import datetime
import uuid

# ------------------------------------------------------------------
# Project task management
# ------------------------------------------------------------------

def _generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"PROJ-{uuid.uuid4().hex[:8].upper()}"

def _project_tasks_path(feature_id: str) -> Path:
    """Get the project tasks file path for a feature."""
    from pathlib import Path
    return Path(".speck-it") / "project_tasks" / f"{feature_id}_tasks.json"

def _load_project_tasks(feature_id: str) -> List[ProjectTask]:
    """Load project tasks from file."""
    tasks_path = _project_tasks_path(feature_id)
    if not tasks_path.exists():
        return []
    
    try:
        data = json.loads(tasks_path.read_text(encoding="utf-8"))
        return [ProjectTask.from_dict(task_data) for task_data in data]
    except (json.JSONDecodeError, KeyError):
        return []

def _save_project_tasks(feature_id: str, tasks: List[ProjectTask]) -> None:
    """Save project tasks to file."""
    tasks_path = _project_tasks_path(feature_id)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(
        json.dumps([task.to_dict() for task in tasks], indent=2),
        encoding="utf-8"
    )