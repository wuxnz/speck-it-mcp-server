"""Data models for Speck-It workflow management.

This module contains the core data structures used throughout the Speck-It system,
representing features, tasks, analysis results, and project status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FeatureAnalysis:
    """Structured interpretation of a feature request."""

    feature_id: str
    feature_name: str
    description: str
    primary_story: str
    actors: List[str]
    actions: List[str]
    goals: List[str]
    clarifications: List[str]
    edge_cases: List[str]
    summary: str
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "description": self.description,
            "primary_story": self.primary_story,
            "actors": self.actors,
            "actions": self.actions,
            "goals": self.goals,
            "clarifications": self.clarifications,
            "edge_cases": self.edge_cases,
            "summary": self.summary,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureAnalysis":
        """Create from dictionary representation."""
        return cls(
            feature_id=data["feature_id"],
            feature_name=data["feature_name"],
            description=data["description"],
            primary_story=data["primary_story"],
            actors=data["actors"],
            actions=data["actions"],
            goals=data.get("goals", []),
            clarifications=data.get("clarifications", []),
            edge_cases=data.get("edge_cases", []),
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
        )

    def validate(self) -> List[str]:
        """Validate the analysis and return any issues."""
        issues = []
        
        if not self.feature_id:
            issues.append("Feature ID is required")
        if not self.feature_name:
            issues.append("Feature name is required")
        if not self.description:
            issues.append("Description is required")
        if not self.actors:
            issues.append("At least one actor is required")
        if not self.actions:
            issues.append("At least one action is required")
        if not self.goals:
            issues.append("At least one goal is required")
            
        return issues


@dataclass(slots=True)
class FeatureArtifacts:
    """Filesystem pointers for generated markdown artifacts."""

    feature_id: str
    feature_dir: Path
    spec_path: Optional[Path] = None
    plan_path: Optional[Path] = None
    tasks_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary representation."""
        return {
            "feature_id": self.feature_id,
            "feature_dir": str(self.feature_dir),
            "spec_path": str(self.spec_path) if self.spec_path else None,
            "plan_path": str(self.plan_path) if self.plan_path else None,
            "tasks_path": str(self.tasks_path) if self.tasks_path else None,
        }

    def exists(self) -> bool:
        """Check if all non-None paths exist."""
        paths = [p for p in [self.spec_path, self.plan_path, self.tasks_path] if p is not None]
        return all(p.exists() for p in paths)

    def get_missing_paths(self) -> List[str]:
        """Get list of missing artifact paths."""
        missing = []
        if self.spec_path and not self.spec_path.exists():
            missing.append(f"spec: {self.spec_path}")
        if self.plan_path and not self.plan_path.exists():
            missing.append(f"plan: {self.plan_path}")
        if self.tasks_path and not self.tasks_path.exists():
            missing.append(f"tasks: {self.tasks_path}")
        return missing


@dataclass(slots=True)
class TaskItem:
    """Representation of a single tasks.md checklist entry."""

    task_id: str
    description: str
    completed: bool
    notes: List[str] = field(default_factory=list)
    line_index: int = -1
    indent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "completed": self.completed,
            "notes": list(self.notes),
        }

    def add_note(self, note: str) -> None:
        """Add a note to this task."""
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.notes.append(f"{timestamp}: {note}")

    def mark_completed(self, note: Optional[str] = None) -> None:
        """Mark this task as completed."""
        self.completed = True
        if note:
            self.add_note(note)
        else:
            self.add_note("Task marked as completed")


@dataclass(slots=True)
class ProjectTask:
    """Enhanced task representation for project-level task management."""

    task_id: str
    feature_id: str
    description: str
    task_type: str  # 'workflow', 'spec', 'plan', 'implementation', 'validation', 'custom'
    priority: int = 5  # 1-10 scale, lower number = higher priority
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'blocked', 'cancelled'
    dependencies: List[str] = field(default_factory=list)  # List of task_ids this depends on
    prerequisites: List[str] = field(default_factory=list)  # List of prerequisite conditions
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    assigned_to: Optional[str] = None  # Could be AI model name or user
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)  # Task IDs that block this one
    blocks: List[str] = field(default_factory=list)  # Task IDs blocked by this one

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "feature_id": self.feature_id,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "prerequisites": list(self.prerequisites),
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "notes": list(self.notes),
            "tags": list(self.tags),
            "blocked_by": list(self.blocked_by),
            "blocks": list(self.blocks),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectTask":
        """Create from dictionary representation."""
        return cls(
            task_id=data["task_id"],
            feature_id=data["feature_id"],
            description=data["description"],
            task_type=data["task_type"],
            priority=data.get("priority", 5),
            status=data.get("status", "pending"),
            dependencies=data.get("dependencies", []),
            prerequisites=data.get("prerequisites", []),
            estimated_hours=data.get("estimated_hours"),
            actual_hours=data.get("actual_hours"),
            assigned_to=data.get("assigned_to"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", []),
            tags=data.get("tags", []),
            blocked_by=data.get("blocked_by", []),
            blocks=data.get("blocks", []),
        )

    def update_status(self, new_status: str, note: Optional[str] = None) -> None:
        """Update task status with optional note."""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow().isoformat() + "Z"
        
        if new_status == "completed" and old_status != "completed":
            self.completed_at = datetime.utcnow().isoformat() + "Z"
            if note:
                self.add_note(f"Status changed to {new_status}: {note}")
            else:
                self.add_note(f"Status changed to {new_status}")
        elif note:
            self.add_note(f"Status changed from {old_status} to {new_status}: {note}")
        else:
            self.add_note(f"Status changed from {old_status} to {new_status}")

    def add_note(self, note: str) -> None:
        """Add a note with timestamp."""
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.notes.append(f"{timestamp}: {note}")
        self.updated_at = timestamp

    def is_executable(self) -> bool:
        """Check if task can be executed (not completed/cancelled/blocked)."""
        return self.status not in ["completed", "cancelled", "blocked"]

    def is_high_priority(self) -> bool:
        """Check if task is high priority (1-3)."""
        return self.priority <= 3

    def validate(self) -> List[str]:
        """Validate task data and return any issues."""
        issues = []
        
        if not self.task_id:
            issues.append("Task ID is required")
        if not self.feature_id:
            issues.append("Feature ID is required")
        if not self.description:
            issues.append("Description is required")
        if self.task_type not in ["workflow", "spec", "plan", "implementation", "validation", "custom"]:
            issues.append(f"Invalid task type: {self.task_type}")
        if not 1 <= self.priority <= 10:
            issues.append(f"Priority must be 1-10, got: {self.priority}")
        if self.status not in ["pending", "in_progress", "completed", "blocked", "cancelled"]:
            issues.append(f"Invalid status: {self.status}")
        if self.estimated_hours is not None and self.estimated_hours <= 0:
            issues.append("Estimated hours must be positive")
        if self.actual_hours is not None and self.actual_hours <= 0:
            issues.append("Actual hours must be positive")
            
        return issues


@dataclass(slots=True)
class ProjectStatus:
    """Project-level status tracking across all features."""

    project_name: str
    total_features: int = 0
    completed_features: int = 0
    in_progress_features: int = 0
    blocked_features: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    in_progress_tasks: int = 0
    blocked_tasks: int = 0
    high_priority_tasks: int = 0
    overdue_tasks: int = 0
    estimated_hours_remaining: float = 0.0
    actual_hours_spent: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    blockers: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_name": self.project_name,
            "total_features": self.total_features,
            "completed_features": self.completed_features,
            "in_progress_features": self.in_progress_features,
            "blocked_features": self.blocked_features,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "in_progress_tasks": self.in_progress_tasks,
            "blocked_tasks": self.blocked_tasks,
            "high_priority_tasks": self.high_priority_tasks,
            "overdue_tasks": self.overdue_tasks,
            "estimated_hours_remaining": self.estimated_hours_remaining,
            "actual_hours_spent": self.actual_hours_spent,
            "last_updated": self.last_updated,
            "blockers": list(self.blockers),
            "milestones": list(self.milestones),
        }

    def get_completion_rate(self) -> float:
        """Get task completion rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    def get_feature_completion_rate(self) -> float:
        """Get feature completion rate as percentage."""
        if self.total_features == 0:
            return 0.0
        return (self.completed_features / self.total_features) * 100

    def is_healthy(self) -> bool:
        """Check if project is in a healthy state."""
        # Healthy if less than 20% of tasks are blocked and completion rate > 10%
        blocked_rate = (self.blocked_tasks / self.total_tasks) if self.total_tasks > 0 else 0
        completion_rate = self.get_completion_rate()
        return blocked_rate < 0.2 and completion_rate > 10.0

    def add_milestone(self, milestone: str) -> None:
        """Add a new milestone."""
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.milestones.append(f"{timestamp}: {milestone}")
        self.last_updated = timestamp

    def add_blocker(self, blocker: str) -> None:
        """Add a new blocker."""
        if blocker not in self.blockers:
            self.blockers.append(blocker)
            self.last_updated = datetime.utcnow().isoformat() + "Z"

    def remove_blocker(self, blocker: str) -> None:
        """Remove a blocker."""
        if blocker in self.blockers:
            self.blockers.remove(blocker)
            self.last_updated = datetime.utcnow().isoformat() + "Z"


@dataclass(slots=True)
class WorkflowStep:
    """Represents a single step in the Speck-It workflow."""

    step_number: int
    name: str
    tool_name: str
    description: str
    purpose: str
    prerequisites: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    is_completed: bool = False
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_number": self.step_number,
            "name": self.name,
            "tool_name": self.tool_name,
            "description": self.description,
            "purpose": self.purpose,
            "prerequisites": list(self.prerequisites),
            "expected_output": self.expected_output,
            "is_completed": self.is_completed,
            "completed_at": self.completed_at,
        }

    def mark_completed(self) -> None:
        """Mark this step as completed."""
        self.is_completed = True
        self.completed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if this step can be executed based on prerequisites."""
        return all(prereq in completed_steps for prereq in self.prerequisites)


# Workflow step definitions
WORKFLOW_STEPS = [
    WorkflowStep(
        step_number=1,
        name="Constitution Setup",
        tool_name="set_constitution",
        description="Establish project constitution and foundational principles",
        purpose="Define project guidelines and standards for all development",
        expected_output="Constitution saved to .speck-it/memory/constitution.md"
    ),
    WorkflowStep(
        step_number=2,
        name="Feature Root Registration",
        tool_name="set_feature_root",
        description="Register the project root directory for feature artifacts",
        purpose="Establish where specifications and plans will be stored",
        prerequisites=["Constitution Setup"],
        expected_output="Feature root registered and .speck-it/ directory created"
    ),
    WorkflowStep(
        step_number=3,
        name="Specification Generation",
        tool_name="generate_spec",
        description="Create detailed feature specification from description",
        purpose="Generate comprehensive requirements and analysis",
        prerequisites=["Feature Root Registration"],
        expected_output="Specification saved to .speck-it/specs/{feature_id}/spec.md"
    ),
    WorkflowStep(
        step_number=4,
        name="Implementation Planning",
        tool_name="generate_plan",
        description="Create implementation plan from specification",
        purpose="Develop technical approach and architecture",
        prerequisites=["Specification Generation"],
        expected_output="Plan saved to .speck-it/specs/{feature_id}/plan.md"
    ),
    WorkflowStep(
        step_number=5,
        name="Task Generation",
        tool_name="generate_tasks",
        description="Break down plan into actionable TDD-oriented tasks",
        purpose="Create executable checklist for implementation",
        prerequisites=["Implementation Planning"],
        expected_output="Tasks saved to .speck-it/specs/{feature_id}/tasks.md"
    ),
    WorkflowStep(
        step_number=6,
        name="Task Execution",
        tool_name="list_tasks, update_task, complete_task",
        description="Execute tasks sequentially with progress tracking",
        purpose="Implement feature following TDD methodology",
        prerequisites=["Task Generation"],
        expected_output="All tasks completed with implementation"
    ),
    WorkflowStep(
        step_number=7,
        name="Feature Finalization",
        tool_name="finalize_feature",
        description="Mark feature complete after all tasks are done",
        purpose="Validate completion and archive feature artifacts",
        prerequisites=["Task Execution"],
        expected_output="Feature marked as completed and archived"
    ),
]