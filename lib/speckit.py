"""Spec Kit-inspired utilities for MCP server.

This module provides a pragmatic, template-driven implementation of the
Spec-Driven Development workflow used by GitHub's Spec Kit project. It
extracts structured insights from a free-form feature description and
renders markdown artifacts (specification, plan, tasks) that mirror the
shape and rigor of the upstream toolkit. The goal is to allow MCP tools to
bootstrap high-quality feature artifacts offline without invoking the
original CLI.
"""

from __future__ import annotations

import json
import re
import textwrap
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Dataclasses representing persisted artifacts and derived analysis
# ---------------------------------------------------------------------------


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


@dataclass(slots=True)
class FeatureArtifacts:
    """Filesystem pointers for generated markdown artifacts."""

    feature_id: str
    feature_dir: Path
    spec_path: Optional[Path] = None
    plan_path: Optional[Path] = None
    tasks_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "feature_id": self.feature_id,
            "feature_dir": str(self.feature_dir),
            "spec_path": str(self.spec_path) if self.spec_path else None,
            "plan_path": str(self.plan_path) if self.plan_path else None,
            "tasks_path": str(self.tasks_path) if self.tasks_path else None,
        }


# ---------------------------------------------------------------------------
# Task management primitives
# ---------------------------------------------------------------------------


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
        return {
            "task_id": self.task_id,
            "description": self.description,
            "completed": self.completed,
            "notes": list(self.notes),
        }


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


# ---------------------------------------------------------------------------
# Helper utilities for lightweight NLP-style processing
# ---------------------------------------------------------------------------

_FEATURE_ROOT_REGISTRY: Dict[str, Path] = {}


def register_feature_root(feature_id: str, root: Path | str) -> Path:
    """Record the canonical project root for a feature's artifacts."""

    resolved = Path(root).resolve()
    _FEATURE_ROOT_REGISTRY[feature_id.lower()] = resolved
    return resolved


def lookup_feature_root(feature_id: str) -> Optional[Path]:
    """Return the registered project root for the feature, if any."""

    return _FEATURE_ROOT_REGISTRY.get(feature_id.lower())


_AMBIGUOUS_TERM_HINTS: Dict[str, str] = {
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

_ACTOR_KEYWORDS: Dict[str, Iterable[str]] = {
    "administrators": ("admin", "administrator", "admins"),
    "operators": ("operator", "operators"),
    "customers": ("customer", "customers"),
    "team members": ("team", "teams", "teammate", "teammates"),
    "managers": ("manager", "managers"),
    "developers": ("developer", "developers", "engineer", "engineers"),
    "analysts": ("analyst", "analysts"),
    "moderators": ("moderator", "moderators"),
}

_WORD_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

_TASK_LINE_PATTERN = re.compile(r"^(?P<prefix>\s*)-\s*\[(?P<mark> |x|X)\]\s*(?P<rest>.+)$")
_TASK_ID_PATTERN = re.compile(r"\bT\d{3,}\b", re.IGNORECASE)
_NOTE_LINE_PATTERN = re.compile(r"^\s{2,}-\s*Note:\s*(?P<note>.+)$", re.IGNORECASE)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "feature"


def _split_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def _normalize_sentence(sentence: str) -> str:
    return sentence.strip().rstrip(".?!")


def _sentence_keywords(sentence: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", sentence.lower()) if len(w) > 3]


def _extract_actors(description: str) -> List[str]:
    detected: List[str] = []
    lower = description.lower()
    for canonical, variants in _ACTOR_KEYWORDS.items():
        if any(term in lower for term in variants):
            detected.append(canonical)
    if "user" in lower and "users" not in detected:
        detected.append("users")
    if not detected:
        detected.append("users")
    return detected


def _extract_actions(sentences: Iterable[str]) -> List[str]:
    actions: List[str] = []
    for sentence in sentences:
        normalized = _normalize_sentence(sentence)
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


def _extract_goals(actions: Iterable[str]) -> List[str]:
    goals: List[str] = []
    for action in actions:
        goal = action
        if goal.lower().startswith("enable users to "):
            goal = goal[len("Enable users to ") :]
        elif goal.lower().startswith("allow users to "):
            goal = goal[len("Allow users to ") :]
        goals.append(goal.rstrip("."))
    return goals


def _build_primary_story(actors: List[str], goals: List[str], feature_name: str, description: str) -> str:
    actor_text = ", ".join(actors)
    if goals:
        main_goal = goals[0]
    else:
        main_goal = description.strip().rstrip(".")
    return f"As {actor_text}, I want to {main_goal} so that {feature_name} delivers value."


def _identify_clarifications(description: str, actions: Iterable[str]) -> List[str]:
    clarifications: List[str] = []
    lower = description.lower()
    for term, hint in _AMBIGUOUS_TERM_HINTS.items():
        if term in lower:
            clarifications.append(f"{term.upper()} requirement: {hint}")
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


def _build_edge_cases(actions: Iterable[str]) -> List[str]:
    edge_cases: List[str] = []
    for action in actions:
        base = action.rstrip(".")
        edge_cases.append(f"What happens when {base.lower()} fails due to invalid input or system errors?")
        edge_cases.append(f"How should {base.lower()} behave when the user lacks permissions?")
    return sorted(set(edge_cases))


def _build_summary(feature_name: str, actors: Iterable[str], goals: Iterable[str]) -> str:
    actor_text = ", ".join(actors)
    goals_text = "; ".join(goals)
    return f"{feature_name} empowers {actor_text} by enabling: {goals_text}."


# ---------------------------------------------------------------------------
# Workspace orchestrator
# ---------------------------------------------------------------------------


class SpecKitWorkspace:
    """Manage Spec Kit-inspired artifacts within a repository."""

    STORAGE_DIR_ENV = "SPECKIT_STORAGE_DIR"
    STORAGE_DIR_CANDIDATES = (".speck-it")

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        preferred_name = os.getenv(self.STORAGE_DIR_ENV)
        candidates: List[str] = []
        if preferred_name:
            candidates.append(preferred_name)
        candidates.extend(self.STORAGE_DIR_CANDIDATES)

        base_dir = None
        for name in candidates:
            candidate = self.root / name
            if candidate.exists():
                base_dir = candidate
                break

        if base_dir is None:
            base_dir = self.root / (preferred_name or self.STORAGE_DIR_CANDIDATES[0])

        self.base_dir = base_dir
        self.memory_dir = self.base_dir / "memory"
        self.specs_dir = self.base_dir / "specs"
        self.data_dir = self.base_dir / "state"
        self.tasks_dir = self.base_dir / "project_tasks"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _remember_feature_root(self, feature_id: str) -> None:
        register_feature_root(feature_id, self.root)

    def spec_exists(self, feature_id: str) -> bool:
        """Check if a specification exists for the given feature."""
        feature_dir = self._feature_dir(feature_id)
        return (feature_dir / "spec.md").exists()

    def plan_exists(self, feature_id: str) -> bool:
        """Check if an implementation plan exists for the given feature."""
        feature_dir = self._feature_dir(feature_id)
        return (feature_dir / "plan.md").exists()

    # ------------------------------------------------------------------
    # Constitution management
    # ------------------------------------------------------------------

    @property
    def constitution_path(self) -> Path:
        return self.memory_dir / "constitution.md"

    def save_constitution(self, content: str, *, mode: str = "replace") -> Path:
        if mode not in {"replace", "append"}:
            raise ValueError("mode must be 'replace' or 'append'")
        path = self.constitution_path
        if mode == "replace" or not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + content.strip() + "\n")
        return path

    def load_constitution(self) -> Optional[str]:
        if not self.constitution_path.exists():
            return None
        return self.constitution_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def list_features(self) -> List[Dict[str, str]]:
        features: List[Dict[str, str]] = []
        for path in sorted(self.specs_dir.glob("*/")):
            feature_id = path.name.rstrip("/")
            self._remember_feature_root(feature_id)
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
        highest = 0
        for directory in self.specs_dir.glob("*/"):
            name = directory.name.rstrip("/")
            match = re.match(r"(\d{3})-", name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _feature_identifier(self, feature_name: str, feature_id: Optional[str]) -> str:
        if feature_id:
            slug = _slugify(feature_id)
            if not re.match(r"^\d{3}-", slug):
                slug = f"{self._next_feature_number():03d}-{slug}"
            return slug
        slug = _slugify(feature_name)
        number = self._next_feature_number()
        candidate = f"{number:03d}-{slug}"
        while (self.specs_dir / candidate).exists():
            number += 1
            candidate = f"{number:03d}-{slug}"
        return candidate

    def _feature_dir(self, feature_id: str) -> Path:
        self._remember_feature_root(feature_id)
        path = self.specs_dir / feature_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _analysis_path(self, feature_dir: Path) -> Path:
        return feature_dir / "analysis.json"

    def _save_analysis(self, feature_dir: Path, analysis: FeatureAnalysis) -> None:
        path = self._analysis_path(feature_dir)
        path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")

    def _load_analysis(self, feature_id: str) -> Optional[FeatureAnalysis]:
        feature_dir = self.specs_dir / feature_id
        path = self._analysis_path(feature_dir)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return FeatureAnalysis.from_dict(data)

    # ------------------------------------------------------------------
    # Artifact generation
    # ------------------------------------------------------------------

    def generate_spec(
        self,
        feature_name: str,
        description: str,
        *,
        feature_id: Optional[str] = None,
        save: bool = True,
    ) -> tuple[FeatureArtifacts, FeatureAnalysis, str]:
        feature_identifier = self._feature_identifier(feature_name, feature_id)
        self._remember_feature_root(feature_identifier)
        actors = _extract_actors(description)
        sentences = _split_sentences(description)
        actions = _extract_actions(sentences)
        goals = _extract_goals(actions)
        primary_story = _build_primary_story(actors, goals, feature_name, description)
        clarifications = _identify_clarifications(description, actions)
        edge_cases = _build_edge_cases(actions)
        summary = _build_summary(feature_name, actors, goals)
        keywords = sorted({kw for sentence in sentences for kw in _sentence_keywords(sentence)})

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

        spec_content = self._render_spec(analysis)

        feature_dir = self._feature_dir(feature_identifier) if save else self.specs_dir / feature_identifier
        spec_path = feature_dir / "spec.md" if save else None
        if save:
            feature_dir.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(spec_content, encoding="utf-8")
            self._save_analysis(feature_dir, analysis)

        artifacts = FeatureArtifacts(
            feature_id=feature_identifier,
            feature_dir=feature_dir,
            spec_path=spec_path,
        )
        return artifacts, analysis, spec_content

    def generate_plan(
        self,
        feature_id: str,
        *,
        tech_context: Optional[str] = None,
        save: bool = True,
    ) -> tuple[FeatureArtifacts, FeatureAnalysis, str]:
        self._remember_feature_root(feature_id)
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
        self._remember_feature_root(feature_id)
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

    def implementation_playbook(self, feature_id: str) -> Dict[str, List[str]]:
        self._remember_feature_root(feature_id)
        analysis = self._load_analysis(feature_id)
        if not analysis:
            raise FileNotFoundError(
                f"Feature '{feature_id}' does not have analysis data. Generate a spec first."
            )
        feature_dir = self._feature_dir(feature_id)
        tasks_path = feature_dir / "tasks.md"
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        playbook = {
            "feature_id": feature_id,
            "spec_path": str(spec_path) if spec_path.exists() else None,
            "plan_path": str(plan_path) if plan_path.exists() else None,
            "tasks_path": str(tasks_path) if tasks_path.exists() else None,
            "phases": [],
            "quality_gates": [],
        }
        playbook["phases"].append(
            textwrap.dedent(
                f"""
                Phase 1 — Research & Clarification: Resolve outstanding questions.
                Outstanding clarifications: {', '.join(analysis.clarifications) or 'None'}
                """
            ).strip()
        )
        playbook["phases"].append(
            textwrap.dedent(
                f"""
                Phase 2 — Design Deliverables: Produce research.md, data-model.md,
                contracts, and quickstart docs guided by the plan checklist.
                """
            ).strip()
        )
        playbook["phases"].append(
            "Phase 3 — Task Execution: Follow tasks.md sequentially, honoring dependencies and parallel markers."
        )
        playbook["phases"].append(
            "Phase 4 — Implementation & Validation: Implement code, make tests pass, and run QA per plan."
        )

        playbook["quality_gates"].extend(
            [
                "All [NEEDS CLARIFICATION] items resolved before coding.",
                "Contract and integration tests authored before implementation (TDD).",
                "Constitution principles verified after each phase (spec, plan, tasks, implementation).",
                "All tasks closed with corresponding commits or notes.",
            ]
        )
        return playbook

    # ------------------------------------------------------------------
    # Task progress management
    # ------------------------------------------------------------------

    def _tasks_path(self, feature_id: str) -> Path:
        feature_dir = self._feature_dir(feature_id)
        return feature_dir / "tasks.md"

    def list_tasks(self, feature_id: str) -> List[Dict[str, Any]]:
        tasks = self._read_tasks(feature_id)
        return [task.to_dict() for task in tasks]

    def update_task(
        self,
        feature_id: str,
        task_id: str,
        *,
        completed: Optional[bool] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        tasks_path = self._tasks_path(feature_id)
        if not tasks_path.exists():
            raise FileNotFoundError(
                f"No tasks.md found for feature '{feature_id}'. Generate tasks before updating."
            )

        tasks = self._read_tasks(feature_id)
        task_map = {task.task_id.upper(): task for task in tasks}
        lookup = task_map.get(task_id.upper())
        if not lookup:
            raise ValueError(f"Task '{task_id}' not found for feature '{feature_id}'.")

        lines = tasks_path.read_text(encoding="utf-8").splitlines()
        line = lines[lookup.line_index]
        match = _TASK_LINE_PATTERN.match(line)
        if not match:
            raise RuntimeError(f"Unable to parse stored task line for '{task_id}'.")

        mark = match.group("mark")
        rest = match.group("rest")
        previously_completed = lookup.completed

        if completed is not None:
            new_mark = "x" if completed else " "
            mark = new_mark
            lookup.completed = completed

        note_to_add = note
        if note_to_add is None and completed:
            timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            if previously_completed:
                note_to_add = f"Completion reconfirmed {timestamp}"
            else:
                note_to_add = f"Completed via API {timestamp}"

        if note_to_add:
            note_line = f"{lookup.indent}  - Note: {note_to_add}"
            insert_index = lookup.line_index + 1
            while insert_index < len(lines) and lines[insert_index].startswith(lookup.indent + "  - "):
                insert_index += 1
            lines.insert(insert_index, note_line)
            lookup.notes.append(note_to_add)

        lines[lookup.line_index] = f"{lookup.indent}- [{mark}] {rest}"
        tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        status = self.feature_status(feature_id)
        next_task = status["tasks"]["incomplete"][0] if status["tasks"]["incomplete"] else None

        return {
            "task": lookup.to_dict(),
            "tasks_path": str(tasks_path),
            "remaining": status["tasks"]["remaining"],
            "all_completed": status["tasks"]["all_completed"],
            "next_task": next_task,
        }

    def next_open_task(self, feature_id: str) -> Optional[Dict[str, Any]]:
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

    def finalize_feature(self, feature_id: str) -> Dict[str, Any]:
        status = self.feature_status(feature_id)

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
                "Cannot finalize feature '{feature_id}': pending tasks remain -> "
                + ", ".join(pending_ids)
            )

        status["finalized_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        status_path = self.data_dir / f"{feature_id}_status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        status["status_path"] = str(status_path)
        return status

    def _read_tasks(self, feature_id: str) -> List[TaskItem]:
        tasks_path = self._tasks_path(feature_id)
        if not tasks_path.exists():
            raise FileNotFoundError(
                f"No tasks.md found for feature '{feature_id}'. Generate tasks before listing."
            )
        lines = tasks_path.read_text(encoding="utf-8").splitlines()
        tasks: List[TaskItem] = []
        current_task: Optional[TaskItem] = None
        for idx, line in enumerate(lines):
            task_match = _TASK_LINE_PATTERN.match(line)
            if task_match:
                rest = task_match.group("rest").strip()
                task_id_match = _TASK_ID_PATTERN.search(rest)
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

            note_match = _NOTE_LINE_PATTERN.match(line)
            if note_match and current_task:
                current_task.notes.append(note_match.group("note").strip())

        return tasks

    # ------------------------------------------------------------------
    # Project-level task management
    # ------------------------------------------------------------------

    def _project_tasks_path(self) -> Path:
        """Get the path to the project tasks storage file."""
        return self.tasks_dir / "project_tasks.json"

    def _load_project_tasks(self) -> Dict[str, ProjectTask]:
        """Load all project tasks from storage."""
        path = self._project_tasks_path()
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tasks = {}
            for task_id, task_data in data.items():
                tasks[task_id] = ProjectTask.from_dict(task_data)
            return tasks
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_project_tasks(self, tasks: Dict[str, ProjectTask]) -> None:
        """Save all project tasks to storage."""
        path = self._project_tasks_path()
        data = {task_id: task.to_dict() for task_id, task in tasks.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
        """Create a new project-level task."""
        tasks = self._load_project_tasks()

        # Generate unique task ID
        task_counter = 1
        while True:
            task_id = f"PROJ-{task_counter:03d}"
            if task_id not in tasks:
                break
            task_counter += 1

        task = ProjectTask(
            task_id=task_id,
            feature_id=feature_id,
            description=description,
            task_type=task_type,
            priority=priority,
            dependencies=dependencies or [],
            prerequisites=prerequisites or [],
            estimated_hours=estimated_hours,
            tags=tags or [],
        )

        tasks[task_id] = task
        self._save_project_tasks(tasks)
        return task

    def get_project_tasks(
        self,
        feature_id: Optional[str] = None,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        priority_range: Optional[tuple[int, int]] = None,
    ) -> List[ProjectTask]:
        """Get project tasks with optional filtering."""
        tasks = self._load_project_tasks()

        filtered_tasks = []
        for task in tasks.values():
            if feature_id and task.feature_id != feature_id:
                continue
            if status and task.status != status:
                continue
            if task_type and task.task_type != task_type:
                continue
            if priority_range and not (priority_range[0] <= task.priority <= priority_range[1]):
                continue
            filtered_tasks.append(task)

        # Sort by priority (lower number = higher priority), then by creation date
        filtered_tasks.sort(key=lambda t: (t.priority, t.created_at))
        return filtered_tasks

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
        tasks = self._load_project_tasks()
        task = tasks.get(task_id)

        if not task:
            return None

        if status:
            task.status = status
            task.updated_at = datetime.utcnow().isoformat() + "Z"
            if status == "completed" and not task.completed_at:
                task.completed_at = datetime.utcnow().isoformat() + "Z"

        if priority is not None:
            task.priority = priority
            task.updated_at = datetime.utcnow().isoformat() + "Z"

        if actual_hours is not None:
            task.actual_hours = actual_hours
            task.updated_at = datetime.utcnow().isoformat() + "Z"

        if add_note:
            task.notes.append(add_note)
            task.updated_at = datetime.utcnow().isoformat() + "Z"

        if add_tag and add_tag not in task.tags:
            task.tags.append(add_tag)
            task.updated_at = datetime.utcnow().isoformat() + "Z"

        if remove_tag and remove_tag in task.tags:
            task.tags.remove(remove_tag)
            task.updated_at = datetime.utcnow().isoformat() + "Z"

        self._save_project_tasks(tasks)
        return task

    def validate_task_prerequisites(self, task_id: str) -> Dict[str, Any]:
        """Validate that all prerequisites for a task are met."""
        tasks = self._load_project_tasks()
        task = tasks.get(task_id)

        if not task:
            return {"valid": False, "error": f"Task '{task_id}' not found"}

        issues = []

        # Check feature root registration
        if "feature_root_registered" in task.prerequisites:
            if not lookup_feature_root(task.feature_id):
                issues.append(f"Feature root not registered for feature '{task.feature_id}'")

        # Check if constitution exists
        if "constitution_exists" in task.prerequisites:
            if not self.constitution_path.exists():
                issues.append("Project constitution not set")

        # Check if spec exists
        if "spec_exists" in task.prerequisites:
            if not self.spec_exists(task.feature_id):
                issues.append(f"Specification not found for feature '{task.feature_id}'")

        # Check if plan exists
        if "plan_exists" in task.prerequisites:
            if not self.plan_exists(task.feature_id):
                issues.append(f"Implementation plan not found for feature '{task.feature_id}'")

        # Check if tasks exist
        if "tasks_exist" in task.prerequisites:
            try:
                existing_tasks = self.list_tasks(task.feature_id)
                if not existing_tasks:
                    issues.append(f"No task list found for feature '{task.feature_id}'")
            except FileNotFoundError:
                issues.append(f"Task list not generated for feature '{task.feature_id}'")

        # Check dependency tasks
        for dep_id in task.dependencies:
            dep_task = tasks.get(dep_id)
            if not dep_task:
                issues.append(f"Dependency task '{dep_id}' not found")
            elif dep_task.status != "completed":
                issues.append(f"Dependency task '{dep_id}' is not completed (status: {dep_task.status})")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "can_proceed": len(issues) == 0
        }

    def get_next_executable_tasks(self) -> List[ProjectTask]:
        """Get tasks that are ready to be executed (prerequisites met, no blocking dependencies)."""
        tasks = self._load_project_tasks()
        executable = []

        for task in tasks.values():
            if task.status in ["completed", "cancelled"]:
                continue

            # Check if all dependencies are completed
            deps_met = True
            for dep_id in task.dependencies:
                dep_task = tasks.get(dep_id)
                if not dep_task or dep_task.status != "completed":
                    deps_met = False
                    break

            if not deps_met:
                continue

            # Validate prerequisites
            validation = self.validate_task_prerequisites(task.task_id)
            if validation["can_proceed"]:
                executable.append(task)

        # Sort by priority and creation date
        executable.sort(key=lambda t: (t.priority, t.created_at))
        return executable

    def get_project_status(self) -> ProjectStatus:
        """Get comprehensive project status across all features."""
        tasks = self._load_project_tasks()
        features = self.list_features()

        # Calculate feature statistics
        total_features = len(features)
        completed_features = sum(1 for f in features if self._is_feature_completed(f["feature_id"]))
        in_progress_features = sum(1 for f in features if self._is_feature_in_progress(f["feature_id"]))
        blocked_features = sum(1 for f in features if self._is_feature_blocked(f["feature_id"]))

        # Calculate task statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
        in_progress_tasks = sum(1 for t in tasks.values() if t.status == "in_progress")
        blocked_tasks = sum(1 for t in tasks.values() if t.status == "blocked")
        high_priority_tasks = sum(1 for t in tasks.values() if t.priority <= 3)

        # Calculate time estimates
        estimated_hours_remaining = sum(
            t.estimated_hours or 0 for t in tasks.values()
            if t.status not in ["completed", "cancelled"]
        )
        actual_hours_spent = sum(
            t.actual_hours or 0 for t in tasks.values()
            if t.status == "completed"
        )

        # Find blockers
        blockers = []
        for task in tasks.values():
            if task.status == "blocked":
                blockers.extend(task.blocked_by)

        # Get milestones (completed high-priority tasks)
        milestones = [
            t.description for t in tasks.values()
            if t.status == "completed" and t.priority <= 2
        ]

        return ProjectStatus(
            project_name=self.root.name,
            total_features=total_features,
            completed_features=completed_features,
            in_progress_features=in_progress_features,
            blocked_features=blocked_features,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            in_progress_tasks=in_progress_tasks,
            blocked_tasks=blocked_tasks,
            high_priority_tasks=high_priority_tasks,
            estimated_hours_remaining=estimated_hours_remaining,
            actual_hours_spent=actual_hours_spent,
            blockers=list(set(blockers)),
            milestones=milestones[:10],  # Limit to 10 most recent
        )

    def _is_feature_completed(self, feature_id: str) -> bool:
        """Check if a feature is completed."""
        try:
            status = self.feature_status(feature_id)
            return status["tasks"]["all_completed"]
        except:
            return False

    def _is_feature_in_progress(self, feature_id: str) -> bool:
        """Check if a feature is in progress."""
        try:
            status = self.feature_status(feature_id)
            total = status["tasks"]["total"]
            completed = status["tasks"]["completed"]
            return total > 0 and completed < total
        except:
            return False

    def _is_feature_blocked(self, feature_id: str) -> bool:
        """Check if a feature is blocked."""
        tasks = self.get_project_tasks(feature_id=feature_id)
        return any(t.status == "blocked" for t in tasks)

    def auto_update_task_status(self, feature_id: str, action: str) -> List[str]:
        """Automatically update task statuses based on tool executions."""
        updated_tasks = []
        tasks = self._load_project_tasks()

        for task in tasks.values():
            if task.feature_id != feature_id:
                continue

            # Update workflow tasks based on actions
            if action == "constitution_set" and "constitution_exists" in task.prerequisites:
                if task.status == "pending":
                    task.status = "completed"
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
                    task.completed_at = datetime.utcnow().isoformat() + "Z"
                    task.notes.append("Automatically completed: constitution was set")
                    updated_tasks.append(task.task_id)

            elif action == "feature_root_set" and "feature_root_registered" in task.prerequisites:
                if task.status == "pending":
                    task.status = "completed"
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
                    task.completed_at = datetime.utcnow().isoformat() + "Z"
                    task.notes.append("Automatically completed: feature root was registered")
                    updated_tasks.append(task.task_id)

            elif action == "spec_generated" and "spec_exists" in task.prerequisites:
                if task.status == "pending":
                    task.status = "completed"
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
                    task.completed_at = datetime.utcnow().isoformat() + "Z"
                    task.notes.append("Automatically completed: specification was generated")
                    updated_tasks.append(task.task_id)

            elif action == "plan_generated" and "plan_exists" in task.prerequisites:
                if task.status == "pending":
                    task.status = "completed"
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
                    task.completed_at = datetime.utcnow().isoformat() + "Z"
                    task.notes.append("Automatically completed: implementation plan was generated")
                    updated_tasks.append(task.task_id)

            elif action == "tasks_generated" and "tasks_exist" in task.prerequisites:
                if task.status == "pending":
                    task.status = "completed"
                    task.updated_at = datetime.utcnow().isoformat() + "Z"
                    task.completed_at = datetime.utcnow().isoformat() + "Z"
                    task.notes.append("Automatically completed: task list was generated")
                    updated_tasks.append(task.task_id)

        if updated_tasks:
            self._save_project_tasks(tasks)

        return updated_tasks

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_spec(self, analysis: FeatureAnalysis) -> str:
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
        constitution = self.load_constitution()
        if not constitution:
            return "No constitution recorded. Use set_constitution to define project principles."
        lines = [line.strip() for line in constitution.splitlines() if line.strip()]
        excerpt = "\n".join(f"- {line}" for line in lines[:8])
        return excerpt or "Constitution exists but is empty."

    def _parse_context(self, tech_context: Optional[str]) -> Dict[str, str]:
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


__all__ = ["SpecKitWorkspace", "FeatureArtifacts", "FeatureAnalysis", "ProjectTask", "ProjectStatus"]
