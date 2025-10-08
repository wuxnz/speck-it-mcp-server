"""Unit tests for Speck-It models.

This module tests the core data structures and their validation,
serialization, and business logic methods.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.models import (
    FeatureAnalysis,
    FeatureArtifacts,
    TaskItem,
    ProjectTask,
    ProjectStatus,
    WorkflowStep,
    WORKFLOW_STEPS,
)


class TestFeatureAnalysis:
    """Test cases for FeatureAnalysis model."""

    def test_feature_analysis_creation(self):
        """Test creating a valid FeatureAnalysis."""
        analysis = FeatureAnalysis(
            feature_id="test-001",
            feature_name="Test Feature",
            description="A test feature for validation",
            primary_story="As a user, I want to test so that validation works",
            actors=["users"],
            actions=["Enable users to test"],
            goals=["test validation"],
            clarifications=[],
            edge_cases=[],
            summary="Test feature enables users to test validation"
        )

        assert analysis.feature_id == "test-001"
        assert analysis.feature_name == "Test Feature"
        assert analysis.actors == ["users"]
        assert analysis.actions == ["Enable users to test"]
        assert analysis.goals == ["test validation"]

    def test_feature_analysis_to_dict(self):
        """Test converting FeatureAnalysis to dictionary."""
        analysis = FeatureAnalysis(
            feature_id="test-001",
            feature_name="Test Feature",
            description="A test feature",
            primary_story="As a user, I want to test",
            actors=["users"],
            actions=["Enable users to test"],
            goals=["test"],
            clarifications=[],
            edge_cases=[],
            summary="Test feature"
        )

        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["feature_id"] == "test-001"
        assert result["feature_name"] == "Test Feature"
        assert result["actors"] == ["users"]
        assert result["actions"] == ["Enable users to test"]
        assert result["goals"] == ["test"]

    def test_feature_analysis_from_dict(self):
        """Test creating FeatureAnalysis from dictionary."""
        data = {
            "feature_id": "test-001",
            "feature_name": "Test Feature",
            "description": "A test feature",
            "primary_story": "As a user, I want to test",
            "actors": ["users"],
            "actions": ["Enable users to test"],
            "goals": ["test"],
            "clarifications": [],
            "edge_cases": [],
            "summary": "Test feature",
            "keywords": ["test", "feature"]
        }

        analysis = FeatureAnalysis.from_dict(data)
        
        assert analysis.feature_id == "test-001"
        assert analysis.feature_name == "Test Feature"
        assert analysis.keywords == ["test", "feature"]

    def test_feature_analysis_validation_success(self):
        """Test successful validation of FeatureAnalysis."""
        analysis = FeatureAnalysis(
            feature_id="test-001",
            feature_name="Test Feature",
            description="A test feature",
            primary_story="As a user, I want to test",
            actors=["users"],
            actions=["Enable users to test"],
            goals=["test"],
            clarifications=[],
            edge_cases=[],
            summary="Test feature"
        )

        issues = analysis.validate()
        assert issues == []

    def test_feature_analysis_validation_failures(self):
        """Test validation failures for FeatureAnalysis."""
        analysis = FeatureAnalysis(
            feature_id="",  # Invalid
            feature_name="",  # Invalid
            description="",  # Invalid
            primary_story="As a user, I want to test",
            actors=[],  # Invalid
            actions=[],  # Invalid
            goals=[],  # Invalid
            clarifications=[],
            edge_cases=[],
            summary="Test feature"
        )

        issues = analysis.validate()
        assert len(issues) == 5
        assert "Feature ID is required" in issues
        assert "Feature name is required" in issues
        assert "Description is required" in issues
        assert "At least one actor is required" in issues
        assert "At least one action is required" in issues


class TestFeatureArtifacts:
    """Test cases for FeatureArtifacts model."""

    def test_feature_artifacts_creation(self):
        """Test creating FeatureArtifacts."""
        feature_dir = Path("/tmp/test-feature")
        artifacts = FeatureArtifacts(
            feature_id="test-001",
            feature_dir=feature_dir,
            spec_path=feature_dir / "spec.md",
            plan_path=feature_dir / "plan.md",
            tasks_path=feature_dir / "tasks.md"
        )

        assert artifacts.feature_id == "test-001"
        assert artifacts.feature_dir == feature_dir
        assert artifacts.spec_path.name == "spec.md"
        assert artifacts.plan_path.name == "plan.md"
        assert artifacts.tasks_path.name == "tasks.md"

    def test_feature_artifacts_to_dict(self):
        """Test converting FeatureArtifacts to dictionary."""
        feature_dir = Path("/tmp/test-feature")
        artifacts = FeatureArtifacts(
            feature_id="test-001",
            feature_dir=feature_dir,
            spec_path=feature_dir / "spec.md",
            plan_path=None,
            tasks_path=feature_dir / "tasks.md"
        )

        result = artifacts.to_dict()
        
        assert isinstance(result, dict)
        assert result["feature_id"] == "test-001"
        assert result["spec_path"] == str(feature_dir / "spec.md")
        assert result["plan_path"] is None
        assert result["tasks_path"] == str(feature_dir / "tasks.md")

    def test_feature_artifacts_exists_all_exist(self, tmp_path):
        """Test exists() when all files exist."""
        feature_dir = tmp_path / "test-feature"
        feature_dir.mkdir()
        
        spec_path = feature_dir / "spec.md"
        plan_path = feature_dir / "plan.md"
        tasks_path = feature_dir / "tasks.md"
        
        # Create all files
        for path in [spec_path, plan_path, tasks_path]:
            path.write_text("test content")

        artifacts = FeatureArtifacts(
            feature_id="test-001",
            feature_dir=feature_dir,
            spec_path=spec_path,
            plan_path=plan_path,
            tasks_path=tasks_path
        )

        assert artifacts.exists() is True

    def test_feature_artifacts_exists_missing_files(self, tmp_path):
        """Test exists() when some files are missing."""
        feature_dir = tmp_path / "test-feature"
        feature_dir.mkdir()
        
        spec_path = feature_dir / "spec.md"
        plan_path = feature_dir / "plan.md"
        tasks_path = feature_dir / "tasks.md"
        
        # Create only spec file
        spec_path.write_text("test content")

        artifacts = FeatureArtifacts(
            feature_id="test-001",
            feature_dir=feature_dir,
            spec_path=spec_path,
            plan_path=plan_path,  # Missing
            tasks_path=tasks_path  # Missing
        )

        assert artifacts.exists() is False

    def test_feature_artifacts_get_missing_paths(self, tmp_path):
        """Test get_missing_paths() method."""
        feature_dir = tmp_path / "test-feature"
        feature_dir.mkdir()
        
        spec_path = feature_dir / "spec.md"
        plan_path = feature_dir / "plan.md"
        tasks_path = feature_dir / "tasks.md"
        
        # Create only spec file
        spec_path.write_text("test content")

        artifacts = FeatureArtifacts(
            feature_id="test-001",
            feature_dir=feature_dir,
            spec_path=spec_path,
            plan_path=plan_path,  # Missing
            tasks_path=tasks_path  # Missing
        )

        missing = artifacts.get_missing_paths()
        assert len(missing) == 2
        assert any("plan" in path for path in missing)
        assert any("tasks" in path for path in missing)


class TestTaskItem:
    """Test cases for TaskItem model."""

    def test_task_item_creation(self):
        """Test creating a TaskItem."""
        task = TaskItem(
            task_id="T001",
            description="Test task description",
            completed=False,
            notes=["Initial note"],
            line_index=5,
            indent="  "
        )

        assert task.task_id == "T001"
        assert task.description == "Test task description"
        assert task.completed is False
        assert task.notes == ["Initial note"]
        assert task.line_index == 5
        assert task.indent == "  "

    def test_task_item_to_dict(self):
        """Test converting TaskItem to dictionary."""
        task = TaskItem(
            task_id="T001",
            description="Test task",
            completed=True,
            notes=["Note 1", "Note 2"]
        )

        result = task.to_dict()
        
        assert isinstance(result, dict)
        assert result["task_id"] == "T001"
        assert result["description"] == "Test task"
        assert result["completed"] is True
        assert result["notes"] == ["Note 1", "Note 2"]

    def test_task_item_add_note(self):
        """Test adding a note to TaskItem."""
        task = TaskItem(
            task_id="T001",
            description="Test task",
            completed=False
        )

        initial_note_count = len(task.notes)
        task.add_note("Test note")
        
        assert len(task.notes) == initial_note_count + 1
        assert "Test note" in task.notes[-1]
        assert any(":" in note for note in task.notes)  # Should contain timestamp

    def test_task_item_mark_completed(self):
        """Test marking TaskItem as completed."""
        task = TaskItem(
            task_id="T001",
            description="Test task",
            completed=False
        )

        task.mark_completed()
        assert task.completed is True
        assert len(task.notes) > 0
        assert "completed" in task.notes[-1].lower()

        task.mark_completed("Custom completion note")
        assert len(task.notes) > 1
        assert "Custom completion note" in task.notes[-1]


class TestProjectTask:
    """Test cases for ProjectTask model."""

    def test_project_task_creation(self):
        """Test creating a ProjectTask."""
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test project task",
            task_type="implementation",
            priority=3,
            status="pending",
            dependencies=["PROJ-000"],
            prerequisites=["spec_exists"],
            estimated_hours=8.0,
            tags=["backend", "api"]
        )

        assert task.task_id == "PROJ-001"
        assert task.feature_id == "feature-001"
        assert task.task_type == "implementation"
        assert task.priority == 3
        assert task.status == "pending"
        assert task.dependencies == ["PROJ-000"]
        assert task.prerequisites == ["spec_exists"]
        assert task.estimated_hours == 8.0
        assert task.tags == ["backend", "api"]

    def test_project_task_to_dict(self):
        """Test converting ProjectTask to dictionary."""
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test project task",
            task_type="implementation",
            priority=3
        )

        result = task.to_dict()
        
        assert isinstance(result, dict)
        assert result["task_id"] == "PROJ-001"
        assert result["feature_id"] == "feature-001"
        assert result["task_type"] == "implementation"
        assert result["priority"] == 3

    def test_project_task_from_dict(self):
        """Test creating ProjectTask from dictionary."""
        data = {
            "task_id": "PROJ-001",
            "feature_id": "feature-001",
            "description": "Test project task",
            "task_type": "implementation",
            "priority": 3,
            "status": "in_progress",
            "estimated_hours": 5.0,
            "actual_hours": 3.5,
            "tags": ["test"]
        }

        task = ProjectTask.from_dict(data)
        
        assert task.task_id == "PROJ-001"
        assert task.feature_id == "feature-001"
        assert task.status == "in_progress"
        assert task.estimated_hours == 5.0
        assert task.actual_hours == 3.5
        assert task.tags == ["test"]

    def test_project_task_update_status(self):
        """Test updating ProjectTask status."""
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test project task",
            task_type="implementation",
            status="pending"
        )

        # Update to in_progress
        task.update_status("in_progress", "Started working on task")
        assert task.status == "in_progress"
        assert len(task.notes) > 0
        assert "in_progress" in task.notes[-1]

        # Update to completed
        task.update_status("completed")
        assert task.status == "completed"
        assert task.completed_at is not None

    def test_project_task_add_note(self):
        """Test adding a note to ProjectTask."""
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test project task",
            task_type="implementation"
        )

        initial_note_count = len(task.notes)
        task.add_note("Test note")
        
        assert len(task.notes) == initial_note_count + 1
        assert "Test note" in task.notes[-1]
        assert any(":" in note for note in task.notes)  # Should contain timestamp

    def test_project_task_is_executable(self):
        """Test is_executable() method."""
        # Executable task
        task1 = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test task",
            task_type="implementation",
            status="pending"
        )
        assert task1.is_executable() is True

        # Non-executable tasks
        task2 = ProjectTask(
            task_id="PROJ-002",
            feature_id="feature-001",
            description="Completed task",
            task_type="implementation",
            status="completed"
        )
        assert task2.is_executable() is False

        task3 = ProjectTask(
            task_id="PROJ-003",
            feature_id="feature-001",
            description="Blocked task",
            task_type="implementation",
            status="blocked"
        )
        assert task3.is_executable() is False

    def test_project_task_is_high_priority(self):
        """Test is_high_priority() method."""
        # High priority task
        high_priority_task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="High priority task",
            task_type="implementation",
            priority=2
        )
        assert high_priority_task.is_high_priority() is True

        # Normal priority task
        normal_priority_task = ProjectTask(
            task_id="PROJ-002",
            feature_id="feature-001",
            description="Normal priority task",
            task_type="implementation",
            priority=5
        )
        assert normal_priority_task.is_high_priority() is False

    def test_project_task_validation_success(self):
        """Test successful validation of ProjectTask."""
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id="feature-001",
            description="Test project task",
            task_type="implementation",
            priority=5,
            status="pending",
            estimated_hours=8.0,
            actual_hours=6.0
        )

        issues = task.validate()
        assert issues == []

    def test_project_task_validation_failures(self):
        """Test validation failures for ProjectTask."""
        task = ProjectTask(
            task_id="",  # Invalid
            feature_id="",  # Invalid
            description="",  # Invalid
            task_type="invalid_type",  # Invalid
            priority=15,  # Invalid (must be 1-10)
            status="invalid_status",  # Invalid
            estimated_hours=-5.0,  # Invalid
            actual_hours=-2.0  # Invalid
        )

        issues = task.validate()
        assert len(issues) >= 6
        assert "Task ID is required" in issues
        assert "Feature ID is required" in issues
        assert "Description is required" in issues
        assert "Invalid task type" in issues
        assert "Priority must be 1-10" in issues
        assert "Invalid status" in issues
        assert "Estimated hours must be positive" in issues
        assert "Actual hours must be positive" in issues


class TestProjectStatus:
    """Test cases for ProjectStatus model."""

    def test_project_status_creation(self):
        """Test creating ProjectStatus."""
        status = ProjectStatus(
            project_name="Test Project",
            total_features=5,
            completed_features=2,
            total_tasks=20,
            completed_tasks=8,
            estimated_hours_remaining=40.0,
            actual_hours_spent=25.5
        )

        assert status.project_name == "Test Project"
        assert status.total_features == 5
        assert status.completed_features == 2
        assert status.total_tasks == 20
        assert status.completed_tasks == 8
        assert status.estimated_hours_remaining == 40.0
        assert status.actual_hours_spent == 25.5

    def test_project_status_to_dict(self):
        """Test converting ProjectStatus to dictionary."""
        status = ProjectStatus(
            project_name="Test Project",
            total_features=3,
            completed_features=1,
            total_tasks=15,
            completed_tasks=5
        )

        result = status.to_dict()
        
        assert isinstance(result, dict)
        assert result["project_name"] == "Test Project"
        assert result["total_features"] == 3
        assert result["completed_features"] == 1
        assert result["total_tasks"] == 15
        assert result["completed_tasks"] == 5

    def test_project_status_get_completion_rate(self):
        """Test get_completion_rate() method."""
        # No tasks
        status1 = ProjectStatus(project_name="Test", total_tasks=0, completed_tasks=0)
        assert status1.get_completion_rate() == 0.0

        # Half completed
        status2 = ProjectStatus(project_name="Test", total_tasks=10, completed_tasks=5)
        assert status2.get_completion_rate() == 50.0

        # All completed
        status3 = ProjectStatus(project_name="Test", total_tasks=8, completed_tasks=8)
        assert status3.get_completion_rate() == 100.0

    def test_project_status_get_feature_completion_rate(self):
        """Test get_feature_completion_rate() method."""
        # No features
        status1 = ProjectStatus(project_name="Test", total_features=0, completed_features=0)
        assert status1.get_feature_completion_rate() == 0.0

        # One third completed
        status2 = ProjectStatus(project_name="Test", total_features=3, completed_features=1)
        assert status2.get_feature_completion_rate() == pytest.approx(33.33, rel=1e-2)

    def test_project_status_is_healthy(self):
        """Test is_healthy() method."""
        # Healthy project
        healthy_status = ProjectStatus(
            project_name="Test",
            total_tasks=20,
            completed_tasks=10,
            blocked_tasks=2  # 10% blocked
        )
        assert healthy_status.is_healthy() is True

        # Unhealthy due to too many blocked tasks
        unhealthy_status1 = ProjectStatus(
            project_name="Test",
            total_tasks=20,
            completed_tasks=5,
            blocked_tasks=5  # 25% blocked
        )
        assert unhealthy_status1.is_healthy() is False

        # Unhealthy due to low completion rate
        unhealthy_status2 = ProjectStatus(
            project_name="Test",
            total_tasks=20,
            completed_tasks=1,  # 5% completion
            blocked_tasks=1  # 5% blocked
        )
        assert unhealthy_status2.is_healthy() is False

    def test_project_status_add_milestone(self):
        """Test add_milestone() method."""
        status = ProjectStatus(project_name="Test")
        initial_milestone_count = len(status.milestones)
        
        status.add_milestone("Completed feature X")
        
        assert len(status.milestones) == initial_milestone_count + 1
        assert "Completed feature X" in status.milestones[-1]
        assert any(":" in milestone for milestone in status.milestones)  # Should contain timestamp

    def test_project_status_add_blocker(self):
        """Test add_blocker() method."""
        status = ProjectStatus(project_name="Test")
        initial_blocker_count = len(status.blockers)
        
        status.add_blocker("Dependency on external API")
        
        assert len(status.blockers) == initial_blocker_count + 1
        assert "Dependency on external API" in status.blockers

        # Adding the same blocker again should not duplicate
        status.add_blocker("Dependency on external API")
        assert len(status.blockers) == initial_blocker_count + 1

    def test_project_status_remove_blocker(self):
        """Test remove_blocker() method."""
        status = ProjectStatus(project_name="Test")
        status.add_blocker("Test blocker")
        assert len(status.blockers) == 1
        
        status.remove_blocker("Test blocker")
        assert len(status.blockers) == 0
        
        # Removing non-existent blocker should not fail
        status.remove_blocker("Non-existent blocker")
        assert len(status.blockers) == 0


class TestWorkflowStep:
    """Test cases for WorkflowStep model."""

    def test_workflow_step_creation(self):
        """Test creating WorkflowStep."""
        step = WorkflowStep(
            step_number=1,
            name="Test Step",
            tool_name="test_tool",
            description="A test workflow step",
            purpose="To test the workflow",
            prerequisites=["Previous Step"],
            expected_output="Test output"
        )

        assert step.step_number == 1
        assert step.name == "Test Step"
        assert step.tool_name == "test_tool"
        assert step.description == "A test workflow step"
        assert step.purpose == "To test the workflow"
        assert step.prerequisites == ["Previous Step"]
        assert step.expected_output == "Test output"
        assert step.is_completed is False
        assert step.completed_at is None

    def test_workflow_step_to_dict(self):
        """Test converting WorkflowStep to dictionary."""
        step = WorkflowStep(
            step_number=2,
            name="Second Step",
            tool_name="second_tool",
            description="Second workflow step",
            purpose="To continue testing"
        )

        result = step.to_dict()
        
        assert isinstance(result, dict)
        assert result["step_number"] == 2
        assert result["name"] == "Second Step"
        assert result["tool_name"] == "second_tool"
        assert result["description"] == "Second workflow step"
        assert result["purpose"] == "To continue testing"

    def test_workflow_step_mark_completed(self):
        """Test marking WorkflowStep as completed."""
        step = WorkflowStep(
            step_number=1,
            name="Test Step",
            tool_name="test_tool",
            description="A test workflow step",
            purpose="To test the workflow"
        )

        assert step.is_completed is False
        assert step.completed_at is None
        
        step.mark_completed()
        
        assert step.is_completed is True
        assert step.completed_at is not None
        assert any(":" in step.completed_at for _ in [1])  # Should contain timestamp

    def test_workflow_step_can_execute(self):
        """Test can_execute() method."""
        step = WorkflowStep(
            step_number=2,
            name="Second Step",
            tool_name="second_tool",
            description="Second workflow step",
            purpose="To continue testing",
            prerequisites=["First Step"]
        )

        # Cannot execute without prerequisite
        assert step.can_execute([]) is False
        assert step.can_execute(["Other Step"]) is False
        
        # Can execute with prerequisite
        assert step.can_execute(["First Step"]) is True
        assert step.can_execute(["First Step", "Other Step"]) is True

    def test_workflow_steps_constants(self):
        """Test WORKFLOW_STEPS constants."""
        assert isinstance(WORKFLOW_STEPS, list)
        assert len(WORKFLOW_STEPS) == 7  # Should have 7 workflow steps
        
        # Check that all steps have required fields
        for step in WORKFLOW_STEPS:
            assert isinstance(step, WorkflowStep)
            assert step.step_number >= 1
            assert step.name
            assert step.tool_name
            assert step.description
            assert step.purpose


class TestWorkflowIntegration:
    """Integration tests for workflow models working together."""

    def test_feature_analysis_to_project_task_conversion(self):
        """Test converting FeatureAnalysis to ProjectTask."""
        analysis = FeatureAnalysis(
            feature_id="test-001",
            feature_name="Test Feature",
            description="A test feature",
            primary_story="As a user, I want to test",
            actors=["users"],
            actions=["Enable users to test"],
            goals=["test validation"],
            clarifications=[],
            edge_cases=[],
            summary="Test feature enables testing"
        )

        # Create a project task based on the analysis
        task = ProjectTask(
            task_id="PROJ-001",
            feature_id=analysis.feature_id,
            description=f"Implement {analysis.feature_name}",
            task_type="implementation",
            priority=3,
            prerequisites=["spec_exists", "plan_exists"]
        )

        assert task.feature_id == analysis.feature_id
        assert analysis.feature_name in task.description
        assert "spec_exists" in task.prerequisites

    def test_project_status_aggregation(self):
        """Test ProjectStatus aggregation from multiple tasks."""
        tasks = [
            ProjectTask(
                task_id="PROJ-001",
                feature_id="feature-001",
                description="Task 1",
                task_type="implementation",
                priority=2,
                status="completed",
                estimated_hours=5.0,
                actual_hours=4.5
            ),
            ProjectTask(
                task_id="PROJ-002",
                feature_id="feature-001",
                description="Task 2",
                task_type="implementation",
                priority=3,
                status="in_progress",
                estimated_hours=8.0,
                actual_hours=3.0
            ),
            ProjectTask(
                task_id="PROJ-003",
                feature_id="feature-002",
                description="Task 3",
                task_type="validation",
                priority=1,
                status="blocked",
                estimated_hours=3.0
            )
        ]

        # Create project status based on tasks
        status = ProjectStatus(
            project_name="Test Project",
            total_features=2,
            completed_features=0,
            total_tasks=len(tasks),
            completed_tasks=len([t for t in tasks if t.status == "completed"]),
            in_progress_tasks=len([t for t in tasks if t.status == "in_progress"]),
            blocked_tasks=len([t for t in tasks if t.status == "blocked"]),
            high_priority_tasks=len([t for t in tasks if t.is_high_priority()]),
            estimated_hours_remaining=sum(t.estimated_hours or 0 for t in tasks if t.status != "completed"),
            actual_hours_spent=sum(t.actual_hours or 0 for t in tasks if t.status == "completed")
        )

        assert status.total_tasks == 3
        assert status.completed_tasks == 1
        assert status.in_progress_tasks == 1
        assert status.blocked_tasks == 1
        assert status.high_priority_tasks == 1
        assert status.estimated_hours_remaining == 11.0  # 8.0 + 3.0
        assert status.actual_hours_spent == 4.5
        assert status.get_completion_rate() == pytest.approx(33.33, rel=1e-2)