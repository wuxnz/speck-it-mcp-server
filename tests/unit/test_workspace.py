"""Unit tests for Speck-It workspace functionality.

This module tests the core workspace operations including
artifact generation, file management, and workflow validation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.models import FeatureAnalysis, FeatureArtifacts
from src.workspace import Workspace


class TestWorkspaceInitialization:
    """Test cases for workspace initialization."""

    def test_workspace_creation(self, tmp_path):
        """Test creating a new workspace."""
        workspace = Workspace(tmp_path)
        
        assert workspace.root == tmp_path.resolve()
        assert workspace.base_dir == tmp_path / ".speck-it"
        assert workspace.memory_dir == tmp_path / ".speck-it" / "memory"
        assert workspace.specs_dir == tmp_path / ".speck-it" / "specs"
        assert workspace.data_dir == tmp_path / ".speck-it" / "state"
        assert workspace.tasks_dir == tmp_path / ".speck-it" / "project_tasks"
        
        # Check directories were created
        assert workspace.base_dir.exists()
        assert workspace.memory_dir.exists()
        assert workspace.specs_dir.exists()
        assert workspace.data_dir.exists()
        assert workspace.tasks_dir.exists()

    def test_workspace_with_custom_storage_dir(self, tmp_path, monkeypatch):
        """Test workspace with custom storage directory."""
        monkeypatch.setenv("SPECKIT_STORAGE_DIR", ".custom-speck-it")
        workspace = Workspace(tmp_path)
        
        assert workspace.base_dir == tmp_path / ".custom-speck-it"
        assert workspace.base_dir.exists()

    def test_workspace_with_string_path(self):
        """Test workspace creation with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Workspace(temp_dir)
            assert workspace.root == Path(temp_dir).resolve()


class TestConstitutionManagement:
    """Test cases for constitution management."""

    def test_save_constitution_replace(self, tmp_path):
        """Test saving constitution with replace mode."""
        workspace = Workspace(tmp_path)
        content = "# Test Constitution\n\nThis is a test constitution."
        
        path = workspace.save_constitution(content, mode="replace")
        
        assert path == workspace.constitution_path
        assert path.exists()
        assert path.read_text(encoding="utf-8") == content + "\n"

    def test_save_constitution_append(self, tmp_path):
        """Test saving constitution with append mode."""
        workspace = Workspace(tmp_path)
        initial_content = "# Initial Constitution\n\nInitial content."
        additional_content = "## Additional Section\n\nMore content."
        
        # Save initial content
        workspace.save_constitution(initial_content, mode="replace")
        
        # Append additional content
        path = workspace.save_constitution(additional_content, mode="append")
        
        assert path.exists()
        full_content = path.read_text(encoding="utf-8")
        assert initial_content in full_content
        assert additional_content in full_content

    def test_save_constitution_invalid_mode(self, tmp_path):
        """Test saving constitution with invalid mode."""
        workspace = Workspace(tmp_path)
        
        with pytest.raises(ValueError, match="mode must be 'replace' or 'append'"):
            workspace.save_constitution("content", mode="invalid")

    def test_load_constitution_exists(self, tmp_path):
        """Test loading existing constitution."""
        workspace = Workspace(tmp_path)
        content = "# Test Constitution\n\nThis is a test."
        
        workspace.save_constitution(content)
        loaded_content = workspace.load_constitution()
        
        assert loaded_content == content

    def test_load_constitution_not_exists(self, tmp_path):
        """Test loading constitution when it doesn't exist."""
        workspace = Workspace(tmp_path)
        
        loaded_content = workspace.load_constitution()
        assert loaded_content is None


class TestFeatureManagement:
    """Test cases for feature management."""

    def test_list_features_empty(self, tmp_path):
        """Test listing features when none exist."""
        workspace = Workspace(tmp_path)
        
        features = workspace.list_features()
        assert features == []

    def test_list_features_with_features(self, tmp_path):
        """Test listing features when some exist."""
        workspace = Workspace(tmp_path)
        
        # Create feature directories
        feature1_dir = workspace.specs_dir / "001-test-feature"
        feature2_dir = workspace.specs_dir / "002-another-feature"
        feature1_dir.mkdir()
        feature2_dir.mkdir()
        
        # Create analysis files
        analysis1 = FeatureAnalysis(
            feature_id="001-test-feature",
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
        
        analysis_data = analysis1.to_dict()
        (feature1_dir / "analysis.json").write_text(json.dumps(analysis_data))
        
        # Create spec files
        (feature1_dir / "spec.md").write_text("# Test Spec")
        (feature1_dir / "plan.md").write_text("# Test Plan")
        (feature2_dir / "spec.md").write_text("# Another Spec")
        
        features = workspace.list_features()
        
        assert len(features) == 2
        assert features[0]["feature_id"] == "001-test-feature"
        assert features[0]["feature_name"] == "Test Feature"
        assert features[0]["spec_path"] is not None
        assert features[0]["plan_path"] is not None
        assert features[0]["tasks_path"] is None
        
        assert features[1]["feature_id"] == "002-another-feature"
        assert features[1]["feature_name"] == "002-another-feature"
        assert features[1]["spec_path"] is not None
        assert features[1]["plan_path"] is None

    def test_next_feature_number_empty(self, tmp_path):
        """Test getting next feature number when no features exist."""
        workspace = Workspace(tmp_path)
        
        assert workspace._next_feature_number() == 1

    def test_next_feature_number_with_existing(self, tmp_path):
        """Test getting next feature number with existing features."""
        workspace = Workspace(tmp_path)
        
        # Create existing feature directories
        (workspace.specs_dir / "001-feature").mkdir()
        (workspace.specs_dir / "003-another").mkdir()
        (workspace.specs_dir / "non-numeric").mkdir()
        
        assert workspace._next_feature_number() == 4

    def test_feature_identifier_with_id(self, tmp_path):
        """Test generating feature identifier with provided ID."""
        workspace = Workspace(tmp_path)
        
        identifier = workspace._feature_identifier("Test Feature", "custom-id")
        assert identifier == "001-custom-id"

    def test_feature_identifier_without_id(self, tmp_path):
        """Test generating feature identifier without provided ID."""
        workspace = Workspace(tmp_path)
        
        identifier = workspace._feature_identifier("Test Feature", None)
        assert identifier.startswith("001-")
        assert "test-feature" in identifier

    def test_feature_dir_creation(self, tmp_path):
        """Test feature directory creation."""
        workspace = Workspace(tmp_path)
        
        feature_dir = workspace._feature_dir("test-feature")
        expected_dir = workspace.specs_dir / "test-feature"
        
        assert feature_dir == expected_dir
        assert feature_dir.exists()

    def test_save_and_load_analysis(self, tmp_path):
        """Test saving and loading feature analysis."""
        workspace = Workspace(tmp_path)
        feature_dir = workspace._feature_dir("test-feature")
        
        analysis = FeatureAnalysis(
            feature_id="test-feature",
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
        
        # Save analysis
        workspace._save_analysis(feature_dir, analysis)
        
        # Load analysis
        loaded_analysis = workspace._load_analysis("test-feature")
        
        assert loaded_analysis is not None
        assert loaded_analysis.feature_id == analysis.feature_id
        assert loaded_analysis.feature_name == analysis.feature_name
        assert loaded_analysis.description == analysis.description

    def test_load_analysis_not_exists(self, tmp_path):
        """Test loading analysis when it doesn't exist."""
        workspace = Workspace(tmp_path)
        
        analysis = workspace._load_analysis("non-existent-feature")
        assert analysis is None


class TestArtifactExistenceChecks:
    """Test cases for artifact existence checks."""

    def test_spec_exists_true(self, tmp_path):
        """Test spec_exists when spec exists."""
        workspace = Workspace(tmp_path)
        feature_dir = workspace._feature_dir("test-feature")
        (feature_dir / "spec.md").write_text("# Test Spec")
        
        assert workspace.spec_exists("test-feature") is True

    def test_spec_exists_false(self, tmp_path):
        """Test spec_exists when spec doesn't exist."""
        workspace = Workspace(tmp_path)
        
        assert workspace.spec_exists("test-feature") is False

    def test_plan_exists_true(self, tmp_path):
        """Test plan_exists when plan exists."""
        workspace = Workspace(tmp_path)
        feature_dir = workspace._feature_dir("test-feature")
        (feature_dir / "plan.md").write_text("# Test Plan")
        
        assert workspace.plan_exists("test-feature") is True

    def test_plan_exists_false(self, tmp_path):
        """Test plan_exists when plan doesn't exist."""
        workspace = Workspace(tmp_path)
        
        assert workspace.plan_exists("test-feature") is False

    def test_tasks_exists_true(self, tmp_path):
        """Test tasks_exists when tasks exist."""
        workspace = Workspace(tmp_path)
        feature_dir = workspace._feature_dir("test-feature")
        (feature_dir / "tasks.md").write_text("# Test Tasks")
        
        assert workspace.tasks_exists("test-feature") is True

    def test_tasks_exists_false(self, tmp_path):
        """Test tasks_exists when tasks don't exist."""
        workspace = Workspace(tmp_path)
        
        assert workspace.tasks_exists("test-feature") is False


class TestArtifactGeneration:
    """Test cases for artifact generation."""

    def test_generate_spec_basic(self, tmp_path):
        """Test basic specification generation."""
        workspace = Workspace(tmp_path)
        
        artifacts, analysis, content = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature that enables users to test functionality",
            save=True
        )
        
        # Check artifacts
        assert isinstance(artifacts, FeatureArtifacts)
        assert artifacts.feature_id.startswith("001-")
        assert artifacts.spec_path is not None
        assert artifacts.spec_path.exists()
        
        # Check analysis
        assert isinstance(analysis, FeatureAnalysis)
        assert analysis.feature_name == "Test Feature"
        assert analysis.description == "A test feature that enables users to test functionality"
        assert "users" in analysis.actors
        assert len(analysis.actions) > 0
        assert len(analysis.goals) > 0
        
        # Check content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "# Feature Specification: Test Feature" in content
        assert "## Quick Summary" in content
        assert "## User Scenarios & Testing" in content
        assert "## Requirements" in content

    def test_generate_spec_without_saving(self, tmp_path):
        """Test specification generation without saving."""
        workspace = Workspace(tmp_path)
        
        artifacts, analysis, content = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=False
        )
        
        assert artifacts.spec_path is None
        assert not artifacts.feature_dir.exists()
        assert len(content) > 0

    def test_generate_spec_with_custom_id(self, tmp_path):
        """Test specification generation with custom feature ID."""
        workspace = Workspace(tmp_path)
        
        artifacts, analysis, content = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            feature_id="custom-001",
            save=True
        )
        
        assert artifacts.feature_id == "001-custom-001"
        assert artifacts.spec_path.parent.name == "001-custom-001"

    def test_generate_plan_success(self, tmp_path):
        """Test successful plan generation."""
        workspace = Workspace(tmp_path)
        
        # First generate a spec
        spec_artifacts, spec_analysis, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature for plan generation",
            save=True
        )
        
        # Then generate a plan
        plan_artifacts, plan_analysis, plan_content = workspace.generate_plan(
            feature_id=spec_artifacts.feature_id,
            save=True
        )
        
        # Check artifacts
        assert plan_artifacts.feature_id == spec_artifacts.feature_id
        assert plan_artifacts.plan_path is not None
        assert plan_artifacts.plan_path.exists()
        
        # Check analysis
        assert plan_analysis.feature_id == spec_analysis.feature_id
        
        # Check content
        assert isinstance(plan_content, str)
        assert len(plan_content) > 0
        assert "# Implementation Plan: Test Feature" in plan_content
        assert "## Technical Context" in plan_content
        assert "## Design Deliverables" in plan_content
        assert "## Task Generation Strategy" in plan_content

    def test_generate_plan_without_spec(self, tmp_path):
        """Test plan generation when spec doesn't exist."""
        workspace = Workspace(tmp_path)
        
        with pytest.raises(FileNotFoundError, match="Feature.*does not have analysis data"):
            workspace.generate_plan("non-existent-feature")

    def test_generate_tasks_success(self, tmp_path):
        """Test successful task generation."""
        workspace = Workspace(tmp_path)
        
        # Generate spec first
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature for task generation",
            save=True
        )
        
        # Generate plan
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        
        # Generate tasks
        task_artifacts, task_analysis, task_content = workspace.generate_tasks(
            feature_id=spec_artifacts.feature_id,
            save=True
        )
        
        # Check artifacts
        assert task_artifacts.feature_id == spec_artifacts.feature_id
        assert task_artifacts.tasks_path is not None
        assert task_artifacts.tasks_path.exists()
        
        # Check analysis
        assert task_analysis.feature_id == spec_artifacts.feature_id
        
        # Check content
        assert isinstance(task_content, str)
        assert len(task_content) > 0
        assert "# Tasks: Test Feature" in task_content
        assert "## Task List" in task_content
        assert "- [ ] T001" in task_content

    def test_generate_tasks_without_plan(self, tmp_path):
        """Test task generation when plan doesn't exist."""
        workspace = Workspace(tmp_path)
        
        # Generate spec only
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        
        with pytest.raises(FileNotFoundError, match="Expected plan at"):
            workspace.generate_tasks(feature_id=spec_artifacts.feature_id)


class TestTaskManagement:
    """Test cases for task management."""

    def test_list_tasks_success(self, tmp_path):
        """Test successful task listing."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # List tasks
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all("task_id" in task for task in tasks)
        assert all("description" in task for task in tasks)
        assert all("completed" in task for task in tasks)

    def test_list_tasks_no_tasks_file(self, tmp_path):
        """Test listing tasks when tasks file doesn't exist."""
        workspace = Workspace(tmp_path)
        
        with pytest.raises(FileNotFoundError, match="No tasks.md found"):
            workspace.list_tasks("non-existent-feature")

    def test_update_task_completion(self, tmp_path):
        """Test updating task completion status."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Get first task
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        first_task = tasks[0]
        
        # Update task completion
        result = workspace.update_task(
            feature_id=spec_artifacts.feature_id,
            task_id=first_task["task_id"],
            completed=True,
            note="Task completed successfully"
        )
        
        assert result["task"]["completed"] is True
        assert "Task completed successfully" in str(result["task"]["notes"])
        assert result["remaining"] >= 0

    def test_update_task_add_note(self, tmp_path):
        """Test updating task with note."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Get first task
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        first_task = tasks[0]
        
        # Add note without changing completion
        initial_completed = first_task["completed"]
        result = workspace.update_task(
            feature_id=spec_artifacts.feature_id,
            task_id=first_task["task_id"],
            note="Added a note"
        )
        
        assert result["task"]["completed"] == initial_completed
        assert "Added a note" in str(result["task"]["notes"])

    def test_update_task_not_found(self, tmp_path):
        """Test updating non-existent task."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        with pytest.raises(ValueError, match="Task 'NON-EXISTENT' not found"):
            workspace.update_task(
                feature_id=spec_artifacts.feature_id,
                task_id="NON-EXISTENT",
                completed=True
            )

    def test_next_open_task(self, tmp_path):
        """Test getting next open task."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Get next task
        next_task = workspace.next_open_task(spec_artifacts.feature_id)
        
        assert next_task is not None
        assert next_task["completed"] is False
        assert "task_id" in next_task
        assert "description" in next_task

    def test_next_open_task_all_completed(self, tmp_path):
        """Test getting next task when all are completed."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Complete all tasks
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        for task in tasks:
            workspace.update_task(
                feature_id=spec_artifacts.feature_id,
                task_id=task["task_id"],
                completed=True
            )
        
        # Get next task
        next_task = workspace.next_open_task(spec_artifacts.feature_id)
        assert next_task is None

    def test_complete_task(self, tmp_path):
        """Test completing a task."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Get first task
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        first_task = tasks[0]
        
        # Complete task
        result = workspace.complete_task(
            feature_id=spec_artifacts.feature_id,
            task_id=first_task["task_id"],
            note="Task completed via test"
        )
        
        assert result["task"]["completed"] is True
        assert "Task completed via test" in str(result["task"]["notes"])
        assert result["all_completed"] is False  # Should have other tasks remaining

    def test_feature_status(self, tmp_path):
        """Test getting feature status."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Get feature status
        status = workspace.feature_status(spec_artifacts.feature_id)
        
        assert status["feature_id"] == spec_artifacts.feature_id
        assert status["spec_path"] is not None
        assert status["plan_path"] is not None
        assert status["tasks_path"] is not None
        assert status["tasks"]["total"] > 0
        assert status["tasks"]["completed"] == 0
        assert status["tasks"]["remaining"] > 0
        assert status["tasks"]["all_completed"] is False

    def test_finalize_feature_success(self, tmp_path):
        """Test successful feature finalization."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Complete all tasks
        tasks = workspace.list_tasks(spec_artifacts.feature_id)
        for task in tasks:
            workspace.complete_task(
                feature_id=spec_artifacts.feature_id,
                task_id=task["task_id"]
            )
        
        # Finalize feature
        result = workspace.finalize_feature(spec_artifacts.feature_id)
        
        assert result["feature_id"] == spec_artifacts.feature_id
        assert "finalized_at" in result
        assert result["tasks"]["all_completed"] is True

    def test_finalize_feature_incomplete_tasks(self, tmp_path):
        """Test finalizing feature with incomplete tasks."""
        workspace = Workspace(tmp_path)
        
        # Generate complete workflow
        spec_artifacts, _, _ = workspace.generate_spec(
            feature_name="Test Feature",
            description="A test feature",
            save=True
        )
        workspace.generate_plan(feature_id=spec_artifacts.feature_id, save=True)
        workspace.generate_tasks(feature_id=spec_artifacts.feature_id, save=True)
        
        # Don't complete tasks - try to finalize
        with pytest.raises(ValueError, match="Cannot finalize feature.*pending tasks remain"):
            workspace.finalize_feature(spec_artifacts.feature_id)


class TestTextProcessingUtilities:
    """Test cases for text processing utility methods."""

    def test_slugify(self, tmp_path):
        """Test slugify method."""
        workspace = Workspace(tmp_path)
        
        assert workspace._slugify("Test Feature") == "test-feature"
        assert workspace._slugify("Test Feature with Spaces") == "test-feature-with-spaces"
        assert workspace._slugify("Test_Feature-With_Special!@#Chars") == "test-feature-with-special-chars"
        assert workspace._slugify("") == "feature"

    def test_split_sentences(self, tmp_path):
        """Test sentence splitting."""
        workspace = Workspace(tmp_path)
        
        text = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = workspace._split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one"
        assert sentences[1] == "This is sentence two"
        assert sentences[2] == "This is sentence three"

    def test_sentence_keywords(self, tmp_path):
        """Test keyword extraction from sentences."""
        workspace = Workspace(tmp_path)
        
        sentence = "This is a test sentence with some keywords"
        keywords = workspace._sentence_keywords(sentence)
        
        assert "this" in keywords
        assert "test" in keywords
        assert "sentence" in keywords
        assert "keywords" in keywords
        assert "is" not in keywords  # Too short
        assert "a" not in keywords    # Too short
        assert "with" not in keywords  # Too short

    def test_extract_actors(self, tmp_path):
        """Test actor extraction."""
        workspace = Workspace(tmp_path)
        
        # Test with user mention
        description1 = "This feature allows users to manage their data"
        actors1 = workspace._extract_actors(description1)
        assert "users" in actors1
        
        # Test with admin mention
        description2 = "Administrators can configure system settings"
        actors2 = workspace._extract_actors(description2)
        assert "administrators" in actors2
        
        # Test with multiple actors
        description3 = "Developers and team members can collaborate on projects"
        actors3 = workspace._extract_actors(description3)
        assert "developers" in actors3
        assert "team members" in actors3
        
        # Test with no clear actors (should default to users)
        description4 = "This is a generic feature description"
        actors4 = workspace._extract_actors(description4)
        assert "users" in actors4

    def test_extract_actions(self, tmp_path):
        """Test action extraction."""
        workspace = Workspace(tmp_path)
        
        sentences = [
            "Allow users to create new accounts",
            "Enable administrators to manage settings",
            "Support file uploads and downloads",
            "This feature provides data visualization"
        ]
        
        actions = workspace._extract_actions(sentences)
        
        assert len(actions) == 4
        assert any("create new accounts" in action.lower() for action in actions)
        assert any("manage settings" in action.lower() for action in actions)
        assert any("file uploads" in action.lower() for action in actions)
        assert any("data visualization" in action.lower() for action in actions)

    def test_extract_goals(self, tmp_path):
        """Test goal extraction."""
        workspace = Workspace(tmp_path)
        
        actions = [
            "Enable users to create new accounts",
            "Allow administrators to manage settings",
            "Support file uploads and downloads"
        ]
        
        goals = workspace._extract_goals(actions)
        
        assert len(goals) == 3
        assert "create new accounts" in goals
        assert "manage settings" in goals
        assert "file uploads and downloads" in goals

    def test_build_primary_story(self, tmp_path):
        """Test primary story building."""
        workspace = Workspace(tmp_path)
        
        actors = ["users", "administrators"]
        goals = ["manage data", "configure settings"]
        feature_name = "Management System"
        description = "A system for managing data and settings"
        
        story = workspace._build_primary_story(actors, goals, feature_name, description)
        
        assert "As users, administrators" in story
        assert "I want to manage data" in story
        assert "so that Management System delivers value" in story

    def test_identify_clarifications(self, tmp_path):
        """Test clarification identification."""
        workspace = Workspace(tmp_path)
        
        # Test with ambiguous terms
        description1 = "This feature should be fast and secure"
        actions1 = ["Enable users to access the system quickly"]
        clarifications1 = workspace._identify_clarifications(description1, actions1)
        
        assert any("FAST requirement" in clarification for clarification in clarifications1)
        assert any("SECURE requirement" in clarification for clarification in clarifications1)
        
        # Test with authentication
        description2 = "Users can login with their credentials"
        actions2 = ["Enable users to authenticate"]
        clarifications2 = workspace._identify_clarifications(description2, actions2)
        
        assert any("Authentication method" in clarification for clarification in clarifications2)

    def test_build_edge_cases(self, tmp_path):
        """Test edge case building."""
        workspace = Workspace(tmp_path)
        
        actions = [
            "Enable users to create accounts",
            "Allow administrators to delete data"
        ]
        
        edge_cases = workspace._build_edge_cases(actions)
        
        assert len(edge_cases) >= 4  # 2 per action
        assert any("create accounts" in case.lower() for case in edge_cases)
        assert any("delete data" in case.lower() for case in edge_cases)
        assert any("fails due to invalid input" in case.lower() for case in edge_cases)
        assert any("lacks permissions" in case.lower() for case in edge_cases)

    def test_build_summary(self, tmp_path):
        """Test summary building."""
        workspace = Workspace(tmp_path)
        
        feature_name = "Test Feature"
        actors = ["users", "administrators"]
        goals = ["manage data", "configure settings"]
        
        summary = workspace._build_summary(feature_name, actors, goals)
        
        assert "Test Feature empowers" in summary
        assert "users, administrators" in summary
        assert "manage data" in summary
        assert "configure settings" in summary

    def test_parse_context(self, tmp_path):
        """Test context parsing."""
        workspace = Workspace(tmp_path)
        
        context_text = """
        language: Python
        framework: FastAPI
        database: PostgreSQL
        testing: pytest
        performance: <100ms response time
        """
        
        context = workspace._parse_context(context_text)
        
        assert context["language"] == "Python"
        assert context["primary_dependencies"] == "FastAPI"
        assert context["storage"] == "PostgreSQL"
        assert context["testing"] == "pytest"
        assert context["performance_goals"] == "<100ms response time"

    def test_constitution_excerpt(self, tmp_path):
        """Test constitution excerpt generation."""
        workspace = Workspace(tmp_path)
        
        # Test with no constitution
        excerpt = workspace._constitution_excerpt()
        assert "No constitution recorded" in excerpt
        
        # Test with constitution
        constitution = "# Project Constitution\n\n## Principles\n- Quality first\n- Test-driven development\n\n## Standards\n- Follow best practices\n- Write clean code"
        workspace.save_constitution(constitution)
        
        excerpt = workspace._constitution_excerpt()
        assert "Project Constitution" in excerpt
        assert "Quality first" in excerpt
        assert "Test-driven development" in excerpt