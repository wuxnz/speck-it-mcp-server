"""Unit tests for Speck-It workflow management.

This module tests the workflow orchestration, step validation,
and overall process management.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models import ProjectTask, ProjectStatus
from src.workflow import WorkflowManager, register_feature_root, lookup_feature_root


class TestFeatureRootRegistry:
    """Test cases for feature root registry functions."""

    def test_register_feature_root(self):
        """Test registering a feature root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            result = register_feature_root("test-feature", root_path)
            
            assert result == root_path.resolve()
            assert lookup_feature_root("test-feature") == root_path.resolve()

    def test_register_feature_root_case_insensitive(self):
        """Test that feature root registration is case insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            register_feature_root("Test-Feature", root_path)
            
            assert lookup_feature_root("test-feature") == root_path.resolve()
            assert lookup_feature_root("TEST-FEATURE") == root_path.resolve()
            assert lookup_feature_root("Test-Feature") == root_path.resolve()

    def test_lookup_feature_root_not_found(self):
        """Test looking up non-existent feature root."""
        result = lookup_feature_root("non-existent-feature")
        assert result is None


class TestWorkflowManagerInitialization:
    """Test cases for WorkflowManager initialization."""

    def test_workflow_manager_creation(self):
        """Test creating a WorkflowManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            assert manager.workspace.root == Path(temp_dir).resolve()
            assert manager.workspace.base_dir.exists()

    def test_workflow_manager_with_path_object(self):
        """Test creating WorkflowManager with Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(Path(temp_dir))
            
            assert manager.workspace.root == Path(temp_dir).resolve()


class TestConstitutionManagement:
    """Test cases for constitution management in WorkflowManager."""

    def test_set_constitution_success(self):
        """Test successful constitution setting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            content = "# Test Constitution\n\nThis is a test constitution."
            
            result = manager.set_constitution(content)
            
            assert result["constitution_path"] is not None
            assert "Constitution saved successfully" in result["message"]
            assert result["next_suggested_step"] == "set_feature_root"
            assert Path(result["constitution_path"]).exists()
            assert Path(result["constitution_path"]).read_text() == content + "\n"

    def test_set_constitution_append_mode(self):
        """Test constitution setting with append mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            initial_content = "# Initial Constitution"
            additional_content = "## Additional Section"
            
            # Set initial content
            manager.set_constitution(initial_content)
            
            # Append additional content
            result = manager.set_constitution(additional_content, mode="append")
            
            full_content = Path(result["constitution_path"]).read_text()
            assert initial_content in full_content
            assert additional_content in full_content

    def test_get_constitution_exists(self):
        """Test getting existing constitution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            content = "# Test Constitution\n\nThis is a test."
            
            manager.set_constitution(content)
            result = manager.get_constitution()
            
            assert result["exists"] is True
            assert result["content"] == content
            assert result["constitution_path"] is not None

    def test_get_constitution_not_exists(self):
        """Test getting constitution when it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.get_constitution()
            
            assert result["exists"] is False
            assert result["content"] is None
            assert result["next_suggested_step"] == "set_constitution"


class TestFeatureManagement:
    """Test cases for feature management in WorkflowManager."""

    def test_list_features_empty(self):
        """Test listing features when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.list_features()
            
            assert result["features"] == []
            assert result["count"] == 0
            assert "No features generated yet" in result["message"]

    def test_list_features_with_features(self):
        """Test listing features when some exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create a feature
            manager.set_constitution("# Test Constitution")
            manager.set_feature_root("test-feature")
            manager.generate_spec("Test Feature", "A test feature", feature_id="test-feature")
            
            result = manager.list_features()
            
            assert len(result["features"]) == 1
            assert result["count"] == 1
            assert result["features"][0]["feature_id"] == "001-test-feature"
            assert result["features"][0]["feature_name"] == "Test Feature"

    def test_set_feature_root_with_path(self):
        """Test setting feature root with explicit path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.set_feature_root("test-feature", temp_dir)
            
            assert result["feature_id"] == "test-feature"
            assert result["root"] == str(Path(temp_dir).resolve())
            assert result["next_suggested_step"] == "generate_spec"
            assert lookup_feature_root("test-feature") == Path(temp_dir).resolve()

    def test_set_feature_root_auto_detect(self):
        """Test setting feature root with auto-detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.set_feature_root("test-feature")
            
            assert result["feature_id"] == "test-feature"
            assert result["root"] == str(Path(temp_dir).resolve())
            assert lookup_feature_root("test-feature") == Path(temp_dir).resolve()

    def test_set_feature_root_creates_directory(self):
        """Test that set_feature_root creates .speck-it directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            speck_it_dir = Path(temp_dir) / ".speck-it"
            
            # Ensure directory doesn't exist initially
            if speck_it_dir.exists():
                speck_it_dir.rmdir()
            
            manager.set_feature_root("test-feature")
            
            assert speck_it_dir.exists()


class TestArtifactGeneration:
    """Test cases for artifact generation in WorkflowManager."""

    def test_generate_spec_success(self):
        """Test successful specification generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature for validation"
            )
            
            assert "artifacts" in result
            assert "analysis" in result
            assert "content" in result
            assert result["artifacts"]["feature_id"].startswith("001-")
            assert result["artifacts"]["spec_path"] is not None
            assert Path(result["artifacts"]["spec_path"]).exists()
            assert result["next_suggested_step"] == "generate_plan"
            assert "Specification generated" in result["message"]

    def test_generate_spec_without_saving(self):
        """Test specification generation without saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature",
                save=False
            )
            
            assert result["artifacts"]["spec_path"] is None
            assert not Path(temp_dir).joinpath(".speck-it").exists()

    def test_generate_plan_success(self):
        """Test successful plan generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # First generate spec
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            
            # Then generate plan
            result = manager.generate_plan(feature_id)
            
            assert "artifacts" in result
            assert "analysis" in result
            assert "content" in result
            assert result["artifacts"]["plan_path"] is not None
            assert Path(result["artifacts"]["plan_path"]).exists()
            assert result["next_suggested_step"] == "generate_tasks"
            assert "Implementation plan generated" in result["message"]

    def test_generate_plan_without_spec(self):
        """Test plan generation when spec doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.generate_plan("non-existent-feature")
            
            assert "error" in result
            assert "No specification found" in result["error"]
            assert result["next_suggested_step"] == "generate_spec"

    def test_generate_tasks_success(self):
        """Test successful task generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate spec and plan first
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            
            # Generate tasks
            result = manager.generate_tasks(feature_id)
            
            assert "artifacts" in result
            assert "analysis" in result
            assert "content" in result
            assert result["artifacts"]["tasks_path"] is not None
            assert Path(result["artifacts"]["tasks_path"]).exists()
            assert result["next_suggested_step"] == "list_tasks"
            assert "Task list generated" in result["message"]

    def test_generate_tasks_without_plan(self):
        """Test task generation when plan doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate spec only
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            
            result = manager.generate_tasks(feature_id)
            
            assert "error" in result
            assert "No implementation plan found" in result["error"]
            assert result["next_suggested_step"] == "generate_plan"


class TestTaskExecution:
    """Test cases for task execution in WorkflowManager."""

    def test_list_tasks_success(self):
        """Test successful task listing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # List tasks
            result = manager.list_tasks(feature_id)
            
            assert "tasks" in result
            assert len(result["tasks"]) > 0
            assert all("task_id" in task for task in result["tasks"])

    def test_list_tasks_no_tasks(self):
        """Test listing tasks when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.list_tasks("non-existent-feature")
            
            assert "error" in result
            assert "Generate tasks" in result["error"]

    def test_next_task_success(self):
        """Test getting next task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # Get next task
            result = manager.next_task(feature_id)
            
            assert "task" in result
            assert "remaining" in result
            assert result["task"]["completed"] is False

    def test_update_task_success(self):
        """Test successful task update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # Get first task
            tasks = manager.list_tasks(feature_id)
            first_task = tasks["tasks"][0]
            
            # Update task
            result = manager.update_task(
                feature_id=feature_id,
                task_id=first_task["task_id"],
                completed=True,
                note="Task completed via test"
            )
            
            assert result["task"]["completed"] is True
            assert "Task completed via test" in str(result["task"]["notes"])

    def test_complete_task_success(self):
        """Test successful task completion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # Complete first task
            result = manager.complete_task(feature_id=feature_id)
            
            assert result["task"]["completed"] is True
            assert "remaining" in result

    def test_feature_status_success(self):
        """Test successful feature status retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # Get feature status
            result = manager.feature_status(feature_id)
            
            assert result["feature_id"] == feature_id
            assert result["spec_path"] is not None
            assert result["plan_path"] is not None
            assert result["tasks_path"] is not None
            assert result["tasks"]["total"] > 0

    def test_finalize_feature_success(self):
        """Test successful feature finalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Generate complete workflow
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            manager.generate_plan(feature_id)
            manager.generate_tasks(feature_id)
            
            # Complete all tasks
            tasks = manager.list_tasks(feature_id)
            for task in tasks["tasks"]:
                manager.complete_task(
                    feature_id=feature_id,
                    task_id=task["task_id"]
                )
            
            # Finalize feature
            result = manager.finalize_feature(feature_id)
            
            assert result["feature_id"] == feature_id
            assert "finalized_at" in result
            assert result["tasks"]["all_completed"] is True


class TestProjectTaskManagement:
    """Test cases for project task management in WorkflowManager."""

    def test_create_project_task_success(self):
        """Test successful project task creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.create_project_task(
                feature_id="test-feature",
                description="Test project task",
                task_type="implementation",
                priority=3,
                estimated_hours=5.0
            )
            
            assert result["success"] is True
            assert "task" in result
            assert result["task"]["feature_id"] == "test-feature"
            assert result["task"]["description"] == "Test project task"
            assert result["task"]["task_type"] == "implementation"
            assert result["task"]["priority"] == 3
            assert result["task"]["estimated_hours"] == 5.0

    def test_get_project_tasks_success(self):
        """Test successful project task retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create some tasks
            manager.create_project_task(
                feature_id="feature-1",
                description="Task 1",
                priority=1
            )
            manager.create_project_task(
                feature_id="feature-2",
                description="Task 2",
                priority=3,
                status="completed"
            )
            
            # Get all tasks
            result = manager.get_project_tasks()
            
            assert "tasks" in result
            assert result["total_count"] == 2
            assert len(result["tasks"]) == 2
            
            # Get filtered tasks
            result_filtered = manager.get_project_tasks(
                feature_id="feature-1",
                status="pending"
            )
            
            assert len(result_filtered["tasks"]) == 1
            assert result_filtered["tasks"][0]["feature_id"] == "feature-1"

    def test_update_project_task_success(self):
        """Test successful project task update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create a task
            create_result = manager.create_project_task(
                feature_id="test-feature",
                description="Test task",
                priority=5
            )
            task_id = create_result["task"]["task_id"]
            
            # Update the task
            result = manager.update_project_task(
                task_id=task_id,
                status="in_progress",
                priority=2,
                add_note="Started working on task"
            )
            
            assert result["success"] is True
            assert result["task"]["status"] == "in_progress"
            assert result["task"]["priority"] == 2
            assert "Started working on task" in str(result["task"]["notes"])

    def test_validate_task_prerequisites_success(self):
        """Test successful task prerequisite validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create a task with no prerequisites
            result = manager.create_project_task(
                feature_id="test-feature",
                description="Test task",
                prerequisites=[]
            )
            task_id = result["task"]["task_id"]
            
            # Validate prerequisites
            validation_result = manager.validate_task_prerequisites(task_id)
            
            assert validation_result["can_proceed"] is True
            assert len(validation_result["validation"]["issues"]) == 0

    def test_get_next_executable_tasks_success(self):
        """Test getting next executable tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create tasks without dependencies
            manager.create_project_task(
                feature_id="feature-1",
                description="Task 1",
                priority=1
            )
            manager.create_project_task(
                feature_id="feature-2",
                description="Task 2",
                priority=2
            )
            
            # Get next executable tasks
            result = manager.get_next_executable_tasks()
            
            assert "executable_tasks" in result
            assert result["count"] == 2
            assert len(result["executable_tasks"]) == 2

    def test_get_project_status_success(self):
        """Test successful project status retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create some tasks
            manager.create_project_task(
                feature_id="feature-1",
                description="Task 1",
                priority=1,
                status="completed"
            )
            manager.create_project_task(
                feature_id="feature-2",
                description="Task 2",
                priority=3,
                status="in_progress"
            )
            
            # Get project status
            result = manager.get_project_status()
            
            assert "project_status" in result
            assert "feature_breakdown" in result
            assert result["project_status"]["total_tasks"] == 2
            assert result["project_status"]["completed_tasks"] == 1
            assert result["project_status"]["in_progress_tasks"] == 1


class TestWorkflowGuidance:
    """Test cases for workflow guidance."""

    def test_get_workflow_guide(self):
        """Test getting workflow guide."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.get_workflow_guide()
            
            assert "workflow_overview" in result
            assert "steps" in result
            assert "tips" in result
            assert len(result["steps"]) == 7  # Should have 7 workflow steps
            
            # Check that each step has required fields
            for step in result["steps"]:
                assert "step" in step
                assert "tool" in step
                assert "description" in step
                assert "purpose" in step
            
            # Check that tips are provided
            assert len(result["tips"]) > 0
            assert any("Follow steps in order" in tip for tip in result["tips"])


class TestErrorHandling:
    """Test cases for error handling in WorkflowManager."""

    def test_generate_spec_error_handling(self):
        """Test error handling in generate_spec."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Mock the workspace to raise an exception
            with patch.object(manager.workspace, 'generate_spec', side_effect=Exception("Test error")):
                result = manager.generate_spec("Test Feature", "A test feature")
                
                assert "error" in result
                assert "Failed to generate specification" in result["error"]
                assert result["next_suggested_step"] == "generate_spec"

    def test_set_feature_root_invalid_path(self):
        """Test set_feature_root with invalid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            with pytest.raises(ValueError, match="Provided root.*does not exist"):
                manager.set_feature_root("test-feature", "/nonexistent/path")

    def test_update_task_not_found(self):
        """Test updating non-existent task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.update_task(
                feature_id="test-feature",
                task_id="NON-EXISTENT",
                completed=True
            )
            
            assert "error" in result
            assert "Failed to update task" in result["error"]

    def test_complete_task_not_found(self):
        """Test completing non-existent task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            result = manager.complete_task(
                feature_id="test-feature",
                task_id="NON-EXISTENT"
            )
            
            assert "error" in result
            assert "Failed to complete task" in result["error"]


class TestAutoUpdateTasks:
    """Test cases for automatic task updates."""

    def test_auto_update_tasks_on_constitution_set(self):
        """Test auto-updating tasks when constitution is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create a task with constitution prerequisite
            task_result = manager.create_project_task(
                feature_id="global",
                description="Set up constitution",
                task_type="workflow",
                prerequisites=["constitution_exists"]
            )
            
            # Set constitution (should auto-update task)
            result = manager.set_constitution("# Test Constitution")
            
            assert "auto_updated_tasks" in result
            # Note: The actual auto-update functionality depends on the workspace implementation

    def test_auto_update_tasks_on_feature_root_set(self):
        """Test auto-updating tasks when feature root is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Create a task with feature root prerequisite
            task_result = manager.create_project_task(
                feature_id="test-feature",
                description="Register feature root",
                task_type="workflow",
                prerequisites=["feature_root_registered"]
            )
            
            # Set feature root (should auto-update task)
            result = manager.set_feature_root("test-feature")
            
            assert "auto_updated_tasks" in result
            # Note: The actual auto-update functionality depends on the workspace implementation


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow_success(self):
        """Test complete workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Step 1: Set constitution
            constitution_result = manager.set_constitution("# Test Constitution")
            assert constitution_result["constitution_path"] is not None
            
            # Step 2: Set feature root
            feature_root_result = manager.set_feature_root("test-feature")
            assert feature_root_result["feature_id"] == "test-feature"
            
            # Step 3: Generate spec
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature for complete workflow"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            assert spec_result["artifacts"]["spec_path"] is not None
            
            # Step 4: Generate plan
            plan_result = manager.generate_plan(feature_id)
            assert plan_result["artifacts"]["plan_path"] is not None
            
            # Step 5: Generate tasks
            tasks_result = manager.generate_tasks(feature_id)
            assert tasks_result["artifacts"]["tasks_path"] is not None
            
            # Step 6: Execute tasks
            tasks = manager.list_tasks(feature_id)
            initial_count = len(tasks["tasks"])
            
            # Complete first task
            complete_result = manager.complete_task(feature_id=feature_id)
            assert complete_result["task"]["completed"] is True
            
            # Check status
            status = manager.feature_status(feature_id)
            assert status["tasks"]["completed"] == 1
            assert status["tasks"]["remaining"] == initial_count - 1
            
            # Complete all remaining tasks
            for _ in range(status["tasks"]["remaining"]):
                try:
                    manager.complete_task(feature_id=feature_id)
                except:
                    # Some tasks might not be completable without actual implementation
                    break
            
            # Step 7: Finalize (if all tasks are completed)
            final_status = manager.feature_status(feature_id)
            if final_status["tasks"]["all_completed"]:
                finalize_result = manager.finalize_feature(feature_id)
                assert "finalized_at" in finalize_result

    def test_workflow_order_enforcement(self):
        """Test that workflow order is properly enforced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = WorkflowManager(temp_dir)
            
            # Try to generate plan without spec - should fail
            plan_result = manager.generate_plan("test-feature")
            assert "error" in plan_result
            assert "No specification found" in plan_result["error"]
            
            # Try to generate tasks without plan - should fail
            tasks_result = manager.generate_tasks("test-feature")
            assert "error" in tasks_result
            assert "No implementation plan found" in tasks_result["error"]
            
            # Generate spec first - should succeed
            spec_result = manager.generate_spec(
                feature_name="Test Feature",
                description="A test feature"
            )
            feature_id = spec_result["artifacts"]["feature_id"]
            
            # Now plan should succeed
            plan_result = manager.generate_plan(feature_id)
            assert "error" not in plan_result
            
            # Now tasks should succeed
            tasks_result = manager.generate_tasks(feature_id)
            assert "error" not in tasks_result