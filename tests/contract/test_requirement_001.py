"""
Contract test for Requirement FR-001:
System MUST Enable users to this tool should be called after the spec is generated with the "generate_spec" 
and before the plan is generated with the "generate_plan" tool and should plan out all the technical requirements, 
data flow, design principles and details, and any other technical aspect of the project.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from src.workspace import Workspace as SpecKitWorkspace
from src.models import FeatureArtifacts, FeatureAnalysis


class TestRequirement001WorkflowOrder:
    """Contract tests for workflow order requirement - generate_spec must be called before generate_plan."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SpecKitWorkspace(Path(temp_dir))
            # Create .speck-it directory structure
            speck_it_dir = workspace.base_dir / ".speck-it"
            speck_it_dir.mkdir(exist_ok=True)
            (speck_it_dir / "memory").mkdir(exist_ok=True)
            (speck_it_dir / "specs").mkdir(exist_ok=True)
            (speck_it_dir / "state").mkdir(exist_ok=True)
            yield workspace
    
    @pytest.fixture
    def sample_feature_data(self):
        """Sample feature data for testing."""
        return {
            "feature_name": "Test Feature",
            "description": "A test feature for contract testing",
            "feature_id": "test-001"
        }
    
    def test_generate_plan_requires_existing_spec(self, temp_workspace, sample_feature_data):
        """
        Contract Test: Verify that generate_plan fails when no spec exists.
        
        Given: A workspace without a generated spec
        When: generate_plan is called
        Then: It should raise an appropriate error indicating spec is required
        """
        feature_id = sample_feature_data["feature_id"]
        
        # Ensure no spec exists
        assert not temp_workspace.spec_exists(feature_id)
        
        # Attempt to generate plan without spec
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_id)
    
    def test_generate_plan_succeeds_after_spec_generation(self, temp_workspace, sample_feature_data):
        """
        Contract Test: Verify that generate_plan succeeds after spec is generated.
        
        Given: A workspace with a generated spec
        When: generate_plan is called
        Then: It should successfully generate a plan
        """
        feature_id = sample_feature_data["feature_id"]
        feature_name = sample_feature_data["feature_name"]
        description = sample_feature_data["description"]
        
        # First, generate a spec
        artifacts, analysis, content = temp_workspace.generate_spec(
            feature_name, description, feature_id=feature_id
        )
        
        # Verify spec was created
        assert temp_workspace.spec_exists(feature_id)
        assert artifacts.spec_path.exists()
        
        # Now generate plan - should succeed
        plan_artifacts, plan_analysis, plan_content = temp_workspace.generate_plan(feature_id)
        
        # Verify plan was created successfully
        assert plan_artifacts.plan_path.exists()
        assert plan_content is not None
        assert len(plan_content) > 0
    
    def test_workflow_order_enforcement(self, temp_workspace, sample_feature_data):
        """
        Contract Test: Verify the complete workflow order is enforced.
        
        Given: A new workspace
        When: Following the complete workflow
        Then: Each step must be completed in the correct order
        """
        feature_id = sample_feature_data["feature_id"]
        feature_name = sample_feature_data["feature_name"]
        description = sample_feature_data["description"]
        
        # Step 1: Cannot generate plan without spec
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_id)
        
        # Step 2: Generate spec - should succeed
        spec_artifacts, spec_analysis, spec_content = temp_workspace.generate_spec(
            feature_name, description, feature_id=feature_id
        )
        assert temp_workspace.spec_exists(feature_id)
        
        # Step 3: Generate plan - should now succeed
        plan_artifacts, plan_analysis, plan_content = temp_workspace.generate_plan(feature_id)
        assert temp_workspace.plan_exists(feature_id)
        
        # Step 4: Generate tasks - should succeed
        task_artifacts, task_analysis, task_content = temp_workspace.generate_tasks(feature_id)
        assert temp_workspace.tasks_exists(feature_id)
    
    def test_plan_content_includes_technical_requirements(self, temp_workspace, sample_feature_data):
        """
        Contract Test: Verify generated plan includes technical requirements and design details.
        
        Given: A generated spec
        When: generate_plan is called
        Then: The plan should contain technical requirements, data flow, design principles
        """
        feature_id = sample_feature_data["feature_id"]
        feature_name = sample_feature_data["feature_name"]
        description = "A feature that requires technical planning with data flow and design principles"
        
        # Generate spec first
        temp_workspace.generate_spec(feature_name, description, feature_id=feature_id)
        
        # Generate plan
        artifacts, analysis, content = temp_workspace.generate_plan(feature_id)
        
        # Verify plan contains expected technical sections
        assert "Technical Context" in content
        assert "Design Deliverables" in content
        assert "Research Agenda" in content
        assert "Task Generation Strategy" in content
        assert "Risks & Mitigations" in content
        
        # Verify plan includes technical aspects
        assert any(term in content.lower() for term in [
            "technical", "requirements", "data", "flow", "design", "principles"
        ])
    
    def test_multiple_features_workflow_isolation(self, temp_workspace):
        """
        Contract Test: Verify workflow order is enforced per feature.
        
        Given: Multiple features in the same workspace
        When: Working with different features
        Then: Workflow order should be enforced independently for each feature
        """
        feature_a = "feature-a"
        feature_b = "feature-b"
        
        # Generate spec for feature A only
        temp_workspace.generate_spec("Feature A", "Description A", feature_id=feature_a)
        
        # Feature A should be able to generate plan
        assert temp_workspace.spec_exists(feature_a)
        temp_workspace.generate_plan(feature_a)  # Should succeed
        
        # Feature B should NOT be able to generate plan (no spec yet)
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_b)
        
        # Generate spec for feature B
        temp_workspace.generate_spec("Feature B", "Description B", feature_id=feature_b)
        
        # Now feature B should be able to generate plan
        temp_workspace.generate_plan(feature_b)  # Should succeed
    
    def test_error_handling_invalid_spec_state(self, temp_workspace, sample_feature_data):
        """
        Contract Test: Verify proper error handling for invalid spec states.
        
        Given: Various invalid spec states
        When: generate_plan is called
        Then: Appropriate errors should be raised
        """
        feature_id = sample_feature_data["feature_id"]
        
        # Test 1: Non-existent feature directory
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_id)
        
        # Test 2: Empty spec file
        spec_dir = temp_workspace.base_dir / ".speck-it" / "specs" / feature_id
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_file = spec_dir / "spec.md"
        spec_file.write_text("")  # Empty spec
        
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_id)
        
        # Test 3: Corrupted spec file (invalid markdown)
        spec_file.write_text("Invalid content without proper spec structure")
        
        # This should still fail gracefully
        with pytest.raises(ValueError, match="No specification found"):
            temp_workspace.generate_plan(feature_id)
    
    def test_plan_generation_with_comprehensive_spec(self, temp_workspace):
        """
        Contract Test: Verify plan generation works with comprehensive spec data.
        
        Given: A comprehensive specification with all required sections
        When: generate_plan is called
        Then: Plan should be generated with all technical aspects
        """
        feature_id = "comprehensive-feature"
        
        # Create a comprehensive spec
        comprehensive_spec = """
# Feature Specification: Comprehensive Feature

## Quick Summary
This is a comprehensive feature with detailed requirements.

## User Scenarios & Testing
### Primary User Story
As a user, I want comprehensive functionality so that I can achieve my goals.

### Acceptance Scenarios
1. Given the feature is available, When I use it, Then it works correctly.

## Requirements
### Functional Requirements
- FR-001: System must provide core functionality
- FR-002: System must handle data processing
- FR-003: System must ensure security

### Key Entities
- User: Represents system users
- Data: Represents processed information
- Security: Represents security constraints
"""
        
        # Create spec directory and file
        spec_dir = temp_workspace.base_dir / ".speck-it" / "specs" / feature_id
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_file = spec_dir / "spec.md"
        spec_file.write_text(comprehensive_spec)
        
        # Generate plan
        artifacts, analysis, content = temp_workspace.generate_plan(feature_id)
        
        # Verify comprehensive plan content
        assert "Technical Context" in content
        assert "Design Deliverables" in content
        assert "Task Generation Strategy" in content
        assert len(content) > 1000  # Should be substantial
        
        # Verify plan addresses the requirements from spec
        assert "FR-001" in content or "core functionality" in content.lower()
        assert "FR-002" in content or "data processing" in content.lower()
        assert "FR-003" in content or "security" in content.lower()


class TestRequirement001Integration:
    """Integration tests for the workflow requirement."""
    
    def test_end_to_end_workflow_with_mcp_tools(self):
        """
        Contract Test: Verify the workflow works end-to-end with MCP tools.
        
        This test simulates the actual MCP tool usage pattern.
        """
        # This would be an integration test that actually calls the MCP tools
        # For now, we'll test the underlying workspace logic
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SpecKitWorkspace(Path(temp_dir))
            
            # Setup workspace structure
            speck_it_dir = workspace.base_dir / ".speck-it"
            speck_it_dir.mkdir(exist_ok=True)
            (speck_it_dir / "memory").mkdir(exist_ok=True)
            (speck_it_dir / "specs").mkdir(exist_ok=True)
            (speck_it_dir / "state").mkdir(exist_ok=True)
            
            feature_id = "integration-test"
            
            # Step 1: Try to generate plan (should fail)
            with pytest.raises(ValueError, match="No specification found"):
                workspace.generate_plan(feature_id)
            
            # Step 2: Generate spec
            spec_artifacts, spec_analysis, spec_content = workspace.generate_spec(
                "Integration Test Feature",
                "A feature for integration testing",
                feature_id=feature_id
            )
            
            # Step 3: Generate plan (should succeed)
            plan_artifacts, plan_analysis, plan_content = workspace.generate_plan(feature_id)
            
            # Verify the workflow completed successfully
            assert spec_artifacts.spec_path.exists()
            assert plan_artifacts.plan_path.exists()
            assert len(spec_content) > 0
            assert len(plan_content) > 0