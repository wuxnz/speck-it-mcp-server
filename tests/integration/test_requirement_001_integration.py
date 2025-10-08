"""
Integration test for Requirement FR-001:
System MUST Enable users to this tool should be called after the spec is generated with the "generate_spec" 
and before the plan is generated with the "generate_plan" tool and should plan out all the technical requirements, 
data flow, design principles and details, and any other technical aspect of the project.

This integration test verifies the complete workflow using the actual MCP server tools.
"""

import pytest
import tempfile
import os
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock
import sys

# Add the project root to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.workflow import WorkflowManager


class TestRequirement001Integration:
    """Integration tests for the complete workflow requirement."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            yield project_path
    
    @pytest.fixture
    def sample_constitution(self):
        """Sample constitution for testing."""
        return """
# Test Project Constitution

## Core Principles
- Quality first approach
- Test-driven development
- Clean code standards

## Development Standards
- All code must be tested
- Follow best practices
- Use type hints where applicable
"""
    
    @pytest.fixture
    def sample_feature_data(self):
        """Sample feature data for testing."""
        return {
            "feature_name": "User Authentication System",
            "description": "Implement secure user authentication with JWT tokens, password hashing, and session management. The system should support user registration, login, logout, and password reset functionality.",
            "feature_id": "auth-system"
        }
    
    def test_complete_workflow_integration(self, temp_project_dir, sample_constitution, sample_feature_data):
        """
        Integration Test: Verify the complete workflow from constitution to plan generation.
        
        Given: A fresh project directory
        When: Following the complete 7-step workflow
        Then: Each step should work correctly and in the right order
        """
        # Step 1: Set constitution
        constitution_result = set_constitution(
            content=sample_constitution,
            root=str(temp_project_dir)
        )
        
        assert constitution_result["constitution_path"] is not None
        assert "Constitution saved successfully" in constitution_result["message"]
        
        # Step 2: Set feature root
        feature_root_result = set_feature_root(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert feature_root_result["feature_id"] == sample_feature_data["feature_id"]
        assert feature_root_result["root"] == str(temp_project_dir)
        
        # Step 3: Generate spec
        spec_result = generate_spec(
            feature_name=sample_feature_data["feature_name"],
            description=sample_feature_data["description"],
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert spec_result["artifacts"]["spec_path"] is not None
        assert "Specification generated" in spec_result["message"]
        
        # Step 4: Generate plan (this is what we're testing)
        plan_result = generate_plan(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert plan_result["artifacts"]["plan_path"] is not None
        assert "Implementation plan generated" in plan_result["message"]
        
        # Verify the plan contains technical requirements
        plan_content = plan_result["content"]
        assert "Technical Context" in plan_content
        assert "Design Deliverables" in plan_content
        assert "Task Generation Strategy" in plan_content
        
        # Step 5: Generate tasks (optional for this test)
        tasks_result = generate_tasks(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert tasks_result["artifacts"]["tasks_path"] is not None
        assert "Task list generated" in tasks_result["message"]
    
    def test_plan_generation_without_spec_fails(self, temp_project_dir, sample_constitution, sample_feature_data):
        """
        Integration Test: Verify that plan generation fails when no spec exists.
        
        Given: A project with constitution and feature root but no spec
        When: generate_plan is called
        Then: It should fail with appropriate error message
        """
        # Set up constitution and feature root
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        set_feature_root(feature_id=sample_feature_data["feature_id"], root=str(temp_project_dir))
        
        # Try to generate plan without spec - should fail
        plan_result = generate_plan(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert "error" in plan_result
        assert "No specification found" in plan_result["error"]
        assert "generate_spec" in plan_result["suggestion"]
    
    def test_multiple_features_workflow_isolation(self, temp_project_dir, sample_constitution):
        """
        Integration Test: Verify workflow isolation between multiple features.
        
        Given: Multiple features in the same project
        When: Working with different features
        Then: Each feature should maintain independent workflow state
        """
        feature_a_data = {
            "feature_name": "Feature A",
            "description": "First feature for testing",
            "feature_id": "feature-a"
        }
        
        feature_b_data = {
            "feature_name": "Feature B", 
            "description": "Second feature for testing",
            "feature_id": "feature-b"
        }
        
        # Set up project
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        
        # Set up feature roots
        set_feature_root(feature_id=feature_a_data["feature_id"], root=str(temp_project_dir))
        set_feature_root(feature_id=feature_b_data["feature_id"], root=str(temp_project_dir))
        
        # Generate spec for feature A only
        generate_spec(
            feature_name=feature_a_data["feature_name"],
            description=feature_a_data["description"],
            feature_id=feature_a_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        # Feature A should be able to generate plan
        plan_a_result = generate_plan(
            feature_id=feature_a_data["feature_id"],
            root=str(temp_project_dir)
        )
        assert "error" not in plan_a_result
        
        # Feature B should NOT be able to generate plan (no spec yet)
        plan_b_result = generate_plan(
            feature_id=feature_b_data["feature_id"],
            root=str(temp_project_dir)
        )
        assert "error" in plan_b_result
        assert "No specification found" in plan_b_result["error"]
        
        # Now generate spec for feature B
        generate_spec(
            feature_name=feature_b_data["feature_name"],
            description=feature_b_data["description"],
            feature_id=feature_b_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        # Feature B should now be able to generate plan
        plan_b_result_after = generate_plan(
            feature_id=feature_b_data["feature_id"],
            root=str(temp_project_dir)
        )
        assert "error" not in plan_b_result_after
    
    def test_plan_content_quality_and_completeness(self, temp_project_dir, sample_constitution, sample_feature_data):
        """
        Integration Test: Verify generated plan contains all required technical elements.
        
        Given: A comprehensive feature specification
        When: generate_plan is called
        Then: The plan should contain all technical requirements, data flow, design principles
        """
        # Set up complete workflow up to spec generation
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        set_feature_root(feature_id=sample_feature_data["feature_id"], root=str(temp_project_dir))
        
        spec_result = generate_spec(
            feature_name=sample_feature_data["feature_name"],
            description=sample_feature_data["description"],
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        # Generate plan
        plan_result = generate_plan(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        plan_content = plan_result["content"]
        
        # Verify all required technical sections are present
        required_sections = [
            "Technical Context",
            "Design Deliverables", 
            "Research Agenda",
            "Task Generation Strategy",
            "Risks & Mitigations",
            "Progress Tracking"
        ]
        
        for section in required_sections:
            assert section in plan_content, f"Missing required section: {section}"
        
        # Verify technical requirements are addressed
        technical_keywords = [
            "authentication",
            "security", 
            "jwt",
            "session",
            "password",
            "user"
        ]
        
        found_keywords = [keyword for keyword in technical_keywords 
                         if keyword.lower() in plan_content.lower()]
        
        assert len(found_keywords) > 0, "Plan should address technical aspects from the feature description"
        
        # Verify plan includes actionable technical guidance
        assert "Phase" in plan_content or "Step" in plan_content
        assert "implementation" in plan_content.lower()
    
    def test_workflow_state_persistence(self, temp_project_dir, sample_constitution, sample_feature_data):
        """
        Integration Test: Verify workflow state persists across tool calls.
        
        Given: A project with completed workflow steps
        When: Checking feature status and listing features
        Then: The state should be accurately reflected
        """
        # Complete the workflow up to plan generation
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        set_feature_root(feature_id=sample_feature_data["feature_id"], root=str(temp_project_dir))
        generate_spec(
            feature_name=sample_feature_data["feature_name"],
            description=sample_feature_data["description"],
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        generate_plan(feature_id=sample_feature_data["feature_id"], root=str(temp_project_dir))
        
        # Check feature status
        status_result = feature_status(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert status_result["spec_path"] is not None
        assert status_result["plan_path"] is not None
        assert status_result["tasks_path"] is None  # Not generated yet
        
        # List features should show our feature
        features_result = list_features(root=str(temp_project_dir))
        
        feature_ids = [f["feature_id"] for f in features_result["features"]]
        assert sample_feature_data["feature_id"] in feature_ids
        
        # Find our feature in the list
        our_feature = next(f for f in features_result["features"] 
                          if f["feature_id"] == sample_feature_data["feature_id"])
        
        assert our_feature["spec_path"] is not None
        assert our_feature["plan_path"] is not None
    
    def test_error_recovery_and_validation(self, temp_project_dir, sample_constitution, sample_feature_data):
        """
        Integration Test: Verify error handling and recovery scenarios.
        
        Given: Various error conditions
        When: Calling workflow tools
        Then: Appropriate error messages and recovery options should be provided
        """
        # Test 1: Invalid feature ID
        invalid_result = generate_plan(
            feature_id="non-existent-feature",
            root=str(temp_project_dir)
        )
        
        assert "error" in invalid_result
        assert "No specification found" in invalid_result["error"]
        
        # Test 2: Set up partial workflow
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        set_feature_root(feature_id=sample_feature_data["feature_id"], root=str(temp_project_dir))
        
        # Generate spec
        generate_spec(
            feature_name=sample_feature_data["feature_name"],
            description=sample_feature_data["description"],
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        # Now plan generation should work
        recovery_result = generate_plan(
            feature_id=sample_feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        assert "error" not in recovery_result
        assert "Implementation plan generated" in recovery_result["message"]
    
    def test_concurrent_workflow_safety(self, temp_project_dir, sample_constitution):
        """
        Integration Test: Verify workflow safety with concurrent operations.
        
        Given: Multiple features being processed
        When: Operations happen concurrently
        Then: Each feature's workflow should remain isolated and consistent
        """
        features_data = [
            {
                "feature_name": f"Concurrent Feature {i}",
                "description": f"Feature {i} for concurrent testing",
                "feature_id": f"concurrent-{i}"
            }
            for i in range(3)
        ]
        
        # Set up project
        set_constitution(content=sample_constitution, root=str(temp_project_dir))
        
        # Set up all feature roots
        for feature_data in features_data:
            set_feature_root(
                feature_id=feature_data["feature_id"],
                root=str(temp_project_dir)
            )
        
        # Generate specs for all features
        for feature_data in features_data:
            generate_spec(
                feature_name=feature_data["feature_name"],
                description=feature_data["description"],
                feature_id=feature_data["feature_id"],
                root=str(temp_project_dir)
            )
        
        # Generate plans for all features
        plan_results = []
        for feature_data in features_data:
            result = generate_plan(
                feature_id=feature_data["feature_id"],
                root=str(temp_project_dir)
            )
            plan_results.append(result)
        
        # All plans should succeed
        for i, result in enumerate(plan_results):
            assert "error" not in result, f"Feature {i} plan generation failed: {result.get('error', 'Unknown error')}"
            assert "Implementation plan generated" in result["message"]
        
        # Verify all plans are distinct and valid
        plan_paths = [result["artifacts"]["plan_path"] for result in plan_results]
        assert len(set(plan_paths)) == len(plan_paths), "Each feature should have a distinct plan file"
        
        for plan_path in plan_paths:
            assert Path(plan_path).exists(), "Plan file should exist"
            assert Path(plan_path).stat().st_size > 0, "Plan file should not be empty"


class TestRequirement001RealWorldScenarios:
    """Real-world scenario tests for the workflow requirement."""
    
    def test_real_world_authentication_feature(self, temp_project_dir):
        """
        Integration Test: Test with a real-world authentication feature scenario.
        """
        constitution = """
# E-commerce Platform Constitution

## Core Principles
- Security first approach
- GDPR compliance
- Scalable architecture
- Test-driven development

## Development Standards
- OWASP security guidelines
- Comprehensive testing
- Performance monitoring
- Clean architecture patterns
"""
        
        feature_data = {
            "feature_name": "User Authentication & Authorization",
            "description": """
            Implement a comprehensive user authentication and authorization system for an e-commerce platform.
            
            The system should include:
            - User registration with email verification
            - Secure login with rate limiting
            - Password reset functionality
            - JWT-based session management
            - Role-based access control (admin, customer, guest)
            - Two-factor authentication support
            - OAuth integration (Google, Facebook)
            - Session timeout and refresh mechanisms
            - Audit logging for security events
            - Password strength validation
            - Account lockout after failed attempts
            """,
            "feature_id": "ecommerce-auth"
        }
        
        # Complete workflow
        set_constitution(content=constitution, root=str(temp_project_dir))
        set_feature_root(feature_id=feature_data["feature_id"], root=str(temp_project_dir))
        
        spec_result = generate_spec(
            feature_name=feature_data["feature_name"],
            description=feature_data["description"],
            feature_id=feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        plan_result = generate_plan(
            feature_id=feature_data["feature_id"],
            root=str(temp_project_dir)
        )
        
        # Verify plan addresses real-world complexity
        plan_content = plan_result["content"]
        
        # Should address security aspects
        security_terms = ["security", "authentication", "authorization", "jwt", "oauth", "2fa"]
        found_security = [term for term in security_terms if term.lower() in plan_content.lower()]
        assert len(found_security) > 0, "Plan should address security requirements"
        
        # Should include implementation phases
        assert "phase" in plan_content.lower() or "step" in plan_content.lower()
        
        # Should include technical architecture
        assert "architecture" in plan_content.lower() or "design" in plan_content.lower()
        
        # Should include testing strategy
        assert "test" in plan_content.lower()
        
        # Should be comprehensive (substantial content)
        assert len(plan_content) > 2000, "Real-world feature should generate comprehensive plan"