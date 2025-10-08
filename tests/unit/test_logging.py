"""Unit tests for Speck-It logging and observability.

This module tests the logging infrastructure, performance monitoring,
and observability hooks.
"""

import json
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.logging import (
    setup_logging,
    JsonFormatter,
    PerformanceMonitor,
    log_performance,
    log_operation,
    ObservabilityHooks,
    log_workflow_step,
    log_error_with_context,
    observability_hooks,
)


class TestJsonFormatter:
    """Test cases for JsonFormatter."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JsonFormatter()
        
        # Create a log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            "test",
            logging.INFO,
            "Test message",
            (),  # args
            None  # exc_info
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse JSON
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert "module" in data
        assert "function" in data
        assert "line" in data

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JsonFormatter()
        
        # Create a log record with exception
        logger = logging.getLogger("test")
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logger.makeRecord(
                "test",
                logging.ERROR,
                "Test message",
                (),  # args
                True  # exc_info
            )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse JSON
        data = json.loads(formatted)
        
        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JsonFormatter()
        
        # Create a log record with extra fields
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            "test",
            logging.INFO,
            "Test message",
            (),  # args
            None  # exc_info
        )
        record.extra_fields = {"custom_field": "custom_value"}
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse JSON
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["custom_field"] == "custom_value"


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    def test_record_metric(self):
        """Test recording a performance metric."""
        monitor = PerformanceMonitor()
        
        # Record a metric
        monitor.record_metric("test_metric", 42, {"tag": "test"})
        
        # Get metrics
        metrics = monitor.get_metrics("test_metric")
        
        assert len(metrics) == 1
        assert metrics["test_metric"][0]["value"] == 42
        assert metrics["test_metric"][0]["tags"]["tag"] == "test"
        assert "timestamp" in metrics["test_metric"][0]

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        monitor = PerformanceMonitor()
        
        # Record multiple metrics
        monitor.record_metric("metric1", 1)
        monitor.record_metric("metric2", 2)
        monitor.record_metric("metric1", 3)
        
        # Get all metrics
        all_metrics = monitor.get_metrics()
        
        assert len(all_metrics) == 2
        assert len(all_metrics["metric1"]) == 2
        assert len(all_metrics["metric2"]) == 1
        assert all_metrics["metric1"][0]["value"] == 1
        assert all_metrics["metric1"][1]["value"] == 3
        assert all_metrics["metric2"][0]["value"] == 2


class TestLogPerformance:
    """Test cases for log_performance decorator."""

    def test_log_performance_decorator(self):
        """Test the log_performance decorator."""
        @log_performance("test_operation")
        def test_function():
            return "test_result"
        
        # Call the decorated function
        result = test_function()
        
        assert result == "test_result"
        
        # Check that metric was recorded
        metrics = performance_monitor.get_metrics("test_operation_duration")
        assert len(metrics) == 1
        assert metrics["test_operation_duration"][0]["value"] > 0
        assert metrics["test_operation_duration"][0]["tags"]["status"] == "success"

    def test_log_performance_decorator_with_exception(self):
        """Test the log_performance decorator with exception."""
        @log_performance("test_operation")
        def test_function():
            raise ValueError("Test error")
        
        # Call the decorated function
        with pytest.raises(ValueError):
            test_function()
        
        # Check that metric was recorded
        metrics = performance_monitor.get_metrics("test_operation_duration")
        assert len(metrics) == 1
        assert metrics["test_operation_duration"][0]["tags"]["status"] == "error"
        assert metrics["test_operation_duration"][0]["tags"]["error_type"] == "ValueError"


class TestLogOperation:
    """Test cases for log_operation context manager."""

    def test_log_operation_success(self):
        """Test successful operation logging."""
        with patch("src.logging.logging.getLogger") as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            with log_operation("test_operation", param1="value1"):
                pass
            
            # Check that operation was logged
            assert mock_logger_instance.info.called
            assert mock_logger_instance.error.called is False

    def test_log_operation_with_exception(self):
        """Test operation logging with exception."""
        with patch("src.logging.logging.getLogger") as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            with pytest.raises(ValueError):
                with log_operation("test_operation"):
                    raise ValueError("Test error")
            
            # Check that error was logged
            assert mock_logger_instance.error.called
            assert "Test error" in str(mock_logger_instance.error.call_args)


class TestObservabilityHooks:
    """Test cases for ObservabilityHooks."""

    def test_register_and_trigger_hooks(self):
        """Test registering and triggering hooks."""
        hooks = ObservabilityHooks()
        callback_called = False
        
        def test_callback(**data):
            nonlocal callback_called
            callback_called = True
            assert "test_param" in data
        
        # Register hook
        hooks.register_hook("test_event", test_callback)
        
        # Trigger hook
        hooks.trigger_hooks("test_event", test_param="test_value")
        
        # Check that callback was called
        assert callback_called

    def test_log_workflow_event(self):
        """Test logging workflow events."""
        hooks = ObservabilityHooks()
        
        # Log a workflow event
        hooks.log_workflow_event("test_event", feature_id="test-feature", param="value")
        
        # Check that the event was logged (we can't easily test this without mocking)
        # This test mainly ensures the method doesn't raise exceptions
        assert True

    def test_hook_failure_handling(self):
        """Test that hook failures don't crash the system."""
        hooks = ObservabilityHooks()
        
        def failing_callback(event_type, **data):
            raise ValueError("Hook failed")
        
        # Register failing hook
        hooks.register_hook("test_event", failing_callback)
        
        # Trigger hook - should not raise exception
        hooks.trigger_hooks("test_event", param="value")
        
        # If we get here, the exception was handled properly
        assert True


class TestLoggingFunctions:
    """Test cases for logging convenience functions."""

    def test_log_workflow_step(self):
        """Test log_workflow_step function."""
        with patch("src.logging.observability_hooks") as mock_hooks:
            log_workflow_step("test_step", feature_id="test-feature")
            
            # Check that the event was logged
            mock_hooks.log_workflow_event.assert_called_once()
            call_args = mock_hooks.log_workflow_event.call_args
            assert call_args[0] == "workflow_step_test_step"
            assert call_args[1]["feature_id"] == "test-feature"
            assert call_args[1]["step_name"] == "test_step"

    def test_log_error_with_context(self):
        """Test log_error_with_context function."""
        with patch("src.logging.logging.getLogger") as mock_logger:
            error = ValueError("Test error")
            context = {"operation": "test_operation", "param": "value"}
            
            log_error_with_context(error, context, extra_param="extra_value")
            
            # Check that error was logged
            assert mock_logger.return_value.error.called
            call_args = mock_logger.return_value.error.call_args
            assert "Test error" in str(call_args)
            assert call_args[1]["extra_fields"]["operation"] == "test_operation"
            assert call_args[1]["extra_fields"]["param"] == "value"
            assert call_args[1]["extra_fields"]["extra_param"] == "extra_value"
            assert call_args[1]["extra_fields"]["error_type"] == "ValueError"


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_setup_logging(self):
        """Test setting up logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Setup logging
            setup_logging(log_level=logging.DEBUG, log_file=log_file)
            
            # Get logger
            logger = logging.getLogger("speckit.test")
            
            # Log a message
            logger.info("Test message")
            
            # Check that log file was created and contains the message
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
            
            # Check that content is valid JSON
            lines = content.strip().split("\n")
            for line in lines:
                json.loads(line)  # Should not raise exception

    def test_end_to_end_logging_flow(self):
        """Test end-to-end logging flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Setup logging
            setup_logging(log_level=logging.INFO, log_file=log_file)
            
            # Register a hook
            def test_hook(event_type, **data):
                pass
            
            observability_hooks.register_hook("test_event", test_hook)
            
            # Log a workflow step
            log_workflow_step("test_step", feature_id="test-feature")
            
            # Trigger the hook
            observability_hooks.trigger_hooks("test_event", param="value")
            
            # Record a metric
            performance_monitor.record_metric("test_metric", 42)
            
            # Check that log file contains our events
            content = log_file.read_text()
            assert "workflow_step_test_step" in content
            assert "Metric recorded: test_metric=42" in content