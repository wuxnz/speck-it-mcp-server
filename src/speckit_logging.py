"""Logging and observability utilities for Speck-It.

This module provides structured logging, performance monitoring,
and observability hooks for the Speck-It workflow system.
"""

from __future__ import annotations

import json
import time
import logging as std_logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import wraps

# Configure structured logging
def setup_logging(log_level: Union[str, int] = std_logging.INFO, log_file: Optional[Path] = None) -> None:
    """Setup structured logging for Speck-It."""
    
    # Create logger
    logger = std_logging.getLogger("speckit")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = std_logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    json_formatter = JsonFormatter()
    
    # Console handler
    console_handler = std_logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = std_logging.FileHandler(log_file)
        file_handler.setLevel(std_logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    logger.info("Speck-It logging initialized")


class JsonFormatter(std_logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: std_logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class PerformanceMonitor:
    """Monitor performance metrics for Speck-It operations."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": name,
            "value": value,
            "tags": tags or {}
        }
        
        # Store in memory (in production, this would go to a monitoring system)
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)
        
        # Log the metric
        logger = std_logging.getLogger("speckit.performance")
        logger.info(f"Metric recorded: {name}={value}", extra={"extra_fields": metric})
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get recorded metrics."""
        if name:
            return {name: self.metrics.get(name, [])}
        return self.metrics.copy()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def log_performance(operation_name: str):
    """Decorator to log performance metrics for operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = std_logging.getLogger("speckit.performance")
            
            try:
                # Log operation start
                logger.info(f"Starting operation: {operation_name}")
                
                # Execute the operation
                result = func(*args, **kwargs)
                
                # Calculate and log performance
                duration = time.time() - start_time
                performance_monitor.record_metric(
                    f"{operation_name}_duration",
                    duration,
                    {"status": "success"}
                )
                
                logger.info(
                    f"Completed operation: {operation_name} in {duration:.3f}s",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration": duration,
                        "status": "success"
                    }}
                )
                
                return result
                
            except Exception as e:
                # Calculate and log performance even for failed operations
                duration = time.time() - start_time
                performance_monitor.record_metric(
                    f"{operation_name}_duration",
                    duration,
                    {"status": "error", "error_type": type(e).__name__}
                )
                
                logger.error(
                    f"Failed operation: {operation_name} after {duration:.3f}s - {str(e)}",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration": duration,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }},
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


@contextmanager
def log_operation(operation_name: str, **extra_fields):
    """Context manager to log operations with custom fields."""
    logger = std_logging.getLogger("speckit.operations")
    start_time = time.time()
    
    # Log operation start
    logger.info(f"Starting operation: {operation_name}", extra={"extra_fields": {
        "operation": operation_name,
        "status": "started",
        **extra_fields
    }})
    
    try:
        yield
        
        # Log operation success
        duration = time.time() - start_time
        logger.info(f"Completed operation: {operation_name} in {duration:.3f}s", extra={"extra_fields": {
            "operation": operation_name,
            "status": "completed",
            "duration": duration,
            **extra_fields
        }})
        
    except Exception as e:
        # Log operation failure
        duration = time.time() - start_time
        logger.error(f"Failed operation: {operation_name} after {duration:.3f}s - {str(e)}", extra={"extra_fields": {
            "operation": operation_name,
            "status": "failed",
            "duration": duration,
            "error_type": type(e).__name__,
            "error_message": str(e),
            **extra_fields
        }}, exc_info=True)
        
        raise


class ObservabilityHooks:
    """Observability hooks for Speck-It workflow events."""
    
    def __init__(self):
        self.hooks: Dict[str, List[callable]] = {}
        self.logger = std_logging.getLogger("speckit.observability")
    
    def register_hook(self, event_type: str, callback: callable) -> None:
        """Register a callback for a specific event type."""
        if event_type not in self.hooks:
            self.hooks[event_type] = []
        self.hooks[event_type].append(callback)
        self.logger.debug(f"Registered hook for event: {event_type}")
    
    def trigger_hooks(self, event_type: str, **data) -> None:
        """Trigger all callbacks for a specific event type."""
        if event_type in self.hooks:
            self.logger.debug(f"Triggering {len(self.hooks[event_type])} hooks for event: {event_type}")
            for hook in self.hooks[event_type]:
                try:
                    hook(**data)
                except Exception as e:
                    self.logger.error(f"Hook failed for event {event_type}: {e}")
    
    def log_workflow_event(self, event_type: str, feature_id: Optional[str] = None, **data) -> None:
        """Log a workflow event and trigger hooks."""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "feature_id": feature_id,
            **data
        }
        
        # Log the event
        self.logger.info(f"Workflow event: {event_type}", extra={"extra_fields": event_data})
        
        # Trigger hooks with event_data but without event_type to avoid duplicate argument
        hook_data = {k: v for k, v in event_data.items() if k != "event_type"}
        self.trigger_hooks(event_type, **hook_data)


# Global observability hooks instance
observability_hooks = ObservabilityHooks()


def log_workflow_step(step_name: str, feature_id: Optional[str] = None, **extra_fields):
    """Log a workflow step with observability."""
    observability_hooks.log_workflow_event(
        f"workflow_step_{step_name.lower()}",
        feature_id=feature_id,
        step_name=step_name,
        **extra_fields
    )


def log_artifact_event(event_type: str, artifact_type: str, feature_id: str, **extra_fields):
    """Log an artifact-related event."""
    observability_hooks.log_workflow_event(
        f"artifact_{event_type.lower()}",
        feature_id=feature_id,
        artifact_type=artifact_type,
        **extra_fields
    )


def log_error_with_context(error: Exception, context: Dict[str, Any], **extra_fields):
    """Log an error with rich context information."""
    logger = std_logging.getLogger("speckit.errors")
    
    error_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        **extra_fields
    }
    
    logger.error(
        f"Error in {context.get('operation', 'unknown operation')}: {str(error)}",
        extra={"extra_fields": error_data},
        exc_info=True
    )


# Convenience functions for common operations
def log_spec_generation(feature_id: str, feature_name: str, **extra_fields):
    """Log specification generation event."""
    log_artifact_event("generated", "spec", feature_id, feature_name=feature_name, **extra_fields)


def log_plan_generation(feature_id: str, **extra_fields):
    """Log plan generation event."""
    log_artifact_event("generated", "plan", feature_id, **extra_fields)


def log_task_generation(feature_id: str, task_count: int, **extra_fields):
    """Log task generation event."""
    log_artifact_event("generated", "tasks", feature_id, task_count=task_count, **extra_fields)


def log_task_update(feature_id: str, task_id: str, completed: bool, **extra_fields):
    """Log task update event."""
    log_artifact_event("updated", "task", feature_id, task_id=task_id, completed=completed, **extra_fields)


def log_feature_finalization(feature_id: str, **extra_fields):
    """Log feature finalization event."""
    log_artifact_event("finalized", "feature", feature_id, **extra_fields)


# Initialize default logging when module is imported
# This is done lazily to avoid circular import issues
_default_logging_initialized = False

def _initialize_default_logging():
    """Initialize default logging if not already done."""
    global _default_logging_initialized
    if not _default_logging_initialized:
        setup_logging()
        _default_logging_initialized = True