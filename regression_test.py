#!/usr/bin/env python3
"""
Speck-It Final Regression Test and Metrics Collection

This script performs regression testing and collects performance metrics
for the Speck-It MCP Server without importing the modules directly.
"""

import sys
import os
import json
import time
import subprocess
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse


class FinalRegressionTester:
    """Final regression testing and metrics collection for Speck-It."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize regression tester."""
        self.output_dir = output_dir or Path("regression_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "tests": [],
            "metrics": {},
            "summary": {}
        }
        
        # Setup basic logging
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("speckit.regression")
        
        # Performance metrics collection
        self.performance_data = []
        
    def run_mcp_command_with_metrics(self, command: str, params: Dict[str, Any], test_name: str) -> Dict[str, Any]:
        """Run an MCP command and collect performance metrics."""
        self.logger.info(f"Running MCP command: {command}")
        
        # Record start time
        start_time = time.time()
        
        # Create input for the MCP server
        input_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": command,
            "params": params
        }
        
        # Run the command
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )
        
        stdout, stderr = process.communicate(input=json.dumps(input_data))
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse response
        try:
            result = json.loads(stdout)
            success = process.returncode == 0 and "error" not in result
        except json.JSONDecodeError:
            result = {"error": f"Failed to parse response: {stdout}"}
            success = False
        
        # Collect performance metrics
        metrics = {
            "command": command,
            "params": params,
            "duration": duration,
            "success": success,
            "return_code": process.returncode,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr),
            "memory_usage": self._get_memory_usage()
        }
        
        # Record performance data
        self.performance_data.append(metrics)
        
        # Log result
        if success:
            self.logger.info(f"[PASS] {command} completed in {duration:.3f}s")
        else:
            self.logger.error(f"[FAIL] {command} failed in {duration:.3f}s")
            if "error" in result:
                self.logger.error(f"Error: {result['error']}")
        
        return {
            "result": result,
            "metrics": metrics,
            "success": success
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_complete_workflow(self) -> Dict[str, Any]:
        """Test the complete Speck-It workflow."""
        self.logger.info("Testing complete workflow...")
        
        workflow_results = {
            "test_name": "complete_workflow",
            "steps": [],
            "start_time": time.time()
        }
        
        try:
            # Step 1: Set constitution
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "set_constitution",
                {
                    "content": "# Test Constitution\n\nThis is a test constitution for regression testing.",
                    "mode": "replace"
                },
                "set_constitution"
            )
            workflow_results["steps"].append({
                "step": "set_constitution",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to set constitution")
            
            # Step 2: Register feature root
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "set_feature_root",
                {"feature_id": "regression-test"},
                "set_feature_root"
            )
            workflow_results["steps"].append({
                "step": "set_feature_root",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to register feature root")
            
            # Step 3: Generate spec
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "generate_spec",
                {
                    "feature_name": "Regression Test Feature",
                    "description": "A feature for regression testing the complete workflow",
                    "feature_id": "regression-test"
                },
                "generate_spec"
            )
            workflow_results["steps"].append({
                "step": "generate_spec",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to generate specification")
            
            feature_id = result["result"]["artifacts"]["feature_id"]
            
            # Step 4: Generate plan
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "generate_plan",
                {"feature_id": feature_id},
                "generate_plan"
            )
            workflow_results["steps"].append({
                "step": "generate_plan",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to generate plan")
            
            # Step 5: Generate tasks
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "generate_tasks",
                {"feature_id": feature_id},
                "generate_tasks"
            )
            workflow_results["steps"].append({
                "step": "generate_tasks",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to generate tasks")
            
            # Step 6: List tasks
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "list_tasks",
                {"feature_id": feature_id},
                "list_tasks"
            )
            workflow_results["steps"].append({
                "step": "list_tasks",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to list tasks")
            
            tasks = result["result"]["tasks"]
            
            # Step 7: Complete all tasks
            for i, task in enumerate(tasks, 1):
                step_start = time.time()
                result = self.run_mcp_command_with_metrics(
                    "complete_task",
                    {
                        "feature_id": feature_id,
                        "task_id": task["task_id"]
                    },
                    f"complete_task_{i}"
                )
                workflow_results["steps"].append({
                    "step": f"complete_task_{i}",
                    "duration": time.time() - step_start,
                    "success": result["success"]
                })
                
                if not result["success"]:
                    raise ValueError(f"Failed to complete task {task['task_id']}")
            
            # Step 8: Finalize feature
            step_start = time.time()
            result = self.run_mcp_command_with_metrics(
                "finalize_feature",
                {"feature_id": feature_id},
                "finalize_feature"
            )
            workflow_results["steps"].append({
                "step": "finalize_feature",
                "duration": time.time() - step_start,
                "success": result["success"]
            })
            
            if not result["success"]:
                raise ValueError("Failed to finalize feature")
            
            workflow_results["end_time"] = time.time()
            workflow_results["total_duration"] = workflow_results["end_time"] - workflow_results["start_time"]
            workflow_results["success"] = True
            
            self.logger.info(f"[PASS] Complete workflow test passed in {workflow_results['total_duration']:.3f}s")
            
        except Exception as e:
            workflow_results["end_time"] = time.time()
            workflow_results["total_duration"] = workflow_results["end_time"] - workflow_results["start_time"]
            workflow_results["success"] = False
            workflow_results["error"] = str(e)
            
            self.logger.error(f"[FAIL] Complete workflow test failed: {e}")
        
        return workflow_results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks."""
        self.logger.info("Running performance benchmarks...")
        
        benchmark_results = {
            "test_name": "performance_benchmarks",
            "benchmarks": [],
            "start_time": time.time()
        }
        
        # Test specification generation performance
        spec_times = []
        for i in range(3):
            start_time = time.time()
            result = self.run_mcp_command_with_metrics(
                "set_constitution",
                {
                    "content": f"# Test Constitution {i}\n\nThis is test constitution {i} for benchmarking.",
                    "mode": "replace"
                },
                f"set_constitution_benchmark_{i}"
            )
            spec_times.append(result["metrics"]["duration"])
        
        benchmark_results["benchmarks"].append({
            "operation": "set_constitution",
            "iterations": 3,
            "times": spec_times,
            "average": statistics.mean(spec_times),
            "min": min(spec_times),
            "max": max(spec_times),
            "std_dev": statistics.stdev(spec_times) if len(spec_times) > 1 else 0
        })
        
        # Test plan generation performance
        plan_times = []
        for i in range(3):
            # First set up a feature
            self.run_mcp_command_with_metrics(
                "set_constitution",
                {
                    "content": f"# Test Constitution {i}\n\nThis is test constitution {i} for benchmarking.",
                    "mode": "replace"
                },
                f"set_constitution_benchmark_plan_{i}"
            )
            
            self.run_mcp_command_with_metrics(
                "set_feature_root",
                {"feature_id": f"benchmark-plan-{i}"},
                f"set_feature_root_benchmark_plan_{i}"
            )
            
            self.run_mcp_command_with_metrics(
                "generate_spec",
                {
                    "feature_name": f"Benchmark Feature {i}",
                    "description": f"A feature for benchmarking plan generation {i}",
                    "feature_id": f"benchmark-plan-{i}"
                },
                f"generate_spec_benchmark_plan_{i}"
            )
            
            start_time = time.time()
            result = self.run_mcp_command_with_metrics(
                "generate_plan",
                {"feature_id": f"benchmark-plan-{i}"},
                f"generate_plan_benchmark_{i}"
            )
            plan_times.append(result["metrics"]["duration"])
        
        benchmark_results["benchmarks"].append({
            "operation": "generate_plan",
            "iterations": 3,
            "times": plan_times,
            "average": statistics.mean(plan_times),
            "min": min(plan_times),
            "max": max(plan_times),
            "std_dev": statistics.stdev(plan_times) if len(plan_times) > 1 else 0
        })
        
        benchmark_results["end_time"] = time.time()
        benchmark_results["total_duration"] = benchmark_results["end_time"] - benchmark_results["start_time"]
        benchmark_results["success"] = True
        
        self.logger.info(f"[PASS] Performance benchmarks completed in {benchmark_results['total_duration']:.3f}s")
        
        return benchmark_results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        self.logger.info("Testing error handling...")
        
        error_results = {
            "test_name": "error_handling",
            "tests": [],
            "start_time": time.time()
        }
        
        # Test invalid command
        start_time = time.time()
        result = self.run_mcp_command_with_metrics(
            "invalid_command",
            {},
            "invalid_command"
        )
        error_results["tests"].append({
            "test": "invalid_command",
            "duration": time.time() - start_time,
            "success": not result["success"],
            "expected_error": True
        })
        
        # Test missing required parameters
        start_time = time.time()
        result = self.run_mcp_command_with_metrics(
            "generate_spec",
            {},  # Missing required parameters
            "missing_params"
        )
        error_results["tests"].append({
            "test": "missing_params",
            "duration": time.time() - start_time,
            "success": not result["success"],
            "expected_error": True
        })
        
        # Test workflow order enforcement
        start_time = time.time()
        result = self.run_mcp_command_with_metrics(
            "generate_plan",
            {"feature_id": "non-existent-feature"},
            "workflow_order_enforcement"
        )
        error_results["tests"].append({
            "test": "workflow_order_enforcement",
            "duration": time.time() - start_time,
            "success": not result["success"],
            "expected_error": True
        })
        
        error_results["end_time"] = time.time()
        error_results["total_duration"] = error_results["end_time"] - error_results["start_time"]
        error_results["success"] = True
        
        self.logger.info(f"[PASS] Error handling tests completed in {error_results['total_duration']:.3f}s")
        
        return error_results
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system and performance metrics."""
        self.logger.info("Collecting system metrics...")
        
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
        
        # Collect memory metrics
        memory_metrics = self._get_memory_usage()
        if "error" not in memory_metrics:
            metrics["memory"] = memory_metrics
        
        # Collect disk usage metrics
        try:
            import shutil
            disk_usage = shutil.disk_usage(Path(__file__).parent)
            metrics["disk"] = {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free
            }
        except Exception as e:
            metrics["disk"] = {"error": str(e)}
        
        # Calculate performance statistics
        if self.performance_data:
            command_metrics = {}
            for metric in self.performance_data:
                command = metric["command"]
                if command not in command_metrics:
                    command_metrics[command] = []
                command_metrics[command].append(metric["duration"])
            
            for command, durations in command_metrics.items():
                if durations:
                    metrics[f"{command}_performance"] = {
                        "count": len(durations),
                        "average": statistics.mean(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
                    }
        
        self.logger.info("[PASS] System metrics collected")
        
        return metrics
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run all regression tests."""
        self.logger.info("Starting regression tests...")
        
        start_time = time.time()
        
        # Run complete workflow test
        workflow_result = self.test_complete_workflow()
        self.results["tests"].append(workflow_result)
        
        # Run performance benchmarks
        benchmark_result = self.test_performance_benchmarks()
        self.results["tests"].append(benchmark_result)
        
        # Run error handling tests
        error_result = self.test_error_handling()
        self.results["tests"].append(error_result)
        
        # Collect system metrics
        metrics_result = self.collect_system_metrics()
        self.results["metrics"] = metrics_result
        
        # Calculate summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        passed_tests = sum(1 for test in self.results["tests"] if test.get("success", False))
        total_tests = len(self.results["tests"])
        
        self.results["summary"] = {
            "total_duration": total_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Regression tests completed in {total_duration:.3f}s")
        self.logger.info(f"Results: {passed_tests}/{total_tests} tests passed ({self.results['summary']['success_rate']:.1%})")
        
        return self.results
    
    def save_results(self) -> Path:
        """Save regression test results to file."""
        results_file = self.output_dir / f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        return results_file
    
    def generate_report(self) -> Path:
        """Generate a human-readable regression report."""
        report_file = self.output_dir / f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Speck-It Regression Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Python Version: {self.results['python_version']}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = self.results["summary"]
            f.write(f"Total Duration: {summary.get('total_duration', 0):.3f}s\n")
            f.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
            f.write(f"Passed Tests: {summary.get('passed_tests', 0)}\n")
            f.write(f"Failed Tests: {summary.get('failed_tests', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0):.1%}\n\n")
            
            # Test Results
            f.write("TEST RESULTS\n")
            f.write("-" * 20 + "\n")
            for test in self.results["tests"]:
                f.write(f"{test['test_name']}\n")
                f.write(f"  Success: {'PASS' if test.get('success') else 'FAIL'}\n")
                if 'total_duration' in test:
                    f.write(f"  Duration: {test['total_duration']:.3f}s\n")
                
                if test['test_name'] == 'complete_workflow':
                    f.write(f"  Steps: {len(test.get('steps', []))}\n")
                    for step in test.get('steps', []):
                        status = 'PASS' if step.get('success') else 'FAIL'
                        f.write(f"    {step['step']}: {status} ({step['duration']:.3f}s)\n")
                
                f.write("\n")
            
            # Performance Metrics
            for key, value in self.results["metrics"].items():
                if key.endswith("_performance"):
                    operation_name = key.replace("_performance", "")
                    f.write(f"{operation_name.upper()} PERFORMANCE\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Count: {value['count']}\n")
                    f.write(f"Average: {value['average']:.3f}s\n")
                    f.write(f"Min: {value['min']:.3f}s\n")
                    f.write(f"Max: {value['max']:.3f}s\n")
                    f.write(f"Std Dev: {value['std_dev']:.3f}s\n")
                    f.write("\n")
            
            # System Metrics
            if "memory" in self.results["metrics"]:
                f.write("SYSTEM METRICS\n")
                f.write("-" * 20 + "\n")
                memory = self.results["metrics"]["memory"]
                f.write(f"RSS: {memory['rss'] / 1024 / 1024:.1f} MB\n")
                f.write(f"VMS: {memory['vms'] / 1024 / 1024:.1f} MB\n")
                f.write(f"Percent: {memory['percent']:.1f}%\n\n")
            
            if "disk" in self.results["metrics"]:
                disk = self.results["metrics"]["disk"]
                if "error" not in disk:
                    f.write("DISK USAGE\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Total: {disk['total'] / 1024 / 1024 / 1024:.1f} GB\n")
                    f.write(f"Used: {disk['used'] / 1024 / 1024 / 1024:.1f} GB\n")
                    f.write(f"Free: {disk['free'] / 1024 / 1024 / 1024:.1f} GB\n\n")
        
        self.logger.info(f"Report generated: {report_file}")
        return report_file


def main():
    """Main entry point for regression testing."""
    parser = argparse.ArgumentParser(description="Speck-It Regression Testing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regression_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick regression tests only"
    )
    
    args = parser.parse_args()
    
    # Create regression tester
    tester = FinalRegressionTester(args.output_dir)
    
    # Run regression tests
    if args.quick:
        # Run only the complete workflow test
        result = tester.test_complete_workflow()
        tester.results["tests"].append(result)
        tester.results["summary"] = {
            "total_tests": 1,
            "passed_tests": 1 if result.get("success") else 0,
            "failed_tests": 0 if result.get("success") else 1,
            "success_rate": 1 if result.get("success") else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        # Run full regression test suite
        tester.run_regression_tests()
    
    # Save results
    tester.save_results()
    
    # Generate report
    tester.generate_report()
    
    # Exit with appropriate code
    success = tester.results["summary"]["success_rate"] == 1.0
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()