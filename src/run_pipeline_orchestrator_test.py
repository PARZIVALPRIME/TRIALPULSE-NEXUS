# src/run_pipeline_orchestrator_test.py
"""
TRIALPULSE NEXUS 10X - Pipeline Orchestrator Test
Phase 11.1: Pipeline Orchestration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta


def run_tests():
    """Run all pipeline orchestrator tests"""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 11.1 PIPELINE ORCHESTRATOR TEST")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    # Reset for clean test
    try:
        from src.orchestration.pipeline_orchestrator import reset_pipeline_orchestrator
        reset_pipeline_orchestrator()
    except:
        pass
    
    # =========================================================================
    # TEST 1: Initialize Orchestrator
    # =========================================================================
    print("-" * 70)
    print("TEST 1: Initialize Pipeline Orchestrator")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import (
            get_pipeline_orchestrator,
            PipelineType,
            PipelineStatus,
            TaskStatus
        )
        
        orchestrator = get_pipeline_orchestrator()
        
        print(f"   ✅ Orchestrator initialized")
        print(f"   Registered pipelines: {len(orchestrator.pipelines)}")
        print(f"   Registered tasks: {len(orchestrator.task_registry.tasks)}")
        
        for pid, pipeline in orchestrator.pipelines.items():
            print(f"   - {pid}: {pipeline.pipeline_name} ({len(pipeline.tasks)} tasks)")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 2: Run Data Refresh Pipeline
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 2: Run Data Refresh Pipeline")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import run_data_refresh
        
        run = run_data_refresh(triggered_by="test")
        
        print(f"   ✅ Data refresh pipeline completed")
        print(f"   Run ID: {run.run_id}")
        print(f"   Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        print(f"   Tasks: {run.tasks_completed}/{run.tasks_total} completed")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 3: Run Model Inference Pipeline
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 3: Run Model Inference Pipeline (Parallel)")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import run_model_inference
        
        run = run_model_inference(triggered_by="test")
        
        print(f"   ✅ Model inference pipeline completed")
        print(f"   Run ID: {run.run_id}")
        print(f"   Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        print(f"   Tasks: {run.tasks_completed}/{run.tasks_total} completed")
        print(f"   Success Rate: {run.success_rate:.1%}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 4: Run Agent Tasks Pipeline
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 4: Run Agent Tasks Pipeline")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import run_agent_tasks
        
        run = run_agent_tasks(triggered_by="test")
        
        print(f"   ✅ Agent tasks pipeline completed")
        print(f"   Run ID: {run.run_id}")
        print(f"   Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        print(f"   Tasks: {run.tasks_completed}/{run.tasks_total} completed")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 5: Run Dashboard Update Pipeline
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 5: Run Dashboard Update Pipeline")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import run_dashboard_update
        
        run = run_dashboard_update(triggered_by="test")
        
        print(f"   ✅ Dashboard update pipeline completed")
        print(f"   Run ID: {run.run_id}")
        print(f"   Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        print(f"   Tasks: {run.tasks_completed}/{run.tasks_total} completed")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 6: Run Full Pipeline
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 6: Run Full Pipeline (All Components)")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import run_full_pipeline
        
        run = run_full_pipeline(triggered_by="test")
        
        print(f"   ✅ Full pipeline completed")
        print(f"   Run ID: {run.run_id}")
        print(f"   Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        print(f"   Tasks: {run.tasks_completed}/{run.tasks_total} completed")
        print(f"   Failed: {run.tasks_failed}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 7: Pipeline Scheduling
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 7: Pipeline Scheduling")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import (
            get_pipeline_orchestrator, ScheduleFrequency
        )
        
        orchestrator = get_pipeline_orchestrator()
        
        # Schedule data refresh daily
        schedule = orchestrator.schedule_pipeline(
            "PIPE-DATA-REFRESH",
            ScheduleFrequency.DAILY,
            created_by="test"
        )
        
        print(f"   ✅ Pipeline scheduled")
        print(f"   Schedule ID: {schedule.schedule_id}")
        print(f"   Frequency: {schedule.frequency.value}")
        print(f"   Next Run: {schedule.next_run}")
        print(f"   Enabled: {schedule.enabled}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 8: Get Recent Runs
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 8: Get Recent Runs")
    print("-" * 70)
    
    try:
        orchestrator = get_pipeline_orchestrator()
        
        recent = orchestrator.get_recent_runs(limit=5)
        
        print(f"   ✅ Retrieved {len(recent)} recent runs")
        for run in recent:
            print(f"   - {run.pipeline_name}: {run.status.value} ({run.duration_seconds:.2f}s)")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 9: Get Pipeline Status
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 9: Get Pipeline Status")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import get_pipeline_status
        
        status = get_pipeline_status("PIPE-DATA-REFRESH")
        
        print(f"   ✅ Pipeline status retrieved")
        print(f"   Pipeline: {status['pipeline_name']}")
        print(f"   Type: {status['pipeline_type']}")
        print(f"   Tasks: {status['task_count']}")
        print(f"   Recent Runs: {len(status['recent_runs'])}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 10: Get Statistics
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 10: Get Orchestration Statistics")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import get_pipeline_stats
        
        stats = get_pipeline_stats()
        
        print(f"   ✅ Statistics retrieved")
        print(f"   Total Runs: {stats['total_runs']}")
        print(f"   Registered Pipelines: {stats['registered_pipelines']}")
        print(f"   Scheduled Pipelines: {stats['scheduled_pipelines']}")
        print(f"   Active Runs: {stats['active_runs']}")
        print(f"   Average Duration: {stats['average_duration_seconds']:.2f}s")
        print(f"   Recent Success Rate: {stats['recent_success_rate']:.1f}%")
        
        if stats.get('by_status'):
            print(f"   By Status: {stats['by_status']}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
        # =========================================================================
    # TEST 11: Health Status
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 11: Get Health Status")
    print("-" * 70)
    
    try:
        orchestrator = get_pipeline_orchestrator()
        
        health = orchestrator.get_health_status()
        
        print(f"   ✅ Health status retrieved")
        print(f"   Status: {health['status']}")
        print(f"   Active Runs: {health['active_runs']}")
        print(f"   Recent Failures: {health['recent_failures']}")
        print(f"   Executor Active: {health['executor_active']}")
        print(f"   Registered Tasks: {health['registered_tasks']}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 12: Task Registry
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 12: Task Registry")
    print("-" * 70)
    
    try:
        orchestrator = get_pipeline_orchestrator()
        registry = orchestrator.task_registry
        
        # List all registered tasks
        task_names = list(registry.tasks.keys())
        
        print(f"   ✅ Task registry has {len(task_names)} tasks")
        print(f"   Tasks: {', '.join(task_names[:10])}...")
        
        # Execute a task directly
        result = registry.execute("data_ingestion", task_id="TEST-001")
        
        print(f"   Direct execution: {result.success}")
        print(f"   Result: {result.result}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 13: Custom Pipeline Creation
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 13: Custom Pipeline Creation")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import (
            Pipeline, PipelineTask, PipelineType, TaskPriority
        )
        
        orchestrator = get_pipeline_orchestrator()
        
        # Create custom pipeline
        custom_pipeline = Pipeline(
            pipeline_id="PIPE-CUSTOM-001",
            pipeline_name="Custom Test Pipeline",
            pipeline_type=PipelineType.CUSTOM,
            description="A custom test pipeline",
            tasks=[
                PipelineTask(
                    task_id="PIPE-CUSTOM-001-T01",
                    task_name="Custom Task 1",
                    task_type="custom",
                    pipeline_id="PIPE-CUSTOM-001",
                    function_name="cache_invalidation",
                    function_module="orchestration",
                    order=1,
                    priority=TaskPriority.HIGH
                ),
                PipelineTask(
                    task_id="PIPE-CUSTOM-001-T02",
                    task_name="Custom Task 2",
                    task_type="custom",
                    pipeline_id="PIPE-CUSTOM-001",
                    function_name="alert_processing",
                    function_module="orchestration",
                    depends_on=["PIPE-CUSTOM-001-T01"],
                    order=2
                )
            ],
            parallel_execution=False,
            tags=["custom", "test"]
        )
        
        # Register it
        orchestrator.pipelines[custom_pipeline.pipeline_id] = custom_pipeline
        
        # Run it
        run = orchestrator.run_pipeline(
            "PIPE-CUSTOM-001",
            triggered_by="test",
            trigger_reason="Testing custom pipeline"
        )
        
        print(f"   ✅ Custom pipeline created and executed")
        print(f"   Pipeline ID: {custom_pipeline.pipeline_id}")
        print(f"   Tasks: {len(custom_pipeline.tasks)}")
        print(f"   Run Status: {run.status.value}")
        print(f"   Duration: {run.duration_seconds:.2f}s")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 14: Pipeline with Dependencies
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 14: Pipeline with Task Dependencies")
    print("-" * 70)
    
    try:
        # The full pipeline has complex dependencies
        orchestrator = get_pipeline_orchestrator()
        pipeline = orchestrator.get_pipeline("PIPE-FULL")
        
        # Count dependencies
        tasks_with_deps = sum(1 for t in pipeline.tasks if t.depends_on)
        total_deps = sum(len(t.depends_on) for t in pipeline.tasks)
        
        print(f"   ✅ Pipeline dependency analysis")
        print(f"   Pipeline: {pipeline.pipeline_name}")
        print(f"   Total Tasks: {len(pipeline.tasks)}")
        print(f"   Tasks with Dependencies: {tasks_with_deps}")
        print(f"   Total Dependencies: {total_deps}")
        print(f"   Parallel Execution: {pipeline.parallel_execution}")
        print(f"   Max Parallel: {pipeline.max_parallel_tasks}")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # TEST 15: Convenience Functions
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 15: Convenience Functions")
    print("-" * 70)
    
    try:
        from src.orchestration.pipeline_orchestrator import (
            run_data_refresh,
            run_model_inference,
            run_agent_tasks,
            run_dashboard_update,
            run_full_pipeline,
            get_pipeline_status,
            get_pipeline_stats
        )
        
        # All functions should be importable
        print(f"   ✅ All convenience functions imported")
        print(f"   - run_data_refresh")
        print(f"   - run_model_inference")
        print(f"   - run_agent_tasks")
        print(f"   - run_dashboard_update")
        print(f"   - run_full_pipeline")
        print(f"   - get_pipeline_status")
        print(f"   - get_pipeline_stats")
        
        passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Total: {passed + failed}")
    print()
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)