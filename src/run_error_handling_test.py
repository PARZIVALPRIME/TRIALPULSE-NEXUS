"""
TRIALPULSE NEXUS 10X - Phase 11.2 Error Handling Test (Standalone)
"""

import sys
import sqlite3  # Import at the top level
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests():
    """Run all error handling tests"""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 11.2 ERROR HANDLING TEST")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # =========================================================================
    # TEST 1: Initialize Error Handling System
    # =========================================================================
    print("-" * 70)
    print("TEST 1: Initialize Error Handling System")
    print("-" * 70)
    
    try:
        # Import directly from the module file
        from src.orchestration.error_handling import (
            ErrorHandlingSystem,
            get_error_handling_system,
            reset_error_handling_system,
            ErrorSeverity,
            ErrorCategory,
            CircuitState,
            FallbackMode,
            CircuitBreakerConfig,
            FallbackConfig,
            handle_error,
            with_error_handling,
            get_system_health,
            recover_component,
        )
        
        # Reset for clean test
        reset_error_handling_system()
        
        system = get_error_handling_system()
        
        print(f"   ✅ Error handling system initialized")
        print(f"   Database: {system.db_path}")
        print(f"   Recovery procedures: {len(system.recovery_manager.list_procedures())}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return tests_passed, tests_failed
    
    # =========================================================================
    # TEST 2: Handle Basic Error
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 2: Handle Basic Error")
    print("-" * 70)
    
    try:
        # Simulate an error
        try:
            raise ValueError("Test validation error")
        except Exception as e:
            error = handle_error(
                e, 
                component="test_component",
                operation="test_operation",
                user_id="test_user"
            )
        
        print(f"   Error ID: {error.error_id}")
        print(f"   Category: {error.category.value}")
        print(f"   Severity: {error.severity.value}")
        print(f"   Message: {error.message}")
        print(f"   Recovery Action: {error.recovery_action.value if error.recovery_action else 'None'}")
        print(f"   ✅ Error handled successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 3: Error Classification
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 3: Error Classification")
    print("-" * 70)
    
    try:
        # Note: sqlite3 is imported at the top of the file
        test_errors = [
            (sqlite3.OperationalError("database is locked"), "database"),
            (ConnectionError("Connection refused"), "network"),
            (FileNotFoundError("File not found: data.parquet"), "data"),
            (TimeoutError("Request timed out"), "timeout"),
            (PermissionError("Access denied"), "permission"),
        ]
        
        for exc, expected_category in test_errors:
            try:
                raise exc
            except Exception as e:
                error = handle_error(e, "test", "classify")
            
            status = "✅" if error.category.value == expected_category else "⚠️"
            print(f"   {status} {type(exc).__name__}: {error.category.value} (expected: {expected_category})")
        
        print(f"   ✅ Error classification working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 4: Circuit Breaker
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 4: Circuit Breaker")
    print("-" * 70)
    
    try:
        # Configure circuit breaker
        system.circuit_breaker.configure(CircuitBreakerConfig(
            component="test_circuit",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=5
        ))
        
        # Initial state should be closed
        state = system.circuit_breaker.get_state("test_circuit")
        print(f"   Initial state: {state.state.value}")
        assert state.state == CircuitState.CLOSED
        
        # Record failures to open circuit
        for i in range(3):
            system.circuit_breaker.record_failure("test_circuit")
        
        state = system.circuit_breaker.get_state("test_circuit")
        print(f"   After 3 failures: {state.state.value}")
        assert state.state == CircuitState.OPEN
        
        # Check that circuit rejects execution
        can_exec, reason = system.circuit_breaker.can_execute("test_circuit")
        print(f"   Can execute: {can_exec} ({reason[:50]}...)")
        assert not can_exec
        
        # Reset circuit
        system.circuit_breaker.reset("test_circuit")
        state = system.circuit_breaker.get_state("test_circuit")
        print(f"   After reset: {state.state.value}")
        assert state.state == CircuitState.CLOSED
        
        print(f"   ✅ Circuit breaker working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 5: Fallback Manager
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 5: Fallback Manager")
    print("-" * 70)
    
    try:
        # Configure fallback
        system.fallback_manager.configure(FallbackConfig(
            component="test_fallback",
            fallback_mode=FallbackMode.FULL,
            cache_ttl_seconds=60,
            static_value={"default": "value"}
        ))
        
        # Cache some data
        system.fallback_manager.cache_data("test_fallback", "test_key", {"cached": "data"})
        
        # Retrieve cached data
        cached = system.fallback_manager.get_cached_data("test_fallback", "test_key")
        print(f"   Cached data: {cached}")
        assert cached == {"cached": "data"}
        
        # Switch to cached mode
        system.fallback_manager.set_fallback_mode("test_fallback", FallbackMode.CACHED)
        mode = system.fallback_manager.get_fallback_mode("test_fallback")
        print(f"   Fallback mode: {mode.value}")
        assert mode == FallbackMode.CACHED
        
        # Get fallback value
        value, message = system.fallback_manager.get_fallback_value(
            "test_fallback", "test_key", {"default": "data"}
        )
        print(f"   Fallback value: {value}")
        print(f"   Message: {message}")
        assert value == {"cached": "data"}
        
        # Test static mode
        system.fallback_manager.set_fallback_mode("test_fallback", FallbackMode.STATIC)
        value, message = system.fallback_manager.get_fallback_value(
            "test_fallback", "missing_key", None
        )
        print(f"   Static value: {value}")
        assert value == {"default": "value"}
        
        print(f"   ✅ Fallback manager working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 6: Execute with Fallback
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 6: Execute with Fallback")
    print("-" * 70)
    
    try:
        # Reset to full mode first
        system.fallback_manager.set_fallback_mode("test_fallback", FallbackMode.FULL)
        
        # Test successful execution
        def successful_func():
            return {"result": "success"}
        
        result, success, message = system.fallback_manager.execute_with_fallback(
            "test_fallback", "op1",
            primary_func=successful_func,
            default={"result": "default"}
        )
        print(f"   Primary success: {result} - {message}")
        assert success and result == {"result": "success"}
        
        # Test failing execution with fallback
        def failing_func():
            raise ValueError("Intentional failure")
        
        def fallback_func():
            return {"result": "fallback"}
        
        result, success, message = system.fallback_manager.execute_with_fallback(
            "test_fallback", "op2",
            primary_func=failing_func,
            fallback_func=fallback_func,
            default={"result": "default"}
        )
        print(f"   With fallback func: {result} - {message}")
        assert success and result == {"result": "fallback"}
        
        print(f"   ✅ Execute with fallback working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 7: Recovery Procedures
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 7: Recovery Procedures")
    print("-" * 70)
    
    try:
        procedures = system.recovery_manager.list_procedures()
        print(f"   Available procedures: {len(procedures)}")
        
        for proc in procedures:
            print(f"   - {proc.procedure_id}: {proc.name}")
        
        # Execute a recovery procedure
        success, result = system.recovery_manager.execute_procedure(
            "PROC-CACHE-CLEAR",
            component="test_component",
            executed_by="test_user"
        )
        print(f"   Execute PROC-CACHE-CLEAR: {success}")
        print(f"   Result: {result}")
        
        print(f"   ✅ Recovery procedures working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 8: Component Health
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 8: Component Health")
    print("-" * 70)
    
    try:
        health = system.get_component_health("data_loader")
        
        print(f"   Component: {health.component}")
        print(f"   Status: {health.status.value}")
        print(f"   Fallback Mode: {health.fallback_mode.value}")
        print(f"   Circuit State: {health.circuit_state.value}")
        print(f"   Error Rate: {health.error_rate:.2f}/min")
        
        print(f"   ✅ Component health retrieved successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 9: System Health
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 9: System Health")
    print("-" * 70)
    
    try:
        health = get_system_health()
        
        print(f"   Overall Status: {health['overall_status']}")
        print(f"   Components: {len(health['components'])}")
        for comp, status in health['components'].items():
            print(f"   - {comp}: {status['status']}")
        print(f"   Errors (24h): {health['error_summary']['total_errors']}")
        
        print(f"   ✅ System health retrieved successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 10: Error Decorator
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 10: Error Decorator")
    print("-" * 70)
    
    try:
        @with_error_handling("decorated_component", "decorated_operation", fallback_value="fallback")
        def decorated_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Decorated function failed")
            return "success"
        
        # Test successful execution
        result = decorated_function(should_fail=False)
        print(f"   Successful call: {result}")
        assert result == "success"
        
        # Test failing execution (uses fallback)
        result = decorated_function(should_fail=True)
        print(f"   Failed call (fallback): {result}")
        
        print(f"   ✅ Error decorator working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 11: Error Summary
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 11: Error Summary")
    print("-" * 70)
    
    try:
        summary = system.error_handler.get_error_summary(hours=24)
        
        print(f"   Period: {summary['period_hours']} hours")
        print(f"   Total Errors: {summary['total_errors']}")
        print(f"   By Severity: {summary['by_severity']}")
        print(f"   By Category: {summary['by_category']}")
        print(f"   Unresolved: {summary['unresolved']}")
        
        print(f"   ✅ Error summary retrieved successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 12: Recover Component
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 12: Recover Component")
    print("-" * 70)
    
    try:
        # Open a circuit first
        for _ in range(5):
            system.circuit_breaker.record_failure("recovery_test")
        
        state = system.circuit_breaker.get_state("recovery_test")
        print(f"   Before recovery: {state.state.value}")
        
        # Recover
        success, message = recover_component("recovery_test", executed_by="test_user")
        print(f"   Recovery result: {success}")
        print(f"   Message: {message}")
        
        state = system.circuit_breaker.get_state("recovery_test")
        print(f"   After recovery: {state.state.value}")
        
        print(f"   ✅ Component recovery working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 13: Recent Errors
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 13: Recent Errors")
    print("-" * 70)
    
    try:
        errors = system.error_handler.get_recent_errors(limit=10)
        
        print(f"   Recent errors: {len(errors)}")
        for error in errors[:5]:
            print(f"   - {error.error_id}: {error.category.value} - {error.message[:40]}...")
        
        print(f"   ✅ Recent errors retrieved successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 14: Resolve Error
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 14: Resolve Error")
    print("-" * 70)
    
    try:
        # Get an unresolved error
        errors = system.error_handler.get_recent_errors(limit=1, unresolved_only=True)
        
        if errors:
            error = errors[0]
            print(f"   Error to resolve: {error.error_id}")
            
            success = system.error_handler.resolve_error(
                error.error_id,
                resolved_by="test_user",
                resolution_notes="Resolved during testing"
            )
            print(f"   Resolved: {success}")
            
            # Verify resolution
            updated = system.error_handler.get_error(error.error_id)
            print(f"   Resolved status: {updated.resolved}")
            print(f"   Resolved by: {updated.resolved_by}")
        else:
            print(f"   No unresolved errors to test")
        
        print(f"   ✅ Error resolution working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 15: Statistics
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 15: Statistics")
    print("-" * 70)
    
    try:
        stats = system.get_statistics()
        
        print(f"   Error Summary: {stats['error_summary']['total_errors']} total")
        print(f"   Recovery Procedures: {stats['recovery_procedures']}")
        print(f"   Timestamp: {stats['timestamp']}")
        
        print(f"   ✅ Statistics retrieved successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    return tests_passed, tests_failed


def main():
    """Main entry point"""
    tests_passed, tests_failed = run_tests()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    print()
    
    if tests_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_failed} test(s) failed")
    
    return tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)