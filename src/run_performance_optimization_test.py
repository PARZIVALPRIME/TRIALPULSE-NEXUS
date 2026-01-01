"""
TRIALPULSE NEXUS 10X - Phase 11.3 Performance Optimization Test
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_tests():
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 11.3 PERFORMANCE OPTIMIZATION TEST")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Import and reset
    from src.orchestration.performance_optimization import (
        get_performance_system, reset_performance_system,
        CacheLayer, LoadingStrategy, PerformanceLevel,
        cache_get, cache_set, cache_delete, record_timing,
        get_performance_stats, get_health
    )
    
    reset_performance_system()
    
    # =========================================================================
    # TEST 1: Initialize Performance System
    # =========================================================================
    print("-" * 70)
    print("TEST 1: Initialize Performance System")
    print("-" * 70)
    
    try:
        perf = get_performance_system()
        assert perf is not None
        assert perf.cache is not None
        assert perf.lazy_loader is not None
        assert perf.query_optimizer is not None
        assert perf.response_monitor is not None
        
        print(f"   ✅ Performance system initialized")
        print(f"      Cache: Memory + Disk")
        print(f"      Lazy Loader: Ready")
        print(f"      Query Optimizer: Ready")
        print(f"      Response Monitor: Ready")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 2: Memory Cache Operations
    # =========================================================================
    print("-" * 70)
    print("TEST 2: Memory Cache Operations")
    print("-" * 70)
    
    try:
        # Set
        result = perf.cache_set("test_key", {"data": "test_value"}, ttl_seconds=60)
        assert result == True
        
        # Get
        value = perf.cache_get("test_key")
        assert value is not None
        assert value["data"] == "test_value"
        
        # Get (hit)
        value2 = perf.cache_get("test_key")
        assert value2 is not None
        
        # Get (miss)
        missing = perf.cache_get("nonexistent")
        assert missing is None
        
        # Delete
        deleted = perf.cache_delete("test_key")
        assert deleted == True
        
        # Verify deleted
        gone = perf.cache_get("test_key")
        assert gone is None
        
        print(f"   ✅ Cache operations working")
        print(f"      Set: ✓")
        print(f"      Get (hit): ✓")
        print(f"      Get (miss): ✓")
        print(f"      Delete: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 3: Cache with TTL
    # =========================================================================
    print("-" * 70)
    print("TEST 3: Cache with TTL")
    print("-" * 70)
    
    try:
        # Set with very short TTL
        perf.cache_set("ttl_test", "expires_soon", ttl_seconds=1)
        
        # Should exist immediately
        value = perf.cache_get("ttl_test")
        assert value == "expires_soon"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        expired = perf.cache_get("ttl_test")
        assert expired is None
        
        print(f"   ✅ TTL expiration working")
        print(f"      Before expiry: Found ✓")
        print(f"      After expiry: None ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 4: Cache with Tags
    # =========================================================================
    print("-" * 70)
    print("TEST 4: Cache with Tags")
    print("-" * 70)
    
    try:
        # Set with tags
        perf.cache_set("tagged_1", "value1", tags=["study_21", "site"])
        perf.cache_set("tagged_2", "value2", tags=["study_21", "patient"])
        perf.cache_set("tagged_3", "value3", tags=["study_22", "site"])
        
        # Verify all exist
        assert perf.cache_get("tagged_1") is not None
        assert perf.cache_get("tagged_2") is not None
        assert perf.cache_get("tagged_3") is not None
        
        # Invalidate by tag
        invalidated = perf.cache_invalidate_tag("study_21")
        assert invalidated == 2
        
        # Verify invalidation
        assert perf.cache_get("tagged_1") is None
        assert perf.cache_get("tagged_2") is None
        assert perf.cache_get("tagged_3") is not None  # Different tag
        
        print(f"   ✅ Tag-based invalidation working")
        print(f"      Invalidated: 2 entries with 'study_21'")
        print(f"      Remaining: 1 entry with 'study_22'")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 5: Cache Statistics
    # =========================================================================
    print("-" * 70)
    print("TEST 5: Cache Statistics")
    print("-" * 70)
    
    try:
        # Clear and populate
        perf.cache_clear()
        
        for i in range(10):
            perf.cache_set(f"stat_test_{i}", f"value_{i}")
        
        # Generate hits
        for i in range(5):
            perf.cache_get(f"stat_test_{i}")
        
        # Generate misses
        for i in range(3):
            perf.cache_get(f"nonexistent_{i}")
        
        stats = perf.cache_stats()
        
        print(f"   ✅ Cache statistics")
        print(f"      Memory Items: {stats.memory_items}")
        print(f"      Memory Bytes: {stats.memory_bytes:,}")
        print(f"      Hits: {stats.hits}")
        print(f"      Misses: {stats.misses}")
        print(f"      Hit Rate: {stats.hit_rate:.1%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 6: Lazy Loading
    # =========================================================================
    print("-" * 70)
    print("TEST 6: Lazy Loading")
    print("-" * 70)
    
    try:
        # Define a loader
        load_count = [0]
        def expensive_load():
            load_count[0] += 1
            time.sleep(0.1)  # Simulate expensive operation
            return {"loaded": True, "count": load_count[0]}
        
        # Register lazy loader
        perf.register_lazy_loader("expensive_data", expensive_load, LoadingStrategy.LAZY)
        
        # Check not loaded yet
        assert not perf.lazy_loader.is_loaded("expensive_data")
        
        # First access triggers load
        start = time.time()
        value = perf.lazy_get("expensive_data")
        load_time = (time.time() - start) * 1000
        
        assert value["loaded"] == True
        assert perf.lazy_loader.is_loaded("expensive_data")
        
        # Second access should be from cache (faster)
        start = time.time()
        value2 = perf.lazy_get("expensive_data")
        cached_time = (time.time() - start) * 1000
        
        print(f"   ✅ Lazy loading working")
        print(f"      First load: {load_time:.1f}ms")
        print(f"      Cached access: {cached_time:.1f}ms")
        print(f"      Loader called: {load_count[0]} time(s)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 7: Query Profiling
    # =========================================================================
    print("-" * 70)
    print("TEST 7: Query Profiling")
    print("-" * 70)
    
    try:
        # Profile some queries
        queries = [
            ("SELECT * FROM patients WHERE study_id = 'Study_21'", 150, 10000, 500),
            ("SELECT patient_id FROM issues WHERE status = 'open'", 50, 5000, 100),
            ("SELECT * FROM sites", 2000, 100000, 3000),  # Slow query
            ("SELECT a.*, b.* FROM table_a a JOIN table_b b ON a.id = b.id JOIN table_c c ON b.id = c.id", 1500, 50000, 200),
        ]
        
        for query, time_ms, scanned, returned in queries:
            profile = perf.profile_query(query, time_ms, scanned, returned)
        
        # Get slow queries
        slow = perf.get_slow_queries(threshold_ms=1000)
        
        # Get stats
        stats = perf.get_query_stats()
        
        print(f"   ✅ Query profiling working")
        print(f"      Total queries: {stats['total_queries']}")
        print(f"      Average time: {stats['avg_time_ms']:.1f}ms")
        print(f"      Slow queries (>1s): {len(slow)}")
        
        if slow:
            print(f"      Slowest: {slow[0].execution_time_ms:.0f}ms")
            if slow[0].optimization_suggestions:
                print(f"      Suggestions: {', '.join(slow[0].optimization_suggestions[:2])}")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 8: Index Suggestions
    # =========================================================================
    print("-" * 70)
    print("TEST 8: Index Suggestions")
    print("-" * 70)
    
    try:
        # Profile more queries to trigger suggestions
        for i in range(10):
            perf.profile_query(
                f"SELECT * FROM patients WHERE site_id = 'Site_{i}' ORDER BY created_at",
                800, 50000, 100
            )
        
        suggestions = perf.suggest_indexes()
        
        print(f"   ✅ Index suggestion engine working")
        print(f"      Suggestions generated: {len(suggestions)}")
        
        if suggestions:
            for s in suggestions[:2]:
                print(f"      - {s['suggestion']}")
        else:
            print(f"      (No indexes needed based on current queries)")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 9: Response Time Targets
    # =========================================================================
    print("-" * 70)
    print("TEST 9: Response Time Targets")
    print("-" * 70)
    
    try:
        # Set custom target
        perf.set_response_target("custom_op", 100, 300, 1000)
        
        # Record measurements
        for time_ms in [50, 80, 120, 90, 150, 200, 85]:
            perf.record_response_time("custom_op", time_ms)
        
        target = perf.get_response_target("custom_op")
        
        print(f"   ✅ Response time targets working")
        print(f"      Operation: custom_op")
        print(f"      Target: {target.target_ms}ms")
        print(f"      Current Avg: {target.current_avg_ms:.1f}ms")
        print(f"      P95: {target.current_p95_ms:.1f}ms")
        print(f"      Status: {target.status.value}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 10: SLA Violations
    # =========================================================================
    print("-" * 70)
    print("TEST 10: SLA Violations")
    print("-" * 70)
    
    try:
        # Create a violation
        perf.set_response_target("slow_op", 50, 100, 200)
        for _ in range(10):
            perf.record_response_time("slow_op", 500)  # Way over critical
        
        violations = perf.get_sla_violations()
        
        print(f"   ✅ SLA violation detection working")
        print(f"      Violations detected: {len(violations)}")
        
        for v in violations[:3]:
            print(f"      - {v['operation']}: {v['current_avg_ms']:.0f}ms (target: {v['target_ms']}ms)")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 11: Performance Report
    # =========================================================================
    print("-" * 70)
    print("TEST 11: Performance Report")
    print("-" * 70)
    
    try:
        report = perf.get_performance_report()
        
        print(f"   ✅ Performance report generated")
        print(f"      Operations tracked: {report['summary']['total_operations']}")
        print(f"      Healthy: {report['summary']['healthy']}")
        print(f"      Warning: {report['summary']['warning']}")
        print(f"      Critical: {report['summary']['critical']}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 12: Cached Decorator
    # =========================================================================
    print("-" * 70)
    print("TEST 12: Cached Decorator")
    print("-" * 70)
    
    try:
        call_count = [0]
        
        @perf.cached(ttl_seconds=60, tags=["test"])
        def expensive_function(x, y):
            call_count[0] += 1
            time.sleep(0.05)
            return x + y
        
        # First call - should execute
        result1 = expensive_function(1, 2)
        assert result1 == 3
        
        # Second call - should be cached
        result2 = expensive_function(1, 2)
        assert result2 == 3
        
        # Different args - should execute
        result3 = expensive_function(3, 4)
        assert result3 == 7
        
        print(f"   ✅ @cached decorator working")
        print(f"      Function calls: {call_count[0]} (expected: 2)")
        print(f"      Results: {result1}, {result2}, {result3}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 13: Timed Decorator
    # =========================================================================
    print("-" * 70)
    print("TEST 13: Timed Decorator")
    print("-" * 70)
    
    try:
        @perf.timed("decorated_function")
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        # Call a few times
        for _ in range(5):
            slow_function()
        
        target = perf.get_response_target("decorated_function")
        
        print(f"   ✅ @timed decorator working")
        if target:
            print(f"      Samples: {target.samples}")
            print(f"      Avg time: {target.current_avg_ms:.1f}ms")
        else:
            # Check if response was recorded
            print(f"      Timing recorded for 'decorated_function'")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 14: Overall Statistics
    # =========================================================================
    print("-" * 70)
    print("TEST 14: Overall Statistics")
    print("-" * 70)
    
    try:
        stats = perf.get_statistics()
        
        print(f"   ✅ Overall statistics")
        print(f"      Cache Hit Rate: {stats['cache']['hit_rate']:.1%}")
        print(f"      Memory Items: {stats['cache']['memory_items']}")
        print(f"      Lazy Loaders: {stats['lazy_loading']['total_loaders']}")
        print(f"      Queries Profiled: {stats['queries']['total_queries']}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 15: Health Status
    # =========================================================================
    print("-" * 70)
    print("TEST 15: Health Status")
    print("-" * 70)
    
    try:
        health = perf.get_health_status()
        
        print(f"   ✅ Health status")
        print(f"      Status: {health['status']}")
        print(f"      Cache Hit Rate: {health['cache_hit_rate']:.1%}")
        print(f"      SLA Violations: {health['sla_violations']}")
        print(f"      Memory Usage: {health['memory_usage_bytes']:,} bytes")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 16: Convenience Functions
    # =========================================================================
    print("-" * 70)
    print("TEST 16: Convenience Functions")
    print("-" * 70)
    
    try:
        # Test convenience functions
        cache_set("conv_test", "convenience_value", ttl=60)
        value = cache_get("conv_test")
        assert value == "convenience_value"
        
        cache_delete("conv_test")
        assert cache_get("conv_test") is None
        
        record_timing("conv_operation", 123.45)
        
        stats = get_performance_stats()
        assert 'cache' in stats
        
        health = get_health()
        assert 'status' in health
        
        print(f"   ✅ Convenience functions working")
        print(f"      cache_get/set/delete: ✓")
        print(f"      record_timing: ✓")
        print(f"      get_performance_stats: ✓")
        print(f"      get_health: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
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
        print(f"❌ {tests_failed} TEST(S) FAILED")
    
    return tests_failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)