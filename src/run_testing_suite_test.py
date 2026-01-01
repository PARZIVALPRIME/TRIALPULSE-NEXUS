"""
TRIALPULSE NEXUS 10X - Phase 11.4 Testing Suite Test Runner
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_tests():
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 11.4 TESTING SUITE TEST")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Import and reset
    from src.orchestration.testing_suite import (
        get_testing_suite, reset_testing_suite,
        TestType, TestStatus, TestPriority, TestCategory,
        TestCase, Assertions,
        run_unit_tests, run_integration_tests, run_e2e_tests,
        run_all_tests, get_test_stats
    )
    
    reset_testing_suite()
    
    # =========================================================================
    # TEST 1: Initialize Testing Suite
    # =========================================================================
    print("-" * 70)
    print("TEST 1: Initialize Testing Suite")
    print("-" * 70)
    
    try:
        suite = get_testing_suite()
        assert suite is not None
        assert suite.registry is not None
        assert suite.runner is not None
        
        test_count = suite.registry.count()
        print(f"   ✅ Testing suite initialized")
        print(f"      Total tests registered: {test_count}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 2: Assertions Class
    # =========================================================================
    print("-" * 70)
    print("TEST 2: Assertions Class")
    print("-" * 70)
    
    try:
        assertions = Assertions()
        
        # Test assertTrue
        assertions.assertTrue(True, "Should pass")
        assert assertions.passed == 1
        
        # Test assertEqual
        assertions.assertEqual(5, 5, "5 equals 5")
        assert assertions.passed == 2
        
        # Test assertGreater
        assertions.assertGreater(10, 5, "10 > 5")
        assert assertions.passed == 3
        
        # Test assertIn
        assertions.assertIn("a", ["a", "b", "c"], "a in list")
        assert assertions.passed == 4
        
        # Test assertIsNotNone
        assertions.assertIsNotNone("value", "Not none")
        assert assertions.passed == 5
        
        # Test assertAlmostEqual
        assertions.assertAlmostEqual(3.14159, 3.14160, places=3)
        assert assertions.passed == 6
        
        print(f"   ✅ Assertions class working")
        print(f"      Assertions passed: {assertions.passed}")
        print(f"      Assertions failed: {assertions.failed}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 3: Test Registration
    # =========================================================================
    print("-" * 70)
    print("TEST 3: Test Registration")
    print("-" * 70)
    
    try:
        # Register a custom test
        def custom_test(assertions):
            assertions.assertTrue(True, "Custom test passes")
        
        custom_case = TestCase(
            test_id="CUSTOM-001",
            name="Custom Test",
            description="A custom test for testing",
            test_type=TestType.UNIT,
            category=TestCategory.DATA,
            priority=TestPriority.LOW,
            test_function=custom_test,
            tags=["custom", "test"]
        )
        
        test_id = suite.register_test(custom_case)
        assert test_id == "CUSTOM-001"
        
        # Retrieve it
        retrieved = suite.get_test("CUSTOM-001")
        assert retrieved is not None
        assert retrieved.name == "Custom Test"
        
        print(f"   ✅ Test registration working")
        print(f"      Registered: {test_id}")
        print(f"      Retrieved: {retrieved.name}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 4: Run Unit Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 4: Run Unit Tests")
    print("-" * 70)
    
    try:
        result = suite.run_unit_tests()
        
        print(f"   ✅ Unit tests executed")
        print(f"      Suite: {result.suite_name}")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Failed: {result.failed}")
        print(f"      Duration: {result.duration_seconds:.2f}s")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 5: Run Integration Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 5: Run Integration Tests")
    print("-" * 70)
    
    try:
        result = suite.run_integration_tests()
        
        print(f"   ✅ Integration tests executed")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Failed: {result.failed}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 6: Run E2E Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 6: Run E2E Tests")
    print("-" * 70)
    
    try:
        result = suite.run_e2e_tests()
        
        print(f"   ✅ E2E tests executed")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Failed: {result.failed}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 7: Run UAT Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 7: Run UAT Tests")
    print("-" * 70)
    
    try:
        result = suite.run_uat_tests()
        
        print(f"   ✅ UAT tests executed")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Scenarios: {len(suite.get_uat_scenarios())}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 8: Run Critical Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 8: Run Critical Tests")
    print("-" * 70)
    
    try:
        result = suite.run_critical_tests()
        
        print(f"   ✅ Critical tests executed")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Success Rate: {result.success_rate:.1%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 9: Run by Category
    # =========================================================================
    print("-" * 70)
    print("TEST 9: Run Tests by Category")
    print("-" * 70)
    
    try:
        result = suite.run_by_category(TestCategory.DATA)
        
        print(f"   ✅ Category tests executed")
        print(f"      Category: DATA")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 10: Run by Tag
    # =========================================================================
    print("-" * 70)
    print("TEST 10: Run Tests by Tag")
    print("-" * 70)
    
    try:
        result = suite.run_by_tag("data")
        
        print(f"   ✅ Tag-based tests executed")
        print(f"      Tag: data")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 11: Get Recent Runs
    # =========================================================================
    print("-" * 70)
    print("TEST 11: Get Recent Runs")
    print("-" * 70)
    
    try:
        runs = suite.get_recent_runs(5)
        
        print(f"   ✅ Recent runs retrieved")
        print(f"      Count: {len(runs)}")
        if runs:
            print(f"      Latest: {runs[0]['suite_name']} ({runs[0]['passed']}/{runs[0]['total_tests']} passed)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 12: Get Statistics
    # =========================================================================
    print("-" * 70)
    print("TEST 12: Get Statistics")
    print("-" * 70)
    
    try:
        stats = suite.get_statistics()
        
        print(f"   ✅ Statistics retrieved")
        print(f"      Total tests: {stats['total_tests_registered']}")
        print(f"      By type:")
        for t, count in stats['by_type'].items():
            print(f"         {t}: {count}")
        print(f"      By priority:")
        for p, count in stats['by_priority'].items():
            print(f"         {p}: {count}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 13: Generate Report
    # =========================================================================
    print("-" * 70)
    print("TEST 13: Generate Report")
    print("-" * 70)
    
    try:
        result = suite.run_unit_tests()
        report = suite.generate_report(result)
        
        assert len(report) > 100
        assert "SUMMARY" in report
        assert "RESULTS" in report
        
        print(f"   ✅ Report generated")
        print(f"      Report length: {len(report)} chars")
        print(f"      Contains summary: ✓")
        print(f"      Contains results: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 14: UAT Scenarios
    # =========================================================================
    print("-" * 70)
    print("TEST 14: UAT Scenarios")
    print("-" * 70)
    
    try:
        scenarios = suite.get_uat_scenarios()
        
        print(f"   ✅ UAT scenarios retrieved")
        print(f"      Total scenarios: {len(scenarios)}")
        for sc in scenarios:
            print(f"      - {sc.title} ({len(sc.steps)} steps)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 15: List Tests by Type
    # =========================================================================
    print("-" * 70)
    print("TEST 15: List Tests by Type")
    print("-" * 70)
    
    try:
        unit_tests = suite.list_tests(TestType.UNIT)
        int_tests = suite.list_tests(TestType.INTEGRATION)
        e2e_tests = suite.list_tests(TestType.E2E)
        uat_tests = suite.list_tests(TestType.UAT)
        
        print(f"   ✅ Tests listed by type")
        print(f"      Unit: {len(unit_tests)}")
        print(f"      Integration: {len(int_tests)}")
        print(f"      E2E: {len(e2e_tests)}")
        print(f"      UAT: {len(uat_tests)}")
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
        stats = get_test_stats()
        assert 'total_tests_registered' in stats
        
        print(f"   ✅ Convenience functions working")
        print(f"      get_test_stats: ✓")
        print(f"      run_unit_tests: ✓")
        print(f"      run_integration_tests: ✓")
        print(f"      run_e2e_tests: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 17: Run All Tests
    # =========================================================================
    print("-" * 70)
    print("TEST 17: Run All Tests")
    print("-" * 70)
    
    try:
        result = suite.run_all_tests()
        
        print(f"   ✅ All tests executed")
        print(f"      Total: {result.total_tests}")
        print(f"      Passed: {result.passed}")
        print(f"      Failed: {result.failed}")
        print(f"      Errors: {result.errors}")
        print(f"      Skipped: {result.skipped}")
        print(f"      Success Rate: {result.success_rate:.1%}")
        print(f"      Duration: {result.duration_seconds:.2f}s")
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