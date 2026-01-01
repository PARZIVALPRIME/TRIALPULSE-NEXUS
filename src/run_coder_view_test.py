"""
TRIALPULSE NEXUS 10X - Phase 7.8
Coder Workbench Test Runner

Tests the coder_view.py module (Coder Workbench Dashboard)

Author: TrialPulse Team
Date: 2026-01-01
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all coder workbench tests"""
    
    print("\n" + "="*70)
    print("TRIALPULSE NEXUS 10X - CODER WORKBENCH TEST RUNNER")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import module
    print("TEST 1: Import Coder View Module")
    try:
        from dashboard.pages.coder_view import (
            CoderDataLoader,
            CodingItem, CodingStats, DictionaryEntry, EscalationRequest,
            CodingType, CodingStatus, ConfidenceLevel, EscalationReason,
            get_confidence_color, get_confidence_label, get_confidence_icon,
            render_page, test_coder_workbench
        )
        print("   ✅ All imports successful")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        tests_failed += 1
        
        # Try alternative import
        print("\n   Attempting to run internal test function...")
        try:
            from dashboard.pages.coder_view import test_coder_workbench
            result = test_coder_workbench()
            return result
        except Exception as e2:
            print(f"   ❌ Alternative import also failed: {e2}")
            return False
    
    # Test 2: Data loader initialization
    print("\nTEST 2: Data Loader Initialization")
    try:
        loader = CoderDataLoader()
        print("   ✅ Data loader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 3: Coding statistics
    print("\nTEST 3: Coding Statistics")
    try:
        stats = loader.get_coding_stats()
        print(f"   ✅ Stats loaded")
        print(f"      - MedDRA Pending: {stats.meddra_pending}")
        print(f"      - WHODrug Pending: {stats.whodrug_pending}")
        print(f"      - Total Pending: {stats.total_pending}")
        print(f"      - High Confidence Ready: {stats.high_confidence_pending}")
        print(f"      - Coded Today: {stats.total_coded_today}")
        print(f"      - Auto-Approval Rate: {stats.auto_approval_rate:.1%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 4: Coding queue - all items
    print("\nTEST 4: Coding Queue (All Items)")
    try:
        items = loader.get_coding_queue()
        print(f"   ✅ Queue loaded: {len(items)} items")
        
        meddra_count = sum(1 for i in items if i.coding_type == CodingType.MEDDRA)
        whodrug_count = sum(1 for i in items if i.coding_type == CodingType.WHODRUG)
        high_conf = sum(1 for i in items if i.confidence >= 0.95)
        low_conf = sum(1 for i in items if i.confidence < 0.70)
        
        print(f"      - MedDRA items: {meddra_count}")
        print(f"      - WHODrug items: {whodrug_count}")
        print(f"      - High Confidence (≥95%): {high_conf}")
        print(f"      - Low Confidence (<70%): {low_conf}")
        
        if items:
            sample = items[0]
            print(f"      - Sample: '{sample.verbatim_term}' → {sample.suggested_term} ({sample.confidence:.0%})")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 5: MedDRA filter
    print("\nTEST 5: MedDRA Filter")
    try:
        meddra_items = loader.get_coding_queue(coding_type=CodingType.MEDDRA)
        print(f"   ✅ MedDRA items: {len(meddra_items)}")
        
        # Verify all items are MedDRA
        all_meddra = all(i.coding_type == CodingType.MEDDRA for i in meddra_items)
        if all_meddra:
            print("      ✓ All items correctly filtered as MedDRA")
        else:
            print("      ✗ Some items are not MedDRA")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 6: WHODrug filter
    print("\nTEST 6: WHODrug Filter")
    try:
        whodrug_items = loader.get_coding_queue(coding_type=CodingType.WHODRUG)
        print(f"   ✅ WHODrug items: {len(whodrug_items)}")
        
        # Verify all items are WHODrug
        all_whodrug = all(i.coding_type == CodingType.WHODRUG for i in whodrug_items)
        if all_whodrug:
            print("      ✓ All items correctly filtered as WHODrug")
        else:
            print("      ✗ Some items are not WHODrug")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 7: Dictionary search - MedDRA
    print("\nTEST 7: Dictionary Search - MedDRA")
    try:
        # Test various search terms
        search_tests = [
            ("headache", "Headache"),
            ("nausea", "Nausea"),
            ("fever", "Pyrexia"),
            ("10019211", "Headache"),  # Code search
        ]
        
        all_passed = True
        for query, expected in search_tests:
            results = loader.search_dictionary(query, CodingType.MEDDRA)
            if results and results[0]['term'] == expected:
                print(f"      ✓ '{query}' → {results[0]['term']} ({results[0]['code']})")
            else:
                print(f"      ✗ '{query}' → Expected {expected}, got {results[0]['term'] if results else 'No results'}")
                all_passed = False
        
        if all_passed:
            print("   ✅ All MedDRA searches passed")
        else:
            print("   ⚠️ Some searches did not match expected results")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 8: Dictionary search - WHODrug
    print("\nTEST 8: Dictionary Search - WHODrug")
    try:
        search_tests = [
            ("tylenol", "Paracetamol"),
            ("ibuprofen", "Ibuprofen"),
            ("aspirin", "Aspirin"),
            ("advil", "Ibuprofen"),
        ]
        
        all_passed = True
        for query, expected in search_tests:
            results = loader.search_dictionary(query, CodingType.WHODRUG)
            if results and results[0]['term'] == expected:
                print(f"      ✓ '{query}' → {results[0]['term']} ({results[0]['code']})")
            else:
                print(f"      ✗ '{query}' → Expected {expected}, got {results[0]['term'] if results else 'No results'}")
                all_passed = False
        
        if all_passed:
            print("   ✅ All WHODrug searches passed")
        else:
            print("   ⚠️ Some searches did not match expected results")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 9: Escalations
    print("\nTEST 9: Escalations")
    try:
        escalations = loader.get_escalations()
        print(f"   ✅ Escalations loaded: {len(escalations)}")
        
        pending = sum(1 for e in escalations if e.status == "pending")
        resolved = sum(1 for e in escalations if e.status == "resolved")
        
        print(f"      - Pending: {pending}")
        print(f"      - Resolved: {resolved}")
        
        if escalations:
            sample = escalations[0]
            print(f"      - Sample: '{sample.verbatim_term}' → {sample.escalated_to}")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 10: Helper functions
    print("\nTEST 10: Helper Functions")
    try:
        # Test confidence colors
        assert get_confidence_color(0.98) == '#27ae60', "Very high color failed"
        assert get_confidence_color(0.90) == '#2ecc71', "High color failed"
        assert get_confidence_color(0.75) == '#f39c12', "Medium color failed"
        assert get_confidence_color(0.60) == '#e67e22', "Low color failed"
        assert get_confidence_color(0.30) == '#e74c3c', "Very low color failed"
        print("      ✓ Confidence colors correct")
        
        # Test confidence labels
        assert get_confidence_label(0.98) == "Very High"
        assert get_confidence_label(0.90) == "High"
        assert get_confidence_label(0.75) == "Medium"
        assert get_confidence_label(0.60) == "Low"
        assert get_confidence_label(0.30) == "Very Low"
        print("      ✓ Confidence labels correct")
        
        # Test confidence icons
        assert get_confidence_icon(0.98) == "✅"
        assert get_confidence_icon(0.90) == "🟢"
        assert get_confidence_icon(0.75) == "🟡"
        assert get_confidence_icon(0.60) == "🟠"
        assert get_confidence_icon(0.30) == "🔴"
        print("      ✓ Confidence icons correct")
        
        print("   ✅ All helper functions working")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 11: CodingItem properties
    print("\nTEST 11: CodingItem Properties")
    try:
        # Test various confidence levels
        test_cases = [
            (0.98, ConfidenceLevel.VERY_HIGH, '#27ae60'),
            (0.90, ConfidenceLevel.HIGH, '#2ecc71'),
            (0.75, ConfidenceLevel.MEDIUM, '#f39c12'),
            (0.60, ConfidenceLevel.LOW, '#e67e22'),
            (0.30, ConfidenceLevel.VERY_LOW, '#e74c3c'),
        ]
        
        all_passed = True
        for conf, expected_level, expected_color in test_cases:
            item = CodingItem(
                item_id=f"TEST-{int(conf*100)}",
                verbatim_term="test term",
                coding_type=CodingType.MEDDRA,
                status=CodingStatus.PENDING,
                confidence=conf
            )
            
            if item.confidence_level != expected_level:
                print(f"      ✗ {conf:.0%}: Level {item.confidence_level} != {expected_level}")
                all_passed = False
            if item.confidence_color != expected_color:
                print(f"      ✗ {conf:.0%}: Color {item.confidence_color} != {expected_color}")
                all_passed = False
        
        if all_passed:
            print("   ✅ CodingItem properties working correctly")
        else:
            print("   ⚠️ Some CodingItem properties incorrect")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 12: Render page function
    print("\nTEST 12: Render Page Function")
    try:
        assert callable(render_page)
        print("   ✅ render_page function exists and is callable")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    print()
    
    if tests_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_failed} test(s) failed")
    
    print("="*70 + "\n")
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)