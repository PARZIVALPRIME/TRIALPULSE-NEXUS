"""
TRIALPULSE NEXUS 10X - Phase 7.7
Site Portal Test Runner

Author: TrialPulse Team
Date: 2026-01-01
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all site portal tests"""
    
    print("\n" + "="*70)
    print("TRIALPULSE NEXUS 10X - SITE PORTAL TEST RUNNER")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import module
    print("TEST 1: Import Site Portal Module")
    try:
        from dashboard.pages.site_portal import (
            SitePortalDataLoader,
            SiteAction, CRAMessage, HelpRequest, SiteProgress,
            ActionStatus, ActionPriority, MessageType, HelpRequestStatus,
            get_priority_color, get_status_color, format_time_remaining,
            render_page
        )
        print("   ✅ All imports successful")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: Data loader initialization
    print("\nTEST 2: Data Loader Initialization")
    try:
        loader = SitePortalDataLoader()
        print("   ✅ Data loader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 3: Load UPR
    print("\nTEST 3: Load UPR Data")
    try:
        upr = loader.load_upr()
        if upr is not None:
            print(f"   ✅ UPR loaded: {len(upr)} patients, {len(upr.columns)} columns")
            tests_passed += 1
        else:
            print("   ⚠️ UPR file not found (using sample data)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 4: Get sites list
    print("\nTEST 4: Get Sites List")
    try:
        sites = loader.get_sites_list()
        print(f"   ✅ Found {len(sites)} sites")
        if sites:
            print(f"      Sample: {sites[:3]}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 5: Site progress
    print("\nTEST 5: Get Site Progress")
    try:
        site_id = sites[0] if sites else 'Site_1'
        progress = loader.get_site_progress(site_id)
        print(f"   ✅ Progress for {site_id}:")
        print(f"      - Total Patients: {progress.total_patients}")
        print(f"      - Clean Patients: {progress.clean_patients}")
        print(f"      - DB Lock Ready: {progress.db_lock_ready}")
        print(f"      - Open Issues: {progress.open_issues}")
        print(f"      - DQI Score: {progress.dqi_score:.1f}")
        print(f"      - Completion Rate: {progress.completion_rate:.1f}%")
        print(f"      - Trend: {progress.trend}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 6: Site actions
    print("\nTEST 6: Get Site Actions")
    try:
        actions = loader.get_site_actions(site_id)
        print(f"   ✅ Generated {len(actions)} actions")
        
        # Count by priority
        urgent = sum(1 for a in actions if a.priority == ActionPriority.URGENT)
        high = sum(1 for a in actions if a.priority == ActionPriority.HIGH)
        medium = sum(1 for a in actions if a.priority == ActionPriority.MEDIUM)
        low = sum(1 for a in actions if a.priority == ActionPriority.LOW)
        print(f"      By Priority: Urgent={urgent}, High={high}, Medium={medium}, Low={low}")
        
        # Count by status
        pending = sum(1 for a in actions if a.status == ActionStatus.PENDING)
        in_prog = sum(1 for a in actions if a.status == ActionStatus.IN_PROGRESS)
        overdue = sum(1 for a in actions if a.is_overdue)
        print(f"      By Status: Pending={pending}, In Progress={in_prog}, Overdue={overdue}")
        
        if actions:
            print(f"      Top Action: {actions[0].title}")
            print(f"         Priority: {actions[0].priority.value}")
            print(f"         Category: {actions[0].category}")
            print(f"         Due: {actions[0].due_date.strftime('%Y-%m-%d')}")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 7: CRA messages
    print("\nTEST 7: Get CRA Messages")
    try:
        messages = loader.get_cra_messages(site_id)
        print(f"   ✅ Generated {len(messages)} messages")
        
        unread = sum(1 for m in messages if not m.is_read)
        starred = sum(1 for m in messages if m.is_starred)
        print(f"      Unread: {unread}, Starred: {starred}")
        
        # Count by type
        types = {}
        for m in messages:
            t = m.message_type.value
            types[t] = types.get(t, 0) + 1
        print(f"      By Type: {types}")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 8: Help requests
    print("\nTEST 8: Get Help Requests")
    try:
        help_reqs = loader.get_help_requests(site_id)
        print(f"   ✅ Generated {len(help_reqs)} help requests")
        
        open_reqs = sum(1 for r in help_reqs if r.status not in [HelpRequestStatus.RESOLVED, HelpRequestStatus.CLOSED])
        print(f"      Open: {open_reqs}, Resolved: {len(help_reqs) - open_reqs}")
        
        # Count by category
        categories = {}
        for r in help_reqs:
            c = r.category.value
            categories[c] = categories.get(c, 0) + 1
        print(f"      By Category: {categories}")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 9: Helper functions
    print("\nTEST 9: Helper Functions")
    try:
        # Priority colors
        assert get_priority_color(ActionPriority.URGENT) == '#e74c3c'
        assert get_priority_color(ActionPriority.HIGH) == '#f39c12'
        assert get_priority_color(ActionPriority.MEDIUM) == '#3498db'
        assert get_priority_color(ActionPriority.LOW) == '#27ae60'
        print("   ✅ Priority colors correct")
        
        # Status colors
        assert get_status_color(ActionStatus.PENDING) == '#f39c12'
        assert get_status_color(ActionStatus.COMPLETED) == '#27ae60'
        assert get_status_color(ActionStatus.OVERDUE) == '#e74c3c'
        print("   ✅ Status colors correct")
        
        # Time formatting
        assert 'overdue' in format_time_remaining(-2).lower()
        assert 'today' in format_time_remaining(0).lower()
        assert 'tomorrow' in format_time_remaining(1).lower()
        assert 'remaining' in format_time_remaining(5).lower()
        print("   ✅ Time formatting correct")
        
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 10: SiteAction class
    print("\nTEST 10: SiteAction Class")
    try:
        from datetime import datetime, timedelta
        
        # Create test action
        action = SiteAction(
            action_id="TEST-001",
            title="Test Action",
            description="Test description",
            priority=ActionPriority.HIGH,
            status=ActionStatus.PENDING,
            due_date=datetime.now() + timedelta(days=3),
            category="Query"
        )
        
        assert action.action_id == "TEST-001"
        assert action.is_overdue == False
        assert action.days_remaining >= 2
        assert action.urgency_score > 0
        
        # Test overdue action
        overdue_action = SiteAction(
            action_id="TEST-002",
            title="Overdue Action",
            description="Overdue description",
            priority=ActionPriority.URGENT,
            status=ActionStatus.PENDING,
            due_date=datetime.now() - timedelta(days=2),
            category="Safety"
        )
        
        assert overdue_action.is_overdue == True
        assert overdue_action.days_remaining < 0
        
        print("   ✅ SiteAction class working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 11: render_page function exists
    print("\nTEST 11: Render Page Function")
    try:
        assert callable(render_page)
        print("   ✅ render_page function exists and is callable")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"TEST SUMMARY")
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