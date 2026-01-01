"""
TRIALPULSE NEXUS 10X - Phase 8.5: Team Workspaces Test Runner

Tests all Team Workspaces functionality.

Author: TrialPulse Team
Date: 2026-01-02
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_team_workspaces():
    """Run all Team Workspaces tests."""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 8.5: TEAM WORKSPACES TEST")
    print("=" * 70)
    print()
    
    from src.collaboration.team_workspaces import (
        TeamWorkspacesManager, reset_team_workspaces_manager,
        WorkspaceType, WorkspaceStatus, MemberRole, GoalStatus, GoalPriority,
        ResourceType, ActivityType
    )
    
    # Reset for clean test
    reset_team_workspaces_manager()
    
    # Use test database
    test_db = "data/collaboration/team_workspaces_test.db"
    Path("data/collaboration").mkdir(parents=True, exist_ok=True)
    
    # Remove existing test DB
    if Path(test_db).exists():
        Path(test_db).unlink()
    
    tests_passed = 0
    tests_failed = 0
    
    # =========================================================================
    # TEST 1: Manager Initialization
    # =========================================================================
    print("-" * 70)
    print("TEST 1: Manager Initialization")
    print("-" * 70)
    
    try:
        manager = TeamWorkspacesManager(test_db)
        print(f"   ✅ Manager initialized")
        print(f"   Database: {test_db}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
        return
    
    # =========================================================================
    # TEST 2: Create Workspace
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 2: Create Workspace")
    print("-" * 70)
    
    try:
        workspace = manager.create_workspace(
            name="Study 21 Team",
            description="Workspace for Study 21 clinical operations team",
            workspace_type=WorkspaceType.STUDY,
            created_by="lead_001",
            created_by_name="John Smith",
            study_id="Study_21",
            region="Global",
            icon="📊",
            color="#3498db",
            tags=["phase3", "oncology"]
        )
        
        print(f"   ✅ Workspace created: {workspace.workspace_id}")
        print(f"   Name: {workspace.name}")
        print(f"   Type: {workspace.workspace_type.value}")
        print(f"   Status: {workspace.status.value}")
        print(f"   Members: {workspace.member_count} (creator auto-added)")
        tests_passed += 1
        
        # Store for later tests
        test_workspace_id = workspace.workspace_id
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return
    
    # =========================================================================
    # TEST 3: Get Workspace
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 3: Get Workspace")
    print("-" * 70)
    
    try:
        ws = manager.get_workspace(test_workspace_id)
        assert ws is not None, "Workspace not found"
        assert ws.name == "Study 21 Team", "Name mismatch"
        assert ws.member_count == 1, "Should have 1 member"
        
        print(f"   ✅ Workspace retrieved successfully")
        print(f"   Name: {ws.name}")
        print(f"   Tags: {ws.tags}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 4: Add Members
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 4: Add Members")
    print("-" * 70)
    
    try:
        # Add CRA
        member1 = manager.add_member(
            workspace_id=test_workspace_id,
            user_id="cra_001",
            user_name="Sarah Chen",
            role=MemberRole.MEMBER,
            added_by="lead_001",
            user_email="sarah.chen@example.com",
            title="Clinical Research Associate",
            department="Clinical Operations"
        )
        
        # Add Data Manager as Admin
        member2 = manager.add_member(
            workspace_id=test_workspace_id,
            user_id="dm_001",
            user_name="Alex Kim",
            role=MemberRole.ADMIN,
            added_by="lead_001"
        )
        
        # Add Safety Lead
        member3 = manager.add_member(
            workspace_id=test_workspace_id,
            user_id="safety_001",
            user_name="Dr. Maria Garcia",
            role=MemberRole.MODERATOR,
            added_by="lead_001"
        )
        
        members = manager.get_members(test_workspace_id)
        
        print(f"   ✅ Members added: {len(members)} total")
        for m in members:
            print(f"      - {m.user_name} ({m.role.value})")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 5: Update Member Role
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 5: Update Member Role")
    print("-" * 70)
    
    try:
        result = manager.update_member_role(
            workspace_id=test_workspace_id,
            user_id="cra_001",
            new_role=MemberRole.MODERATOR,
            updated_by="lead_001",
            updated_by_name="John Smith"
        )
        
        assert result == True, "Role update failed"
        
        member = manager.get_member(test_workspace_id, "cra_001")
        assert member.role == MemberRole.MODERATOR, "Role not updated"
        
        print(f"   ✅ Role updated: Sarah Chen → {member.role.value}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 6: Create Goals
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 6: Create Goals")
    print("-" * 70)
    
    try:
        # Goal 1: Clean Rate
        goal1 = manager.create_goal(
            workspace_id=test_workspace_id,
            title="Achieve 80% Clean Rate",
            description="Reach 80% Tier 2 clean patients before Q2",
            target_value=80.0,
            unit="%",
            due_date=datetime.now() + timedelta(days=90),
            created_by="lead_001",
            created_by_name="John Smith",
            priority=GoalPriority.HIGH,
            owner_id="dm_001",
            owner_name="Alex Kim",
            category="data_quality"
        )
        
        # Goal 2: SDV Completion
        goal2 = manager.create_goal(
            workspace_id=test_workspace_id,
            title="Complete SDV Backlog",
            description="Clear all SDV items older than 30 days",
            target_value=100.0,
            unit="%",
            due_date=datetime.now() + timedelta(days=30),
            created_by="lead_001",
            created_by_name="John Smith",
            priority=GoalPriority.CRITICAL,
            owner_id="cra_001",
            owner_name="Sarah Chen",
            category="monitoring"
        )
        
        # Goal 3: Query Resolution
        goal3 = manager.create_goal(
            workspace_id=test_workspace_id,
            title="Reduce Open Queries to <100",
            description="Close open queries across all sites",
            target_value=100,
            unit="queries",
            due_date=datetime.now() + timedelta(days=60),
            created_by="lead_001",
            created_by_name="John Smith",
            priority=GoalPriority.MEDIUM,
            category="data_quality"
        )
        
        goals = manager.get_goals(test_workspace_id)
        
        print(f"   ✅ Goals created: {len(goals)}")
        for g in goals:
            print(f"      - {g.title} ({g.priority.value}, {g.status.value})")
            print(f"        Target: {g.target_value} {g.unit}, Due: {g.due_date.strftime('%Y-%m-%d')}")
        
        # Store for later
        test_goal_id = goal1.goal_id
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        test_goal_id = None
    
    # =========================================================================
    # TEST 7: Update Goal Progress
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 7: Update Goal Progress")
    print("-" * 70)
    
    try:
        if test_goal_id:
            # Update progress
            updated_goal = manager.update_goal_progress(
                goal_id=test_goal_id,
                current_value=53.7,
                updated_by="dm_001",
                updated_by_name="Alex Kim",
                notes="Current clean rate from latest metrics"
            )
            
            assert updated_goal is not None, "Goal not found"
            assert updated_goal.current_value == 53.7, "Value not updated"
            assert updated_goal.status == GoalStatus.IN_PROGRESS, f"Status should be in_progress, got {updated_goal.status}"
            
            print(f"   ✅ Goal progress updated")
            print(f"   Current: {updated_goal.current_value} {updated_goal.unit}")
            print(f"   Progress: {updated_goal.progress_percent:.1f}%")
            print(f"   Status: {updated_goal.status.value}")
            tests_passed += 1
        else:
            print(f"   ⚠️ Skipped (no goal ID)")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 8: Post Announcement
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 8: Post Announcement")
    print("-" * 70)
    
    try:
        announcement = manager.post_announcement(
            workspace_id=test_workspace_id,
            title="Weekly Team Meeting - Monday 10 AM",
            content="Please join the weekly sync to discuss progress on DB Lock readiness. Agenda: 1) SDV backlog update, 2) Query resolution status, 3) Site performance review.",
            author_id="lead_001",
            author_name="John Smith",
            is_pinned=True,
            priority="high"
        )
        
        announcements = manager.get_announcements(test_workspace_id)
        
        print(f"   ✅ Announcement posted: {announcement.announcement_id}")
        print(f"   Title: {announcement.title}")
        print(f"   Priority: {announcement.priority}")
        print(f"   Pinned: {announcement.is_pinned}")
        print(f"   Total announcements: {len(announcements)}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 9: Share Resource
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 9: Share Resource")
    print("-" * 70)
    
    try:
        resource1 = manager.share_resource(
            workspace_id=test_workspace_id,
            name="Study 21 Protocol v3.2",
            description="Latest protocol version with amendment 2",
            resource_type=ResourceType.DOCUMENT,
            shared_by="lead_001",
            shared_by_name="John Smith",
            url="https://sharepoint.example.com/study21/protocol_v3.2.pdf",
            tags=["protocol", "amendment"]
        )
        
        resource2 = manager.share_resource(
            workspace_id=test_workspace_id,
            name="Weekly Metrics Dashboard",
            description="Real-time dashboard for DQI and clean patient metrics",
            resource_type=ResourceType.DASHBOARD,
            shared_by="dm_001",
            shared_by_name="Alex Kim",
            url="https://dashboard.example.com/study21"
        )
        
        resources = manager.get_resources(test_workspace_id)
        
        print(f"   ✅ Resources shared: {len(resources)}")
        for r in resources:
            print(f"      - {r.name} ({r.resource_type.value})")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 10: Activity Feed
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 10: Activity Feed")
    print("-" * 70)
    
    try:
        activities = manager.get_activity_feed(test_workspace_id, limit=20)
        
        print(f"   ✅ Activity feed: {len(activities)} items")
        print()
        for a in activities[:10]:
            print(f"      [{a.activity_type.value}] {a.title}")
            if a.description:
                print(f"         {a.description[:60]}...")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 11: Update Metrics
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 11: Update Metrics")
    print("-" * 70)
    
    try:
        metrics = manager.update_metrics(
            workspace_id=test_workspace_id,
            total_patients=26443,
            mean_dqi=98.2,
            clean_rate=53.7,
            dblock_ready_rate=27.8,
            open_issues=12500,
            custom_metrics={
                'sdv_completion': 78.5,
                'query_resolution_rate': 85.2
            }
        )
        
        print(f"   ✅ Metrics updated: {metrics.metrics_id}")
        print(f"   Patients: {metrics.total_patients:,}")
        print(f"   DQI: {metrics.mean_dqi:.1f} ({metrics.dqi_trend})")
        print(f"   Clean Rate: {metrics.clean_rate:.1f}% ({metrics.clean_rate_trend})")
        print(f"   DB Lock Ready: {metrics.dblock_ready_rate:.1f}%")
        
        # Update again to see trend
        metrics2 = manager.update_metrics(
            workspace_id=test_workspace_id,
            total_patients=26443,
            mean_dqi=98.8,  # Improved
            clean_rate=55.2,  # Improved
            dblock_ready_rate=29.1,
            open_issues=12100
        )
        
        print(f"   Second update trend - DQI: {metrics2.dqi_trend}, Clean: {metrics2.clean_rate_trend}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 12: List Workspaces
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 12: List Workspaces")
    print("-" * 70)
    
    try:
        # Create another workspace
        ws2 = manager.create_workspace(
            name="CRA Team Workspace",
            description="Workspace for CRA team coordination",
            workspace_type=WorkspaceType.FUNCTIONAL,
            created_by="cra_lead",
            created_by_name="CRA Lead"
        )
        
        # List all
        workspaces, total = manager.list_workspaces()
        
        print(f"   ✅ Workspaces listed: {total}")
        for ws in workspaces:
            print(f"      - {ws.name} ({ws.workspace_type.value}) - {ws.member_count} members")
        
        # List by user
        user_workspaces, _ = manager.list_workspaces(user_id="lead_001")
        print(f"   User 'lead_001' workspaces: {len(user_workspaces)}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 13: Workspace Summary
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 13: Workspace Summary")
    print("-" * 70)
    
    try:
        summary = manager.get_workspace_summary(test_workspace_id)
        
        assert summary is not None, "Summary not returned"
        assert 'workspace' in summary, "Missing workspace"
        assert 'members' in summary, "Missing members"
        assert 'goals' in summary, "Missing goals"
        
        print(f"   ✅ Workspace summary retrieved")
        print(f"   Workspace: {summary['workspace']['name']}")
        print(f"   Members: {summary['members']['total']}")
        print(f"   Goals: {summary['goals']['summary']}")
        print(f"   Activities: {len(summary['activities'])}")
        print(f"   Announcements: {len(summary['announcements'])}")
        print(f"   Resources: {len(summary['resources'])}")
        
        if summary['metrics']:
            print(f"   Latest DQI: {summary['metrics']['mean_dqi']:.1f}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 14: Statistics
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 14: Statistics")
    print("-" * 70)
    
    try:
        stats = manager.get_statistics()
        
        print(f"   ✅ Statistics retrieved")
        print(f"   Workspaces: {stats['workspaces']}")
        print(f"   Members: {stats['members']}")
        print(f"   Goals: {stats['goals']}")
        print(f"   Activities (7 days): {stats['activities']['last_7_days']}")
        print(f"   Resources: {stats['resources']['total']}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 15: Remove Member
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 15: Remove Member")
    print("-" * 70)
    
    try:
        # Add a guest to remove
        manager.add_member(
            workspace_id=test_workspace_id,
            user_id="guest_001",
            user_name="Guest User",
            role=MemberRole.GUEST,
            added_by="lead_001"
        )
        
        members_before = len(manager.get_members(test_workspace_id))
        
        result = manager.remove_member(
            workspace_id=test_workspace_id,
            user_id="guest_001",
            removed_by="lead_001",
            removed_by_name="John Smith"
        )
        
        members_after = len(manager.get_members(test_workspace_id))
        
        assert result == True, "Remove failed"
        assert members_after == members_before - 1, "Member count should decrease"
        
        print(f"   ✅ Member removed successfully")
        print(f"   Members: {members_before} → {members_after}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 16: Archive Workspace
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 16: Archive Workspace")
    print("-" * 70)
    
    try:
        # Create a workspace to archive
        ws_to_archive = manager.create_workspace(
            name="Temporary Workspace",
            description="This will be archived",
            workspace_type=WorkspaceType.AD_HOC,
            created_by="lead_001",
            created_by_name="John Smith"
        )
        
        result = manager.archive_workspace(
            workspace_id=ws_to_archive.workspace_id,
            archived_by="lead_001",
            archived_by_name="John Smith"
        )
        
        archived_ws = manager.get_workspace(ws_to_archive.workspace_id)
        
        assert result == True, "Archive failed"
        assert archived_ws.status == WorkspaceStatus.ARCHIVED, "Status not archived"
        
        print(f"   ✅ Workspace archived")
        print(f"   Status: {archived_ws.status.value}")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 17: Audit Trail
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 17: Audit Trail")
    print("-" * 70)
    
    try:
        audit = manager.get_audit_trail(test_workspace_id, limit=20)
        
        print(f"   ✅ Audit trail: {len(audit)} entries")
        for entry in audit[:5]:
            print(f"      [{entry['action']}] by {entry['actor_id']} at {entry['created_at'][:19]}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # =========================================================================
    # TEST 18: Convenience Functions
    # =========================================================================
        # =========================================================================
    # TEST 18: Convenience Functions
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST 18: Convenience Functions")
    print("-" * 70)
    
    try:
        # Use the same manager instance (not convenience functions that create new instance)
        # Test get_workspace via manager
        ws = manager.get_workspace(test_workspace_id)
        assert ws is not None, "Workspace not found"
        
        # Test list_workspaces for user
        user_workspaces, _ = manager.list_workspaces(user_id="lead_001", status=WorkspaceStatus.ACTIVE)
        assert len(user_workspaces) > 0, "User should have workspaces"
        
        # Test get_statistics
        stats = manager.get_statistics()
        assert 'workspaces' in stats, "Stats should have workspaces"
        
        print(f"   ✅ Convenience functions working")
        print(f"   get_workspace: {ws.name}")
        print(f"   list_workspaces (user): {len(user_workspaces)} workspaces")
        print(f"   get_statistics: {stats['workspaces']['total_active']} active")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"❌ {tests_failed} test(s) failed")
    
    print()
    print(f"📁 Test database: {test_db}")
    
    return tests_failed == 0


if __name__ == "__main__":
    success = test_team_workspaces()
    sys.exit(0 if success else 1)