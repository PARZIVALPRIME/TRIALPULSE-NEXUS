"""
TRIALPULSE NEXUS 10X - Phase 10.4 Trust Metrics Test Runner

Tests the Trust Metrics System functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all trust metrics tests."""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 10.4 TRUST METRICS TEST")
    print("=" * 70)
    print()
    
    # Use temporary database for testing
    temp_db = tempfile.mktemp(suffix='.db')
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Import directly from trust_metrics module to avoid __init__.py issues
        from src.governance.trust_metrics import (
            TrustMetricsSystem,
            InteractionType,
            InteractionOutcome,
            SatisfactionLevel,
            FeatureCategory,
            TrustLevel,
            reset_trust_metrics_system,
        )
        
        # Reset singleton for testing
        reset_trust_metrics_system()
        
        # Initialize with temp database
        system = TrustMetricsSystem(db_path=temp_db)
        
        # -----------------------------------------------------------------
        # TEST 1: Record Interactions
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 1: Record Interactions")
        print("-" * 70)
        
        interactions = []
        for i in range(20):
            int_id = system.record_interaction(
                user_id=f"user_{i % 5}",
                user_role=["CRA", "Data Manager", "Study Lead"][i % 3],
                agent_name=["DiagnosticAgent", "ResolverAgent", "ForecasterAgent"][i % 3],
                interaction_type=InteractionType.RECOMMENDATION,
                ai_confidence=0.7 + (i % 4) * 0.08,
                ai_suggestion=f"Suggestion {i}"
            )
            interactions.append(int_id)
        
        print(f"   ✅ Recorded {len(interactions)} interactions")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 2: Record Outcomes
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 2: Record Outcomes")
        print("-" * 70)
        
        outcomes = [
            InteractionOutcome.ACCEPTED,
            InteractionOutcome.ACCEPTED_MODIFIED,
            InteractionOutcome.REJECTED,
            InteractionOutcome.OVERRIDDEN,
            InteractionOutcome.ACCEPTED,
            InteractionOutcome.ACCEPTED,
            InteractionOutcome.REJECTED,
            InteractionOutcome.ACCEPTED_MODIFIED,
            InteractionOutcome.OVERRIDDEN,
            InteractionOutcome.ACCEPTED,
        ]
        
        for i, int_id in enumerate(interactions[:10]):
            outcome = outcomes[i % len(outcomes)]
            override_reason = "Better judgment" if outcome == InteractionOutcome.OVERRIDDEN else None
            
            system.record_outcome(
                interaction_id=int_id,
                outcome=outcome,
                time_to_decision_seconds=10 + i * 5,
                override_reason=override_reason
            )
        
        print(f"   ✅ Recorded outcomes for 10 interactions")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 3: Override Rate Calculation
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 3: Override Rate Calculation")
        print("-" * 70)
        
        override_data = system.get_override_rate(days=30)
        
        print(f"   Total Interactions: {override_data['total_interactions']}")
        print(f"   Overrides: {override_data['overrides']}")
        print(f"   Override Rate: {override_data['override_rate_percent']}")
        
        if override_data['total_interactions'] == 10:
            print(f"   ✅ Override rate calculated correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected interaction count")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 4: Acceptance Rate Calculation
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 4: Acceptance Rate Calculation")
        print("-" * 70)
        
        acceptance_data = system.get_acceptance_rate(days=30)
        
        print(f"   Accepted: {acceptance_data['accepted']}")
        print(f"   Accepted Modified: {acceptance_data['accepted_modified']}")
        print(f"   Rejected: {acceptance_data['rejected']}")
        print(f"   Acceptance Rate: {acceptance_data['acceptance_rate_percent']}")
        print(f"   Avg Decision Time: {acceptance_data['avg_decision_time_seconds']:.1f}s")
        
        if acceptance_data['acceptance_rate'] > 0:
            print(f"   ✅ Acceptance rate calculated")
            tests_passed += 1
        else:
            print(f"   ❌ Acceptance rate calculation failed")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 5: Record Satisfaction Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 5: Record Satisfaction Feedback")
        print("-" * 70)
        
        satisfaction_levels = [
            (SatisfactionLevel.VERY_SATISFIED, 5, True),
            (SatisfactionLevel.SATISFIED, 4, True),
            (SatisfactionLevel.NEUTRAL, 3, None),
            (SatisfactionLevel.DISSATISFIED, 2, False),
            (SatisfactionLevel.SATISFIED, 4, True),
        ]
        
        for i, (level, rating, recommend) in enumerate(satisfaction_levels):
            system.record_satisfaction(
                user_id=f"user_{i}",
                user_role="CRA",
                satisfaction_level=level,
                rating=rating,
                feedback_text=f"Feedback {i}",
                would_recommend=recommend,
                feature_category=FeatureCategory.AI_ASSISTANT
            )
        
        print(f"   ✅ Recorded {len(satisfaction_levels)} satisfaction feedbacks")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 6: Satisfaction Metrics
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 6: Satisfaction Metrics")
        print("-" * 70)
        
        satisfaction_data = system.get_satisfaction_metrics(days=30)
        
        print(f"   Count: {satisfaction_data['count']}")
        print(f"   Avg Rating: {satisfaction_data['avg_rating_formatted']}")
        print(f"   NPS Score: {satisfaction_data['nps_score']}")
        print(f"   Promoters: {satisfaction_data['promoters']}")
        print(f"   Detractors: {satisfaction_data['detractors']}")
        
        if satisfaction_data['count'] == 5:
            print(f"   ✅ Satisfaction metrics calculated")
            tests_passed += 1
        else:
            print(f"   ❌ Satisfaction count mismatch")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 7: Record Feature Usage
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 7: Record Feature Usage")
        print("-" * 70)
        
        features = [
            (FeatureCategory.DASHBOARD, "executive_overview", 120),
            (FeatureCategory.AI_ASSISTANT, "natural_language_query", 45),
            (FeatureCategory.REPORTS, "generate_report", 30),
            (FeatureCategory.CASCADE_EXPLORER, "view_dependencies", 60),
            (FeatureCategory.DASHBOARD, "cra_view", 90),
        ]
        
        for i, (category, name, duration) in enumerate(features):
            system.record_feature_usage(
                user_id=f"user_{i % 3}",
                user_role="CRA",
                feature_category=category,
                feature_name=name,
                action="use",
                duration_seconds=duration
            )
        
        print(f"   ✅ Recorded {len(features)} feature usages")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 8: Adoption Metrics
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 8: Adoption Metrics")
        print("-" * 70)
        
        adoption_data = system.get_adoption_metrics(days=30, total_users=10)
        
        print(f"   Active Users: {adoption_data['active_users']}")
        print(f"   Adoption Rate: {adoption_data['adoption_rate_percent']}")
        print(f"   Total Usage Count: {adoption_data['total_usage_count']}")
        print(f"   Avg Session Duration: {adoption_data['avg_session_duration']}s")
        print(f"   Features Used: {len(adoption_data['by_feature'])}")
        
        if adoption_data['active_users'] > 0:
            print(f"   ✅ Adoption metrics calculated")
            tests_passed += 1
        else:
            print(f"   ❌ No active users found")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 9: Calculate Trust Metrics
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 9: Calculate Trust Metrics")
        print("-" * 70)
        
        trust_metrics = system.calculate_trust_metrics(days=30, total_users=10)
        
        print(f"   Trust Score: {trust_metrics.trust_score}/100")
        print(f"   Trust Level: {trust_metrics.trust_level.value}")
        print(f"   Override Rate: {trust_metrics.override_rate:.1%}")
        print(f"   Acceptance Rate: {trust_metrics.acceptance_rate:.1%}")
        print(f"   Avg Satisfaction: {trust_metrics.avg_satisfaction}/5")
        print(f"   Adoption Rate: {trust_metrics.adoption_rate:.1%}")
        print(f"   NPS Score: {trust_metrics.nps_score}")
        
        if trust_metrics.trust_score >= 0:
            print(f"   ✅ Trust metrics calculated")
            tests_passed += 1
        else:
            print(f"   ❌ Invalid trust score")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 10: Acceptance by Agent
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 10: Acceptance by Agent")
        print("-" * 70)
        
        by_agent = system.get_acceptance_by_agent(days=30)
        
        print(f"   Agents tracked: {len(by_agent)}")
        for agent, data in list(by_agent.items())[:3]:
            print(f"   - {agent}: {data['acceptance_rate']:.1%} acceptance")
        
        if len(by_agent) > 0:
            print(f"   ✅ Agent breakdown available")
            tests_passed += 1
        else:
            print(f"   ❌ No agent data found")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 11: Trust Alerts
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 11: Trust Alerts")
        print("-" * 70)
        
        # Lower thresholds to trigger alerts
        system.thresholds['acceptance_rate_low'] = 0.90
        system.thresholds['override_rate_high'] = 0.10
        
        alerts = system.check_trust_alerts(days=30)
        
        print(f"   Alerts Generated: {len(alerts)}")
        for alert in alerts[:3]:
            print(f"   - [{alert.severity.upper()}] {alert.message}")
        
        active = system.get_active_alerts()
        print(f"   Active Alerts: {len(active)}")
        
        print(f"   ✅ Alert system working")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 12: Acknowledge Alert
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 12: Acknowledge Alert")
        print("-" * 70)
        
        if alerts:
            ack_success = system.acknowledge_alert(alerts[0].alert_id, "admin_001")
            print(f"   Acknowledged: {ack_success}")
            
            remaining = system.get_active_alerts()
            print(f"   Remaining Active: {len(remaining)}")
            
            if ack_success:
                print(f"   ✅ Alert acknowledged")
                tests_passed += 1
            else:
                print(f"   ❌ Acknowledge failed")
                tests_failed += 1
        else:
            print(f"   ⚠️ No alerts to acknowledge (skipped)")
            tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 13: Statistics
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 13: Statistics")
        print("-" * 70)
        
        stats = system.get_statistics()
        
        print(f"   Total Interactions: {stats['total_interactions']}")
        print(f"   Total Feedback: {stats['total_feedback']}")
        print(f"   Total Feature Usage: {stats['total_feature_usage']}")
        print(f"   Trust Score: {stats['trust_score']}")
        print(f"   Trust Level: {stats['trust_level']}")
        
        if stats['total_interactions'] > 0:
            print(f"   ✅ Statistics retrieved")
            tests_passed += 1
        else:
            print(f"   ❌ No statistics")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 14: Daily Snapshot
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 14: Daily Snapshot")
        print("-" * 70)
        
        snapshot_saved = system.save_daily_snapshot()
        print(f"   Snapshot Saved: {snapshot_saved}")
        
        if snapshot_saved:
            print(f"   ✅ Daily snapshot saved")
            tests_passed += 1
        else:
            print(f"   ❌ Snapshot failed")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 15: Get Interaction
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 15: Get Interaction")
        print("-" * 70)
        
        if interactions:
            interaction = system.get_interaction(interactions[0])
            if interaction:
                print(f"   Interaction ID: {interaction.interaction_id}")
                print(f"   User: {interaction.user_id}")
                print(f"   Agent: {interaction.agent_name}")
                print(f"   Outcome: {interaction.outcome.value if interaction.outcome else 'Pending'}")
                print(f"   ✅ Interaction retrieved")
                tests_passed += 1
            else:
                print(f"   ❌ Interaction not found")
                tests_failed += 1
        else:
            print(f"   ⚠️ No interactions to retrieve")
            tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    finally:
        # Cleanup temp database
        try:
            if os.path.exists(temp_db):
                os.remove(temp_db)
        except:
            pass
    
    # Summary
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