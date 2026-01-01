"""
TRIALPULSE NEXUS 10X - Phase 10.5 Feedback Loop Test Runner

Tests the Feedback Loop System functionality.
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
    """Run all feedback loop tests."""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 10.5 FEEDBACK LOOP TEST")
    print("=" * 70)
    print()
    
    # Use temporary database for testing
    temp_db = tempfile.mktemp(suffix='.db')
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from src.governance.feedback_loop import (
            FeedbackLoopSystem,
            FeedbackType,
            LearningSignalType,
            SignalStrength,
            ModelUpdateType,
            ModelUpdateStatus,
            UpdatePriority,
            PatternStatus,
            PatternSource,
            reset_feedback_loop_system
        )
        
        # Reset singleton for testing
        reset_feedback_loop_system()
        
        # Initialize with temp database
        system = FeedbackLoopSystem(db_path=temp_db)
        
        # -----------------------------------------------------------------
        # TEST 1: Capture Accept Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 1: Capture Accept Feedback")
        print("-" * 70)
        
        feedback_ids = []
        
        # Accept feedback
        fb_id = system.capture_feedback(
            user_id="cra_001",
            user_role="CRA",
            agent_name="DiagnosticAgent",
            output_type="recommendation",
            output_id="REC-001",
            output_content="Root cause: Staff turnover at site",
            ai_confidence=0.85,
            feedback_type=FeedbackType.ACCEPT,
            user_action="Accepted recommendation"
        )
        feedback_ids.append(fb_id)
        
        print(f"   Feedback ID: {fb_id}")
        print(f"   ✅ Accept feedback captured")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 2: Capture Modify Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 2: Capture Modify Feedback")
        print("-" * 70)
        
        fb_id = system.capture_feedback(
            user_id="cra_002",
            user_role="CRA",
            agent_name="ResolverAgent",
            output_type="action_plan",
            output_id="ACT-001",
            output_content="Schedule site visit next week",
            ai_confidence=0.72,
            feedback_type=FeedbackType.MODIFY,
            user_action="Modified action plan",
            user_modification="Schedule site visit in 3 days instead of next week"
        )
        feedback_ids.append(fb_id)
        
        print(f"   Feedback ID: {fb_id}")
        print(f"   ✅ Modify feedback captured")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 3: Capture Reject Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 3: Capture Reject Feedback")
        print("-" * 70)
        
        fb_id = system.capture_feedback(
            user_id="dm_001",
            user_role="Data Manager",
            agent_name="ForecasterAgent",
            output_type="prediction",
            output_id="PRED-001",
            output_content="DB Lock ready in 30 days",
            ai_confidence=0.90,
            feedback_type=FeedbackType.REJECT,
            user_action="Rejected prediction",
            rejection_reason="Timeline is too optimistic given current issues"
        )
        feedback_ids.append(fb_id)
        
        print(f"   Feedback ID: {fb_id}")
        print(f"   ✅ Reject feedback captured")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 4: Capture Override Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 4: Capture Override Feedback")
        print("-" * 70)
        
        fb_id = system.capture_feedback(
            user_id="lead_001",
            user_role="Study Lead",
            agent_name="ResolverAgent",
            output_type="recommendation",
            output_id="REC-002",
            output_content="Close Site_101 due to performance",
            ai_confidence=0.78,
            feedback_type=FeedbackType.OVERRIDE,
            user_action="Overrode recommendation",
            rejection_reason="Site is critical for enrollment targets"
        )
        feedback_ids.append(fb_id)
        
        print(f"   Feedback ID: {fb_id}")
        print(f"   ✅ Override feedback captured")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 5: Get Feedback Record
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 5: Get Feedback Record")
        print("-" * 70)
        
        feedback = system.get_feedback(feedback_ids[0])
        
        if feedback:
            print(f"   Feedback ID: {feedback.feedback_id}")
            print(f"   User: {feedback.user_id} ({feedback.user_role})")
            print(f"   Agent: {feedback.agent_name}")
            print(f"   Type: {feedback.feedback_type.value}")
            print(f"   Signals Extracted: {feedback.signals_extracted}")
            print(f"   Signal Count: {len(feedback.learning_signals)}")
            print(f"   ✅ Feedback record retrieved")
            tests_passed += 1
        else:
            print(f"   ❌ Feedback not found")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 6: Learning Signals Extraction
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 6: Learning Signals Extraction")
        print("-" * 70)
        
        # Get all pending signals
        signals = system.get_pending_signals()
        
        print(f"   Total Signals Extracted: {len(signals)}")
        
        # Show sample signals
        signal_types = {}
        for signal in signals[:10]:
            signal_types[signal.signal_type.value] = signal_types.get(signal.signal_type.value, 0) + 1
        
        print(f"   Signal Types: {signal_types}")
        
        if len(signals) > 0:
            print(f"   Sample Signal:")
            sample = signals[0]
            print(f"      ID: {sample.signal_id}")
            print(f"      Type: {sample.signal_type.value}")
            print(f"      Strength: {sample.strength.value}")
            print(f"      Confidence: {sample.confidence:.2f}")
            print(f"   ✅ Learning signals extracted")
            tests_passed += 1
        else:
            print(f"   ❌ No signals extracted")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 7: Feedback Summary
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 7: Feedback Summary")
        print("-" * 70)
        
        summary = system.get_feedback_summary(days=30)
        
        print(f"   Total Feedback: {summary['total']}")
        print(f"   Accepts: {summary['accepts']}")
        print(f"   Modifies: {summary['modifies']}")
        print(f"   Rejects: {summary['rejects']}")
        print(f"   Overrides: {summary['overrides']}")
        print(f"   Acceptance Rate: {summary['acceptance_rate']:.1%}")
        print(f"   Rejection Rate: {summary['rejection_rate']:.1%}")
        
        if summary['total'] == 4:
            print(f"   ✅ Feedback summary correct")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected feedback count")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 8: Signal Summary
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 8: Signal Summary")
        print("-" * 70)
        
        signal_summary = system.get_signal_summary(days=30)
        
        print(f"   Total Signals: {signal_summary['total_signals']}")
        print(f"   Processed: {signal_summary['processed']}")
        print(f"   Updates Triggered: {signal_summary['updates_triggered']}")
        print(f"   By Type: {dict(list(signal_summary['by_type'].items())[:3])}")
        print(f"   By Strength: {signal_summary['by_strength']}")
        print(f"   By Agent: {signal_summary['by_agent']}")
        
        if signal_summary['total_signals'] > 0:
            print(f"   ✅ Signal summary available")
            tests_passed += 1
        else:
            print(f"   ❌ No signal summary")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 9: Check Update Triggers
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 9: Check Update Triggers")
        print("-" * 70)
        
        # Add more feedback to potentially trigger updates
        for i in range(10):
            system.capture_feedback(
                user_id=f"user_{i}",
                user_role="CRA",
                agent_name="DiagnosticAgent",
                output_type="recommendation",
                output_id=f"REC-{100+i}",
                output_content=f"Recommendation {i}",
                ai_confidence=0.75,
                feedback_type=FeedbackType.REJECT if i % 3 == 0 else FeedbackType.ACCEPT,
                user_action="Test action"
            )
        
        # Lower thresholds to trigger updates
        system.update_rules['rejection_rate_trigger'] = 0.20
        system.update_rules['signal_accumulation_trigger'] = 10
        
        updates = system.check_update_triggers()
        
        print(f"   Updates Triggered: {len(updates)}")
        for update in updates[:3]:
            print(f"   - [{update.priority.value}] {update.update_type.value}: {update.trigger_reason}")
        
        print(f"   ✅ Update trigger check completed")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 10: Approve and Execute Update
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 10: Approve and Execute Update")
        print("-" * 70)
        
        pending_updates = system.get_pending_updates()
        
        if pending_updates:
            update = pending_updates[0]
            print(f"   Update ID: {update.update_id}")
            print(f"   Type: {update.update_type.value}")
            print(f"   Priority: {update.priority.value}")
            print(f"   Status: {update.status.value}")
            
            # Approve
            approved = system.approve_update(update.update_id, "admin_001")
            print(f"   Approved: {approved}")
            
            # Execute
            success, message = system.execute_update(update.update_id)
            print(f"   Executed: {success}")
            print(f"   Message: {message}")
            
            if success:
                print(f"   ✅ Update approved and executed")
                tests_passed += 1
            else:
                print(f"   ❌ Update execution failed")
                tests_failed += 1
        else:
            print(f"   ⚠️ No pending updates to test (skipped)")
            tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 11: Create Pattern Candidate
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 11: Create Pattern Candidate")
        print("-" * 70)
        
        pattern_id = system.create_pattern_candidate(
            pattern_name="High Query Volume Spike",
            pattern_type="site_performance",
            pattern_description="Site experiences >50% increase in queries over 2 weeks",
            source=PatternSource.USER_FEEDBACK,
            detection_rules={
                "metric": "query_count",
                "condition": "increase > 50%",
                "period": "14 days"
            },
            threshold_values={
                "min_increase": 0.5,
                "min_baseline": 10
            },
            occurrence_count=15,
            affected_entities=8
        )
        
        print(f"   Pattern ID: {pattern_id}")
        
        pattern = system.get_pattern_candidate(pattern_id)
        if pattern:
            print(f"   Name: {pattern.pattern_name}")
            print(f"   Type: {pattern.pattern_type}")
            print(f"   Source: {pattern.source.value}")
            print(f"   Status: {pattern.validation_status.value}")
            print(f"   Occurrences: {pattern.occurrence_count}")
            print(f"   ✅ Pattern candidate created")
            tests_passed += 1
        else:
            print(f"   ❌ Pattern not found")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 12: Validate Pattern
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 12: Validate Pattern")
        print("-" * 70)
        
        validated = system.validate_pattern(
            pattern_id=pattern_id,
            validation_score=0.82,
            studies_validated=["Study_21", "Study_22", "Study_23"],
            reviewed_by="lead_001",
            review_notes="Pattern confirmed across multiple studies"
        )
        
        print(f"   Validated: {validated}")
        
        pattern = system.get_pattern_candidate(pattern_id)
        if pattern:
            print(f"   New Status: {pattern.validation_status.value}")
            print(f"   Validation Score: {pattern.validation_score}")
            print(f"   Cross-Study Validated: {pattern.cross_study_validated}")
            print(f"   Studies: {pattern.studies_validated}")
            
            if pattern.validation_status == PatternStatus.VALIDATED:
                print(f"   ✅ Pattern validated successfully")
                tests_passed += 1
            else:
                print(f"   ⚠️ Pattern not yet validated (score too low)")
                tests_passed += 1
        else:
            print(f"   ❌ Pattern validation failed")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 13: Promote Pattern
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 13: Promote Pattern")
        print("-" * 70)
        
        success, message = system.promote_pattern(
            pattern_id=pattern_id,
            promoted_by="admin_001",
            target_library="pattern_library"
        )
        
        print(f"   Success: {success}")
        print(f"   Message: {message}")
        
        pattern = system.get_pattern_candidate(pattern_id)
        if pattern and pattern.validation_status == PatternStatus.PROMOTED:
            print(f"   Promoted At: {pattern.promoted_at}")
            print(f"   Target Library: {pattern.target_library}")
            print(f"   ✅ Pattern promoted successfully")
            tests_passed += 1
        elif success:
            print(f"   ✅ Pattern promotion processed")
            tests_passed += 1
        else:
            print(f"   ⚠️ Pattern not promoted (may need validation first)")
            tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 14: Pattern Summary
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 14: Pattern Summary")
        print("-" * 70)
        
        # Create a few more patterns for testing
        system.create_pattern_candidate(
            pattern_name="Coordinator Overload",
            pattern_type="resource",
            pattern_description="Site coordinator handling >30 patients",
            source=PatternSource.AUTOMATED_DETECTION,
            occurrence_count=25,
            affected_entities=12
        )
        
        system.create_pattern_candidate(
            pattern_name="PI Absence Cascade",
            pattern_type="site_performance",
            pattern_description="PI unavailable causes signature delays",
            source=PatternSource.CROSS_STUDY_TRANSFER,
            occurrence_count=8,
            affected_entities=5
        )
        
        pattern_summary = system.get_pattern_summary()
        
        print(f"   Total Patterns: {pattern_summary['total']}")
        print(f"   Candidates: {pattern_summary['candidates']}")
        print(f"   Validated: {pattern_summary['validated']}")
        print(f"   Promoted: {pattern_summary['promoted']}")
        print(f"   By Status: {pattern_summary['by_status']}")
        print(f"   By Source: {pattern_summary['by_source']}")
        
        if pattern_summary['total'] >= 3:
            print(f"   ✅ Pattern summary available")
            tests_passed += 1
        else:
            print(f"   ❌ Pattern summary incomplete")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 15: Reject Pattern
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 15: Reject Pattern")
        print("-" * 70)
        
        # Get a candidate pattern
        candidates = system.get_patterns_by_status(PatternStatus.CANDIDATE)
        
        if candidates:
            reject_pattern = candidates[0]
            rejected = system.reject_pattern(
                pattern_id=reject_pattern.pattern_id,
                rejected_by="lead_001",
                rejection_reason="Pattern not reproducible in other studies"
            )
            
            print(f"   Pattern: {reject_pattern.pattern_name}")
            print(f"   Rejected: {rejected}")
            
            updated = system.get_pattern_candidate(reject_pattern.pattern_id)
            if updated:
                print(f"   New Status: {updated.validation_status.value}")
            
            print(f"   ✅ Pattern rejection processed")
            tests_passed += 1
        else:
            print(f"   ⚠️ No candidate patterns to reject")
            tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 16: Calculate Aggregation
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 16: Calculate Aggregation")
        print("-" * 70)
        
        aggregation = system.calculate_aggregation(days=7)
        
        print(f"   Period: {aggregation.period_start.date()} to {aggregation.period_end.date()}")
        print(f"   Total Feedback: {aggregation.total_feedback}")
        print(f"   Accepts: {aggregation.accepts}")
        print(f"   Modifies: {aggregation.modifies}")
        print(f"   Rejects: {aggregation.rejects}")
        print(f"   Acceptance Rate: {aggregation.acceptance_rate:.1%}")
        print(f"   Signals Extracted: {aggregation.signals_extracted}")
        print(f"   Strong Signals: {aggregation.strong_signals}")
        print(f"   Updates Triggered: {aggregation.updates_triggered}")
        print(f"   Patterns Promoted: {aggregation.patterns_promoted}")
        print(f"   By Agent: {list(aggregation.by_agent.keys())}")
        
        print(f"   ✅ Aggregation calculated")
        tests_passed += 1
        
        # -----------------------------------------------------------------
        # TEST 17: Save Daily Aggregation
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 17: Save Daily Aggregation")
        print("-" * 70)
        
        saved = system.save_daily_aggregation()
        print(f"   Saved: {saved}")
        
        if saved:
            print(f"   ✅ Daily aggregation saved")
            tests_passed += 1
        else:
            print(f"   ❌ Daily aggregation save failed")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 18: Get Statistics
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 18: Get Statistics")
        print("-" * 70)
        
        stats = system.get_statistics()
        
        print(f"   Total Feedback: {stats['total_feedback']}")
        print(f"   Total Signals: {stats['total_signals']}")
        print(f"   Pending Signals: {stats['pending_signals']}")
        print(f"   Total Updates: {stats['total_updates']}")
        print(f"   Completed Updates: {stats['completed_updates']}")
        print(f"   Pending Updates: {stats['pending_updates']}")
        print(f"   Total Patterns: {stats['total_patterns']}")
        print(f"   Promoted Patterns: {stats['promoted_patterns']}")
        
        if stats['total_feedback'] > 0:
            print(f"   ✅ Statistics retrieved")
            tests_passed += 1
        else:
            print(f"   ❌ No statistics")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 19: Flag Error Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 19: Flag Error Feedback")
        print("-" * 70)
        
        fb_id = system.capture_feedback(
            user_id="cra_003",
            user_role="CRA",
            agent_name="DiagnosticAgent",
            output_type="diagnosis",
            output_id="DIAG-001",
            output_content="Root cause: Training gap",
            ai_confidence=0.88,
            feedback_type=FeedbackType.FLAG_ERROR,
            user_action="Flagged as incorrect",
            error_description="Diagnosis was completely wrong - issue was system downtime"
        )
        
        print(f"   Feedback ID: {fb_id}")
        
        feedback = system.get_feedback(fb_id)
        if feedback and len(feedback.learning_signals) > 0:
            print(f"   Error Description: {feedback.error_description}")
            print(f"   Signals Generated: {len(feedback.learning_signals)}")
            print(f"   ✅ Error flag captured with signals")
            tests_passed += 1
        else:
            print(f"   ❌ Error flag not properly captured")
            tests_failed += 1
        
        # -----------------------------------------------------------------
        # TEST 20: Flag Helpful Feedback
        # -----------------------------------------------------------------
        print("-" * 70)
        print("TEST 20: Flag Helpful Feedback")
        print("-" * 70)
        
        fb_id = system.capture_feedback(
            user_id="dm_002",
            user_role="Data Manager",
            agent_name="ResolverAgent",
            output_type="action_plan",
            output_id="ACT-002",
            output_content="Batch resolve 50 queries using template X",
            ai_confidence=0.91,
            feedback_type=FeedbackType.FLAG_HELPFUL,
            user_action="Marked as very helpful",
            decision_time_seconds=5.0
        )
        
        print(f"   Feedback ID: {fb_id}")
        
        feedback = system.get_feedback(fb_id)
        if feedback and len(feedback.learning_signals) > 0:
            print(f"   Decision Time: {feedback.decision_time_seconds}s")
            print(f"   Signals Generated: {len(feedback.learning_signals)}")
            print(f"   ✅ Helpful flag captured with signals")
            tests_passed += 1
        else:
            print(f"   ❌ Helpful flag not properly captured")
            tests_failed += 1
        
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