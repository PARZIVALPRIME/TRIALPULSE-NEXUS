# src/run_executor_agent_test.py
"""
Test runner for Enhanced EXECUTOR Agent v1.0
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.executor_enhanced import (
    EnhancedExecutorAgent,
    get_executor_agent,
    ActionValidator,
    ApprovalWorkflow,
    ExecutionEngine,
    AuditTrail,
    ExecutionStatus,
    RiskLevel,
    ApprovalLevel,
    ValidationResult
)


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def create_test_action(
    action_id: str = "ACT-001",
    title: str = "Test Action",
    priority: str = "medium",
    issue_type: str = "open_queries",
    category: str = "data_quality",
    effort_hours: float = 1.0,
    impact_score: float = 50.0,
    entity_id: str = "Site_1"
) -> Dict[str, Any]:
    """Create a test action."""
    return {
        "action_id": action_id,
        "title": title,
        "description": f"Test action for {issue_type}",
        "priority": priority,
        "issue_type": issue_type,
        "category": category,
        "effort_hours": effort_hours,
        "impact_score": impact_score,
        "entity_id": entity_id,
        "responsible_role": "Data Manager",
        "steps": ["Step 1", "Step 2", "Step 3"],
        "dependencies": [],
        "due_date": (datetime.now() + timedelta(days=7)).isoformat()
    }


def test_action_validator():
    """Test action validation."""
    print_separator("TEST 1: Action Validator")
    
    validator = ActionValidator()
    
    # Test valid action
    print("\nValidating valid action:")
    valid_action = create_test_action()
    report = validator.validate_action(valid_action)
    
    print(f"  Overall Result: {report.overall_result.value}")
    print(f"  Can Execute: {report.can_execute}")
    print(f"  Checks Passed: {len([c for c in report.checks if c.result == ValidationResult.PASSED])}")
    print(f"  Warnings: {len(report.warnings)}")
    print(f"  Blockers: {len(report.blockers)}")
    
    # Test invalid action (missing fields)
    print("\nValidating invalid action (missing fields):")
    invalid_action = {"title": "Incomplete Action"}
    report = validator.validate_action(invalid_action)
    
    print(f"  Overall Result: {report.overall_result.value}")
    print(f"  Can Execute: {report.can_execute}")
    print(f"  Blockers: {report.blockers}")
    
    # Test prohibited action
    print("\nValidating prohibited action:")
    prohibited_action = create_test_action(issue_type="sae_causality_assessment")
    report = validator.validate_action(prohibited_action)
    
    print(f"  Overall Result: {report.overall_result.value}")
    print(f"  Can Execute: {report.can_execute}")
    
    return True


def test_approval_workflow():
    """Test approval workflow."""
    print_separator("TEST 2: Approval Workflow")
    
    workflow = ApprovalWorkflow()
    
    # Test risk level determination
    print("\nRisk Level Determination:")
    
    test_cases = [
        ("Low risk", create_test_action(priority="low", effort_hours=0.5)),
        ("Medium risk", create_test_action(priority="high", impact_score=90)),
        ("High risk", create_test_action(priority="critical")),
        ("Critical risk", create_test_action(category="safety", issue_type="sae_dm_pending"))
    ]
    
    for name, action in test_cases:
        risk = workflow.determine_risk_level(action)
        requires = workflow.requires_approval(action)
        print(f"  {name}: {risk.value} (requires approval: {requires})")
    
    # Test approval request creation
    print("\nCreating Approval Request:")
    action = create_test_action(priority="high")
    request = workflow.create_approval_request(action, "user_123")
    
    print(f"  Request ID: {request.request_id}")
    print(f"  Risk Level: {request.risk_level.value}")
    print(f"  Approval Level: {request.approval_level.value}")
    print(f"  Approvers Required: {request.approvers_required}")
    print(f"  Status: {request.status}")
    print(f"  Expires: {request.expires_at.strftime('%Y-%m-%d %H:%M') if request.expires_at else 'Never'}")
    
    # Test approval submission
    print("\nSubmitting Approval:")
    approval, updated_request = workflow.submit_approval(
        request_id=request.request_id,
        approver_id="approver_1",
        approver_name="John Doe",
        approver_role="Study Lead",
        decision="approved",
        comments="Looks good"
    )
    
    print(f"  Approval ID: {approval.approval_id}")
    print(f"  Decision: {approval.decision}")
    print(f"  Request Status: {updated_request.status}")
    print(f"  Is Approved: {updated_request.is_approved}")
    
    return True


def test_execution_engine():
    """Test execution engine."""
    print_separator("TEST 3: Execution Engine")
    
    engine = ExecutionEngine()
    
    # Test normal execution
    print("\nExecuting action:")
    action = create_test_action()
    execution = engine.execute_action(action, "system")
    
    print(f"  Execution ID: {execution.execution_id}")
    print(f"  Status: {execution.status.value}")
    print(f"  Result: {execution.result}")
    print(f"  Duration: {execution.duration_seconds:.2f}s" if execution.duration_seconds else "  Duration: N/A")
    print(f"  Rollback Available: {execution.rollback_available}")
    
    # Test dry run
    print("\nDry run execution:")
    dry_execution = engine.execute_action(action, "system", dry_run=True)
    
    print(f"  Status: {dry_execution.status.value}")
    print(f"  Result: {dry_execution.result}")
    print(f"  Output: {dry_execution.output.get('dry_run')}")
    
    # Test rollback
    print("\nTesting rollback:")
    if execution.rollback_available:
        rollback = engine.rollback_action(execution.execution_id, "admin")
        print(f"  Rollback Status: {rollback.status.value}")
        print(f"  Original Status After Rollback: {engine.get_execution_status(execution.execution_id).status.value}")
    
    return True


def test_audit_trail():
    """Test audit trail."""
    print_separator("TEST 4: Audit Trail")
    
    audit = AuditTrail()
    
    # Log several entries
    print("\nLogging audit entries:")
    
    entries = [
        ("ACT-001", "validation", "Action validated", "system"),
        ("ACT-001", "approval_request", "Approval requested", "user_123"),
        ("ACT-001", "approval", "Action approved", "approver_1"),
        ("ACT-001", "execution", "Action executed", "system"),
        ("ACT-002", "validation", "Action validated", "system")
    ]
    
    for action_id, event_type, description, actor in entries:
        entry = audit.log(
            action_id=action_id,
            event_type=event_type,
            event_description=description,
            actor=actor
        )
        print(f"  {entry.entry_id}: {event_type} - {description}")
    
    # Test retrieval
    print(f"\nTotal Entries: {len(audit.entries)}")
    
    print("\nEntries for ACT-001:")
    for entry in audit.get_entries_for_action("ACT-001"):
        print(f"  - {entry.event_type}: {entry.event_description}")
    
    print("\nEntries by actor 'system':")
    system_entries = audit.get_entries_by_actor("system")
    print(f"  Count: {len(system_entries)}")
    
    # Test integrity verification
    print("\nVerifying integrity:")
    is_valid, issues = audit.verify_integrity()
    print(f"  Valid: {is_valid}")
    print(f"  Issues: {len(issues)}")
    
    return True


def test_full_workflow():
    """Test complete execution workflow."""
    print_separator("TEST 5: Full Execution Workflow")
    
    agent = get_executor_agent()
    
    # Create test actions with different risk levels
    actions = [
        create_test_action("ACT-001", "Low Risk Query Resolution", "low", effort_hours=0.5),
        create_test_action("ACT-002", "Medium Risk SDV", "high", "sdv_incomplete", effort_hours=2.0),
        create_test_action("ACT-003", "Critical SAE Reconciliation", "critical", "sae_dm_pending", "safety")
    ]
    
    print(f"\nProcessing {len(actions)} actions:")
    
    result = agent.process_actions(
        actions=actions,
        executed_by="test_user",
        auto_execute_low_risk=True,
        dry_run=True  # Dry run for testing
    )
    
    print(f"\n--- RESULTS ---")
    print(f"Actions Processed: {result.actions_processed}")
    print(f"Actions Executed: {result.actions_executed}")
    print(f"Pending Approval: {result.actions_pending_approval}")
    print(f"Failed: {result.actions_failed}")
    
    print(f"\nValidation Reports: {len(result.validation_reports)}")
    for report in result.validation_reports:
        status = "✅" if report.can_execute else "❌"
        print(f"  {status} {report.action_id}: {report.overall_result.value}")
    
    print(f"\nExecution Records: {len(result.execution_records)}")
    for execution in result.execution_records:
        print(f"  - {execution.action_id}: {execution.status.value}")
    
    print(f"\nApproval Requests: {len(result.approval_requests)}")
    for request in result.approval_requests:
        print(f"  - {request.action_id}: {request.risk_level.value} risk, awaiting {request.approvers_required} approver(s)")
    
    print(f"\nAudit Entries: {len(result.audit_entries)}")
    
    return result.actions_processed == 3


def test_approval_process():
    """Test the approval process end-to-end."""
    print_separator("TEST 6: Approval Process")
    
    agent = get_executor_agent()
    
    # Create high-risk action
    action = create_test_action(
        "ACT-HIGH",
        "High Risk Action",
        "critical",
        "signature_gaps"
    )
    
    print("\n1. Validating action:")
    validation = agent.validate_action(action)
    print(f"   Can Execute: {validation.can_execute}")
    
    print("\n2. Requesting approval:")
    request = agent.request_approval(action, "data_manager")
    print(f"   Request ID: {request.request_id}")
    print(f"   Risk Level: {request.risk_level.value}")
    print(f"   Approvers Needed: {request.approvers_required}")
    
    print("\n3. First approval:")
    approval1, updated = agent.approve_action(
        request_id=request.request_id,
        approver_id="lead_001",
        approver_name="Jane Smith",
        approver_role="Study Lead",
        decision="approved",
        comments="Approved for execution"
    )
    print(f"   Status After: {updated.status}")
    print(f"   Approvals: {len(updated.approvals_received)}/{updated.approvers_required}")
    
    if updated.approvers_required > 1 and not updated.is_approved:
        print("\n4. Second approval (if needed):")
        approval2, final = agent.approve_action(
            request_id=request.request_id,
            approver_id="mgr_001",
            approver_name="Bob Wilson",
            approver_role="Manager",
            decision="approved"
        )
        print(f"   Final Status: {final.status}")
        print(f"   Is Approved: {final.is_approved}")
    
    return True


def test_rollback_capability():
    """Test rollback capability."""
    print_separator("TEST 7: Rollback Capability")
    
    agent = get_executor_agent()
    
    # Execute an action
    action = create_test_action("ACT-ROLLBACK", "Reversible Action")
    
    print("\n1. Executing action:")
    execution = agent.execute_action(
        action,
        "system",
        skip_validation=True,
        skip_approval=True
    )
    print(f"   Execution ID: {execution.execution_id}")
    print(f"   Status: {execution.status.value}")
    print(f"   Rollback Available: {execution.rollback_available}")
    
    if execution.rollback_available and execution.status == ExecutionStatus.COMPLETED:
        print("\n2. Rolling back:")
        rollback = agent.rollback_execution(execution.execution_id, "admin")
        print(f"   Rollback Status: {rollback.status.value}")
        print(f"   Original Now: {agent.execution_engine.get_execution_status(execution.execution_id).status.value}")
    
    print("\n3. Audit trail for this action:")
    entries = agent.audit_trail.get_entries_for_action("ACT-ROLLBACK")
    for entry in entries:
        print(f"   - {entry.event_type}: {entry.event_description}")
    
    return True


def test_natural_language_queries():
    """Test natural language query handling."""
    print_separator("TEST 8: Natural Language Queries")
    
    agent = get_executor_agent()
    
    # First, create some state
    actions = [
        create_test_action("ACT-NL-1", "Test 1", "low"),
        create_test_action("ACT-NL-2", "Test 2", "high")
    ]
    agent.process_actions(actions, auto_execute_low_risk=True, dry_run=True)
    
    queries = [
        "Show pending approvals",
        "What's the execution status?",
        "Show audit trail",
        "What's available for rollback?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        result = agent.execute_from_query(query)
        print(f"Summary: {result.summary}")
    
    return True


def test_batch_processing():
    """Test batch action processing."""
    print_separator("TEST 9: Batch Processing")
    
    agent = get_executor_agent()
    
    # Create batch of actions
    actions = [
        create_test_action(f"BATCH-{i:03d}", f"Batch Action {i}", 
                          priority=["low", "medium", "high"][i % 3],
                          effort_hours=0.5 + (i * 0.2))
        for i in range(10)
    ]
    
    print(f"\nProcessing {len(actions)} actions in batch:")
    
    start = time.time()
    result = agent.process_actions(
        actions=actions,
        executed_by="batch_processor",
        auto_execute_low_risk=True,
        dry_run=True
    )
    duration = time.time() - start
    
    print(f"\n--- BATCH RESULTS ---")
    print(f"Duration: {duration:.2f}s")
    print(f"Actions/Second: {len(actions)/duration:.1f}")
    print(f"Executed (Low Risk): {result.actions_executed}")
    print(f"Pending Approval: {result.actions_pending_approval}")
    print(f"Failed: {result.actions_failed}")
    
    print(f"\nBy Risk Level:")
    risk_counts: Dict[str, int] = {}
    for req in result.approval_requests:
        risk = req.risk_level.value
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count}")
    
    return result.actions_processed == 10


def run_all_tests():
    """Run all executor agent tests."""
    print("\n" + "=" * 70)
    print(" ENHANCED EXECUTOR AGENT v1.0 - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Action Validator
    try:
        results['action_validator'] = test_action_validator()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['action_validator'] = False
    
    # Test 2: Approval Workflow
    try:
        results['approval_workflow'] = test_approval_workflow()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['approval_workflow'] = False
    
    # Test 3: Execution Engine
    try:
        results['execution_engine'] = test_execution_engine()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['execution_engine'] = False
    
    # Test 4: Audit Trail
    try:
        results['audit_trail'] = test_audit_trail()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['audit_trail'] = False
    
    # Test 5: Full Workflow
    try:
        results['full_workflow'] = test_full_workflow()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['full_workflow'] = False
    
    # Test 6: Approval Process
    try:
        results['approval_process'] = test_approval_process()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['approval_process'] = False
    
    # Test 7: Rollback Capability
    try:
        results['rollback'] = test_rollback_capability()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['rollback'] = False
    
    # Test 8: Natural Language Queries
    try:
        results['nl_queries'] = test_natural_language_queries()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['nl_queries'] = False
    
    # Test 9: Batch Processing
    try:
        results['batch_processing'] = test_batch_processing()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['batch_processing'] = False
    
    # Summary
    print_separator("TEST RESULTS SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASSED" if passed_flag else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)