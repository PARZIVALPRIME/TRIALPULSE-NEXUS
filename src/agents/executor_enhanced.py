# src/agents/executor_enhanced.py
"""
TRIALPULSE NEXUS 10X - Enhanced EXECUTOR Agent v1.0

Purpose: Validate, approve, execute, and track actions with full audit trail,
         human-in-the-loop controls, and rollback capabilities.

Features:
- Action validation with pre-execution checks
- Multi-level approval workflow
- Execution tracking and status management
- Rollback capability for reversible actions
- Complete audit trail logging
- Auto-execution rules for low-risk actions
- Human-in-the-loop controls for high-risk actions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json
import hashlib
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of action execution."""
    PENDING = "pending"
    VALIDATING = "validating"
    VALIDATION_FAILED = "validation_failed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ApprovalLevel(Enum):
    """Approval levels required for actions."""
    NONE = "none"                    # Auto-execute
    SINGLE = "single"                # One approver
    DUAL = "dual"                    # Two approvers
    MANAGER = "manager"              # Manager approval
    STUDY_LEAD = "study_lead"        # Study Lead approval
    SPONSOR = "sponsor"              # Sponsor approval required


class RiskLevel(Enum):
    """Risk level of an action."""
    LOW = "low"           # Can auto-execute
    MEDIUM = "medium"     # Single approval
    HIGH = "high"         # Dual approval
    CRITICAL = "critical" # Manager + Study Lead approval


class ValidationResult(Enum):
    """Result of validation check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """A single validation check."""
    check_id: str
    name: str
    description: str
    result: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "description": self.description,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationReport:
    """Complete validation report for an action."""
    report_id: str
    action_id: str
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_result: ValidationResult = ValidationResult.PASSED
    can_execute: bool = True
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_check(self, check: ValidationCheck):
        self.checks.append(check)
        self._update_overall_result()
    
    def _update_overall_result(self):
        """Update overall result based on all checks."""
        results = [c.result for c in self.checks]
        
        if ValidationResult.FAILED in results:
            self.overall_result = ValidationResult.FAILED
            self.can_execute = False
            self.blockers = [c.message for c in self.checks if c.result == ValidationResult.FAILED]
        elif ValidationResult.WARNING in results:
            self.overall_result = ValidationResult.WARNING
            self.can_execute = True
            self.warnings = [c.message for c in self.checks if c.result == ValidationResult.WARNING]
        else:
            self.overall_result = ValidationResult.PASSED
            self.can_execute = True
    
    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "action_id": self.action_id,
            "checks": [c.to_dict() for c in self.checks],
            "overall_result": self.overall_result.value,
            "can_execute": self.can_execute,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "checks_passed": len([c for c in self.checks if c.result == ValidationResult.PASSED]),
            "checks_failed": len([c for c in self.checks if c.result == ValidationResult.FAILED]),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Approval:
    """An approval decision."""
    approval_id: str
    action_id: str
    approver_id: str
    approver_name: str
    approver_role: str
    decision: str  # approved, rejected
    comments: str = ""
    conditions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "approval_id": self.approval_id,
            "action_id": self.action_id,
            "approver_id": self.approver_id,
            "approver_name": self.approver_name,
            "approver_role": self.approver_role,
            "decision": self.decision,
            "comments": self.comments,
            "conditions": self.conditions,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ApprovalRequest:
    """Request for approval."""
    request_id: str
    action_id: str
    action_title: str
    action_description: str
    risk_level: RiskLevel
    approval_level: ApprovalLevel
    requested_by: str
    approvers_required: int = 1
    approvals_received: List[Approval] = field(default_factory=list)
    rejections_received: List[Approval] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, expired
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_approved(self) -> bool:
        return len(self.approvals_received) >= self.approvers_required
    
    @property
    def is_rejected(self) -> bool:
        return len(self.rejections_received) > 0
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def add_approval(self, approval: Approval):
        if approval.decision == "approved":
            self.approvals_received.append(approval)
            if self.is_approved:
                self.status = "approved"
        else:
            self.rejections_received.append(approval)
            self.status = "rejected"
    
    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "action_id": self.action_id,
            "action_title": self.action_title,
            "action_description": self.action_description,
            "risk_level": self.risk_level.value,
            "approval_level": self.approval_level.value,
            "requested_by": self.requested_by,
            "approvers_required": self.approvers_required,
            "approvals_received": len(self.approvals_received),
            "rejections_received": len(self.rejections_received),
            "status": self.status,
            "is_approved": self.is_approved,
            "is_rejected": self.is_rejected,
            "is_expired": self.is_expired,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ExecutionRecord:
    """Record of an action execution."""
    execution_id: str
    action_id: str
    action_title: str
    status: ExecutionStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    executed_by: str = "system"
    result: str = ""
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    rollback_available: bool = False
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict:
        return {
            "execution_id": self.execution_id,
            "action_id": self.action_id,
            "action_title": self.action_title,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "executed_by": self.executed_by,
            "result": self.result,
            "output": self.output,
            "error": self.error,
            "rollback_available": self.rollback_available,
            "retry_count": self.retry_count
        }


@dataclass
class AuditEntry:
    """An entry in the audit trail."""
    entry_id: str
    action_id: str
    event_type: str  # validation, approval_request, approval, execution, rollback, etc.
    event_description: str
    actor: str
    actor_role: str = ""
    old_state: Dict[str, Any] = field(default_factory=dict)
    new_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for audit integrity."""
        data = f"{self.entry_id}{self.action_id}{self.event_type}{self.actor}{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "action_id": self.action_id,
            "event_type": self.event_type,
            "event_description": self.event_description,
            "actor": self.actor,
            "actor_role": self.actor_role,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum
        }


@dataclass
class ExecutorResult:
    """Complete result of executor operations."""
    result_id: str
    query: str
    actions_processed: int = 0
    actions_executed: int = 0
    actions_pending_approval: int = 0
    actions_failed: int = 0
    validation_reports: List[ValidationReport] = field(default_factory=list)
    approval_requests: List[ApprovalRequest] = field(default_factory=list)
    execution_records: List[ExecutionRecord] = field(default_factory=list)
    audit_entries: List[AuditEntry] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "query": self.query,
            "actions_processed": self.actions_processed,
            "actions_executed": self.actions_executed,
            "actions_pending_approval": self.actions_pending_approval,
            "actions_failed": self.actions_failed,
            "validation_reports": [v.to_dict() for v in self.validation_reports],
            "approval_requests": [a.to_dict() for a in self.approval_requests],
            "execution_records": [e.to_dict() for e in self.execution_records],
            "audit_entries_count": len(self.audit_entries),
            "summary": self.summary,
            "recommendations": self.recommendations,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


class ActionValidator:
    """Validates actions before execution."""
    
    # Actions that should never be auto-executed
    NEVER_AUTO_EXECUTE = [
        "sae_causality",
        "protocol_deviation",
        "medical_judgment",
        "regulatory_submission",
        "site_closure",
        "locked_data_change",
        "unblinding"
    ]
    
    # Required fields for each action category
    REQUIRED_FIELDS = {
        "data_quality": ["entity_id", "issue_type", "responsible_role"],
        "safety": ["entity_id", "issue_type", "responsible_role", "priority"],
        "monitoring": ["entity_id", "responsible_role"],
        "site_management": ["entity_id", "responsible_role"],
        "communication": ["recipient", "message_type"]
    }
    
    def __init__(self):
        self._check_counter = 0
    
    def _generate_check_id(self) -> str:
        self._check_counter += 1
        return f"CHK-{self._check_counter:04d}"
    
    def validate_action(self, action: Dict[str, Any]) -> ValidationReport:
        """Run all validation checks on an action."""
        report = ValidationReport(
            report_id=f"VAL-{datetime.now().strftime('%H%M%S')}",
            action_id=action.get('action_id', 'unknown')
        )
        
        # Check 1: Required fields
        report.add_check(self._check_required_fields(action))
        
        # Check 2: Action not in never-auto list
        report.add_check(self._check_not_prohibited(action))
        
        # Check 3: Entity exists
        report.add_check(self._check_entity_exists(action))
        
        # Check 4: Responsible role assigned
        report.add_check(self._check_role_assigned(action))
        
        # Check 5: Effort is reasonable
        report.add_check(self._check_effort_reasonable(action))
        
        # Check 6: Priority is valid
        report.add_check(self._check_priority_valid(action))
        
        # Check 7: Dependencies met
        report.add_check(self._check_dependencies_met(action))
        
        # Check 8: Due date is valid
        report.add_check(self._check_due_date_valid(action))
        
        return report
    
    def _check_required_fields(self, action: Dict) -> ValidationCheck:
        """Check that required fields are present."""
        missing = []
        required = ['action_id', 'title', 'entity_id', 'issue_type']
        
        for field in required:
            if not action.get(field):
                missing.append(field)
        
        if missing:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Required Fields",
                description="Check that all required fields are present",
                result=ValidationResult.FAILED,
                message=f"Missing required fields: {', '.join(missing)}",
                details={"missing_fields": missing}
            )
        
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Required Fields",
            description="Check that all required fields are present",
            result=ValidationResult.PASSED,
            message="All required fields present"
        )
    
    def _check_not_prohibited(self, action: Dict) -> ValidationCheck:
        """Check action is not in prohibited list."""
        issue_type = action.get('issue_type', '').lower()
        title = action.get('title', '').lower()
        
        for prohibited in self.NEVER_AUTO_EXECUTE:
            if prohibited in issue_type or prohibited in title:
                return ValidationCheck(
                    check_id=self._generate_check_id(),
                    name="Prohibited Action Check",
                    description="Check action is not prohibited from auto-execution",
                    result=ValidationResult.FAILED,
                    message=f"Action type '{prohibited}' requires manual handling",
                    details={"prohibited_type": prohibited}
                )
        
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Prohibited Action Check",
            description="Check action is not prohibited from auto-execution",
            result=ValidationResult.PASSED,
            message="Action is allowed for execution"
        )
    
    def _check_entity_exists(self, action: Dict) -> ValidationCheck:
        """Check that the target entity exists."""
        entity_id = action.get('entity_id', '')
        
        if not entity_id or entity_id == 'unknown':
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Entity Existence",
                description="Check that target entity exists",
                result=ValidationResult.WARNING,
                message="Entity ID not specified or unknown"
            )
        
        # In production, would check against database
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Entity Existence",
            description="Check that target entity exists",
            result=ValidationResult.PASSED,
            message=f"Entity {entity_id} validated"
        )
    
    def _check_role_assigned(self, action: Dict) -> ValidationCheck:
        """Check that a responsible role is assigned."""
        role = action.get('responsible_role', '')
        
        if not role:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Role Assignment",
                description="Check that responsible role is assigned",
                result=ValidationResult.FAILED,
                message="No responsible role assigned"
            )
        
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Role Assignment",
            description="Check that responsible role is assigned",
            result=ValidationResult.PASSED,
            message=f"Assigned to: {role}"
        )
    
    def _check_effort_reasonable(self, action: Dict) -> ValidationCheck:
        """Check that effort estimate is reasonable."""
        effort = action.get('effort_hours', 0)
        
        if effort <= 0:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Effort Estimate",
                description="Check effort estimate is reasonable",
                result=ValidationResult.WARNING,
                message="No effort estimate provided"
            )
        
        if effort > 40:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Effort Estimate",
                description="Check effort estimate is reasonable",
                result=ValidationResult.WARNING,
                message=f"High effort estimate ({effort}h) - consider breaking into smaller tasks"
            )
        
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Effort Estimate",
            description="Check effort estimate is reasonable",
            result=ValidationResult.PASSED,
            message=f"Effort estimate: {effort} hours"
        )
    
    def _check_priority_valid(self, action: Dict) -> ValidationCheck:
        """Check that priority is valid."""
        priority = action.get('priority', '')
        valid_priorities = ['critical', 'high', 'medium', 'low', 'optional']
        
        if not priority:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Priority Validation",
                description="Check priority is valid",
                result=ValidationResult.WARNING,
                message="No priority specified"
            )
        
        if priority.lower() not in valid_priorities:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Priority Validation",
                description="Check priority is valid",
                result=ValidationResult.FAILED,
                message=f"Invalid priority: {priority}"
            )
        
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Priority Validation",
            description="Check priority is valid",
            result=ValidationResult.PASSED,
            message=f"Priority: {priority}"
        )
    
    def _check_dependencies_met(self, action: Dict) -> ValidationCheck:
        """Check that action dependencies are met."""
        dependencies = action.get('dependencies', [])
        
        if not dependencies:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Dependencies",
                description="Check action dependencies are met",
                result=ValidationResult.PASSED,
                message="No dependencies"
            )
        
        # In production, would check if dependent actions are completed
        return ValidationCheck(
            check_id=self._generate_check_id(),
            name="Dependencies",
            description="Check action dependencies are met",
            result=ValidationResult.WARNING,
            message=f"{len(dependencies)} dependencies - verify completion before executing"
        )
    
    def _check_due_date_valid(self, action: Dict) -> ValidationCheck:
        """Check that due date is valid and not expired."""
        due_date = action.get('due_date')
        
        if not due_date:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Due Date",
                description="Check due date is valid",
                result=ValidationResult.PASSED,
                message="No due date specified"
            )
        
        try:
            if isinstance(due_date, str):
                due_dt = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            else:
                due_dt = due_date
            
            if due_dt < datetime.now():
                return ValidationCheck(
                    check_id=self._generate_check_id(),
                    name="Due Date",
                    description="Check due date is valid",
                    result=ValidationResult.WARNING,
                    message="Due date has passed"
                )
            
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Due Date",
                description="Check due date is valid",
                result=ValidationResult.PASSED,
                message=f"Due: {due_dt.strftime('%Y-%m-%d')}"
            )
        except Exception:
            return ValidationCheck(
                check_id=self._generate_check_id(),
                name="Due Date",
                description="Check due date is valid",
                result=ValidationResult.WARNING,
                message="Could not parse due date"
            )


class ApprovalWorkflow:
    """Manages approval workflows for actions."""
    
    # Approval requirements by risk level
    APPROVAL_REQUIREMENTS = {
        RiskLevel.LOW: (ApprovalLevel.NONE, 0),
        RiskLevel.MEDIUM: (ApprovalLevel.SINGLE, 1),
        RiskLevel.HIGH: (ApprovalLevel.DUAL, 2),
        RiskLevel.CRITICAL: (ApprovalLevel.STUDY_LEAD, 2)
    }
    
    # Expiration times by approval level
    EXPIRATION_HOURS = {
        ApprovalLevel.NONE: 0,
        ApprovalLevel.SINGLE: 72,
        ApprovalLevel.DUAL: 48,
        ApprovalLevel.MANAGER: 24,
        ApprovalLevel.STUDY_LEAD: 24,
        ApprovalLevel.SPONSOR: 168
    }
    
    def __init__(self):
        self._request_counter = 0
        self._approval_counter = 0
        self.pending_requests: Dict[str, ApprovalRequest] = {}
    
    def _generate_request_id(self) -> str:
        self._request_counter += 1
        return f"REQ-{self._request_counter:04d}"
    
    def _generate_approval_id(self) -> str:
        self._approval_counter += 1
        return f"APR-{self._approval_counter:04d}"
    
    def determine_risk_level(self, action: Dict) -> RiskLevel:
        """Determine the risk level of an action."""
        priority = action.get('priority', 'medium').lower()
        issue_type = action.get('issue_type', '').lower()
        category = action.get('category', '').lower()
        
        # Safety-related actions are critical
        if 'safety' in category or 'sae' in issue_type:
            return RiskLevel.CRITICAL
        
        # Critical priority maps to high risk
        if priority == 'critical':
            return RiskLevel.HIGH
        
        # High priority with high impact
        impact = action.get('impact_score', 0)
        if priority == 'high' and impact > 80:
            return RiskLevel.MEDIUM
        
        # Low effort, low priority = low risk
        effort = action.get('effort_hours', 1)
        if effort <= 1 and priority in ['low', 'optional']:
            return RiskLevel.LOW
        
        # Default to medium
        return RiskLevel.MEDIUM
    
    def create_approval_request(
        self,
        action: Dict,
        requested_by: str = "system"
    ) -> ApprovalRequest:
        """Create an approval request for an action."""
        risk_level = self.determine_risk_level(action)
        approval_level, approvers_required = self.APPROVAL_REQUIREMENTS[risk_level]
        
        # Calculate expiration
        exp_hours = self.EXPIRATION_HOURS.get(approval_level, 72)
        expires_at = datetime.now() + timedelta(hours=exp_hours) if exp_hours > 0 else None
        
        request = ApprovalRequest(
            request_id=self._generate_request_id(),
            action_id=action.get('action_id', 'unknown'),
            action_title=action.get('title', 'Untitled Action'),
            action_description=action.get('description', ''),
            risk_level=risk_level,
            approval_level=approval_level,
            requested_by=requested_by,
            approvers_required=approvers_required,
            expires_at=expires_at
        )
        
        self.pending_requests[request.request_id] = request
        
        return request
    
    def submit_approval(
        self,
        request_id: str,
        approver_id: str,
        approver_name: str,
        approver_role: str,
        decision: str,
        comments: str = "",
        conditions: List[str] = None
    ) -> Tuple[Approval, ApprovalRequest]:
        """Submit an approval decision."""
        request = self.pending_requests.get(request_id)
        
        if not request:
            raise ValueError(f"Approval request {request_id} not found")
        
        if request.is_expired:
            request.status = "expired"
            raise ValueError(f"Approval request {request_id} has expired")
        
        approval = Approval(
            approval_id=self._generate_approval_id(),
            action_id=request.action_id,
            approver_id=approver_id,
            approver_name=approver_name,
            approver_role=approver_role,
            decision=decision,
            comments=comments,
            conditions=conditions or []
        )
        
        request.add_approval(approval)
        
        return approval, request
    
    def get_pending_requests(self, role: str = None) -> List[ApprovalRequest]:
        """Get pending approval requests, optionally filtered by role."""
        pending = [r for r in self.pending_requests.values() if r.status == "pending"]
        
        # In production, would filter by role-based access
        return pending
    
    def requires_approval(self, action: Dict) -> bool:
        """Check if an action requires approval."""
        risk_level = self.determine_risk_level(action)
        return risk_level != RiskLevel.LOW


class ExecutionEngine:
    """Executes validated and approved actions."""
    
    def __init__(self):
        self._execution_counter = 0
        self.execution_history: Dict[str, ExecutionRecord] = {}
    
    def _generate_execution_id(self) -> str:
        self._execution_counter += 1
        return f"EXE-{self._execution_counter:04d}"
    
    def execute_action(
        self,
        action: Dict,
        executed_by: str = "system",
        dry_run: bool = False
    ) -> ExecutionRecord:
        """Execute an action."""
        execution = ExecutionRecord(
            execution_id=self._generate_execution_id(),
            action_id=action.get('action_id', 'unknown'),
            action_title=action.get('title', 'Untitled'),
            status=ExecutionStatus.IN_PROGRESS,
            started_at=datetime.now(),
            executed_by=executed_by
        )
        
        try:
            if dry_run:
                # Simulate execution
                execution.result = "Dry run - no changes made"
                execution.output = {"dry_run": True, "would_execute": action}
                execution.status = ExecutionStatus.COMPLETED
            else:
                # In production, would execute actual action
                # For now, simulate successful execution
                execution.result = f"Action '{action.get('title')}' executed successfully"
                execution.output = {
                    "action_id": action.get('action_id'),
                    "entity_id": action.get('entity_id'),
                    "issue_type": action.get('issue_type'),
                    "steps_completed": len(action.get('steps', [])),
                    "simulated": True
                }
                execution.status = ExecutionStatus.COMPLETED
                
                # Store rollback data for reversible actions
                if self._is_reversible(action):
                    execution.rollback_available = True
                    execution.rollback_data = {
                        "original_action": action,
                        "execution_time": datetime.now().isoformat()
                    }
            
            execution.completed_at = datetime.now()
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            
            if execution.retry_count < execution.max_retries:
                execution.retry_count += 1
        
        self.execution_history[execution.execution_id] = execution
        return execution
    
    def rollback_action(self, execution_id: str, rolled_back_by: str = "system") -> ExecutionRecord:
        """Rollback a previously executed action."""
        original = self.execution_history.get(execution_id)
        
        if not original:
            raise ValueError(f"Execution {execution_id} not found")
        
        if not original.rollback_available:
            raise ValueError(f"Execution {execution_id} is not reversible")
        
        rollback_execution = ExecutionRecord(
            execution_id=self._generate_execution_id(),
            action_id=f"ROLLBACK-{original.action_id}",
            action_title=f"Rollback: {original.action_title}",
            status=ExecutionStatus.IN_PROGRESS,
            started_at=datetime.now(),
            executed_by=rolled_back_by
        )
        
        try:
            # In production, would perform actual rollback
            rollback_execution.result = f"Rollback of {execution_id} completed"
            rollback_execution.output = {
                "original_execution_id": execution_id,
                "rollback_data": original.rollback_data,
                "simulated": True
            }
            rollback_execution.status = ExecutionStatus.COMPLETED
            
            # Mark original as rolled back
            original.status = ExecutionStatus.ROLLED_BACK
            
        except Exception as e:
            rollback_execution.status = ExecutionStatus.FAILED
            rollback_execution.error = str(e)
        
        rollback_execution.completed_at = datetime.now()
        self.execution_history[rollback_execution.execution_id] = rollback_execution
        
        return rollback_execution
    
    def _is_reversible(self, action: Dict) -> bool:
        """Determine if an action is reversible."""
        # Most data quality actions are not truly reversible
        # But we track state for audit purposes
        non_reversible = ['protocol_deviation', 'site_closure', 'regulatory_submission']
        issue_type = action.get('issue_type', '').lower()
        
        for nr in non_reversible:
            if nr in issue_type:
                return False
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get the status of an execution."""
        return self.execution_history.get(execution_id)


class AuditTrail:
    """Manages audit trail for all executor operations."""
    
    def __init__(self, persist_path: Optional[Path] = None):
        self._entry_counter = 0
        self.entries: List[AuditEntry] = []
        self.persist_path = persist_path
    
    def _generate_entry_id(self) -> str:
        self._entry_counter += 1
        return f"AUD-{self._entry_counter:05d}"
    
    def log(
        self,
        action_id: str,
        event_type: str,
        event_description: str,
        actor: str,
        actor_role: str = "",
        old_state: Dict = None,
        new_state: Dict = None,
        metadata: Dict = None
    ) -> AuditEntry:
        """Log an audit entry."""
        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            action_id=action_id,
            event_type=event_type,
            event_description=event_description,
            actor=actor,
            actor_role=actor_role,
            old_state=old_state or {},
            new_state=new_state or {},
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        
        # Persist to file if configured
        if self.persist_path:
            self._persist_entry(entry)
        
        return entry
    
    def _persist_entry(self, entry: AuditEntry):
        """Persist audit entry to file."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.persist_path, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")
    
    def get_entries_for_action(self, action_id: str) -> List[AuditEntry]:
        """Get all audit entries for an action."""
        return [e for e in self.entries if e.action_id == action_id]
    
    def get_entries_by_actor(self, actor: str) -> List[AuditEntry]:
        """Get all audit entries by an actor."""
        return [e for e in self.entries if e.actor == actor]
    
    def get_entries_by_type(self, event_type: str) -> List[AuditEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.entries if e.event_type == event_type]
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the audit trail."""
        issues = []
        
        for entry in self.entries:
            expected_checksum = entry._calculate_checksum()
            if entry.checksum != expected_checksum:
                issues.append(f"Checksum mismatch for entry {entry.entry_id}")
        
        return len(issues) == 0, issues


class EnhancedExecutorAgent:
    """
    Enhanced EXECUTOR Agent for action validation, approval, and execution.
    
    Capabilities:
    - Pre-execution validation
    - Multi-level approval workflow
    - Execution with status tracking
    - Rollback capability
    - Complete audit trail
    - Auto-execution for low-risk actions
    """
    
    def __init__(self, llm_wrapper=None, audit_path: Optional[str] = None):
        self.validator = ActionValidator()
        self.approval_workflow = ApprovalWorkflow()
        self.execution_engine = ExecutionEngine()
        self.audit_trail = AuditTrail(
            persist_path=Path(audit_path) if audit_path else None
        )
        self.llm = llm_wrapper
        self._result_counter = 0
        
        logger.info("EnhancedExecutorAgent initialized")
    
    def _generate_result_id(self) -> str:
        self._result_counter += 1
        return f"EXR-{self._result_counter:04d}"
    
    def validate_action(self, action: Dict) -> ValidationReport:
        """Validate an action before execution."""
        report = self.validator.validate_action(action)
        
        # Log validation
        self.audit_trail.log(
            action_id=action.get('action_id', 'unknown'),
            event_type="validation",
            event_description=f"Validation {'passed' if report.can_execute else 'failed'}",
            actor="system",
            new_state={"validation_result": report.overall_result.value}
        )
        
        return report
    
    def request_approval(
        self,
        action: Dict,
        requested_by: str = "system"
    ) -> ApprovalRequest:
        """Request approval for an action."""
        request = self.approval_workflow.create_approval_request(action, requested_by)
        
        # Log approval request
        self.audit_trail.log(
            action_id=action.get('action_id', 'unknown'),
            event_type="approval_request",
            event_description=f"Approval requested - {request.risk_level.value} risk",
            actor=requested_by,
            new_state={
                "request_id": request.request_id,
                "approval_level": request.approval_level.value,
                "approvers_required": request.approvers_required
            }
        )
        
        return request
    
    def approve_action(
        self,
        request_id: str,
        approver_id: str,
        approver_name: str,
        approver_role: str,
        decision: str,
        comments: str = ""
    ) -> Tuple[Approval, ApprovalRequest]:
        """Submit an approval decision."""
        approval, request = self.approval_workflow.submit_approval(
            request_id=request_id,
            approver_id=approver_id,
            approver_name=approver_name,
            approver_role=approver_role,
            decision=decision,
            comments=comments
        )
        
        # Log approval
        self.audit_trail.log(
            action_id=request.action_id,
            event_type="approval",
            event_description=f"Action {decision} by {approver_name}",
            actor=approver_id,
            actor_role=approver_role,
            old_state={"status": "pending"},
            new_state={
                "status": request.status,
                "decision": decision,
                "comments": comments
            }
        )
        
        return approval, request
    
    def execute_action(
        self,
        action: Dict,
        executed_by: str = "system",
        skip_validation: bool = False,
        skip_approval: bool = False,
        dry_run: bool = False
    ) -> ExecutionRecord:
        """Execute an action with full workflow."""
        action_id = action.get('action_id', 'unknown')
        
        # Step 1: Validation
        if not skip_validation:
            validation = self.validate_action(action)
            if not validation.can_execute:
                return ExecutionRecord(
                    execution_id=self.execution_engine._generate_execution_id(),
                    action_id=action_id,
                    action_title=action.get('title', 'Unknown'),
                    status=ExecutionStatus.VALIDATION_FAILED,
                    error=f"Validation failed: {', '.join(validation.blockers)}"
                )
        
        # Step 2: Check if approval required
        if not skip_approval and self.approval_workflow.requires_approval(action):
            request = self.request_approval(action, executed_by)
            return ExecutionRecord(
                execution_id=self.execution_engine._generate_execution_id(),
                action_id=action_id,
                action_title=action.get('title', 'Unknown'),
                status=ExecutionStatus.AWAITING_APPROVAL,
                result=f"Awaiting approval - Request ID: {request.request_id}"
            )
        
        # Step 3: Execute
        execution = self.execution_engine.execute_action(action, executed_by, dry_run)
        
        # Log execution
        self.audit_trail.log(
            action_id=action_id,
            event_type="execution",
            event_description=f"Action executed - {execution.status.value}",
            actor=executed_by,
            old_state={"status": "pending"},
            new_state={
                "status": execution.status.value,
                "result": execution.result,
                "execution_id": execution.execution_id
            }
        )
        
        return execution
    
    def rollback_execution(
        self,
        execution_id: str,
        rolled_back_by: str = "system"
    ) -> ExecutionRecord:
        """Rollback a previous execution."""
        rollback = self.execution_engine.rollback_action(execution_id, rolled_back_by)
        
        # Log rollback
        self.audit_trail.log(
            action_id=rollback.action_id,
            event_type="rollback",
            event_description=f"Execution {execution_id} rolled back",
            actor=rolled_back_by,
            old_state={"original_execution": execution_id},
            new_state={"rollback_execution": rollback.execution_id}
        )
        
        return rollback
    
    def process_actions(
        self,
        actions: List[Dict],
        executed_by: str = "system",
        auto_execute_low_risk: bool = True,
        dry_run: bool = False
    ) -> ExecutorResult:
        """Process multiple actions."""
        start_time = datetime.now()
        
        result = ExecutorResult(
            result_id=self._generate_result_id(),
            query=f"Process {len(actions)} actions"
        )
        
        for action in actions:
            result.actions_processed += 1
            
            # Validate
            validation = self.validate_action(action)
            result.validation_reports.append(validation)
            
            if not validation.can_execute:
                result.actions_failed += 1
                continue
            
            # Check approval requirement
            risk_level = self.approval_workflow.determine_risk_level(action)
            
            if risk_level == RiskLevel.LOW and auto_execute_low_risk:
                # Auto-execute low risk actions
                execution = self.execute_action(
                    action, executed_by, 
                    skip_validation=True, 
                    skip_approval=True,
                    dry_run=dry_run
                )
                result.execution_records.append(execution)
                
                if execution.status == ExecutionStatus.COMPLETED:
                    result.actions_executed += 1
                else:
                    result.actions_failed += 1
            else:
                # Create approval request
                request = self.request_approval(action, executed_by)
                result.approval_requests.append(request)
                result.actions_pending_approval += 1
        
        # Collect audit entries
        result.audit_entries = self.audit_trail.entries[-len(actions) * 3:]  # Approximate
        
        # Generate summary
        result.summary = self._generate_summary(result)
        result.recommendations = self._generate_recommendations(result)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def execute_from_query(self, query: str) -> ExecutorResult:
        """Execute actions based on natural language query."""
        start_time = datetime.now()
        query_lower = query.lower()
        
        result = ExecutorResult(
            result_id=self._generate_result_id(),
            query=query
        )
        
        if 'pending' in query_lower or 'approval' in query_lower:
            # Show pending approvals
            pending = self.approval_workflow.get_pending_requests()
            result.approval_requests = pending
            result.summary = f"Found {len(pending)} pending approval requests"
        
        elif 'status' in query_lower:
            # Show execution status
            history = list(self.execution_engine.execution_history.values())
            result.execution_records = history[-10:]  # Last 10
            result.summary = f"Last {len(result.execution_records)} executions"
        
        elif 'audit' in query_lower:
            # Show audit trail
            result.audit_entries = self.audit_trail.entries[-20:]
            result.summary = f"Last {len(result.audit_entries)} audit entries"
        
        elif 'rollback' in query_lower:
            # Info about rollback capability
            rollback_available = [
                e for e in self.execution_engine.execution_history.values()
                if e.rollback_available and e.status == ExecutionStatus.COMPLETED
            ]
            result.execution_records = rollback_available
            result.summary = f"{len(rollback_available)} executions available for rollback"
        
        else:
            result.summary = "Use 'pending approvals', 'execution status', 'audit trail', or 'rollback available'"
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result
    
    def _generate_summary(self, result: ExecutorResult) -> str:
        """Generate execution summary."""
        summary = f"""
EXECUTION SUMMARY
{'=' * 50}
Actions Processed: {result.actions_processed}
Actions Executed: {result.actions_executed}
Pending Approval: {result.actions_pending_approval}
Failed: {result.actions_failed}

VALIDATION:
  Passed: {len([v for v in result.validation_reports if v.can_execute])}
  Failed: {len([v for v in result.validation_reports if not v.can_execute])}
  Warnings: {sum(len(v.warnings) for v in result.validation_reports)}

EXECUTION RESULTS:
"""
        for exec_rec in result.execution_records:
            summary += f"  - {exec_rec.action_title}: {exec_rec.status.value}\n"
        
        if result.approval_requests:
            summary += f"\nPENDING APPROVALS:\n"
            for req in result.approval_requests:
                summary += f"  - [{req.request_id}] {req.action_title} ({req.risk_level.value})\n"
        
        return summary.strip()
    
    def _generate_recommendations(self, result: ExecutorResult) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if result.actions_pending_approval > 0:
            recommendations.append(
                f"Review and approve {result.actions_pending_approval} pending action(s)"
            )
        
        if result.actions_failed > 0:
            recommendations.append(
                f"Investigate {result.actions_failed} failed action(s) - check validation reports"
            )
        
        warnings = sum(len(v.warnings) for v in result.validation_reports)
        if warnings > 0:
            recommendations.append(
                f"Address {warnings} validation warning(s) before proceeding"
            )
        
        if not recommendations:
            recommendations.append("All actions processed successfully")
        
        return recommendations
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Main processing method for orchestrator integration."""
        
        # Check if context contains actions to process
        if context and 'actions' in context:
            result = self.process_actions(
                actions=context['actions'],
                executed_by=context.get('executed_by', 'system'),
                auto_execute_low_risk=context.get('auto_execute', True),
                dry_run=context.get('dry_run', False)
            )
        else:
            result = self.execute_from_query(query)
        
        return result.to_dict()


def get_executor_agent(llm_wrapper=None, audit_path: str = None) -> EnhancedExecutorAgent:
    """Factory function to get executor agent instance."""
    return EnhancedExecutorAgent(llm_wrapper=llm_wrapper, audit_path=audit_path)