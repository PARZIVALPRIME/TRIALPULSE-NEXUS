"""
TRIALPULSE NEXUS 10X - Causal Hypothesis Engine v1.2
Phase 4.4: Root cause analysis with evidence chains and confidence scoring

FIXES in v1.2:
- Fixed issue detection: columns contain 1/0 not True/False
- Fixed _has_issue to handle numeric values
- Fixed evidence gathering to use patient_issues.parquet columns
- Added proper count column detection

AUTHOR: TrialPulse Team
VERSION: 1.2
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Evidence:
    """Single piece of evidence supporting a hypothesis"""
    evidence_id: str
    source: str
    description: str
    value: Any
    expected_value: Any
    deviation: float
    weight: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def strength(self) -> float:
        return self.deviation * self.weight


@dataclass
class EvidenceChain:
    """Collection of evidence supporting a hypothesis"""
    chain_id: str
    evidences: List[Evidence] = field(default_factory=list)
    
    def add_evidence(self, evidence: Evidence):
        self.evidences.append(evidence)
    
    @property
    def total_strength(self) -> float:
        return sum(e.strength for e in self.evidences)
    
    @property
    def weighted_strength(self) -> float:
        if not self.evidences:
            return 0.0
        total_weight = sum(e.weight for e in self.evidences)
        if total_weight == 0:
            return 0.0
        return sum(e.strength for e in self.evidences) / total_weight


@dataclass
class Hypothesis:
    """A causal hypothesis for an issue"""
    hypothesis_id: str
    issue_type: str
    entity_id: str
    entity_type: str
    root_cause: str
    description: str
    mechanism: str
    evidence_chain: EvidenceChain
    confidence: float
    confidence_interval: Tuple[float, float]
    confounders: List[str]
    verification_steps: List[str]
    recommendations: List[str]
    priority: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'hypothesis_id': self.hypothesis_id,
            'issue_type': self.issue_type,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'root_cause': self.root_cause,
            'description': self.description,
            'mechanism': self.mechanism,
            'evidence_chain': {
                'chain_id': self.evidence_chain.chain_id,
                'evidences': [asdict(e) for e in self.evidence_chain.evidences],
                'total_strength': self.evidence_chain.total_strength,
                'weighted_strength': self.evidence_chain.weighted_strength
            },
            'confidence': self.confidence,
            'confidence_interval': list(self.confidence_interval),
            'confounders': self.confounders,
            'verification_steps': self.verification_steps,
            'recommendations': self.recommendations,
            'priority': self.priority,
            'created_at': self.created_at
        }


# =============================================================================
# ISSUE TYPES - Standard names used in patient_issues.parquet
# =============================================================================

ISSUE_TYPES = [
    'sdv_incomplete',
    'open_queries',
    'signature_gaps',
    'broken_signatures',
    'sae_dm_pending',
    'sae_safety_pending',
    'missing_visits',
    'missing_pages',
    'lab_issues',
    'edrr_issues',
    'inactivated_forms',
    'meddra_uncoded',
    'whodrug_uncoded',
    'high_query_volume'
]


# =============================================================================
# ROOT CAUSE TEMPLATES
# =============================================================================

ROOT_CAUSE_TEMPLATES = {
    'sdv_incomplete': [
        {
            'root_cause': 'CRA Resource Constraint',
            'description': 'Insufficient CRA capacity to complete SDV for assigned sites',
            'mechanism': 'High patient-to-CRA ratio → Delayed monitoring visits → SDV backlog accumulation',
            'confounders': ['Site accessibility', 'Travel restrictions', 'Holiday periods'],
            'verification': ['Check CRA assignment ratio', 'Review monitoring visit logs', 'Compare with peer CRA performance'],
            'priority_if_confirmed': 'High'
        },
        {
            'root_cause': 'Source Document Unavailability',
            'description': 'Source documents not available during monitoring visits',
            'mechanism': 'Missing/incomplete source docs → CRA cannot verify → SDV pending',
            'confounders': ['COVID restrictions', 'Site staff changes', 'EMR access issues'],
            'verification': ['Review query types', 'Check source doc queries percentage', 'Site communication logs'],
            'priority_if_confirmed': 'Medium'
        }
    ],
    'open_queries': [
        {
            'root_cause': 'Site Response Delay',
            'description': 'Site not responding to queries within expected timeframe',
            'mechanism': 'Low site engagement → Queries age → Open query accumulation',
            'confounders': ['Site workload', 'Staff turnover', 'Language barriers'],
            'verification': ['Calculate average response time', 'Compare with site SLA', 'Review escalation history'],
            'priority_if_confirmed': 'High'
        },
        {
            'root_cause': 'Coordinator Overload',
            'description': 'Site coordinator managing too many patients',
            'mechanism': 'High patient load → Prioritization challenges → Query response delays',
            'confounders': ['Coordinator experience', 'Part-time staff', 'Multiple studies'],
            'verification': ['Calculate coordinator ratio', 'Compare with site capacity', 'Review escalation patterns'],
            'priority_if_confirmed': 'High'
        }
    ],
    'signature_gaps': [
        {
            'root_cause': 'PI Absence',
            'description': 'Principal Investigator unavailable for signatures',
            'mechanism': 'PI travel/leave → No delegation → Signature accumulation',
            'confounders': ['Conference schedule', 'Clinical duties', 'Multiple studies'],
            'verification': ['Check PI signature pattern', 'Review delegation log', 'Compare with sub-I activity'],
            'priority_if_confirmed': 'High'
        },
        {
            'root_cause': 'Batch Signing Pattern',
            'description': 'Site accumulates forms and signs in batches',
            'mechanism': 'Efficiency preference → Delayed signing → Periodic backlog',
            'confounders': ['Site workflow', 'PI preference', 'EDC access frequency'],
            'verification': ['Analyze signature timestamps', 'Check for clustering', 'Review sign-off SLA'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'broken_signatures': [
        {
            'root_cause': 'Post-Signature Data Changes',
            'description': 'Data modified after PI signature requiring re-signature',
            'mechanism': 'Data correction after sign → Signature invalidated → Re-sign required',
            'confounders': ['Data entry quality', 'Query volume', 'Protocol amendments'],
            'verification': ['Review audit trail', 'Identify change triggers', 'Check timing of changes vs signatures'],
            'priority_if_confirmed': 'Medium'
        },
        {
            'root_cause': 'Query Resolution Process',
            'description': 'Query resolution triggering signature breaks',
            'mechanism': 'Query answered → Form updated → Previous signature broken → Re-sign queue',
            'confounders': ['Query timing', 'Auto-query triggers', 'Edit check configuration'],
            'verification': ['Correlate queries with signature breaks', 'Review workflow sequence', 'Check edit check logs'],
            'priority_if_confirmed': 'Medium'
        }
    ],
    'sae_dm_pending': [
        {
            'root_cause': 'Data Manager Workload',
            'description': 'High SAE volume exceeding DM capacity',
            'mechanism': 'SAE surge → DM backlog → Reconciliation delays',
            'confounders': ['Study phase', 'Patient population', 'Adverse event profile'],
            'verification': ['Calculate SAE/DM ratio', 'Check reconciliation SLA', 'Compare with baseline'],
            'priority_if_confirmed': 'Critical'
        },
        {
            'root_cause': 'Cross-System Mismatch',
            'description': 'Discrepancies between safety database and EDC',
            'mechanism': 'Data entry differences → Reconciliation failures → Manual review required',
            'confounders': ['System integration', 'Data entry timing', 'Form mapping'],
            'verification': ['Review mismatch categories', 'Check integration logs', 'Analyze error patterns'],
            'priority_if_confirmed': 'High'
        }
    ],
    'sae_safety_pending': [
        {
            'root_cause': 'Medical Review Bottleneck',
            'description': 'Safety physician capacity limiting SAE review',
            'mechanism': 'Limited medical reviewer time → Pending queue grows → SLA risk',
            'confounders': ['Case complexity', 'Additional info requests', 'Causality assessment difficulty'],
            'verification': ['Calculate cases per reviewer', 'Check review turnaround', 'Compare with target SLA'],
            'priority_if_confirmed': 'Critical'
        }
    ],
    'missing_visits': [
        {
            'root_cause': 'Patient Non-Compliance',
            'description': 'Patients missing scheduled visits',
            'mechanism': 'Patient no-show → Visit not completed → Data gaps',
            'confounders': ['Patient population', 'Visit burden', 'Transportation issues'],
            'verification': ['Check patient compliance rate', 'Review retention patterns', 'Analyze missed visit reasons'],
            'priority_if_confirmed': 'Medium'
        },
        {
            'root_cause': 'Visit Window Violation',
            'description': 'Visits conducted outside protocol windows',
            'mechanism': 'Scheduling delays → Window violation → Visit marked missing in EDC',
            'confounders': ['Site scheduling', 'Patient availability', 'Holiday periods'],
            'verification': ['Review visit date calculations', 'Check window definitions', 'Compare actual vs expected'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'missing_pages': [
        {
            'root_cause': 'Data Entry Backlog',
            'description': 'Site behind on CRF data entry',
            'mechanism': 'High enrollment → Data entry capacity exceeded → Pages pending',
            'confounders': ['Site staffing', 'Protocol complexity', 'Competing priorities'],
            'verification': ['Calculate data entry lag', 'Check pages per patient', 'Review entry patterns'],
            'priority_if_confirmed': 'High'
        },
        {
            'root_cause': 'Conditional Form Logic',
            'description': 'Forms not triggered due to missing prerequisites',
            'mechanism': 'Prerequisite data missing → Conditional form not triggered → Appears as missing',
            'confounders': ['EDC configuration', 'Form dependencies', 'Protocol design'],
            'verification': ['Check form trigger logic', 'Review prerequisite completion', 'Analyze form dependencies'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'lab_issues': [
        {
            'root_cause': 'Lab Range Configuration',
            'description': 'Lab normal ranges not configured in EDC',
            'mechanism': 'Missing lab ranges → Values cannot be flagged → Manual review required',
            'confounders': ['Central vs local labs', 'Lab vendor changes', 'Protocol amendments'],
            'verification': ['Review lab configuration', 'Check range completeness', 'Compare with lab vendor data'],
            'priority_if_confirmed': 'Medium'
        }
    ],
    'edrr_issues': [
        {
            'root_cause': 'Vendor Data Mismatch',
            'description': 'Third-party vendor data not reconciling with EDC',
            'mechanism': 'Data format/timing differences → Reconciliation failures → Issues flagged',
            'confounders': ['Vendor systems', 'Transfer frequency', 'Data mapping complexity'],
            'verification': ['Review reconciliation errors', 'Check vendor data timing', 'Analyze mismatch types'],
            'priority_if_confirmed': 'Medium'
        }
    ],
    'inactivated_forms': [
        {
            'root_cause': 'Protocol Deviation Cleanup',
            'description': 'Forms inactivated due to protocol deviations',
            'mechanism': 'Deviation occurred → Form data invalid → Form inactivated',
            'confounders': ['Protocol complexity', 'Amendment timing', 'Training gaps'],
            'verification': ['Review inactivation reasons', 'Correlate with deviations', 'Check approval workflow'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'meddra_uncoded': [
        {
            'root_cause': 'Coding Backlog',
            'description': 'Medical coder capacity exceeded by term volume',
            'mechanism': 'High AE volume → Coding queue grows → Uncoded terms accumulate',
            'confounders': ['Study phase', 'Patient population', 'Protocol amendments'],
            'verification': ['Calculate terms per coder', 'Check coding SLA', 'Review prioritization'],
            'priority_if_confirmed': 'Medium'
        },
        {
            'root_cause': 'Ambiguous Terms',
            'description': 'Verbatim terms requiring medical review for coding',
            'mechanism': 'Unclear verbatim → Auto-code fails → Manual review queue',
            'confounders': ['Language', 'Entry quality', 'Dictionary version'],
            'verification': ['Check auto-code success rate', 'Review manual queue', 'Analyze term patterns'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'whodrug_uncoded': [
        {
            'root_cause': 'Non-Standard Drug Names',
            'description': 'Drug names entered in non-standard format',
            'mechanism': 'Brand/generic confusion → Dictionary lookup fails → Manual coding needed',
            'confounders': ['Regional drug names', 'Generic vs brand', 'Spelling variations'],
            'verification': ['Review failed auto-codes', 'Check dictionary coverage', 'Analyze naming patterns'],
            'priority_if_confirmed': 'Low'
        }
    ],
    'high_query_volume': [
        {
            'root_cause': 'Data Entry Quality Issues',
            'description': 'Poor data entry generating excessive queries',
            'mechanism': 'Training gaps → Entry errors → Edit check failures → Queries generated',
            'confounders': ['Site experience', 'Protocol complexity', 'EDC usability'],
            'verification': ['Analyze query types', 'Review edit check triggers', 'Check training records'],
            'priority_if_confirmed': 'High'
        },
        {
            'root_cause': 'Overly Sensitive Edit Checks',
            'description': 'Edit checks generating false positive queries',
            'mechanism': 'Aggressive edit check logic → Valid data flagged → Query overload',
            'confounders': ['Edit check tuning', 'Population variability', 'Protocol amendments'],
            'verification': ['Review edit check logic', 'Calculate false positive rate', 'Check resolution patterns'],
            'priority_if_confirmed': 'Medium'
        }
    ]
}


# =============================================================================
# CAUSAL HYPOTHESIS ENGINE
# =============================================================================

class CausalHypothesisEngine:
    """
    Engine for generating causal hypotheses with evidence chains.
    """
    
    def __init__(self, data_dir: Path = None):
        """Initialize the Causal Hypothesis Engine"""
        self.data_dir = Path(data_dir) if data_dir else Path('data/processed')
        self.templates = ROOT_CAUSE_TEMPLATES
        
        # Data sources
        self.patient_data = None
        self.site_data = None
        self.pattern_matches = None
        
        # Column names (from patient_issues.parquet)
        self.issue_columns = {}  # issue_type -> column name
        self.count_columns = {}  # issue_type -> column name
        self.severity_columns = {}  # issue_type -> column name
        
        # Statistics
        self.stats = {
            'hypotheses_generated': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'by_issue_type': defaultdict(int),
            'by_root_cause': defaultdict(int)
        }
        
        logger.info("CausalHypothesisEngine initialized")
    
    def _detect_columns(self, df: pd.DataFrame):
        """Detect issue, count, and severity columns"""
        for issue_type in ISSUE_TYPES:
            # Issue column: issue_{type}
            issue_col = f'issue_{issue_type}'
            if issue_col in df.columns:
                self.issue_columns[issue_type] = issue_col
            
            # Count column: count_{type}
            count_col = f'count_{issue_type}'
            if count_col in df.columns:
                self.count_columns[issue_type] = count_col
            
            # Severity column: severity_{type}
            severity_col = f'severity_{issue_type}'
            if severity_col in df.columns:
                self.severity_columns[issue_type] = severity_col
    
    def load_data(self) -> bool:
        """Load all required data sources"""
        logger.info("Loading data sources...")
        
        try:
            # Load patient_issues.parquet (primary source)
            patient_file = self.data_dir / 'analytics' / 'patient_issues.parquet'
            if patient_file.exists():
                self.patient_data = pd.read_parquet(patient_file)
                logger.info(f"  Loaded patient_issues: {len(self.patient_data)} rows")
                
                # Detect columns
                self._detect_columns(self.patient_data)
                logger.info(f"  Issue columns detected: {len(self.issue_columns)}")
                logger.info(f"  Count columns detected: {len(self.count_columns)}")
                logger.info(f"  Severity columns detected: {len(self.severity_columns)}")
            else:
                logger.error(f"  Patient file not found: {patient_file}")
                return False
            
            # Load site data
            site_file = self.data_dir / 'metrics' / 'site_metrics.parquet'
            if site_file.exists():
                self.site_data = pd.read_parquet(site_file)
                logger.info(f"  Loaded site_metrics: {len(self.site_data)} rows")
            
            # Load pattern matches
            pattern_file = self.data_dir / 'analytics' / 'pattern_library' / 'pattern_matches.parquet'
            if pattern_file.exists():
                self.pattern_matches = pd.read_parquet(pattern_file)
                logger.info(f"  Loaded pattern_matches: {len(self.pattern_matches)} rows")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _has_issue(self, patient_row: pd.Series, issue_type: str) -> bool:
        """Check if patient has a specific issue (handles 1/0 or True/False)"""
        if issue_type not in self.issue_columns:
            return False
        
        col = self.issue_columns[issue_type]
        val = patient_row.get(col, 0)
        
        # Handle various truthy values
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val > 0
        return bool(val)
    
    def _get_count(self, patient_row: pd.Series, issue_type: str) -> float:
        """Get issue count for a patient"""
        if issue_type not in self.count_columns:
            return 0.0
        
        col = self.count_columns[issue_type]
        val = patient_row.get(col, 0)
        
        if pd.isna(val):
            return 0.0
        return float(val)
    
    def _get_severity(self, patient_row: pd.Series, issue_type: str) -> float:
        """Get issue severity for a patient"""
        if issue_type not in self.severity_columns:
            return 0.0
        
        col = self.severity_columns[issue_type]
        val = patient_row.get(col, 0)
        
        if pd.isna(val):
            return 0.0
        return float(val)
    
    def _get_patients_with_issues(self) -> pd.DataFrame:
        """Get patients that have at least one issue"""
        if self.patient_data is None:
            return pd.DataFrame()
        
        # Use has_any_issue column if available
        if 'has_any_issue' in self.patient_data.columns:
            mask = self.patient_data['has_any_issue'] > 0
            return self.patient_data[mask]
        
        # Fallback: check all issue columns
        issue_cols = list(self.issue_columns.values())
        if not issue_cols:
            return pd.DataFrame()
        
        mask = (self.patient_data[issue_cols] > 0).any(axis=1)
        return self.patient_data[mask]
    
    def _gather_evidence(self, 
                         issue_type: str, 
                         patient_row: pd.Series,
                         template: Dict) -> EvidenceChain:
        """Gather evidence for a hypothesis"""
        chain = EvidenceChain(
            chain_id=f"EC-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
        )
        
        evidence_id = 0
        
        # Evidence 1: Issue is present
        if self._has_issue(patient_row, issue_type):
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='patient_issues',
                description=f"Issue {issue_type} is present",
                value=True,
                expected_value=False,
                deviation=1.0,
                weight=0.5
            ))
            evidence_id += 1
        
        # Evidence 2: Issue count
        count = self._get_count(patient_row, issue_type)
        if count > 0:
            # Normalize: 1-2 = low, 3-5 = medium, 5+ = high
            if count <= 2:
                deviation = 0.3
            elif count <= 5:
                deviation = 0.6
            else:
                deviation = min(count / 10, 1.0)
            
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='patient_issues',
                description=f"Issue count for {issue_type}",
                value=int(count),
                expected_value=0,
                deviation=deviation,
                weight=0.8
            ))
            evidence_id += 1
        
        # Evidence 3: Issue severity
        severity = self._get_severity(patient_row, issue_type)
        if severity > 0:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='patient_issues',
                description=f"Severity score for {issue_type}",
                value=round(severity, 2),
                expected_value=0,
                deviation=min(severity, 1.0),
                weight=0.7
            ))
            evidence_id += 1
        
        # Evidence 4: Total issues (context)
        total_issues = patient_row.get('total_issues', 0)
        if pd.notna(total_issues) and total_issues > 1:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='patient_issues',
                description="Multiple issues present (compound complexity)",
                value=int(total_issues),
                expected_value=0,
                deviation=min(total_issues / 5, 0.8),
                weight=0.4
            ))
            evidence_id += 1
        
        # Evidence 5: Priority tier
        priority_tier = patient_row.get('priority_tier', '')
        if priority_tier in ['critical', 'high']:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='patient_issues',
                description="High priority classification",
                value=priority_tier,
                expected_value='none',
                deviation=0.9 if priority_tier == 'critical' else 0.7,
                weight=0.6
            ))
            evidence_id += 1
        
        # Issue-specific evidence
        if issue_type == 'sdv_incomplete' and count > 3:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='analysis',
                description="Significant SDV backlog (>3 CRFs)",
                value=int(count),
                expected_value=0,
                deviation=0.8,
                weight=0.7
            ))
            evidence_id += 1
        
        elif issue_type == 'sae_dm_pending' and count > 0:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='analysis',
                description="Safety reconciliation pending (regulatory impact)",
                value=int(count),
                expected_value=0,
                deviation=0.9,
                weight=0.9
            ))
            evidence_id += 1
        
        elif issue_type == 'signature_gaps' and count > 2:
            chain.add_evidence(Evidence(
                evidence_id=f"E-{evidence_id:03d}",
                source='analysis',
                description="Multiple unsigned CRFs (PI attention needed)",
                value=int(count),
                expected_value=0,
                deviation=0.7,
                weight=0.75
            ))
            evidence_id += 1
        
        return chain
    
    def _calculate_confidence(self, evidence_chain: EvidenceChain) -> Tuple[float, Tuple[float, float]]:
        """Calculate confidence score and interval"""
        if not evidence_chain.evidences:
            return 0.0, (0.0, 0.0)
        
        # Weighted average of evidence strengths
        total_weight = sum(e.weight for e in evidence_chain.evidences)
        if total_weight == 0:
            return 0.0, (0.0, 0.0)
        
        weighted_sum = sum(e.strength for e in evidence_chain.evidences)
        base_confidence = weighted_sum / total_weight
        
        # Boost confidence with more evidence (up to 20% boost for 5+ pieces)
        n_evidence = len(evidence_chain.evidences)
        evidence_boost = min(0.2, n_evidence * 0.04)
        
        confidence = min(base_confidence + evidence_boost, 1.0)
        
        # Uncertainty decreases with more evidence
        uncertainty = 0.15 / np.sqrt(max(n_evidence, 1))
        lower = max(0.0, confidence - uncertainty)
        upper = min(1.0, confidence + uncertainty)
        
        return round(confidence, 3), (round(lower, 3), round(upper, 3))
    
    def _assign_priority(self, confidence: float, issue_type: str, template: Dict) -> str:
        """Assign priority based on confidence and issue criticality"""
        base_priority = template.get('priority_if_confirmed', 'Medium')
        
        # Safety issues always elevated
        if issue_type in ['sae_dm_pending', 'sae_safety_pending']:
            if confidence >= 0.5:
                return 'Critical'
            return 'High'
        
        if confidence < 0.3:
            return 'Low'
        elif confidence < 0.5:
            return 'Medium' if base_priority in ['Critical', 'High'] else 'Low'
        elif confidence < 0.7:
            return base_priority if base_priority != 'Critical' else 'High'
        else:
            return base_priority
    
    def _generate_recommendations(self, template: Dict, confidence: float, issue_type: str) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        root_cause = template['root_cause']
        
        # Issue-specific recommendations
        if issue_type == 'sdv_incomplete':
            recommendations.append("Schedule SDV completion during next monitoring visit")
            recommendations.append("Ensure source documents are available")
        elif issue_type == 'open_queries':
            recommendations.append("Send query reminder to site coordinator")
            recommendations.append("Review query aging and escalate if >14 days")
        elif issue_type == 'signature_gaps':
            recommendations.append("Request PI to complete pending signatures")
            recommendations.append("Verify delegation log is current")
        elif issue_type == 'sae_dm_pending':
            recommendations.append("Prioritize SAE reconciliation (regulatory impact)")
            recommendations.append("Verify safety database sync status")
        elif issue_type == 'broken_signatures':
            recommendations.append("Review data changes triggering signature breaks")
            recommendations.append("Batch re-signature request to PI")
        
        # Root cause based recommendations
        if 'Resource' in root_cause or 'Workload' in root_cause:
            recommendations.append("Review workload distribution and capacity")
        if 'Delay' in root_cause or 'Response' in root_cause:
            recommendations.append("Implement escalation protocol")
        if 'Absence' in root_cause:
            recommendations.append("Ensure backup/delegation is in place")
        
        # Add verification step if low confidence
        if confidence < 0.5 and template.get('verification'):
            recommendations.insert(0, f"VERIFY FIRST: {template['verification'][0]}")
        
        # Ensure we have at least 2 recommendations
        if len(recommendations) < 2:
            recommendations.append("Monitor issue and track resolution progress")
        
        return recommendations[:4]
    
    def generate_hypothesis(self,
                           issue_type: str,
                           patient_row: pd.Series,
                           template_index: int = 0) -> Optional[Hypothesis]:
        """Generate a single hypothesis for a patient issue"""
        if issue_type not in self.templates:
            return None
        
        templates = self.templates[issue_type]
        if template_index >= len(templates):
            template_index = 0
        
        template = templates[template_index]
        patient_id = patient_row.get('patient_key', 'unknown')
        
        # Gather evidence
        evidence_chain = self._gather_evidence(issue_type, patient_row, template)
        
        # Skip if no meaningful evidence
        if len(evidence_chain.evidences) < 2:
            return None
        
        # Calculate confidence
        confidence, ci = self._calculate_confidence(evidence_chain)
        
        # Create hypothesis
        hypothesis = Hypothesis(
            hypothesis_id=f"HYP-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
            issue_type=issue_type,
            entity_id=patient_id,
            entity_type='patient',
            root_cause=template['root_cause'],
            description=template['description'],
            mechanism=template['mechanism'],
            evidence_chain=evidence_chain,
            confidence=confidence,
            confidence_interval=ci,
            confounders=template['confounders'],
            verification_steps=template['verification'],
            recommendations=self._generate_recommendations(template, confidence, issue_type),
            priority=self._assign_priority(confidence, issue_type, template)
        )
        
        # Update stats
        self.stats['hypotheses_generated'] += 1
        if confidence >= 0.7:
            self.stats['high_confidence'] += 1
        elif confidence >= 0.4:
            self.stats['medium_confidence'] += 1
        else:
            self.stats['low_confidence'] += 1
        self.stats['by_issue_type'][issue_type] += 1
        self.stats['by_root_cause'][template['root_cause']] += 1
        
        return hypothesis
    
    def generate_all_hypotheses_for_patient(self, patient_row: pd.Series) -> List[Hypothesis]:
        """Generate all hypotheses for a patient"""
        hypotheses = []
        
        for issue_type in ISSUE_TYPES:
            if self._has_issue(patient_row, issue_type):
                # Generate hypothesis for each template
                for i in range(len(self.templates.get(issue_type, []))):
                    h = self.generate_hypothesis(issue_type, patient_row, i)
                    if h:
                        hypotheses.append(h)
        
        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses
    
    def analyze_population(self, sample_size: int = 100) -> List[Hypothesis]:
        """Analyze a sample of patients with issues"""
        all_hypotheses = []
        
        patients_with_issues = self._get_patients_with_issues()
        
        if len(patients_with_issues) == 0:
            logger.warning("No patients with issues found")
            return all_hypotheses
        
        logger.info(f"  Found {len(patients_with_issues)} patients with issues")
        
        # Sample patients
        n = min(sample_size, len(patients_with_issues))
        sample = patients_with_issues.sample(n=n, random_state=42)
        
        logger.info(f"  Analyzing {n} patients...")
        
        for idx, row in sample.iterrows():
            hypotheses = self.generate_all_hypotheses_for_patient(row)
            all_hypotheses.extend(hypotheses)
        
        logger.info(f"  Generated {len(all_hypotheses)} hypotheses")
        
        return all_hypotheses
    
    def format_hypothesis_report(self, hypothesis: Hypothesis) -> str:
        """Format hypothesis as human-readable report"""
        report = []
        report.append("=" * 70)
        report.append("CAUSAL HYPOTHESIS REPORT")
        report.append("=" * 70)
        report.append(f"ID: {hypothesis.hypothesis_id}")
        report.append(f"Patient: {hypothesis.entity_id}")
        report.append(f"Created: {hypothesis.created_at}")
        report.append("")
        report.append(f"ISSUE TYPE: {hypothesis.issue_type}")
        report.append(f"PRIORITY: {hypothesis.priority}")
        report.append("")
        report.append("ROOT CAUSE HYPOTHESIS:")
        report.append(f"  {hypothesis.root_cause}")
        report.append("")
        report.append("DESCRIPTION:")
        report.append(f"  {hypothesis.description}")
        report.append("")
        report.append("MECHANISM:")
        report.append(f"  {hypothesis.mechanism}")
        report.append("")
        report.append(f"CONFIDENCE: {hypothesis.confidence:.1%} (CI: {hypothesis.confidence_interval[0]:.1%} - {hypothesis.confidence_interval[1]:.1%})")
        report.append("")
        report.append("EVIDENCE CHAIN:")
        for e in hypothesis.evidence_chain.evidences:
            report.append(f"  [{e.source}] {e.description}: {e.value} (expected: {e.expected_value})")
        report.append(f"  Total Strength: {hypothesis.evidence_chain.total_strength:.2f}")
        report.append("")
        report.append("POTENTIAL CONFOUNDERS:")
        for c in hypothesis.confounders:
            report.append(f"  - {c}")
        report.append("")
        report.append("VERIFICATION STEPS:")
        for i, v in enumerate(hypothesis.verification_steps, 1):
            report.append(f"  {i}. {v}")
        report.append("")
        report.append("RECOMMENDATIONS:")
        for i, r in enumerate(hypothesis.recommendations, 1):
            report.append(f"  {i}. {r}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_results(self, hypotheses: List[Hypothesis], output_dir: Path) -> Dict[str, str]:
        """Save hypotheses to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_saved = {}
        
        # Save as JSON
        records = [h.to_dict() for h in hypotheses]
        json_file = output_dir / 'causal_hypotheses.json'
        with open(json_file, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        files_saved['json'] = str(json_file)
        
        # Save summary CSV
        summary_records = []
        for h in hypotheses:
            summary_records.append({
                'hypothesis_id': h.hypothesis_id,
                'patient_id': h.entity_id,
                'issue_type': h.issue_type,
                'root_cause': h.root_cause,
                'confidence': h.confidence,
                'confidence_lower': h.confidence_interval[0],
                'confidence_upper': h.confidence_interval[1],
                'priority': h.priority,
                'evidence_count': len(h.evidence_chain.evidences),
                'evidence_strength': round(h.evidence_chain.total_strength, 2),
                'top_recommendation': h.recommendations[0] if h.recommendations else ''
            })
        
        df = pd.DataFrame(summary_records)
        csv_file = output_dir / 'causal_hypotheses_summary.csv'
        df.to_csv(csv_file, index=False)
        files_saved['summary_csv'] = str(csv_file)
        
        # Save high confidence hypotheses
        high_conf = [h for h in hypotheses if h.confidence >= 0.7]
        if high_conf:
            high_conf_file = output_dir / 'high_confidence_hypotheses.csv'
            high_df = pd.DataFrame([{
                'patient_id': h.entity_id,
                'issue_type': h.issue_type,
                'root_cause': h.root_cause,
                'confidence': h.confidence,
                'priority': h.priority,
                'recommendation': h.recommendations[0] if h.recommendations else ''
            } for h in high_conf])
            high_df.to_csv(high_conf_file, index=False)
            files_saved['high_confidence'] = str(high_conf_file)
        
        # Save statistics
        stats_file = output_dir / 'causal_hypothesis_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'total_hypotheses': self.stats['hypotheses_generated'],
                'high_confidence': self.stats['high_confidence'],
                'medium_confidence': self.stats['medium_confidence'],
                'low_confidence': self.stats['low_confidence'],
                'by_issue_type': dict(self.stats['by_issue_type']),
                'by_root_cause': dict(self.stats['by_root_cause']),
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        files_saved['stats'] = str(stats_file)
        
        logger.info(f"Saved {len(hypotheses)} hypotheses to {output_dir}")
        
        return files_saved
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_hypotheses': self.stats['hypotheses_generated'],
            'confidence_distribution': {
                'high (>=70%)': self.stats['high_confidence'],
                'medium (40-70%)': self.stats['medium_confidence'],
                'low (<40%)': self.stats['low_confidence']
            },
            'by_issue_type': dict(self.stats['by_issue_type']),
            'top_root_causes': dict(sorted(
                self.stats['by_root_cause'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'template_coverage': len(self.templates),
            'total_templates': sum(len(v) for v in self.templates.values())
        }


def main():
    """Test the engine"""
    print("Testing CausalHypothesisEngine v1.2...")
    
    engine = CausalHypothesisEngine()
    if not engine.load_data():
        print("Failed to load data")
        return
    
    hypotheses = engine.analyze_population(sample_size=50)
    print(f"Generated {len(hypotheses)} hypotheses")
    
    if hypotheses:
        print("\nTop hypothesis:")
        print(engine.format_hypothesis_report(hypotheses[0]))
    
    return engine


if __name__ == "__main__":
    main()