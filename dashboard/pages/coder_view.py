"""
TRIALPULSE NEXUS 10X - Phase 7.8
Coder Workbench Dashboard v1.0

A specialized workbench for Medical Coders with:
- Confidence-sorted coding queue
- Batch approval for high-confidence terms
- Dictionary search (MedDRA/WHODrug)
- Expert escalation workflow
- Coding statistics and productivity metrics

Author: TrialPulse Team
Date: 2026-01-01
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import re

# =============================================================================
# CONFIGURATION & ENUMS
# =============================================================================

class CodingType(Enum):
    """Type of medical coding"""
    MEDDRA = "meddra"
    WHODRUG = "whodrug"

class CodingStatus(Enum):
    """Status of coding item"""
    PENDING = "pending"
    AUTO_CODED = "auto_coded"
    MANUALLY_CODED = "manually_coded"
    NEEDS_REVIEW = "needs_review"
    ESCALATED = "escalated"
    APPROVED = "approved"
    REJECTED = "rejected"

class ConfidenceLevel(Enum):
    """Confidence level for auto-coding"""
    VERY_HIGH = "very_high"    # >= 95%
    HIGH = "high"              # 85-95%
    MEDIUM = "medium"          # 70-85%
    LOW = "low"                # 50-70%
    VERY_LOW = "very_low"      # < 50%

class EscalationReason(Enum):
    """Reasons for escalation"""
    AMBIGUOUS_TERM = "ambiguous_term"
    MULTIPLE_MATCHES = "multiple_matches"
    NO_MATCH = "no_match"
    MEDICAL_REVIEW = "medical_review"
    NEW_TERM = "new_term"
    COMPLEX_CASE = "complex_case"

# Color schemes
CONFIDENCE_COLORS = {
    'very_high': '#27ae60',
    'high': '#2ecc71',
    'medium': '#f39c12',
    'low': '#e67e22',
    'very_low': '#e74c3c'
}

STATUS_COLORS = {
    'pending': '#95a5a6',
    'auto_coded': '#3498db',
    'manually_coded': '#9b59b6',
    'needs_review': '#f39c12',
    'escalated': '#e74c3c',
    'approved': '#27ae60',
    'rejected': '#c0392b'
}

CODING_TYPE_COLORS = {
    'meddra': '#3498db',
    'whodrug': '#9b59b6'
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CodingItem:
    """Represents a single term to be coded"""
    item_id: str
    verbatim_term: str
    coding_type: CodingType
    status: CodingStatus
    confidence: float
    suggested_code: Optional[str] = None
    suggested_term: Optional[str] = None
    patient_id: str = ""
    study_id: str = ""
    site_id: str = ""
    form_name: str = ""
    field_name: str = ""
    context: str = ""
    alternatives: List[Dict] = field(default_factory=list)
    coded_by: Optional[str] = None
    coded_at: Optional[datetime] = None
    escalated_to: Optional[str] = None
    escalation_reason: Optional[EscalationReason] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    @property
    def confidence_color(self) -> str:
        return CONFIDENCE_COLORS.get(self.confidence_level.value, '#95a5a6')

@dataclass
class DictionaryEntry:
    """Represents a dictionary entry (MedDRA or WHODrug)"""
    code: str
    term: str
    dictionary_type: CodingType
    level: str  # PT, LLT, SOC for MedDRA; ATC for WHODrug
    parent_code: Optional[str] = None
    parent_term: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    
@dataclass
class EscalationRequest:
    """Represents an escalation request"""
    request_id: str
    item_id: str
    verbatim_term: str
    coding_type: CodingType
    reason: EscalationReason
    escalated_by: str
    escalated_to: str
    escalated_at: datetime
    status: str  # pending, resolved, cancelled
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    notes: str = ""

@dataclass
class CodingStats:
    """Coding statistics"""
    total_pending: int
    total_coded_today: int
    total_escalated: int
    auto_approval_rate: float
    avg_confidence: float
    meddra_pending: int
    whodrug_pending: int
    high_confidence_pending: int
    productivity_trend: str

# =============================================================================
# MOCK DICTIONARY DATA
# =============================================================================

MEDDRA_DICTIONARY = [
    {"code": "10019211", "term": "Headache", "level": "PT", "soc": "Nervous system disorders"},
    {"code": "10028813", "term": "Nausea", "level": "PT", "soc": "Gastrointestinal disorders"},
    {"code": "10047700", "term": "Vomiting", "level": "PT", "soc": "Gastrointestinal disorders"},
    {"code": "10012735", "term": "Diarrhoea", "level": "PT", "soc": "Gastrointestinal disorders"},
    {"code": "10016558", "term": "Fatigue", "level": "PT", "soc": "General disorders"},
    {"code": "10013573", "term": "Dizziness", "level": "PT", "soc": "Nervous system disorders"},
    {"code": "10003239", "term": "Arthralgia", "level": "PT", "soc": "Musculoskeletal disorders"},
    {"code": "10034835", "term": "Pharyngitis", "level": "PT", "soc": "Respiratory disorders"},
    {"code": "10037660", "term": "Pyrexia", "level": "PT", "soc": "General disorders"},
    {"code": "10011224", "term": "Cough", "level": "PT", "soc": "Respiratory disorders"},
    {"code": "10002855", "term": "Anxiety", "level": "PT", "soc": "Psychiatric disorders"},
    {"code": "10022891", "term": "Insomnia", "level": "PT", "soc": "Psychiatric disorders"},
    {"code": "10037844", "term": "Rash", "level": "PT", "soc": "Skin disorders"},
    {"code": "10035664", "term": "Pruritus", "level": "PT", "soc": "Skin disorders"},
    {"code": "10000081", "term": "Abdominal pain", "level": "PT", "soc": "Gastrointestinal disorders"},
    {"code": "10005886", "term": "Back pain", "level": "PT", "soc": "Musculoskeletal disorders"},
    {"code": "10029354", "term": "Myalgia", "level": "PT", "soc": "Musculoskeletal disorders"},
    {"code": "10002034", "term": "Anaemia", "level": "PT", "soc": "Blood disorders"},
    {"code": "10020772", "term": "Hypertension", "level": "PT", "soc": "Vascular disorders"},
    {"code": "10021097", "term": "Hypotension", "level": "PT", "soc": "Vascular disorders"},
]

WHODRUG_DICTIONARY = [
    {"code": "N02BE01", "term": "Paracetamol", "atc": "N02BE", "category": "Anilides"},
    {"code": "N02BA01", "term": "Acetylsalicylic acid", "atc": "N02BA", "category": "Salicylic acid"},
    {"code": "M01AE01", "term": "Ibuprofen", "atc": "M01AE", "category": "Propionic acid derivatives"},
    {"code": "C09AA01", "term": "Captopril", "atc": "C09AA", "category": "ACE inhibitors"},
    {"code": "C07AB02", "term": "Metoprolol", "atc": "C07AB", "category": "Beta blockers"},
    {"code": "A02BC01", "term": "Omeprazole", "atc": "A02BC", "category": "Proton pump inhibitors"},
    {"code": "C10AA01", "term": "Simvastatin", "atc": "C10AA", "category": "HMG CoA reductase inhibitors"},
    {"code": "A10BA02", "term": "Metformin", "atc": "A10BA", "category": "Biguanides"},
    {"code": "N06AB06", "term": "Sertraline", "atc": "N06AB", "category": "SSRIs"},
    {"code": "J01CA04", "term": "Amoxicillin", "atc": "J01CA", "category": "Penicillins"},
    {"code": "R06AE07", "term": "Cetirizine", "atc": "R06AE", "category": "Piperazine derivatives"},
    {"code": "B01AC06", "term": "Aspirin", "atc": "B01AC", "category": "Platelet aggregation inhibitors"},
    {"code": "N05BA06", "term": "Lorazepam", "atc": "N05BA", "category": "Benzodiazepines"},
    {"code": "C03CA01", "term": "Furosemide", "atc": "C03CA", "category": "Sulfonamides"},
    {"code": "H02AB06", "term": "Prednisolone", "atc": "H02AB", "category": "Glucocorticoids"},
]

# Common misspellings and variations
TERM_VARIATIONS = {
    "headach": "Headache",
    "headache": "Headache",
    "head ache": "Headache",
    "nausea": "Nausea",
    "nausia": "Nausea",
    "vomit": "Vomiting",
    "vomitting": "Vomiting",
    "diarrhea": "Diarrhoea",
    "diarrhoea": "Diarrhoea",
    "fatique": "Fatigue",
    "fatigue": "Fatigue",
    "dizzy": "Dizziness",
    "dizziness": "Dizziness",
    "fever": "Pyrexia",
    "pyrexia": "Pyrexia",
    "rash": "Rash",
    "skin rash": "Rash",
    "itching": "Pruritus",
    "pruritus": "Pruritus",
    "paracetamol": "Paracetamol",
    "acetaminophen": "Paracetamol",
    "tylenol": "Paracetamol",
    "aspirin": "Aspirin",
    "ibuprofen": "Ibuprofen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
}

# =============================================================================
# DATA LOADER
# =============================================================================

class CoderDataLoader:
    """Data loader for Coder Workbench"""
    
    def __init__(self):
        self.base_path = Path("data/processed")
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300
    
    def _load_parquet(self, path: Path) -> Optional[pd.DataFrame]:
        """Load parquet file with caching"""
        cache_key = str(path)
        now = datetime.now().timestamp()
        
        if cache_key in self._cache:
            if now - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]
        
        if path.exists():
            df = pd.read_parquet(path)
            self._cache[cache_key] = df
            self._cache_time[cache_key] = now
            return df
        return None
    
    def load_patient_issues(self) -> Optional[pd.DataFrame]:
        """Load patient issues"""
        return self._load_parquet(self.base_path / "analytics" / "patient_issues.parquet")
    
    def load_coding_meddra(self) -> Optional[pd.DataFrame]:
        """Load MedDRA coding data"""
        return self._load_parquet(self.base_path / "cleaned" / "coding_meddra_agg.parquet")
    
    def load_coding_whodrug(self) -> Optional[pd.DataFrame]:
        """Load WHODrug coding data"""
        return self._load_parquet(self.base_path / "cleaned" / "coding_whodrug_agg.parquet")
    
    def get_coding_stats(self) -> CodingStats:
        """Get coding statistics"""
        issues = self.load_patient_issues()
        
        meddra_pending = 0
        whodrug_pending = 0
        
        if issues is not None:
            if 'issue_meddra_uncoded' in issues.columns:
                meddra_pending = int(issues['issue_meddra_uncoded'].sum())
            if 'issue_whodrug_uncoded' in issues.columns:
                whodrug_pending = int(issues['issue_whodrug_uncoded'].sum())
        
        total_pending = meddra_pending + whodrug_pending
        
        # Simulate other stats
        np.random.seed(42)
        
        return CodingStats(
            total_pending=total_pending,
            total_coded_today=np.random.randint(50, 150),
            total_escalated=np.random.randint(5, 20),
            auto_approval_rate=np.random.uniform(0.75, 0.92),
            avg_confidence=np.random.uniform(0.82, 0.95),
            meddra_pending=meddra_pending,
            whodrug_pending=whodrug_pending,
            high_confidence_pending=int(total_pending * 0.4),
            productivity_trend='improving'
        )
    
    def get_coding_queue(self, coding_type: Optional[CodingType] = None, 
                         confidence_filter: Optional[str] = None,
                         limit: int = 100) -> List[CodingItem]:
        """Get coding queue with items to be coded"""
        issues = self.load_patient_issues()
        
        items = []
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        # Sample verbatim terms for MedDRA
        meddra_verbatims = [
            ("headach", "Headache", 0.97),
            ("severe headache", "Headache", 0.92),
            ("nausea and vomiting", "Nausea", 0.78),
            ("mild nausea", "Nausea", 0.95),
            ("diarrea", "Diarrhoea", 0.88),
            ("loose stools", "Diarrhoea", 0.72),
            ("feeling tired", "Fatigue", 0.85),
            ("extreme fatigue", "Fatigue", 0.91),
            ("dizzy spells", "Dizziness", 0.82),
            ("vertigo", "Dizziness", 0.65),
            ("joint pain", "Arthralgia", 0.89),
            ("knee pain", "Arthralgia", 0.76),
            ("sore throat", "Pharyngitis", 0.84),
            ("fever", "Pyrexia", 0.96),
            ("high temperature", "Pyrexia", 0.88),
            ("coughing", "Cough", 0.94),
            ("dry cough", "Cough", 0.91),
            ("skin rash", "Rash", 0.93),
            ("itchy rash", "Rash", 0.87),
            ("stomach ache", "Abdominal pain", 0.79),
            ("abdominal discomfort", "Abdominal pain", 0.83),
            ("lower back pain", "Back pain", 0.90),
            ("muscle ache", "Myalgia", 0.86),
            ("anxiety symptoms", "Anxiety", 0.81),
            ("sleep problems", "Insomnia", 0.77),
            ("itching", "Pruritus", 0.92),
            ("high blood pressure", "Hypertension", 0.94),
            ("low blood pressure", "Hypotension", 0.93),
            ("unusual symptom xyz", None, 0.25),  # No match
            ("complex medical event", None, 0.35),  # Low confidence
        ]
        
        # Sample verbatim terms for WHODrug
        whodrug_verbatims = [
            ("tylenol", "Paracetamol", 0.98),
            ("paracetamol 500mg", "Paracetamol", 0.96),
            ("acetaminophen", "Paracetamol", 0.94),
            ("advil", "Ibuprofen", 0.97),
            ("ibuprofen 400", "Ibuprofen", 0.95),
            ("aspirin 81mg", "Aspirin", 0.96),
            ("baby aspirin", "Aspirin", 0.88),
            ("metformin 500", "Metformin", 0.94),
            ("glucophage", "Metformin", 0.85),
            ("omeprazole 20mg", "Omeprazole", 0.97),
            ("prilosec", "Omeprazole", 0.89),
            ("lipitor", "Simvastatin", 0.72),  # Actually atorvastatin
            ("simvastatin 40mg", "Simvastatin", 0.96),
            ("zoloft", "Sertraline", 0.91),
            ("sertraline 50", "Sertraline", 0.95),
            ("amoxicillin 500mg", "Amoxicillin", 0.97),
            ("amoxil", "Amoxicillin", 0.86),
            ("zyrtec", "Cetirizine", 0.93),
            ("ativan", "Lorazepam", 0.90),
            ("lasix", "Furosemide", 0.88),
            ("prednisone 10mg", "Prednisolone", 0.78),
            ("unknown medication", None, 0.20),
            ("herbal supplement", None, 0.30),
        ]
        
        # Generate MedDRA items
        if coding_type is None or coding_type == CodingType.MEDDRA:
            for i, (verbatim, suggested, conf) in enumerate(meddra_verbatims):
                # Apply confidence filter
                if confidence_filter:
                    if confidence_filter == "very_high" and conf < 0.95:
                        continue
                    elif confidence_filter == "high" and (conf < 0.85 or conf >= 0.95):
                        continue
                    elif confidence_filter == "medium" and (conf < 0.70 or conf >= 0.85):
                        continue
                    elif confidence_filter == "low" and conf >= 0.70:
                        continue
                
                # Find code
                code = None
                if suggested:
                    for entry in MEDDRA_DICTIONARY:
                        if entry["term"] == suggested:
                            code = entry["code"]
                            break
                
                # Generate alternatives for medium/low confidence
                alternatives = []
                if 0.50 <= conf < 0.90:
                    alt_terms = np.random.choice(
                        [e["term"] for e in MEDDRA_DICTIONARY if e["term"] != suggested],
                        size=min(3, len(MEDDRA_DICTIONARY) - 1),
                        replace=False
                    )
                    for alt in alt_terms:
                        for entry in MEDDRA_DICTIONARY:
                            if entry["term"] == alt:
                                alternatives.append({
                                    "code": entry["code"],
                                    "term": entry["term"],
                                    "confidence": round(conf * np.random.uniform(0.6, 0.9), 2)
                                })
                                break
                
                item = CodingItem(
                    item_id=f"MED-{uuid.uuid4().hex[:8].upper()}",
                    verbatim_term=verbatim,
                    coding_type=CodingType.MEDDRA,
                    status=CodingStatus.PENDING if conf < 0.95 else CodingStatus.AUTO_CODED,
                    confidence=conf,
                    suggested_code=code,
                    suggested_term=suggested,
                    patient_id=f"Study_{np.random.randint(1,24)}|Site_{np.random.randint(1,100)}|Subject_{np.random.randint(1,1000)}",
                    study_id=f"Study_{np.random.randint(1, 24)}",
                    site_id=f"Site_{np.random.randint(1, 100)}",
                    form_name="Adverse Events",
                    field_name="AE Term",
                    context=f"Reported on Day {np.random.randint(1, 100)} of treatment",
                    alternatives=alternatives
                )
                items.append(item)
        
        # Generate WHODrug items
        if coding_type is None or coding_type == CodingType.WHODRUG:
            for i, (verbatim, suggested, conf) in enumerate(whodrug_verbatims):
                # Apply confidence filter
                if confidence_filter:
                    if confidence_filter == "very_high" and conf < 0.95:
                        continue
                    elif confidence_filter == "high" and (conf < 0.85 or conf >= 0.95):
                        continue
                    elif confidence_filter == "medium" and (conf < 0.70 or conf >= 0.85):
                        continue
                    elif confidence_filter == "low" and conf >= 0.70:
                        continue
                
                # Find code
                code = None
                if suggested:
                    for entry in WHODRUG_DICTIONARY:
                        if entry["term"] == suggested:
                            code = entry["code"]
                            break
                
                # Generate alternatives
                alternatives = []
                if 0.50 <= conf < 0.90:
                    alt_terms = np.random.choice(
                        [e["term"] for e in WHODRUG_DICTIONARY if e["term"] != suggested],
                        size=min(3, len(WHODRUG_DICTIONARY) - 1),
                        replace=False
                    )
                    for alt in alt_terms:
                        for entry in WHODRUG_DICTIONARY:
                            if entry["term"] == alt:
                                alternatives.append({
                                    "code": entry["code"],
                                    "term": entry["term"],
                                    "confidence": round(conf * np.random.uniform(0.6, 0.9), 2)
                                })
                                break
                
                item = CodingItem(
                    item_id=f"WHO-{uuid.uuid4().hex[:8].upper()}",
                    verbatim_term=verbatim,
                    coding_type=CodingType.WHODRUG,
                    status=CodingStatus.PENDING if conf < 0.95 else CodingStatus.AUTO_CODED,
                    confidence=conf,
                    suggested_code=code,
                    suggested_term=suggested,
                    patient_id=f"Study_{np.random.randint(1,24)}|Site_{np.random.randint(1,100)}|Subject_{np.random.randint(1,1000)}",
                    study_id=f"Study_{np.random.randint(1, 24)}",
                    site_id=f"Site_{np.random.randint(1, 100)}",
                    form_name="Concomitant Medications",
                    field_name="Medication Name",
                    context=f"Started on Day {np.random.randint(-30, 100)}",
                    alternatives=alternatives
                )
                items.append(item)
        
        # Sort by confidence (lowest first for manual review queue)
        items.sort(key=lambda x: (-x.confidence if x.confidence >= 0.95 else x.confidence))
        
        return items[:limit]
    
    def search_dictionary(self, query: str, coding_type: CodingType, 
                          limit: int = 20) -> List[Dict]:
        """Search dictionary for matching terms"""
        query_lower = query.lower().strip()
        results = []
        
        if coding_type == CodingType.MEDDRA:
            dictionary = MEDDRA_DICTIONARY
        else:
            dictionary = WHODRUG_DICTIONARY
        
        # Check variations first
        if query_lower in TERM_VARIATIONS:
            standard_term = TERM_VARIATIONS[query_lower]
            for entry in dictionary:
                if entry["term"] == standard_term:
                    results.append({
                        **entry,
                        "match_type": "exact_variation",
                        "score": 1.0
                    })
        
        # Search by term
        for entry in dictionary:
            term_lower = entry["term"].lower()
            
            if query_lower == term_lower:
                if not any(r["code"] == entry["code"] for r in results):
                    results.append({
                        **entry,
                        "match_type": "exact",
                        "score": 1.0
                    })
            elif query_lower in term_lower:
                if not any(r["code"] == entry["code"] for r in results):
                    results.append({
                        **entry,
                        "match_type": "contains",
                        "score": len(query_lower) / len(term_lower)
                    })
            elif term_lower in query_lower:
                if not any(r["code"] == entry["code"] for r in results):
                    results.append({
                        **entry,
                        "match_type": "partial",
                        "score": len(term_lower) / len(query_lower)
                    })
        
        # Search by code
        for entry in dictionary:
            if query_lower in entry["code"].lower():
                if not any(r["code"] == entry["code"] for r in results):
                    results.append({
                        **entry,
                        "match_type": "code",
                        "score": 0.9
                    })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:limit]
    
    def get_escalations(self) -> List[EscalationRequest]:
        """Get list of escalation requests"""
        escalations = [
            EscalationRequest(
                request_id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
                item_id="MED-12345678",
                verbatim_term="unusual neurological event",
                coding_type=CodingType.MEDDRA,
                reason=EscalationReason.AMBIGUOUS_TERM,
                escalated_by="Coder_Kim",
                escalated_to="Medical Monitor",
                escalated_at=datetime.now() - timedelta(hours=4),
                status="pending",
                notes="Multiple possible matches - need medical input"
            ),
            EscalationRequest(
                request_id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
                item_id="WHO-87654321",
                verbatim_term="traditional medicine mixture",
                coding_type=CodingType.WHODRUG,
                reason=EscalationReason.NO_MATCH,
                escalated_by="Coder_Kim",
                escalated_to="Drug Safety",
                escalated_at=datetime.now() - timedelta(days=1),
                status="pending",
                notes="No standard drug code available"
            ),
            EscalationRequest(
                request_id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
                item_id="MED-ABCDEF12",
                verbatim_term="cardiac symptoms post infusion",
                coding_type=CodingType.MEDDRA,
                reason=EscalationReason.MEDICAL_REVIEW,
                escalated_by="Coder_Kim",
                escalated_to="Medical Monitor",
                escalated_at=datetime.now() - timedelta(days=2),
                status="resolved",
                resolution="Code as 'Infusion related reaction' (10051792)",
                resolved_at=datetime.now() - timedelta(hours=6),
                notes="Safety-related term"
            )
        ]
        return escalations
    
    def get_studies_list(self) -> List[str]:
        """Get list of studies"""
        issues = self.load_patient_issues()
        if issues is not None and 'study_id' in issues.columns:
            return sorted(issues['study_id'].dropna().unique().tolist())
        return [f"Study_{i}" for i in range(1, 24)]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 0.95:
        return CONFIDENCE_COLORS['very_high']
    elif confidence >= 0.85:
        return CONFIDENCE_COLORS['high']
    elif confidence >= 0.70:
        return CONFIDENCE_COLORS['medium']
    elif confidence >= 0.50:
        return CONFIDENCE_COLORS['low']
    else:
        return CONFIDENCE_COLORS['very_low']

def get_confidence_label(confidence: float) -> str:
    """Get label for confidence level"""
    if confidence >= 0.95:
        return "Very High"
    elif confidence >= 0.85:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    elif confidence >= 0.50:
        return "Low"
    else:
        return "Very Low"

def get_confidence_icon(confidence: float) -> str:
    """Get icon for confidence level"""
    if confidence >= 0.95:
        return "✅"
    elif confidence >= 0.85:
        return "🟢"
    elif confidence >= 0.70:
        return "🟡"
    elif confidence >= 0.50:
        return "🟠"
    else:
        return "🔴"


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_coder_header(stats: CodingStats):
    """Render coder workbench header"""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 28px;">🏷️ Medical Coder Workbench</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 16px;">
                    MedDRA & WHODrug Coding • {datetime.now().strftime('%B %d, %Y')}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 36px; font-weight: bold;">{stats.total_pending}</div>
                <div style="opacity: 0.9;">Terms Pending</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_section(stats: CodingStats):
    """Render statistics section"""
    
    st.markdown("### 📊 Today's Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 28px; font-weight: bold;">{stats.meddra_pending}</div>
            <div style="font-size: 12px; opacity: 0.9;">MedDRA Pending</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 28px; font-weight: bold;">{stats.whodrug_pending}</div>
            <div style="font-size: 12px; opacity: 0.9;">WHODrug Pending</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 28px; font-weight: bold;">{stats.total_coded_today}</div>
            <div style="font-size: 12px; opacity: 0.9;">Coded Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 28px; font-weight: bold;">{stats.high_confidence_pending}</div>
            <div style="font-size: 12px; opacity: 0.9;">High Conf. Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 28px; font-weight: bold;">{stats.total_escalated}</div>
            <div style="font-size: 12px; opacity: 0.9;">Escalated</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row - metrics
    st.markdown("<br>", unsafe_allow_html=True)
    
    mcol1, mcol2, mcol3 = st.columns(3)
    
    with mcol1:
        st.metric(
            "Auto-Approval Rate",
            f"{stats.auto_approval_rate:.1%}",
            delta="+2.3%" if stats.productivity_trend == 'improving' else "-1.2%"
        )
    
    with mcol2:
        st.metric(
            "Avg Confidence",
            f"{stats.avg_confidence:.1%}",
            delta="+0.5%" if stats.productivity_trend == 'improving' else None
        )
    
    with mcol3:
        st.metric(
            "Productivity Trend",
            stats.productivity_trend.title(),
            delta="📈" if stats.productivity_trend == 'improving' else "📉"
        )


def render_coding_queue(items: List[CodingItem], loader: CoderDataLoader):
    """Render the coding queue"""
    
    st.markdown("### 📋 Coding Queue")
    
    # Filters
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    
    with fcol1:
        type_filter = st.selectbox(
            "Coding Type",
            ["All", "MedDRA", "WHODrug"],
            key="queue_type_filter"
        )
    
    with fcol2:
        conf_filter = st.selectbox(
            "Confidence Level",
            ["All", "Very High (≥95%)", "High (85-95%)", "Medium (70-85%)", "Low (<70%)"],
            key="queue_conf_filter"
        )
    
    with fcol3:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "Auto-coded", "Needs Review"],
            key="queue_status_filter"
        )
    
    with fcol4:
        sort_order = st.selectbox(
            "Sort By",
            ["Confidence (Low First)", "Confidence (High First)", "Recent First"],
            key="queue_sort"
        )
    
    # Apply filters
    filtered_items = items
    
    if type_filter == "MedDRA":
        filtered_items = [i for i in filtered_items if i.coding_type == CodingType.MEDDRA]
    elif type_filter == "WHODrug":
        filtered_items = [i for i in filtered_items if i.coding_type == CodingType.WHODRUG]
    
    if conf_filter == "Very High (≥95%)":
        filtered_items = [i for i in filtered_items if i.confidence >= 0.95]
    elif conf_filter == "High (85-95%)":
        filtered_items = [i for i in filtered_items if 0.85 <= i.confidence < 0.95]
    elif conf_filter == "Medium (70-85%)":
        filtered_items = [i for i in filtered_items if 0.70 <= i.confidence < 0.85]
    elif conf_filter == "Low (<70%)":
        filtered_items = [i for i in filtered_items if i.confidence < 0.70]
    
    if status_filter == "Pending":
        filtered_items = [i for i in filtered_items if i.status == CodingStatus.PENDING]
    elif status_filter == "Auto-coded":
        filtered_items = [i for i in filtered_items if i.status == CodingStatus.AUTO_CODED]
    elif status_filter == "Needs Review":
        filtered_items = [i for i in filtered_items if i.status == CodingStatus.NEEDS_REVIEW]
    
    # Sort
    if sort_order == "Confidence (Low First)":
        filtered_items.sort(key=lambda x: x.confidence)
    elif sort_order == "Confidence (High First)":
        filtered_items.sort(key=lambda x: x.confidence, reverse=True)
    
    st.markdown(f"**Showing {len(filtered_items)} items**")
    
    st.markdown("---")
    
    # Batch actions for high confidence
    high_conf_items = [i for i in filtered_items if i.confidence >= 0.95]
    if high_conf_items:
        with st.expander(f"⚡ Batch Approve {len(high_conf_items)} High-Confidence Items (≥95%)", expanded=False):
            st.markdown("These items have very high confidence and can be batch approved:")
            
            # Show preview table
            batch_data = []
            for item in high_conf_items[:10]:
                batch_data.append({
                    "Verbatim": item.verbatim_term,
                    "Suggested": item.suggested_term or "No match",
                    "Code": item.suggested_code or "-",
                    "Confidence": f"{item.confidence:.1%}",
                    "Type": item.coding_type.value.upper()
                })
            
            st.dataframe(pd.DataFrame(batch_data), use_container_width=True, hide_index=True)
            
            if len(high_conf_items) > 10:
                st.info(f"... and {len(high_conf_items) - 10} more items")
            
            bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
            with bcol1:
                if st.button("✅ Approve All", key="batch_approve", use_container_width=True, type="primary"):
                    st.success(f"Batch approved {len(high_conf_items)} items!")
            with bcol2:
                if st.button("👁️ Review First", key="batch_review", use_container_width=True):
                    st.info("Opening review mode...")
    
    st.markdown("---")
    
    # Individual items
    for item in filtered_items[:20]:
        render_coding_item(item, loader)


def render_coding_item(item: CodingItem, loader: CoderDataLoader):
    """Render a single coding item"""
    
    type_color = CODING_TYPE_COLORS.get(item.coding_type.value, '#95a5a6')
    conf_color = get_confidence_color(item.confidence)
    conf_icon = get_confidence_icon(item.confidence)
    conf_label = get_confidence_label(item.confidence)
    
    with st.container():
        st.markdown(f"""
        <div style="
            border-left: 4px solid {conf_color};
            background: white;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <span style="
                            background: {type_color};
                            color: white;
                            padding: 2px 8px;
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: 600;
                        ">{item.coding_type.value.upper()}</span>
                        <span style="font-size: 16px; font-weight: 600; color: #2c3e50;">
                            "{item.verbatim_term}"
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #7f8c8d; margin-bottom: 8px;">
                        {item.form_name} • {item.field_name} • {item.patient_id}
                    </div>
                    {f'''<div style="font-size: 14px; color: #2c3e50; margin-bottom: 8px;">
                        <strong>Suggested:</strong> {item.suggested_term or "No match found"} 
                        {f"({item.suggested_code})" if item.suggested_code else ""}
                    </div>''' if item.suggested_term else '<div style="font-size: 14px; color: #e74c3c; margin-bottom: 8px;"><strong>No automatic match found</strong></div>'}
                </div>
                <div style="text-align: right; min-width: 120px;">
                    <div style="
                        background: {conf_color}20;
                        color: {conf_color};
                        padding: 5px 12px;
                        border-radius: 15px;
                        font-size: 14px;
                        font-weight: 600;
                        display: inline-block;
                    ">
                        {conf_icon} {item.confidence:.0%}
                    </div>
                    <div style="font-size: 11px; color: #7f8c8d; margin-top: 5px;">
                        {conf_label} Confidence
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action row
        with st.expander("Actions & Details", expanded=False):
            # Alternatives
            if item.alternatives:
                st.markdown("**Alternative Matches:**")
                alt_data = []
                for alt in item.alternatives:
                    alt_data.append({
                        "Term": alt["term"],
                        "Code": alt["code"],
                        "Confidence": f"{alt['confidence']:.0%}"
                    })
                st.dataframe(pd.DataFrame(alt_data), use_container_width=True, hide_index=True)
            
            # Dictionary search
            st.markdown("**Dictionary Search:**")
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input(
                    "Search term",
                    value=item.verbatim_term,
                    key=f"search_{item.item_id}",
                    label_visibility="collapsed"
                )
            with search_col2:
                if st.button("🔍 Search", key=f"search_btn_{item.item_id}", use_container_width=True):
                    results = loader.search_dictionary(search_query, item.coding_type)
                    if results:
                        st.session_state[f"search_results_{item.item_id}"] = results
            
            # Show search results
            if f"search_results_{item.item_id}" in st.session_state:
                results = st.session_state[f"search_results_{item.item_id}"]
                st.markdown(f"**Found {len(results)} matches:**")
                for r in results[:5]:
                    rcol1, rcol2 = st.columns([4, 1])
                    with rcol1:
                        st.markdown(f"**{r['term']}** ({r['code']}) - Match: {r['match_type']}")
                    with rcol2:
                        if st.button("Select", key=f"select_{item.item_id}_{r['code']}", use_container_width=True):
                            st.success(f"Selected: {r['term']}")
            
            st.markdown("---")
            
            # Action buttons
            acol1, acol2, acol3, acol4, acol5 = st.columns(5)
            
            with acol1:
                if st.button("✅ Approve", key=f"approve_{item.item_id}", use_container_width=True, 
                            type="primary" if item.confidence >= 0.85 else "secondary"):
                    st.success(f"Approved: {item.suggested_term}")
            
            with acol2:
                if st.button("✏️ Edit", key=f"edit_{item.item_id}", use_container_width=True):
                    st.info("Opening editor...")
            
            with acol3:
                if st.button("❌ Reject", key=f"reject_{item.item_id}", use_container_width=True):
                    st.warning("Marked for review")
            
            with acol4:
                if st.button("⬆️ Escalate", key=f"escalate_{item.item_id}", use_container_width=True):
                    st.session_state[f"escalate_{item.item_id}"] = True
            
            with acol5:
                if st.button("⏭️ Skip", key=f"skip_{item.item_id}", use_container_width=True):
                    st.info("Skipped")
            
            # Escalation form
            if st.session_state.get(f"escalate_{item.item_id}", False):
                st.markdown("**Escalation Details:**")
                esc_reason = st.selectbox(
                    "Reason",
                    [r.value.replace("_", " ").title() for r in EscalationReason],
                    key=f"esc_reason_{item.item_id}"
                )
                esc_to = st.selectbox(
                    "Escalate To",
                    ["Medical Monitor", "Drug Safety", "Senior Coder", "Study Lead"],
                    key=f"esc_to_{item.item_id}"
                )
                esc_notes = st.text_area(
                    "Notes",
                    key=f"esc_notes_{item.item_id}",
                    height=80
                )
                
                esc_col1, esc_col2 = st.columns(2)
                with esc_col1:
                    if st.button("📤 Submit Escalation", key=f"submit_esc_{item.item_id}", use_container_width=True):
                        st.success(f"Escalated to {esc_to}")
                        st.session_state[f"escalate_{item.item_id}"] = False
                with esc_col2:
                    if st.button("Cancel", key=f"cancel_esc_{item.item_id}", use_container_width=True):
                        st.session_state[f"escalate_{item.item_id}"] = False


def render_dictionary_search(loader: CoderDataLoader):
    """Render dictionary search section"""
    
    st.markdown("### 🔍 Dictionary Search")
    
    scol1, scol2 = st.columns([3, 1])
    
    with scol1:
        search_term = st.text_input(
            "Search MedDRA or WHODrug dictionary",
            placeholder="Enter term, code, or synonym...",
            key="dict_search"
        )
    
    with scol2:
        dict_type = st.selectbox(
            "Dictionary",
            ["MedDRA", "WHODrug"],
            key="dict_type"
        )
    
    if search_term:
        coding_type = CodingType.MEDDRA if dict_type == "MedDRA" else CodingType.WHODRUG
        results = loader.search_dictionary(search_term, coding_type)
        
        if results:
            st.markdown(f"**Found {len(results)} matches:**")
            
            # Display as cards
            for result in results:
                match_color = '#27ae60' if result['score'] >= 0.9 else '#f39c12' if result['score'] >= 0.7 else '#95a5a6'
                
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    padding: 12px 15px;
                    border-radius: 8px;
                    margin: 8px 0;
                    border-left: 3px solid {match_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 15px; font-weight: 600; color: #2c3e50;">
                                {result['term']}
                            </div>
                            <div style="font-size: 13px; color: #7f8c8d;">
                                Code: {result['code']} | 
                                Level: {result.get('level', result.get('atc', 'N/A'))} |
                                Match: {result['match_type'].replace('_', ' ').title()}
                            </div>
                        </div>
                        <div style="
                            background: {match_color}20;
                            color: {match_color};
                            padding: 3px 10px;
                            border-radius: 10px;
                            font-size: 12px;
                        ">
                            {result['score']:.0%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matches found. Try a different search term.")
    else:
        # Show popular terms
        st.markdown("**Popular MedDRA Terms:**")
        pop_terms = ["Headache", "Nausea", "Fatigue", "Dizziness", "Pyrexia", "Rash", "Cough"]
        cols = st.columns(7)
        for i, term in enumerate(pop_terms):
            with cols[i]:
                if st.button(term, key=f"pop_{term}", use_container_width=True):
                    st.session_state['dict_search'] = term
                    st.rerun()


def render_escalations(escalations: List[EscalationRequest]):
    """Render escalations section"""
    
    st.markdown("### ⬆️ Escalations")
    
    pending = [e for e in escalations if e.status == "pending"]
    resolved = [e for e in escalations if e.status == "resolved"]
    
    st.markdown(f"**{len(pending)} pending** | {len(resolved)} resolved")
    
    if pending:
        st.markdown("#### Pending Escalations")
        for esc in pending:
            type_color = CODING_TYPE_COLORS.get(esc.coding_type.value, '#95a5a6')
            
            with st.expander(f"🔴 {esc.verbatim_term} - {esc.reason.value.replace('_', ' ').title()}"):
                st.markdown(f"""
                <div style="padding: 10px; background: #fff5f5; border-radius: 8px;">
                    <p><strong>Escalated to:</strong> {esc.escalated_to}</p>
                    <p><strong>Reason:</strong> {esc.reason.value.replace('_', ' ').title()}</p>
                    <p><strong>When:</strong> {esc.escalated_at.strftime('%Y-%m-%d %H:%M')}</p>
                    <p><strong>Notes:</strong> {esc.notes}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.button("📧 Send Reminder", key=f"remind_{esc.request_id}", use_container_width=True)
                with col2:
                    st.button("❌ Cancel", key=f"cancel_{esc.request_id}", use_container_width=True)
    
    if resolved:
        st.markdown("#### Recently Resolved")
        for esc in resolved:
            with st.expander(f"✅ {esc.verbatim_term} - Resolved"):
                st.markdown(f"""
                <div style="padding: 10px; background: #f0fff0; border-radius: 8px;">
                    <p><strong>Resolution:</strong> {esc.resolution}</p>
                    <p><strong>Resolved at:</strong> {esc.resolved_at.strftime('%Y-%m-%d %H:%M') if esc.resolved_at else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)


def render_productivity_chart(stats: CodingStats):
    """Render productivity chart"""
    
    st.markdown("### 📈 Productivity")
    
    # Mock data for chart
    dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
    data = {
        'Date': dates,
        'Coded': np.random.randint(80, 200, 14),
        'Auto-Approved': np.random.randint(50, 150, 14),
        'Manual': np.random.randint(20, 60, 14)
    }
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Auto-Approved'],
        name='Auto-Approved',
        marker_color='#27ae60'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Manual'],
        name='Manual',
        marker_color='#9b59b6'
    ))
    
    fig.update_layout(
        barmode='stack',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_title="",
        yaxis_title="Terms Coded"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_page(user=None):
    """Main render function for Coder Workbench"""
    
    # Initialize data loader
    loader = CoderDataLoader()
    
    # Get stats
    stats = loader.get_coding_stats()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏷️ Coder Workbench")
        
        st.markdown("#### Quick Stats")
        st.metric("Pending", stats.total_pending)
        st.metric("Coded Today", stats.total_coded_today)
        st.metric("Escalated", stats.total_escalated)
        
        st.markdown("---")
        
        st.markdown("#### Quick Filters")
        quick_type = st.radio(
            "Coding Type",
            ["All", "MedDRA Only", "WHODrug Only"],
            key="sidebar_type"
        )
        
        quick_conf = st.radio(
            "Confidence",
            ["All", "High (≥85%)", "Needs Review (<85%)"],
            key="sidebar_conf"
        )
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Queue", use_container_width=True):
            loader._cache.clear()
            st.rerun()
    
    # Header
    render_coder_header(stats)
    
    # Stats section
    render_stats_section(stats)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Coding Queue", 
        "🔍 Dictionary Search", 
        "⬆️ Escalations",
        "📈 Productivity"
    ])
    
    with tab1:
        # Get items based on sidebar filters
        coding_type = None
        if quick_type == "MedDRA Only":
            coding_type = CodingType.MEDDRA
        elif quick_type == "WHODrug Only":
            coding_type = CodingType.WHODRUG
        
        items = loader.get_coding_queue(coding_type=coding_type)
        render_coding_queue(items, loader)
    
    with tab2:
        render_dictionary_search(loader)
    
    with tab3:
        escalations = loader.get_escalations()
        render_escalations(escalations)
    
    with tab4:
        render_productivity_chart(stats)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 12px; padding: 20px;">
        TrialPulse NEXUS 10X • Coder Workbench v1.0 • 
        MedDRA v26.0 • WHODrug March 2024
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_coder_workbench():
    """Test the coder workbench components"""
    print("\n" + "="*70)
    print("TRIALPULSE NEXUS 10X - CODER WORKBENCH TEST")
    print("="*70 + "\n")
    
    loader = CoderDataLoader()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Stats
    print("TEST 1: Coding Statistics")
    try:
        stats = loader.get_coding_stats()
        print(f"   ✅ Stats loaded")
        print(f"      - MedDRA Pending: {stats.meddra_pending}")
        print(f"      - WHODrug Pending: {stats.whodrug_pending}")
        print(f"      - Total Pending: {stats.total_pending}")
        print(f"      - High Confidence: {stats.high_confidence_pending}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 2: Coding Queue
    print("\nTEST 2: Coding Queue")
    try:
        items = loader.get_coding_queue()
        print(f"   ✅ Queue loaded: {len(items)} items")
        
        meddra_count = sum(1 for i in items if i.coding_type == CodingType.MEDDRA)
        whodrug_count = sum(1 for i in items if i.coding_type == CodingType.WHODRUG)
        high_conf = sum(1 for i in items if i.confidence >= 0.95)
        
        print(f"      - MedDRA: {meddra_count}")
        print(f"      - WHODrug: {whodrug_count}")
        print(f"      - High Confidence (≥95%): {high_conf}")
        
        if items:
            print(f"      - Sample: '{items[0].verbatim_term}' → {items[0].suggested_term} ({items[0].confidence:.0%})")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 3: MedDRA Filter
    print("\nTEST 3: MedDRA Filter")
    try:
        meddra_items = loader.get_coding_queue(coding_type=CodingType.MEDDRA)
        print(f"   ✅ MedDRA items: {len(meddra_items)}")
        assert all(i.coding_type == CodingType.MEDDRA for i in meddra_items)
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 4: WHODrug Filter
    print("\nTEST 4: WHODrug Filter")
    try:
        whodrug_items = loader.get_coding_queue(coding_type=CodingType.WHODRUG)
        print(f"   ✅ WHODrug items: {len(whodrug_items)}")
        assert all(i.coding_type == CodingType.WHODRUG for i in whodrug_items)
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 5: Dictionary Search - MedDRA
    print("\nTEST 5: Dictionary Search - MedDRA")
    try:
        results = loader.search_dictionary("headache", CodingType.MEDDRA)
        print(f"   ✅ Search 'headache': {len(results)} results")
        if results:
            print(f"      - Top match: {results[0]['term']} ({results[0]['code']})")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 6: Dictionary Search - WHODrug
    print("\nTEST 6: Dictionary Search - WHODrug")
    try:
        results = loader.search_dictionary("tylenol", CodingType.WHODRUG)
        print(f"   ✅ Search 'tylenol': {len(results)} results")
        if results:
            print(f"      - Top match: {results[0]['term']} ({results[0]['code']})")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 7: Escalations
    print("\nTEST 7: Escalations")
    try:
        escalations = loader.get_escalations()
        print(f"   ✅ Escalations: {len(escalations)}")
        pending = sum(1 for e in escalations if e.status == "pending")
        resolved = sum(1 for e in escalations if e.status == "resolved")
        print(f"      - Pending: {pending}")
        print(f"      - Resolved: {resolved}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 8: Helper Functions
    print("\nTEST 8: Helper Functions")
    try:
        assert get_confidence_color(0.98) == '#27ae60'
        assert get_confidence_color(0.90) == '#2ecc71'
        assert get_confidence_color(0.75) == '#f39c12'
        assert get_confidence_color(0.60) == '#e67e22'
        assert get_confidence_color(0.30) == '#e74c3c'
        
        assert get_confidence_label(0.98) == "Very High"
        assert get_confidence_label(0.90) == "High"
        assert get_confidence_label(0.75) == "Medium"
        
        assert get_confidence_icon(0.98) == "✅"
        assert get_confidence_icon(0.90) == "🟢"
        assert get_confidence_icon(0.75) == "🟡"
        
        print("   ✅ All helper functions working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 9: CodingItem Properties
    print("\nTEST 9: CodingItem Properties")
    try:
        item = CodingItem(
            item_id="TEST-001",
            verbatim_term="test headache",
            coding_type=CodingType.MEDDRA,
            status=CodingStatus.PENDING,
            confidence=0.92
        )
        
        assert item.confidence_level == ConfidenceLevel.HIGH
        assert item.confidence_color == '#2ecc71'
        
        print(f"   ✅ CodingItem properties working")
        print(f"      - Confidence Level: {item.confidence_level.value}")
        print(f"      - Confidence Color: {item.confidence_color}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 10: Render Page Callable
    print("\nTEST 10: Render Page Function")
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
    test_coder_workbench()