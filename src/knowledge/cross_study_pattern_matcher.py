"""
TRIALPULSE NEXUS 10X - Phase 4.5: Cross-Study Pattern Matcher v1.0

Purpose:
- Pattern embedding index across studies
- Cross-study similarity matching
- Pattern validation tracking
- Pattern transfer recommendations
- Study-specific adaptations

Author: TrialPulse Team
Version: 1.0
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using fallback.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StudyPattern:
    """Represents a pattern detected in a specific study."""
    pattern_id: str
    study_id: str
    pattern_type: str
    description: str
    issue_types: List[str]
    affected_sites: int
    affected_patients: int
    severity: str
    confidence: float
    first_detected: str
    last_seen: str
    resolution_rate: float = 0.0
    avg_resolution_days: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class PatternMatch:
    """Represents a match between patterns across studies."""
    source_pattern_id: str
    source_study_id: str
    target_pattern_id: str
    target_study_id: str
    similarity_score: float
    match_type: str  # exact, similar, related
    transferable: bool
    adaptation_needed: List[str]
    confidence: float


@dataclass
class PatternValidation:
    """Tracks validation status of a pattern."""
    pattern_id: str
    study_id: str
    validation_status: str  # pending, validated, rejected, partial
    validated_by: str
    validation_date: str
    effectiveness_score: float
    notes: str
    transfer_success_rate: float = 0.0


class CrossStudyPatternMatcher:
    """
    Cross-Study Pattern Matcher for TRIALPULSE NEXUS.
    
    Capabilities:
    1. Pattern embedding index across all studies
    2. Cross-study similarity matching
    3. Pattern validation tracking
    4. Pattern transfer recommendations
    5. Study-specific adaptations
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analytics" / "cross_study_patterns"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.model = None
            self.embedding_dim = 384
        
        # Pattern storage
        self.study_patterns: Dict[str, List[StudyPattern]] = defaultdict(list)
        self.pattern_embeddings: Dict[str, np.ndarray] = {}
        self.pattern_matches: List[PatternMatch] = []
        self.validations: Dict[str, PatternValidation] = {}
        
        # Pattern type definitions
        self.pattern_types = {
            'coordinator_overload': {
                'description': 'Site coordinator managing too many subjects',
                'issue_types': ['sdv_incomplete', 'signature_gaps', 'open_queries'],
                'transferable': True,
                'adaptation_factors': ['site_size', 'therapeutic_area']
            },
            'pi_absence': {
                'description': 'Principal Investigator unavailable for signatures',
                'issue_types': ['signature_gaps', 'broken_signatures'],
                'transferable': True,
                'adaptation_factors': ['site_structure', 'delegation_log']
            },
            'query_cascade': {
                'description': 'Queries creating downstream signature blocks',
                'issue_types': ['open_queries', 'signature_gaps'],
                'transferable': True,
                'adaptation_factors': ['query_complexity', 'visit_schedule']
            },
            'sdv_backlog': {
                'description': 'Source data verification falling behind',
                'issue_types': ['sdv_incomplete'],
                'transferable': True,
                'adaptation_factors': ['monitoring_frequency', 'crf_complexity']
            },
            'sae_processing_delay': {
                'description': 'SAE reconciliation taking too long',
                'issue_types': ['sae_dm_pending', 'sae_safety_pending'],
                'transferable': True,
                'adaptation_factors': ['safety_team_size', 'sae_volume']
            },
            'coding_backlog': {
                'description': 'Medical coding falling behind',
                'issue_types': ['meddra_uncoded', 'whodrug_uncoded'],
                'transferable': True,
                'adaptation_factors': ['terminology_complexity', 'coder_availability']
            },
            'data_entry_lag': {
                'description': 'CRF data entry delayed at site',
                'issue_types': ['missing_pages', 'missing_visits'],
                'transferable': True,
                'adaptation_factors': ['visit_frequency', 'edc_complexity']
            },
            'end_of_month_rush': {
                'description': 'Spike in issues at month end',
                'issue_types': ['open_queries', 'sdv_incomplete', 'signature_gaps'],
                'transferable': True,
                'adaptation_factors': ['reporting_cycle', 'sponsor_deadlines']
            },
            'new_site_ramp': {
                'description': 'New site learning curve issues',
                'issue_types': ['open_queries', 'missing_pages', 'signature_gaps'],
                'transferable': True,
                'adaptation_factors': ['site_experience', 'training_completeness']
            },
            'cross_functional_gap': {
                'description': 'Issues at handoff points between teams',
                'issue_types': ['sae_dm_pending', 'edrr_issues'],
                'transferable': True,
                'adaptation_factors': ['team_structure', 'communication_channels']
            },
            'seasonal_variation': {
                'description': 'Performance variations during holidays/seasons',
                'issue_types': ['sdv_incomplete', 'signature_gaps', 'open_queries'],
                'transferable': False,
                'adaptation_factors': ['geography', 'cultural_calendar']
            },
            'system_integration_issue': {
                'description': 'Issues from EDC/safety system integration',
                'issue_types': ['edrr_issues', 'inactivated_forms'],
                'transferable': False,
                'adaptation_factors': ['system_versions', 'integration_type']
            },
            'regulatory_submission_pressure': {
                'description': 'Data quality issues near submission deadlines',
                'issue_types': ['open_queries', 'sdv_incomplete', 'signature_gaps'],
                'transferable': True,
                'adaptation_factors': ['submission_timeline', 'regulatory_region']
            }
        }
        
        logger.info("CrossStudyPatternMatcher initialized")
    
    def load_existing_patterns(self) -> int:
        """Load existing patterns from pattern library."""
        pattern_file = self.data_dir / "analytics" / "pattern_library" / "pattern_matches.parquet"
        
        if not pattern_file.exists():
            logger.warning(f"Pattern file not found: {pattern_file}")
            return 0
        
        df = pd.read_parquet(pattern_file)
        logger.info(f"Loaded {len(df)} pattern matches from library")
        
        # Group by study and pattern type
        for study_id in df['study_id'].unique():
            study_df = df[df['study_id'] == study_id]
            
            for pattern_id in study_df['pattern_id'].unique():
                pattern_df = study_df[study_df['pattern_id'] == pattern_id]
                
                # Get pattern type from pattern_id (e.g., "PAT-DQ-001" -> "query_overload")
                pattern_type = self._map_pattern_id_to_type(pattern_id)
                
                # Create StudyPattern
                pattern = StudyPattern(
                    pattern_id=f"{study_id}_{pattern_id}",
                    study_id=study_id,
                    pattern_type=pattern_type,
                    description=pattern_df['pattern_name'].iloc[0] if 'pattern_name' in pattern_df.columns else pattern_type,
                    issue_types=self._get_issue_types_for_pattern(pattern_type),
                    affected_sites=pattern_df['site_id'].nunique() if 'site_id' in pattern_df.columns else len(pattern_df),
                    affected_patients=len(pattern_df),
                    severity=pattern_df['severity'].iloc[0] if 'severity' in pattern_df.columns else 'Medium',
                    confidence=pattern_df['confidence'].mean() if 'confidence' in pattern_df.columns else 0.7,
                    first_detected=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat(),
                    features=self._extract_pattern_features(pattern_df)
                )
                
                self.study_patterns[study_id].append(pattern)
        
        total_patterns = sum(len(patterns) for patterns in self.study_patterns.values())
        logger.info(f"Created {total_patterns} study patterns across {len(self.study_patterns)} studies")
        
        return total_patterns
    
    def _map_pattern_id_to_type(self, pattern_id: str) -> str:
        """Map pattern ID to pattern type."""
        mapping = {
            'PAT-DQ-001': 'query_cascade',
            'PAT-DQ-002': 'sdv_backlog',
            'PAT-DQ-003': 'data_entry_lag',
            'PAT-DQ-004': 'coding_backlog',
            'PAT-RC-001': 'coordinator_overload',
            'PAT-RC-002': 'pi_absence',
            'PAT-SF-001': 'sae_processing_delay',
            'PAT-SF-002': 'sae_processing_delay',
            'PAT-TL-001': 'regulatory_submission_pressure',
            'PAT-SP-001': 'new_site_ramp',
            'PAT-SP-002': 'coordinator_overload',
            'PAT-CP-001': 'system_integration_issue',
            'PAT-CP-002': 'cross_functional_gap'
        }
        return mapping.get(pattern_id, 'unknown')
    
    def _get_issue_types_for_pattern(self, pattern_type: str) -> List[str]:
        """Get issue types associated with a pattern type."""
        if pattern_type in self.pattern_types:
            return self.pattern_types[pattern_type]['issue_types']
        return []
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from pattern data for similarity matching."""
        features = {
            'patient_count': len(df),
            'site_count': df['site_id'].nunique() if 'site_id' in df.columns else 1,
        }
        
        # Add numeric features if available
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['patient_count', 'site_count']:
                features[f'{col}_mean'] = float(df[col].mean())
                features[f'{col}_std'] = float(df[col].std()) if len(df) > 1 else 0.0
        
        return features
    
    def generate_pattern_embeddings(self) -> int:
        """Generate embeddings for all patterns."""
        if not self.model:
            logger.warning("No embedding model available. Using feature-based similarity.")
            return 0
        
        count = 0
        for study_id, patterns in self.study_patterns.items():
            for pattern in patterns:
                # Create text representation of pattern
                text = self._pattern_to_text(pattern)
                
                # Generate embedding
                embedding = self.model.encode(text)
                pattern.embedding = embedding.tolist()
                self.pattern_embeddings[pattern.pattern_id] = embedding
                count += 1
        
        logger.info(f"Generated {count} pattern embeddings")
        return count
    
    def _pattern_to_text(self, pattern: StudyPattern) -> str:
        """Convert pattern to text for embedding."""
        text_parts = [
            f"Pattern type: {pattern.pattern_type}",
            f"Description: {pattern.description}",
            f"Issue types: {', '.join(pattern.issue_types)}",
            f"Severity: {pattern.severity}",
            f"Affected patients: {pattern.affected_patients}",
            f"Affected sites: {pattern.affected_sites}"
        ]
        
        # Add features
        for key, value in pattern.features.items():
            text_parts.append(f"{key}: {value}")
        
        return ". ".join(text_parts)
    
    def find_cross_study_matches(self, similarity_threshold: float = 0.7) -> List[PatternMatch]:
        """Find similar patterns across different studies."""
        self.pattern_matches = []
        
        # Get all patterns as flat list
        all_patterns = []
        for study_id, patterns in self.study_patterns.items():
            all_patterns.extend(patterns)
        
        if len(all_patterns) < 2:
            logger.warning("Not enough patterns for cross-study matching")
            return []
        
        # Compare each pair of patterns from different studies
        for i, pattern1 in enumerate(all_patterns):
            for j, pattern2 in enumerate(all_patterns[i+1:], i+1):
                # Skip same study patterns
                if pattern1.study_id == pattern2.study_id:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(pattern1, pattern2)
                
                if similarity >= similarity_threshold:
                    # Determine match type
                    if similarity >= 0.95:
                        match_type = 'exact'
                    elif similarity >= 0.85:
                        match_type = 'similar'
                    else:
                        match_type = 'related'
                    
                    # Check transferability
                    transferable, adaptations = self._check_transferability(pattern1, pattern2)
                    
                    match = PatternMatch(
                        source_pattern_id=pattern1.pattern_id,
                        source_study_id=pattern1.study_id,
                        target_pattern_id=pattern2.pattern_id,
                        target_study_id=pattern2.study_id,
                        similarity_score=similarity,
                        match_type=match_type,
                        transferable=transferable,
                        adaptation_needed=adaptations,
                        confidence=min(pattern1.confidence, pattern2.confidence) * similarity
                    )
                    self.pattern_matches.append(match)
        
        logger.info(f"Found {len(self.pattern_matches)} cross-study pattern matches")
        return self.pattern_matches
    
    def _calculate_similarity(self, pattern1: StudyPattern, pattern2: StudyPattern) -> float:
        """Calculate similarity between two patterns."""
        # Embedding-based similarity if available
        if pattern1.pattern_id in self.pattern_embeddings and pattern2.pattern_id in self.pattern_embeddings:
            emb1 = self.pattern_embeddings[pattern1.pattern_id]
            emb2 = self.pattern_embeddings[pattern2.pattern_id]
            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(cosine_sim)
        
        # Feature-based similarity as fallback
        score = 0.0
        weights = {'pattern_type': 0.4, 'issue_types': 0.3, 'severity': 0.2, 'features': 0.1}
        
        # Pattern type match
        if pattern1.pattern_type == pattern2.pattern_type:
            score += weights['pattern_type']
        
        # Issue types overlap
        overlap = len(set(pattern1.issue_types) & set(pattern2.issue_types))
        union = len(set(pattern1.issue_types) | set(pattern2.issue_types))
        if union > 0:
            score += weights['issue_types'] * (overlap / union)
        
        # Severity match
        if pattern1.severity == pattern2.severity:
            score += weights['severity']
        
        # Feature similarity
        common_features = set(pattern1.features.keys()) & set(pattern2.features.keys())
        if common_features:
            feature_sim = 0.0
            for feat in common_features:
                v1, v2 = pattern1.features[feat], pattern2.features[feat]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if max(abs(v1), abs(v2)) > 0:
                        feature_sim += 1 - abs(v1 - v2) / max(abs(v1), abs(v2))
            score += weights['features'] * (feature_sim / len(common_features))
        
        return score
    
    def _check_transferability(self, source: StudyPattern, target: StudyPattern) -> Tuple[bool, List[str]]:
        """Check if a pattern solution can be transferred between studies."""
        adaptations = []
        
        # Check if pattern type is inherently transferable
        if source.pattern_type in self.pattern_types:
            type_info = self.pattern_types[source.pattern_type]
            if not type_info['transferable']:
                return False, ["Pattern type not transferable across studies"]
            
            # Check adaptation factors
            adaptations = type_info['adaptation_factors']
        
        # Check feature differences that might require adaptation
        if 'patient_count_mean' in source.features and 'patient_count_mean' in target.features:
            ratio = source.features['patient_count_mean'] / max(target.features['patient_count_mean'], 1)
            if ratio < 0.5 or ratio > 2.0:
                adaptations.append("scale_adjustment")
        
        # Consider severity differences
        severity_order = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        source_sev = severity_order.get(source.severity, 2)
        target_sev = severity_order.get(target.severity, 2)
        if abs(source_sev - target_sev) > 1:
            adaptations.append("severity_calibration")
        
        return True, adaptations
    
    def generate_transfer_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for transferring successful patterns."""
        recommendations = []
        
        # Group matches by target study
        by_target = defaultdict(list)
        for match in self.pattern_matches:
            if match.transferable and match.confidence >= 0.6:
                by_target[match.target_study_id].append(match)
        
        for target_study, matches in by_target.items():
            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            for match in matches[:5]:  # Top 5 per study
                # Find source pattern
                source_pattern = None
                for patterns in self.study_patterns.values():
                    for p in patterns:
                        if p.pattern_id == match.source_pattern_id:
                            source_pattern = p
                            break
                
                if source_pattern:
                    rec = {
                        'target_study': target_study,
                        'source_study': match.source_study_id,
                        'pattern_type': source_pattern.pattern_type,
                        'description': source_pattern.description,
                        'similarity': round(match.similarity_score, 3),
                        'confidence': round(match.confidence, 3),
                        'adaptations_needed': match.adaptation_needed,
                        'expected_impact': self._estimate_impact(source_pattern),
                        'implementation_steps': self._get_implementation_steps(source_pattern),
                        'priority': self._calculate_priority(match, source_pattern)
                    }
                    recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} transfer recommendations")
        return recommendations
    
    def _estimate_impact(self, pattern: StudyPattern) -> Dict[str, Any]:
        """Estimate impact of applying a pattern solution."""
        return {
            'patients_affected': pattern.affected_patients,
            'sites_affected': pattern.affected_sites,
            'estimated_reduction': f"{int(pattern.resolution_rate * 100)}%" if pattern.resolution_rate > 0 else "TBD",
            'estimated_days': pattern.avg_resolution_days if pattern.avg_resolution_days > 0 else "TBD"
        }
    
    def _get_implementation_steps(self, pattern: StudyPattern) -> List[str]:
        """Get implementation steps for a pattern type."""
        steps_by_type = {
            'coordinator_overload': [
                "1. Assess current coordinator-to-subject ratio",
                "2. Identify high-volume sites needing support",
                "3. Implement workload balancing or additional resources",
                "4. Set up weekly check-ins for overloaded sites"
            ],
            'pi_absence': [
                "1. Review delegation log for signature authority",
                "2. Ensure sub-investigators are trained and authorized",
                "3. Implement batch signing schedules",
                "4. Set up PI absence notification system"
            ],
            'query_cascade': [
                "1. Prioritize blocking queries in resolution queue",
                "2. Clear queries preventing signature completion",
                "3. Implement cascading priority algorithm",
                "4. Monitor downstream unblocking"
            ],
            'sdv_backlog': [
                "1. Calculate current SDV completion rate",
                "2. Prioritize subjects by DB Lock eligibility",
                "3. Schedule focused SDV visits",
                "4. Consider remote SDV options"
            ],
            'sae_processing_delay': [
                "1. Review SAE reconciliation SLAs",
                "2. Identify bottlenecks in DM-Safety handoff",
                "3. Implement daily reconciliation meetings",
                "4. Set up escalation for aging SAEs"
            ]
        }
        
        return steps_by_type.get(pattern.pattern_type, [
            "1. Analyze pattern root cause",
            "2. Develop mitigation strategy",
            "3. Implement corrective actions",
            "4. Monitor effectiveness"
        ])
    
    def _calculate_priority(self, match: PatternMatch, pattern: StudyPattern) -> str:
        """Calculate priority for a transfer recommendation."""
        score = match.confidence * 0.4
        
        # Factor in severity
        severity_scores = {'Critical': 1.0, 'High': 0.75, 'Medium': 0.5, 'Low': 0.25}
        score += severity_scores.get(pattern.severity, 0.5) * 0.3
        
        # Factor in affected population
        if pattern.affected_patients >= 100:
            score += 0.3
        elif pattern.affected_patients >= 50:
            score += 0.2
        elif pattern.affected_patients >= 20:
            score += 0.1
        
        if score >= 0.8:
            return 'Critical'
        elif score >= 0.6:
            return 'High'
        elif score >= 0.4:
            return 'Medium'
        return 'Low'
    
    def validate_pattern(self, pattern_id: str, status: str, validated_by: str,
                         effectiveness_score: float, notes: str = "") -> PatternValidation:
        """Record validation for a pattern."""
        # Find pattern
        pattern = None
        study_id = None
        for sid, patterns in self.study_patterns.items():
            for p in patterns:
                if p.pattern_id == pattern_id:
                    pattern = p
                    study_id = sid
                    break
        
        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_id}")
        
        validation = PatternValidation(
            pattern_id=pattern_id,
            study_id=study_id,
            validation_status=status,
            validated_by=validated_by,
            validation_date=datetime.now().isoformat(),
            effectiveness_score=effectiveness_score,
            notes=notes
        )
        
        self.validations[pattern_id] = validation
        
        # Update pattern resolution rate if validated
        if status == 'validated' and effectiveness_score > 0:
            pattern.resolution_rate = effectiveness_score
        
        logger.info(f"Validated pattern {pattern_id}: {status} ({effectiveness_score:.2f})")
        return validation
    
    def get_study_pattern_summary(self, study_id: str) -> Dict[str, Any]:
        """Get pattern summary for a specific study."""
        if study_id not in self.study_patterns:
            return {'error': f'Study not found: {study_id}'}
        
        patterns = self.study_patterns[study_id]
        
        # Find matches for this study
        incoming_matches = [m for m in self.pattern_matches if m.target_study_id == study_id]
        outgoing_matches = [m for m in self.pattern_matches if m.source_study_id == study_id]
        
        return {
            'study_id': study_id,
            'total_patterns': len(patterns),
            'pattern_types': list(set(p.pattern_type for p in patterns)),
            'total_affected_patients': sum(p.affected_patients for p in patterns),
            'total_affected_sites': sum(p.affected_sites for p in patterns),
            'severity_distribution': self._count_by_key(patterns, 'severity'),
            'incoming_matches': len(incoming_matches),
            'outgoing_matches': len(outgoing_matches),
            'validated_patterns': len([p for p in patterns if p.pattern_id in self.validations])
        }
    
    def _count_by_key(self, items: List, key: str) -> Dict[str, int]:
        """Count items by a key attribute."""
        counts = defaultdict(int)
        for item in items:
            value = getattr(item, key, 'Unknown')
            counts[value] += 1
        return dict(counts)
    
    def generate_cross_study_report(self) -> Dict[str, Any]:
        """Generate comprehensive cross-study pattern report."""
        all_patterns = []
        for patterns in self.study_patterns.values():
            all_patterns.extend(patterns)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_studies': len(self.study_patterns),
                'total_patterns': len(all_patterns),
                'total_matches': len(self.pattern_matches),
                'validated_patterns': len(self.validations),
                'transferable_matches': len([m for m in self.pattern_matches if m.transferable])
            },
            'by_pattern_type': {},
            'by_study': {},
            'top_matches': [],
            'validation_summary': {
                'validated': len([v for v in self.validations.values() if v.validation_status == 'validated']),
                'pending': len([v for v in self.validations.values() if v.validation_status == 'pending']),
                'rejected': len([v for v in self.validations.values() if v.validation_status == 'rejected'])
            }
        }
        
        # By pattern type
        for pattern_type in set(p.pattern_type for p in all_patterns):
            type_patterns = [p for p in all_patterns if p.pattern_type == pattern_type]
            report['by_pattern_type'][pattern_type] = {
                'count': len(type_patterns),
                'studies': list(set(p.study_id for p in type_patterns)),
                'total_patients': sum(p.affected_patients for p in type_patterns),
                'avg_confidence': np.mean([p.confidence for p in type_patterns]) if type_patterns else 0
            }
        
        # By study
        for study_id in self.study_patterns:
            report['by_study'][study_id] = self.get_study_pattern_summary(study_id)
        
        # Top matches
        sorted_matches = sorted(self.pattern_matches, key=lambda x: x.confidence, reverse=True)
        for match in sorted_matches[:20]:
            report['top_matches'].append({
                'source': f"{match.source_study_id}/{match.source_pattern_id}",
                'target': f"{match.target_study_id}/{match.target_pattern_id}",
                'similarity': round(match.similarity_score, 3),
                'confidence': round(match.confidence, 3),
                'match_type': match.match_type,
                'transferable': match.transferable
            })
        
        return report
    
    def save_results(self) -> Dict[str, str]:
        """Save all results to files."""
        saved_files = {}
        
        # 1. Pattern embeddings
        if self.pattern_embeddings:
            embeddings_array = np.array(list(self.pattern_embeddings.values()))
            np.save(self.output_dir / "pattern_embeddings.npy", embeddings_array)
            
            # Save pattern ID mapping
            with open(self.output_dir / "pattern_embedding_ids.json", 'w') as f:
                json.dump(list(self.pattern_embeddings.keys()), f, indent=2)
            saved_files['embeddings'] = str(self.output_dir / "pattern_embeddings.npy")
        
        # 2. Study patterns
        patterns_data = []
        for study_id, patterns in self.study_patterns.items():
            for p in patterns:
                pattern_dict = asdict(p)
                pattern_dict.pop('embedding', None)  # Don't save embeddings in JSON
                patterns_data.append(pattern_dict)
        
        with open(self.output_dir / "study_patterns.json", 'w') as f:
            json.dump(patterns_data, f, indent=2, default=str)
        saved_files['patterns'] = str(self.output_dir / "study_patterns.json")
        
        # 3. Pattern matches
        if self.pattern_matches:
            matches_data = [asdict(m) for m in self.pattern_matches]
            with open(self.output_dir / "pattern_matches.json", 'w') as f:
                json.dump(matches_data, f, indent=2)
            
            # Also save as CSV
            matches_df = pd.DataFrame(matches_data)
            matches_df.to_csv(self.output_dir / "pattern_matches.csv", index=False)
            saved_files['matches'] = str(self.output_dir / "pattern_matches.csv")
        
        # 4. Transfer recommendations
        recommendations = self.generate_transfer_recommendations()
        if recommendations:
            with open(self.output_dir / "transfer_recommendations.json", 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            # CSV version
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_csv(self.output_dir / "transfer_recommendations.csv", index=False)
            saved_files['recommendations'] = str(self.output_dir / "transfer_recommendations.csv")
        
        # 5. Validations
        if self.validations:
            validations_data = {k: asdict(v) for k, v in self.validations.items()}
            with open(self.output_dir / "pattern_validations.json", 'w') as f:
                json.dump(validations_data, f, indent=2)
            saved_files['validations'] = str(self.output_dir / "pattern_validations.json")
        
        # 6. Cross-study report
        report = self.generate_cross_study_report()
        with open(self.output_dir / "cross_study_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        saved_files['report'] = str(self.output_dir / "cross_study_report.json")
        
        # 7. Summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'statistics': {
                'total_studies': len(self.study_patterns),
                'total_patterns': sum(len(p) for p in self.study_patterns.values()),
                'total_embeddings': len(self.pattern_embeddings),
                'total_matches': len(self.pattern_matches),
                'total_validations': len(self.validations),
                'total_recommendations': len(recommendations) if recommendations else 0
            },
            'files_saved': saved_files
        }
        
        with open(self.output_dir / "cross_study_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(saved_files)} files to {self.output_dir}")
        return saved_files


def main():
    """Main execution function."""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - Phase 4.5: Cross-Study Pattern Matcher")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Initialize matcher
    matcher = CrossStudyPatternMatcher(data_dir="data/processed")
    
    # Step 1: Load existing patterns
    print("\n📊 Step 1: Loading existing patterns...")
    pattern_count = matcher.load_existing_patterns()
    print(f"   ✅ Loaded patterns from {len(matcher.study_patterns)} studies")
    print(f"   ✅ Total patterns: {pattern_count}")
    
    # Step 2: Generate embeddings
    print("\n🧠 Step 2: Generating pattern embeddings...")
    embedding_count = matcher.generate_pattern_embeddings()
    print(f"   ✅ Generated {embedding_count} embeddings")
    
    # Step 3: Find cross-study matches
    print("\n🔗 Step 3: Finding cross-study pattern matches...")
    matches = matcher.find_cross_study_matches(similarity_threshold=0.7)
    print(f"   ✅ Found {len(matches)} cross-study matches")
    
    # Breakdown by match type
    match_types = {}
    for m in matches:
        match_types[m.match_type] = match_types.get(m.match_type, 0) + 1
    for mt, count in sorted(match_types.items()):
        print(f"      - {mt}: {count}")
    
    # Step 4: Generate transfer recommendations
    print("\n📋 Step 4: Generating transfer recommendations...")
    recommendations = matcher.generate_transfer_recommendations()
    print(f"   ✅ Generated {len(recommendations)} transfer recommendations")
    
    # Show top recommendations
    if recommendations:
        print("\n   Top 5 Transfer Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec['source_study']} → {rec['target_study']}")
            print(f"      Pattern: {rec['pattern_type']}")
            print(f"      Confidence: {rec['confidence']:.2f}, Priority: {rec['priority']}")
    
    # Step 5: Generate report
    print("\n📈 Step 5: Generating cross-study report...")
    report = matcher.generate_cross_study_report()
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-STUDY PATTERN SUMMARY")
    print("=" * 70)
    print(f"Studies Analyzed:        {report['summary']['total_studies']}")
    print(f"Total Patterns:          {report['summary']['total_patterns']}")
    print(f"Cross-Study Matches:     {report['summary']['total_matches']}")
    print(f"Transferable Matches:    {report['summary']['transferable_matches']}")
    
    # Pattern types
    print("\n📊 Patterns by Type:")
    for ptype, info in sorted(report['by_pattern_type'].items(), 
                               key=lambda x: x[1]['count'], reverse=True):
        print(f"   {ptype}: {info['count']} patterns across {len(info['studies'])} studies")
    
    # Step 6: Save results
    print("\n💾 Step 6: Saving results...")
    saved_files = matcher.save_results()
    print(f"   ✅ Saved {len(saved_files)} files:")
    for name, path in saved_files.items():
        print(f"      - {name}: {path}")
    
    # Duration
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4.5 COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Output: data/processed/analytics/cross_study_patterns/")
    
    return matcher


if __name__ == "__main__":
    main()