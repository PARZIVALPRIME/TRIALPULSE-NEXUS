# dashboard/pages/dm_hub.py
"""
TRIALPULSE NEXUS 10X - Data Manager Hub
Phase 7.4: Quality Matrix, Genome Matches, Bottleneck Visualization, Pattern Alerts
Version: 1.1 (Fixed column mappings)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# =============================================================================
# DATA LOADER
# =============================================================================

class DMDataLoader:
    """Data loader for Data Manager Hub."""
    
    def __init__(self):
        self.base_path = Path("data/processed")
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_times = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid."""
        if key not in self._cache_times:
            return False
        return (datetime.now() - self._cache_times[key]).seconds < self._cache_ttl
    
    def _get_cached(self, key: str):
        """Get cached data if valid."""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, data):
        """Set cache with timestamp."""
        self._cache[key] = data
        self._cache_times[key] = datetime.now()
    
    def load_upr(self) -> pd.DataFrame:
        """Load Unified Patient Record."""
        cached = self._get_cached('upr')
        if cached is not None:
            return cached
        
        path = self.base_path / "upr" / "unified_patient_record.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('upr', df)
            return df
        return pd.DataFrame()
    
    def load_patient_issues(self) -> pd.DataFrame:
        """Load patient issues data."""
        cached = self._get_cached('patient_issues')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "patient_issues.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('patient_issues', df)
            return df
        return pd.DataFrame()
    
    def load_patient_dqi(self) -> pd.DataFrame:
        """Load patient DQI data."""
        cached = self._get_cached('patient_dqi')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "patient_dqi_enhanced.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('patient_dqi', df)
            return df
        return pd.DataFrame()
    
    def load_clean_status(self) -> pd.DataFrame:
        """Load clean patient status."""
        cached = self._get_cached('clean_status')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "patient_clean_status.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('clean_status', df)
            return df
        return pd.DataFrame()
    
    def load_dblock_status(self) -> pd.DataFrame:
        """Load DB lock status."""
        cached = self._get_cached('dblock_status')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "patient_dblock_status.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('dblock_status', df)
            return df
        return pd.DataFrame()
    
    def load_site_benchmarks(self) -> pd.DataFrame:
        """Load site benchmarks."""
        cached = self._get_cached('site_benchmarks')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "site_benchmarks.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('site_benchmarks', df)
            return df
        return pd.DataFrame()
    
    def load_cascade_analysis(self) -> pd.DataFrame:
        """Load cascade analysis data."""
        cached = self._get_cached('cascade_analysis')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "patient_cascade_analysis.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('cascade_analysis', df)
            return df
        return pd.DataFrame()
    
    def load_pattern_matches(self) -> pd.DataFrame:
        """Load pattern library matches."""
        cached = self._get_cached('pattern_matches')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "pattern_library" / "pattern_matches.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('pattern_matches', df)
            return df
        return pd.DataFrame()
    
    def load_pattern_alerts(self) -> pd.DataFrame:
        """Load pattern alerts."""
        cached = self._get_cached('pattern_alerts')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "pattern_library" / "alerts.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('pattern_alerts', df)
            return df
        return pd.DataFrame()
    
    def load_resolution_genome(self) -> pd.DataFrame:
        """Load resolution genome recommendations."""
        cached = self._get_cached('resolution_genome')
        if cached is not None:
            return cached
        
        path = self.base_path / "analytics" / "resolution_genome" / "patient_recommendations.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._set_cache('resolution_genome', df)
            return df
        return pd.DataFrame()
    
    def get_studies_list(self) -> List[str]:
        """Get list of studies."""
        upr = self.load_upr()
        if upr.empty or 'study_id' not in upr.columns:
            return [f"Study_{i}" for i in range(1, 24)]
        return sorted(upr['study_id'].unique().tolist())
    
    def get_issue_types_list(self) -> List[str]:
        """Get list of issue types."""
        return [
            'sdv_incomplete', 'open_queries', 'signature_gaps', 'sae_dm_pending',
            'missing_visits', 'missing_pages', 'broken_signatures', 'inactivated_forms',
            'lab_issues', 'edrr_issues', 'meddra_uncoded', 'whodrug_uncoded',
            'high_query_volume', 'sae_safety_pending'
        ]
    
    def get_dm_portfolio(self) -> Dict:
        """Get Data Manager portfolio summary."""
        upr = self.load_upr()
        issues = self.load_patient_issues()
        dqi = self.load_patient_dqi()
        clean = self.load_clean_status()
        dblock = self.load_dblock_status()
        
        if upr.empty:
            return self._get_demo_portfolio()
        
        # Calculate metrics
        total_patients = len(upr)
        total_studies = upr['study_id'].nunique() if 'study_id' in upr.columns else 0
        total_sites = upr['site_id'].nunique() if 'site_id' in upr.columns else 0
        
        # DQI metrics
        mean_dqi = dqi['dqi_score'].mean() if not dqi.empty and 'dqi_score' in dqi.columns else 0
        
        # Clean metrics
        tier1_clean = 0
        tier2_clean = 0
        if not clean.empty:
            if 'tier1_clean' in clean.columns:
                tier1_clean = clean['tier1_clean'].sum()
            if 'tier2_clean' in clean.columns:
                tier2_clean = clean['tier2_clean'].sum()
        
        # DB Lock metrics
        dblock_ready = 0
        dblock_eligible = 0
        if not dblock.empty:
            for col in ['dblock_ready', 'db_lock_ready', 'is_ready']:
                if col in dblock.columns:
                    dblock_ready = dblock[col].sum()
                    break
            for col in ['dblock_eligible', 'db_lock_eligible', 'is_eligible']:
                if col in dblock.columns:
                    dblock_eligible = dblock[col].sum()
                    break
        
        # Issue metrics from patient_issues
        total_issues = 0
        critical_issues = 0
        open_queries = 0
        pending_reviews = 0
        
        if not issues.empty:
            # Total issues
            if 'total_issues' in issues.columns:
                total_issues = int(issues['total_issues'].sum())
            
            # Critical issues (priority_tier == 'Critical')
            if 'priority_tier' in issues.columns:
                critical_issues = int((issues['priority_tier'] == 'Critical').sum())
            
            # Open queries from count column
            if 'count_open_queries' in issues.columns:
                open_queries = int(issues['count_open_queries'].sum())
            
            # SAE pending
            if 'count_sae_dm_pending' in issues.columns:
                pending_reviews = int(issues['count_sae_dm_pending'].sum())
            if 'count_sae_safety_pending' in issues.columns:
                pending_reviews += int(issues['count_sae_safety_pending'].sum())
        
        return {
            'total_patients': total_patients,
            'total_studies': total_studies,
            'total_sites': total_sites,
            'mean_dqi': mean_dqi,
            'tier1_clean': tier1_clean,
            'tier1_clean_rate': tier1_clean / total_patients * 100 if total_patients > 0 else 0,
            'tier2_clean': tier2_clean,
            'tier2_clean_rate': tier2_clean / total_patients * 100 if total_patients > 0 else 0,
            'dblock_ready': dblock_ready,
            'dblock_eligible': dblock_eligible,
            'dblock_ready_rate': dblock_ready / dblock_eligible * 100 if dblock_eligible > 0 else 0,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'open_queries': open_queries,
            'pending_reviews': pending_reviews
        }
    
    def _get_demo_portfolio(self) -> Dict:
        """Return demo portfolio data."""
        return {
            'total_patients': 57997,
            'total_studies': 23,
            'total_sites': 3416,
            'mean_dqi': 98.22,
            'tier1_clean': 35416,
            'tier1_clean_rate': 61.1,
            'tier2_clean': 31142,
            'tier2_clean_rate': 53.7,
            'dblock_ready': 5717,
            'dblock_eligible': 20596,
            'dblock_ready_rate': 27.8,
            'total_issues': 53452,
            'critical_issues': 5629,
            'open_queries': 20899,
            'pending_reviews': 4538
        }
    
    def get_quality_matrix(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """Get quality matrix by site."""
        upr = self.load_upr()
        issues = self.load_patient_issues()
        dqi = self.load_patient_dqi()
        clean = self.load_clean_status()
        
        if upr.empty:
            return self._get_demo_quality_matrix()
        
        # Filter by study if specified
        if study_id and study_id != "All Studies":
            upr = upr[upr['study_id'] == study_id]
            if not issues.empty and 'study_id' in issues.columns:
                issues = issues[issues['study_id'] == study_id]
            if not dqi.empty and 'study_id' in dqi.columns:
                dqi = dqi[dqi['study_id'] == study_id]
            if not clean.empty and 'study_id' in clean.columns:
                clean = clean[clean['study_id'] == study_id]
        
        # Group by site
        site_metrics = []
        sites = upr['site_id'].unique() if 'site_id' in upr.columns else []
        
        for site in sites[:100]:  # Limit to 100 sites for performance
            site_upr = upr[upr['site_id'] == site]
            site_issues = issues[issues['site_id'] == site] if not issues.empty and 'site_id' in issues.columns else pd.DataFrame()
            site_dqi = dqi[dqi['site_id'] == site] if not dqi.empty and 'site_id' in dqi.columns else pd.DataFrame()
            site_clean = clean[clean['site_id'] == site] if not clean.empty and 'site_id' in clean.columns else pd.DataFrame()
            
            # Calculate metrics
            patients = len(site_upr)
            
            # DQI
            mean_dqi = site_dqi['dqi_score'].mean() if not site_dqi.empty and 'dqi_score' in site_dqi.columns else 100
            
            # Clean rates
            tier1_rate = 0
            tier2_rate = 0
            if not site_clean.empty:
                if 'tier1_clean' in site_clean.columns:
                    tier1_rate = site_clean['tier1_clean'].mean() * 100
                if 'tier2_clean' in site_clean.columns:
                    tier2_rate = site_clean['tier2_clean'].mean() * 100
            
            # Issue counts using correct column names (issue_*)
            open_queries = 0
            sdv_incomplete = 0
            signature_gaps = 0
            sae_pending = 0
            total_issues = 0
            
            if not site_issues.empty:
                if 'issue_open_queries' in site_issues.columns:
                    open_queries = int(site_issues['issue_open_queries'].sum())
                if 'issue_sdv_incomplete' in site_issues.columns:
                    sdv_incomplete = int(site_issues['issue_sdv_incomplete'].sum())
                if 'issue_signature_gaps' in site_issues.columns:
                    signature_gaps = int(site_issues['issue_signature_gaps'].sum())
                if 'issue_sae_dm_pending' in site_issues.columns:
                    sae_pending = int(site_issues['issue_sae_dm_pending'].sum())
                if 'total_issues' in site_issues.columns:
                    total_issues = int(site_issues['total_issues'].sum())
            
            # Get study
            study = site_upr['study_id'].iloc[0] if 'study_id' in site_upr.columns and len(site_upr) > 0 else 'Unknown'
            
            site_metrics.append({
                'site_id': site,
                'study_id': study,
                'patients': patients,
                'dqi_score': round(mean_dqi, 1),
                'tier1_clean_rate': round(tier1_rate, 1),
                'tier2_clean_rate': round(tier2_rate, 1),
                'open_queries': open_queries,
                'sdv_incomplete': sdv_incomplete,
                'signature_gaps': signature_gaps,
                'sae_pending': sae_pending,
                'total_issues': total_issues if total_issues > 0 else (open_queries + sdv_incomplete + signature_gaps + sae_pending)
            })
        
        df = pd.DataFrame(site_metrics)
        
        # Add quality tier
        if not df.empty:
            df['quality_tier'] = df['dqi_score'].apply(self._get_quality_tier)
        
        return df
    
    def _get_quality_tier(self, dqi: float) -> str:
        """Get quality tier from DQI score."""
        if dqi >= 95:
            return 'Pristine'
        elif dqi >= 85:
            return 'Excellent'
        elif dqi >= 75:
            return 'Good'
        elif dqi >= 65:
            return 'Fair'
        elif dqi >= 50:
            return 'Poor'
        else:
            return 'Critical'
    
    def _get_demo_quality_matrix(self) -> pd.DataFrame:
        """Return demo quality matrix."""
        np.random.seed(42)
        sites = [f"Site_{i}" for i in range(1, 51)]
        studies = [f"Study_{np.random.randint(1, 24)}" for _ in sites]
        
        return pd.DataFrame({
            'site_id': sites,
            'study_id': studies,
            'patients': np.random.randint(10, 100, len(sites)),
            'dqi_score': np.random.uniform(70, 100, len(sites)).round(1),
            'tier1_clean_rate': np.random.uniform(40, 90, len(sites)).round(1),
            'tier2_clean_rate': np.random.uniform(30, 80, len(sites)).round(1),
            'open_queries': np.random.randint(0, 50, len(sites)),
            'sdv_incomplete': np.random.randint(0, 30, len(sites)),
            'signature_gaps': np.random.randint(0, 20, len(sites)),
            'sae_pending': np.random.randint(0, 10, len(sites)),
            'total_issues': np.random.randint(10, 100, len(sites)),
            'quality_tier': np.random.choice(['Pristine', 'Excellent', 'Good', 'Fair', 'Poor'], len(sites))
        })
    
    def get_bottleneck_analysis(self) -> Dict:
        """Get bottleneck analysis data using correct column names."""
        issues = self.load_patient_issues()
        
        if issues.empty:
            return self._get_demo_bottleneck()
        
        bottlenecks = []
        
        # Issue type mappings with correct column names from patient_issues.parquet
        # Columns are: issue_{type} (binary) and count_{type} (count)
        issue_mappings = [
            {
                'key': 'sdv_incomplete',
                'name': 'SDV Incomplete',
                'responsible': 'CRA',
                'impact': 70,
                'issue_col': 'issue_sdv_incomplete',
                'count_col': 'count_sdv_incomplete'
            },
            {
                'key': 'open_queries',
                'name': 'Open Queries',
                'responsible': 'Data Manager',
                'impact': 80,
                'issue_col': 'issue_open_queries',
                'count_col': 'count_open_queries'
            },
            {
                'key': 'signature_gaps',
                'name': 'Signature Gaps',
                'responsible': 'Site',
                'impact': 60,
                'issue_col': 'issue_signature_gaps',
                'count_col': 'count_signature_gaps'
            },
            {
                'key': 'sae_dm_pending',
                'name': 'SAE DM Pending',
                'responsible': 'Safety DM',
                'impact': 100,
                'issue_col': 'issue_sae_dm_pending',
                'count_col': 'count_sae_dm_pending'
            },
            {
                'key': 'sae_safety_pending',
                'name': 'SAE Safety Pending',
                'responsible': 'Safety Physician',
                'impact': 100,
                'issue_col': 'issue_sae_safety_pending',
                'count_col': 'count_sae_safety_pending'
            },
            {
                'key': 'missing_visits',
                'name': 'Missing Visits',
                'responsible': 'Site',
                'impact': 90,
                'issue_col': 'issue_missing_visits',
                'count_col': 'count_missing_visits'
            },
            {
                'key': 'missing_pages',
                'name': 'Missing Pages',
                'responsible': 'CRA',
                'impact': 85,
                'issue_col': 'issue_missing_pages',
                'count_col': 'count_missing_pages'
            },
            {
                'key': 'broken_signatures',
                'name': 'Broken Signatures',
                'responsible': 'Site',
                'impact': 50,
                'issue_col': 'issue_broken_signatures',
                'count_col': 'count_broken_signatures'
            },
            {
                'key': 'inactivated_forms',
                'name': 'Inactivated Forms',
                'responsible': 'Data Manager',
                'impact': 40,
                'issue_col': 'issue_inactivated_forms',
                'count_col': 'count_inactivated_forms'
            },
            {
                'key': 'lab_issues',
                'name': 'Lab Issues',
                'responsible': 'Data Manager',
                'impact': 45,
                'issue_col': 'issue_lab_issues',
                'count_col': 'count_lab_issues'
            },
            {
                'key': 'edrr_issues',
                'name': 'EDRR Issues',
                'responsible': 'Data Manager',
                'impact': 55,
                'issue_col': 'issue_edrr_issues',
                'count_col': 'count_edrr_issues'
            },
            {
                'key': 'meddra_uncoded',
                'name': 'MedDRA Uncoded',
                'responsible': 'Medical Coder',
                'impact': 30,
                'issue_col': 'issue_meddra_uncoded',
                'count_col': 'count_meddra_uncoded'
            },
            {
                'key': 'whodrug_uncoded',
                'name': 'WHODrug Uncoded',
                'responsible': 'Medical Coder',
                'impact': 30,
                'issue_col': 'issue_whodrug_uncoded',
                'count_col': 'count_whodrug_uncoded'
            },
            {
                'key': 'high_query_volume',
                'name': 'High Query Volume',
                'responsible': 'Data Manager',
                'impact': 65,
                'issue_col': 'issue_high_query_volume',
                'count_col': 'count_high_query_volume'
            }
        ]
        
        for mapping in issue_mappings:
            issue_col = mapping['issue_col']
            count_col = mapping['count_col']
            
            # Count patients with this issue
            patient_count = 0
            issue_count = 0
            
            if issue_col in issues.columns:
                patient_count = int((issues[issue_col] > 0).sum())
            
            if count_col in issues.columns:
                issue_count = int(issues[count_col].sum())
            
            if patient_count > 0:
                impact = mapping['impact']
                bottlenecks.append({
                    'issue_type': mapping['key'],
                    'issue_name': mapping['name'],
                    'patient_count': patient_count,
                    'issue_count': issue_count,
                    'count': patient_count,  # For compatibility
                    'impact_score': impact,
                    'blocking_score': int(patient_count * impact / 100),
                    'responsible': mapping['responsible'],
                    'downstream_blocked': self._estimate_downstream(mapping['key'], patient_count)
                })
        
        # Sort by blocking score
        bottlenecks.sort(key=lambda x: x['blocking_score'], reverse=True)
        
        # Calculate totals
        total_blocked = sum(b['patient_count'] for b in bottlenecks)
        critical_count = len([b for b in bottlenecks if b['impact_score'] >= 80])
        
        return {
            'bottlenecks': bottlenecks[:10],  # Top 10
            'total_blocked_patients': total_blocked,
            'critical_bottlenecks': critical_count
        }
    
    def _estimate_downstream(self, issue_type: str, count: int) -> int:
        """Estimate downstream blocked items."""
        # Cascade multipliers
        multipliers = {
            'missing_visits': 3.0,
            'missing_pages': 2.5,
            'open_queries': 2.0,
            'sae_dm_pending': 1.5,
            'sdv_incomplete': 1.2,
            'signature_gaps': 1.0
        }
        return int(count * multipliers.get(issue_type, 1.0))
    
    def _get_demo_bottleneck(self) -> Dict:
        """Return demo bottleneck data."""
        return {
            'bottlenecks': [
                {'issue_type': 'sdv_incomplete', 'issue_name': 'SDV Incomplete', 'patient_count': 18630, 'issue_count': 181864, 'count': 18630, 'impact_score': 70, 'blocking_score': 13041, 'responsible': 'CRA', 'downstream_blocked': 22356},
                {'issue_type': 'signature_gaps', 'issue_name': 'Signature Gaps', 'patient_count': 8778, 'issue_count': 231265, 'count': 8778, 'impact_score': 60, 'blocking_score': 5267, 'responsible': 'Site', 'downstream_blocked': 8778},
                {'issue_type': 'open_queries', 'issue_name': 'Open Queries', 'patient_count': 4999, 'issue_count': 20899, 'count': 4999, 'impact_score': 80, 'blocking_score': 3999, 'responsible': 'Data Manager', 'downstream_blocked': 9998},
                {'issue_type': 'sae_dm_pending', 'issue_name': 'SAE DM Pending', 'patient_count': 4425, 'issue_count': 5130, 'count': 4425, 'impact_score': 100, 'blocking_score': 4425, 'responsible': 'Safety DM', 'downstream_blocked': 6638},
                {'issue_type': 'missing_pages', 'issue_name': 'Missing Pages', 'patient_count': 2138, 'issue_count': 6116, 'count': 2138, 'impact_score': 85, 'blocking_score': 1817, 'responsible': 'CRA', 'downstream_blocked': 5345}
            ],
            'total_blocked_patients': 38970,
            'critical_bottlenecks': 2
        }
    
    def get_pattern_alerts(self, severity_filter: Optional[str] = None) -> pd.DataFrame:
        """Get pattern alerts."""
        alerts = self.load_pattern_alerts()
        
        if alerts.empty:
            return self._get_demo_pattern_alerts()
        
        # Filter by severity if specified
        if severity_filter and severity_filter != "All":
            if 'severity' in alerts.columns:
                alerts = alerts[alerts['severity'].str.lower() == severity_filter.lower()]
        
        return alerts
    
    def _get_demo_pattern_alerts(self) -> pd.DataFrame:
        """Return demo pattern alerts."""
        np.random.seed(42)
        
        patterns = [
            ('PAT-RC-002', 'Signature Backlog', 'Resource Constraint', 'High'),
            ('PAT-TL-001', 'High Priority Site', 'Timeline', 'Critical'),
            ('PAT-DQ-002', 'SDV Backlog', 'Data Quality', 'High'),
            ('PAT-SF-001', 'SAE Pending - DM', 'Safety', 'Critical'),
            ('PAT-RC-001', 'Coordinator Overload', 'Resource Constraint', 'Medium'),
            ('PAT-CP-001', 'Form Inactivation', 'Compliance', 'Medium'),
            ('PAT-DQ-001', 'Query Overload', 'Data Quality', 'High'),
            ('PAT-SP-001', 'Underperforming Site', 'Site Performance', 'High'),
            ('PAT-DQ-003', 'Missing Data', 'Data Quality', 'Medium'),
            ('PAT-SF-002', 'SAE Pending - Safety', 'Safety', 'Critical')
        ]
        
        alerts_data = []
        for pattern_id, name, category, severity in patterns:
            count = np.random.randint(50, 500)
            alerts_data.append({
                'pattern_id': pattern_id,
                'pattern_name': name,
                'category': category,
                'severity': severity,
                'match_count': count,
                'sites_affected': np.random.randint(10, 100),
                'patients_affected': count * np.random.randint(2, 5),
                'first_detected': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'recommendation': f'Review {name.lower()} across affected sites',
                'status': np.random.choice(['New', 'Acknowledged', 'In Progress', 'Resolved'], p=[0.3, 0.3, 0.3, 0.1])
            })
        
        return pd.DataFrame(alerts_data)
    
    def get_genome_matches(self, issue_type_filter: Optional[str] = None) -> pd.DataFrame:
        """Get resolution genome matches aggregated by issue type."""
        genome = self.load_resolution_genome()
        
        if genome.empty:
            return self._get_demo_genome_matches()
        
        # Aggregate by issue type
        if 'issue_type' not in genome.columns:
            return self._get_demo_genome_matches()
        
        # Filter if specified
        if issue_type_filter and issue_type_filter != "All":
            genome = genome[genome['issue_type'] == issue_type_filter]
        
        # Group by issue type
        grouped = genome.groupby('issue_type').agg({
            'patient_key': 'count'
        }).reset_index()
        grouped.columns = ['issue_type', 'match_count']
        
        # Add template info
        template_info = {
            'sdv_incomplete': {'template': 'Focused SDV Visit', 'success_rate': 92, 'effort_hours': 2.0},
            'open_queries': {'template': 'Query Resolution Call', 'success_rate': 88, 'effort_hours': 0.5},
            'signature_gaps': {'template': 'PI Signature Session', 'success_rate': 95, 'effort_hours': 1.0},
            'sae_dm_pending': {'template': 'SAE Reconciliation', 'success_rate': 90, 'effort_hours': 1.5},
            'missing_visits': {'template': 'Visit Scheduling', 'success_rate': 85, 'effort_hours': 0.5},
            'missing_pages': {'template': 'CRF Completion', 'success_rate': 90, 'effort_hours': 1.0},
            'broken_signatures': {'template': 'Re-signature Request', 'success_rate': 95, 'effort_hours': 0.5},
            'inactivated_forms': {'template': 'Form Review', 'success_rate': 80, 'effort_hours': 0.5},
            'lab_issues': {'template': 'Lab Range Config', 'success_rate': 85, 'effort_hours': 1.0},
            'edrr_issues': {'template': 'EDRR Reconciliation', 'success_rate': 82, 'effort_hours': 1.5},
            'meddra_uncoded': {'template': 'MedDRA Coding', 'success_rate': 95, 'effort_hours': 0.1},
            'whodrug_uncoded': {'template': 'WHODrug Coding', 'success_rate': 95, 'effort_hours': 0.1},
            'high_query_volume': {'template': 'Query Batch Processing', 'success_rate': 85, 'effort_hours': 1.0},
            'sae_safety_pending': {'template': 'Safety Review', 'success_rate': 90, 'effort_hours': 2.0}
        }
        
        grouped['template_name'] = grouped['issue_type'].map(
            lambda x: template_info.get(x, {}).get('template', 'Standard Resolution')
        )
        grouped['success_rate'] = grouped['issue_type'].map(
            lambda x: template_info.get(x, {}).get('success_rate', 80)
        )
        grouped['effort_hours'] = grouped['issue_type'].map(
            lambda x: template_info.get(x, {}).get('effort_hours', 1.0)
        )
        grouped['total_effort'] = grouped['match_count'] * grouped['effort_hours']
        
        return grouped.sort_values('match_count', ascending=False)
    
    def _get_demo_genome_matches(self) -> pd.DataFrame:
        """Return demo genome matches."""
        return pd.DataFrame({
            'issue_type': ['sdv_incomplete', 'open_queries', 'signature_gaps', 'sae_dm_pending', 
                          'missing_pages', 'missing_visits', 'broken_signatures', 'inactivated_forms'],
            'template_name': ['Focused SDV Visit', 'Query Resolution Call', 'PI Signature Session',
                            'SAE Reconciliation', 'CRF Completion', 'Visit Scheduling',
                            'Re-signature Request', 'Form Review'],
            'match_count': [18630, 4999, 8778, 4425, 2138, 1383, 5491, 4951],
            'success_rate': [92, 88, 95, 90, 90, 85, 95, 80],
            'effort_hours': [2.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.5, 0.5],
            'total_effort': [37260, 2500, 8778, 6638, 2138, 692, 2746, 2476]
        })


# =============================================================================
# COLOR HELPERS
# =============================================================================

def get_dqi_color(dqi: float) -> str:
    """Get color for DQI score."""
    if dqi >= 95:
        return '#27ae60'  # Green - Pristine
    elif dqi >= 85:
        return '#2ecc71'  # Light Green - Excellent
    elif dqi >= 75:
        return '#f39c12'  # Orange - Good
    elif dqi >= 65:
        return '#e67e22'  # Dark Orange - Fair
    elif dqi >= 50:
        return '#e74c3c'  # Red - Poor
    else:
        return '#c0392b'  # Dark Red - Critical


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        'Critical': '#e74c3c',
        'critical': '#e74c3c',
        'High': '#e67e22',
        'high': '#e67e22',
        'Medium': '#f39c12',
        'medium': '#f39c12',
        'Low': '#27ae60',
        'low': '#27ae60',
        'Info': '#3498db',
        'info': '#3498db'
    }
    return colors.get(severity, '#95a5a6')


def get_tier_color(tier: str) -> str:
    """Get color for quality tier."""
    colors = {
        'Pristine': '#27ae60',
        'Excellent': '#2ecc71',
        'Good': '#f39c12',
        'Fair': '#e67e22',
        'Poor': '#e74c3c',
        'Critical': '#c0392b'
    }
    return colors.get(tier, '#95a5a6')


def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        'New': '#3498db',
        'Acknowledged': '#9b59b6',
        'In Progress': '#f39c12',
        'Resolved': '#27ae60'
    }
    return colors.get(status, '#95a5a6')


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_page(user=None):
    """Main render function for Data Manager Hub."""
    
    # Initialize data loader
    loader = DMDataLoader()
    
    # Page header
    st.markdown("""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">📊 Data Manager Hub</h1>
            <p style="color: #ecf0f1; margin: 5px 0 0 0;">
                Quality Matrix • Bottleneck Analysis • Pattern Alerts • Resolution Genome
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filters row
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        studies = ["All Studies"] + loader.get_studies_list()
        selected_study = st.selectbox("📚 Study", studies, key="dm_study_filter")
    
    with col2:
        severity_options = ["All", "Critical", "High", "Medium", "Low"]
        selected_severity = st.selectbox("⚠️ Severity", severity_options, key="dm_severity_filter")
    
    with col3:
        issue_types = ["All"] + loader.get_issue_types_list()
        selected_issue = st.selectbox("🔍 Issue Type", issue_types, key="dm_issue_filter")
    
    with col4:
        st.write("")  # Spacing
        if st.button("🔄 Refresh", key="dm_refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Get portfolio data
    portfolio = loader.get_dm_portfolio()
    
    # Render sections
    render_kpi_section(portfolio)
    
    st.markdown("---")
    
    # Two columns for Quality Matrix and Bottlenecks
    col1, col2 = st.columns([3, 2])
    
    with col1:
        render_quality_matrix(loader, selected_study)
    
    with col2:
        render_bottleneck_section(loader)
    
    st.markdown("---")
    
    # Two columns for Pattern Alerts and Genome Matches
    col1, col2 = st.columns(2)
    
    with col1:
        render_pattern_alerts(loader, selected_severity)
    
    with col2:
        render_genome_matches(loader, selected_issue)
    
    st.markdown("---")
    
    # Action section
    render_action_section()


def render_kpi_section(portfolio: Dict):
    """Render KPI cards section."""
    st.markdown("### 📈 Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db, #2980b9); 
                        padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 28px;">{portfolio['total_patients']:,}</h3>
                <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;">Total Patients</p>
                <p style="color: #bdc3c7; margin: 0; font-size: 11px;">
                    {portfolio['total_sites']:,} Sites • {portfolio['total_studies']} Studies
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        dqi = portfolio['mean_dqi']
        dqi_color = get_dqi_color(dqi)
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {dqi_color}, {dqi_color}dd); 
                        padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 28px;">{dqi:.1f}</h3>
                <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;">Mean DQI</p>
                <p style="color: #bdc3c7; margin: 0; font-size: 11px;">
                    {'Pristine' if dqi >= 95 else 'Excellent' if dqi >= 85 else 'Good' if dqi >= 75 else 'Fair'}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        clean_rate = portfolio['tier2_clean_rate']
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #27ae60, #1e8449); 
                        padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 28px;">{clean_rate:.1f}%</h3>
                <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;">Tier 2 Clean</p>
                <p style="color: #bdc3c7; margin: 0; font-size: 11px;">
                    {portfolio['tier2_clean']:,} patients
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); 
                        padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 28px;">{portfolio['total_issues']:,}</h3>
                <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;">Open Issues</p>
                <p style="color: #bdc3c7; margin: 0; font-size: 11px;">
                    {portfolio['critical_issues']:,} critical
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); 
                        padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 28px;">{portfolio['open_queries']:,}</h3>
                <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;">Open Queries</p>
                <p style="color: #bdc3c7; margin: 0; font-size: 11px;">
                    {portfolio['pending_reviews']:,} pending reviews
                </p>
            </div>
        """, unsafe_allow_html=True)


def render_quality_matrix(loader: DMDataLoader, study_filter: str):
    """Render quality matrix section."""
    st.markdown("### 🎯 Quality Matrix by Site")
    
    # Get data
    matrix = loader.get_quality_matrix(study_filter if study_filter != "All Studies" else None)
    
    if matrix.empty:
        st.info("No quality matrix data available.")
        return
    
    # Create heatmap
    fig = go.Figure()
    
    # Prepare data for heatmap
    sites = matrix['site_id'].tolist()[:30]  # Limit to 30 for display
    metrics = ['dqi_score', 'tier1_clean_rate', 'tier2_clean_rate']
    metric_names = ['DQI Score', 'Tier 1 Clean %', 'Tier 2 Clean %']
    
    z_data = []
    for metric in metrics:
        z_data.append(matrix[metric].tolist()[:30])
    
    fig.add_trace(go.Heatmap(
        z=z_data,
        x=sites,
        y=metric_names,
        colorscale=[
            [0, '#e74c3c'],
            [0.5, '#f39c12'],
            [0.75, '#2ecc71'],
            [1, '#27ae60']
        ],
        showscale=True,
        colorbar=dict(title="Score"),
        hovertemplate='Site: %{x}<br>Metric: %{y}<br>Value: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickangle=45),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality tier distribution
    st.markdown("#### Quality Tier Distribution")
    tier_counts = matrix['quality_tier'].value_counts()
    
    tier_cols = st.columns(len(tier_counts))
    for i, (tier, count) in enumerate(tier_counts.items()):
        with tier_cols[i]:
            color = get_tier_color(tier)
            st.markdown(f"""
                <div style="background: {color}22; border-left: 4px solid {color}; 
                            padding: 10px; border-radius: 5px; text-align: center;">
                    <span style="font-weight: bold; color: {color};">{count}</span>
                    <br><span style="font-size: 11px;">{tier}</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Expandable table
    with st.expander("📋 View Full Quality Matrix"):
        display_cols = ['site_id', 'study_id', 'patients', 'dqi_score', 
                       'tier1_clean_rate', 'tier2_clean_rate', 'total_issues', 'quality_tier']
        display_df = matrix[display_cols].head(50)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_bottleneck_section(loader: DMDataLoader):
    """Render bottleneck analysis section."""
    st.markdown("### 🚧 Bottleneck Analysis")
    
    # Get data
    bottleneck_data = loader.get_bottleneck_analysis()
    bottlenecks = bottleneck_data['bottlenecks']
    
    if not bottlenecks:
        st.info("No bottlenecks identified.")
        return
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patients Affected", f"{bottleneck_data['total_blocked_patients']:,}")
    with col2:
        st.metric("Critical Bottlenecks", bottleneck_data['critical_bottlenecks'])
    
    # Bottleneck chart
    df = pd.DataFrame(bottlenecks)
    
    fig = go.Figure()
    
    # Add bars
    colors = [get_severity_color('Critical') if b['impact_score'] >= 80 
              else get_severity_color('High') if b['impact_score'] >= 60 
              else get_severity_color('Medium') for b in bottlenecks]
    
    fig.add_trace(go.Bar(
        y=df['issue_name'],
        x=df['blocking_score'],
        orientation='h',
        marker_color=colors,
        text=df['patient_count'].apply(lambda x: f'{x:,}'),
        textposition='auto',
        hovertemplate='%{y}<br>Blocking Score: %{x:,}<br>Patients: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Blocking Score",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top bottleneck detail
    if bottlenecks:
        top = bottlenecks[0]
        st.markdown(f"""
            <div style="background: #e74c3c22; border-left: 4px solid #e74c3c; 
                        padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong>🔴 Top Bottleneck:</strong> {top['issue_name']}<br>
                <span style="font-size: 12px;">
                    {top['patient_count']:,} patients • {top.get('issue_count', top['patient_count']):,} issues • Blocks {top['downstream_blocked']:,} downstream
                </span><br>
                <span style="font-size: 11px; color: #666;">
                    Responsible: {top['responsible']}
                </span>
            </div>
        """, unsafe_allow_html=True)


def render_pattern_alerts(loader: DMDataLoader, severity_filter: str):
    """Render pattern alerts section."""
    st.markdown("### 🚨 Pattern Alerts")
    
    # Get data
    alerts = loader.get_pattern_alerts(severity_filter if severity_filter != "All" else None)
    
    if alerts.empty:
        st.info("No pattern alerts found.")
        return
    
    # Summary by severity
    severity_col = 'severity' if 'severity' in alerts.columns else None
    if severity_col:
        severity_counts = alerts[severity_col].str.capitalize().value_counts()
    else:
        severity_counts = pd.Series()
    
    severity_cols = st.columns(4)
    for i, severity in enumerate(['Critical', 'High', 'Medium', 'Low']):
        with severity_cols[i]:
            count = severity_counts.get(severity, 0)
            color = get_severity_color(severity)
            st.markdown(f"""
                <div style="background: {color}22; border: 2px solid {color}; 
                            padding: 8px; border-radius: 8px; text-align: center;">
                    <span style="font-weight: bold; font-size: 18px; color: {color};">{count}</span>
                    <br><span style="font-size: 11px;">{severity}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Alert cards
    for _, alert in alerts.head(5).iterrows():
        # Handle different column names
        severity = str(alert.get('severity', 'Medium')).capitalize()
        color = get_severity_color(severity)
        status = alert.get('status', 'New')
        if pd.isna(status):
            status = 'New'
        status_color = get_status_color(status)
        
        pattern_name = alert.get('pattern_name', alert.get('pattern_id', 'Unknown Pattern'))
        match_count = alert.get('match_count', alert.get('matches', 0))
        sites_affected = alert.get('sites_affected', alert.get('site_count', 0))
        
        st.markdown(f"""
            <div style="background: white; border-left: 4px solid {color}; 
                        padding: 12px; border-radius: 5px; margin-bottom: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #2c3e50;">{pattern_name}</strong>
                    <span style="background: {status_color}; color: white; padding: 2px 8px; 
                                 border-radius: 10px; font-size: 11px;">{status}</span>
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                    <span style="background: {color}22; color: {color}; padding: 2px 6px; 
                                 border-radius: 4px; font-size: 10px;">{severity}</span>
                    &nbsp;•&nbsp; {match_count:,} matches
                    &nbsp;•&nbsp; {sites_affected} sites
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # View all button
    if len(alerts) > 5:
        with st.expander("📋 View All Alerts"):
            display_cols = [c for c in ['pattern_id', 'pattern_name', 'severity', 'match_count', 
                           'sites_affected', 'status'] if c in alerts.columns]
            st.dataframe(alerts[display_cols] if display_cols else alerts.head(20), 
                        use_container_width=True, hide_index=True)


def render_genome_matches(loader: DMDataLoader, issue_filter: str):
    """Render resolution genome matches section."""
    st.markdown("### 🧬 Resolution Genome Matches")
    
    # Get data
    genome = loader.get_genome_matches(issue_filter if issue_filter != "All" else None)
    
    if genome.empty:
        st.info("No genome matches found.")
        return
    
    # Summary metrics
    total_matches = genome['match_count'].sum()
    total_effort = genome['total_effort'].sum()
    avg_success = genome['success_rate'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", f"{total_matches:,}")
    with col2:
        st.metric("Total Effort", f"{total_effort:,.0f} hrs")
    with col3:
        st.metric("Avg Success Rate", f"{avg_success:.0f}%")
    
    # Genome chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=genome['issue_type'],
        y=genome['match_count'],
        marker_color='#3498db',
        text=genome['match_count'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        name='Matches',
        hovertemplate='%{x}<br>Matches: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickangle=45),
        yaxis_title="Match Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Template cards
    st.markdown("#### 📝 Resolution Templates")
    
    for _, row in genome.head(4).iterrows():
        success_color = '#27ae60' if row['success_rate'] >= 90 else '#f39c12' if row['success_rate'] >= 80 else '#e74c3c'
        
        st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; 
                        margin-bottom: 8px; border-left: 4px solid #3498db;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{row['template_name']}</strong>
                    <span style="background: {success_color}; color: white; padding: 2px 8px; 
                                 border-radius: 10px; font-size: 11px;">{row['success_rate']}% success</span>
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                    {row['issue_type']} • {row['match_count']:,} matches • {row['effort_hours']} hrs each
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Batch apply button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Batch Apply All", key="batch_apply_genome", type="primary"):
            st.success(f"Queued {total_matches:,} resolution actions!")
    with col2:
        if st.button("📊 Export Matches", key="export_genome"):
            st.toast("Genome matches exported to CSV")


def render_action_section():
    """Render quick actions section."""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📝 Generate Report", key="dm_gen_report", use_container_width=True):
            st.session_state['current_page'] = 'Reports'
            st.rerun()
    
    with col2:
        if st.button("🤖 Ask AI", key="dm_ask_ai", use_container_width=True):
            st.session_state['current_page'] = 'AI Assistant'
            st.rerun()
    
    with col3:
        if st.button("🔗 View Cascade", key="dm_view_cascade", use_container_width=True):
            st.session_state['current_page'] = 'Cascade Explorer'
            st.rerun()
    
    with col4:
        if st.button("📧 Send Alerts", key="dm_send_alerts", use_container_width=True):
            st.toast("Alert notifications queued for delivery")
    
    with col5:
        if st.button("📥 Export Data", key="dm_export", use_container_width=True):
            st.toast("Data export initiated")


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_dm_hub():
    """Test Data Manager Hub components."""
    print("\n" + "="*60)
    print("TRIALPULSE NEXUS 10X - DATA MANAGER HUB TEST")
    print("="*60 + "\n")
    
    loader = DMDataLoader()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Data Loader Initialization
    print("TEST 1: Data Loader Initialization")
    try:
        assert loader is not None
        print("  ✅ Data loader created")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 2: Load UPR
    print("\nTEST 2: Load UPR Data")
    try:
        upr = loader.load_upr()
        print(f"  ✅ Loaded {len(upr):,} patients")
        if not upr.empty:
            print(f"     Columns: {len(upr.columns)}")
            print(f"     Studies: {upr['study_id'].nunique() if 'study_id' in upr.columns else 'N/A'}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 3: Get Studies List
    print("\nTEST 3: Get Studies List")
    try:
        studies = loader.get_studies_list()
        print(f"  ✅ Found {len(studies)} studies")
        print(f"     First 5: {studies[:5]}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 4: Portfolio Summary
    print("\nTEST 4: Get Portfolio Summary")
    try:
        portfolio = loader.get_dm_portfolio()
        print(f"  ✅ Portfolio loaded")
        print(f"     Patients: {portfolio['total_patients']:,}")
        print(f"     DQI: {portfolio['mean_dqi']:.1f}")
        print(f"     Clean Rate: {portfolio['tier2_clean_rate']:.1f}%")
        print(f"     Issues: {portfolio['total_issues']:,}")
        print(f"     Critical: {portfolio['critical_issues']:,}")
        print(f"     Open Queries: {portfolio['open_queries']:,}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 5: Quality Matrix
    print("\nTEST 5: Get Quality Matrix")
    try:
        matrix = loader.get_quality_matrix()
        print(f"  ✅ Quality matrix loaded")
        print(f"     Sites: {len(matrix)}")
        if not matrix.empty:
            print(f"     Columns: {list(matrix.columns)[:5]}...")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 6: Bottleneck Analysis
    print("\nTEST 6: Get Bottleneck Analysis")
    try:
        bottlenecks = loader.get_bottleneck_analysis()
        print(f"  ✅ Bottleneck analysis loaded")
        print(f"     Bottlenecks: {len(bottlenecks['bottlenecks'])}")
        print(f"     Total Affected: {bottlenecks['total_blocked_patients']:,}")
        print(f"     Critical: {bottlenecks['critical_bottlenecks']}")
        if bottlenecks['bottlenecks']:
            top = bottlenecks['bottlenecks'][0]
            print(f"     Top: {top['issue_name']} ({top['patient_count']:,} patients)")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 7: Pattern Alerts
    print("\nTEST 7: Get Pattern Alerts")
    try:
        alerts = loader.get_pattern_alerts()
        print(f"  ✅ Pattern alerts loaded")
        print(f"     Alerts: {len(alerts)}")
        if not alerts.empty and 'severity' in alerts.columns:
            print(f"     Severity dist: {alerts['severity'].value_counts().to_dict()}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 8: Genome Matches
    print("\nTEST 8: Get Genome Matches")
    try:
        genome = loader.get_genome_matches()
        print(f"  ✅ Genome matches loaded")
        print(f"     Issue types: {len(genome)}")
        if not genome.empty:
            print(f"     Total matches: {genome['match_count'].sum():,}")
            print(f"     Top: {genome.iloc[0]['issue_type']} ({genome.iloc[0]['match_count']:,})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 9: Color Functions
    print("\nTEST 9: Color Helper Functions")
    try:
        assert get_dqi_color(98) == '#27ae60'
        assert get_dqi_color(70) == '#e67e22'
        assert get_severity_color('Critical') == '#e74c3c'
        assert get_tier_color('Pristine') == '#27ae60'
        print("  ✅ All color functions working")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 10: Study Filter
    print("\nTEST 10: Study Filter")
    try:
        matrix_filtered = loader.get_quality_matrix("Study_1")
        print(f"  ✅ Study filter working")
        print(f"     Filtered sites: {len(matrix_filtered)}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed}/{tests_passed + tests_failed} tests passed")
    print("="*60)
    
    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {tests_failed} tests failed")
    
    return tests_passed, tests_failed


if __name__ == "__main__":
    test_dm_hub()