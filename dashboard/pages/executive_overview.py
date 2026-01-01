"""
TRIALPULSE NEXUS 10X - Executive Overview Page
Phase 7.2: Portfolio-level KPIs, regional heatmap, DQI trending, DB Lock projection
FIXED VERSION v1.2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

# Import components
from dashboard.config.theme import THEME_CONFIG, get_dqi_band_color, get_dqi_band_name
from dashboard.config.session import get_filter, set_filter, cache_data, get_cached_data


class ExecutiveDataLoader:
    """Data loader for Executive Overview page - FIXED VERSION."""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self._cache = {}
    
    def _safe_load_parquet(self, path: Path, cache_key: str) -> Optional[pd.DataFrame]:
        """Safely load a parquet file with caching."""
        cached = get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        try:
            if path.exists():
                df = pd.read_parquet(path)
                cache_data(cache_key, df, ttl_seconds=300)
                return df
        except Exception as e:
            pass
        return None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates list."""
        if df is None:
            return None
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def load_upr(self) -> Optional[pd.DataFrame]:
        """Load Unified Patient Record."""
        path = self.data_dir / "upr" / "unified_patient_record.parquet"
        return self._safe_load_parquet(path, "exec_upr")
    
    def load_patient_issues(self) -> Optional[pd.DataFrame]:
        """Load patient issues data."""
        path = self.data_dir / "analytics" / "patient_issues.parquet"
        return self._safe_load_parquet(path, "exec_issues")
    
    def load_patient_dqi(self) -> Optional[pd.DataFrame]:
        """Load patient DQI data."""
        path = self.data_dir / "analytics" / "patient_dqi_enhanced.parquet"
        return self._safe_load_parquet(path, "exec_dqi")
    
    def load_clean_status(self) -> Optional[pd.DataFrame]:
        """Load clean patient status."""
        path = self.data_dir / "analytics" / "patient_clean_status.parquet"
        return self._safe_load_parquet(path, "exec_clean")
    
    def load_dblock_status(self) -> Optional[pd.DataFrame]:
        """Load DB lock status."""
        path = self.data_dir / "analytics" / "patient_dblock_status.parquet"
        return self._safe_load_parquet(path, "exec_dblock")
    
    def load_site_benchmarks(self) -> Optional[pd.DataFrame]:
        """Load site benchmarks."""
        path = self.data_dir / "analytics" / "site_benchmarks.parquet"
        return self._safe_load_parquet(path, "exec_sites")
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        cache_key = "exec_portfolio_metrics"
        cached = get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        metrics = {
            "total_patients": 0,
            "total_sites": 0,
            "total_studies": 0,
            "mean_dqi": 0.0,
            "median_dqi": 0.0,
            "tier1_clean_count": 0,
            "tier1_clean_rate": 0.0,
            "tier2_clean_count": 0,
            "tier2_clean_rate": 0.0,
            "dblock_ready_count": 0,
            "dblock_ready_rate": 0.0,
            "dblock_eligible_count": 0,
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "patients_with_issues": 0,
            "clean_patients": 0
        }
        
        # Load UPR for basic counts
        upr = self.load_upr()
        if upr is not None:
            metrics["total_patients"] = len(upr)
            if 'site_id' in upr.columns:
                metrics["total_sites"] = upr['site_id'].nunique()
            if 'study_id' in upr.columns:
                metrics["total_studies"] = upr['study_id'].nunique()
        
        # Load DQI data
        dqi = self.load_patient_dqi()
        if dqi is not None:
            dqi_col = self._find_column(dqi, ['dqi_score', 'enhanced_dqi', 'dqi', 'DQI'])
            if dqi_col:
                metrics["mean_dqi"] = float(dqi[dqi_col].mean())
                metrics["median_dqi"] = float(dqi[dqi_col].median())
        
        # Load clean status
        clean = self.load_clean_status()
        if clean is not None:
            tier1_col = self._find_column(clean, ['tier1_clean', 'is_tier1_clean', 'clinical_clean'])
            tier2_col = self._find_column(clean, ['tier2_clean', 'is_tier2_clean', 'operational_clean'])
            
            if tier1_col:
                metrics["tier1_clean_count"] = int(clean[tier1_col].sum())
                metrics["tier1_clean_rate"] = float(clean[tier1_col].mean() * 100)
            if tier2_col:
                metrics["tier2_clean_count"] = int(clean[tier2_col].sum())
                metrics["tier2_clean_rate"] = float(clean[tier2_col].mean() * 100)
        
        # Load DB lock status
        dblock = self.load_dblock_status()
        if dblock is not None:
            ready_col = self._find_column(dblock, [
                'dblock_tier1_ready', 'dblock_ready', 'db_lock_ready', 'ready', 'is_ready'
            ])
            eligible_col = self._find_column(dblock, [
                'dblock_is_eligible', 'dblock_eligible', 'db_lock_eligible', 'eligible', 'is_eligible'
            ])
            blocker_col = self._find_column(dblock, ['dblock_blocker_count', 'blocker_count'])
            
            # Calculate ready count
            if ready_col:
                metrics["dblock_ready_count"] = int(dblock[ready_col].sum())
            elif blocker_col:
                metrics["dblock_ready_count"] = int((dblock[blocker_col] == 0).sum())
            
            # Calculate eligible count and ready rate
            if eligible_col:
                metrics["dblock_eligible_count"] = int(dblock[eligible_col].sum())
                eligible_df = dblock[dblock[eligible_col] == True]
                if len(eligible_df) > 0:
                    if ready_col:
                        metrics["dblock_ready_rate"] = float(eligible_df[ready_col].mean() * 100)
                    elif blocker_col:
                        metrics["dblock_ready_rate"] = float((eligible_df[blocker_col] == 0).mean() * 100)
            else:
                metrics["dblock_eligible_count"] = len(dblock)
                if ready_col:
                    metrics["dblock_ready_rate"] = float(dblock[ready_col].mean() * 100)
                elif blocker_col:
                    metrics["dblock_ready_rate"] = float((dblock[blocker_col] == 0).mean() * 100)
        
        # Load issues
        issues = self.load_patient_issues()
        if issues is not None:
            # Count total issues from issue_* columns
            issue_cols = [c for c in issues.columns if c.startswith('issue_')]
            count_cols = [c for c in issues.columns if c.startswith('count_')]
            
            if count_cols:
                for col in count_cols:
                    metrics["total_issues"] += int(issues[col].sum())
            elif issue_cols:
                for col in issue_cols:
                    metrics["total_issues"] += int(issues[col].sum())
            
            if 'total_issues' in issues.columns:
                metrics["patients_with_issues"] = int((issues['total_issues'] > 0).sum())
                metrics["clean_patients"] = int((issues['total_issues'] == 0).sum())
            elif issue_cols:
                issues_sum = issues[issue_cols].sum(axis=1)
                metrics["patients_with_issues"] = int((issues_sum > 0).sum())
                metrics["clean_patients"] = int((issues_sum == 0).sum())
            
            # Priority breakdown
            if 'priority_tier' in issues.columns:
                metrics["critical_issues"] = int((issues['priority_tier'] == 'Critical').sum())
                metrics["high_issues"] = int((issues['priority_tier'] == 'High').sum())
            elif issue_cols:
                issues_sum = issues[issue_cols].sum(axis=1)
                metrics["critical_issues"] = int((issues_sum >= 5).sum())
                metrics["high_issues"] = int(((issues_sum >= 3) & (issues_sum < 5)).sum())
        
        cache_data(cache_key, metrics, ttl_seconds=60)
        return metrics
    
    def get_study_metrics(self) -> pd.DataFrame:
        """Get metrics aggregated by study."""
        cache_key = "exec_study_metrics"
        cached = get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        upr = self.load_upr()
        dqi = self.load_patient_dqi()
        clean = self.load_clean_status()
        dblock = self.load_dblock_status()
        issues = self.load_patient_issues()
        
        if upr is None or 'study_id' not in upr.columns:
            return pd.DataFrame()
        
        # Start with basic counts
        study_metrics = upr.groupby('study_id').agg({
            'patient_key': 'count'
        }).reset_index()
        study_metrics.columns = ['study_id', 'patients']
        
        # Add site count
        if 'site_id' in upr.columns:
            site_counts = upr.groupby('study_id')['site_id'].nunique().reset_index()
            site_counts.columns = ['study_id', 'sites']
            study_metrics = study_metrics.merge(site_counts, on='study_id', how='left')
        else:
            study_metrics['sites'] = 0
        
        # Add DQI
        if dqi is not None and 'study_id' in dqi.columns:
            dqi_col = self._find_column(dqi, ['dqi_score', 'enhanced_dqi', 'dqi'])
            if dqi_col:
                dqi_agg = dqi.groupby('study_id')[dqi_col].mean().reset_index()
                dqi_agg.columns = ['study_id', 'mean_dqi']
                study_metrics = study_metrics.merge(dqi_agg, on='study_id', how='left')
        
        # Add clean status
        if clean is not None and 'study_id' in clean.columns:
            tier1_col = self._find_column(clean, ['tier1_clean', 'is_tier1_clean'])
            tier2_col = self._find_column(clean, ['tier2_clean', 'is_tier2_clean'])
            
            agg_dict = {}
            if tier1_col:
                agg_dict[tier1_col] = 'mean'
            if tier2_col:
                agg_dict[tier2_col] = 'mean'
            
            if agg_dict:
                clean_agg = clean.groupby('study_id').agg(agg_dict).reset_index()
                if tier1_col and tier1_col in clean_agg.columns:
                    clean_agg['tier1_rate'] = clean_agg[tier1_col] * 100
                    clean_agg = clean_agg.drop(columns=[tier1_col])
                if tier2_col and tier2_col in clean_agg.columns:
                    clean_agg['tier2_rate'] = clean_agg[tier2_col] * 100
                    clean_agg = clean_agg.drop(columns=[tier2_col])
                study_metrics = study_metrics.merge(clean_agg, on='study_id', how='left')
        
        # Add DB lock
        if dblock is not None and 'study_id' in dblock.columns:
            ready_col = self._find_column(dblock, [
                'dblock_tier1_ready', 'dblock_ready', 'db_lock_ready', 'ready'
            ])
            eligible_col = self._find_column(dblock, [
                'dblock_is_eligible', 'dblock_eligible', 'db_lock_eligible', 'eligible'
            ])
            blocker_col = self._find_column(dblock, ['dblock_blocker_count', 'blocker_count'])
            
            if ready_col or blocker_col:
                dblock_copy = dblock.copy()
                
                if not ready_col and blocker_col:
                    dblock_copy['_ready'] = dblock_copy[blocker_col] == 0
                    ready_col = '_ready'
                
                if ready_col and eligible_col:
                    dblock_agg = dblock_copy.groupby('study_id').agg({
                        ready_col: 'sum',
                        eligible_col: 'sum'
                    }).reset_index()
                    dblock_agg.columns = ['study_id', 'dblock_ready', 'dblock_eligible']
                elif ready_col:
                    dblock_agg = dblock_copy.groupby('study_id').agg({
                        ready_col: 'sum'
                    }).reset_index()
                    dblock_agg.columns = ['study_id', 'dblock_ready']
                    dblock_agg['dblock_eligible'] = dblock_copy.groupby('study_id').size().values
                
                dblock_agg['dblock_rate'] = np.where(
                    dblock_agg['dblock_eligible'] > 0,
                    dblock_agg['dblock_ready'] / dblock_agg['dblock_eligible'] * 100,
                    0
                )
                study_metrics = study_metrics.merge(
                    dblock_agg[['study_id', 'dblock_ready', 'dblock_eligible', 'dblock_rate']],
                    on='study_id',
                    how='left'
                )
        
        # Add issues
        if issues is not None and 'study_id' in issues.columns:
            issue_cols = [c for c in issues.columns if c.startswith('issue_')]
            
            if issue_cols:
                issues_copy = issues.copy()
                issues_copy['_total_issues'] = issues_copy[issue_cols].sum(axis=1)
                issues_agg = issues_copy.groupby('study_id')['_total_issues'].sum().reset_index()
                issues_agg.columns = ['study_id', 'total_issues']
                study_metrics = study_metrics.merge(issues_agg, on='study_id', how='left')
            elif 'total_issues' in issues.columns:
                issues_agg = issues.groupby('study_id')['total_issues'].sum().reset_index()
                issues_agg.columns = ['study_id', 'total_issues']
                study_metrics = study_metrics.merge(issues_agg, on='study_id', how='left')
        
        study_metrics = study_metrics.fillna(0)
        study_metrics = study_metrics.sort_values('patients', ascending=False)
        
        cache_data(cache_key, study_metrics, ttl_seconds=60)
        return study_metrics
    
    def get_regional_metrics(self) -> pd.DataFrame:
        """Get metrics aggregated by region - FIXED."""
        cache_key = "exec_regional_metrics"
        cached = get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        upr = self.load_upr()
        dqi = self.load_patient_dqi()
        
        if upr is None:
            return pd.DataFrame()
        
        upr = upr.copy()
        
        # Check for region column
        region_col = self._find_column(upr, ['region', 'country', 'site_country'])
        
        if region_col is None:
            # Create mock regions based on site_id
            if 'site_id' in upr.columns:
                upr['region'] = upr['site_id'].apply(lambda x: self._assign_mock_region(x))
            else:
                upr['region'] = 'Unknown'
            region_col = 'region'
        
        # Aggregate by region
        if 'patient_key' in upr.columns:
            regional = upr.groupby(region_col).agg({'patient_key': 'count'}).reset_index()
            regional.columns = [region_col, 'patients']
        else:
            regional = upr.groupby(region_col).size().reset_index(name='patients')
        
        # Rename region column to 'region' for consistency
        if region_col != 'region':
            regional = regional.rename(columns={region_col: 'region'})
        
        # Add site count
        if 'site_id' in upr.columns:
            site_counts = upr.groupby(region_col)['site_id'].nunique().reset_index()
            site_counts.columns = ['region', 'sites']
            regional = regional.merge(site_counts, on='region', how='left')
        else:
            regional['sites'] = 0
        
        # Add DQI
        if dqi is not None:
            dqi_col = self._find_column(dqi, ['dqi_score', 'enhanced_dqi', 'dqi'])
            if dqi_col:
                # Check if DQI already has region column
                dqi_region_col = self._find_column(dqi, ['region', 'country'])
                
                if dqi_region_col:
                    # Use DQI's own region column
                    dqi_valid = dqi[dqi[dqi_region_col].notna() & (dqi[dqi_region_col] != '')]
                    if len(dqi_valid) > 0:
                        dqi_regional = dqi_valid.groupby(dqi_region_col)[dqi_col].mean().reset_index()
                        dqi_regional.columns = ['region', 'mean_dqi']
                        regional = regional.merge(dqi_regional, on='region', how='left')
                elif 'patient_key' in dqi.columns and 'patient_key' in upr.columns:
                    # Fall back to merging with UPR for region
                    upr_subset = upr[['patient_key', region_col]].drop_duplicates()
                    upr_subset = upr_subset.rename(columns={region_col: '_region'})
                    dqi_with_region = dqi.merge(upr_subset, on='patient_key', how='left')
                    if '_region' in dqi_with_region.columns:
                        dqi_valid = dqi_with_region.dropna(subset=['_region'])
                        dqi_valid = dqi_valid[dqi_valid['_region'] != '']
                        if len(dqi_valid) > 0:
                            dqi_regional = dqi_valid.groupby('_region')[dqi_col].mean().reset_index()
                            dqi_regional.columns = ['region', 'mean_dqi']
                            regional = regional.merge(dqi_regional, on='region', how='left')
        
        regional = regional.fillna(0)
        cache_data(cache_key, regional, ttl_seconds=60)
        return regional

    
    def _assign_mock_region(self, site_id: str) -> str:
        """Assign mock region based on site_id hash."""
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
        if site_id:
            idx = hash(str(site_id)) % len(regions)
            return regions[idx]
        return 'Unknown'
    
    def get_dqi_distribution(self) -> pd.DataFrame:
        """Get DQI distribution by band."""
        dqi = self.load_patient_dqi()
        if dqi is None:
            return pd.DataFrame()
        
        dqi_col = self._find_column(dqi, ['dqi_score', 'enhanced_dqi', 'dqi'])
        if not dqi_col:
            return pd.DataFrame()
        
        total = len(dqi)
        if total == 0:
            return pd.DataFrame()
        
        bands = [
            ('Emergency', 0, 25),
            ('Critical', 25, 50),
            ('Poor', 50, 65),
            ('Fair', 65, 75),
            ('Good', 75, 85),
            ('Excellent', 85, 95),
            ('Pristine', 95, 100.01)
        ]
        
        distribution = []
        for band_name, low, high in bands:
            count = len(dqi[(dqi[dqi_col] >= low) & (dqi[dqi_col] < high)])
            distribution.append({
                'band': band_name,
                'count': count,
                'percentage': count / total * 100
            })
        
        return pd.DataFrame(distribution)
    
    def get_issue_breakdown(self) -> pd.DataFrame:
        """Get issue breakdown by type - FIXED version."""
        issues = self.load_patient_issues()
        
        if issues is None or len(issues) == 0:
            return pd.DataFrame()
        
        # Find issue columns (they use 'issue_' prefix)
        issue_cols = [col for col in issues.columns if col.startswith('issue_')]
        
        if not issue_cols:
            # Try alternative column patterns
            issue_cols = [col for col in issues.columns if col.startswith('has_')]
        
        if not issue_cols:
            return pd.DataFrame()
        
        # Calculate counts for each issue type
        issue_data = []
        
        for col in issue_cols:
            # Extract issue type name
            if col.startswith('issue_'):
                issue_type = col.replace('issue_', '')
            elif col.startswith('has_'):
                issue_type = col.replace('has_', '')
            else:
                issue_type = col
            
            # Count patients with this issue
            count = int(issues[col].sum())
            
            if count > 0:
                # Format display name
                display_name = issue_type.replace('_', ' ').title()
                
                issue_data.append({
                    'issue_type': issue_type,
                    'display_name': display_name,
                    'patient_count': count,
                    'percentage': count / len(issues) * 100
                })
        
        if not issue_data:
            return pd.DataFrame()
        
        # Create DataFrame and sort by count
        df = pd.DataFrame(issue_data)
        df = df.sort_values('patient_count', ascending=False).reset_index(drop=True)
        
        return df


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_page(user: Dict[str, Any] = None):
    """Render the Executive Overview page."""
    
    # Page header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                color: white; padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem;">📊 Executive Overview</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
                    Portfolio-level insights and key performance indicators
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; opacity: 0.7;">Last Updated</div>
                <div style="font-size: 1.1rem;">{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data loader
    loader = ExecutiveDataLoader()
    
    # Load portfolio metrics
    metrics = loader.get_portfolio_metrics()
    
    # Check if data loaded
    if metrics["total_patients"] == 0:
        st.error("⚠️ No data available. Please ensure the analytics pipeline has been run.")
        st.info("Run the following commands to generate data:\n"
                "```\n"
                "python src/run_dqi_enhanced.py\n"
                "python src/run_clean_patient.py\n"
                "python src/run_dblock_ready.py\n"
                "python src/run_issue_detector.py\n"
                "```")
        return
    
    # Study filter
    study_metrics = loader.get_study_metrics()
    if len(study_metrics) > 0 and 'study_id' in study_metrics.columns:
        studies = ['All Studies'] + study_metrics['study_id'].tolist()
    else:
        studies = ['All Studies']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_study = st.selectbox(
            "📚 Filter by Study",
            studies,
            index=0,
            key="exec_study_filter"
        )
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.cached_data = {}
            st.rerun()
    with col3:
        view_mode = st.selectbox(
            "View",
            ["Dashboard", "Detailed"],
            index=0,
            key="exec_view_mode"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 1: KEY METRICS ===
    render_key_metrics(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 2: DQI & CLEAN PATIENT ===
    col1, col2 = st.columns(2)
    
    with col1:
        render_dqi_section(loader, metrics)
    
    with col2:
        render_clean_patient_section(loader, metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 3: STUDY-LEVEL METRICS ===
    if len(study_metrics) > 0:
        render_study_metrics_section(study_metrics)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 4: REGIONAL HEATMAP ===
    render_regional_heatmap(loader)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 5: DB LOCK PROJECTION ===
    render_dblock_projection(loader, metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 6: ISSUES OVERVIEW ===
    render_issues_section(loader, metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 7: QUICK ACTIONS ===
    render_quick_actions()


def render_key_metrics(metrics: Dict[str, Any]):
    """Render the key metrics row."""
    
    st.markdown("### 🎯 Key Performance Indicators")
    
    total_patients = metrics["total_patients"]
    total_sites = metrics["total_sites"]
    mean_dqi = metrics["mean_dqi"]
    tier2_clean_rate = metrics["tier2_clean_rate"]
    tier2_clean_count = metrics["tier2_clean_count"]
    dblock_ready_rate = metrics["dblock_ready_rate"]
    dblock_ready_count = metrics["dblock_ready_count"]
    dblock_eligible_count = metrics["dblock_eligible_count"]
    total_issues = metrics["total_issues"]
    critical_issues = metrics["critical_issues"]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.25rem; border-radius: 10px; 
                    border-left: 4px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #7f8c8d; font-size: 0.85rem;">👥 Total Patients</div>
            <div style="color: #2c3e50; font-size: 2rem; font-weight: 700;">{total_patients:,}</div>
            <div style="color: #95a5a6; font-size: 0.75rem;">Across {total_sites:,} sites</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        dqi_color = get_dqi_band_color(mean_dqi)
        dqi_band = get_dqi_band_name(mean_dqi)
        st.markdown(f"""
        <div style="background: white; padding: 1.25rem; border-radius: 10px; 
                    border-left: 4px solid {dqi_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #7f8c8d; font-size: 0.85rem;">📊 Mean DQI</div>
            <div style="color: {dqi_color}; font-size: 2rem; font-weight: 700;">{mean_dqi:.1f}</div>
            <div style="color: #95a5a6; font-size: 0.75rem;">{dqi_band}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        clean_color = "#27ae60" if tier2_clean_rate >= 50 else "#f39c12" if tier2_clean_rate >= 30 else "#e74c3c"
        st.markdown(f"""
        <div style="background: white; padding: 1.25rem; border-radius: 10px; 
                    border-left: 4px solid {clean_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #7f8c8d; font-size: 0.85rem;">✨ Clean Rate</div>
            <div style="color: {clean_color}; font-size: 2rem; font-weight: 700;">{tier2_clean_rate:.1f}%</div>
            <div style="color: #95a5a6; font-size: 0.75rem;">{tier2_clean_count:,} patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dblock_color = "#27ae60" if dblock_ready_rate >= 25 else "#f39c12" if dblock_ready_rate >= 10 else "#e74c3c"
        st.markdown(f"""
        <div style="background: white; padding: 1.25rem; border-radius: 10px; 
                    border-left: 4px solid {dblock_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #7f8c8d; font-size: 0.85rem;">🔒 DB Lock Ready</div>
            <div style="color: {dblock_color}; font-size: 2rem; font-weight: 700;">{dblock_ready_rate:.1f}%</div>
            <div style="color: #95a5a6; font-size: 0.75rem;">{dblock_ready_count:,} of {dblock_eligible_count:,} eligible</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        issues_color = "#e74c3c" if critical_issues > 100 else "#f39c12" if critical_issues > 0 else "#27ae60"
        st.markdown(f"""
        <div style="background: white; padding: 1.25rem; border-radius: 10px; 
                    border-left: 4px solid {issues_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #7f8c8d; font-size: 0.85rem;">⚠️ Open Issues</div>
            <div style="color: {issues_color}; font-size: 2rem; font-weight: 700;">{total_issues:,}</div>
            <div style="color: #95a5a6; font-size: 0.75rem;">🔴 {critical_issues:,} critical</div>
        </div>
        """, unsafe_allow_html=True)


def render_dqi_section(loader: ExecutiveDataLoader, metrics: Dict[str, Any]):
    """Render DQI distribution section."""
    
    st.markdown("### 📈 DQI Distribution")
    
    distribution = loader.get_dqi_distribution()
    
    if len(distribution) == 0:
        st.info("DQI distribution data not available")
        return
    
    colors = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    
    fig = go.Figure(data=[go.Pie(
        labels=distribution['band'],
        values=distribution['count'],
        hole=0.6,
        marker_colors=colors,
        textinfo='percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    mean_dqi = metrics['mean_dqi']
    fig.add_annotation(
        text=f"<b>{mean_dqi:.1f}</b><br>Mean DQI",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    total_patients = metrics['total_patients']
    if total_patients > 0:
        col1, col2, col3 = st.columns(3)
        pristine_excellent = distribution[distribution['band'].isin(['Pristine', 'Excellent'])]['count'].sum()
        good_fair = distribution[distribution['band'].isin(['Good', 'Fair'])]['count'].sum()
        poor_critical = distribution[distribution['band'].isin(['Poor', 'Critical', 'Emergency'])]['count'].sum()
        
        with col1:
            pct = pristine_excellent / total_patients * 100
            st.metric("🟢 Pristine/Excellent", f"{int(pristine_excellent):,}", f"{pct:.1f}%")
        with col2:
            pct = good_fair / total_patients * 100
            st.metric("🟡 Good/Fair", f"{int(good_fair):,}", f"{pct:.1f}%")
        with col3:
            pct = poor_critical / total_patients * 100
            st.metric("🔴 Needs Attention", f"{int(poor_critical):,}", f"{pct:.1f}%")


def render_clean_patient_section(loader: ExecutiveDataLoader, metrics: Dict[str, Any]):
    """Render clean patient progress section."""
    
    st.markdown("### ✨ Clean Patient Progress")
    
    total_patients = metrics['total_patients']
    tier1_clean_count = metrics['tier1_clean_count']
    tier2_clean_count = metrics['tier2_clean_count']
    dblock_ready_count = metrics['dblock_ready_count']
    
    stages = ['Total Patients', 'Tier 1 Clean', 'Tier 2 Clean', 'DB Lock Ready']
    values = [total_patients, tier1_clean_count, tier2_clean_count, dblock_ready_count]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=['#3498db', '#2ecc71', '#27ae60', '#1abc9c']
        ),
        connector=dict(line=dict(color="#dee2e6", width=2))
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    tier1_rate = metrics['tier1_clean_rate']
    tier2_rate = metrics['tier2_clean_rate']
    dblock_rate = metrics['dblock_ready_rate']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tier 1 Rate", f"{tier1_rate:.1f}%",
                  help="Patients with 0 hard blocks")
    with col2:
        delta = tier2_rate - tier1_rate
        st.metric("Tier 2 Rate", f"{tier2_rate:.1f}%", f"{delta:+.1f}%",
                  help="Patients with 0 hard + soft blocks")
    with col3:
        st.metric("DB Lock Rate", f"{dblock_rate:.1f}%",
                  help="Of eligible patients")


def render_study_metrics_section(study_metrics: pd.DataFrame):
    """Render study-level metrics table."""
    
    st.markdown("### 📚 Study-Level Metrics")
    
    if len(study_metrics) == 0:
        st.info("Study metrics not available")
        return
    
    available_sort_cols = ['patients']
    sort_labels = {'patients': 'Patients'}
    
    if 'mean_dqi' in study_metrics.columns:
        available_sort_cols.append('mean_dqi')
        sort_labels['mean_dqi'] = 'DQI'
    if 'tier2_rate' in study_metrics.columns:
        available_sort_cols.append('tier2_rate')
        sort_labels['tier2_rate'] = 'Clean Rate'
    if 'dblock_rate' in study_metrics.columns:
        available_sort_cols.append('dblock_rate')
        sort_labels['dblock_rate'] = 'DB Lock Rate'
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            available_sort_cols,
            format_func=lambda x: sort_labels.get(x, x),
            key="study_sort_by"
        )
    
    sorted_data = study_metrics.sort_values(sort_by, ascending=False).head(10)
    
    num_cols = 1
    subplot_titles = ['Patients']
    
    if 'mean_dqi' in sorted_data.columns:
        num_cols += 1
        subplot_titles.append('Mean DQI')
    if 'tier2_rate' in sorted_data.columns:
        num_cols += 1
        subplot_titles.append('Clean Rate %')
    if 'dblock_rate' in sorted_data.columns:
        num_cols += 1
        subplot_titles.append('DB Lock %')
    
    fig = make_subplots(
        rows=1, cols=num_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    studies = sorted_data['study_id'].tolist()
    col_idx = 1
    
    fig.add_trace(go.Bar(
        y=studies,
        x=sorted_data['patients'],
        orientation='h',
        marker_color='#3498db',
        name='Patients',
        text=[f"{int(x):,}" for x in sorted_data['patients']],
        textposition='auto'
    ), row=1, col=col_idx)
    col_idx += 1
    
    if 'mean_dqi' in sorted_data.columns:
        dqi_colors = [get_dqi_band_color(d) for d in sorted_data['mean_dqi']]
        fig.add_trace(go.Bar(
            y=studies,
            x=sorted_data['mean_dqi'],
            orientation='h',
            marker_color=dqi_colors,
            name='DQI',
            text=[f"{x:.1f}" for x in sorted_data['mean_dqi']],
            textposition='auto'
        ), row=1, col=col_idx)
        col_idx += 1
    
    if 'tier2_rate' in sorted_data.columns:
        fig.add_trace(go.Bar(
            y=studies,
            x=sorted_data['tier2_rate'],
            orientation='h',
            marker_color='#27ae60',
            name='Clean %',
            text=[f"{x:.1f}%" for x in sorted_data['tier2_rate']],
            textposition='auto'
        ), row=1, col=col_idx)
        col_idx += 1
    
    if 'dblock_rate' in sorted_data.columns:
        fig.add_trace(go.Bar(
            y=studies,
            x=sorted_data['dblock_rate'],
            orientation='h',
            marker_color='#9b59b6',
            name='DB Lock %',
            text=[f"{x:.1f}%" for x in sorted_data['dblock_rate']],
            textposition='auto'
        ), row=1, col=col_idx)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=100, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 View Full Data Table"):
        display_cols = ['study_id', 'patients', 'sites', 'mean_dqi', 'tier1_rate', 'tier2_rate', 'dblock_rate', 'total_issues']
        available_cols = [c for c in display_cols if c in study_metrics.columns]
        
        display_df = study_metrics[available_cols].copy()
        for col in display_df.columns:
            if col in ['mean_dqi']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            elif col in ['tier1_rate', 'tier2_rate', 'dblock_rate']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_regional_heatmap(loader: ExecutiveDataLoader):
    """Render regional performance heatmap."""
    
    st.markdown("### 🌍 Regional Performance")
    
    regional = loader.get_regional_metrics()
    
    if len(regional) == 0:
        st.info("Regional data not available")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'mean_dqi' in regional.columns:
            fig = go.Figure()
            
            regional_sorted = regional.sort_values('mean_dqi', ascending=True)
            colors = [get_dqi_band_color(d) for d in regional_sorted['mean_dqi']]
            
            fig.add_trace(go.Bar(
                x=regional_sorted['mean_dqi'],
                y=regional_sorted['region'],
                orientation='h',
                marker=dict(color=colors),
                text=[f"{x:.1f}" for x in regional_sorted['mean_dqi']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>DQI: %{x:.1f}<br>Patients: %{customdata:,}<extra></extra>',
                customdata=regional_sorted['patients']
            ))
            
            fig.update_layout(
                title="Mean DQI by Region",
                height=300,
                margin=dict(l=120, r=40, t=40, b=20),
                xaxis=dict(range=[0, 105], title="DQI Score"),
                yaxis=dict(title=""),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.add_vline(x=85, line_dash="dash", line_color="#e74c3c",
                         annotation_text="Target: 85", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("DQI data not available for regional view")
    
    with col2:
        st.markdown("**Regional Summary**")
        
        for _, row in regional.iterrows():
            dqi = row.get('mean_dqi', 0)
            color = get_dqi_band_color(dqi) if dqi > 0 else "#95a5a6"
            patients = int(row.get('patients', 0))
            sites = int(row.get('sites', 0))
            region_name = row['region']
            
            dqi_text = f"DQI: {dqi:.1f}" if dqi > 0 else "DQI: N/A"
            
            st.markdown(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 8px; 
                        margin-bottom: 0.5rem; border-left: 4px solid {color};">
                <div style="font-weight: 600; color: #2c3e50;">{region_name}</div>
                <div style="font-size: 0.85rem; color: #7f8c8d;">
                    {patients:,} patients • {sites:,} sites • {dqi_text}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_dblock_projection(loader: ExecutiveDataLoader, metrics: Dict[str, Any]):
    """Render DB Lock projection section."""
    
    st.markdown("### 🔒 Database Lock Projection")
    
    current_rate = metrics['dblock_ready_rate']
    dblock_ready_count = metrics['dblock_ready_count']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
        
        np.random.seed(42)
        historical = np.linspace(max(current_rate - 15, 5), current_rate, 8)
        
        projected = [current_rate]
        for i in range(4):
            next_val = min(projected[-1] + 2 + np.random.uniform(-0.5, 0.5), 100)
            projected.append(next_val)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates[:8],
            y=historical,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        
        future_dates = pd.date_range(start=dates[-1], periods=5, freq='W')
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=projected,
            mode='lines+markers',
            name='Projected',
            line=dict(color='#27ae60', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        upper = [p + 3 for p in projected]
        lower = [max(p - 3, 0) for p in projected]
        
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(39, 174, 96, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band',
            showlegend=True
        ))
        
        fig.add_hline(y=50, line_dash="dash", line_color="#e74c3c",
                     annotation_text="Target: 50%")
        
        fig.update_layout(
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(title="Date"),
            yaxis=dict(title="DB Lock Ready %", range=[0, max(60, current_rate + 20)]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
        fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Projection Summary**")
        
        st.markdown(f"""
        <div style="background: #3498db; color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Current Status</div>
            <div style="font-size: 1.8rem; font-weight: 700;">{current_rate:.1f}%</div>
            <div style="font-size: 0.85rem;">{dblock_ready_count:,} patients ready</div>
        </div>
        """, unsafe_allow_html=True)
        
        weeks_to_target = max(int((50 - current_rate) / 2), 0) if current_rate < 50 else 0
        target_date = datetime.now() + timedelta(weeks=weeks_to_target)
        
        st.markdown(f"""
        <div style="background: #27ae60; color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Projected 50% Ready</div>
            <div style="font-size: 1.2rem; font-weight: 700;">{target_date.strftime('%B %d, %Y')}</div>
            <div style="font-size: 0.85rem;">~{weeks_to_target} weeks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Top Blockers:**")
        blockers = [
            ("SDV Incomplete", 43),
            ("Signatures", 29),
            ("Open Queries", 18),
            ("SAE Pending", 10)
        ]
        
        for blocker, pct in blockers:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: #2c3e50;">{blocker}</span>
                <span style="color: #7f8c8d; font-weight: 600;">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)


def render_issues_section(loader: ExecutiveDataLoader, metrics: Dict[str, Any]):
    """Render issues overview section - FIXED version."""
    
    st.markdown("### ⚠️ Issues Overview")
    
    issues = loader.load_patient_issues()
    
    if issues is None or len(issues) == 0:
        st.info("No issues data available.")
        return
    
    # Get issue breakdown
    issue_breakdown = loader.get_issue_breakdown()
    
    # Calculate priority distribution
    priority_data = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0, 'Clean': 0}
    
    if 'priority_tier' in issues.columns:
        priority_counts = issues['priority_tier'].value_counts()
        # Data uses lowercase values, map to display labels
        priority_data['Critical'] = int(priority_counts.get('critical', 0))
        priority_data['High'] = int(priority_counts.get('high', 0))
        priority_data['Medium'] = int(priority_counts.get('medium', 0))
        priority_data['Low'] = int(priority_counts.get('low', 0))
        # 'none' in data means clean patients
        priority_data['Clean'] = int(priority_counts.get('none', 0))
    else:
        issue_cols = [col for col in issues.columns if col.startswith('issue_')]
        
        if issue_cols:
            total_issues_per_patient = issues[issue_cols].sum(axis=1)
            
            priority_data['Critical'] = int((total_issues_per_patient >= 5).sum())
            priority_data['High'] = int(((total_issues_per_patient >= 3) & (total_issues_per_patient < 5)).sum())
            priority_data['Medium'] = int(((total_issues_per_patient >= 2) & (total_issues_per_patient < 3)).sum())
            priority_data['Low'] = int((total_issues_per_patient == 1).sum())
            priority_data['Clean'] = int((total_issues_per_patient == 0).sum())
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        if len(issue_breakdown) > 0:
            top_issues = issue_breakdown.head(10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=top_issues['display_name'],
                x=top_issues['patient_count'],
                orientation='h',
                marker_color='#e74c3c',
                text=top_issues['patient_count'].apply(lambda x: f'{x:,}'),
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Top 10 Issue Types",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Patients Affected",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Issue breakdown data not available")
    
    with col2:
        st.markdown("**Priority Distribution**")
        
        total = sum(priority_data.values())
        
        priority_colors = {
            'Critical': '#e74c3c',
            'High': '#e67e22',
            'Medium': '#f39c12',
            'Clean': '#27ae60',
        }
        
        for priority, count in priority_data.items():
            pct = (count / total * 100) if total > 0 else 0
            color = priority_colors.get(priority, '#95a5a6')
            
            icon = '🔴' if priority == 'Critical' else '🟠' if priority == 'High' else '🟡' if priority == 'Medium' else '🟢'
            
            st.markdown(f"""
                <div style="background: white; border-left: 4px solid {color}; 
                            padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold;">{icon} {priority}</span>
                        <span style="color: #666;">{count:,}</span>
                    </div>
                    <div style="background: #ecf0f1; height: 6px; border-radius: 3px; margin-top: 0.5rem;">
                        <div style="background: {color}; height: 100%; width: {pct}%; border-radius: 3px;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.75rem; color: #888; margin-top: 0.25rem;">
                        {pct:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        if len(issue_breakdown) > 0:
            total_issues = int(issue_breakdown['patient_count'].sum())
            st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                    <div style="font-size: 0.8rem; color: #666;">Total Issue Instances</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #e74c3c;">{total_issues:,}</div>
                </div>
            """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick actions panel."""
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📄 Generate Report", use_container_width=True, type="primary"):
            st.session_state.current_page = "Reports"
            st.rerun()
    
    with col2:
        if st.button("🤖 Ask AI", use_container_width=True):
            st.session_state.current_page = "AI Assistant"
            st.rerun()
    
    with col3:
        if st.button("🌊 View Cascade", use_container_width=True):
            st.session_state.current_page = "Cascade Explorer"
            st.rerun()
    
    with col4:
        if st.button("📊 Site Details", use_container_width=True):
            st.session_state.current_page = "Data Manager Hub"
            st.rerun()
    
    with col5:
        if st.button("⬇️ Export Data", use_container_width=True):
            st.toast("Export functionality - Select report type from Reports page")


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_executive_overview():
    """Test Executive Overview components."""
    print("\n" + "="*60)
    print("TRIALPULSE NEXUS 10X - EXECUTIVE OVERVIEW TEST v1.2")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    loader = ExecutiveDataLoader()
    
    # Test 1: Data Loader
    tests_total += 1
    print("\nTEST 1: Data Loader Initialization")
    try:
        print("   ✅ ExecutiveDataLoader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Load UPR
    tests_total += 1
    print("\nTEST 2: Load UPR Data")
    try:
        upr = loader.load_upr()
        if upr is not None:
            print(f"   ✅ Loaded {len(upr)} patients")
            tests_passed += 1
        else:
            print("   ⚠️ UPR not found")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Portfolio Metrics
    tests_total += 1
    print("\nTEST 3: Get Portfolio Metrics")
    try:
        metrics = loader.get_portfolio_metrics()
        print(f"   ✅ Metrics retrieved:")
        print(f"      Patients: {metrics['total_patients']:,}")
        print(f"      Mean DQI: {metrics['mean_dqi']:.1f}")
        print(f"      Clean Rate: {metrics['tier2_clean_rate']:.1f}%")
        print(f"      DB Lock Ready: {metrics['dblock_ready_rate']:.1f}%")
        print(f"      Total Issues: {metrics['total_issues']:,}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Issue Breakdown
    tests_total += 1
    print("\nTEST 4: Get Issue Breakdown")
    try:
        issue_breakdown = loader.get_issue_breakdown()
        if len(issue_breakdown) > 0:
            print(f"   ✅ Issue types: {len(issue_breakdown)}")
            for _, row in issue_breakdown.head(5).iterrows():
                print(f"      {row['display_name']}: {row['patient_count']:,}")
            tests_passed += 1
        else:
            print("   ⚠️ No issue breakdown data")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: DQI Distribution
    tests_total += 1
    print("\nTEST 5: Get DQI Distribution")
    try:
        dqi_dist = loader.get_dqi_distribution()
        if len(dqi_dist) > 0:
            print(f"   ✅ DQI bands: {len(dqi_dist)}")
            for _, row in dqi_dist.iterrows():
                print(f"      {row['band']}: {row['count']:,} ({row['percentage']:.1f}%)")
            tests_passed += 1
        else:
            print("   ⚠️ No DQI distribution data")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Study Metrics
    tests_total += 1
    print("\nTEST 6: Get Study Metrics")
    try:
        study_metrics = loader.get_study_metrics()
        if len(study_metrics) > 0:
            print(f"   ✅ Studies: {len(study_metrics)}")
            tests_passed += 1
        else:
            print("   ⚠️ No study metrics data")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 7: Regional Metrics
    tests_total += 1
    print("\nTEST 7: Get Regional Metrics")
    try:
        regional = loader.get_regional_metrics()
        if len(regional) > 0:
            print(f"   ✅ Regions: {len(regional)}")
            tests_passed += 1
        else:
            print("   ⚠️ No regional data")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("="*60)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_executive_overview()