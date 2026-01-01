"""
TRIALPULSE NEXUS 10X - Dashboard Data Loader
Phase 7.1: Centralized data loading with caching
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import streamlit as st

from dashboard.config.session import cache_data, get_cached_data


class DashboardDataLoader:
    """Centralized data loader for dashboard with caching."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base directory for data files
        """
        if data_dir is None:
            # Default to project data directory
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        self._data_cache = {}
        self._last_load = None
    
    @property
    def upr_path(self) -> Path:
        return self.data_dir / "upr" / "unified_patient_record.parquet"
    
    @property
    def analytics_path(self) -> Path:
        return self.data_dir / "analytics"
    
    def load_upr(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load Unified Patient Record."""
        cache_key = "upr"
        
        if use_cache:
            cached = get_cached_data(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.upr_path.exists():
                df = pd.read_parquet(self.upr_path)
                cache_data(cache_key, df, ttl_seconds=300)
                return df
        except Exception as e:
            st.warning(f"Could not load UPR: {e}")
        
        return None
    
    def load_patient_issues(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load patient issues data."""
        return self._load_analytics_file("patient_issues.parquet", use_cache)
    
    def load_patient_dqi(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load patient DQI data."""
        return self._load_analytics_file("patient_dqi_enhanced.parquet", use_cache)
    
    def load_patient_clean_status(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load patient clean status data."""
        return self._load_analytics_file("patient_clean_status.parquet", use_cache)
    
    def load_patient_dblock(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load patient DB lock status data."""
        return self._load_analytics_file("patient_dblock_status.parquet", use_cache)
    
    def load_site_benchmarks(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load site benchmarks data."""
        return self._load_analytics_file("site_benchmarks.parquet", use_cache)
    
    def load_cascade_analysis(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load cascade analysis data."""
        return self._load_analytics_file("patient_cascade_analysis.parquet", use_cache)
    
    def load_pattern_matches(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load pattern matches data."""
        path = self.analytics_path / "pattern_library" / "pattern_matches.parquet"
        return self._load_file(path, "pattern_matches", use_cache)
    
    def load_resolution_recommendations(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load resolution recommendations data."""
        path = self.analytics_path / "resolution_genome" / "patient_recommendations.parquet"
        return self._load_file(path, "resolution_recommendations", use_cache)
    
    def _load_analytics_file(self, filename: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load a file from the analytics directory."""
        path = self.analytics_path / filename
        cache_key = filename.replace(".parquet", "")
        return self._load_file(path, cache_key, use_cache)
    
    def _load_file(self, path: Path, cache_key: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load a parquet file with caching."""
        if use_cache:
            cached = get_cached_data(cache_key)
            if cached is not None:
                return cached
        
        try:
            if path.exists():
                df = pd.read_parquet(path)
                cache_data(cache_key, df, ttl_seconds=300)
                return df
        except Exception as e:
            st.warning(f"Could not load {cache_key}: {e}")
        
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio-level summary statistics."""
        cache_key = "portfolio_summary"
        cached = get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        summary = {
            "total_patients": 0,
            "total_sites": 0,
            "total_studies": 0,
            "mean_dqi": 0.0,
            "tier2_clean_rate": 0.0,
            "dblock_ready_rate": 0.0,
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0
        }
        
        upr = self.load_upr()
        if upr is not None:
            summary["total_patients"] = len(upr)
            summary["total_sites"] = upr['site_id'].nunique() if 'site_id' in upr.columns else 0
            summary["total_studies"] = upr['study_id'].nunique() if 'study_id' in upr.columns else 0
        
        dqi = self.load_patient_dqi()
        if dqi is not None and 'dqi_score' in dqi.columns:
            summary["mean_dqi"] = dqi['dqi_score'].mean()
        
        clean = self.load_patient_clean_status()
        if clean is not None and 'tier2_clean' in clean.columns:
            summary["tier2_clean_rate"] = clean['tier2_clean'].mean() * 100
        
        dblock = self.load_patient_dblock()
        if dblock is not None and 'dblock_ready' in dblock.columns:
            summary["dblock_ready_rate"] = dblock['dblock_ready'].mean() * 100
        
        issues = self.load_patient_issues()
        if issues is not None:
            if 'total_issues' in issues.columns:
                summary["total_issues"] = issues['total_issues'].sum()
            if 'priority' in issues.columns:
                summary["critical_issues"] = len(issues[issues['priority'] == 'Critical'])
                summary["high_issues"] = len(issues[issues['priority'] == 'High'])
        
        cache_data(cache_key, summary, ttl_seconds=60)
        return summary
    
    def get_study_list(self) -> list:
        """Get list of all studies."""
        upr = self.load_upr()
        if upr is not None and 'study_id' in upr.columns:
            return sorted(upr['study_id'].unique().tolist())
        return []
    
    def get_site_list(self, study_id: Optional[str] = None) -> list:
        """Get list of sites, optionally filtered by study."""
        upr = self.load_upr()
        if upr is not None and 'site_id' in upr.columns:
            if study_id and study_id != "All Studies":
                filtered = upr[upr['study_id'] == study_id]
                return sorted(filtered['site_id'].unique().tolist())
            return sorted(upr['site_id'].unique().tolist())
        return []
    
    def refresh_all(self):
        """Clear all cached data to force refresh."""
        st.session_state.cached_data = {}
        st.session_state.data_loaded = False
        self._last_load = datetime.now()


# Singleton instance
_data_loader: Optional[DashboardDataLoader] = None


def get_data_loader() -> DashboardDataLoader:
    """Get the singleton data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader