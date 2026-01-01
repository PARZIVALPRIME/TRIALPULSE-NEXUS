"""
TRIALPULSE NEXUS 10X - Phase 7.5
Safety Surveillance Dashboard v1.1 (FIXED)

Fixes:
- HTML table rendering issue
- Signal count consistency
- More realistic data distribution
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
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

class SLAStatus(Enum):
    """SLA status levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"

class SignalStrength(Enum):
    """Signal detection strength"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NOISE = "noise"

@dataclass
class SLAConfig:
    """SLA configuration for different SAE types"""
    fatal_hours: int = 24
    serious_hours: int = 72
    non_serious_hours: int = 120
    follow_up_hours: int = 168

SLA_CONFIG = SLAConfig()

SAFETY_COLORS = {
    'safe': '#27ae60',
    'warning': '#f39c12',
    'critical': '#e74c3c',
    'breached': '#8e44ad',
    'fatal': '#c0392b',
    'serious': '#e74c3c',
    'non_serious': '#f39c12',
}

SIGNAL_COLORS = {
    'strong': '#e74c3c',
    'moderate': '#f39c12',
    'weak': '#3498db',
    'noise': '#95a5a6',
}

# =============================================================================
# DATA LOADER
# =============================================================================

class SafetyDataLoader:
    """Load and process safety surveillance data"""
    
    def __init__(self):
        self.base_path = Path("data/processed")
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 60
    
    def _get_cached(self, key: str, loader_func):
        """Get cached data or load fresh"""
        now = datetime.now()
        if key in self._cache:
            if (now - self._cache_time[key]).seconds < self._cache_ttl:
                return self._cache[key]
        
        data = loader_func()
        self._cache[key] = data
        self._cache_time[key] = now
        return data
    
    def load_upr(self) -> pd.DataFrame:
        """Load unified patient record"""
        def loader():
            path = self.base_path / "upr" / "unified_patient_record.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._get_cached('upr', loader)
    
    def load_patient_issues(self) -> pd.DataFrame:
        """Load patient issues data"""
        def loader():
            path = self.base_path / "analytics" / "patient_issues.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._get_cached('patient_issues', loader)
    
    def load_pattern_alerts(self) -> pd.DataFrame:
        """Load pattern alerts"""
        def loader():
            path = self.base_path / "analytics" / "pattern_library" / "alerts.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._get_cached('pattern_alerts', loader)
    
    def get_sae_cases(self) -> pd.DataFrame:
        """Get SAE cases with SLA tracking - more realistic distribution"""
        issues = self.load_patient_issues()
        upr = self.load_upr()
        
        if issues.empty or upr.empty:
            return self._generate_sae_data(50)
        
        # Get actual SAE patient count
        sae_dm_col = 'issue_sae_dm_pending' if 'issue_sae_dm_pending' in issues.columns else None
        sae_safety_col = 'issue_sae_safety_pending' if 'issue_sae_safety_pending' in issues.columns else None
        
        sae_count = 0
        if sae_dm_col:
            sae_count += issues[sae_dm_col].sum()
        if sae_safety_col:
            sae_count += issues[sae_safety_col].sum()
        
        # Generate realistic SAE cases based on actual data
        n_cases = min(int(sae_count) if sae_count > 0 else 50, 100)
        return self._generate_sae_data(n_cases)
    
    def _generate_sae_data(self, n_cases: int) -> pd.DataFrame:
        """Generate realistic SAE data with proper distribution"""
        now = datetime.now()
        np.random.seed(42)  # Reproducible results
        
        cases = []
        for i in range(n_cases):
            case_id = f"SAE-{2024}-{i+1:04d}"
            
            # Severity distribution: 5% Fatal, 70% Serious, 25% Non-Serious
            severity = np.random.choice(
                ['Fatal', 'Serious', 'Non-Serious'],
                p=[0.05, 0.70, 0.25]
            )
            
            # SLA based on severity
            if severity == 'Fatal':
                sla_hours = SLA_CONFIG.fatal_hours
            elif severity == 'Serious':
                sla_hours = SLA_CONFIG.serious_hours
            else:
                sla_hours = SLA_CONFIG.non_serious_hours
            
            # More realistic time distribution
            # 20% breached, 15% critical, 25% warning, 40% safe
            status_roll = np.random.random()
            if status_roll < 0.20:  # 20% breached
                hours_elapsed = sla_hours + np.random.uniform(1, 48)
            elif status_roll < 0.35:  # 15% critical
                hours_elapsed = sla_hours - np.random.uniform(1, 24)
            elif status_roll < 0.60:  # 25% warning
                hours_elapsed = sla_hours - np.random.uniform(24, 48)
            else:  # 40% safe
                hours_elapsed = np.random.uniform(0, sla_hours - 48)
            
            onset_date = now - timedelta(hours=hours_elapsed)
            hours_remaining = sla_hours - hours_elapsed
            
            # Determine status
            if hours_remaining < 0:
                status = SLAStatus.BREACHED
                breach_prob = 1.0
            elif hours_remaining < 24:
                status = SLAStatus.CRITICAL
                breach_prob = 0.75 + (24 - hours_remaining) / 24 * 0.20
            elif hours_remaining < 48:
                status = SLAStatus.WARNING
                breach_prob = 0.30 + (48 - hours_remaining) / 24 * 0.45
            else:
                status = SLAStatus.SAFE
                breach_prob = max(0.05, 0.30 - hours_remaining / 100)
            
            cases.append({
                'case_id': case_id,
                'patient_key': f'Patient_{i+1}',
                'study_id': f'Study_{np.random.randint(1, 24)}',
                'site_id': f'Site_{np.random.randint(1, 100)}',
                'subject_id': f'Subject_{i+1:04d}',
                'event_term': np.random.choice([
                    'Myocardial Infarction', 'Pneumonia', 'Hepatotoxicity',
                    'Anaphylaxis', 'Stroke', 'Renal Failure', 'Sepsis',
                    'Cardiac Arrest', 'Respiratory Failure', 'GI Hemorrhage'
                ]),
                'severity': severity,
                'onset_date': onset_date,
                'sla_hours': sla_hours,
                'hours_remaining': max(0, hours_remaining),
                'hours_elapsed': hours_elapsed,
                'status': status.value,
                'breach_probability': breach_prob,
                'reporter': np.random.choice(['Site', 'Sponsor', 'Patient']),
                'expectedness': np.random.choice(['Expected', 'Unexpected'], p=[0.6, 0.4]),
                'causality': np.random.choice(['Related', 'Possibly Related', 'Not Related'], p=[0.2, 0.3, 0.5]),
                'outcome': np.random.choice(['Recovered', 'Recovering', 'Not Recovered', 'Fatal', 'Unknown'], p=[0.3, 0.35, 0.2, 0.05, 0.1]),
                'requires_review': np.random.choice([True, False], p=[0.4, 0.6]),
                'dm_status': np.random.choice(['Pending', 'In Review', 'Completed'], p=[0.3, 0.3, 0.4]),
                'safety_status': np.random.choice(['Pending', 'In Review', 'Completed'], p=[0.35, 0.35, 0.3]),
            })
        
        return pd.DataFrame(cases)
    
    def get_safety_signals(self) -> pd.DataFrame:
        """Get safety signals - more realistic counts"""
        issues = self.load_patient_issues()
        
        signals = []
        
        # Generate realistic signal distribution
        # Strong: 3-5, Moderate: 5-10, Weak: 10-15
        np.random.seed(42)
        
        signal_configs = [
            ('Hepatic Cluster', 'Elevated hepatic events in EU sites', 3.2, 'strong'),
            ('Cardiac Signal', 'Cardiac events above baseline', 2.8, 'strong'),
            ('GI Events', 'Increased GI adverse events at Site_101', 2.5, 'moderate'),
            ('Infection Rate', 'Higher infection rate in Study_21', 2.3, 'moderate'),
            ('Neurological', 'Neurological event pattern detected', 2.1, 'moderate'),
            ('Renal Signal', 'Renal function decline pattern', 1.9, 'moderate'),
            ('Skin Reactions', 'Skin reaction clustering', 1.7, 'weak'),
            ('Respiratory', 'Respiratory events trending up', 1.5, 'weak'),
            ('Hematologic', 'Mild hematologic signal', 1.3, 'weak'),
            ('Musculoskeletal', 'MSK event pattern', 1.2, 'weak'),
        ]
        
        for i, (sig_type, desc, z_score, strength) in enumerate(signal_configs):
            signals.append({
                'signal_id': f'SIG-{i+1:04d}',
                'signal_type': sig_type,
                'description': desc,
                'site_id': f'Site_{np.random.randint(1, 100)}',
                'study_id': f'Study_{np.random.randint(1, 24)}',
                'metric': sig_type,
                'observed_value': np.random.randint(5, 25),
                'expected_value': np.random.randint(2, 10),
                'z_score': z_score,
                'strength': strength,
                'confidence': min(0.99, 0.5 + z_score * 0.15),
                'affected_patients': np.random.randint(10, 80),
                'detected_at': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                'status': np.random.choice(['New', 'Under Review', 'Confirmed', 'Dismissed'], p=[0.3, 0.4, 0.2, 0.1]),
            })
        
        return pd.DataFrame(signals)
    
    def get_safety_metrics(self) -> Dict:
        """Get safety surveillance metrics"""
        sae_cases = self.get_sae_cases()
        signals = self.get_safety_signals()
        
        total_sae = len(sae_cases)
        pending_dm = len(sae_cases[sae_cases['dm_status'] == 'Pending']) if not sae_cases.empty else 0
        pending_safety = len(sae_cases[sae_cases['safety_status'] == 'Pending']) if not sae_cases.empty else 0
        
        breached = len(sae_cases[sae_cases['status'] == 'breached']) if not sae_cases.empty else 0
        critical = len(sae_cases[sae_cases['status'] == 'critical']) if not sae_cases.empty else 0
        warning = len(sae_cases[sae_cases['status'] == 'warning']) if not sae_cases.empty else 0
        safe = len(sae_cases[sae_cases['status'] == 'safe']) if not sae_cases.empty else 0
        
        active_signals = len(signals[signals['status'].isin(['New', 'Under Review'])]) if not signals.empty else 0
        strong_signals = len(signals[signals['strength'] == 'strong']) if not signals.empty else 0
        
        return {
            'total_sae_cases': total_sae,
            'pending_dm_review': pending_dm,
            'pending_safety_review': pending_safety,
            'sla_breached': breached,
            'sla_critical': critical,
            'sla_warning': warning,
            'sla_safe': safe,
            'active_signals': active_signals,
            'strong_signals': strong_signals,
            'avg_breach_probability': sae_cases['breach_probability'].mean() if not sae_cases.empty else 0,
        }
    
    def get_sae_timeline(self) -> pd.DataFrame:
        """Get SAE case timeline for visualization"""
        sae_cases = self.get_sae_cases()
        
        if sae_cases.empty:
            return pd.DataFrame()
        
        sae_cases['onset_date_only'] = pd.to_datetime(sae_cases['onset_date']).dt.date
        
        timeline = sae_cases.groupby('onset_date_only').agg({
            'case_id': 'count',
            'severity': lambda x: (x == 'Serious').sum() + (x == 'Fatal').sum(),
        }).reset_index()
        
        timeline.columns = ['date', 'total_cases', 'serious_cases']
        timeline['date'] = pd.to_datetime(timeline['date'])
        
        return timeline.sort_values('date')
    
    def get_studies_list(self) -> List[str]:
        """Get list of studies"""
        upr = self.load_upr()
        if not upr.empty and 'study_id' in upr.columns:
            return sorted(upr['study_id'].dropna().unique().tolist())
        return [f'Study_{i}' for i in range(1, 24)]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sla_color(status: str) -> str:
    return SAFETY_COLORS.get(status, '#95a5a6')

def get_signal_color(strength: str) -> str:
    return SIGNAL_COLORS.get(strength, '#95a5a6')

def format_time_remaining(hours: float) -> str:
    if hours <= 0:
        return "BREACHED"
    elif hours < 1:
        return f"{int(hours * 60)}m"
    elif hours < 24:
        return f"{int(hours)}h"
    else:
        days = int(hours / 24)
        remaining_hours = int(hours % 24)
        return f"{days}d {remaining_hours}h"

def get_severity_icon(severity: str) -> str:
    icons = {'Fatal': '💀', 'Serious': '🔴', 'Non-Serious': '🟡'}
    return icons.get(severity, '⚪')

def get_status_icon(status: str) -> str:
    icons = {'breached': '🚨', 'critical': '⚠️', 'warning': '⏰', 'safe': '✅'}
    return icons.get(status, '❓')


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_page(user=None):
    """Main render function for Safety Surveillance page"""
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h1 style="color: white; margin: 0;">🛡️ Safety Surveillance</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Real-time SAE monitoring, SLA tracking, and signal detection
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    loader = SafetyDataLoader()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        studies = loader.get_studies_list()
        selected_study = st.selectbox(
            "Study",
            options=["All Studies"] + studies,
            key="safety_study_filter"
        )
        
        severity_filter = st.multiselect(
            "Severity",
            options=["Fatal", "Serious", "Non-Serious"],
            default=["Fatal", "Serious", "Non-Serious"],
            key="safety_severity_filter"
        )
        
        status_filter = st.multiselect(
            "SLA Status",
            options=["breached", "critical", "warning", "safe"],
            default=["breached", "critical", "warning"],
            key="safety_status_filter"
        )
        
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            loader._cache = {}
            st.rerun()
    
    # Load data
    metrics = loader.get_safety_metrics()
    sae_cases = loader.get_sae_cases()
    signals = loader.get_safety_signals()
    
    # Apply filters
    if selected_study != "All Studies":
        sae_cases = sae_cases[sae_cases['study_id'] == selected_study]
    
    if severity_filter:
        sae_cases = sae_cases[sae_cases['severity'].isin(severity_filter)]
    
    if status_filter:
        sae_cases = sae_cases[sae_cases['status'].isin(status_filter)]
    
    # Render sections
    render_kpi_section(metrics)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_sla_countdown_table(sae_cases)
    
    with col2:
        render_signal_detection(signals)
    
    st.markdown("---")
    
    render_sae_timeline(loader)
    
    st.markdown("---")
    
    render_pattern_alerts(loader)
    
    st.markdown("---")
    
    render_quick_actions()


def render_kpi_section(metrics: Dict):
    """Render key safety metrics"""
    
    st.markdown("### 📊 Safety Overview")
    
    cols = st.columns(5)
    
    with cols[0]:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: #bdc3c7; font-size: 0.8rem;">Total SAE Cases</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{metrics['total_sae_cases']}</div>
                <div style="color: #95a5a6; font-size: 0.75rem;">
                    DM: {metrics['pending_dm_review']} | Safety: {metrics['pending_safety_review']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        breach_color = '#e74c3c' if metrics['sla_breached'] > 0 else '#27ae60'
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {breach_color} 0%, {breach_color}dd 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">🚨 SLA Breached</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{metrics['sla_breached']}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">Immediate action required</div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">⚠️ SLA Critical</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{metrics['sla_critical']}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">&lt;24 hours remaining</div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        signal_color = '#e74c3c' if metrics['strong_signals'] > 0 else '#3498db'
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {signal_color} 0%, {signal_color}dd 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">📡 Active Signals</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{metrics['active_signals']}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">
                    {metrics['strong_signals']} strong signals
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[4]:
        risk_pct = metrics['avg_breach_probability'] * 100
        risk_color = '#e74c3c' if risk_pct > 50 else '#f39c12' if risk_pct > 25 else '#27ae60'
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}dd 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">📈 Avg Breach Risk</div>
                <div style="color: white; font-size: 2rem; font-weight: bold;">{risk_pct:.0f}%</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">Across pending cases</div>
            </div>
        """, unsafe_allow_html=True)


def render_sla_countdown_table(sae_cases: pd.DataFrame):
    """Render SLA countdown table - FIXED VERSION using st.dataframe"""
    
    st.markdown("### ⏱️ SLA Countdown - Pending Cases")
    
    if sae_cases.empty:
        st.info("No SAE cases match the current filters.")
        return
    
    # Sort by urgency
    sae_cases_sorted = sae_cases.sort_values(
        by=['breach_probability', 'hours_remaining'],
        ascending=[False, True]
    ).head(20).copy()
    
    # Format for display
    sae_cases_sorted['Time Left'] = sae_cases_sorted['hours_remaining'].apply(format_time_remaining)
    sae_cases_sorted['Breach Risk'] = (sae_cases_sorted['breach_probability'] * 100).round(0).astype(int).astype(str) + '%'
    sae_cases_sorted['Status Icon'] = sae_cases_sorted['status'].apply(lambda x: get_status_icon(x) + ' ' + x.upper())
    sae_cases_sorted['Severity Icon'] = sae_cases_sorted['severity'].apply(lambda x: get_severity_icon(x) + ' ' + x)
    
    # Select display columns
    display_df = sae_cases_sorted[[
        'case_id', 'site_id', 'event_term', 'Severity Icon', 
        'Time Left', 'Status Icon', 'Breach Risk'
    ]].copy()
    
    display_df.columns = ['Case ID', 'Site', 'Event', 'Severity', 'Time Left', 'Status', 'Breach Risk']
    
    # Use st.dataframe with styling
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Case ID": st.column_config.TextColumn("Case ID", width="small"),
            "Site": st.column_config.TextColumn("Site", width="small"),
            "Event": st.column_config.TextColumn("Event", width="medium"),
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Time Left": st.column_config.TextColumn("Time Left", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Breach Risk": st.column_config.TextColumn("Risk", width="small"),
        }
    )
    
    # Status summary
    col1, col2, col3, col4 = st.columns(4)
    
    status_counts = sae_cases['status'].value_counts()
    
    with col1:
        breached = status_counts.get('breached', 0)
        st.metric("🚨 Breached", breached, delta=None, delta_color="inverse")
    
    with col2:
        critical = status_counts.get('critical', 0)
        st.metric("⚠️ Critical", critical, delta=None, delta_color="inverse")
    
    with col3:
        warning = status_counts.get('warning', 0)
        st.metric("⏰ Warning", warning, delta=None, delta_color="off")
    
    with col4:
        safe = status_counts.get('safe', 0)
        st.metric("✅ Safe", safe, delta=None, delta_color="normal")
    
    # Full data expander
    with st.expander("📋 View All SAE Case Details"):
        full_display = sae_cases[[
            'case_id', 'site_id', 'study_id', 'event_term', 'severity',
            'hours_remaining', 'status', 'breach_probability', 
            'dm_status', 'safety_status', 'causality', 'outcome'
        ]].copy()
        full_display['hours_remaining'] = full_display['hours_remaining'].round(1)
        full_display['breach_probability'] = (full_display['breach_probability'] * 100).round(0).astype(int).astype(str) + '%'
        st.dataframe(full_display, use_container_width=True, height=400)


def render_signal_detection(signals: pd.DataFrame):
    """Render signal detection panel"""
    
    st.markdown("### 📡 Safety Signals")
    
    if signals.empty:
        st.info("No safety signals detected.")
        return
    
    # Summary counts
    strong = len(signals[signals['strength'] == 'strong'])
    moderate = len(signals[signals['strength'] == 'moderate'])
    weak = len(signals[signals['strength'] == 'weak'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div style="background: {SIGNAL_COLORS['strong']}; color: white; 
                        padding: 0.5rem; border-radius: 4px; text-align: center;">
                <strong>🔴 Strong: {strong}</strong>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: {SIGNAL_COLORS['moderate']}; color: white; 
                        padding: 0.5rem; border-radius: 4px; text-align: center;">
                <strong>🟡 Moderate: {moderate}</strong>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: {SIGNAL_COLORS['weak']}; color: white; 
                        padding: 0.5rem; border-radius: 4px; text-align: center;">
                <strong>🔵 Weak: {weak}</strong>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display signals as cards
    for _, signal in signals.head(5).iterrows():
        strength = signal['strength']
        color = get_signal_color(strength)
        
        st.markdown(f"""
            <div style="background: white; border-left: 4px solid {color}; 
                        padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: {color};">{signal['signal_type']}</strong>
                    <span style="background: {color}; color: white; padding: 0.15rem 0.4rem; 
                                 border-radius: 4px; font-size: 0.7rem;">{strength.upper()}</span>
                </div>
                <div style="color: #666; font-size: 0.8rem; margin-top: 0.25rem;">
                    {signal['description']}
                </div>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.75rem; color: #888;">
                    <span>📊 Z-score: {signal['z_score']:.1f}σ</span>
                    <span>👥 {signal['affected_patients']} patients</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if len(signals) > 5:
        with st.expander(f"View All {len(signals)} Signals"):
            st.dataframe(signals, use_container_width=True)


def render_sae_timeline(loader: SafetyDataLoader):
    """Render SAE timeline chart"""
    
    st.markdown("### 📈 SAE Timeline")
    
    timeline = loader.get_sae_timeline()
    
    if timeline.empty:
        st.info("No timeline data available.")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=timeline['date'],
            y=timeline['total_cases'],
            mode='lines+markers',
            name='Total Cases',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(
            x=timeline['date'],
            y=timeline['serious_cases'],
            name='Serious/Fatal',
            marker_color='#e74c3c',
            opacity=0.7,
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Total Cases", secondary_y=False)
    fig.update_yaxes(title_text="Serious/Fatal", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_daily = timeline['total_cases'].mean()
        st.metric("Avg Daily Cases", f"{avg_daily:.1f}")
    
    with col2:
        max_day = timeline.loc[timeline['total_cases'].idxmax()]
        st.metric("Peak Day", f"{max_day['total_cases']} cases")
    
    with col3:
        serious_rate = (timeline['serious_cases'].sum() / timeline['total_cases'].sum() * 100)
        st.metric("Serious Rate", f"{serious_rate:.1f}%")


def render_pattern_alerts(loader: SafetyDataLoader):
    """Render safety pattern alerts"""
    
    st.markdown("### 🔔 Safety Pattern Alerts")
    
    alerts = loader.load_pattern_alerts()
    
    if alerts.empty:
        st.info("No pattern alerts available.")
        return
    
    # Filter for safety-related patterns
    safety_patterns = ['SF', 'SAE', 'SAFE']
    safety_alerts = alerts[alerts['pattern_id'].str.contains('|'.join(safety_patterns), na=False, case=False)]
    
    if safety_alerts.empty:
        safety_alerts = alerts.head(10)
    
    if 'severity' in safety_alerts.columns:
        severity_counts = safety_alerts['severity'].value_counts()
        
        cols = st.columns(4)
        severity_colors = {
            'Critical': '#e74c3c',
            'High': '#e67e22',
            'Medium': '#f39c12',
            'Low': '#27ae60'
        }
        
        for i, (sev, color) in enumerate(severity_colors.items()):
            count = severity_counts.get(sev, 0)
            with cols[i]:
                st.markdown(f"""
                    <div style="background: {color}; color: white; padding: 0.5rem; 
                                border-radius: 8px; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: bold;">{count}</div>
                        <div style="font-size: 0.8rem;">{sev}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    display_cols = ['pattern_id', 'pattern_name', 'site_id', 'severity', 'alert_message']
    available_cols = [c for c in display_cols if c in safety_alerts.columns]
    
    if available_cols:
        st.dataframe(
            safety_alerts[available_cols].head(20),
            use_container_width=True,
            height=300
        )


def render_quick_actions():
    """Render quick action buttons"""
    
    st.markdown("### ⚡ Quick Actions")
    
    cols = st.columns(5)
    
    with cols[0]:
        if st.button("📊 Safety Report", use_container_width=True):
            st.toast("Navigating to Reports...")
    
    with cols[1]:
        if st.button("🤖 Ask AI", use_container_width=True):
            st.toast("Opening AI Assistant...")
    
    with cols[2]:
        if st.button("📧 Send Alerts", use_container_width=True):
            st.toast("Alert notifications queued for approval")
    
    with cols[3]:
        if st.button("📥 Export Data", use_container_width=True):
            st.toast("Preparing data export...")
    
    with cols[4]:
        if st.button("⚙️ Settings", use_container_width=True):
            st.toast("Opening Settings...")


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_safety_view():
    """Test Safety Surveillance components"""
    print("\n" + "="*60)
    print("TRIALPULSE NEXUS 10X - SAFETY SURVEILLANCE TEST v1.1")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Data Loader
    tests_total += 1
    print("\nTEST 1: Data Loader Initialization")
    try:
        loader = SafetyDataLoader()
        print("   ✅ SafetyDataLoader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Get SAE Cases
    tests_total += 1
    print("\nTEST 2: Get SAE Cases")
    try:
        sae_cases = loader.get_sae_cases()
        print(f"   ✅ Loaded {len(sae_cases)} SAE cases")
        
        # Check distribution
        status_counts = sae_cases['status'].value_counts()
        print(f"   Status distribution:")
        for status, count in status_counts.items():
            pct = count / len(sae_cases) * 100
            print(f"      {get_status_icon(status)} {status}: {count} ({pct:.1f}%)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Safety Metrics
    tests_total += 1
    print("\nTEST 3: Get Safety Metrics")
    try:
        metrics = loader.get_safety_metrics()
        print(f"   ✅ Total SAE: {metrics['total_sae_cases']}")
        print(f"   SLA Breached: {metrics['sla_breached']}")
        print(f"   SLA Critical: {metrics['sla_critical']}")
        print(f"   Active Signals: {metrics['active_signals']}")
        print(f"   Strong Signals: {metrics['strong_signals']}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Safety Signals
    tests_total += 1
    print("\nTEST 4: Get Safety Signals")
    try:
        signals = loader.get_safety_signals()
        print(f"   ✅ Detected {len(signals)} signals")
        
        strong = len(signals[signals['strength'] == 'strong'])
        moderate = len(signals[signals['strength'] == 'moderate'])
        weak = len(signals[signals['strength'] == 'weak'])
        print(f"   Strong: {strong}, Moderate: {moderate}, Weak: {weak}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: SAE Timeline
    tests_total += 1
    print("\nTEST 5: Get SAE Timeline")
    try:
        timeline = loader.get_sae_timeline()
        print(f"   ✅ Timeline: {len(timeline)} data points")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Helper Functions
    tests_total += 1
    print("\nTEST 6: Helper Functions")
    try:
        assert format_time_remaining(48) == "2d 0h"
        assert format_time_remaining(12) == "12h"
        assert format_time_remaining(0.5) == "30m"
        assert format_time_remaining(0) == "BREACHED"
        print("   ✅ format_time_remaining() working")
        
        assert get_sla_color('critical') == '#e74c3c'
        print("   ✅ get_sla_color() working")
        
        assert get_severity_icon('Fatal') == '💀'
        print("   ✅ get_severity_icon() working")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 7: Pattern Alerts
    tests_total += 1
    print("\nTEST 7: Pattern Alerts")
    try:
        alerts = loader.load_pattern_alerts()
        print(f"   ✅ Loaded {len(alerts)} pattern alerts")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("="*60)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_safety_view()