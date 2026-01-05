# dashboard/pages/collaboration_hub.py
"""
TRIALPULSE NEXUS 10X - Collaboration Hub
Phase 8: Investigation Rooms, Team Workspaces, @Tagging, Escalation, Alerts, Issues
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config.theme import THEME_CONFIG

# =============================================================================
# DATA LOADER
# =============================================================================

class CollaborationDataLoader:
    """Data loader for Collaboration Hub with demo fallbacks."""
    
    def __init__(self):
        self._managers_loaded = False
        self._rooms_manager = None
        self._workspaces_manager = None
        self._tagging_system = None
        self._escalation_engine = None
        self._alert_system = None
        self._issue_registry = None
    
    def _load_managers(self):
        """Lazy load collaboration managers."""
        if self._managers_loaded:
            return
        try:
            from src.collaboration import (
                get_investigation_rooms_manager,
                get_team_workspaces_manager,
                get_tagging_system,
                get_escalation_engine,
                get_alert_system,
                get_issue_registry
            )
            self._rooms_manager = get_investigation_rooms_manager()
            self._workspaces_manager = get_team_workspaces_manager()
            self._tagging_system = get_tagging_system()
            self._escalation_engine = get_escalation_engine()
            self._alert_system = get_alert_system()
            self._issue_registry = get_issue_registry()
        except Exception:
            pass
        self._managers_loaded = True
    
    def get_investigation_rooms(self) -> List[Dict]:
        """Get investigation rooms."""
        self._load_managers()
        if self._rooms_manager:
            try:
                rooms = self._rooms_manager.get_all_rooms()
                return [r.to_dict() if hasattr(r, 'to_dict') else r for r in rooms]
            except Exception:
                pass
        return self._demo_rooms()
    
    def get_workspaces(self) -> List[Dict]:
        """Get team workspaces."""
        self._load_managers()
        if self._workspaces_manager:
            try:
                ws = self._workspaces_manager.get_all_workspaces()
                return [w.to_dict() if hasattr(w, 'to_dict') else w for w in ws]
            except Exception:
                pass
        return self._demo_workspaces()
    
    def get_escalations(self) -> List[Dict]:
        """Get active escalations."""
        self._load_managers()
        if self._escalation_engine:
            try:
                esc = self._escalation_engine.get_active_escalations()
                return [e.to_dict() if hasattr(e, 'to_dict') else e for e in esc]
            except Exception:
                pass
        return self._demo_escalations()
    
    def get_alerts(self, user_id: str = "current_user") -> List[Dict]:
        """Get user alerts."""
        self._load_managers()
        if self._alert_system:
            try:
                alerts = self._alert_system.get_user_alerts(user_id)
                return [a.to_dict() if hasattr(a, 'to_dict') else a for a in alerts]
            except Exception:
                pass
        return self._demo_alerts()
    
    def get_issues(self) -> List[Dict]:
        """Get issues from registry."""
        self._load_managers()
        if self._issue_registry:
            try:
                issues = self._issue_registry.get_all_issues()
                return [i.to_dict() if hasattr(i, 'to_dict') else i for i in issues]
            except Exception:
                pass
        return self._demo_issues()
    
    def get_collaboration_stats(self) -> Dict:
        """Get overall collaboration statistics."""
        rooms = self.get_investigation_rooms()
        workspaces = self.get_workspaces()
        escalations = self.get_escalations()
        issues = self.get_issues()
        
        active_rooms = len([r for r in rooms if r.get('status') == 'active'])
        active_workspaces = len([w for w in workspaces if w.get('status') == 'active'])
        pending_escalations = len([e for e in escalations if e.get('status') != 'resolved'])
        open_issues = len([i for i in issues if i.get('status') not in ['resolved', 'closed']])
        
        return {
            'total_rooms': len(rooms),
            'active_rooms': active_rooms,
            'total_workspaces': len(workspaces),
            'active_workspaces': active_workspaces,
            'total_escalations': len(escalations),
            'pending_escalations': pending_escalations,
            'total_issues': len(issues),
            'open_issues': open_issues
        }
    
    # Demo data methods
    def _demo_rooms(self) -> List[Dict]:
        return [
            {'room_id': 'ROOM-001', 'title': 'ASIA TQI Drop Investigation', 'room_type': 'issue', 
             'status': 'active', 'created_at': datetime.now() - timedelta(days=2), 
             'participant_count': 5, 'evidence_count': 12, 'thread_count': 8,
             'description': 'Investigating 12-point TQI drop in ASIA region'},
            {'room_id': 'ROOM-002', 'title': 'Site JP-101 Performance Review', 'room_type': 'site',
             'status': 'active', 'created_at': datetime.now() - timedelta(days=5),
             'participant_count': 4, 'evidence_count': 8, 'thread_count': 5,
             'description': 'Review underperforming site patterns'},
            {'room_id': 'ROOM-003', 'title': 'SAE Cluster EU Analysis', 'room_type': 'safety',
             'status': 'pending_review', 'created_at': datetime.now() - timedelta(days=1),
             'participant_count': 6, 'evidence_count': 15, 'thread_count': 12,
             'description': 'Hepatic cluster analysis - 3 cases above baseline'},
            {'room_id': 'ROOM-004', 'title': 'Q4 DB Lock Readiness', 'room_type': 'study',
             'status': 'active', 'created_at': datetime.now() - timedelta(days=10),
             'participant_count': 8, 'evidence_count': 20, 'thread_count': 15,
             'description': 'Tracking DB lock readiness across all regions'}
        ]
    
    def _demo_workspaces(self) -> List[Dict]:
        return [
            {'workspace_id': 'WS-ASIA', 'name': 'ASIA Region Team', 'workspace_type': 'region',
             'status': 'active', 'member_count': 12, 'goal_count': 5, 'active_goal_count': 3,
             'description': 'Regional coordination for ASIA sites', 'tqi': 72},
            {'workspace_id': 'WS-EU', 'name': 'EU Region Team', 'workspace_type': 'region',
             'status': 'active', 'member_count': 15, 'goal_count': 4, 'active_goal_count': 2,
             'description': 'Regional coordination for EU sites', 'tqi': 81},
            {'workspace_id': 'WS-LATAM', 'name': 'LATAM Region Team', 'workspace_type': 'region',
             'status': 'active', 'member_count': 8, 'goal_count': 6, 'active_goal_count': 4,
             'description': 'Regional coordination for LATAM sites', 'tqi': 68},
            {'workspace_id': 'WS-SAFETY', 'name': 'Safety Committee', 'workspace_type': 'committee',
             'status': 'active', 'member_count': 6, 'goal_count': 3, 'active_goal_count': 2,
             'description': 'Cross-functional safety oversight', 'tqi': 85},
            {'workspace_id': 'WS-DM', 'name': 'Data Management Hub', 'workspace_type': 'functional',
             'status': 'active', 'member_count': 10, 'goal_count': 8, 'active_goal_count': 5,
             'description': 'Central DM coordination', 'tqi': 92}
        ]
    
    def _demo_escalations(self) -> List[Dict]:
        return [
            {'escalation_id': 'ESC-001', 'issue_id': 'ISS-2341', 'level': 4, 'status': 'active',
             'title': 'SAE SLA Breach Risk - D-2341', 'created_at': datetime.now() - timedelta(hours=46),
             'sla_remaining_hours': 2, 'assigned_to': 'Study Lead', 'priority': 'critical'},
            {'escalation_id': 'ESC-002', 'issue_id': 'ISS-1892', 'level': 3, 'status': 'active',
             'title': 'Site JP-101 Signature Backlog', 'created_at': datetime.now() - timedelta(days=14),
             'sla_remaining_hours': 48, 'assigned_to': 'CRA Lead', 'priority': 'high'},
            {'escalation_id': 'ESC-003', 'issue_id': 'ISS-3102', 'level': 2, 'status': 'active',
             'title': 'Query Resolution Delay - BR-201', 'created_at': datetime.now() - timedelta(days=7),
             'sla_remaining_hours': 120, 'assigned_to': 'Data Manager', 'priority': 'medium'},
            {'escalation_id': 'ESC-004', 'issue_id': 'ISS-4521', 'level': 5, 'status': 'active',
             'title': 'Critical Protocol Deviation', 'created_at': datetime.now() - timedelta(hours=12),
             'sla_remaining_hours': 0, 'assigned_to': 'Sponsor', 'priority': 'critical'}
        ]
    
    def _demo_alerts(self) -> List[Dict]:
        return [
            {'alert_id': 'ALT-001', 'title': 'You were mentioned in ROOM-001', 'category': 'mention',
             'priority': 'high', 'status': 'unread', 'created_at': datetime.now() - timedelta(hours=1)},
            {'alert_id': 'ALT-002', 'title': 'New escalation assigned to you', 'category': 'escalation',
             'priority': 'critical', 'status': 'unread', 'created_at': datetime.now() - timedelta(hours=2)},
            {'alert_id': 'ALT-003', 'title': 'Goal deadline approaching: DQI Target', 'category': 'goal',
             'priority': 'medium', 'status': 'read', 'created_at': datetime.now() - timedelta(hours=6)},
            {'alert_id': 'ALT-004', 'title': 'Weekly digest: 12 issues resolved', 'category': 'digest',
             'priority': 'low', 'status': 'read', 'created_at': datetime.now() - timedelta(days=1)},
            {'alert_id': 'ALT-005', 'title': 'Pattern detected: Coordinator Overload', 'category': 'pattern',
             'priority': 'high', 'status': 'unread', 'created_at': datetime.now() - timedelta(hours=3)}
        ]
    
    def _demo_issues(self) -> List[Dict]:
        return [
            {'issue_id': 'ISS-2341', 'title': 'SAE Pending Review - Patient 1001', 'category': 'safety',
             'priority': 'critical', 'severity': 'high', 'status': 'open', 'site_id': 'JP-101',
             'assigned_to': 'Safety Lead', 'created_at': datetime.now() - timedelta(days=2), 'age_days': 2},
            {'issue_id': 'ISS-1892', 'title': 'PI Signature Backlog - 23 pending', 'category': 'signature',
             'priority': 'high', 'severity': 'medium', 'status': 'in_progress', 'site_id': 'JP-101',
             'assigned_to': 'Site Coordinator', 'created_at': datetime.now() - timedelta(days=14), 'age_days': 14},
            {'issue_id': 'ISS-3102', 'title': 'Query Volume Spike - BR-201', 'category': 'query',
             'priority': 'medium', 'severity': 'medium', 'status': 'open', 'site_id': 'BR-201',
             'assigned_to': 'Data Manager', 'created_at': datetime.now() - timedelta(days=7), 'age_days': 7},
            {'issue_id': 'ISS-4521', 'title': 'Protocol Deviation - Consent Issue', 'category': 'compliance',
             'priority': 'critical', 'severity': 'critical', 'status': 'escalated', 'site_id': 'EU-044',
             'assigned_to': 'Study Lead', 'created_at': datetime.now() - timedelta(hours=12), 'age_days': 0},
            {'issue_id': 'ISS-5678', 'title': 'Missing Visit Data - 5 patients', 'category': 'data_quality',
             'priority': 'high', 'severity': 'medium', 'status': 'open', 'site_id': 'US-102',
             'assigned_to': 'CRA', 'created_at': datetime.now() - timedelta(days=5), 'age_days': 5}
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        'active': THEME_CONFIG.success,
        'pending_review': THEME_CONFIG.warning,
        'open': THEME_CONFIG.info,
        'in_progress': THEME_CONFIG.warning,
        'escalated': THEME_CONFIG.danger,
        'resolved': THEME_CONFIG.text_muted,
        'closed': THEME_CONFIG.text_muted,
        'unread': THEME_CONFIG.accent,
        'read': THEME_CONFIG.text_secondary
    }
    return colors.get(status, THEME_CONFIG.text_muted)


def get_priority_color(priority: str) -> str:
    """Get color for priority."""
    colors = {
        'critical': THEME_CONFIG.danger,
        'high': THEME_CONFIG.warning,
        'medium': THEME_CONFIG.info,
        'low': THEME_CONFIG.success
    }
    return colors.get(priority.lower(), THEME_CONFIG.text_muted)


def get_room_type_icon(room_type: str) -> str:
    """Get icon for room type."""
    icons = {
        'issue': '🔍', 'pattern': '📊', 'site': '🏥', 'study': '📋',
        'safety': '🛡️', 'audit': '📝', 'ad_hoc': '💬'
    }
    return icons.get(room_type, '📁')


def get_workspace_type_icon(ws_type: str) -> str:
    """Get icon for workspace type."""
    icons = {
        'region': '🌍', 'study': '📋', 'functional': '⚙️',
        'committee': '👥', 'project': '🎯', 'ad_hoc': '💬'
    }
    return icons.get(ws_type, '📁')


def get_escalation_level_color(level: int) -> str:
    """Get color for escalation level."""
    colors = {1: THEME_CONFIG.success, 2: THEME_CONFIG.info, 3: THEME_CONFIG.warning,
              4: THEME_CONFIG.danger, 5: '#ff0000'}
    return colors.get(level, THEME_CONFIG.text_muted)


# =============================================================================
# RENDER COMPONENTS
# =============================================================================

def render_stat_card(label: str, value: str, subtitle: str, color: str):
    """Render a statistics card."""
    theme = THEME_CONFIG
    st.markdown(f"""
        <div style="background: {theme.gradient_card}; border: 1px solid {theme.glass_border};
                    border-radius: {theme.border_radius}; padding: 1.25rem; text-align: center;
                    backdrop-filter: blur(10px);">
            <div style="color: {theme.text_muted}; font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 1px; margin-bottom: 0.5rem;">{label}</div>
            <div style="color: {color}; font-size: 2rem; font-weight: 700; line-height: 1;">{value}</div>
            <div style="color: {theme.text_secondary}; font-size: 0.8rem; margin-top: 0.5rem;">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def render_investigation_rooms_tab(loader: CollaborationDataLoader):
    """Render Investigation Rooms tab."""
    theme = THEME_CONFIG
    rooms = loader.get_investigation_rooms()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h3 style='color: {theme.text_primary}; margin-bottom: 0.5rem;'>🔍 Investigation Rooms</h3>", 
                   unsafe_allow_html=True)
        st.markdown(f"<p style='color: {theme.text_muted};'>Collaborative spaces for issue analysis and root cause identification</p>", 
                   unsafe_allow_html=True)
    with col2:
        if st.button("➕ Create Room", key="create_room", type="primary"):
            st.info("Room creation form would open here")
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    # Filter
    status_filter = st.selectbox("Filter by Status", ["All", "Active", "Pending Review", "Closed"], key="room_status")
    
    # Room cards
    for room in rooms:
        if status_filter != "All" and room.get('status', '').replace('_', ' ').title() != status_filter:
            continue
        
        status = room.get('status', 'unknown')
        status_color = get_status_color(status)
        room_type = room.get('room_type', 'issue')
        icon = get_room_type_icon(room_type)
        
        created = room.get('created_at', datetime.now())
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace('Z', '+00:00'))
        age = (datetime.now() - created).days
        
        st.markdown(f"""
            <div style="background: {theme.gradient_card}; border: 1px solid {theme.glass_border};
                        border-radius: {theme.border_radius}; padding: 1.25rem; margin-bottom: 1rem;
                        border-left: 4px solid {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="font-size: 1.5rem;">{icon}</span>
                            <div>
                                <div style="color: {theme.text_primary}; font-weight: 600; font-size: 1.1rem;">
                                    {room.get('title', 'Untitled Room')}</div>
                                <div style="color: {theme.text_muted}; font-size: 0.85rem;">
                                    {room.get('room_id', '')} • {room_type.title()} • {age}d ago</div>
                            </div>
                        </div>
                        <p style="color: {theme.text_secondary}; margin: 0.75rem 0 0 2.25rem; font-size: 0.9rem;">
                            {room.get('description', '')}</p>
                    </div>
                    <div style="background: {status_color}20; color: {status_color}; padding: 0.3rem 0.75rem;
                                border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">
                        {status.replace('_', ' ')}</div>
                </div>
                <div style="display: flex; gap: 2rem; margin-top: 1rem; padding-top: 1rem; 
                            border-top: 1px solid {theme.glass_border};">
                    <div style="color: {theme.text_secondary}; font-size: 0.85rem;">
                        👥 {room.get('participant_count', 0)} participants</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.85rem;">
                        📎 {room.get('evidence_count', 0)} evidence</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.85rem;">
                        💬 {room.get('thread_count', 0)} threads</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("Enter Room", key=f"enter_{room.get('room_id')}"):
                st.session_state['selected_room'] = room.get('room_id')
        with col_b:
            if st.button("View Timeline", key=f"timeline_{room.get('room_id')}"):
                st.info(f"Timeline for {room.get('room_id')}")


def render_team_workspaces_tab(loader: CollaborationDataLoader):
    """Render Team Workspaces tab."""
    theme = THEME_CONFIG
    workspaces = loader.get_workspaces()
    
    st.markdown(f"<h3 style='color: {theme.text_primary};'>🏢 Team Workspaces</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {theme.text_muted};'>Regional and functional team collaboration spaces</p>", 
               unsafe_allow_html=True)
    
    # Workspace type filter
    ws_types = list(set(w.get('workspace_type', 'other') for w in workspaces))
    type_filter = st.selectbox("Filter by Type", ["All"] + [t.title() for t in ws_types], key="ws_type")
    
    for ws in workspaces:
        if type_filter != "All" and ws.get('workspace_type', '').title() != type_filter:
            continue
        
        ws_type = ws.get('workspace_type', 'other')
        icon = get_workspace_type_icon(ws_type)
        tqi = ws.get('tqi', 80)
        tqi_color = THEME_CONFIG.success if tqi >= 80 else THEME_CONFIG.warning if tqi >= 70 else THEME_CONFIG.danger
        
        st.markdown(f"""
            <div style="background: {theme.gradient_card}; border: 1px solid {theme.glass_border};
                        border-radius: {theme.border_radius}; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.5rem;">{icon}</span>
                        <div>
                            <div style="color: {theme.text_primary}; font-weight: 600; font-size: 1.1rem;">
                                {ws.get('name', 'Untitled')}</div>
                            <div style="color: {theme.text_muted}; font-size: 0.85rem;">{ws_type.title()}</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {tqi_color}; font-size: 1.5rem; font-weight: 700;">TQI {tqi}</div>
                        <div style="color: {theme.text_muted}; font-size: 0.75rem;">Regional Score</div>
                    </div>
                </div>
                <p style="color: {theme.text_secondary}; margin: 0.75rem 0; font-size: 0.9rem;">
                    {ws.get('description', '')}</p>
                <div style="display: flex; gap: 1.5rem; padding-top: 0.75rem; border-top: 1px solid {theme.glass_border};">
                    <div style="color: {theme.text_secondary}; font-size: 0.85rem;">
                        👥 {ws.get('member_count', 0)} members</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.85rem;">
                        🎯 {ws.get('active_goal_count', 0)}/{ws.get('goal_count', 0)} goals active</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Workspace", key=f"open_ws_{ws.get('workspace_id')}"):
            st.session_state['selected_workspace'] = ws.get('workspace_id')


def render_escalation_tab(loader: CollaborationDataLoader):
    """Render Escalation Pipeline tab."""
    theme = THEME_CONFIG
    escalations = loader.get_escalations()
    
    st.markdown(f"<h3 style='color: {theme.text_primary};'>⚡ Escalation Pipeline</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {theme.text_muted};'>5-Level escalation tracking with SLA monitoring</p>", 
               unsafe_allow_html=True)
    
    # Level indicators
    st.markdown(f"""
        <div style="display: flex; gap: 0.5rem; margin: 1rem 0;">
            <div style="flex: 1; text-align: center; padding: 0.5rem; background: {THEME_CONFIG.success}20; 
                        border-radius: 8px; border: 1px solid {THEME_CONFIG.success}40;">
                <div style="color: {THEME_CONFIG.success}; font-weight: 700;">L1</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">Auto-remind</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 0.5rem; background: {THEME_CONFIG.info}20; 
                        border-radius: 8px; border: 1px solid {THEME_CONFIG.info}40;">
                <div style="color: {THEME_CONFIG.info}; font-weight: 700;">L2</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">CRA Alert</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 0.5rem; background: {THEME_CONFIG.warning}20; 
                        border-radius: 8px; border: 1px solid {THEME_CONFIG.warning}40;">
                <div style="color: {THEME_CONFIG.warning}; font-weight: 700;">L3</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">Investigation</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 0.5rem; background: {THEME_CONFIG.danger}20; 
                        border-radius: 8px; border: 1px solid {THEME_CONFIG.danger}40;">
                <div style="color: {THEME_CONFIG.danger}; font-weight: 700;">L4</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">Study Lead</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 0.5rem; background: #ff000020; 
                        border-radius: 8px; border: 1px solid #ff000040;">
                <div style="color: #ff0000; font-weight: 700;">L5</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">Sponsor</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Escalation cards
    for esc in sorted(escalations, key=lambda x: x.get('level', 1), reverse=True):
        level = esc.get('level', 1)
        level_color = get_escalation_level_color(level)
        sla_hours = esc.get('sla_remaining_hours', 0)
        sla_status = "🔴 BREACHED" if sla_hours <= 0 else f"⏱️ {sla_hours}h remaining"
        
        st.markdown(f"""
            <div style="background: {theme.gradient_card}; border: 1px solid {level_color}40;
                        border-radius: {theme.border_radius}; padding: 1.25rem; margin-bottom: 1rem;
                        border-left: 4px solid {level_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="background: {level_color}; color: white; padding: 0.2rem 0.5rem;
                                        border-radius: 4px; font-weight: 700; font-size: 0.85rem;">L{level}</span>
                            <span style="color: {theme.text_primary}; font-weight: 600;">
                                {esc.get('title', 'Untitled')}</span>
                        </div>
                        <div style="color: {theme.text_muted}; font-size: 0.85rem; margin-top: 0.5rem;">
                            {esc.get('escalation_id', '')} • Assigned to: {esc.get('assigned_to', 'Unassigned')}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {'#ff0000' if sla_hours <= 0 else theme.warning}; font-weight: 600;">
                            {sla_status}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_alerts_tab(loader: CollaborationDataLoader):
    """Render Alert Center tab."""
    theme = THEME_CONFIG
    alerts = loader.get_alerts()
    
    st.markdown(f"<h3 style='color: {theme.text_primary};'>🔔 Alert Center</h3>", unsafe_allow_html=True)
    
    unread = len([a for a in alerts if a.get('status') == 'unread'])
    st.markdown(f"<p style='color: {theme.text_muted};'>{unread} unread notifications</p>", unsafe_allow_html=True)
    
    # Mark all read button
    if unread > 0:
        if st.button("Mark All as Read", key="mark_all_read"):
            st.success("All notifications marked as read")
    
    for alert in alerts:
        status = alert.get('status', 'read')
        priority = alert.get('priority', 'low')
        priority_color = get_priority_color(priority)
        is_unread = status == 'unread'
        
        bg_opacity = '0.8' if is_unread else '0.4'
        
        st.markdown(f"""
            <div style="background: rgba(30, 41, 59, {bg_opacity}); border: 1px solid {theme.glass_border};
                        border-radius: {theme.border_radius}; padding: 1rem; margin-bottom: 0.75rem;
                        border-left: 3px solid {priority_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        {'🔵' if is_unread else '⚪'}
                        <div>
                            <div style="color: {theme.text_primary}; font-weight: {'600' if is_unread else '400'};">
                                {alert.get('title', 'Notification')}</div>
                            <div style="color: {theme.text_muted}; font-size: 0.8rem;">
                                {alert.get('category', '').title()} • {alert.get('created_at', datetime.now()).strftime('%b %d, %H:%M') if isinstance(alert.get('created_at'), datetime) else 'Recently'}</div>
                        </div>
                    </div>
                    <span style="background: {priority_color}20; color: {priority_color}; padding: 0.2rem 0.5rem;
                                border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">
                        {priority}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_issues_tab(loader: CollaborationDataLoader):
    """Render Issue Registry tab."""
    theme = THEME_CONFIG
    issues = loader.get_issues()
    
    st.markdown(f"<h3 style='color: {theme.text_primary};'>📋 Issue Registry</h3>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.selectbox("Priority", ["All", "Critical", "High", "Medium", "Low"], key="issue_priority")
    with col2:
        status_filter = st.selectbox("Status", ["All", "Open", "In Progress", "Escalated", "Resolved"], key="issue_status")
    with col3:
        category_filter = st.selectbox("Category", ["All", "Safety", "Query", "Signature", "Compliance", "Data Quality"], key="issue_cat")
    
    # Issues table
    for issue in issues:
        if priority_filter != "All" and issue.get('priority', '').lower() != priority_filter.lower():
            continue
        if status_filter != "All" and issue.get('status', '').replace('_', ' ').title() != status_filter:
            continue
        
        priority = issue.get('priority', 'medium')
        priority_color = get_priority_color(priority)
        status_color = get_status_color(issue.get('status', 'open'))
        
        st.markdown(f"""
            <div style="background: {theme.gradient_card}; border: 1px solid {theme.glass_border};
                        border-radius: {theme.border_radius}; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="background: {priority_color}20; color: {priority_color}; 
                                        padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; 
                                        font-weight: 600; text-transform: uppercase;">{priority}</span>
                            <span style="color: {theme.text_primary}; font-weight: 600;">
                                {issue.get('title', 'Untitled Issue')}</span>
                        </div>
                        <div style="color: {theme.text_muted}; font-size: 0.85rem; margin-top: 0.4rem;">
                            {issue.get('issue_id', '')} • {issue.get('site_id', '')} • 
                            {issue.get('category', '').replace('_', ' ').title()} • {issue.get('age_days', 0)}d old</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {status_color}20; color: {status_color}; 
                                    padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                            {issue.get('status', '').replace('_', ' ').title()}</span>
                        <div style="color: {theme.text_muted}; font-size: 0.8rem; margin-top: 0.4rem;">
                            → {issue.get('assigned_to', 'Unassigned')}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_tagging_tab():
    """Render @Tagging System tab."""
    theme = THEME_CONFIG
    
    st.markdown(f"<h3 style='color: {theme.text_primary};'>🏷️ @Tagging System</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {theme.text_muted};'>Mention people, entities, and topics across the platform</p>", 
               unsafe_allow_html=True)
    
    # Tag input demo
    st.markdown(f"""
        <div style="background: {theme.gradient_card}; border: 1px solid {theme.glass_border};
                    border-radius: {theme.border_radius}; padding: 1.5rem; margin: 1rem 0;">
            <div style="color: {theme.text_primary}; font-weight: 600; margin-bottom: 1rem;">Tag Reference</div>
            <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
                <div>
                    <div style="color: {theme.accent}; font-weight: 600;">@People</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.9rem;">@Sarah, @DM-Team</div>
                </div>
                <div>
                    <div style="color: {theme.info}; font-weight: 600;">@Entities</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.9rem;">@Site-JP101, @Subject-1001</div>
                </div>
                <div>
                    <div style="color: {theme.warning}; font-weight: 600;">#Topics</div>
                    <div style="color: {theme.text_secondary}; font-size: 0.9rem;">#consent, #urgent</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Test input
    test_input = st.text_area("Try tagging (use @ for mentions, # for topics)", 
                              value="@Sarah please review @Site-JP101 for #consent issues #urgent",
                              key="tag_input")
    
    if test_input:
        mentions = [w for w in test_input.split() if w.startswith('@')]
        topics = [w for w in test_input.split() if w.startswith('#')]
        
        if mentions or topics:
            st.markdown(f"""
                <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid {theme.accent}40;
                            border-radius: {theme.border_radius}; padding: 1rem; margin-top: 1rem;">
                    <div style="color: {theme.text_primary}; font-weight: 600; margin-bottom: 0.5rem;">
                        Detected Tags:</div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        {''.join([f'<span style="background: {theme.accent}30; color: {theme.accent}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem;">{m}</span>' for m in mentions])}
                        {''.join([f'<span style="background: {theme.warning}30; color: {theme.warning}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem;">{t}</span>' for t in topics])}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_page(user: Optional[Dict] = None):
    """Main render function for Collaboration Hub."""
    theme = THEME_CONFIG
    loader = CollaborationDataLoader()
    
    # Page header
    st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="background: {theme.gradient_accent}; -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; font-size: 2rem; margin-bottom: 0.25rem;">
                🤝 Collaboration Hub</h1>
            <p style="color: {theme.text_secondary}; font-size: 1rem;">
                Investigation Rooms • Team Workspaces • @Tagging • Escalation • Alerts • Issues</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Statistics cards
    stats = loader.get_collaboration_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Active Rooms", str(stats['active_rooms']), 
                        f"{stats['total_rooms']} total", theme.accent)
    with col2:
        render_stat_card("Workspaces", str(stats['active_workspaces']),
                        f"{stats['total_workspaces']} total", theme.info)
    with col3:
        render_stat_card("Escalations", str(stats['pending_escalations']),
                        f"{stats['total_escalations']} total", theme.warning)
    with col4:
        render_stat_card("Open Issues", str(stats['open_issues']),
                        f"{stats['total_issues']} total", theme.danger)
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # Tab navigation
    tabs = st.tabs(["🔍 Investigation Rooms", "🏢 Team Workspaces", "🏷️ @Tagging", 
                    "⚡ Escalation", "🔔 Alerts", "📋 Issues"])
    
    with tabs[0]:
        render_investigation_rooms_tab(loader)
    with tabs[1]:
        render_team_workspaces_tab(loader)
    with tabs[2]:
        render_tagging_tab()
    with tabs[3]:
        render_escalation_tab(loader)
    with tabs[4]:
        render_alerts_tab(loader)
    with tabs[5]:
        render_issues_tab(loader)


# Test entrypoint
if __name__ == "__main__":
    render_page()
