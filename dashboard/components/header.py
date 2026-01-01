"""
TRIALPULSE NEXUS 10X - Header Component
Phase 7.1: Dark theme page header with gradient effects
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

from dashboard.config.session import get_page, get_filter
from dashboard.config.theme import THEME_CONFIG


def render_header(user: Dict[str, Any], title: Optional[str] = None, 
                  subtitle: Optional[str] = None, show_filters: bool = True):
    """Render the page header with Neo-Nexus styling."""
    
    theme = THEME_CONFIG
    current_page = get_page()
    page_title = title or current_page
    page_icon = get_page_icon(current_page)
    
    # Neo-Header Container
    st.markdown(f"""
    <div style="background: {theme.gradient_header}; padding: 1.5rem 0; border-bottom: 1px solid {theme.glass_border}; margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2.5rem; filter: drop-shadow(0 0 10px {theme.accent}60);">{page_icon}</div>
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(90deg, #fff, {theme.text_secondary}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px;">{page_title}</h1>
                    <p style="margin: 0; color: {theme.accent}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; letter-spacing: 0.5px; opacity: 0.9;">
                        // {subtitle or get_page_subtitle(current_page)}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if show_filters:
            render_quick_filters()
    
    with col3:
        render_header_actions(user)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Technical Breadcrumb
    render_breadcrumb(current_page)


def get_page_icon(page_name: str) -> str:
    icons = {
        "Executive Overview": "📊", "CRA Field View": "👩‍⚕️", "Data Manager Hub": "💻",
        "Safety Surveillance": "🛡️", "Study Lead Command": "🎯", "Site Portal": "🏥",
        "Coder Workbench": "🔤", "Cascade Explorer": "🌊", "AI Assistant": "🤖",
        "Reports": "📄", "Settings": "⚙️"
    }
    return icons.get(page_name, "📋")


def get_page_subtitle(page_name: str) -> str:
    subtitles = {
        "Executive Overview": "PORTFOLIO_NEXUS_V1",
        "CRA Field View": "SITE_MONITORING_PROTOCOL",
        "Data Manager Hub": "DATA_INTEGRITY_CORE",
        "Safety Surveillance": "PHARMACOVIGILANCE_MODULE",
        "Study Lead Command": "STRATEGIC_COMMAND",
        "Site Portal": "SITE_INTERFACE",
        "Coder Workbench": "MED_CODING_ENGINE",
        "Cascade Explorer": "DEPENDENCY_VISUALIZER",
        "AI Assistant": "NEURAL_INTERFACE_V7",
        "Reports": "GENERATION_CENTER",
        "Settings": "SYSTEM_CONFIG"
    }
    return subtitles.get(page_name, "SYSTEM_MODULE")


def render_quick_filters():
    theme = THEME_CONFIG
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Context Scope", ["Global / All Studies", "Study_21 (Oncology)", "Study_23 (Cardio)"], key="header_study_filter", label_visibility="collapsed")
    with col2:
        if st.button("🔄 SYNCHRONIZE", key="header_refresh", use_container_width=True):
            st.rerun()


def render_header_actions(user: Dict[str, Any]):
    from dashboard.config.session import set_page
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔔 3", key="notif_btn", help="System Alerts"):
            st.toast("📢 3 new notifications: 2 critical sites, 1 SAE pending review", icon="🔔")
    with col2:
        if st.button("🤖 AI", key="ai_btn", help="Open AI Assistant"):
            set_page("AI Assistant")
            st.rerun()
    with col3:
        if st.button(user.get('avatar', '👤'), key="user_btn", help=f"Logged in as {user.get('name', 'User')}"):
            st.toast(f"👤 {user.get('name', 'User')} ({user.get('role', 'User')})\n\nClick Logout in sidebar to sign out.", icon="👤")


def render_breadcrumb(current_page: str):
    theme = THEME_CONFIG
    path = f"ROOT / NEXUS_OS / {current_page.upper().replace(' ', '_')}"
    st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {theme.text_muted}; margin-top: -1.5rem; margin-bottom: 2rem; padding-left: 0.5rem; border-left: 2px solid {theme.accent};">
            <span style="color: {theme.accent};">current_path:</span> {path}
        </div>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "", actions: list = None, icon: str = ""):
    theme = THEME_CONFIG
    st.markdown(f"""
        <div style="border-bottom: 1px solid {theme.glass_border}; padding-bottom: 1rem; margin-bottom: 2rem;">
            <h2 style="color: {theme.text_primary}; margin: 0; display: flex; align-items: center; gap: 0.5rem;">
                {icon} <span style="background: {theme.gradient_accent}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{title}</span>
            </h2>
            {f'<div style="color: {theme.text_secondary}; font-family: monospace; font-size: 0.85rem; margin-top: 0.25rem;">// {subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)
    
    if actions:
        cols = st.columns(len(actions))
        for i, a in enumerate(actions):
            with cols[i]:
                st.button(a['label'], key=a.get('key'))

def render_section_header(title: str, icon: str = "", action_label: str = None, action_key: str = None):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""<h3 style="display: flex; align-items: center; gap: 0.5rem;">{icon} {title}</h3>""", unsafe_allow_html=True)
    with col2:
        if action_label: st.button(action_label, key=action_key)