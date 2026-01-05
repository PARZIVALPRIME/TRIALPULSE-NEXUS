"""
TRIALPULSE NEXUS 10X - Sidebar Component
Phase 7.1: Dark theme navigation sidebar with role-based access
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any

from dashboard.config.auth import logout, can_access_page
from dashboard.config.session import set_page, get_page
from dashboard.config.theme import THEME_CONFIG


# Page definitions with icons
PAGES = {
    "Executive Overview": {"icon": "📊", "description": "Portfolio-level KPIs and insights"},
    "CRA Field View": {"icon": "👩‍⚕️", "description": "Site monitoring and action queue"},
    "Data Manager Hub": {"icon": "💻", "description": "Data quality and query management"},
    "Safety Surveillance": {"icon": "🛡️", "description": "SAE tracking and safety signals"},
    "Study Lead Command": {"icon": "🎯", "description": "Study oversight and decisions"},
    "Site Portal": {"icon": "🏥", "description": "Site-specific actions and status"},
    "Coder Workbench": {"icon": "🔤", "description": "Medical coding queue"},
    "Collaboration Hub": {"icon": "🤝", "description": "Investigation rooms and team collaboration"},
    "Cascade Explorer": {"icon": "🌊", "description": "Issue dependency visualization"},
    "AI Assistant": {"icon": "🤖", "description": "Natural language queries"},
    "Reports": {"icon": "📄", "description": "Generate and download reports"},
    "Settings": {"icon": "⚙️", "description": "User preferences and configuration"}
}


def render_sidebar(user: Dict[str, Any]):
    """Render the navigation sidebar with dark theme."""
    
    theme = THEME_CONFIG
    
    with st.sidebar:
        # Logo and title with Glow Effect
        st.markdown(f"""
            <div style="text-align: center; padding: 2rem 0; position: relative;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px {theme.accent}); animation: pulse 3s infinite;">🧬</div>
                <div style="background: {theme.gradient_accent}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.5rem; font-weight: 800; letter-spacing: 1px;">TRIALPULSE</div>
                <div style="color: {theme.text_secondary}; font-size: 0.7rem; letter-spacing: 4px; font-weight: 500; margin-top: 0.2rem;">NEXUS 10X</div>
            </div>
            <div style="height: 1px; background: linear-gradient(90deg, transparent, {theme.text_muted}, transparent); opacity: 0.2; margin-bottom: 2rem;"></div>
        """, unsafe_allow_html=True)
        
        # User info
        render_user_info(user)
        
        # Navigation
        render_navigation(user)
        
        # Spacer
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats
        render_quick_stats()
        
        st.markdown(f"<div style='height: 1px; background: {theme.glass_border}; margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            logout()
            st.rerun()
        
        # Version info
        st.markdown(
            f"<div style='text-align: center; color: {theme.text_muted}; font-size: 0.65rem; margin-top: 1rem; opacity: 0.6; font-family: monospace;'>"
            f"SYSTEM ONLINE v10.0.0</div>",
            unsafe_allow_html=True
        )


def render_user_info(user: Dict[str, Any]):
    """Render user information section."""
    
    theme = THEME_CONFIG
    avatar = user.get('avatar', '👤')
    name = user.get('name', 'User')
    role = user.get('role', 'Unknown')
    
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {theme.glass_border}; border-radius: 16px; padding: 1rem; display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
            <div style="font-size: 1.8rem; background: {theme.secondary_bg}; width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">{avatar}</div>
            <div>
                <div style="color: {theme.text_primary}; font-weight: 700; font-size: 0.95rem;">{name}</div>
                <div style="color: {theme.accent_light}; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">{role}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_navigation(user: Dict[str, Any]):
    """Render navigation menu."""
    
    theme = THEME_CONFIG
    current_page = get_page()
    
    st.markdown(f"<div style='color: {theme.text_muted}; font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 0.75rem; text-transform: uppercase; padding-left: 0.5rem;'>Main Modules</div>", unsafe_allow_html=True)
    
    for page_name, page_info in PAGES.items():
        if not can_access_page(page_name):
            continue
        
        icon = page_info["icon"]
        is_current = page_name == current_page
        
        button_label = f"{icon} {page_name}"
        
        if is_current:
            # Active State - Neon Glow
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, rgba(99, 102, 241, 0.15), transparent); 
                            border-left: 3px solid {theme.accent};
                            padding: 0.75rem 1rem; 
                            margin: 0.25rem 0;
                            border-radius: 0 12px 12px 0;
                            color: {theme.text_primary}; 
                            font-weight: 600;
                            font-size: 0.9rem;
                            display: flex; align-items: center; gap: 0.75rem;
                            box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);">
                    <span style="filter: drop-shadow(0 0 5px {theme.accent});">{icon}</span> {page_name}
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(button_label, key=f"nav_{page_name}", use_container_width=True):
                set_page(page_name)
                st.rerun()


def render_quick_stats():
    """Render quick statistics section."""
    
    theme = THEME_CONFIG
    
    st.markdown(f"<div style='color: {theme.text_muted}; font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 1rem; text-transform: uppercase; padding-left: 0.5rem;'>Live Metrics</div>", unsafe_allow_html=True)
    
    # Stats with custom HTML for tighter control over look
    # Using st.columns for layout but HTML for content
    col1, col2 = st.columns(2)
    
    with col1:
        render_mini_stat("Patients", "58k", "+12%", theme.success)
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        render_mini_stat("DQI", "98.2", "Elite", theme.info)
    
    with col2:
        render_mini_stat("Sites", "3.4k", "Active", theme.text_secondary)
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        render_mini_stat("Clean", "54%", "↑", theme.warning)

def render_mini_stat(label, value, tag, color):
    theme = THEME_CONFIG
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid {theme.glass_border}; border-radius: 10px; padding: 0.6rem; text-align: center;">
            <div style="color: {theme.text_muted}; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="color: {theme.text_primary}; font-weight: 700; font-size: 1.1rem; margin: 0.1rem 0;">{value}</div>
            <div style="color: {color}; font-size: 0.6rem; font-weight: 600; background: {color}20; display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px;">{tag}</div>
        </div>
    """, unsafe_allow_html=True)
