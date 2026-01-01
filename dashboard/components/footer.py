"""
TRIALPULSE NEXUS 10X - Footer Component
Phase 7.1: Dark theme page footer with status and links
"""

import streamlit as st
from datetime import datetime

from dashboard.config.theme import THEME_CONFIG


def render_footer():
    """Render the page footer with dark theme styling."""
    
    theme = THEME_CONFIG
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_refresh = st.session_state.get('last_data_refresh', 'Never')
    
    if last_refresh != 'Never':
        try:
            last_refresh = datetime.fromisoformat(last_refresh).strftime("%H:%M:%S")
        except:
            pass
    
    # Simple separator
    st.markdown("---")
    
    # Footer using columns for proper Streamlit rendering
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption(f"🕐 {current_time}")
    
    with col2:
        st.caption(f"🔄 Last refresh: {last_refresh}")
    
    with col3:
        st.caption("🟢 System Online")
    
    with col4:
        st.caption("📚 Docs | ❓ Help | 🎧 Support")
    
    # Copyright
    st.markdown(
        f"<div style='text-align: center; padding: 0.5rem 0; color: {theme.text_muted}; font-size: 0.7rem;'>"
        f"© 2026 TrialPulse Nexus 10X | AI-Powered Clinical Trial Intelligence"
        f"</div>",
        unsafe_allow_html=True
    )


def render_mini_footer():
    """Render a minimal footer for modal/popup contexts."""
    
    theme = THEME_CONFIG
    
    st.markdown(
        f"<div style='text-align: center; padding: 0.75rem 0; margin-top: 1rem; "
        f"border-top: 1px solid {theme.border_color};'>"
        f"<span style='font-size: 0.7rem; color: {theme.text_muted};'>TrialPulse Nexus 10X</span>"
        f"</div>",
        unsafe_allow_html=True
    )