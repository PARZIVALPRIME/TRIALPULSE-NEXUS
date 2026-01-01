"""
TRIALPULSE NEXUS 10X - Settings Page
Phase 7.1: User preferences and system configuration
"""

import streamlit as st
from typing import Dict, Any
from datetime import datetime

from dashboard.config.session import (
    get_preference, 
    set_preference, 
    get_session_stats,
    clear_chat_history,
    mark_all_notifications_read
)
from dashboard.config.auth import get_current_user


def render_page(user: Dict[str, Any]):
    """Render the Settings page."""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <h2 style="margin: 0;">⚙️ Settings</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
            Configure your preferences and system settings
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile", 
        "🎨 Display", 
        "🔔 Notifications", 
        "📊 Session"
    ])
    
    with tab1:
        render_profile_settings(user)
    
    with tab2:
        render_display_settings()
    
    with tab3:
        render_notification_settings()
    
    with tab4:
        render_session_info()


def render_profile_settings(user: Dict[str, Any]):
    """Render profile settings section."""
    
    st.markdown("### Profile Information")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Avatar display
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; 
                    border-radius: 10px; margin-bottom: 1rem;">
            <div style="font-size: 5rem;">{user.get('avatar', '👤')}</div>
            <p style="color: #7f8c8d; margin-top: 0.5rem;">Profile Avatar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # User info (read-only for demo)
        st.text_input("Name", value=user.get('name', ''), disabled=True)
        st.text_input("Email", value=user.get('email', ''), disabled=True)
        st.text_input("Role", value=user.get('role', ''), disabled=True)
        
        # Studies access
        studies = user.get('studies', [])
        studies_display = ", ".join(studies) if studies else "None"
        st.text_input("Studies Access", value=studies_display, disabled=True)
        
        # Sites access (if applicable)
        sites = user.get('sites', [])
        if sites:
            sites_display = ", ".join(sites[:5])
            if len(sites) > 5:
                sites_display += f" (+{len(sites) - 5} more)"
            st.text_input("Sites Access", value=sites_display, disabled=True)
    
    st.markdown("---")
    
    # Password change (placeholder)
    st.markdown("### Change Password")
    st.info("🔒 Password change is disabled in demo mode.")
    
    with st.expander("Change Password Form"):
        current_pw = st.text_input("Current Password", type="password", key="current_pw")
        new_pw = st.text_input("New Password", type="password", key="new_pw")
        confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")
        
        if st.button("Update Password", disabled=True):
            st.warning("Password change is disabled in demo mode.")


def render_display_settings():
    """Render display settings section."""
    
    st.markdown("### Display Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Default view
        default_view = get_preference('default_view', 'cards')
        new_view = st.selectbox(
            "Default View Style",
            options=['cards', 'table', 'compact'],
            index=['cards', 'table', 'compact'].index(default_view),
            help="Choose how data is displayed by default"
        )
        if new_view != default_view:
            set_preference('default_view', new_view)
        
        # Items per page
        items_per_page = get_preference('items_per_page', 25)
        new_items = st.selectbox(
            "Items Per Page",
            options=[10, 25, 50, 100],
            index=[10, 25, 50, 100].index(items_per_page),
            help="Number of items to display in tables"
        )
        if new_items != items_per_page:
            set_preference('items_per_page', new_items)
    
    with col2:
        # Compact mode
        compact_mode = get_preference('compact_mode', False)
        new_compact = st.checkbox(
            "Compact Mode",
            value=compact_mode,
            help="Use a more compact layout with smaller elements"
        )
        if new_compact != compact_mode:
            set_preference('compact_mode', new_compact)
        
        # Show tooltips
        show_tooltips = get_preference('show_tooltips', True)
        new_tooltips = st.checkbox(
            "Show Tooltips",
            value=show_tooltips,
            help="Display helpful tooltips on hover"
        )
        if new_tooltips != show_tooltips:
            set_preference('show_tooltips', new_tooltips)
        
        # Dark mode (placeholder)
        dark_mode = st.checkbox(
            "Dark Mode",
            value=False,
            disabled=True,
            help="Dark mode coming soon"
        )
    
    st.markdown("---")
    
    # Chart preferences
    st.markdown("### Chart Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_animation = st.checkbox(
            "Enable Chart Animations",
            value=get_preference('chart_animation', True),
            help="Animate charts when they load"
        )
        set_preference('chart_animation', chart_animation)
    
    with col2:
        chart_labels = st.checkbox(
            "Show Data Labels",
            value=get_preference('chart_labels', True),
            help="Display values on charts"
        )
        set_preference('chart_labels', chart_labels)


def render_notification_settings():
    """Render notification settings section."""
    
    st.markdown("### Notification Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Email notifications
        email_notif = st.checkbox(
            "Email Notifications",
            value=get_preference('email_notifications', True),
            help="Receive notifications via email"
        )
        set_preference('email_notifications', email_notif)
        
        if email_notif:
            st.markdown("**Email notification types:**")
            
            critical_email = st.checkbox(
                "Critical Alerts",
                value=get_preference('critical_email', True),
                help="Receive emails for critical issues"
            )
            set_preference('critical_email', critical_email)
            
            daily_digest = st.checkbox(
                "Daily Digest",
                value=get_preference('daily_digest', True),
                help="Receive daily summary email"
            )
            set_preference('daily_digest', daily_digest)
            
            report_ready = st.checkbox(
                "Report Ready",
                value=get_preference('report_ready', True),
                help="Notify when reports are generated"
            )
            set_preference('report_ready', report_ready)
    
    with col2:
        # In-app notifications
        in_app_notif = st.checkbox(
            "In-App Notifications",
            value=get_preference('in_app_notifications', True),
            help="Show notifications in the app"
        )
        set_preference('in_app_notifications', in_app_notif)
        
        if in_app_notif:
            st.markdown("**In-app notification types:**")
            
            sound_alerts = st.checkbox(
                "Sound Alerts",
                value=get_preference('sound_alerts', False),
                help="Play sound for notifications"
            )
            set_preference('sound_alerts', sound_alerts)
            
            desktop_notif = st.checkbox(
                "Desktop Notifications",
                value=get_preference('desktop_notifications', False),
                help="Show desktop notifications"
            )
            set_preference('desktop_notifications', desktop_notif)
    
    st.markdown("---")
    
    # Auto-refresh settings
    st.markdown("### Data Refresh")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_refresh = st.checkbox(
            "Auto-Refresh Data",
            value=get_preference('auto_refresh', True),
            help="Automatically refresh data at intervals"
        )
        set_preference('auto_refresh', auto_refresh)
    
    with col2:
        if auto_refresh:
            refresh_interval = get_preference('refresh_interval', 300)
            new_interval = st.selectbox(
                "Refresh Interval",
                options=[60, 120, 300, 600, 900],
                format_func=lambda x: f"{x // 60} minutes",
                index=[60, 120, 300, 600, 900].index(refresh_interval)
            )
            set_preference('refresh_interval', new_interval)
    
    st.markdown("---")
    
    # Clear notifications
    st.markdown("### Manage Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Mark All as Read", use_container_width=True):
            mark_all_notifications_read()
            st.success("✅ All notifications marked as read")
    
    with col2:
        if st.button("Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.success("✅ Chat history cleared")


def render_session_info():
    """Render session information section."""
    
    st.markdown("### Current Session")
    
    stats = get_session_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Session Duration",
            format_duration(stats.get('duration_seconds', 0))
        )
    
    with col2:
        st.metric(
            "Page Views",
            stats.get('page_views', 0)
        )
    
    with col3:
        st.metric(
            "Actions Taken",
            stats.get('actions_taken', 0)
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Queries Made",
            stats.get('queries_made', 0)
        )
    
    with col2:
        st.metric(
            "Notifications",
            stats.get('notifications', 0)
        )
    
    with col3:
        st.metric(
            "Chat Messages",
            stats.get('chat_messages', 0)
        )
    
    st.markdown("---")
    
    # Session details
    st.markdown("### Session Details")
    
    user = get_current_user()
    
    session_info = {
        "Session Start": stats.get('session_start', 'Unknown'),
        "User": user.get('name', 'Unknown') if user else 'Unknown',
        "Role": user.get('role', 'Unknown') if user else 'Unknown',
        "Login Time": user.get('login_time', 'Unknown') if user else 'Unknown',
        "Cached Items": stats.get('cached_items', 0),
        "Unread Notifications": stats.get('unread_notifications', 0)
    }
    
    for key, value in session_info.items():
        st.text(f"{key}: {value}")
    
    st.markdown("---")
    
    # System info
    st.markdown("### System Information")
    
    system_info = {
        "Version": "10.0.0",
        "Phase": "7 - Dashboard",
        "Environment": "Development",
        "Data Source": "Local Parquet Files",
        "LLM": "Ollama (llama3.1:8b)",
        "Last Data Refresh": st.session_state.get('last_data_refresh', 'Never')
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    st.markdown("---")
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_data_refresh = datetime.now().isoformat()
            st.session_state.data_loaded = False
            st.success("✅ Data refresh initiated")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.cached_data = {}
            st.success("✅ Cache cleared")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None or seconds < 0:
        return "0m"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"