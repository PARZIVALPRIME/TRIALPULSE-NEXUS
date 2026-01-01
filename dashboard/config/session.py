"""
TRIALPULSE NEXUS 10X - Session State Management
Phase 7.1: Session state initialization and management
"""

import streamlit as st
from datetime import datetime
from typing import Any, Optional, Dict, List

# Default session state values
DEFAULT_SESSION_STATE = {
    # Authentication
    'authenticated': False,
    'username': None,
    'user': None,
    'session_token': None,
    
    # Navigation
    'current_page': 'Executive Overview',
    'allowed_pages': [],
    'page_history': [],
    
    # Filters
    'selected_study': 'All Studies',
    'selected_site': 'All Sites',
    'selected_region': 'All Regions',
    'selected_status': 'All Statuses',
    'date_range': None,
    
    # UI State
    'sidebar_expanded': True,
    'show_ai_assistant': False,
    'dark_mode': False,
    
    # Data Cache
    'data_loaded': False,
    'last_data_refresh': None,
    'cached_data': {},
    
    # AI Assistant
    'chat_history': [],
    'ai_context': {},
    
    # Notifications
    'notifications': [],
    'unread_notifications': 0,
    
    # User Preferences
    'preferences': {
        'default_view': 'cards',
        'items_per_page': 25,
        'auto_refresh': True,
        'refresh_interval': 300,  # seconds
        'show_tooltips': True,
        'compact_mode': False
    },
    
    # Session Metrics
    'session_start': None,
    'page_views': 0,
    'actions_taken': 0,
    'queries_made': 0,
    
    # Temporary State
    'temp': {}
}


def initialize_session():
    """Initialize session state with default values."""
    
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Set session start time if not set
    if st.session_state.session_start is None:
        st.session_state.session_start = datetime.now().isoformat()


def reset_session():
    """Reset session state to defaults (except authentication)."""
    
    # Save auth state
    auth_keys = ['authenticated', 'username', 'user', 'session_token', 'allowed_pages']
    auth_state = {k: st.session_state.get(k) for k in auth_keys}
    
    # Reset all to defaults
    for key, default_value in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = default_value
    
    # Restore auth state
    for k, v in auth_state.items():
        if v is not None:
            st.session_state[k] = v


def set_page(page_name: str):
    """Set the current page and update history."""
    
    # Add to history
    if 'page_history' not in st.session_state:
        st.session_state.page_history = []
    
    current = st.session_state.get('current_page')
    if current and current != page_name:
        st.session_state.page_history.append({
            'page': current,
            'timestamp': datetime.now().isoformat()
        })
        # Keep only last 10 pages in history
        st.session_state.page_history = st.session_state.page_history[-10:]
    
    st.session_state.current_page = page_name
    st.session_state.page_views = st.session_state.get('page_views', 0) + 1


def get_page() -> str:
    """Get the current page name."""
    return st.session_state.get('current_page', 'Executive Overview')


def set_filter(filter_name: str, value: Any):
    """Set a filter value."""
    st.session_state[f'selected_{filter_name}'] = value


def get_filter(filter_name: str) -> Any:
    """Get a filter value."""
    return st.session_state.get(f'selected_{filter_name}')


def clear_filters():
    """Clear all filters to default values."""
    st.session_state.selected_study = 'All Studies'
    st.session_state.selected_site = 'All Sites'
    st.session_state.selected_region = 'All Regions'
    st.session_state.selected_status = 'All Statuses'
    st.session_state.date_range = None


def add_notification(notification: Dict[str, Any]):
    """Add a notification to the session."""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notification['id'] = len(st.session_state.notifications) + 1
    notification['timestamp'] = datetime.now().isoformat()
    notification['read'] = False
    
    st.session_state.notifications.insert(0, notification)
    st.session_state.unread_notifications = st.session_state.get('unread_notifications', 0) + 1
    
    # Keep only last 100 notifications
    st.session_state.notifications = st.session_state.notifications[:100]


def mark_notification_read(notification_id: int):
    """Mark a notification as read."""
    for notif in st.session_state.get('notifications', []):
        if notif.get('id') == notification_id and not notif.get('read'):
            notif['read'] = True
            st.session_state.unread_notifications = max(0, 
                st.session_state.get('unread_notifications', 0) - 1)
            break


def mark_all_notifications_read():
    """Mark all notifications as read."""
    for notif in st.session_state.get('notifications', []):
        notif['read'] = True
    st.session_state.unread_notifications = 0


def add_to_chat_history(role: str, content: str, metadata: Optional[Dict] = None):
    """Add a message to chat history."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    st.session_state.chat_history.append(message)
    
    # Keep only last 50 messages
    st.session_state.chat_history = st.session_state.chat_history[-50:]


def clear_chat_history():
    """Clear chat history."""
    st.session_state.chat_history = []


def set_preference(key: str, value: Any):
    """Set a user preference."""
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}
    st.session_state.preferences[key] = value


def get_preference(key: str, default: Any = None) -> Any:
    """Get a user preference."""
    return st.session_state.get('preferences', {}).get(key, default)


def cache_data(key: str, data: Any, ttl_seconds: int = 300):
    """Cache data with a TTL."""
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    
    st.session_state.cached_data[key] = {
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'ttl': ttl_seconds
    }


def get_cached_data(key: str) -> Optional[Any]:
    """Get cached data if not expired."""
    cached = st.session_state.get('cached_data', {}).get(key)
    
    if not cached:
        return None
    
    # Check TTL
    timestamp = datetime.fromisoformat(cached['timestamp'])
    ttl = cached.get('ttl', 300)
    
    if (datetime.now() - timestamp).total_seconds() > ttl:
        # Expired
        del st.session_state.cached_data[key]
        return None
    
    return cached['data']


def increment_action():
    """Increment the action counter."""
    st.session_state.actions_taken = st.session_state.get('actions_taken', 0) + 1


def increment_query():
    """Increment the query counter."""
    st.session_state.queries_made = st.session_state.get('queries_made', 0) + 1


def get_session_stats() -> Dict[str, Any]:
    """Get session statistics."""
    session_start = st.session_state.get('session_start')
    duration = None
    
    if session_start:
        start_dt = datetime.fromisoformat(session_start)
        duration = (datetime.now() - start_dt).total_seconds()
    
    return {
        'session_start': session_start,
        'duration_seconds': duration,
        'page_views': st.session_state.get('page_views', 0),
        'actions_taken': st.session_state.get('actions_taken', 0),
        'queries_made': st.session_state.get('queries_made', 0),
        'notifications': len(st.session_state.get('notifications', [])),
        'unread_notifications': st.session_state.get('unread_notifications', 0),
        'chat_messages': len(st.session_state.get('chat_history', [])),
        'cached_items': len(st.session_state.get('cached_data', {}))
    }


def set_temp(key: str, value: Any):
    """Set a temporary value (not persisted)."""
    if 'temp' not in st.session_state:
        st.session_state.temp = {}
    st.session_state.temp[key] = value


def get_temp(key: str, default: Any = None) -> Any:
    """Get a temporary value."""
    return st.session_state.get('temp', {}).get(key, default)


def clear_temp():
    """Clear all temporary values."""
    st.session_state.temp = {}