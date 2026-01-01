"""
Dashboard configuration module.
"""

from .theme import apply_theme, get_theme_css, THEME_CONFIG
from .auth import (
    check_authentication, 
    authenticate, 
    logout, 
    get_current_user,
    DEMO_USERS
)
from .session import (
    initialize_session,
    get_session_stats,
    set_page,
    get_page
)

__all__ = [
    'apply_theme',
    'get_theme_css',
    'THEME_CONFIG',
    'check_authentication',
    'authenticate',
    'logout',
    'get_current_user',
    'DEMO_USERS',
    'initialize_session',
    'get_session_stats',
    'set_page',
    'get_page'
]