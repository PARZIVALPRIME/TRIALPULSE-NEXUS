"""
TRIALPULSE NEXUS 10X - Authentication Module
Phase 7.1: Simple authentication for demo purposes
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import secrets

# Demo users with roles and permissions
DEMO_USERS = {
    "lead": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "John Smith",
        "role": "Study Lead",
        "email": "john.smith@example.com",
        "permissions": ["all"],
        "studies": ["all"],
        "avatar": "👨‍💼"
    },
    "cra": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Sarah Chen",
        "role": "CRA",
        "email": "sarah.chen@example.com",
        "permissions": ["cra_view", "reports", "ai_assistant"],
        "studies": ["Study_21", "Study_22", "Study_23"],
        "sites": ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"],
        "avatar": "👩‍⚕️"
    },
    "dm": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Alex Kim",
        "role": "Data Manager",
        "email": "alex.kim@example.com",
        "permissions": ["dm_hub", "cascade", "reports", "ai_assistant"],
        "studies": ["all"],
        "avatar": "👨‍💻"
    },
    "safety": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Dr. Maria Garcia",
        "role": "Safety Physician",
        "email": "maria.garcia@example.com",
        "permissions": ["safety_view", "reports", "ai_assistant"],
        "studies": ["all"],
        "avatar": "👩‍⚕️"
    },
    "site": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Site Coordinator",
        "role": "Site User",
        "email": "site.coordinator@example.com",
        "permissions": ["site_portal"],
        "studies": ["Study_21"],
        "sites": ["Site_1"],
        "avatar": "🏥"
    },
    "coder": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Mike Johnson",
        "role": "Medical Coder",
        "email": "mike.johnson@example.com",
        "permissions": ["coder_view", "reports"],
        "studies": ["all"],
        "avatar": "👨‍🔬"
    }
}

# Role-based page access
ROLE_PERMISSIONS = {
    "Study Lead": {
        "pages": ["Executive Overview", "CRA Field View", "Data Manager Hub", 
                  "Safety Surveillance", "Study Lead Command", "Site Portal",
                  "Coder Workbench", "Cascade Explorer", "AI Assistant", 
                  "Reports", "Settings"],
        "default_page": "Executive Overview"
    },
    "CRA": {
        "pages": ["CRA Field View", "Site Portal", "AI Assistant", "Reports"],
        "default_page": "CRA Field View"
    },
    "Data Manager": {
        "pages": ["Data Manager Hub", "Cascade Explorer", "AI Assistant", "Reports"],
        "default_page": "Data Manager Hub"
    },
    "Safety Physician": {
        "pages": ["Safety Surveillance", "AI Assistant", "Reports"],
        "default_page": "Safety Surveillance"
    },
    "Site User": {
        "pages": ["Site Portal"],
        "default_page": "Site Portal"
    },
    "Medical Coder": {
        "pages": ["Coder Workbench", "Reports"],
        "default_page": "Coder Workbench"
    }
}


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username: str, password: str) -> bool:
    """
    Authenticate a user with username and password.
    
    Returns True if authentication successful, False otherwise.
    """
    if not username or not password:
        return False
    
    username_lower = username.lower().strip()
    
    if username_lower not in DEMO_USERS:
        return False
    
    user = DEMO_USERS[username_lower]
    password_hash = hash_password(password)
    
    if password_hash == user["password_hash"]:
        # Set session state
        st.session_state.authenticated = True
        st.session_state.username = username_lower
        st.session_state.user = {
            "username": username_lower,
            "name": user["name"],
            "role": user["role"],
            "email": user["email"],
            "permissions": user.get("permissions", []),
            "studies": user.get("studies", []),
            "sites": user.get("sites", []),
            "avatar": user.get("avatar", "👤"),
            "login_time": datetime.now().isoformat()
        }
        st.session_state.session_token = secrets.token_hex(32)
        
        # Set default page based on role
        role_config = ROLE_PERMISSIONS.get(user["role"], {})
        st.session_state.current_page = role_config.get("default_page", "Executive Overview")
        st.session_state.allowed_pages = role_config.get("pages", [])
        
        return True
    
    return False


def check_authentication() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user() -> Optional[Dict[str, Any]]:
    """Get the current authenticated user."""
    if check_authentication():
        return st.session_state.get('user', None)
    return None


def logout():
    """Log out the current user."""
    keys_to_clear = [
        'authenticated', 'username', 'user', 'session_token',
        'current_page', 'allowed_pages'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission."""
    user = get_current_user()
    if not user:
        return False
    
    permissions = user.get('permissions', [])
    if 'all' in permissions:
        return True
    
    return permission in permissions


def can_access_page(page_name: str) -> bool:
    """Check if current user can access a specific page."""
    allowed_pages = st.session_state.get('allowed_pages', [])
    return page_name in allowed_pages


def can_access_study(study_id: str) -> bool:
    """Check if current user can access a specific study."""
    user = get_current_user()
    if not user:
        return False
    
    studies = user.get('studies', [])
    if 'all' in studies:
        return True
    
    return study_id in studies


def can_access_site(site_id: str) -> bool:
    """Check if current user can access a specific site."""
    user = get_current_user()
    if not user:
        return False
    
    # Study Leads and Data Managers can access all sites
    if user.get('role') in ['Study Lead', 'Data Manager']:
        return True
    
    sites = user.get('sites', [])
    return site_id in sites


def get_session_duration() -> Optional[timedelta]:
    """Get the duration of the current session."""
    user = get_current_user()
    if not user:
        return None
    
    login_time = user.get('login_time')
    if login_time:
        login_dt = datetime.fromisoformat(login_time)
        return datetime.now() - login_dt
    
    return None