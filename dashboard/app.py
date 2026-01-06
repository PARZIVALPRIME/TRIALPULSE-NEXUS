"""
TRIALPULSE NEXUS 10X - Main Dashboard Application
Phase 7.1: Streamlit App Setup v1.0

Multi-page clinical trial intelligence dashboard with authentication,
session management, and role-based access control.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# DATA INITIALIZATION (for Streamlit Cloud deployment)
# =============================================================================

def check_and_initialize_data():
    """Check if data exists, run pipelines if not."""
    analytics_dir = PROJECT_ROOT / "data" / "processed" / "analytics"
    required_files = [
        "patient_dqi_enhanced.parquet",
        "patient_clean_status.parquet", 
        "patient_dblock_status.parquet",
        "site_benchmarks.parquet"
    ]
    
    # Check if data exists
    data_exists = analytics_dir.exists() and all(
        (analytics_dir / f).exists() for f in required_files
    )
    
    if not data_exists:
        st.info("🔄 First-time setup: Initializing data pipelines... This may take a few minutes.")
        
        with st.spinner("Running data pipelines..."):
            try:
                from initialize_data import initialize_data
                success = initialize_data()
                if success:
                    st.success("✅ Data initialization complete! Refreshing...")
                    st.rerun()
                else:
                    st.warning("⚠️ Some pipelines failed. Dashboard may have limited data.")
            except Exception as e:
                st.error(f"❌ Data initialization failed: {e}")
                st.info("Please check if raw data files exist in data/raw/")
                return False
    return True

# Run data check before anything else (cached to run once per session)
@st.cache_resource
def _init_data_once():
    return check_and_initialize_data()

# =============================================================================

from dashboard.config.theme import apply_theme, get_theme_css
from dashboard.config.auth import check_authentication, get_current_user, logout
from dashboard.config.session import initialize_session, get_session_stats
from dashboard.components.sidebar import render_sidebar
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="TRIALPULSE NEXUS 10X",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://trialpulse-nexus.docs.io/help',
        'Report a bug': 'https://trialpulse-nexus.docs.io/issues',
        'About': '''
        ## TRIALPULSE NEXUS 10X
        
        **The Self-Evolving Clinical Trial Operating System**
        
        - 9 Data Sources → 1 Truth
        - 6 AI Agents with Human Oversight
        - Predictive Intelligence + Actionable Insights
        
        Version: 10.0.0
        '''
    }
)

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session()
    
    # Apply theme
    apply_theme()
    
    # Check and initialize data (runs pipelines if needed - for Streamlit Cloud)
    if not _init_data_once():
        st.error("Failed to initialize data. Please check logs.")
        return
    
    # Check authentication
    if not check_authentication():
        render_login_page()
        return
    
    # Get current user
    user = get_current_user()
    
    # Render main layout
    render_header(user)
    render_sidebar(user)
    
    # Main content area
    render_main_content(user)
    
    # Footer
    render_footer()


def render_login_page():
    """Render the login page."""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #2c3e50; font-size: 2.5rem;">🧬 TRIALPULSE NEXUS 10X</h1>
            <p style="color: #7f8c8d; font-size: 1.1rem;">Clinical Trial Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            st.subheader("🔐 Login")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                remember_me = st.checkbox("Remember me")
            with col_b:
                st.markdown("<div style='text-align: right;'><a href='#'>Forgot password?</a></div>", 
                           unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submitted:
                from dashboard.config.auth import authenticate
                
                if authenticate(username, password):
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # Demo credentials info
        with st.expander("🔑 Demo Credentials"):
            st.markdown("""
            | Role | Username | Password |
            |------|----------|----------|
            | Study Lead | `lead` | `demo123` |
            | CRA | `cra` | `demo123` |
            | Data Manager | `dm` | `demo123` |
            | Safety Lead | `safety` | `demo123` |
            | Site User | `site` | `demo123` |
            | Medical Coder | `coder` | `demo123` |
            """)


def render_main_content(user: dict):
    """Render the main content based on current page."""
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'Executive Overview')
    
    # Page routing
    if current_page == 'Executive Overview':
        from dashboard.pages.executive_overview import render_page
        render_page(user)
    
    elif current_page == 'CRA Field View':
        from dashboard.pages.cra_view import render_page
        render_page(user)
    
    elif current_page == 'Data Manager Hub':
        from dashboard.pages.dm_hub import render_page
        render_page(user)
    
    elif current_page == 'Safety Surveillance':
        from dashboard.pages.safety_view import render_page
        render_page(user)
    
    elif current_page == 'Study Lead Command':
        from dashboard.pages.study_lead import render_page
        render_page(user)
    
    elif current_page == 'Site Portal':
        from dashboard.pages.site_portal import render_page
        render_page(user)
    
    elif current_page == 'Coder Workbench':
        from dashboard.pages.coder_view import render_page
        render_page(user)
    
    elif current_page == 'Collaboration Hub':
        from dashboard.pages.collaboration_hub import render_page
        render_page(user)
    
    elif current_page == 'Cascade Explorer':
        from dashboard.pages.cascade_explorer import render_page
        render_page(user)
    
    elif current_page == 'AI Assistant':
        from dashboard.pages.ai_assistant import render_page
        render_page(user)
    
    elif current_page == 'Reports':
        from dashboard.pages.reports import render_page
        render_page(user)
    
    elif current_page == 'Settings':
        from dashboard.pages.settings import render_page
        render_page(user)
    
    else:
        # Default to Executive Overview
        from dashboard.pages.executive_overview import render_page
        render_page(user)


if __name__ == "__main__":
    main()