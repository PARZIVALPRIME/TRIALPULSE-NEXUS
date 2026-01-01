"""
TRIALPULSE NEXUS 10X - Dashboard Test Runner
Phase 7.1: Test the dashboard components
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

# Suppress Streamlit warnings when running outside of Streamlit context
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Session state does not function.*')

# Set environment variable to suppress Streamlit logging
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"   {status}: {name}")
    if details:
        print(f"           {details}")


def test_imports():
    """Test that all dashboard modules can be imported."""
    print_header("TEST 1: Module Imports")
    
    all_passed = True
    modules = [
        ("dashboard.config.theme", ["apply_theme", "get_theme_css", "THEME_CONFIG"]),
        ("dashboard.config.auth", ["authenticate", "check_authentication", "logout"]),
        ("dashboard.config.session", ["initialize_session", "set_page", "get_page"]),
        ("dashboard.components.metrics", ["render_metric_card", "render_dqi_metric"]),
        ("dashboard.components.charts", ["render_dqi_gauge", "render_trend_chart"]),
        ("dashboard.components.tables", ["render_data_table", "render_paginated_table"]),
        ("dashboard.components.cards", ["render_info_card", "render_action_card"]),
        ("dashboard.components.filters", ["render_filter_bar", "get_active_filters"]),
        ("dashboard.utils.data_loader", ["DashboardDataLoader", "get_data_loader"]),
        ("dashboard.utils.formatters", ["format_number", "format_percentage"]),
    ]
    
    for module_name, exports in modules:
        try:
            module = __import__(module_name, fromlist=exports)
            for export in exports:
                if not hasattr(module, export):
                    raise ImportError(f"Missing export: {export}")
            print_test(f"{module_name}", True)
        except Exception as e:
            print_test(f"{module_name}", False, str(e))
            all_passed = False
    
    return all_passed


def test_theme_config():
    """Test theme configuration."""
    print_header("TEST 2: Theme Configuration")
    
    from dashboard.config.theme import THEME_CONFIG, get_dqi_band_color, get_dqi_band_name
    
    all_passed = True
    
    # Test theme properties
    try:
        assert THEME_CONFIG.primary == "#2c3e50"
        assert THEME_CONFIG.accent == "#3498db"
        assert THEME_CONFIG.success == "#27ae60"
        print_test("Theme colors defined", True)
    except Exception as e:
        print_test("Theme colors defined", False, str(e))
        all_passed = False
    
    # Test DQI band functions
    test_cases = [
        (98, "Pristine", "#27ae60"),
        (88, "Excellent", "#2ecc71"),
        (78, "Good", "#f1c40f"),
        (68, "Fair", "#f39c12"),
        (55, "Poor", "#e67e22"),
        (35, "Critical", "#e74c3c"),
        (15, "Emergency", "#c0392b"),
    ]
    
    for dqi, expected_band, expected_color in test_cases:
        try:
            band = get_dqi_band_name(dqi)
            color = get_dqi_band_color(dqi)
            assert band == expected_band, f"Expected {expected_band}, got {band}"
            assert color == expected_color, f"Expected {expected_color}, got {color}"
            print_test(f"DQI {dqi} → {band}", True)
        except Exception as e:
            print_test(f"DQI {dqi} → {expected_band}", False, str(e))
            all_passed = False
    
    return all_passed


def test_auth_system():
    """Test authentication system."""
    print_header("TEST 3: Authentication System")
    
    from dashboard.config.auth import (
        DEMO_USERS, 
        hash_password,
        ROLE_PERMISSIONS
    )
    
    all_passed = True
    
    # Test demo users exist
    try:
        expected_users = ["lead", "cra", "dm", "safety", "site", "coder"]
        for user in expected_users:
            assert user in DEMO_USERS, f"Missing user: {user}"
        print_test("Demo users configured", True, f"{len(DEMO_USERS)} users")
    except Exception as e:
        print_test("Demo users configured", False, str(e))
        all_passed = False
    
    # Test password hashing
    try:
        hashed = hash_password("demo123")
        assert len(hashed) == 64  # SHA-256 hex length
        print_test("Password hashing", True)
    except Exception as e:
        print_test("Password hashing", False, str(e))
        all_passed = False
    
    # Test role permissions
    try:
        expected_roles = ["Study Lead", "CRA", "Data Manager", "Safety Physician", "Site User", "Medical Coder"]
        for role in expected_roles:
            assert role in ROLE_PERMISSIONS, f"Missing role: {role}"
            assert "pages" in ROLE_PERMISSIONS[role]
            assert "default_page" in ROLE_PERMISSIONS[role]
        print_test("Role permissions configured", True, f"{len(ROLE_PERMISSIONS)} roles")
    except Exception as e:
        print_test("Role permissions configured", False, str(e))
        all_passed = False
    
    return all_passed


def test_session_management():
    """Test session state management."""
    print_header("TEST 4: Session Management")
    
    from dashboard.config.session import DEFAULT_SESSION_STATE
    
    all_passed = True
    
    # Test default state keys
    try:
        expected_keys = [
            'authenticated', 'username', 'user', 'current_page',
            'selected_study', 'selected_site', 'chat_history',
            'notifications', 'preferences'
        ]
        for key in expected_keys:
            assert key in DEFAULT_SESSION_STATE, f"Missing key: {key}"
        print_test("Default session state", True, f"{len(DEFAULT_SESSION_STATE)} keys")
    except Exception as e:
        print_test("Default session state", False, str(e))
        all_passed = False
    
    # Test preferences structure
    try:
        prefs = DEFAULT_SESSION_STATE['preferences']
        assert 'default_view' in prefs
        assert 'items_per_page' in prefs
        assert 'auto_refresh' in prefs
        print_test("Preferences structure", True)
    except Exception as e:
        print_test("Preferences structure", False, str(e))
        all_passed = False
    
    return all_passed


def test_formatters():
    """Test formatting utilities."""
    print_header("TEST 5: Formatting Utilities")
    
    from dashboard.utils.formatters import (
        format_number,
        format_percentage,
        format_date,
        format_duration,
        format_currency,
        format_relative_time,
        truncate_text
    )
    
    all_passed = True
    
    # Test number formatting
    test_cases = [
        (format_number(1234567), "1,234,567"),
        (format_number(1234.567, decimals=2), "1,234.57"),
        (format_number(100, suffix="%"), "100%"),
    ]
    
    for result, expected in test_cases:
        try:
            assert result == expected, f"Expected {expected}, got {result}"
            print_test(f"format_number → {expected}", True)
        except Exception as e:
            print_test(f"format_number → {expected}", False, str(e))
            all_passed = False
    
    # Test percentage formatting
    try:
        assert format_percentage(0.534) == "53.4%"
        assert format_percentage(75.5) == "75.5%"
        print_test("format_percentage", True)
    except Exception as e:
        print_test("format_percentage", False, str(e))
        all_passed = False
    
    # Test duration formatting
    try:
        assert format_duration(3665, short=True) == "1h 1m"
        assert format_duration(125, short=True) == "2m"
        print_test("format_duration", True)
    except Exception as e:
        print_test("format_duration", False, str(e))
        all_passed = False
    
    # Test currency formatting
    try:
        assert format_currency(1234567) == "$1,234,567"
        assert format_currency(1234.56, decimals=2) == "$1,234.56"
        print_test("format_currency", True)
    except Exception as e:
        print_test("format_currency", False, str(e))
        all_passed = False
    
    # Test truncate
    try:
        assert truncate_text("Hello World", 8) == "Hello..."
        assert truncate_text("Short", 10) == "Short"
        print_test("truncate_text", True)
    except Exception as e:
        print_test("truncate_text", False, str(e))
        all_passed = False
    
    return all_passed


def test_data_loader():
    """Test data loader functionality."""
    print_header("TEST 6: Data Loader")
    
    # Mock streamlit session state for testing
    import streamlit as st
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    
    from dashboard.utils.data_loader import DashboardDataLoader
    
    all_passed = True
    
    # Test initialization
    try:
        loader = DashboardDataLoader()
        print_test("DataLoader initialization", True)
    except Exception as e:
        print_test("DataLoader initialization", False, str(e))
        all_passed = False
    
    # Test path properties
    try:
        loader = DashboardDataLoader()
        assert "upr" in str(loader.upr_path)
        assert "analytics" in str(loader.analytics_path)
        print_test("Path properties", True)
    except Exception as e:
        print_test("Path properties", False, str(e))
        all_passed = False
    
    # Test data loading (will fail gracefully if files don't exist)
    try:
        loader = DashboardDataLoader()
        summary = loader.get_portfolio_summary()
        assert isinstance(summary, dict)
        assert "total_patients" in summary
        assert "mean_dqi" in summary
        print_test("Portfolio summary structure", True)
    except Exception as e:
        print_test("Portfolio summary structure", False, str(e))
        all_passed = False
    
    return all_passed


def test_page_placeholders():
    """Test that all page modules exist."""
    print_header("TEST 7: Page Modules")
    
    pages = [
        "dashboard.pages.executive_overview",
        "dashboard.pages.cra_view",
        "dashboard.pages.dm_hub",
        "dashboard.pages.safety_view",
        "dashboard.pages.study_lead",
        "dashboard.pages.site_portal",
        "dashboard.pages.coder_view",
        "dashboard.pages.cascade_explorer",
        "dashboard.pages.ai_assistant",
        "dashboard.pages.reports",
        "dashboard.pages.settings",
    ]
    
    all_passed = True
    
    for page in pages:
        try:
            module = __import__(page, fromlist=['render_page'])
            assert hasattr(module, 'render_page'), f"Missing render_page function in {page}"
            print_test(f"{page.split('.')[-1]}", True)
        except Exception as e:
            print_test(f"{page.split('.')[-1]}", False, str(e))
            all_passed = False
    
    return all_passed


def test_css_generation():
    """Test CSS generation."""
    print_header("TEST 8: CSS Generation")
    
    from dashboard.config.theme import get_theme_css
    
    all_passed = True
    
    try:
        css = get_theme_css()
        assert "<style>" in css
        assert "</style>" in css
        assert "--primary:" in css
        assert "--accent:" in css
        assert ".metric-card" in css
        assert ".status-badge" in css
        print_test("CSS generated", True, f"{len(css):,} characters")
    except Exception as e:
        print_test("CSS generated", False, str(e))
        all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print()
    print("=" * 70)
    print(" TRIALPULSE NEXUS 10X - PHASE 7.1 DASHBOARD TEST")
    print("=" * 70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Theme Configuration", test_theme_config()))
    results.append(("Authentication System", test_auth_system()))
    results.append(("Session Management", test_session_management()))
    results.append(("Formatting Utilities", test_formatters()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Page Modules", test_page_placeholders()))
    results.append(("CSS Generation", test_css_generation()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    print()
    print(f"   Total: {len(results)} tests")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print()
    
    if failed == 0:
        print("   🎉 ALL TESTS PASSED!")
    else:
        print("   ⚠️  Some tests failed. Check the output above.")
    
    print()
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)