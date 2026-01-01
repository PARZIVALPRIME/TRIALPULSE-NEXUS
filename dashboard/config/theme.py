"""
TRIALPULSE NEXUS 10X - Theme Configuration
Phase 8.0: Ultra-Premium 'Neo-Nexus' Aesthetic
"""

import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ThemeConfig:
    """Theme configuration dataclass - Neo-Nexus Theme."""
    
    # Core Palette - Deep Void & Starlight
    primary: str = "#0f172a"      # Deep Slate (Sidebar/Nav)
    background: str = "#020617"   # Void Black (Main BG)
    secondary_bg: str = "#1e293b" # Lighter Slate (Cards)
    
    # Brand Gradients
    gradient_primary: str = "linear-gradient(135deg, #0f172a 0%, #020617 100%)"
    gradient_accent: str = "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)" # Indigo to Purple
    gradient_header: str = "linear-gradient(90deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.9) 100%)"
    gradient_card: str = "linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.6) 100%)"
    
    # Accents
    accent: str = "#8b5cf6"       # Vivid Violet
    accent_light: str = "#a78bfa" # Light Violet
    accent_glow: str = "0 0 15px rgba(139, 92, 246, 0.5)"
    
    # Functional Colors (Neon/Cyberpunk inspired)
    success: str = "#10b981"      # Emerald Neon
    success_light: str = "#34d399"
    warning: str = "#f59e0b"      # Amber Neon
    warning_light: str = "#fbbf24"
    danger: str = "#ef4444"       # Crimson Neon
    danger_light: str = "#f87171"
    info: str = "#06b6d4"         # Cyan Neon
    info_light: str = "#22d3ee"
    
    # Text
    text_primary: str = "#f8fafc"   # White/Slate 50
    text_secondary: str = "#cbd5e1" # Slate 300
    text_muted: str = "#64748b"     # Slate 500
    
    # Borders & Glass
    border_color: str = "rgba(148, 163, 184, 0.1)"
    glass_border: str = "rgba(255, 255, 255, 0.08)"
    glass_background: str = "rgba(30, 41, 59, 0.4)"
    border_radius: str = "12px"
    border_radius_lg: str = "24px"
    
    # Shadows
    shadow_sm: str = "0 2px 4px rgba(0,0,0,0.3)"
    shadow_md: str = "0 8px 16px rgba(0,0,0,0.4)"
    shadow_glow: str = "0 0 20px rgba(99, 102, 241, 0.25)"


THEME_CONFIG = ThemeConfig()


def get_theme_css() -> str:
    """Generate CSS for the Neo-Nexus theme."""
    
    theme = THEME_CONFIG
    
    return f"""
    <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        :root {{
            --primary: {theme.primary};
            --background: {theme.background};
            --text-primary: {theme.text_primary};
        }}
        
        /* Base App Styling */
        .stApp {{
            font-family: 'Inter', sans-serif;
            background: {theme.background};
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 20%);
            background-attachment: fixed;
            color: {theme.text_primary};
        }}
        
        /* HEADINGS */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif;
            color: {theme.text_primary};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        
        /* BUTTONS - Primary */
        div[data-testid="stButton"] > button[kind="primary"] {{
            background: {theme.gradient_accent};
            color: white;
            border: none;
            border-radius: {theme.border_radius};
            padding: 0.6rem 1.25rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: {theme.shadow_glow};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
        }}
        
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 25px rgba(139, 92, 246, 0.6);
            border: none;
        }}
        
        /* BUTTONS - Secondary */
        div[data-testid="stButton"] > button[kind="secondary"] {{
            background: rgba(255, 255, 255, 0.05);
            color: {theme.text_secondary};
            border: 1px solid {theme.glass_border};
            border-radius: {theme.border_radius};
            transition: all 0.2s;
        }}
        
        div[data-testid="stButton"] > button[kind="secondary"]:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: {theme.text_secondary};
            color: white;
        }}
        
        /* INPUTS & SELECTBOXES */
        div[data-testid="stTextInput"] > div > div > input,
        div[data-testid="stSelectbox"] > div > div > div {{
            background-color: rgba(15, 23, 42, 0.6);
            color: {theme.text_primary};
            border: 1px solid {theme.glass_border};
            border-radius: {theme.border_radius};
        }}
        
        div[data-testid="stTextInput"] > div > div > input:focus,
        div[data-testid="stSelectbox"] > div > div[aria-expanded="true"] {{
            border-color: {theme.accent};
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
        }}
        
        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {{
            background-color: {theme.primary};
            border-right: 1px solid {theme.glass_border};
        }}
        
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}
        
        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {theme.primary}; 
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #334155; 
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {theme.accent}; 
        }}
        
        /* DATAFRAME / TABLE STYLING */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {theme.glass_border};
            border-radius: {theme.border_radius};
            overflow: hidden;
        }}
        
        div[data-testid="stDataFrame"] div[class*="col-header"] {{
            background-color: {theme.secondary_bg};
            color: {theme.text_secondary};
            font-weight: 600;
        }}
        
        /* METRIC CARDS OVERRIDE */
        div[data-testid="stMetric"] {{
            background: transparent;
            padding: 0;
        }}
        
         /* TOAST ALERTS */
        div[class*="stToast"] {{
            background-color: {theme.secondary_bg} !important;
            border: 1px solid {theme.glass_border};
            color: {theme.text_primary} !important;
        }}

        /* KEYFRAMES */
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        
        @keyframes glow {{
            0% {{ box-shadow: 0 0 5px rgba(139, 92, 246, 0.2); }}
            50% {{ box-shadow: 0 0 20px rgba(139, 92, 246, 0.6); }}
            100% {{ box-shadow: 0 0 5px rgba(139, 92, 246, 0.2); }}
        }}
    </style>
    """


def apply_theme():
    """Apply the theme CSS to the application."""
    st.markdown(get_theme_css(), unsafe_allow_html=True)


def get_priority_color(priority: str) -> str:
    """Get color for priority levels."""
    colors = {
        'Critical': THEME_CONFIG.danger,
        'High': THEME_CONFIG.warning,
        'Medium': THEME_CONFIG.info,
        'Low': THEME_CONFIG.success
    }
    return colors.get(priority, THEME_CONFIG.text_muted)


def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        'On Track': THEME_CONFIG.success,
        'At Risk': THEME_CONFIG.warning,
        'Delayed': THEME_CONFIG.danger,
        'Completed': THEME_CONFIG.info,
        'Planned': THEME_CONFIG.text_muted
    }
    return colors.get(status, THEME_CONFIG.text_muted)


def get_dqi_band_color(score: float) -> str:
    """Get color for DQI score band."""
    if score >= 95: return THEME_CONFIG.success
    if score >= 90: return str(THEME_CONFIG.success_light) # Cast to str just in case 
    if score >= 85: return THEME_CONFIG.info
    if score >= 80: return THEME_CONFIG.warning
    return THEME_CONFIG.danger

def get_dqi_band_name(score: float) -> str:
    """Get name for DQI score band."""
    if score >= 95: return "Elite"
    if score >= 90: return "Optimal"
    if score >= 85: return "Standard"
    if score >= 80: return "Risk"
    return "Critical"