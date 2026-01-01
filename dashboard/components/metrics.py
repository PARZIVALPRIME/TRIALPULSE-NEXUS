"""
TRIALPULSE NEXUS 10X - Metrics Components
Phase 7.1: Metric cards and displays
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Union

from dashboard.config.theme import (
    THEME_CONFIG, 
    get_dqi_band_color, 
    get_dqi_band_name,
    get_status_color
)


def render_metric_card(
    title: str,
    value: Union[str, int, float],
    subtitle: Optional[str] = None,
    delta: Optional[Union[str, int, float]] = None,
    delta_color: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    help_text: Optional[str] = None,
    size: str = "medium"  # small, medium, large
):
    """
    Render a metric card.
    
    Args:
        title: Card title
        value: Main metric value
        subtitle: Optional subtitle text
        delta: Optional delta/change value
        delta_color: Color for delta (positive, negative, neutral)
        icon: Optional emoji icon
        color: Optional accent color
        help_text: Optional tooltip text
        size: Card size (small, medium, large)
    """
    
    # Size configurations
    sizes = {
        "small": {"value_size": "1.5rem", "title_size": "0.75rem", "padding": "0.75rem"},
        "medium": {"value_size": "2rem", "title_size": "0.85rem", "padding": "1rem"},
        "large": {"value_size": "2.5rem", "title_size": "1rem", "padding": "1.25rem"}
    }
    
    size_config = sizes.get(size, sizes["medium"])
    accent_color = color or THEME_CONFIG.accent
    
    # Delta styling
    delta_html = ""
    if delta is not None:
        if delta_color == "positive" or (isinstance(delta, (int, float)) and delta > 0):
            delta_style = f"color: {THEME_CONFIG.success};"
            delta_icon = "↑"
        elif delta_color == "negative" or (isinstance(delta, (int, float)) and delta < 0):
            delta_style = f"color: {THEME_CONFIG.danger};"
            delta_icon = "↓"
        else:
            delta_style = f"color: {THEME_CONFIG.text_secondary};"
            delta_icon = "→"
        
        delta_val = f"+{delta}" if isinstance(delta, (int, float)) and delta > 0 else str(delta)
        delta_html = f"""
        <div style="font-size: 0.8rem; margin-top: 0.5rem; {delta_style}">
            {delta_icon} {delta_val}
        </div>
        """
    
    # Icon HTML
    icon_html = f"<span style='margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    
    # Help tooltip
    help_attr = f'title="{help_text}"' if help_text else ""
    
    # Render card
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {accent_color}; padding: {size_config['padding']};" {help_attr}>
        <div style="font-size: {size_config['title_size']}; color: {THEME_CONFIG.text_secondary}; margin-bottom: 0.25rem;">
            {icon_html}{title}
        </div>
        <div style="font-size: {size_config['value_size']}; font-weight: 700; color: {THEME_CONFIG.text_primary};">
            {value}
        </div>
        {f'<div style="font-size: 0.75rem; color: {THEME_CONFIG.text_muted}; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]], columns: int = 4):
    """
    Render a row of metric cards.
    
    Args:
        metrics: List of metric configurations
        columns: Number of columns
    """
    
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(**metric)


def render_dqi_metric(dqi_value: float, size: str = "large"):
    """Render a DQI-specific metric card with color coding."""
    
    band_color = get_dqi_band_color(dqi_value)
    band_name = get_dqi_band_name(dqi_value)
    
    render_metric_card(
        title="Data Quality Index",
        value=f"{dqi_value:.1f}",
        subtitle=band_name,
        icon="📊",
        color=band_color,
        size=size
    )


def render_progress_metric(
    title: str,
    current: Union[int, float],
    total: Union[int, float],
    unit: str = "",
    color: Optional[str] = None
):
    """Render a progress-style metric card."""
    
    percentage = (current / total * 100) if total > 0 else 0
    
    # Determine color based on percentage
    if color is None:
        if percentage >= 90:
            color = THEME_CONFIG.success
        elif percentage >= 70:
            color = THEME_CONFIG.warning
        else:
            color = THEME_CONFIG.danger
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: {THEME_CONFIG.text_secondary}; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {THEME_CONFIG.text_primary};">
            {current:,} / {total:,} {unit}
        </div>
        <div class="progress-bar" style="margin-top: 0.75rem;">
            <div class="fill" style="width: {percentage}%; background: {color};"></div>
        </div>
        <div style="font-size: 0.75rem; color: {THEME_CONFIG.text_muted}; margin-top: 0.25rem; text-align: right;">
            {percentage:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_metric(title: str, status: str, details: Optional[str] = None):
    """Render a status indicator metric."""
    
    status_color = get_status_color(status)
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {status_color};">
        <div style="font-size: 0.85rem; color: {THEME_CONFIG.text_secondary};">
            {title}
        </div>
        <div style="display: flex; align-items: center; margin-top: 0.5rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: {status_color}; margin-right: 0.5rem;"></div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {THEME_CONFIG.text_primary};">
                {status}
            </div>
        </div>
        {f'<div style="font-size: 0.75rem; color: {THEME_CONFIG.text_muted}; margin-top: 0.5rem;">{details}</div>' if details else ''}
    </div>
    """, unsafe_allow_html=True)


def render_comparison_metric(
    title: str,
    value1: Union[int, float],
    label1: str,
    value2: Union[int, float],
    label2: str
):
    """Render a comparison metric showing two values side by side."""
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: {THEME_CONFIG.text_secondary}; margin-bottom: 0.75rem;">
            {title}
        </div>
        <div style="display: flex; justify-content: space-between;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {THEME_CONFIG.text_primary};">
                    {value1:,}
                </div>
                <div style="font-size: 0.75rem; color: {THEME_CONFIG.text_muted};">
                    {label1}
                </div>
            </div>
            <div style="border-left: 1px solid {THEME_CONFIG.border_color}; margin: 0 1rem;"></div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {THEME_CONFIG.text_primary};">
                    {value2:,}
                </div>
                <div style="font-size: 0.75rem; color: {THEME_CONFIG.text_muted};">
                    {label2}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)