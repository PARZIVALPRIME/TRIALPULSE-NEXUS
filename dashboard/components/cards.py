"""
TRIALPULSE NEXUS 10X - Card Components
Phase 7.1: Dark theme reusable card components with glassmorphism
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable

from dashboard.config.theme import THEME_CONFIG, get_priority_color, get_status_color


def render_metric_card(
    value: str,
    label: str,
    icon: Optional[str] = None,
    delta: Optional[str] = None,
    delta_direction: str = "neutral",
    color: Optional[str] = None,
    glow: bool = False
):
    """Neo-Nexus Metric Card."""
    
    theme = THEME_CONFIG
    accent = color or theme.accent
    
    # Delta Badge
    delta_html = ""
    if delta:
        if delta_direction == "positive":
            bg = f"rgba(16, 185, 129, 0.2)"
            text = theme.success
            icon_delta = "↑"
        elif delta_direction == "negative":
            bg = f"rgba(239, 68, 68, 0.2)"
            text = theme.danger
            icon_delta = "↓"
        else:
            bg = f"rgba(148, 163, 184, 0.2)"
            text = theme.text_muted
            icon_delta = ""
            
        delta_html = f"""
        <div style="background: {bg}; color: {text}; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.25rem;">
            {icon_delta} {delta}
        </div>
        """
    
    glow_effect = f"box-shadow: 0 0 20px {accent}40;" if glow else ""
    border_color = accent if glow else theme.glass_border
    
    st.markdown(f"""
    <div style="background: {theme.gradient_card};
                border: 1px solid {border_color};
                border-radius: {theme.border_radius_lg};
                padding: 1.5rem;
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                {glow_effect}"
         onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.5)';"
         onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
        
        <!-- Top Accent Line -->
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, {accent}, transparent);"></div>
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; filter: drop-shadow(0 0 8px {accent}60);">{icon if icon else ''}</div>
            {delta_html}
        </div>
        
        <div style="font-size: 2rem; font-weight: 800; color: {theme.text_primary}; letter-spacing: -0.5px;">
            {value}
        </div>
        
        <div style="font-size: 0.8rem; color: {theme.text_secondary}; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem;">
            {label}
        </div>
        
        <!-- Background Decoration -->
        <div style="position: absolute; bottom: -20px; right: -20px; width: 100px; height: 100px; background: {accent}; opacity: 0.05; border-radius: 50%; blur: 40px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_info_card(
    title: str,
    content: str,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    footer: Optional[str] = None
):
    """Neo-Nexus Info Card."""
    
    theme = THEME_CONFIG
    accent = color or theme.accent
    
    st.markdown(f"""
    <div style="background: {theme.secondary_bg};
                border: 1px solid {theme.glass_border};
                border-left: 3px solid {accent};
                border-radius: {theme.border_radius};
                padding: 1.25rem;
                margin-bottom: 1rem;">
        <h4 style="color: {theme.text_primary}; margin: 0 0 0.75rem 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem;">
            {icon if icon else ''} {title}
        </h4>
        <div style="color: {theme.text_secondary}; font-size: 0.95rem; line-height: 1.6;">
            {content}
        </div>
        {f'<div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px dashed {theme.glass_border}; font-size: 0.8rem; color: {theme.text_muted};">{footer}</div>' if footer else ''}
    </div>
    """, unsafe_allow_html=True)


def render_action_card(
    title: str,
    description: str,
    actions: List[Dict[str, Any]],
    icon: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    impact: Optional[str] = None
):
    """Neo-Nexus Action Card."""
    
    theme = THEME_CONFIG
    priority_color = get_priority_color(priority) if priority else theme.accent
    
    # Badges
    priority_badge = ""
    if priority:
        priority_badge = f"""
        <span style="background: {priority_color}20; color: {priority_color}; border: 1px solid {priority_color}40; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;">{priority}</span>
        """
        
    status_badge = ""
    if status:
        s_color = get_status_color(status)
        status_badge = f"""
        <span style="color: {s_color}; font-size: 0.75rem; font-weight: 500; display: flex; align-items: center; gap: 0.3rem;">
            <span style="width: 6px; height: 6px; background: {s_color}; border-radius: 50%;"></span> {status}
        </span>
        """

    st.markdown(f"""
    <div style="background: {theme.gradient_card};
                border: 1px solid {theme.glass_border};
                border-radius: {theme.border_radius};
                padding: 1.5rem;
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;">
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                {f'<div style="font-size: 1.5rem;">{icon}</div>' if icon else ''}
                <div>
                    <h4 style="color: {theme.text_primary}; margin: 0; font-size: 1rem; font-weight: 600;">{title}</h4>
                    {status_badge}
                </div>
            </div>
            {priority_badge}
        </div>
        
        <p style="color: {theme.text_secondary}; font-size: 0.9rem; margin-bottom: 1rem; line-height: 1.5;">
            {description}
        </p>
        
        {f'<div style="background: rgba(6, 182, 212, 0.1); border-left: 2px solid {theme.info}; padding: 0.5rem 0.75rem; border-radius: 0 4px 4px 0; font-size: 0.8rem; margin-bottom: 1rem;"><strong style="color: {theme.info};">Impact:</strong> <span style="color: {theme.text_secondary};">{impact}</span></div>' if impact else ''}
        
    </div>
    """, unsafe_allow_html=True)
    
    # Render native buttons outside html
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                label = action.get('label', 'Action')
                if action.get('icon'):
                    label = f"{action['icon']} {label}"
                
                kind = "primary" if action.get('type') == 'primary' else "secondary"
                if st.button(label, key=action.get('key', f"act_{title}_{i}"), type=kind, use_container_width=True):
                    if action.get('callback'):
                        action['callback']()


def render_stat_card(
    stats: List[Dict[str, Any]],
    columns: int = 4,
    title: Optional[str] = None
):
    """Neo-Nexus Multi-Stat Card."""
    theme = THEME_CONFIG
    
    if title:
        st.markdown(f"<h3 style='font-size: 1.1rem; margin-bottom: 1rem; color: {theme.text_primary};'>{title}</h3>", unsafe_allow_html=True)
        
    cols = st.columns(columns)
    for i, stat in enumerate(stats):
        with cols[i]:
            val_color = stat.get('color', theme.text_primary)
            st.markdown(f"""
            <div style="background: {theme.secondary_bg}; border: 1px solid {theme.glass_border}; border-radius: {theme.border_radius}; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.8;">{stat.get('icon', '')}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: {val_color};">{stat.get('value')}</div>
                <div style="font-size: 0.75rem; color: {theme.text_muted}; text-transform: uppercase;">{stat.get('label')}</div>
            </div>
            """, unsafe_allow_html=True)


def render_alert_card(
    title: str,
    message: str,
    alert_type: str = "info",
    dismissible: bool = False,
    actions: Optional[List[Dict[str, Any]]] = None
):
    """Neo-Nexus Alert Card."""
    
    theme = THEME_CONFIG
    colors = {
        'success': (theme.success, "rgba(16, 185, 129, 0.1)"),
        'warning': (theme.warning, "rgba(245, 158, 11, 0.1)"),
        'danger': (theme.danger, "rgba(239, 68, 68, 0.1)"),
        'info': (theme.info, "rgba(6, 182, 212, 0.1)")
    }
    
    c_hex, c_bg = colors.get(alert_type, (theme.info, "rgba(6, 182, 212, 0.1)"))
    
    st.markdown(f"""
    <div style="background: {c_bg}; border: 1px solid {c_hex}40; border-radius: {theme.border_radius}; padding: 1rem; margin-bottom: 1rem; display: flex; gap: 1rem;">
        <div style="font-size: 1.25rem; color: {c_hex};">ℹ️</div>
        <div>
            <div style="font-weight: 700; color: {c_hex}; margin-bottom: 0.25rem;">{title}</div>
            <div style="color: {theme.text_secondary}; font-size: 0.9rem;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if actions:
        col_acts = st.columns(len(actions))
        for i, a in enumerate(actions):
            with col_acts[i]:
                st.button(a['label'], key=a.get('key'))

def render_progress_card(title: str, progress: float, subtitle: str = None, color: str = None, show_percentage: bool = True):
    """Neo-Nexus Progress Card."""
    theme = THEME_CONFIG
    c = color or theme.accent
    
    st.markdown(f"""
    <div style="background: {theme.secondary_bg}; border: 1px solid {theme.glass_border}; border-radius: {theme.border_radius}; padding: 1.25rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; color: {theme.text_primary};">{title}</span>
            <span style="font-weight: 700; color: {c};">{progress}%</span>
        </div>
        <div style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
            <div style="width: {progress}%; height: 100%; background: linear-gradient(90deg, {c}, {theme.accent_light}); border-radius: 4px; box-shadow: 0 0 10px {c}60;"></div>
        </div>
        {f'<div style="font-size: 0.75rem; color: {theme.text_muted}; margin-top: 0.5rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_timeline_card(events, title=None):
    theme = THEME_CONFIG
    if title: st.markdown(f"### {title}")
    
    html = ""
    for i, e in enumerate(events):
        color = e.get('color', theme.accent)
        is_last = i == len(events)-1
        html += f"""
        <div style="display: flex; gap: 1rem; padding-bottom: {'0' if is_last else '1.5rem'};">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: {color}; box-shadow: 0 0 10px {color}; z-index: 2;"></div>
                {'' if is_last else f'<div style="width: 2px; flex: 1; background: {theme.glass_border}; margin-top: 4px;"></div>'}
            </div>
            <div>
                <div style="font-weight: 600; color: {theme.text_primary};">{e.get('title')}</div>
                <div style="font-size: 0.85rem; color: {theme.text_secondary};">{e.get('description')}</div>
                <div style="font-size: 0.75rem; color: {theme.text_muted}; margin-top: 0.25rem;">{e.get('time')}</div>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div style="background: {theme.secondary_bg}; border: 1px solid {theme.glass_border}; border-radius: {theme.border_radius}; padding: 1.5rem;">
        {html}
    </div>
    """, unsafe_allow_html=True)

def render_dqi_gauge(dqi_score: float, size: str = "medium"):
    """Neo-Nexus DQI Gauge."""
    from dashboard.config.theme import get_dqi_band_color
    theme = THEME_CONFIG
    color = get_dqi_band_color(dqi_score)
    
    dim = {"small": "80px", "medium": "120px", "large": "160px"}.get(size, "120px")
    fs = {"small": "1.2rem", "medium": "1.8rem", "large": "2.5rem"}.get(size, "1.8rem")
    
    st.markdown(f"""
    <div style="position: relative; width: {dim}; height: {dim}; margin: 0 auto;">
        <div style="width: 100%; height: 100%; border-radius: 50%; background: conic-gradient({color} {dqi_score}%, {theme.glass_border} 0); box-shadow: 0 0 20px {color}20;"></div>
        <div style="position: absolute; top: 10%; left: 10%; width: 80%; height: 80%; background: {theme.secondary_bg}; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <div style="font-size: {fs}; font-weight: 800; color: {theme.text_primary};">{dqi_score}</div>
            <div style="font-size: 0.7rem; color: {color}; font-weight: 600;">SCORE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)