"""
TRIALPULSE NEXUS 10X - Filter Components
Phase 7.1: Reusable filter components
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from dashboard.config.session import set_filter, get_filter, clear_filters


def render_filter_bar(
    studies: List[str] = None,
    sites: List[str] = None,
    regions: List[str] = None,
    statuses: List[str] = None,
    show_date_range: bool = False,
    show_search: bool = False,
    key_prefix: str = "filter"
):
    """
    Render a filter bar with common filter options.
    
    Args:
        studies: List of available studies
        sites: List of available sites
        regions: List of available regions
        statuses: List of available statuses
        show_date_range: Whether to show date range filter
        show_search: Whether to show search box
        key_prefix: Prefix for filter keys
    """
    
    # Calculate number of filters
    num_filters = sum([
        bool(studies),
        bool(sites),
        bool(regions),
        bool(statuses),
        show_date_range,
        show_search
    ])
    
    if num_filters == 0:
        return
    
    with st.container():
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        """, unsafe_allow_html=True)
        
        cols = st.columns(min(num_filters, 4))
        col_idx = 0
        
        # Search filter
        if show_search:
            with cols[col_idx % len(cols)]:
                search = st.text_input(
                    "🔍 Search",
                    value=get_filter('search') or "",
                    key=f"{key_prefix}_search",
                    placeholder="Search..."
                )
                set_filter('search', search)
            col_idx += 1
        
        # Study filter
        if studies:
            with cols[col_idx % len(cols)]:
                study_options = ["All Studies"] + studies
                selected_study = st.selectbox(
                    "📚 Study",
                    study_options,
                    index=study_options.index(get_filter('study') or "All Studies"),
                    key=f"{key_prefix}_study"
                )
                set_filter('study', selected_study)
            col_idx += 1
        
        # Site filter
        if sites:
            with cols[col_idx % len(cols)]:
                site_options = ["All Sites"] + sites
                selected_site = st.selectbox(
                    "🏥 Site",
                    site_options,
                    index=site_options.index(get_filter('site') or "All Sites") if get_filter('site') in site_options else 0,
                    key=f"{key_prefix}_site"
                )
                set_filter('site', selected_site)
            col_idx += 1
        
        # Region filter
        if regions:
            with cols[col_idx % len(cols)]:
                region_options = ["All Regions"] + regions
                selected_region = st.selectbox(
                    "🌍 Region",
                    region_options,
                    index=region_options.index(get_filter('region') or "All Regions") if get_filter('region') in region_options else 0,
                    key=f"{key_prefix}_region"
                )
                set_filter('region', selected_region)
            col_idx += 1
        
        # Status filter
        if statuses:
            with cols[col_idx % len(cols)]:
                status_options = ["All Statuses"] + statuses
                selected_status = st.selectbox(
                    "📊 Status",
                    status_options,
                    index=status_options.index(get_filter('status') or "All Statuses") if get_filter('status') in status_options else 0,
                    key=f"{key_prefix}_status"
                )
                set_filter('status', selected_status)
            col_idx += 1
        
        # Date range filter
        if show_date_range:
            with cols[col_idx % len(cols)]:
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(datetime.now() - timedelta(days=30), datetime.now()),
                    key=f"{key_prefix}_date_range"
                )
                if len(date_range) == 2:
                    set_filter('date_range', date_range)
        
        # Clear filters button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear", key=f"{key_prefix}_clear", use_container_width=True):
                clear_filters()
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_priority_filter(key: str = "priority_filter") -> List[str]:
    """
    Render a priority filter with checkboxes.
    
    Returns:
        List of selected priorities
    """
    
    st.markdown("**Priority**")
    
    priorities = ["Critical", "High", "Medium", "Low"]
    selected = []
    
    cols = st.columns(4)
    for i, priority in enumerate(priorities):
        with cols[i]:
            if st.checkbox(priority, value=True, key=f"{key}_{priority}"):
                selected.append(priority)
    
    return selected


def render_issue_type_filter(
    issue_types: List[str],
    key: str = "issue_filter"
) -> List[str]:
    """
    Render an issue type multi-select filter.
    
    Args:
        issue_types: List of available issue types
        key: Unique key for the component
    
    Returns:
        List of selected issue types
    """
    
    selected = st.multiselect(
        "Issue Types",
        options=issue_types,
        default=issue_types,
        key=key
    )
    
    return selected


def render_quick_filters(
    options: List[Dict[str, Any]],
    key: str = "quick_filter"
) -> Optional[str]:
    """
    Render quick filter buttons.
    
    Args:
        options: List of filter options with 'label', 'value', 'icon'
        key: Unique key for the component
    
    Returns:
        Selected filter value or None
    """
    
    cols = st.columns(len(options))
    selected = None
    
    for i, option in enumerate(options):
        with cols[i]:
            icon = option.get('icon', '')
            label = f"{icon} {option.get('label', '')}" if icon else option.get('label', '')
            
            if st.button(
                label,
                key=f"{key}_{option.get('value', i)}",
                use_container_width=True
            ):
                selected = option.get('value')
    
    return selected


def render_range_filter(
    label: str,
    min_value: float,
    max_value: float,
    default: Tuple[float, float] = None,
    step: float = 1.0,
    key: str = "range_filter"
) -> Tuple[float, float]:
    """
    Render a range slider filter.
    
    Args:
        label: Filter label
        min_value: Minimum value
        max_value: Maximum value
        default: Default range (min, max)
        step: Slider step
        key: Unique key
    
    Returns:
        Tuple of (min_selected, max_selected)
    """
    
    default_range = default or (min_value, max_value)
    
    values = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_range,
        step=step,
        key=key
    )
    
    return values


def get_active_filters() -> Dict[str, Any]:
    """
    Get all active (non-default) filters.
    
    Returns:
        Dictionary of active filter values
    """
    
    filters = {}
    
    study = get_filter('study')
    if study and study != "All Studies":
        filters['study'] = study
    
    site = get_filter('site')
    if site and site != "All Sites":
        filters['site'] = site
    
    region = get_filter('region')
    if region and region != "All Regions":
        filters['region'] = region
    
    status = get_filter('status')
    if status and status != "All Statuses":
        filters['status'] = status
    
    search = get_filter('search')
    if search:
        filters['search'] = search
    
    date_range = get_filter('date_range')
    if date_range:
        filters['date_range'] = date_range
    
    return filters


def render_active_filter_tags():
    """Render tags showing active filters with remove buttons."""
    
    filters = get_active_filters()
    
    if not filters:
        return
    
    st.markdown("**Active Filters:**")
    
    cols = st.columns(len(filters) + 1)
    
    for i, (key, value) in enumerate(filters.items()):
        with cols[i]:
            display_value = str(value)
            if len(display_value) > 20:
                display_value = display_value[:17] + "..."
            
            if st.button(f"✕ {key}: {display_value}", key=f"remove_filter_{key}"):
                if key == 'study':
                    set_filter('study', 'All Studies')
                elif key == 'site':
                    set_filter('site', 'All Sites')
                elif key == 'region':
                    set_filter('region', 'All Regions')
                elif key == 'status':
                    set_filter('status', 'All Statuses')
                else:
                    set_filter(key, None)
                st.rerun()
    
    with cols[-1]:
        if st.button("Clear All", key="clear_all_filters"):
            clear_filters()
            st.rerun()