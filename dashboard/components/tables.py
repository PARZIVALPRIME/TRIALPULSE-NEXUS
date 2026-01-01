"""
TRIALPULSE NEXUS 10X - Table Components
Phase 7.1: Reusable table components
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import math

from dashboard.config.theme import THEME_CONFIG, get_priority_color, get_status_color
from dashboard.config.session import get_preference


def render_data_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
    column_config: Optional[Dict[str, Any]] = None,
    hide_index: bool = True,
    height: Optional[int] = None,
    use_container_width: bool = True,
    selection_mode: Optional[str] = None,
    on_select: Optional[Callable] = None
):
    """
    Render a styled data table.
    
    Args:
        data: DataFrame to display
        title: Optional table title
        columns: Columns to display (None for all)
        column_config: Streamlit column configuration
        hide_index: Whether to hide the index
        height: Fixed height in pixels
        use_container_width: Whether to use full width
        selection_mode: 'single-row', 'multi-row', or None
        on_select: Callback for selection
    """
    
    if title:
        st.markdown(f"### {title}")
    
    # Filter columns if specified
    display_data = data[columns] if columns else data
    
    # Default column config
    default_config = {
        "patient_key": st.column_config.TextColumn("Patient ID", width="medium"),
        "site_id": st.column_config.TextColumn("Site", width="small"),
        "study_id": st.column_config.TextColumn("Study", width="small"),
        "dqi_score": st.column_config.ProgressColumn(
            "DQI",
            min_value=0,
            max_value=100,
            format="%.1f"
        ),
        "priority": st.column_config.TextColumn("Priority", width="small"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "issue_count": st.column_config.NumberColumn("Issues", width="small"),
        "effort_hours": st.column_config.NumberColumn("Effort (hrs)", format="%.1f"),
    }
    
    # Merge with provided config
    final_config = {**default_config, **(column_config or {})}
    
    # Render table
    if selection_mode:
        selected = st.dataframe(
            display_data,
            column_config=final_config,
            hide_index=hide_index,
            height=height,
            use_container_width=use_container_width,
            on_select="rerun" if on_select else "ignore",
            selection_mode=selection_mode
        )
        
        if on_select and selected:
            on_select(selected)
    else:
        st.dataframe(
            display_data,
            column_config=final_config,
            hide_index=hide_index,
            height=height,
            use_container_width=use_container_width
        )


def render_paginated_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
    column_config: Optional[Dict[str, Any]] = None,
    page_size: Optional[int] = None,
    key: str = "paginated_table"
):
    """
    Render a paginated data table.
    
    Args:
        data: DataFrame to display
        title: Optional table title
        columns: Columns to display
        column_config: Column configuration
        page_size: Items per page (uses preference if None)
        key: Unique key for the component
    """
    
    if title:
        st.markdown(f"### {title}")
    
    # Get page size from preferences if not specified
    items_per_page = page_size or get_preference('items_per_page', 25)
    
    # Calculate pagination
    total_items = len(data)
    total_pages = math.ceil(total_items / items_per_page)
    
    # Initialize page in session state
    page_key = f"{key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    current_page = st.session_state[page_key]
    
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️", key=f"{key}_first", disabled=current_page == 1):
            st.session_state[page_key] = 1
            st.rerun()
    
    with col2:
        if st.button("◀️", key=f"{key}_prev", disabled=current_page == 1):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            Page {current_page} of {total_pages} ({total_items:,} items)
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("▶️", key=f"{key}_next", disabled=current_page == total_pages):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    
    with col5:
        if st.button("⏭️", key=f"{key}_last", disabled=current_page == total_pages):
            st.session_state[page_key] = total_pages
            st.rerun()
    
    # Get current page data
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_data = data.iloc[start_idx:end_idx]
    
    # Render table
    render_data_table(
        page_data,
        columns=columns,
        column_config=column_config,
        hide_index=True
    )


def render_action_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    actions: List[Dict[str, Any]] = None,
    key: str = "action_table"
):
    """
    Render a table with action buttons per row.
    
    Args:
        data: DataFrame to display
        title: Optional table title
        actions: List of action definitions with 'label', 'icon', 'callback'
        key: Unique key for the component
    """
    
    if title:
        st.markdown(f"### {title}")
    
    actions = actions or []
    
    # Create header
    cols = st.columns([3] + [1] * len(actions))
    
    with cols[0]:
        st.markdown("**Item**")
    
    for i, action in enumerate(actions):
        with cols[i + 1]:
            st.markdown(f"**{action.get('label', 'Action')}**")
    
    st.markdown("---")
    
    # Render rows
    for idx, row in data.iterrows():
        cols = st.columns([3] + [1] * len(actions))
        
        with cols[0]:
            st.markdown(f"{row.get('name', row.name)}")
        
        for i, action in enumerate(actions):
            with cols[i + 1]:
                icon = action.get('icon', '▶️')
                if st.button(
                    icon,
                    key=f"{key}_{idx}_{action.get('label', i)}",
                    help=action.get('help', '')
                ):
                    if action.get('callback'):
                        action['callback'](row)


def render_priority_table(
    data: pd.DataFrame,
    priority_column: str = "priority",
    title: Optional[str] = None
):
    """
    Render a table with priority-colored rows.
    
    Args:
        data: DataFrame to display
        priority_column: Column containing priority values
        title: Optional table title
    """
    
    if title:
        st.markdown(f"### {title}")
    
    # Style function for priority
    def style_priority(val):
        color = get_priority_color(str(val))
        return f'background-color: {color}20; color: {color}; font-weight: 600;'
    
    # Apply styling
    styled = data.style.applymap(
        style_priority,
        subset=[priority_column]
    )
    
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_comparison_table(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    compare_columns: List[str] = None,
    title: Optional[str] = None
):
    """
    Render a side-by-side comparison table.
    
    Args:
        data1: First DataFrame
        data2: Second DataFrame
        label1: Label for first dataset
        label2: Label for second dataset
        compare_columns: Columns to compare
        title: Optional table title
    """
    
    if title:
        st.markdown(f"### {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{label1}**")
        columns = compare_columns or list(data1.columns)
        st.dataframe(data1[columns], use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f"**{label2}**")
        columns = compare_columns or list(data2.columns)
        st.dataframe(data2[columns], use_container_width=True, hide_index=True)