"""
Dashboard components module - Dark Theme Edition.
"""

from .sidebar import render_sidebar
from .header import render_header, render_page_header, render_section_header
from .footer import render_footer
from .metrics import render_metric_card, render_metric_row
from .charts import render_dqi_gauge, render_trend_chart
from .tables import render_data_table, render_paginated_table
from .cards import (
    render_info_card, 
    render_action_card, 
    render_stat_card,
    render_alert_card,
    render_progress_card,
    render_timeline_card,
    render_dqi_gauge as render_cards_dqi_gauge,
)
from .filters import render_filter_bar

__all__ = [
    # Layout components
    'render_sidebar',
    'render_header', 
    'render_footer',
    'render_page_header',
    'render_section_header',
    
    # Metric components
    'render_metric_card',
    'render_metric_row',
    
    # Chart components
    'render_dqi_gauge',
    'render_trend_chart',
    
    # Table components
    'render_data_table',
    'render_paginated_table',
    
    # Card components
    'render_info_card',
    'render_action_card',
    'render_stat_card',
    'render_alert_card',
    'render_progress_card',
    'render_timeline_card',
    
    # Filter components
    'render_filter_bar'
]