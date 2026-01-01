# dashboard/pages/__init__.py
"""
Dashboard pages module.
"""

from .executive_overview import render_page as render_executive_overview
from .cra_view import render_page as render_cra_view
from .dm_hub import render_page as render_dm_hub

__all__ = [
    'render_executive_overview',
    'render_cra_view', 
    'render_dm_hub'
]