"""
TRIALPULSE NEXUS 10X - Chart Components
Phase 7.1: Reusable chart components using Plotly
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from dashboard.config.theme import THEME_CONFIG, get_dqi_band_color


def render_dqi_gauge(
    value: float,
    title: str = "Data Quality Index",
    height: int = 250
):
    """
    Render a gauge chart for DQI score.
    
    Args:
        value: DQI value (0-100)
        title: Chart title
        height: Chart height in pixels
    """
    
    color = get_dqi_band_color(value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': THEME_CONFIG.text_primary}},
        number={'font': {'size': 40, 'color': THEME_CONFIG.text_primary}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': THEME_CONFIG.border_color
            },
            'bar': {'color': color},
            'bgcolor': THEME_CONFIG.background,
            'borderwidth': 2,
            'bordercolor': THEME_CONFIG.border_color,
            'steps': [
                {'range': [0, 25], 'color': '#c0392b20'},
                {'range': [25, 50], 'color': '#e74c3c20'},
                {'range': [50, 65], 'color': '#e67e2220'},
                {'range': [65, 75], 'color': '#f39c1220'},
                {'range': [75, 85], 'color': '#f1c40f20'},
                {'range': [85, 95], 'color': '#2ecc7120'},
                {'range': [95, 100], 'color': '#27ae6020'}
            ],
            'threshold': {
                'line': {'color': THEME_CONFIG.danger, 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_trend_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Trend",
    color: Optional[str] = None,
    height: int = 300,
    show_area: bool = True
):
    """
    Render a trend line chart.
    
    Args:
        data: DataFrame with trend data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        color: Line color
        height: Chart height
        show_area: Whether to show area fill
    """
    
    line_color = color or THEME_CONFIG.accent
    
    if show_area:
        fig = go.Figure(go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='lines',
            fill='tozeroy',
            fillcolor=f'{line_color}20',
            line=dict(color=line_color, width=2)
        ))
    else:
        fig = go.Figure(go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='lines+markers',
            line=dict(color=line_color, width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor=THEME_CONFIG.border_color,
            linecolor=THEME_CONFIG.border_color
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=THEME_CONFIG.border_color,
            linecolor=THEME_CONFIG.border_color
        ),
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Bar Chart",
    color_column: Optional[str] = None,
    orientation: str = "v",
    height: int = 300,
    show_values: bool = True
):
    """
    Render a bar chart.
    
    Args:
        data: DataFrame with chart data
        x_column: Column for x-axis (or y for horizontal)
        y_column: Column for y-axis (or x for horizontal)
        title: Chart title
        color_column: Optional column for color coding
        orientation: 'v' for vertical, 'h' for horizontal
        height: Chart height
        show_values: Whether to show values on bars
    """
    
    if orientation == 'h':
        fig = px.bar(
            data,
            x=y_column,
            y=x_column,
            orientation='h',
            color=color_column,
            title=title,
            color_discrete_sequence=[THEME_CONFIG.accent]
        )
    else:
        fig = px.bar(
            data,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            color_discrete_sequence=[THEME_CONFIG.accent]
        )
    
    if show_values:
        fig.update_traces(texttemplate='%{value:,.0f}', textposition='outside')
    
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=color_column is not None,
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(
    data: pd.DataFrame,
    values_column: str,
    names_column: str,
    title: str = "Distribution",
    height: int = 300,
    hole: float = 0.4,
    colors: Optional[List[str]] = None
):
    """
    Render a pie/donut chart.
    
    Args:
        data: DataFrame with chart data
        values_column: Column with values
        names_column: Column with category names
        title: Chart title
        height: Chart height
        hole: Hole size for donut (0 for pie)
        colors: Optional list of colors
    """
    
    color_sequence = colors or [
        THEME_CONFIG.accent,
        THEME_CONFIG.success,
        THEME_CONFIG.warning,
        THEME_CONFIG.danger,
        THEME_CONFIG.info,
        THEME_CONFIG.primary_light
    ]
    
    fig = px.pie(
        data,
        values=values_column,
        names=names_column,
        title=title,
        hole=hole,
        color_discrete_sequence=color_sequence
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: %{value:,.0f} (%{percent})<extra></extra>'
    )
    
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    title: str = "Heatmap",
    height: int = 400,
    color_scale: str = "RdYlGn"
):
    """
    Render a heatmap.
    
    Args:
        data: DataFrame with heatmap data
        x_column: Column for x-axis
        y_column: Column for y-axis
        value_column: Column for cell values
        title: Chart title
        height: Chart height
        color_scale: Plotly color scale name
    """
    
    # Pivot data for heatmap
    pivot_data = data.pivot(index=y_column, columns=x_column, values=value_column)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns.tolist(),
        y=pivot_data.index.tolist(),
        colorscale=color_scale,
        hoverongaps=False,
        hovertemplate='%{x}<br>%{y}<br>Value: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=80, r=20, t=50, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: str = "Scatter Plot",
    height: int = 400,
    trendline: bool = False
):
    """
    Render a scatter plot.
    
    Args:
        data: DataFrame with scatter data
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column for color
        size_column: Optional column for point size
        title: Chart title
        height: Chart height
        trendline: Whether to show trendline
    """
    
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        title=title,
        trendline="ols" if trendline else None,
        color_discrete_sequence=[THEME_CONFIG.accent]
    )
    
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_multi_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = "Multi-Line Chart",
    height: int = 350,
    colors: Optional[List[str]] = None
):
    """
    Render a multi-line chart.
    
    Args:
        data: DataFrame with chart data
        x_column: Column for x-axis
        y_columns: List of columns to plot as lines
        title: Chart title
        height: Chart height
        colors: Optional list of colors for lines
    """
    
    color_sequence = colors or [
        THEME_CONFIG.accent,
        THEME_CONFIG.success,
        THEME_CONFIG.warning,
        THEME_CONFIG.danger,
        THEME_CONFIG.info
    ]
    
    fig = go.Figure()
    
    for i, col in enumerate(y_columns):
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[col],
            name=col,
            mode='lines+markers',
            line=dict(color=color_sequence[i % len(color_sequence)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        font={'color': THEME_CONFIG.text_primary},
        xaxis=dict(showgrid=True, gridcolor=THEME_CONFIG.border_color),
        yaxis=dict(showgrid=True, gridcolor=THEME_CONFIG.border_color)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_funnel_chart(
    data: pd.DataFrame,
    stage_column: str,
    value_column: str,
    title: str = "Funnel",
    height: int = 300
):
    """
    Render a funnel chart.
    
    Args:
        data: DataFrame with funnel data
        stage_column: Column with stage names
        value_column: Column with values
        title: Chart title
        height: Chart height
    """
    
    fig = go.Figure(go.Funnel(
        y=data[stage_column],
        x=data[value_column],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=[
                THEME_CONFIG.accent,
                THEME_CONFIG.accent_light,
                THEME_CONFIG.success,
                THEME_CONFIG.warning
            ]
        )
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=100, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': THEME_CONFIG.text_primary}
    )
    
    st.plotly_chart(fig, use_container_width=True)