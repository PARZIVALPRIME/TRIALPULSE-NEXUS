"""
TRIALPULSE NEXUS 10X - Formatting Utilities
Phase 7.1: Common formatting functions
"""

from datetime import datetime, timedelta
from typing import Union, Optional
import math


def format_number(value: Union[int, float], decimals: int = 0, suffix: str = "") -> str:
    """
    Format a number with thousands separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        suffix: Optional suffix (e.g., '%', 'hrs')
    
    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    
    if decimals > 0:
        formatted = f"{value:,.{decimals}f}"
    else:
        formatted = f"{int(value):,}"
    
    return f"{formatted}{suffix}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format (0-100 or 0-1)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    
    # Convert from decimal if needed
    if 0 <= value <= 1:
        value = value * 100
    
    return f"{value:.{decimals}f}%"


def format_date(value: Union[datetime, str], format_str: str = "%Y-%m-%d") -> str:
    """
    Format a date value.
    
    Args:
        value: Date to format
        format_str: strftime format string
    
    Returns:
        Formatted date string
    """
    if value is None:
        return "-"
    
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except:
            return value
    
    return value.strftime(format_str)


def format_datetime(value: Union[datetime, str], format_str: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format a datetime value.
    
    Args:
        value: Datetime to format
        format_str: strftime format string
    
    Returns:
        Formatted datetime string
    """
    return format_date(value, format_str)


def format_duration(seconds: Union[int, float], short: bool = False) -> str:
    """
    Format a duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        short: Use short format (1h 30m vs 1 hour 30 minutes)
    
    Returns:
        Formatted duration string
    """
    if seconds is None or seconds < 0:
        return "-"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if short:
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return f"{secs}s"
    else:
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 and hours == 0:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        
        return " ".join(parts) if parts else "0 seconds"


def format_currency(value: float, currency: str = "USD", decimals: int = 0) -> str:
    """
    Format a value as currency.
    
    Args:
        value: Value to format
        currency: Currency code
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    
    symbol = symbols.get(currency, "$")
    
    if decimals > 0:
        return f"{symbol}{value:,.{decimals}f}"
    else:
        return f"{symbol}{int(value):,}"


def format_relative_time(value: Union[datetime, str]) -> str:
    """
    Format a datetime as relative time (e.g., '2 hours ago').
    
    Args:
        value: Datetime value
    
    Returns:
        Relative time string
    """
    if value is None:
        return "-"
    
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except:
            return value
    
    now = datetime.now()
    diff = now - value
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds // 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def format_trend(current: float, previous: float, as_percentage: bool = True) -> tuple:
    """
    Format a trend comparison.
    
    Args:
        current: Current value
        previous: Previous value
        as_percentage: Show as percentage change
    
    Returns:
        Tuple of (formatted_change, direction)
    """
    if previous == 0:
        return ("N/A", "neutral")
    
    change = current - previous
    
    if as_percentage:
        pct_change = (change / previous) * 100
        formatted = f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}%"
    else:
        formatted = f"{'+' if change >= 0 else ''}{change:,.0f}"
    
    if change > 0:
        direction = "positive"
    elif change < 0:
        direction = "negative"
    else:
        direction = "neutral"
    
    return (formatted, direction)


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if text is None:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix