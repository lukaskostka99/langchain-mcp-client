"""
Utility functions and helpers for the LangChain MCP Client.

This module contains common utility functions, async helpers,
and other shared functionality.
"""

import asyncio
import json
import datetime
from typing import Any, Dict, List
import streamlit as st


def run_async(coro):
    """Run an async function within the stored event loop."""
    return st.session_state.loop.run_until_complete(coro)


def initialize_session_state():
    """Initialize all session state variables with their default values."""
    defaults = {
        "client": None,
        "agent": None,
        "tools": [],
        "chat_history": [],
        "servers": {},
        "current_tab": "Single Server",
        "tool_executions": [],
        "tool_test_results": [],
        "tool_test_stats": {},
        "memory_enabled": False,
        "memory_type": "Short-term (Session)",
        "thread_id": "default",
        "max_messages": 100,
        "checkpointer": None,
    }
    
    # Initialize event loop if not exists
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)
    
    # Set defaults for any missing session state variables
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_connection_state():
    """Reset all connection-related session state variables."""
    st.session_state.client = None
    st.session_state.agent = None
    st.session_state.tools = []
    # Clear config applied flag when connection is reset
    st.session_state.config_applied = False


def format_timestamp(timestamp=None) -> str:
    """
    Format a timestamp for display.
    
    Args:
        timestamp: datetime object or None for current time
    
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_json_string(json_str: str) -> tuple[bool, Any]:
    """
    Validate if a string is valid JSON.
    
    Args:
        json_str: String to validate
    
    Returns:
        Tuple of (is_valid, parsed_data_or_error_message)
    """
    try:
        parsed = json.loads(json_str)
        return True, parsed
    except json.JSONDecodeError as e:
        return False, str(e)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get_nested(dictionary: dict, keys: List[str], default=None):
    """
    Safely get a nested dictionary value.
    
    Args:
        dictionary: Dictionary to search
        keys: List of nested keys
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    current = dictionary
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def create_download_data(data: Any, filename_prefix: str = "export") -> tuple[str, str]:
    """
    Create downloadable data and filename.
    
    Args:
        data: Data to export
        filename_prefix: Prefix for the filename
    
    Returns:
        Tuple of (json_string, filename)
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.json"
    json_str = json.dumps(data, indent=2, default=str)
    return json_str, filename


def count_tokens_roughly(text: str) -> int:
    """
    Roughly estimate token count (4 characters per token).
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def is_valid_thread_id(thread_id: str) -> bool:
    """
    Validate if a thread ID is valid.
    
    Args:
        thread_id: Thread ID to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not thread_id or not isinstance(thread_id, str):
        return False
    
    # Basic validation - no spaces, not too long
    if len(thread_id) > 100 or ' ' in thread_id:
        return False
    
    return True


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'streamlit_version': st.__version__,
        'timestamp': datetime.datetime.now().isoformat()
    } 