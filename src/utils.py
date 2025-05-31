"""
Utility functions and helpers for the LangChain MCP Client.

This module contains common utility functions, async helpers,
and other shared functionality.
"""

import asyncio
import json
import datetime
import contextvars
from typing import Any, Dict, List
import streamlit as st


def run_async(coro):
    """
    Run an async function with improved context management and timeout protection.
    
    This function handles different scenarios:
    1. When there's no event loop running (first call)
    2. When there's already an event loop running (nested calls)
    3. When running in different thread contexts
    """
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is None:
            # No event loop running - create one and run the coroutine
            # Add a timeout to prevent infinite hanging
            future = asyncio.wait_for(coro, timeout=600.0)  # 10 minute max timeout
            return asyncio.run(future)
        else:
            # We're in an async context, but need to handle it carefully
            # Check if we have a stored event loop
            if hasattr(st.session_state, 'loop') and st.session_state.loop:
                # We're in an async context - use asyncio.run with timeout
                return _run_with_timeout_and_new_loop(coro, timeout=600.0)
            else:
                # Create new event loop in thread
                return _run_with_timeout_and_new_loop(coro, timeout=600.0)
                
    except (RuntimeError, asyncio.TimeoutError) as e:
        # Handle specific timeout and context errors
        if "TimeoutError" in str(type(e).__name__) or "timeout" in str(e).lower():
            raise TimeoutError("Operation timed out after 10 minutes") from e
        else:
            # Context conflict - use alternative approach
            return _run_with_timeout_and_new_loop(coro, timeout=600.0)
    except Exception as e:
        # Fallback: always use new thread
        return _run_with_timeout_and_new_loop(coro, timeout=600.0)


def _run_with_timeout_and_new_loop(coro, timeout: float = 600.0):
    """Create a new event loop and run the coroutine with timeout."""
    import threading
    import queue
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    completed = threading.Event()
    
    def run_in_thread():
        try:
            # Create new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            
            # Run with timeout
            future = asyncio.wait_for(coro, timeout=timeout)
            result = new_loop.run_until_complete(future)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
        finally:
            # Clean up the loop
            new_loop.close()
            completed.set()
    
    # Start thread
    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for completion with timeout
    if completed.wait(timeout + 10):  # Extra 10 seconds for thread overhead
        if not result_queue.empty():
            return result_queue.get()
        elif not exception_queue.empty():
            raise exception_queue.get()
        else:
            raise RuntimeError("Thread completed but no result or exception found")
    else:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


def initialize_session_state():
    """Initialize session state variables with improved async handling."""
    # Initialize basic session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'tools' not in st.session_state:
        st.session_state.tools = []
    
    if 'tool_executions' not in st.session_state:
        st.session_state.tool_executions = []
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'client' not in st.session_state:
        st.session_state.client = None
    
    if 'checkpointer' not in st.session_state:
        st.session_state.checkpointer = None
    
    if 'servers' not in st.session_state:
        st.session_state.servers = {}
    
    # Initialize event loop with better error handling
    if 'loop' not in st.session_state:
        try:
            # Try to get the current loop
            st.session_state.loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no loop exists, create a new one
            try:
                st.session_state.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(st.session_state.loop)
            except Exception as e:
                st.warning(f"Could not create event loop: {str(e)}. Will create on-demand loops.")
                st.session_state.loop = None


def reset_connection_state():
    """Reset connection-related session state."""
    st.session_state.agent = None
    st.session_state.client = None
    st.session_state.tools = []
    st.session_state.checkpointer = None


def create_download_data(data: Dict, prefix: str = "export") -> tuple[str, str]:
    """Create downloadable JSON data."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    json_str = json.dumps(data, indent=2)
    return json_str, filename


def is_valid_thread_id(thread_id: str) -> bool:
    """Validate thread ID format."""
    if not thread_id or not isinstance(thread_id, str):
        return False
    
    # Basic validation - alphanumeric, underscores, hyphens
    import re
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', thread_id)) and len(thread_id) <= 100


def format_error_message(error: Exception) -> str:
    """Format error message for display."""
    error_msg = str(error)
    
    # Handle common MCP/SSE errors
    if "cannot enter context" in error_msg or "already entered" in error_msg:
        return "Connection context conflict. This often happens with external MCP servers. The connection has been retried with a new context."
    elif "All connection attempts failed" in error_msg:
        return "Failed to connect to the MCP server. Please check that the server is running and accessible."
    elif "timeout" in error_msg.lower():
        return "Connection timed out. The MCP server may be overloaded or unreachable."
    else:
        return error_msg


def safe_async_call(coro, error_message: str = "Async operation failed", timeout: float = 600.0):
    """
    Safely call an async function with improved error handling and timeout.
    
    Args:
        coro: The coroutine to execute
        error_message: Custom error message prefix
        timeout: Maximum time to wait in seconds
    
    Returns:
        Result of the coroutine or None if failed
    """
    try:
        # Add timeout wrapper to the coroutine
        timeout_coro = asyncio.wait_for(coro, timeout=timeout)
        return run_async(timeout_coro)
    except asyncio.TimeoutError:
        st.error(f"{error_message}: Operation timed out after {timeout} seconds")
        st.info("💡 The server may be overloaded or unreachable. Try again or check the server status.")
        return None
    except Exception as e:
        formatted_error = format_error_message(e)
        st.error(f"{error_message}: {formatted_error}")
        
        # Show additional context for debugging
        if "cannot enter context" in str(e) or "already entered" in str(e):
            st.info("🔄 Context conflict detected - this is handled automatically")
        
        return None


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


def count_tokens_roughly(text: str) -> int:
    """
    Roughly estimate token count (4 characters per token).
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


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


def model_supports_tools(model_name: str) -> bool:
    """
    Check if a model supports tool calling.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if the model supports tools, False otherwise
    """
    # List of known models that don't support tool calling
    non_tool_models = [
        'deepseek-r1',
        # Add other models here as needed
    ]
    
    model_name_lower = model_name.lower()
    for non_tool_model in non_tool_models:
        if non_tool_model in model_name_lower:
            return False
    
    return True 