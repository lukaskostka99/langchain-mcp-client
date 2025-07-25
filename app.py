"""
LangChain MCP Client - Main Application

A Streamlit application for interacting with LangChain MCP (Model Context Protocol) servers.
This refactored version uses modular components for better maintainability.
"""

import streamlit as st
import nest_asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Apply nest_asyncio to allow nested asyncio event loops (needed for Streamlit's execution model)
nest_asyncio.apply()

# Import all our modular components
from src.utils import initialize_session_state
from src.ui_components import render_sidebar
from src.tab_components import (
    render_chat_tab,
    render_test_tools_tab, 
    render_memory_tab,
    render_config_tab,
    render_about_tab,
    display_tool_executions
)


def main():
    """Main application entry point."""
    # Set page configuration
    st.set_page_config(
        page_title="LangChain MCP Client",
        page_icon="logo_transparent.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Set logo
    st.logo("side_logo.png", size="large")

    # Initialize session state with all required variables
    initialize_session_state()

    # Render sidebar with all configuration options
    render_sidebar()

    # Create main tabs
    tab_chat, tab_test, tab_memory, tab_config, tab_about = st.tabs([
        "🗨️ Chat", 
        "🔧 Test Tools", 
        "🧠 Memory", 
        "⚙️ Config",
        "ℹ️ About"
    ])

    # Render tab content
    with tab_chat:
        render_chat_tab()
        display_tool_executions()
    
    with tab_test:
        render_test_tools_tab()
    
    with tab_memory:
        render_memory_tab()
    
    with tab_config:
        render_config_tab()
    
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main() 