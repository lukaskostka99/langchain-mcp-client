"""
UI components and interface functions for the LangChain MCP Client.

This module contains all the user interface components including
sidebar, tabs, and various UI utility functions.
"""

import streamlit as st
import json
import datetime
import traceback
from typing import Dict, List, Optional

from .database import PersistentStorageManager
from .llm_providers import (
    get_available_providers, get_provider_models, get_default_model,
    requires_api_key, create_llm_model, supports_streaming
)
from .mcp_client import (
    setup_mcp_client, get_tools_from_client, create_single_server_config,
    create_multi_server_config
)
from .agent_manager import (
    create_agent_with_tools, run_agent, run_tool,
    extract_tool_executions_from_response, extract_assistant_response,
    prepare_agent_invocation_config
)
from .memory_tools import create_history_tool, calculate_chat_statistics
from .utils import run_async, reset_connection_state, safe_async_call, format_error_message, model_supports_tools


def render_sidebar():
    """Render the main application sidebar with all configuration options."""
    with st.sidebar:
        st.title("LangChain MCP Client")
        st.divider()
        st.header("Configuration")
        
        # LLM Provider configuration
        llm_config = render_llm_configuration()
        
        # Streaming configuration
        render_streaming_configuration(llm_config)
        
        # Memory configuration
        memory_config = render_memory_configuration()
        
        # MCP Server configuration
        server_config = render_server_configuration(llm_config, memory_config)
        
        # Display available tools
        render_available_tools()


def render_llm_configuration() -> Dict:
    """Render LLM provider configuration section."""
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=get_available_providers(),
        index=0,
        on_change=reset_connection_state
    )
    
    # Store LLM provider in session state for Config tab
    st.session_state.llm_provider = llm_provider
    
    # API Key input
    api_key_disabled = not requires_api_key(llm_provider)
    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        help=f"Enter your {llm_provider} API Key",
        disabled=api_key_disabled
    )
    
    # Store API key in session state for Config tab (only if not disabled)
    if not api_key_disabled:
        st.session_state.api_key = api_key
    
    # Model selection
    model_options = get_provider_models(llm_provider)
    default_model = get_default_model(llm_provider)
    default_idx = model_options.index(default_model) if default_model in model_options else 0
    
    model_name = st.selectbox(
        "Model",
        options=model_options,
        index=default_idx,
        on_change=reset_connection_state
    )
    
    # Custom model input for Ollama
    if llm_provider == "Ollama" and model_name == "Other":
        custom_model = st.text_input(
            "Custom Model Name",
            placeholder="Enter custom Ollama model name (e.g. llama3)"
        )
        if custom_model:
            model_name = custom_model
    
    # Store selected model in session state for Config tab
    st.session_state.selected_model = model_name
    
    return {
        "provider": llm_provider,
        "api_key": api_key,
        "model": model_name
    }


def render_streaming_configuration(llm_config: Dict) -> None:
    """Render streaming configuration options."""
    with st.expander("🌊 Streaming Settings", expanded=False):
        provider = llm_config.get("provider", "")
        streaming_supported = supports_streaming(provider) if provider else False
        
        if streaming_supported:
            enable_streaming = st.checkbox(
                "Enable Streaming",
                value=st.session_state.get('enable_streaming', True),
                help="Stream responses token by token for a more interactive experience"
            )
            st.session_state.enable_streaming = enable_streaming
            
            if enable_streaming:
                st.success("✅ Streaming enabled - responses will appear in real-time")
            else:
                st.info("ℹ️ Streaming disabled - responses will appear all at once")
        else:
            st.session_state.enable_streaming = False
            if provider:
                st.warning(f"⚠️ {provider} doesn't support streaming")
            else:
                st.info("ℹ️ Select a provider to see streaming options")


def render_memory_configuration() -> Dict:
    """Render memory configuration section."""
    st.header("Memory Settings")
    
    memory_enabled = st.checkbox(
        "Enable Conversation Memory",
        value=st.session_state.get('memory_enabled', False),
        help="Enable persistent conversation memory across interactions",
        key="sidebar_memory_enabled"
    )
    st.session_state.memory_enabled = memory_enabled
    
    memory_config = {"enabled": memory_enabled}
    
    if memory_enabled:
        # Memory type selection
        memory_type = st.selectbox(
            "Memory Type",
            options=["Short-term (Session)", "Persistent (Cross-session)"],
            index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
            help="Short-term: Remembers within current session\nPersistent: Remembers across sessions"
        )
        st.session_state.memory_type = memory_type
        memory_config["type"] = memory_type
        
        # Initialize persistent storage if needed
        if memory_type == "Persistent (Cross-session)":
            if 'persistent_storage' not in st.session_state:
                st.session_state.persistent_storage = PersistentStorageManager()
            
            render_persistent_storage_section()
        
        # Thread ID management
        thread_id = st.text_input(
            "Conversation ID",
            value=st.session_state.get('thread_id', 'default'),
            help="Unique identifier for this conversation thread"
        )
        st.session_state.thread_id = thread_id
        memory_config["thread_id"] = thread_id
        
        # Memory management options
        render_memory_management_section()
    
    # Reset connection when memory settings change
    if st.session_state.get('_last_memory_enabled') != memory_enabled:
        reset_connection_state()
        st.session_state._last_memory_enabled = memory_enabled
    
    return memory_config


def render_persistent_storage_section():
    """Render persistent storage configuration and management."""
    with st.expander("💾 Database Settings"):
        if hasattr(st.session_state, 'persistent_storage'):
            db_stats = st.session_state.persistent_storage.get_database_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversations", db_stats.get('conversation_count', 0))
                st.metric("Total Messages", db_stats.get('total_messages', 0))
            with col2:
                st.metric("Database Size", f"{db_stats.get('database_size_mb', 0)} MB")
                st.text(f"Path: {db_stats.get('database_path', 'N/A')}")
            
            # Conversation browser
            render_conversation_browser()


def render_conversation_browser():
    """Render conversation browser for persistent storage."""
    if not hasattr(st.session_state, 'persistent_storage'):
        return
    
    conversations = st.session_state.persistent_storage.list_conversations()
    if conversations:
        st.subheader("Saved Conversations")
        for conv in conversations[:5]:  # Show last 5 conversations
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    display_title = conv.get('title') or conv['thread_id']
                    if len(display_title) > 30:
                        display_title = display_title[:30] + "..."
                    st.write(f"**{display_title}**")
                    last_msg = conv.get('last_message', '')
                    if last_msg and len(last_msg) > 50:
                        last_msg = last_msg[:50] + "..."
                    st.caption(f"{conv.get('message_count', 0)} messages • {last_msg}")
                
                with col2:
                    if st.button("📂 Load", key=f"load_{conv['thread_id']}"):
                        st.session_state.thread_id = conv['thread_id']
                        st.session_state.chat_history = []
                        st.success(f"Loaded conversation: {conv['thread_id']}")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Del", key=f"del_{conv['thread_id']}"):
                        if st.session_state.persistent_storage.delete_conversation(conv['thread_id']):
                            st.success("Conversation deleted")
                            st.rerun()
                
                st.divider()


def render_memory_management_section():
    """Render memory management options."""
    with st.expander("Memory Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Memory"):
                if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
                    try:
                        st.session_state.chat_history = []
                        if hasattr(st.session_state, 'agent') and st.session_state.agent:
                            st.session_state.agent = None
                        st.success("Memory cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing memory: {str(e)}")
        
        with col2:
            max_messages = st.number_input(
                "Max Messages",
                min_value=10,
                max_value=1000,
                value=st.session_state.get('max_messages', 100),
                help="Maximum messages to keep in memory"
            )
            st.session_state.max_messages = max_messages
        
        # Memory status
        if 'chat_history' in st.session_state:
            current_messages = len(st.session_state.chat_history)
            st.info(f"Current conversation: {current_messages} messages")
        
        # Persistent storage actions
        render_persistent_storage_actions()


def render_persistent_storage_actions():
    """Render persistent storage action buttons."""
    memory_type = st.session_state.get('memory_type', '')
    if (memory_type == "Persistent (Cross-session)" and 
        hasattr(st.session_state, 'persistent_storage')):
        
        st.subheader("Persistent Storage Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Conversation"):
                if st.session_state.chat_history:
                    # Generate a title from the first user message
                    title = None
                    for msg in st.session_state.chat_history[:3]:
                        if msg.get('role') == 'user':
                            title = msg.get('content', '')[:50] + "..." if len(msg.get('content', '')) > 50 else msg.get('content', '')
                            break
                    
                    thread_id = st.session_state.get('thread_id', 'default')
                    st.session_state.persistent_storage.update_conversation_metadata(
                        thread_id=thread_id,
                        title=title,
                        message_count=len(st.session_state.chat_history),
                        last_message=st.session_state.chat_history[-1].get('content', '') if st.session_state.chat_history else ''
                    )
                    st.success("Conversation metadata saved!")
                else:
                    st.warning("No conversation to save")
        
        with col2:
            if st.button("📤 Export Conversation"):
                thread_id = st.session_state.get('thread_id', 'default')
                export_data = st.session_state.persistent_storage.export_conversation(thread_id)
                if export_data:
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📁 Download Export",
                        data=json_str,
                        file_name=f"conversation_{thread_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("Failed to export conversation")


def render_server_configuration(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render MCP server configuration section."""
    st.header("MCP Server Configuration")
    
    server_mode = st.radio(
        "Server Mode",
        options=["Single Server", "Multiple Servers", "No MCP Server (Chat Only)"],
        index=0,
        on_change=lambda: setattr(st.session_state, "current_tab", server_mode)
    )
    
    if server_mode == "Single Server":
        return render_single_server_config(llm_config, memory_config)
    elif server_mode == "Multiple Servers":
        return render_multiple_servers_config(llm_config, memory_config)
    else:  # Chat-only mode
        return render_chat_only_config(llm_config, memory_config)


def render_single_server_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render single server configuration."""
    server_url = st.text_input(
        "MCP Server URL",
        value="http://localhost:8000/sse",
        help="Enter the URL of your MCP server (SSE endpoint)"
    )
    
    if st.button("Connect to MCP Server"):
        return handle_single_server_connection(llm_config, memory_config, server_url)
    
    return {"mode": "single", "connected": False}


def render_multiple_servers_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render multiple servers configuration."""
    st.subheader("Server Management")
    
    # Server input
    server_name = st.text_input(
        "Server Name",
        value="",
        help="Enter a unique name for this server (e.g., 'weather', 'math')"
    )
    
    server_url = st.text_input(
        "Server URL",
        value="http://localhost:8000/sse",
        help="Enter the URL of the MCP server (SSE endpoint)"
    )
    
    # Add server button
    if st.button("Add Server"):
        handle_add_server(server_name, server_url)
    
    # Display configured servers
    render_configured_servers()
    
    # Connect to all servers
    if st.button("Connect to All Servers"):
        return handle_multiple_servers_connection(llm_config, memory_config)
    
    return {"mode": "multiple", "connected": False}


def render_chat_only_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render chat-only mode configuration."""
    st.subheader("Direct Chat Mode")
    st.info("💬 This mode provides a direct chat interface with the LLM without any MCP tools.")
    
    if st.button("Start Chat Agent", type="primary"):
        return handle_chat_only_connection(llm_config, memory_config)
    
    # Information about chat-only mode
    with st.expander("ℹ️ About Chat-Only Mode"):
        st.markdown("""
        **Chat-Only Mode** provides a direct interface to the selected LLM without any MCP server tools.
        
        **Features:**
        - Direct conversation with the LLM
        - Memory support (if enabled)
        - No external tool dependencies
        - Faster setup and response times
        
        **Use Cases:**
        - General conversation and Q&A
        - Creative writing and brainstorming
        - Learning and explanations
        - Code review and discussion (without execution)
        
        **Note:** In this mode, the agent cannot perform external actions like web searches, file operations, or API calls that would normally be available through MCP tools.
        """)
    
    return {"mode": "chat_only", "connected": False}


def handle_single_server_connection(llm_config: Dict, memory_config: Dict, server_url: str) -> Dict:
    """Handle single server connection logic with improved error handling."""
    if not llm_config["api_key"] and requires_api_key(llm_config["provider"]):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "single", "connected": False}
    elif not server_url:
        st.error("Please enter a valid MCP Server URL")
        return {"mode": "single", "connected": False}
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        with st.spinner("Connecting to MCP server..."):
            try:
                # Show progress steps
                progress_placeholder = st.empty()
                
                # Step 1: Setup configuration
                progress_placeholder.info("🔧 Setting up server configuration...")
                server_config = create_single_server_config(
                    server_url, 
                    timeout=60,  # 1 minute for initial connection
                    sse_read_timeout=300  # 5 minutes for SSE operations
                )
                
                # Step 2: Initialize client
                progress_placeholder.info("🌐 Initializing MCP client...")
                client = safe_async_call(
                    setup_mcp_client(server_config),
                    "Failed to initialize MCP client",
                    timeout=60.0  # 1 minute timeout for client setup
                )
                
                if client is None:
                    progress_placeholder.error("❌ Failed to initialize MCP client")
                    return {"mode": "single", "connected": False}
                
                st.session_state.client = client
                
                # Step 3: Get tools
                progress_placeholder.info("🔍 Retrieving tools from server...")
                tools = safe_async_call(
                    get_tools_from_client(st.session_state.client),
                    "Failed to retrieve tools from MCP server",
                    timeout=600.0  # 10 minutes timeout for tool retrieval
                )
                
                if tools is None:
                    progress_placeholder.error("❌ Failed to retrieve tools")
                    return {"mode": "single", "connected": False}
                
                st.session_state.tools = tools
                
                # Step 4: Create agent
                progress_placeholder.info("🤖 Creating and configuring agent...")
                success = create_and_configure_agent(llm_config, memory_config, st.session_state.tools)
                
                if success:
                    progress_placeholder.empty()  # Clear progress messages
                    st.success(f"✅ Connected to MCP server! Found {len(st.session_state.tools)} tools.")
                    with st.expander("🔧 Connection Details"):
                        st.write(f"**Server URL:** {server_url}")
                        st.write(f"**Tools found:** {len(st.session_state.tools)}")
                        st.write(f"**Connection timeout:** 1 minute")
                        st.write(f"**SSE read timeout:** 5 minutes")
                        if st.session_state.tools:
                            st.write(f"**Available tools:**")
                            for tool in st.session_state.tools:
                                st.write(f"  • {tool.name}")
                    return {"mode": "single", "connected": True}
                else:
                    progress_placeholder.error("❌ Failed to configure agent")
                    return {"mode": "single", "connected": False}
                    
            except Exception as e:
                formatted_error = format_error_message(e)
                st.error(f"❌ Error connecting to MCP server: {formatted_error}")
                
                # Show additional troubleshooting info
                with st.expander("🔍 Troubleshooting"):
                    st.write("**Common solutions:**")
                    st.write("• Check that the MCP server is running and accessible")
                    st.write("• Verify the server URL is correct")
                    st.write("• Try refreshing the page and reconnecting")
                    st.write("• For external servers, ensure they support SSE connections")
                    st.write("• Check if the server has rate limiting or requires authentication")
                    
                    st.write("**Technical details:**")
                    st.code(traceback.format_exc(), language="python")
                
                return {"mode": "single", "connected": False}


def handle_multiple_servers_connection(llm_config: Dict, memory_config: Dict) -> Dict:
    """Handle multiple servers connection logic with improved error handling."""
    if not llm_config["api_key"] and requires_api_key(llm_config["provider"]):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "multiple", "connected": False}
    elif not st.session_state.servers:
        st.error("Please add at least one server")
        return {"mode": "multiple", "connected": False}
    
    with st.spinner("Connecting to MCP servers..."):
        try:
            # Initialize the MCP client with all servers with context isolation
            client = safe_async_call(
                setup_mcp_client(st.session_state.servers),
                "Failed to initialize MCP client for multiple servers"
            )
            
            if client is None:
                return {"mode": "multiple", "connected": False}
            
            st.session_state.client = client
            
            # Get tools from the client with context isolation
            tools = safe_async_call(
                get_tools_from_client(st.session_state.client),
                "Failed to retrieve tools from MCP servers"
            )
            
            if tools is None:
                return {"mode": "multiple", "connected": False}
            
            st.session_state.tools = tools
            
            # Create and configure agent
            success = create_and_configure_agent(llm_config, memory_config, st.session_state.tools)
            
            if success:
                st.success(f"✅ Connected to {len(st.session_state.servers)} MCP servers! Found {len(st.session_state.tools)} tools.")
                with st.expander("🔧 Connection Details"):
                    st.write(f"**Servers connected:** {len(st.session_state.servers)}")
                    for name, config in st.session_state.servers.items():
                        st.write(f"  • {name}: {config['url']}")
                    st.write(f"**Total tools found:** {len(st.session_state.tools)}")
                return {"mode": "multiple", "connected": True}
            else:
                return {"mode": "multiple", "connected": False}
                
        except Exception as e:
            formatted_error = format_error_message(e)
            st.error(f"❌ Error connecting to MCP servers: {formatted_error}")
            
            # Show additional troubleshooting info
            with st.expander("🔍 Troubleshooting"):
                st.write("**Common solutions:**")
                st.write("• Check that all MCP servers are running and accessible")
                st.write("• Verify all server URLs are correct")
                st.write("• Try connecting to servers individually first")
                st.write("• For external servers, ensure they support SSE connections")
                
                st.write("**Configured servers:**")
                for name, config in st.session_state.servers.items():
                    st.write(f"  • {name}: {config['url']}")
                
                st.write("**Technical details:**")
                st.code(traceback.format_exc(), language="python")
            
            return {"mode": "multiple", "connected": False}


def handle_chat_only_connection(llm_config: Dict, memory_config: Dict) -> Dict:
    """Handle chat-only mode connection logic."""
    if not llm_config["api_key"] and requires_api_key(llm_config["provider"]):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "chat_only", "connected": False}
    
    with st.spinner("Initializing chat agent..."):
        try:
            # Clear any existing MCP tools since we're in chat-only mode
            st.session_state.tools = []
            st.session_state.client = None
            
            # Create and configure agent
            success = create_and_configure_agent(llm_config, memory_config, [])
            
            if success:
                # Check if the model supports tools
                model_name = llm_config.get('model', '')
                supports_tools = model_supports_tools(model_name)
                
                # Determine appropriate success message
                if not supports_tools:
                    if memory_config.get("enabled"):
                        st.success("✅ Chat agent ready! (Memory enabled, but model doesn't support tools)")
                        st.info("ℹ️ This model doesn't support tool calling, so memory will work through conversation history only.")
                    else:
                        st.success("✅ Chat agent ready! (Simple chat mode - no tools, no memory)")
                else:
                    if memory_config.get("enabled"):
                        st.success("✅ Chat agent ready! (Memory enabled with history tool)")
                    else:
                        st.success("✅ Chat agent ready! (No additional tools)")
                return {"mode": "chat_only", "connected": True}
            else:
                return {"mode": "chat_only", "connected": False}
                
        except Exception as e:
            error_message = str(e)
            if "does not support tools" in error_message:
                st.error("❌ This model doesn't support tool calling. The agent has been configured in simple chat mode.")
                st.info("ℹ️ Memory and tool features are disabled for this model, but basic conversation works.")
            else:
                st.error(f"Error initializing chat agent: {error_message}")
            st.code(traceback.format_exc(), language="python")
            return {"mode": "chat_only", "connected": False}


def create_and_configure_agent(llm_config: Dict, memory_config: Dict, mcp_tools: List) -> bool:
    """Create and configure the agent with the given parameters."""
    try:
        # Get configuration parameters from session state
        use_custom_config = st.session_state.get('config_use_custom_settings', False)
        
        if use_custom_config:
            # Use custom configuration parameters
            temperature = st.session_state.get('config_temperature', 0.7)
            max_tokens = st.session_state.get('config_max_tokens')
            timeout = st.session_state.get('config_timeout')
            system_prompt = st.session_state.get('config_system_prompt')
        else:
            # Use default parameters
            temperature = 0.7
            max_tokens = None
            timeout = None
            system_prompt = None
        
        # Create the language model with configuration
        llm = create_llm_model(
            llm_provider=llm_config["provider"], 
            api_key=llm_config["api_key"], 
            model_name=llm_config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            system_prompt=system_prompt
        )
        
        # Get persistent storage if needed
        persistent_storage = None
        if (memory_config.get("enabled") and 
            memory_config.get("type") == "Persistent (Cross-session)" and
            hasattr(st.session_state, 'persistent_storage')):
            persistent_storage = st.session_state.persistent_storage
        
        # Create the agent
        agent, checkpointer = create_agent_with_tools(
            llm=llm,
            mcp_tools=mcp_tools,
            memory_enabled=memory_config.get("enabled", False),
            memory_type=memory_config.get("type", "Short-term (Session)"),
            persistent_storage=persistent_storage
        )
        
        st.session_state.agent = agent
        st.session_state.checkpointer = checkpointer
        
        return True
        
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return False


def handle_add_server(server_name: str, server_url: str):
    """Handle adding a new server to the configuration."""
    if not server_name:
        st.error("Please enter a server name")
    elif not server_url:
        st.error("Please enter a valid server URL")
    elif server_name in st.session_state.servers:
        st.error(f"Server '{server_name}' already exists")
    else:
        st.session_state.servers[server_name] = {
            "transport": "sse",
            "url": server_url,
            "headers": None,
            "timeout": 600,
            "sse_read_timeout": 900
        }
        st.success(f"Added server '{server_name}'")


def render_configured_servers():
    """Render the list of configured servers."""
    if st.session_state.servers:
        st.subheader("Configured Servers")
        for name, config in st.session_state.servers.items():
            with st.expander(f"Server: {name}"):
                st.write(f"**URL:** {config['url']}")
                if st.button(f"Remove {name}", key=f"remove_{name}"):
                    del st.session_state.servers[name]
                    st.rerun()


def render_available_tools():
    """Render the available tools section."""
    if st.session_state.tools or (st.session_state.agent and st.session_state.get('memory_enabled', False)):
        st.header("Available Tools")
        
        # Check if the current model supports tools
        model_name = st.session_state.get('selected_model', '')
        supports_tools = model_supports_tools(model_name)
        
        # Show total tool count including history tool
        mcp_tool_count = len(st.session_state.tools)
        memory_tool_count = 1 if st.session_state.get('memory_enabled', False) and supports_tools else 0
        total_tools = mcp_tool_count + memory_tool_count
        
        if not supports_tools and st.session_state.get('memory_enabled', False):
            st.info("📊 Memory enabled (conversation history only - model doesn't support tool calling)")
            st.warning("⚠️ This model doesn't support tools, so the history tool is not available. Memory works through conversation context only.")
        elif mcp_tool_count > 0 and memory_tool_count > 0:
            st.info(f"📊 {total_tools} tools available ({mcp_tool_count} MCP + {memory_tool_count} memory tool)")
        elif mcp_tool_count > 0:
            st.info(f"📊 {mcp_tool_count} MCP tools available")
        elif memory_tool_count > 0:
            st.info(f"📊 {memory_tool_count} memory tool available")
        else:
            st.info("📊 No tools available (Chat-only mode)")
        
        render_tool_selector()
    elif st.session_state.agent:
        # Agent exists but no tools - show appropriate message
        model_name = st.session_state.get('selected_model', '')
        if not model_supports_tools(model_name):
            st.header("Agent Status")
            st.info("💬 Simple chat mode - this model doesn't support tool calling")
        else:
            st.header("Available Tools")
            st.info("📊 No tools available (Chat-only mode)")


def render_tool_selector():
    """Render the tool selection dropdown and information."""
    # Check if the current model supports tools
    model_name = st.session_state.get('selected_model', '')
    supports_tools = model_supports_tools(model_name)
    
    # Add history tool to the dropdown when memory is enabled AND model supports tools
    tool_options = [tool.name for tool in st.session_state.tools]
    if st.session_state.get('memory_enabled', False) and supports_tools:
        tool_options.append("get_conversation_history (Memory)")
    
    # Only show tool selection if there are tools available
    if tool_options:
        selected_tool_name = st.selectbox(
            "Available Tools",
            options=tool_options,
            index=0 if tool_options else None
        )
        
        if selected_tool_name:
            render_tool_information(selected_tool_name)
    else:
        # No tools available
        if st.session_state.get('memory_enabled', False) and not supports_tools:
            st.info("💬 Memory is enabled but works through conversation context only (no tool interface)")
        else:
            st.info("💬 In Chat-Only mode - no external tools available")


def render_tool_information(selected_tool_name: str):
    """Render detailed information about the selected tool."""
    if selected_tool_name == "get_conversation_history (Memory)":
        st.write("**Description:** Retrieve conversation history from the current session")
        st.write("**Parameters:**")
        st.code("message_type: string (optional) [default: all]")
        st.code("last_n_messages: integer (optional) [default: 10]") 
        st.code("search_query: string (optional)")
        st.info("💡 This tool allows the agent to access its conversation history when memory is enabled.")
    else:
        # Find the selected MCP tool
        selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
        
        if selected_tool:
            # Display tool information
            st.write(f"**Description:** {selected_tool.description}")
            
            # Display parameters if available
            if hasattr(selected_tool, 'args_schema'):
                st.write("**Parameters:**")
                render_tool_parameters(selected_tool)


def render_tool_parameters(tool):
    """Render tool parameters information."""
    # Get schema properties directly from the tool
    schema = getattr(tool, 'args_schema', {})
    if isinstance(schema, dict):
        properties = schema.get('properties', {})
        required = schema.get('required', [])
    else:
        # Handle Pydantic schema
        schema_dict = schema.schema()
        properties = schema_dict.get('properties', {})
        required = schema_dict.get('required', [])

    # Display each parameter with its details
    for param_name, param_info in properties.items():
        # Get parameter details
        param_type = param_info.get('type', 'string')
        param_title = param_info.get('title', param_name)
        param_default = param_info.get('default', None)
        is_required = param_name in required

        # Build parameter description
        param_desc = [
            f"{param_title}:",
            f"{param_type}",
            "(required)" if is_required else "(optional)"
        ]
        
        if param_default is not None:
            param_desc.append(f"[default: {param_default}]")

        # Display parameter info
        st.code(" ".join(param_desc)) 