import streamlit as st
import asyncio
import json
import datetime
import traceback
from typing import Dict, List, Optional, Any
import os
import nest_asyncio
from contextlib import ExitStack

# Apply nest_asyncio to allow nested asyncio event loops (needed for Streamlit's execution model)
nest_asyncio.apply()

# Import langchain and related libraries
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# Import memory components
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from pydantic import BaseModel, Field
import sqlite3
import threading
from pathlib import Path
import aiosqlite

# Helper functions
def run_async(coro):
    """Run an async function within the stored event loop."""
    return st.session_state.loop.run_until_complete(coro)

async def setup_mcp_client(server_config: Dict[str, Dict]) -> MultiServerMCPClient:
    """Initialize a MultiServerMCPClient with the provided server configuration."""
    client = MultiServerMCPClient(server_config)
    return client

async def get_tools_from_client(client: MultiServerMCPClient) -> List[BaseTool]:
    """Get tools from the MCP client."""
    return await client.get_tools()

async def run_agent(agent, message: str) -> Dict:
    """Run the agent with the provided message."""
    return await agent.ainvoke({"messages": message})

async def run_tool(tool, **kwargs):
    """Run a tool with the provided parameters."""
    return await tool.ainvoke(kwargs)

# Database Manager for Persistent Storage
class PersistentStorageManager:
    """Manages SQLite database for persistent conversation storage."""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create conversations metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        thread_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        message_count INTEGER DEFAULT 0,
                        last_message TEXT
                    )
                """)
                
                conn.commit()
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
    
    def get_checkpointer(self):
        """Get a SQLite checkpointer using context manager pattern."""
        try:
            # Use ExitStack to manage the context manager manually
            if not hasattr(self, '_stack'):
                self._stack = ExitStack()
            
            # Enter the context manager and keep it open
            checkpointer = self._stack.enter_context(
                AsyncSqliteSaver.from_conn_string(str(self.db_path))
            )
            
            return checkpointer
        except Exception as e:
            st.error(f"Error creating SQLite checkpointer: {str(e)}")
            # Fallback to in-memory if SQLite fails
            return InMemorySaver()
    
    def close_checkpointer(self):
        """Close the checkpointer context."""
        try:
            if hasattr(self, '_stack'):
                self._stack.close()
                delattr(self, '_stack')
        except Exception as e:
            st.warning(f"Error closing checkpointer: {str(e)}")
    
    async def get_async_checkpointer(self):
        """Get an async SQLite checkpointer."""
        try:
            # Create an async SQLite connection
            conn = await aiosqlite.connect(str(self.db_path))
            checkpointer = AsyncSqliteSaver(conn)
            return checkpointer
        except Exception as e:
            st.error(f"Error creating async SQLite checkpointer: {str(e)}")
            # Fallback to in-memory if SQLite fails
            return InMemorySaver()
    
    def get_checkpointer_context_manager(self):
        """Get a SQLite checkpointer context manager for the given thread."""
        return AsyncSqliteSaver.from_conn_string(str(self.db_path))
    
    def list_conversations(self) -> List[Dict]:
        """List all stored conversations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT thread_id, created_at, updated_at, title, message_count, last_message
                    FROM conversation_metadata
                    ORDER BY updated_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error listing conversations: {str(e)}")
            return []
    
    def update_conversation_metadata(self, thread_id: str, title: str = None, 
                                   message_count: int = None, last_message: str = None):
        """Update conversation metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update conversation metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO conversation_metadata 
                    (thread_id, created_at, updated_at, title, message_count, last_message)
                    VALUES (?, 
                            COALESCE((SELECT created_at FROM conversation_metadata WHERE thread_id = ?), CURRENT_TIMESTAMP),
                            CURRENT_TIMESTAMP, 
                            COALESCE(?, (SELECT title FROM conversation_metadata WHERE thread_id = ?)),
                            COALESCE(?, (SELECT message_count FROM conversation_metadata WHERE thread_id = ?)),
                            COALESCE(?, (SELECT last_message FROM conversation_metadata WHERE thread_id = ?)))
                """, (thread_id, thread_id, title, thread_id, message_count, thread_id, last_message, thread_id))
                
                conn.commit()
        except Exception as e:
            st.error(f"Error updating conversation metadata: {str(e)}")
    
    def delete_conversation(self, thread_id: str):
        """Delete a conversation and its metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete metadata
                cursor.execute("DELETE FROM conversation_metadata WHERE thread_id = ?", (thread_id,))
                
                # Note: LangGraph's SQLite checkpointer handles its own tables
                # We only delete our metadata here
                
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error deleting conversation: {str(e)}")
            return False
    
    def export_conversation(self, thread_id: str) -> Dict:
        """Export a conversation's data."""
        try:
            # Get metadata
            metadata = None
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM conversation_metadata WHERE thread_id = ?", (thread_id,))
                row = cursor.fetchone()
                if row:
                    metadata = dict(row)
            
            # Get checkpoints (this would require access to LangGraph's internal tables)
            # For now, we'll return the metadata and note that checkpoint data is handled by LangGraph
            return {
                'thread_id': thread_id,
                'metadata': metadata,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'note': 'Checkpoint data is managed by LangGraph SQLite checkpointer'
            }
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get conversation count
                cursor.execute("SELECT COUNT(*) FROM conversation_metadata")
                conversation_count = cursor.fetchone()[0]
                
                # Get total messages
                cursor.execute("SELECT SUM(message_count) FROM conversation_metadata")
                total_messages = cursor.fetchone()[0] or 0
                
                # Get database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'conversation_count': conversation_count,
                    'total_messages': total_messages,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'database_path': str(self.db_path)
                }
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            return {}

def create_llm_model(llm_provider: str, api_key: str, model_name: str):
    """Create a language model based on the selected provider."""
    if llm_provider == "OpenAI":
        return ChatOpenAI(
            openai_api_key=api_key,
            model=model_name,
            temperature=0.7,
        )
    elif llm_provider == "Anthropic":
        return ChatAnthropic(
            anthropic_api_key=api_key,
            model=model_name,
            temperature=0.7,
        )
    elif llm_provider == "Google":
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=None,
            max_retries=2,
        )
    elif llm_provider == "Ollama":
        return ChatOllama(
            model=model_name,
            temperature=0.7,
            # other params...
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

def reset_connection_state():
    """Reset all connection-related session state variables."""
    st.session_state.client = None
    st.session_state.agent = None
    st.session_state.tools = []

def sidebar():
    # Sidebar for configuration
    with st.sidebar:
        # Main app layout
        st.title("LangChain MCP Client")
        
        st.divider()

        st.header("Configuration")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "Select LLM Provider",
            options=["OpenAI", "Anthropic", "Google", "Ollama"],
            index=0,
            on_change=reset_connection_state
        )
        no_key = False
        if llm_provider == "Ollama":
            no_key = True
        # API Key input
        api_key = st.text_input(
            f"{llm_provider} API Key",
            type="password",
            help=f"Enter your {llm_provider} API Key",
            disabled=no_key # Disable input if no key is needed
        )
        
        # Model selection based on provider
        if llm_provider == "OpenAI":
            model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            default_model_idx = 0
        elif llm_provider == "Anthropic":  # Anthropic
            model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            default_model_idx = 0
        elif llm_provider == "Google":  # Google
            model_options = ["gemini-2.0-flash-001", "gemini-2.5-pro-exp-03-25"]
            default_model_idx = 0
        elif llm_provider == "Ollama":  # Ollama
            model_options = ["granite3.3:8b", "qwen3:4b", "Other"]
            default_model_idx = 0
        
        model_name = st.selectbox(
            "Model",
            options=model_options,
            index=default_model_idx,
            on_change=reset_connection_state
        )
        
        # Display a text input field if "Other" is selected in Ollama mode
        if llm_provider == "Ollama" and model_name == "Other":
            custom_model = st.text_input(
                "Custom Model Name",
                placeholder="Enter custom Ollama model name (e.g. llama3)"
            )
            if custom_model:
                model_name = custom_model
        
        # Memory Configuration
        st.header("Memory Settings")
        
        memory_enabled = st.checkbox(
            "Enable Conversation Memory",
            value=st.session_state.get('memory_enabled', False),
            help="Enable persistent conversation memory across interactions"
        )
        st.session_state.memory_enabled = memory_enabled
        
        if memory_enabled:
            # Memory type selection
            memory_type = st.selectbox(
                "Memory Type",
                options=["Short-term (Session)", "Persistent (Cross-session)"],
                index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
                help="Short-term: Remembers within current session\nPersistent: Remembers across sessions"
            )
            st.session_state.memory_type = memory_type
            
            # Initialize persistent storage manager if persistent memory is selected
            if memory_type == "Persistent (Cross-session)":
                if 'persistent_storage' not in st.session_state:
                    st.session_state.persistent_storage = PersistentStorageManager()
                
                # Database configuration for persistent storage
                with st.expander("💾 Database Settings"):
                    db_stats = st.session_state.persistent_storage.get_database_stats()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Conversations", db_stats.get('conversation_count', 0))
                        st.metric("Total Messages", db_stats.get('total_messages', 0))
                    with col2:
                        st.metric("Database Size", f"{db_stats.get('database_size_mb', 0)} MB")
                        st.text(f"Path: {db_stats.get('database_path', 'N/A')}")
                    
                    # Conversation browser for persistent storage
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
                                        st.session_state.chat_history = []  # Clear current chat
                                        st.success(f"Loaded conversation: {conv['thread_id']}")
                                        st.rerun()
                                
                                with col3:
                                    if st.button("🗑️ Del", key=f"del_{conv['thread_id']}"):
                                        if st.session_state.persistent_storage.delete_conversation(conv['thread_id']):
                                            st.success("Conversation deleted")
                                            st.rerun()
                                
                                st.divider()
            
            # Thread ID for conversation separation
            thread_id = st.text_input(
                "Conversation ID",
                value=st.session_state.get('thread_id', 'default'),
                help="Unique identifier for this conversation thread"
            )
            st.session_state.thread_id = thread_id
            
            # Memory management options
            with st.expander("Memory Management"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Clear Memory"):
                        if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
                            # Clear the memory for this thread
                            try:
                                # Reset conversation state
                                st.session_state.chat_history = []
                                if hasattr(st.session_state, 'agent') and st.session_state.agent:
                                    # Force recreate agent to reset memory
                                    st.session_state.agent = None
                                st.success("Memory cleared successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error clearing memory: {str(e)}")
                
                with col2:
                    # Memory trimming options
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
                if memory_type == "Persistent (Cross-session)" and hasattr(st.session_state, 'persistent_storage'):
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
        
        # Reset connection when memory settings change
        if st.session_state.get('_last_memory_enabled') != memory_enabled:
            reset_connection_state()
            st.session_state._last_memory_enabled = memory_enabled
        
        # MCP Server configuration
        st.header("MCP Server Configuration")
        
        # Choose between single server or multiple servers
        server_mode = st.radio(
            "Server Mode",
            options=["Single Server", "Multiple Servers"],
            index=0,
            on_change=lambda: setattr(st.session_state, "current_tab", server_mode)
        )
        
        if server_mode == "Single Server":
            server_type = "SSE (Server-Sent Events)"
            
            server_url = st.text_input(
                "MCP Server URL",
                value="http://localhost:8000/sse",
                help="Enter the URL of your MCP server (SSE endpoint)"
            )
            
            # Connect button for single server
            if st.button("Connect to MCP Server"):
                if not api_key and (not llm_provider == "Ollama"):
                    st.error(f"Please enter your {llm_provider} API Key")
                elif not server_url:
                    st.error("Please enter a valid MCP Server URL")
                else:
                    with st.spinner("Connecting to MCP server..."):
                        try:
                            # Setup server configuration
                            server_config = {
                                "default_server": {
                                    "transport": "sse",
                                    "url": server_url,
                                    "headers": None,
                                    "timeout": 600,
                                    "sse_read_timeout": 900
                                }
                            }
                            
                            # Initialize the MCP client
                            st.session_state.client = run_async(setup_mcp_client(server_config))
                            
                            # Get tools from the client
                            st.session_state.tools = run_async(get_tools_from_client(st.session_state.client))
                            
                            # Create the language model
                            llm = create_llm_model(llm_provider, api_key, model_name)
                            
                            # Create checkpointer if memory is enabled
                            checkpointer = None
                            agent_tools = st.session_state.tools.copy()  # Start with MCP tools
                            
                            if st.session_state.get('memory_enabled', False):
                                memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
                                if memory_type == "Persistent (Cross-session)" and hasattr(st.session_state, 'persistent_storage'):
                                    # Use SQLite checkpointer for persistent storage
                                    checkpointer = st.session_state.persistent_storage.get_checkpointer()
                                else:
                                    # Use in-memory checkpointer for short-term storage
                                    checkpointer = InMemorySaver()
                                
                                st.session_state.checkpointer = checkpointer
                                # Add history tool when memory is enabled
                                agent_tools.append(create_history_tool())
                            
                            # Create the agent
                            st.session_state.agent = create_react_agent(llm, agent_tools, checkpointer=checkpointer)
                            
                            st.success(f"Connected to MCP server! Found {len(st.session_state.tools)} tools.")
                        except Exception as e:
                            st.error(f"Error connecting to MCP server: {str(e)}")
                            st.code(traceback.format_exc(), language="python")
        
        else:  # Multiple Servers mode
            # Server management section
            st.subheader("Server Management")
            
            # Server name input
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
            
            # Display configured servers
            if st.session_state.servers:
                st.subheader("Configured Servers")
                for name, config in st.session_state.servers.items():
                    with st.expander(f"Server: {name}"):
                        st.write(f"**URL:** {config['url']}")
                        if st.button(f"Remove {name}", key=f"remove_{name}"):
                            del st.session_state.servers[name]
                            st.rerun()
            
            # Connect to all servers button
            if st.button("Connect to All Servers"):
                if not api_key and (not llm_provider == "Ollama"):
                    st.error(f"Please enter your {llm_provider} API Key")
                elif not st.session_state.servers:
                    st.error("Please add at least one server")
                else:
                    with st.spinner("Connecting to MCP servers..."):
                        try:
                            # Initialize the MCP client with all servers
                            st.session_state.client = run_async(setup_mcp_client(st.session_state.servers))
                            
                            # Get tools from the client
                            st.session_state.tools = run_async(get_tools_from_client(st.session_state.client))
                            
                            # Create the language model
                            llm = create_llm_model(llm_provider, api_key, model_name)
                            
                            # Create checkpointer if memory is enabled
                            checkpointer = None
                            agent_tools = st.session_state.tools.copy()  # Start with MCP tools
                            
                            if st.session_state.get('memory_enabled', False):
                                memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
                                if memory_type == "Persistent (Cross-session)" and hasattr(st.session_state, 'persistent_storage'):
                                    # Use SQLite checkpointer for persistent storage
                                    checkpointer = st.session_state.persistent_storage.get_checkpointer()
                                else:
                                    # Use in-memory checkpointer for short-term storage
                                    checkpointer = InMemorySaver()
                                
                                st.session_state.checkpointer = checkpointer
                                # Add history tool when memory is enabled
                                agent_tools.append(create_history_tool())
                            
                            # Create the agent
                            st.session_state.agent = create_react_agent(llm, agent_tools, checkpointer=checkpointer)
                            
                            st.success(f"Connected to {len(st.session_state.servers)} MCP servers! Found {len(st.session_state.tools)} tools.")
                        except Exception as e:
                            st.error(f"Error connecting to MCP servers: {str(e)}")
                            st.code(traceback.format_exc(), language="python")
        
        # Display available tools if connected
        if st.session_state.tools:
            st.header("Available Tools")
            
            # Show total tool count including history tool
            total_tools = len(st.session_state.tools)
            if st.session_state.get('memory_enabled', False):
                total_tools += 1
                st.info(f"📊 {total_tools} tools available ({len(st.session_state.tools)} MCP + 1 memory tool)")
            else:
                st.info(f"📊 {total_tools} MCP tools available")
            
            # Add history tool to the dropdown when memory is enabled
            tool_options = [tool.name for tool in st.session_state.tools]
            if st.session_state.get('memory_enabled', False):
                tool_options.append("get_conversation_history (Memory)")
            
            # Tool selection dropdown
            selected_tool_name = st.selectbox(
                "Available Tools",
                options=tool_options,
                index=0 if tool_options else None
            )
            
            if selected_tool_name:
                # Handle memory tool separately
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
                            
                            # Get schema properties directly from the tool
                            schema = getattr(selected_tool, 'args_schema', {})
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

# Function to display tool execution details
def display_tool_executions():
    if st.session_state.tool_executions:
        with st.expander("Tool Execution History", expanded=False):
            for i, exec_record in enumerate(st.session_state.tool_executions):
                st.markdown(f"### Execution #{i+1}: `{exec_record['tool_name']}`")
                st.markdown(f"**Input:** ```json{json.dumps(exec_record['input'])}```")
                st.markdown(f"**Output:** ```{exec_record['output']}```")
                st.markdown(f"**Time:** {exec_record['timestamp']}")
                st.divider()

def tab_chat():
    # Main chat interface
    st.header("Chat with Agent")

    # Connection and memory status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        connection_status = st.empty()
        if st.session_state.client is not None:
            connection_status.success("📶 Connected to MCP server(s)")
        else:
            connection_status.warning("⚠️ Not connected to any MCP server")
    
    with col2:
        memory_status = st.empty()
        if st.session_state.get('memory_enabled', False):
            thread_id = st.session_state.get('thread_id', 'default')
            memory_status.info(f"🧠 Memory enabled (Thread: {thread_id})")
        else:
            memory_status.info("🧠 Memory disabled")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        if message["role"] == "assistant" and "tool" in message and message["tool"]:
            st.code(message['tool'])
        if message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Check if agent is set up
        if st.session_state.agent is None:
            st.error("Please connect to an MCP server first")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Prepare agent invocation config
                        config = {}
                        if st.session_state.get('memory_enabled', False):
                            thread_id = st.session_state.get('thread_id', 'default')
                            config = {"configurable": {"thread_id": thread_id}}
                        
                        # Run the agent with or without memory
                        if config:
                            response = run_async(st.session_state.agent.ainvoke({"messages": user_input}, config))
                        else:
                            response = run_async(run_agent(st.session_state.agent, user_input))
                        
                        tool_outputs = []  # Store multiple tool outputs
                        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Extract tool executions if available
                        if "messages" in response:
                            # Get the last few messages from the response (the new interaction)
                            # Skip the user message and look for the assistant's response with tools
                            assistant_messages = [
                                msg for msg in response["messages"] 
                                if hasattr(msg, 'tool_calls') and msg.tool_calls
                            ]
                            
                            # Only process tool calls from the most recent assistant message
                            if assistant_messages:
                                latest_assistant_msg = assistant_messages[-1]  # Get the latest one
                                
                                for tool_call in latest_assistant_msg.tool_calls:
                                    # Find corresponding ToolMessage
                                    tool_output = next(
                                        (m.content for m in response["messages"] 
                                         if isinstance(m, ToolMessage) and 
                                         hasattr(m, 'tool_call_id') and
                                         m.tool_call_id == tool_call['id']),
                                        None
                                    )
                                    if tool_output:
                                        # Handle different tool types
                                        if tool_call['name'] != 'get_conversation_history':
                                            tool_outputs.append(tool_output)  # Add to list
                                            st.session_state.tool_executions.append({
                                                "tool_name": tool_call['name'],
                                                "input": tool_call['args'],
                                                "output": tool_output,
                                                "timestamp": current_timestamp
                                            })
                                        else:
                                            # For conversation history, show actual content but use summary in tool_outputs
                                            tool_outputs.append("📋 Conversation history retrieved")
                                            st.session_state.tool_executions.append({
                                                "tool_name": tool_call['name'],
                                                "input": tool_call['args'],
                                                "output": tool_output,  # Show actual history content
                                                "timestamp": current_timestamp
                                            })

                        # Extract and display the response
                        output = ""
                        
                        if "messages" in response:
                            for msg in response["messages"]:
                                if isinstance(msg, HumanMessage):
                                    continue  # Skip human messages
                                elif hasattr(msg, 'name') and msg.name:  # ToolMessage
                                    # Check if this is the conversation history tool
                                    if hasattr(msg, 'tool_call_id'):
                                        # Find the corresponding tool call to get the tool name
                                        tool_call = next(
                                            (tc for m in response["messages"] 
                                             if hasattr(m, 'tool_calls') and m.tool_calls
                                             for tc in m.tool_calls 
                                             if tc['id'] == msg.tool_call_id),
                                            None
                                        )
                                        if tool_call and tool_call['name'] == 'get_conversation_history':
                                            # Display conversation history in a cleaner way
                                            st.markdown(msg.content)
                                        else:
                                            # Display other tool outputs as code
                                            st.code(msg.content)
                                    else:
                                        st.code(msg.content)
                                elif isinstance(msg, AIMessage):  # AIMessage - more explicit check
                                    if hasattr(msg, "content") and msg.content:
                                        # Clean the content and remove any tool call artifacts
                                        content = str(msg.content).strip()
                                        # Remove tool call artifacts that might appear in the content
                                        if content.startswith('<|tool_call|>') and content.endswith('[]'):
                                            continue  # Skip malformed tool call artifacts
                                        if content and content != "<|tool_call|>[]":
                                            output = content
                                            st.write(output)
                        
                        # Fallback: if no output was extracted, try to get a direct response
                        if not output and "messages" in response and response["messages"]:
                            # Look for the last AIMessage content
                            for msg in reversed(response["messages"]):
                                if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                    content = str(msg.content).strip()
                                    if content and content != "<|tool_call|>[]" and not content.startswith('<|tool_call|>'):
                                        output = content
                                        st.write(output)
                                        break
                        
                        # If still no output, there might be an issue with the response
                        if not output:
                            st.warning("The agent didn't provide a clear response. This might be a processing issue.")
                            # Show debug info in development
                            with st.expander("Debug: Raw Response"):
                                st.json(response)

                        # Add assistant message to chat history for display purposes
                        # Always add to chat_history for UI display, regardless of memory settings
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "tool": "\n".join(tool_outputs) if tool_outputs else None,  # Join all tool outputs
                            "content": output
                        })
                        
                        # Auto-save to persistent storage if enabled
                        if (st.session_state.get('memory_enabled', False) and 
                            st.session_state.get('memory_type') == "Persistent (Cross-session)" and
                            hasattr(st.session_state, 'persistent_storage')):
                            
                            try:
                                # Auto-save conversation metadata
                                thread_id = st.session_state.get('thread_id', 'default')
                                # Generate title from first user message if not already set
                                title = None
                                for msg in st.session_state.chat_history[:3]:
                                    if msg.get('role') == 'user':
                                        content = msg.get('content', '')
                                        title = content[:50] + "..." if len(content) > 50 else content
                                        break
                                
                                st.session_state.persistent_storage.update_conversation_metadata(
                                    thread_id=thread_id,
                                    title=title,
                                    message_count=len(st.session_state.chat_history),
                                    last_message=output[:100] + "..." if len(output) > 100 else output
                                )
                            except Exception as e:
                                # Don't break the chat flow if saving fails
                                st.warning(f"Auto-save failed: {str(e)}")
                        
                        # Apply message trimming if memory is enabled and we have too many messages
                        if st.session_state.get('memory_enabled', False):
                            max_messages = st.session_state.get('max_messages', 100)
                            if len(st.session_state.chat_history) > max_messages:
                                # Trim older messages but keep some context
                                keep_messages = max_messages // 2  # Keep half the limit
                                st.session_state.chat_history = st.session_state.chat_history[-keep_messages:]
                    
                        st.rerun()

                    except Exception as e:
                        error_msg = str(e)
                        
                        # Check for Ollama connection error
                        if "ConnectError: All connection attempts failed" in error_msg:
                            st.error("⚠️ Could not connect to Ollama. Please make sure Ollama is running by executing 'ollama serve' in a terminal.")
                            st.info("To start Ollama, open a terminal/command prompt and run: `ollama serve`")
                        else:
                            st.error(f"Error processing your request: {error_msg}")
                            st.code(traceback.format_exc(), language="python")

def tab_about():
    st.image("logo_transparent.png", width=200)
    st.markdown("""
    ### About
    This application demonstrates a Streamlit interface for LangChain MCP (Model Context Protocol) adapters. 
    It allows you to connect to MCP servers and use their tools with different LLM providers.

    For more information, check out:
    - [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
    - [Model Context Protocol](https://modelcontextprotocol.io/introduction)
                
    ### License
    This project is licensed under the MIT License.
    
    ### Acknowledgements
    - This application is built using [Streamlit](https://streamlit.io/) for the frontend.
    
    ### Developer
    - [LinkedIn](https://www.linkedin.com/in/guinacio/)
    - [Github](https://github.com/guinacio)
    """)

def tab_test_tools():
    """Tool testing interface for individually testing MCP tools."""
    st.header("🔧 Test Tools Individually")
    
    # Check if tools are available
    if not st.session_state.tools:
        st.warning("⚠️ No tools available. Please connect to an MCP server first.")
        st.info("Go to the sidebar to connect to an MCP server, then return to this tab to test tools.")
        return
    
    # Tool selection
    st.subheader("Select Tool to Test")
    
    tool_names = [tool.name for tool in st.session_state.tools]
    selected_tool_name = st.selectbox(
        "Choose a tool:",
        options=tool_names,
        key="test_tool_selector"
    )
    
    if not selected_tool_name:
        return
        
    # Find the selected tool
    selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
    
    if not selected_tool:
        st.error("Tool not found!")
        return
    
    # Display tool information
    st.subheader("Tool Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Name:** {selected_tool.name}")
        st.write(f"**Description:** {selected_tool.description}")
    
    with col2:
        # Tool execution statistics
        tool_stats = st.session_state.get('tool_test_stats', {})
        if selected_tool_name in tool_stats:
            stats = tool_stats[selected_tool_name]
            st.metric("Tests Run", stats.get('count', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
            st.metric("Avg Time", f"{stats.get('avg_time', 0):.2f}s")
    
    # Generate dynamic form based on tool schema
    st.subheader("Tool Parameters")
    
    # Get tool schema
    tool_params = {}
    required_params = []
    
    if hasattr(selected_tool, 'args_schema') and selected_tool.args_schema:
        schema = selected_tool.args_schema
        if hasattr(schema, 'schema'):
            # Pydantic model
            schema_dict = schema.schema()
            properties = schema_dict.get('properties', {})
            required_params = schema_dict.get('required', [])
        else:
            # Dictionary schema
            properties = schema.get('properties', {})
            required_params = schema.get('required', [])
        
        if properties:
            st.write("Fill in the parameters below:")
            
            # Create form inputs for each parameter
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'string')
                param_title = param_info.get('title', param_name)
                param_description = param_info.get('description', '')
                param_default = param_info.get('default', None)
                is_required = param_name in required_params
                
                # Create label with required indicator
                label = f"{param_title}"
                if is_required:
                    label += " *"
                
                # Create appropriate input widget based on type
                if param_type == 'integer':
                    value = st.number_input(
                        label,
                        value=param_default if param_default is not None else 0,
                        step=1,
                        help=param_description,
                        key=f"test_param_{param_name}"
                    )
                elif param_type == 'number':
                    value = st.number_input(
                        label,
                        value=float(param_default) if param_default is not None else 0.0,
                        step=0.1,
                        help=param_description,
                        key=f"test_param_{param_name}"
                    )
                elif param_type == 'boolean':
                    value = st.checkbox(
                        label,
                        value=param_default if param_default is not None else False,
                        help=param_description,
                        key=f"test_param_{param_name}"
                    )
                elif param_type == 'array':
                    # For arrays, use text area and split by lines or commas
                    text_value = st.text_area(
                        label,
                        value="",
                        help=f"{param_description}\n(Enter items separated by new lines)",
                        key=f"test_param_{param_name}"
                    )
                    value = [item.strip() for item in text_value.split('\n') if item.strip()] if text_value else []
                else:
                    # Default to string
                    value = st.text_input(
                        label,
                        value=param_default if param_default is not None else "",
                        help=param_description,
                        key=f"test_param_{param_name}"
                    )
                
                tool_params[param_name] = value
            
            # Validate required parameters
            missing_required = []
            for req_param in required_params:
                if req_param in tool_params:
                    value = tool_params[req_param]
                    if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                        missing_required.append(req_param)
            
            if missing_required:
                st.warning(f"⚠️ Required parameters missing: {', '.join(missing_required)}")
        else:
            st.info("This tool doesn't require any parameters.")
    else:
        st.info("This tool doesn't require any parameters.")
    
    # Test execution section
    st.subheader("Execute Tool")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Run Tool", type="primary", disabled=len(missing_required) > 0 if 'missing_required' in locals() else False):
            # Execute the tool
            with st.spinner("Executing tool..."):
                start_time = datetime.datetime.now()
                
                try:
                    # Run the tool with parameters
                    result = run_async(run_tool(selected_tool, **tool_params))
                    end_time = datetime.datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    # Store result in session state
                    if 'tool_test_results' not in st.session_state:
                        st.session_state.tool_test_results = []
                    
                    test_result = {
                        'tool_name': selected_tool_name,
                        'parameters': tool_params.copy(),
                        'result': result,
                        'success': True,
                        'execution_time': execution_time,
                        'timestamp': start_time.isoformat()
                    }
                    
                    st.session_state.tool_test_results.insert(0, test_result)  # Insert at beginning
                    
                    # Update statistics
                    if 'tool_test_stats' not in st.session_state:
                        st.session_state.tool_test_stats = {}
                    
                    if selected_tool_name not in st.session_state.tool_test_stats:
                        st.session_state.tool_test_stats[selected_tool_name] = {
                            'count': 0,
                            'successes': 0,
                            'total_time': 0.0
                        }
                    
                    stats = st.session_state.tool_test_stats[selected_tool_name]
                    stats['count'] += 1
                    stats['successes'] += 1
                    stats['total_time'] += execution_time
                    stats['success_rate'] = (stats['successes'] / stats['count']) * 100
                    stats['avg_time'] = stats['total_time'] / stats['count']
                    
                    st.success(f"✅ Tool executed successfully in {execution_time:.2f} seconds!")
                    st.rerun()
                    
                except Exception as e:
                    end_time = datetime.datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    # Store error result
                    if 'tool_test_results' not in st.session_state:
                        st.session_state.tool_test_results = []
                    
                    test_result = {
                        'tool_name': selected_tool_name,
                        'parameters': tool_params.copy(),
                        'result': None,
                        'error': str(e),
                        'success': False,
                        'execution_time': execution_time,
                        'timestamp': start_time.isoformat()
                    }
                    
                    st.session_state.tool_test_results.insert(0, test_result)
                    
                    # Update statistics
                    if 'tool_test_stats' not in st.session_state:
                        st.session_state.tool_test_stats = {}
                    
                    if selected_tool_name not in st.session_state.tool_test_stats:
                        st.session_state.tool_test_stats[selected_tool_name] = {
                            'count': 0,
                            'successes': 0,
                            'total_time': 0.0
                        }
                    
                    stats = st.session_state.tool_test_stats[selected_tool_name]
                    stats['count'] += 1
                    stats['total_time'] += execution_time
                    stats['success_rate'] = (stats['successes'] / stats['count']) * 100
                    stats['avg_time'] = stats['total_time'] / stats['count']
                    
                    st.error(f"❌ Tool execution failed: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
    
    with col2:
        if st.button("🗑️ Clear Results"):
            if 'tool_test_results' in st.session_state:
                st.session_state.tool_test_results = []
            if 'tool_test_stats' in st.session_state:
                st.session_state.tool_test_stats = {}
            st.success("Results cleared!")
            st.rerun()
    
    with col3:
        if st.button("📋 Copy Parameters as JSON"):
            params_json = json.dumps(tool_params, indent=2)
            st.code(params_json, language="json")
    
    # Display results
    if 'tool_test_results' in st.session_state and st.session_state.tool_test_results:
        st.subheader("Test Results")
        
        # Filter results for current tool
        current_tool_results = [r for r in st.session_state.tool_test_results if r['tool_name'] == selected_tool_name]
        
        if current_tool_results:
            # Show latest result prominently
            latest_result = current_tool_results[0]
            
            if latest_result['success']:
                st.success("✅ Latest Result")
                with st.expander("View Result Details", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Output:**")
                        if isinstance(latest_result['result'], str):
                            st.text(latest_result['result'])
                        else:
                            st.json(latest_result['result'])
                    with col2:
                        st.write("**Execution Info:**")
                        st.write(f"Time: {latest_result['execution_time']:.2f}s")
                        st.write(f"Timestamp: {latest_result['timestamp']}")
                        
                        if latest_result['parameters']:
                            st.write("**Parameters:**")
                            st.json(latest_result['parameters'])
            else:
                st.error("❌ Latest Result")
                with st.expander("View Error Details", expanded=True):
                    st.write("**Error:**")
                    st.code(latest_result['error'])
                    st.write(f"**Execution Time:** {latest_result['execution_time']:.2f}s")
                    if latest_result['parameters']:
                        st.write("**Parameters:**")
                        st.json(latest_result['parameters'])
            
            # Show history if there are multiple results
            if len(current_tool_results) > 1:
                with st.expander(f"Previous Results ({len(current_tool_results) - 1})"):
                    for i, result in enumerate(current_tool_results[1:], 1):
                        status = "✅" if result['success'] else "❌"
                        time_str = datetime.datetime.fromisoformat(result['timestamp']).strftime("%H:%M:%S")
                        
                        st.write(f"**{status} Test #{i + 1}** - {time_str} ({result['execution_time']:.2f}s)")
                        
                        if result['success']:
                            if isinstance(result['result'], str):
                                st.text(result['result'][:200] + "..." if len(str(result['result'])) > 200 else result['result'])
                            else:
                                st.json(result['result'])
                        else:
                            st.error(f"Error: {result['error']}")
                        
                        st.divider()
        else:
            st.info("No test results for this tool yet.")

    # Overall Testing Summary
    if 'tool_test_results' in st.session_state and st.session_state.tool_test_results:
        st.subheader("Testing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_tests = len(st.session_state.tool_test_results)
            successful_tests = len([r for r in st.session_state.tool_test_results if r['success']])
            overall_success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            
            st.metric("Total Tests", total_tests)
            st.metric("Success Rate", f"{overall_success_rate:.1f}%")
        
        with col2:
            if st.session_state.tool_test_results:
                avg_execution_time = sum(r['execution_time'] for r in st.session_state.tool_test_results) / len(st.session_state.tool_test_results)
                tools_tested = len(set(r['tool_name'] for r in st.session_state.tool_test_results))
                
                st.metric("Avg Execution Time", f"{avg_execution_time:.2f}s")
                st.metric("Tools Tested", tools_tested)
        
        with col3:
            # Export results
            if st.button("📁 Export Test Results"):
                import io
                
                # Prepare export data
                export_data = {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'summary': {
                        'total_tests': total_tests,
                        'successful_tests': successful_tests,
                        'success_rate': overall_success_rate,
                        'avg_execution_time': avg_execution_time,
                        'tools_tested': tools_tested
                    },
                    'detailed_results': st.session_state.tool_test_results,
                    'statistics': st.session_state.tool_test_stats
                }
                
                # Create downloadable JSON
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"tool_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def tab_memory():
    """Memory management interface for conversation history and settings."""
    st.header("🧠 Memory Management")
    
    # Memory status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        memory_enabled = st.session_state.get('memory_enabled', False)
        if memory_enabled:
            st.success("Memory: Enabled")
        else:
            st.info("Memory: Disabled")
    
    with col2:
        if memory_enabled:
            memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
            thread_id = st.session_state.get('thread_id', 'default')
            st.info(f"Type: {memory_type.split()[0]}")
            st.info(f"Thread: {thread_id}")
        else:
            st.info("Type: N/A")
            st.info("Thread: N/A")
    
    with col3:
        chat_length = len(st.session_state.get('chat_history', []))
        st.metric("Messages", chat_length)
    
    st.divider()
    
    # Enhanced memory configuration
    st.subheader("Memory Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quick toggle for memory
        new_memory_enabled = st.toggle(
            "Enable Memory",
            value=st.session_state.get('memory_enabled', False),
            help="Enable or disable conversation memory"
        )
        
        if new_memory_enabled != st.session_state.get('memory_enabled', False):
            st.session_state.memory_enabled = new_memory_enabled
            # Reset agent when memory setting changes
            if 'agent' in st.session_state:
                st.session_state.agent = None
            st.success(f"Memory {'enabled' if new_memory_enabled else 'disabled'}. Please reconnect to MCP server.")
            st.rerun()
        
        # Memory type selection
        if new_memory_enabled:
            new_memory_type = st.selectbox(
                "Storage Type",
                options=["Short-term (Session)", "Persistent (Cross-session)"],
                index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
                help="Choose memory persistence level"
            )
            
            if new_memory_type != st.session_state.get('memory_type', 'Short-term (Session)'):
                st.session_state.memory_type = new_memory_type
                # Reset agent when memory type changes
                if 'agent' in st.session_state:
                    st.session_state.agent = None
                st.info(f"Memory type changed to: {new_memory_type}. Please reconnect to MCP server.")
                st.rerun()
        
        # Thread ID management
        if new_memory_enabled:
            new_thread_id = st.text_input(
                "Thread ID",
                value=st.session_state.get('thread_id', 'default'),
                help="Unique identifier for conversation thread"
            )
            
            if new_thread_id != st.session_state.get('thread_id', 'default'):
                st.session_state.thread_id = new_thread_id
                st.session_state.chat_history = []  # Clear chat when switching threads
                st.info(f"Switched to thread: {new_thread_id}")
                st.rerun()
    
    with col2:
        if new_memory_enabled:
            # Memory limits
            max_messages = st.number_input(
                "Maximum Messages",
                min_value=10,
                max_value=1000,
                value=st.session_state.get('max_messages', 100),
                help="Maximum number of messages to keep in memory"
            )
            st.session_state.max_messages = max_messages
            
            # Show persistent storage info if enabled
            memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
            if memory_type == "Persistent (Cross-session)":
                if 'persistent_storage' not in st.session_state:
                    st.session_state.persistent_storage = PersistentStorageManager()
                
                # Database statistics
                db_stats = st.session_state.persistent_storage.get_database_stats()
                st.info("📊 Database Statistics")
                st.text(f"Conversations: {db_stats.get('conversation_count', 0)}")
                st.text(f"Total Messages: {db_stats.get('total_messages', 0)}")
                st.text(f"Size: {db_stats.get('database_size_mb', 0)} MB")
    
    # Database Management (for persistent storage)
    if (new_memory_enabled and 
        st.session_state.get('memory_type') == "Persistent (Cross-session)" and 
        hasattr(st.session_state, 'persistent_storage')):
        
        st.subheader("📚 Conversation Database")
        
        # Database actions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Current", type="primary"):
                if st.session_state.chat_history:
                    # Generate a title from the first user message
                    title = None
                    for msg in st.session_state.chat_history[:3]:
                        if msg.get('role') == 'user':
                            content = msg.get('content', '')
                            title = content[:50] + "..." if len(content) > 50 else content
                            break
                    
                    thread_id = st.session_state.get('thread_id', 'default')
                    st.session_state.persistent_storage.update_conversation_metadata(
                        thread_id=thread_id,
                        title=title,
                        message_count=len(st.session_state.chat_history),
                        last_message=st.session_state.chat_history[-1].get('content', '') if st.session_state.chat_history else ''
                    )
                    st.success("Conversation saved to database!")
                    st.rerun()
                else:
                    st.warning("No conversation to save")
        
        with col2:
            if st.button("🔄 Refresh List"):
                st.rerun()
        
        with col3:
            if st.button("📊 Database Stats"):
                stats = st.session_state.persistent_storage.get_database_stats()
                st.json(stats)
        
        with col4:
            if st.button("🗑️ Clear All"):
                if st.checkbox("Confirm deletion", key="confirm_clear_all"):
                    try:
                        # This would require implementing a method to clear all data
                        st.warning("Clear all functionality needs to be implemented")
                    except Exception as e:
                        st.error(f"Error clearing database: {str(e)}")
        
        # Conversation list
        conversations = st.session_state.persistent_storage.list_conversations()
        if conversations:
            st.write(f"**Stored Conversations ({len(conversations)})**")
            
            # Create a more detailed conversation list
            for i, conv in enumerate(conversations):
                with st.expander(f"📄 {conv.get('title', conv['thread_id'])} ({conv.get('message_count', 0)} messages)"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Thread ID:** {conv['thread_id']}")
                        st.write(f"**Created:** {conv.get('created_at', 'Unknown')}")
                        st.write(f"**Updated:** {conv.get('updated_at', 'Unknown')}")
                        st.write(f"**Messages:** {conv.get('message_count', 0)}")
                        if conv.get('last_message'):
                            last_msg = conv['last_message']
                            if len(last_msg) > 100:
                                last_msg = last_msg[:100] + "..."
                            st.write(f"**Last Message:** {last_msg}")
                    
                    with col2:
                        if st.button("📂 Load", key=f"load_detailed_{i}"):
                            st.session_state.thread_id = conv['thread_id']
                            st.session_state.chat_history = []  # Clear current chat
                            st.success(f"Loaded: {conv['thread_id']}")
                            st.rerun()
                        
                        if st.button("📤 Export", key=f"export_detailed_{i}"):
                            export_data = st.session_state.persistent_storage.export_conversation(conv['thread_id'])
                            if export_data:
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label="📁 Download",
                                    data=json_str,
                                    file_name=f"conversation_{conv['thread_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    key=f"download_detailed_{i}"
                                )
                        
                        if st.button("🗑️ Delete", key=f"delete_detailed_{i}"):
                            if st.session_state.persistent_storage.delete_conversation(conv['thread_id']):
                                st.success("Conversation deleted")
                                st.rerun()
                            else:
                                st.error("Failed to delete conversation")
        else:
            st.info("No conversations stored yet. Start chatting and save your conversations!")
    
    # Memory actions (existing functionality)
    if new_memory_enabled:
        st.subheader("Memory Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear Current Thread", type="primary"):
                st.session_state.chat_history = []
                if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
                    try:
                        # Clear the specific thread from checkpointer
                        thread_id = st.session_state.get('thread_id', 'default')
                        st.success(f"Cleared memory for thread: {thread_id}")
                    except Exception as e:
                        st.error(f"Error clearing memory: {str(e)}")
                else:
                    st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All Memory"):
                st.session_state.chat_history = []
                st.session_state.checkpointer = None
                st.session_state.agent = None
                st.success("All memory reset! Please reconnect to MCP server.")
                st.rerun()
        
        with col3:
            if st.button("💾 Export Current"):
                if st.session_state.chat_history:
                    memory_export = {
                        'thread_id': st.session_state.get('thread_id', 'default'),
                        'export_timestamp': datetime.datetime.now().isoformat(),
                        'message_count': len(st.session_state.chat_history),
                        'chat_history': st.session_state.chat_history,
                        'memory_settings': {
                            'memory_enabled': st.session_state.get('memory_enabled', False),
                            'memory_type': st.session_state.get('memory_type', 'Short-term'),
                            'max_messages': st.session_state.get('max_messages', 100)
                        }
                    }
                    
                    json_str = json.dumps(memory_export, indent=2)
                    st.download_button(
                        label="📁 Download",
                        data=json_str,
                        file_name=f"memory_export_{st.session_state.get('thread_id', 'default')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No chat history to export")
        
        with col4:
            uploaded_file = st.file_uploader("📁 Import Memory", type=['json'])
            if uploaded_file is not None:
                try:
                    memory_data = json.load(uploaded_file)
                    if 'chat_history' in memory_data:
                        st.session_state.chat_history = memory_data['chat_history']
                        if 'memory_settings' in memory_data:
                            settings = memory_data['memory_settings']
                            st.session_state.memory_enabled = settings.get('memory_enabled', True)
                            st.session_state.memory_type = settings.get('memory_type', 'Short-term')
                            st.session_state.max_messages = settings.get('max_messages', 100)
                        st.success(f"Imported {len(memory_data['chat_history'])} messages")
                        st.rerun()
                    else:
                        st.error("Invalid memory export file")
                except Exception as e:
                    st.error(f"Error importing memory: {str(e)}")
    
    # Conversation history viewer
    st.subheader("Conversation History")
    
    if st.session_state.get('chat_history'):
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_user = st.checkbox("Show User Messages", value=True)
        with col2:
            show_assistant = st.checkbox("Show Assistant Messages", value=True)
        with col3:
            show_tools = st.checkbox("Show Tool Executions", value=False)
        
        # Display filtered history
        with st.container():
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user" and show_user:
                    with st.expander(f"👤 User Message #{i+1}"):
                        st.write(message["content"])
                        
                elif message["role"] == "assistant" and show_assistant:
                    with st.expander(f"🤖 Assistant Message #{i+1}"):
                        st.write(message["content"])
                        if show_tools and "tool" in message and message["tool"]:
                            st.code(message["tool"], language="text")
        
        # Memory statistics
        st.subheader("Memory Statistics")
        
        user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        assistant_msgs = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
        tool_msgs = len([m for m in st.session_state.chat_history if m["role"] == "assistant" and m.get("tool")])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User Messages", user_msgs)
        with col2:
            st.metric("Assistant Messages", assistant_msgs)
        with col3:
            st.metric("Tool Executions", tool_msgs)
        with col4:
            memory_usage = len(json.dumps(st.session_state.chat_history))
            st.metric("Memory Usage", f"{memory_usage:,} chars")
        
    else:
        st.info("No conversation history available. Start chatting to see memory content!")
    
    # Memory tips
    with st.expander("💡 Memory Tips"):
        st.markdown("""
        **Memory Types:**
        - **Short-term**: Remembers conversation only within current browser session
        - **Persistent**: Remembers across browser sessions (when implemented)
        
        **Thread Management:**
        - Use different thread IDs for separate conversation topics
        - Thread IDs help organize conversations by context
        - Switch threads to start fresh conversations while keeping history
        
        **Memory Limits:**
        - Set max messages to control memory usage
        - Higher limits = more context but slower performance
        - Lower limits = faster but may lose conversation context
        
        **Best Practices:**
        - Export important conversations before clearing memory
        - Use descriptive thread IDs (e.g., "project_planning", "debugging_session")
        - Clear memory periodically to maintain performance
        """)

# Custom tools for memory-enabled agents
class HistoryFilter(BaseModel):
    """Input for get_history tool."""
    message_type: Optional[str] = Field(
        default="all", 
        description="Filter by message type: 'user', 'assistant', 'tool', or 'all'"
    )
    last_n_messages: Optional[int] = Field(
        default=10, 
        description="Number of recent messages to retrieve (max 50)"
    )
    search_query: Optional[str] = Field(
        default=None, 
        description="Search for messages containing this text"
    )

# Create an instance of the history tool
def create_history_tool():
    """Create a history tool that can access the current session state."""
    
    class ConversationHistoryTool(BaseTool):
        """Custom tool for retrieving conversation history."""
        
        name: str = "get_conversation_history"
        description: str = """
        Retrieve conversation history from the current session.
        
        This tool allows you to access previous messages in the conversation to:
        - Reference earlier discussions
        - Summarize conversation topics
        - Find specific information mentioned before
        - Maintain context across interactions
        """
        args_schema: type[BaseModel] = HistoryFilter
        
        def _run(
            self,
            message_type: str = "all",
            last_n_messages: int = 10,
            search_query: Optional[str] = None
        ) -> str:
            """Synchronous implementation."""
            return self._get_history(message_type, last_n_messages, search_query)
        
        async def _arun(
            self,
            message_type: str = "all",
            last_n_messages: int = 10,
            search_query: Optional[str] = None
        ) -> str:
            """Asynchronous implementation."""
            return self._get_history(message_type, last_n_messages, search_query)
        
        def _get_history(
            self,
            message_type: str = "all",
            last_n_messages: int = 10,
            search_query: Optional[str] = None
        ) -> str:
            """Get conversation history from session state."""
            try:
                # Access session state through streamlit
                import streamlit as st
                
                if not hasattr(st.session_state, 'chat_history') or not st.session_state.chat_history:
                    return "No conversation history available yet."
                
                # Limit the number of messages - ensure we have valid integers
                last_n_messages = int(last_n_messages) if last_n_messages is not None else 10
                last_n_messages = min(max(1, last_n_messages), 50)
                
                # Get recent messages
                recent_messages = st.session_state.chat_history[-last_n_messages:]
                
                # Filter by message type
                filtered_messages = []
                if message_type != "all":
                    if message_type == "tool":
                        # Show messages that have tool executions
                        for msg in recent_messages:
                            if (msg.get("role") == "assistant" and 
                                msg.get("tool") is not None and 
                                str(msg.get("tool")).strip()):
                                filtered_messages.append(msg)
                    else:
                        for msg in recent_messages:
                            if msg.get("role") == message_type:
                                filtered_messages.append(msg)
                else:
                    filtered_messages = recent_messages
                
                # Apply search filter if provided
                if search_query and search_query.strip():
                    search_filtered = []
                    search_term = search_query.lower().strip()
                    for msg in filtered_messages:
                        content = str(msg.get("content", "")).lower()
                        tool_content = str(msg.get("tool", "")).lower()
                        if search_term in content or search_term in tool_content:
                            search_filtered.append(msg)
                    filtered_messages = search_filtered
                
                if not filtered_messages:
                    return f"No messages found matching your criteria (type: {message_type}, search: '{search_query or 'none'}')"
                
                # Format the history in a clean, readable way
                formatted_history = []
                formatted_history.append(f"📋 **Conversation History** ({len(filtered_messages)} message{'s' if len(filtered_messages) != 1 else ''})")
                formatted_history.append("")
                
                for i, msg in enumerate(filtered_messages, 1):
                    role = msg.get("role", "unknown")
                    content = str(msg.get("content", "")).strip()
                    tool_output = msg.get("tool")
                    
                    if role == "user":
                        formatted_history.append(f"**{i}. User:** {content}")
                    elif role == "assistant":
                        formatted_history.append(f"**{i}. Assistant:** {content}")
                        if tool_output and str(tool_output).strip():
                            # Show a summary of tool execution instead of full output
                            tool_lines = str(tool_output).strip().split('\n')
                            if len(tool_lines) > 3:
                                formatted_history.append(f"   🔧 *Tool used (output truncated)*")
                            else:
                                formatted_history.append(f"   🔧 *Tool output: {tool_output.strip()}*")
                    
                    formatted_history.append("")  # Add spacing between messages
                
                return "\n".join(formatted_history)
                
            except Exception as e:
                return f"Error retrieving conversation history: {str(e)}"
    
    return ConversationHistoryTool()

# Create an instance of the history tool
get_conversation_history = create_history_tool()

def main():
    # Set page configuration
    st.set_page_config(
        page_title="LangChain MCP Client",
        page_icon="logo_transparent.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.logo("side_logo.png", size="large")

    # Session state initialization
    if "client" not in st.session_state:
        st.session_state.client = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "tools" not in st.session_state:
        st.session_state.tools = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "servers" not in st.session_state:
        st.session_state.servers = {}
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Single Server"
    if "tool_executions" not in st.session_state:
        st.session_state.tool_executions = []
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)
    # Tool testing session state
    if "tool_test_results" not in st.session_state:
        st.session_state.tool_test_results = []
    if "tool_test_stats" not in st.session_state:
        st.session_state.tool_test_stats = {}
    # Memory-related session state
    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = False
    if "memory_type" not in st.session_state:
        st.session_state.memory_type = "Short-term (Session)"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "default"
    if "max_messages" not in st.session_state:
        st.session_state.max_messages = 100
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = None

    # Sidebar for configuration
    sidebar()

    # Tabs for different sections
    t_chat, t_test, t_memory, t_about, = st.tabs(["🗨️ Chat", "🔧 Test Tools", "🧠 Memory", "ℹ️ About"])

    with t_chat:
        tab_chat()
        # Tool execution display
        display_tool_executions()
    with t_test:
        tab_test_tools()
    with t_memory:
        tab_memory()
    with t_about:
        tab_about()

if __name__ == "__main__":
    main()