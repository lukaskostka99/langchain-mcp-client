import streamlit as st
import asyncio
import json
import datetime
import traceback
from typing import Dict, List, Optional, Any
import os
import nest_asyncio

# Apply nest_asyncio to allow nested asyncio event loops (needed for Streamlit's execution model)
nest_asyncio.apply()

# Import langchain and related libraries
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

# Set page configuration
st.set_page_config(
    page_title="LangChain MCP Client",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Helper functions
def run_async(coro):
    """Run an async function within the stored event loop."""
    return st.session_state.loop.run_until_complete(coro)

async def setup_mcp_client(server_config: Dict[str, Dict]) -> MultiServerMCPClient:
    """Initialize a MultiServerMCPClient with the provided server configuration."""
    client = MultiServerMCPClient(server_config)
    return await client.__aenter__()

async def get_tools_from_client(client: MultiServerMCPClient) -> List[BaseTool]:
    """Get tools from the MCP client."""
    return client.get_tools()

async def run_agent(agent, message: str) -> Dict:
    """Run the agent with the provided message."""
    return await agent.ainvoke({"messages": message})

async def run_tool(tool, **kwargs):
    """Run a tool with the provided parameters."""
    return await tool.ainvoke(**kwargs)

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
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

# Main app layout
st.title("🧩 LangChain MCP Client")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Provider selection
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=["OpenAI", "Anthropic"],
        index=0
    )
    
    # API Key input
    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        help=f"Enter your {llm_provider} API Key"
    )
    
    # Model selection based on provider
    if llm_provider == "OpenAI":
        model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        default_model_idx = 0
    else:  # Anthropic
        model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        default_model_idx = 0
    
    model_name = st.selectbox(
        "Model",
        options=model_options,
        index=default_model_idx
    )
    
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
            if not api_key:
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
                                "timeout": 5,
                                "sse_read_timeout": 300
                            }
                        }
                        
                        # Initialize the MCP client
                        if st.session_state.client is not None:
                            # First close the existing client properly
                            try:
                                run_async(st.session_state.client.__aexit__(None, None, None))
                            except Exception as e:
                                st.warning(f"Error closing previous client: {str(e)}")
                        
                        st.session_state.client = run_async(setup_mcp_client(server_config))
                        
                        # Get tools from the client
                        st.session_state.tools = run_async(get_tools_from_client(st.session_state.client))
                        
                        # Create the language model
                        llm = create_llm_model(llm_provider, api_key, model_name)
                        
                        # Create the agent
                        st.session_state.agent = create_react_agent(llm, st.session_state.tools)
                        
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
                    "timeout": 5,
                    "sse_read_timeout": 300
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
            if not api_key:
                st.error(f"Please enter your {llm_provider} API Key")
            elif not st.session_state.servers:
                st.error("Please add at least one server")
            else:
                with st.spinner("Connecting to MCP servers..."):
                    try:
                        # Initialize the MCP client with all servers
                        if st.session_state.client is not None:
                            # First close the existing client properly
                            try:
                                run_async(st.session_state.client.__aexit__(None, None, None))
                            except Exception as e:
                                st.warning(f"Error closing previous client: {str(e)}")
                        
                        st.session_state.client = run_async(setup_mcp_client(st.session_state.servers))
                        
                        # Get tools from the client
                        st.session_state.tools = run_async(get_tools_from_client(st.session_state.client))
                        
                        # Create the language model
                        llm = create_llm_model(llm_provider, api_key, model_name)
                        
                        # Create the agent
                        st.session_state.agent = create_react_agent(llm, st.session_state.tools)
                        
                        st.success(f"Connected to {len(st.session_state.servers)} MCP servers! Found {len(st.session_state.tools)} tools.")
                    except Exception as e:
                        st.error(f"Error connecting to MCP servers: {str(e)}")
                        st.code(traceback.format_exc(), language="python")
    
    # Display available tools if connected
    if st.session_state.tools:
        st.header("Available Tools")
        
        # Tool testing section
        selected_tool_name = st.selectbox(
            "Select a tool to test",
            options=[tool.name for tool in st.session_state.tools],
            index=0 if st.session_state.tools else None
        )
        
        if selected_tool_name:
            # Find the selected tool
            selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
            
            if selected_tool:
                st.subheader(f"Test Tool: {selected_tool.name}")
                st.write(f"**Description:** {selected_tool.description}")
                
                # Create form for tool parameters
                with st.form(key=f"test_tool_{selected_tool.name}"):
                    param_values = {}
                    
                    if hasattr(selected_tool, 'args_schema') and selected_tool.args_schema:
                        if hasattr(selected_tool.args_schema, 'schema') and 'properties' in selected_tool.args_schema.schema():
                            properties = selected_tool.args_schema.schema().get('properties', {})
                            required_params = selected_tool.args_schema.schema().get('required', [])
                            
                            for param, details in properties.items():
                                param_type = details.get('type', 'string')
                                description = details.get('description', '')
                                is_required = param in required_params
                                
                                st.write(f"**{param}**{' (required)' if is_required else ''}: {description}")
                                
                                if param_type == 'integer' or param_type == 'number':
                                    param_values[param] = st.number_input(
                                        f"Enter {param}",
                                        value=0,
                                        help=description
                                    )
                                elif param_type == 'boolean':
                                    param_values[param] = st.checkbox(
                                        f"Enable {param}",
                                        help=description
                                    )
                                else:  # default to string
                                    param_values[param] = st.text_input(
                                        f"Enter {param}",
                                        help=description
                                    )
                    
                    submit_button = st.form_submit_button("Run Tool")
                
                if submit_button:
                    with st.spinner(f"Running {selected_tool.name}..."):
                        try:
                            # Run the tool
                            result = run_async(run_tool(selected_tool, **param_values))
                            
                            # Record tool execution
                            st.session_state.tool_executions.append({
                                "tool_name": selected_tool.name,
                                "input": param_values,
                                "output": result,
                                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            # Display result
                            st.success("Tool executed successfully!")
                            st.subheader("Result:")
                            st.code(result)
                        except Exception as e:
                            st.error(f"Error executing tool: {str(e)}")
                            st.code(traceback.format_exc(), language="python")
        
        # Tool details in expanders
        st.subheader("Tool Details")
        for tool in st.session_state.tools:
            with st.expander(tool.name):
                st.write(f"**Description:** {tool.description}")
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    st.write("**Parameters:**")
                    if hasattr(tool.args_schema, 'schema') and 'properties' in tool.args_schema.schema():
                        for param, details in tool.args_schema.schema().get('properties', {}).items():
                            st.write(f"- **{param}** ({details.get('type', 'any')}): {details.get('description', '')}")

# Main chat interface
st.header("Chat with Agent")

# Function to display tool execution details
def display_tool_executions():
    if st.session_state.tool_executions:
        with st.expander("Tool Execution History", expanded=False):
            for i, exec_record in enumerate(st.session_state.tool_executions):
                st.markdown(f"### Execution #{i+1}: `{exec_record['tool_name']}`")
                st.markdown(f"**Input:** ```json\n{json.dumps(exec_record['input'], indent=2)}\n```")
                st.markdown(f"**Output:** ```\n{exec_record['output']}\n```")
                st.markdown(f"**Time:** {exec_record['timestamp']}")
                st.divider()

# Connection status indicator
connection_status = st.empty()
if st.session_state.client is not None:
    connection_status.success("📶 Connected to MCP server(s)")
else:
    connection_status.warning("⚠️ Not connected to any MCP server")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Display tool executions if any
display_tool_executions()

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
                    # Run the agent
                    response = run_async(run_agent(st.session_state.agent, user_input))
                    
                    # Extract tool executions if available
                    if "intermediate_steps" in response:
                        for step in response["intermediate_steps"]:
                            if len(step) >= 2:
                                tool_action = step[0]
                                tool_output = step[1]
                                
                                # Record tool execution
                                st.session_state.tool_executions.append({
                                    "tool_name": tool_action.tool,
                                    "input": tool_action.tool_input,
                                    "output": tool_output,
                                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })

                    # Extract and display the response
                    output = ""
                    
                    if "messages" in response:
                        for msg in response["messages"]:
                            if isinstance(msg, HumanMessage):
                                continue  # Skip human messages
                            elif hasattr(msg, 'name') and msg.name:  # ToolMessage
                                st.code(msg.content)
                            else:  # AIMessage
                                if hasattr(msg, "content") and msg.content:
                                    output = str(msg.content)
                                    st.write(output)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": output})
                    
                    # Update tool execution display
                    display_tool_executions()
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
                    st.code(traceback.format_exc(), language="python")

# Footer
st.divider()
st.markdown("""
### About
This application demonstrates a Streamlit interface for LangChain MCP (Model Context Protocol) adapters. 
It allows you to connect to MCP servers and use their tools with different LLM providers.

For more information, check out:
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol](https://modelcontextprotocol.io/introduction)
""")

# Proper cleanup when the session ends
def on_shutdown():
    if st.session_state.client is not None:
        try:
            # Close the client properly
            run_async(st.session_state.client.__aexit__(None, None, None))
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")

# Register the cleanup function
import atexit
atexit.register(on_shutdown)