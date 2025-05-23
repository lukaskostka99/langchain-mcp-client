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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

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
                            
                            # Create the agent
                            st.session_state.agent = create_react_agent(llm, st.session_state.tools)
                            
                            st.success(f"Connected to {len(st.session_state.servers)} MCP servers! Found {len(st.session_state.tools)} tools.")
                        except Exception as e:
                            st.error(f"Error connecting to MCP servers: {str(e)}")
                            st.code(traceback.format_exc(), language="python")
        
        # Display available tools if connected
        if st.session_state.tools:
            st.header("Available Tools")
            
            # Tool selection dropdown
            selected_tool_name = st.selectbox(
                "Available Tools",
                options=[tool.name for tool in st.session_state.tools],
                index=0 if st.session_state.tools else None
            )
            
            if selected_tool_name:
                # Find the selected tool
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
                        # Run the agent
                        response = run_async(run_agent(st.session_state.agent, user_input))
                        
                        tool_outputs = []  # Store multiple tool outputs
                        # Extract tool executions if available
                        if "messages" in response:
                            for msg in response["messages"]:
                                # Look for AIMessage with tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        # Find corresponding ToolMessage
                                        tool_output = next(
                                            (m.content for m in response["messages"] 
                                             if isinstance(m, ToolMessage) and 
                                             m.tool_call_id == tool_call['id']),
                                            None
                                        )
                                        if tool_output:
                                            tool_outputs.append(tool_output)  # Add to list
                                            st.session_state.tool_executions.append({
                                                "tool_name": tool_call['name'],
                                                "input": tool_call['args'],
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
                        
                        # Add assistant message to chat history with all tool outputs
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "tool": "\n".join(tool_outputs) if tool_outputs else None,  # Join all tool outputs
                            "content": output
                        })
                    
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

    # Sidebar for configuration
    sidebar()

    # Tabs for different sections
    t_chat, t_test, t_about, = st.tabs(["🗨️ Chat", "🔧 Test Tools", "ℹ️ About"])

    with t_chat:
        tab_chat()
        # Tool execution display
        display_tool_executions()
    with t_test:
        tab_test_tools()
    with t_about:
        tab_about()

if __name__ == "__main__":
    main()