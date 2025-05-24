"""
Agent management and execution functionality.

This module handles the creation, configuration, and execution
of LangGraph agents with various tools and memory configurations.
"""

from typing import Dict, List, Optional
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from .memory_tools import create_history_tool
from .database import PersistentStorageManager


async def run_agent(agent, message: str) -> Dict:
    """Run the agent with the provided message."""
    return await agent.ainvoke({"messages": [HumanMessage(message)]})


async def run_tool(tool, **kwargs):
    """Run a tool with the provided parameters."""
    return await tool.ainvoke(kwargs)


def create_agent_with_tools(
    llm,
    mcp_tools: List[BaseTool],
    memory_enabled: bool = False,
    memory_type: str = "Short-term (Session)",
    persistent_storage: Optional[PersistentStorageManager] = None
):
    """
    Create a LangGraph agent with the specified configuration.
    
    Args:
        llm: The language model to use
        mcp_tools: List of MCP tools
        memory_enabled: Whether to enable memory
        memory_type: Type of memory to use
        persistent_storage: PersistentStorageManager instance for persistent memory
    
    Returns:
        Configured agent and checkpointer
    """
    # Start with MCP tools
    agent_tools = mcp_tools.copy()
    checkpointer = None
    
    # Add memory functionality if enabled
    if memory_enabled:
        if memory_type == "Persistent (Cross-session)" and persistent_storage:
            # Use SQLite checkpointer for persistent storage
            checkpointer = persistent_storage.get_checkpointer()
        else:
            # Use in-memory checkpointer for short-term storage
            checkpointer = InMemorySaver()
        
        # Add history tool when memory is enabled
        agent_tools.append(create_history_tool())
    
    # Check if LLM has a system prompt
    system_prompt = getattr(llm, '_system_prompt', None)
    
    # Create the agent with optional system prompt
    if system_prompt:
        agent = create_react_agent(llm, agent_tools, checkpointer=checkpointer, prompt=system_prompt)
    else:
        agent = create_react_agent(llm, agent_tools, checkpointer=checkpointer)
    
    return agent, checkpointer


def get_agent_config_summary(
    provider: str,
    model: str,
    mcp_tool_count: int,
    memory_enabled: bool,
    memory_type: str = None,
    thread_id: str = None
) -> Dict[str, str]:
    """
    Generate a summary of the agent configuration.
    
    Args:
        provider: LLM provider name
        model: Model name
        mcp_tool_count: Number of MCP tools
        memory_enabled: Whether memory is enabled
        memory_type: Type of memory if enabled
        thread_id: Thread ID if memory is enabled
    
    Returns:
        Dictionary with configuration summary
    """
    config = {
        "provider": provider,
        "model": model,
        "mcp_tools": str(mcp_tool_count),
        "memory": "Enabled" if memory_enabled else "Disabled"
    }
    
    if memory_enabled:
        config["memory_type"] = memory_type or "Short-term"
        config["thread_id"] = thread_id or "default"
        config["total_tools"] = str(mcp_tool_count + 1)  # +1 for history tool
    else:
        config["total_tools"] = str(mcp_tool_count)
    
    return config


def validate_agent_configuration(
    llm_provider: str,
    api_key: str,
    model_name: str,
    memory_enabled: bool = False,
    memory_type: str = None,
    thread_id: str = None
) -> tuple[bool, str]:
    """
    Validate agent configuration before creation.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Import here to avoid circular imports
    from .llm_providers import validate_provider_config
    from .utils import is_valid_thread_id
    
    # Validate LLM configuration
    is_valid, error = validate_provider_config(llm_provider, api_key, model_name)
    if not is_valid:
        return False, error
    
    # Validate memory configuration
    if memory_enabled:
        if memory_type not in ["Short-term (Session)", "Persistent (Cross-session)"]:
            return False, "Invalid memory type"
        
        if thread_id and not is_valid_thread_id(thread_id):
            return False, "Invalid thread ID format"
    
    return True, ""


def prepare_agent_invocation_config(
    memory_enabled: bool,
    thread_id: str = "default"
) -> Dict:
    """
    Prepare configuration for agent invocation.
    
    Args:
        memory_enabled: Whether memory is enabled
        thread_id: Thread ID for memory
    
    Returns:
        Configuration dictionary for agent invocation
    """
    if memory_enabled:
        return {"configurable": {"thread_id": thread_id}}
    return {}


def extract_tool_executions_from_response(response: Dict) -> List[Dict]:
    """
    Extract tool execution information from agent response.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        List of tool execution records for the CURRENT interaction only
    """
    from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
    import datetime
    
    tool_executions = []
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if "messages" not in response:
        return tool_executions
    
    messages = response["messages"]
    
    # When memory is enabled, we need to find only the NEW messages from this interaction
    # Strategy: Look for the last HumanMessage (current user input) and then get 
    # any AIMessage and ToolMessage that come after it
    
    # Find the index of the last HumanMessage (current interaction)
    last_human_idx = -1
    for i in reversed(range(len(messages))):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    
    if last_human_idx == -1:
        # No human message found, fall back to looking at the last few messages
        # This handles the case where memory is disabled
        recent_messages = messages[-10:]  # Look at last 10 messages max
    else:
        # Get messages after the last human message (these are from current interaction)
        recent_messages = messages[last_human_idx + 1:]
    
    # Find AI messages with tool calls in the recent messages
    for msg in recent_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Process tool calls from this AI message
            for tool_call in msg.tool_calls:
                # Find corresponding ToolMessage in recent messages
                tool_output = next(
                    (m.content for m in recent_messages
                     if isinstance(m, ToolMessage) and 
                     hasattr(m, 'tool_call_id') and
                     m.tool_call_id == tool_call['id']),
                    None
                )
                
                if tool_output:
                    tool_executions.append({
                        "tool_name": tool_call['name'],
                        "input": tool_call['args'],
                        "output": tool_output,
                        "timestamp": current_timestamp
                    })
    
    return tool_executions


def extract_assistant_response(response: Dict) -> str:
    """
    Extract the assistant's text response from the agent response.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        Assistant's text response
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    if "messages" not in response:
        return ""
    
    # When memory is enabled, the response contains the entire conversation history
    # We need to find the LAST (most recent) AIMessage that has actual content
    ai_messages = []
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
            content = str(msg.content).strip()
            # Skip tool call artifacts and empty content
            if (content and 
                content != "<|tool_call|>[]" and 
                not content.startswith('<|tool_call|>') and
                not content.endswith('[]')):
                ai_messages.append(content)
    
    # Return the last (most recent) AI message content
    if ai_messages:
        return ai_messages[-1]
    
    return "" 