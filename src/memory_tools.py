"""
Memory and conversation history tools.

This module provides tools for managing conversation history
and memory functionality within the agent system.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import streamlit as st


class HistoryFilter(BaseModel):
    """Input schema for the conversation history tool."""
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


class ConversationHistoryTool(BaseTool):
    """Custom tool for retrieving conversation history from the current session."""
    
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


def create_history_tool() -> ConversationHistoryTool:
    """Create an instance of the conversation history tool."""
    return ConversationHistoryTool()


def format_chat_history_for_export(chat_history: list) -> dict:
    """
    Format chat history for export.
    
    Args:
        chat_history: List of chat messages
    
    Returns:
        Formatted dictionary ready for export
    """
    import datetime
    
    return {
        'export_timestamp': datetime.datetime.now().isoformat(),
        'message_count': len(chat_history),
        'messages': chat_history,
        'format_version': '1.0'
    }


def calculate_chat_statistics(chat_history: list) -> dict:
    """
    Calculate statistics from chat history.
    
    Args:
        chat_history: List of chat messages
    
    Returns:
        Dictionary with statistics
    """
    if not chat_history:
        return {
            'total_messages': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'tool_executions': 0,
            'estimated_tokens': 0
        }
    
    user_msgs = len([m for m in chat_history if m.get("role") == "user"])
    assistant_msgs = len([m for m in chat_history if m.get("role") == "assistant"])
    tool_msgs = len([m for m in chat_history if m.get("role") == "assistant" and m.get("tool")])
    
    # Rough token estimation (4 characters per token)
    total_chars = sum(len(str(m.get("content", ""))) for m in chat_history)
    estimated_tokens = total_chars // 4
    
    return {
        'total_messages': len(chat_history),
        'user_messages': user_msgs,
        'assistant_messages': assistant_msgs,
        'tool_executions': tool_msgs,
        'estimated_tokens': estimated_tokens,
        'avg_message_length': total_chars / len(chat_history) if chat_history else 0
    } 