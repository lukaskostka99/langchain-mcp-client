# LangChain MCP Client Streamlit App

This Streamlit application provides a user interface for connecting to MCP (Model Context Protocol) servers and interacting with them using different LLM providers (OpenAI, Anthropic, Google, Ollama).

## ⚠️ Development Status

**This application is currently in active development.** While functional, you may encounter bugs, incomplete features, or unexpected behavior. We appreciate your patience and welcome feedback to help improve the application.

## Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic Claude, Google Gemini, and Ollama
- **OpenAI Reasoning Models Support**: Enhanced support for o3-mini, o4-mini with specialized parameter handling
- **Streaming Responses**: Real-time token-by-token streaming for supported models
- **MCP (Model Context Protocol) Integration**: Connect to MCP servers for tool access
- **Advanced Memory Management**: Short-term session memory and persistent cross-session memory
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Tool Testing Interface**: Test individual tools with custom parameters
- **Chat-Only Mode**: Use without MCP servers for simple conversations
- **Advanced Model Configuration**: Custom temperature, max tokens, timeout, and system prompts
- **Intelligent Model Validation**: Automatic parameter validation and compatibility checking
- **Comprehensive Logging**: Track all tool executions and conversations
- **Export/Import**: Save and load conversation history
- **Containerized Deployment**: Easy Docker setup

## NEW - Streaming Support

The application now supports real-time streaming responses for all compatible models:

### Supported Models with Streaming
- ✅ **OpenAI**: All GPT models (GPT-4o, GPT-4, GPT-3.5-turbo) + Reasoning models (o3-mini, o4-mini)
- ✅ **Anthropic**: All Claude models (Claude-3.5-Sonnet, Claude-3-Opus, Claude-3-Haiku)
- ✅ **Google**: All Gemini models (Gemini-2.0-Flash, Gemini-2.5-Pro-Exp)
- ✅ **Ollama**: All local models (Granite, Qwen, etc.)
- ❌ **OpenAI o1 Series**: o1, o1-mini, o1-preview (not supported due to API limitations)

### How It Works
1. **Enable Streaming**: Toggle in the "🌊 Streaming Settings" section in the sidebar
2. **Real-time Display**: Responses appear token by token as they're generated
3. **Tool Integration**: See tool execution status in real-time
4. **Fallback Support**: Automatically falls back to non-streaming if issues occur

### Benefits
- **Better User Experience**: See responses as they're being generated
- **Faster Perceived Response Time**: Start reading while the model is still generating
- **Real-time Feedback**: Know immediately when tools are being executed
- **Interactive Feel**: More engaging conversation experience

## OpenAI Reasoning Models Support

The application now includes enhanced support for OpenAI's reasoning models with specialized handling:

### Supported Reasoning Models
- ✅ **o3-mini**: Fast reasoning model with streaming support
- ✅ **o4-mini**: Advanced reasoning model with streaming support
- ❌ **o1 Series**: o1, o1-mini, o1-preview (not supported due to unique API requirements)

### Reasoning Model Features
- **Automatic Parameter Optimization**: Uses `max_completion_tokens` instead of `max_tokens`
- **Temperature Handling**: Automatically disables temperature for reasoning models (they use fixed temperature)
- **Reasoning Effort**: Automatically sets reasoning effort to "medium" for optimal performance
- **Streaming Support**: o3/o4 series support real-time streaming (o1 series does not)
- **Smart Validation**: Prevents incompatible parameter combinations with clear error messages

### User Experience
- **Clear Warnings**: Visual indicators when selecting reasoning models
- **Alternative Suggestions**: Recommends compatible models when o1 series is selected
- **Seamless Integration**: Works with all existing features (memory, tools, streaming)

## Configuration System

### LLM Providers & Parameters
- **OpenAI**: 
  - **Regular Models**: GPT-4o, GPT-4, GPT-3.5-turbo with temperature (0.0-2.0), max tokens (1-16384), timeout (10-600s)
  - **Reasoning Models**: o3-mini, o4-mini with specialized parameter handling (no temperature, max_completion_tokens, reasoning_effort)
  - **Unsupported**: o1, o1-mini, o1-preview (incompatible API requirements)
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Opus, Claude-3-Haiku with temperature (0.0-1.0), max tokens (1-8192), timeout (10-600s)  
- **Google**: Gemini-2.0-Flash, Gemini-2.5-Pro-Exp with temperature (0.0-2.0), max tokens (1-32768), timeout (10-600s)
- **Ollama**: Local models (Granite3.3:8b, Qwen3:4b) with temperature (0.0-2.0), max tokens (1-32768), timeout (10-600s)

### Custom Model Support
All providers now support an **"Other"** option that allows you to specify custom model names:
- **OpenAI**: Enter any OpenAI model name (e.g., gpt-4-turbo, o3-mini, custom fine-tuned models)
- **Anthropic**: Enter any Anthropic model name (e.g., claude-3-sonnet-20240229)
- **Google**: Enter any Google model name (e.g., gemini-pro, gemini-1.5-pro)
- **Ollama**: Enter any locally available model name (e.g., llama3, codellama, custom models)

### System Prompts
- **Built-in Presets**: Five specialized system prompts for different use cases
- **Custom Prompts**: Create and save your own system prompts
- **Real-time Application**: Apply configuration changes with instant feedback
- **Configuration Status**: Visual indicators showing applied/changed/default states

### Configuration Management
- **Apply Configuration**: Smart button that detects changes and applies settings
- **Export/Import**: Save configurations as JSON files for backup or sharing
- **Reset Options**: Reset to defaults or previously applied settings
- **Change Detection**: Real-time tracking of configuration modifications

## Memory System

### Memory Types

1. **Short-term (Session)**: 
   - Conversations stored in browser session memory
   - Lost when browser is closed or app restarted
   - Uses LangGraph's InMemorySaver

2. **Persistent (Cross-session)**:
   - Conversations stored in SQLite database (`conversations.db`)
   - Persists across browser sessions and app restarts
   - Uses LangGraph's SqliteSaver with custom metadata management
   - Auto-saves conversations during chat
   - Browse, load, and manage saved conversations

### Advanced Memory Features

- **Conversation Import/Export**: Import previous conversations with preview and confirmation system
- **Thread Management**: Switch between different conversation threads with unique IDs
- **Memory Analytics**: Track message counts, memory usage, conversation statistics, and database size
- **Conversation Browser**: View and manage all stored conversations with metadata and search capabilities
- **History Tool Integration**: When memory is enabled, the agent gains access to a `get_conversation_history` tool that allows it to:
  - Search through previous conversations by type or content
  - Reference earlier discussions and maintain context
  - Summarize conversation topics and filter messages
  - Access specific parts of the conversation history with clean, formatted output

### Memory Safety & Performance
- **Import Validation**: Safe import system with format validation and preview before applying
- **Loop Prevention**: Robust safeguards against infinite loops during memory operations
- **Flexible Limits**: Configurable message limits and memory constraints
- **Real-time Status**: Visual indicators for memory state, thread information, and change detection

## Tool Testing & Analytics
- **Individual Tool Testing**: Access via the "🔧 Test Tools" tab
- **Dynamic Parameter Forms**: Auto-generated forms based on tool schemas with real-time validation
- **Performance Tracking**: Execution timing, success/failure rates, and detailed statistics
- **Test History**: Complete test result history with export capabilities
- **Result Analysis**: Success rate calculations and average execution time metrics
- **JSON Export**: Export test results and performance data for external analysis

## Future Improvements

- **STDIO MCP Servers**: Support for connecting to MCP servers using standard input/output (STDIO) for more flexible server configurations
- **RAG (File Upload)**: Enable Retrieval-Augmented Generation (RAG) by allowing users to upload files that the agent can use to enhance its responses
- **Enhanced Tool Validation**: Advanced parameter validation and schema checking for MCP tools
- **Multi-threaded Processing**: Parallel processing for multiple tool executions and server connections

## Enhanced Conversation History Tool

Based on AI feedback, the `get_conversation_history` tool has been significantly enhanced with the following improvements:

### ✨ New Features

**Completeness of Return (Rating: 9/10)**
- ✅ **Timestamps**: Every message now includes precise timestamps
- ✅ **Message IDs**: Unique identifiers for each message (e.g., `msg_0001`, `msg_0002`)
- ✅ **Tool Execution Details**: Rich metadata about tool usage including inputs/outputs
- ✅ **Enhanced Metadata**: Comprehensive information for precise referencing

**Ease of Use (Rating: 10/10)**
- ✅ **Advanced Filtering**: 7 parameters for flexible querying
- ✅ **Date Range Filtering**: Filter by specific date ranges (YYYY-MM-DD format)
- ✅ **Flexible Sorting**: Choose between newest-first or oldest-first ordering
- ✅ **Smart Search**: Search through message content AND tool execution data
- ✅ **Boolean Logic**: Support for AND, OR, NOT operators in search queries
- ✅ **Regex Patterns**: Advanced pattern matching with regex support
- ✅ **Metadata Control**: Option to include/exclude detailed metadata

### 🔧 Enhanced Parameters

```python
get_conversation_history(
    message_type="all",           # 'user', 'assistant', 'tool', or 'all'
    last_n_messages=10,           # Number of messages (max 100)
    search_query=None,            # Advanced search with boolean/regex support
    sort_order="newest_first",    # 'newest_first' or 'oldest_first'
    date_from=None,               # Start date (YYYY-MM-DD)
    date_to=None,                 # End date (YYYY-MM-DD)
    include_metadata=True         # Include timestamps and IDs
)
```

### 🔍 Advanced Search Capabilities

**Simple Text Search:**
```python
search_query="weather"                    # Find messages containing "weather"
```

**Boolean Operators:**
```python
search_query="weather AND temperature"   # Both terms must be present
search_query="sunny OR cloudy OR rainy"  # Any of these terms
search_query="weather NOT rain"          # Contains "weather" but not "rain"
search_query="(weather OR climate) AND NOT error"  # Complex logic
```

**Regex Patterns:**
```python
search_query="regex:\\d{2}°[CF]"         # Find temperatures like "72°F" or "23°C"
search_query="regex:https?://\\S+"       # Find URLs
search_query="regex:\\$\\d+(\\.\\d{2})?" # Find dollar amounts like "$25.99"
search_query="regex:\\b\\d{4}-\\d{2}-\\d{2}\\b"  # Find dates like "2024-01-15"
```

**Tool-Specific Search:**
```python
search_query="tool AND (success OR complete) NOT error"  # Successful tool executions
search_query="regex:API.*key AND NOT expired"            # API key related messages
```

### 🎯 Use Cases

- **Precise Referencing**: "Show me message msg_0015 from yesterday"
- **Date-based Analysis**: "What did we discuss between 2024-01-10 and 2024-01-12?"
- **Tool Usage Tracking**: "Show me all messages where tools were used"
- **Content Search**: "Find conversations mentioning 'weather' or 'temperature'"
- **Chronological Review**: "Show the oldest 20 messages to see how our conversation started"
- **Boolean Search**: "Find messages about (weather OR climate) AND NOT rain"
- **Pattern Matching**: "Find all URLs or email addresses in our conversation"
- **Error Analysis**: "Show tool executions that contain 'error' OR 'failed'"

### 📈 Improvement Summary

**Addressing AI Feedback:**

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Completeness** | Basic content only | Timestamps, IDs, tool metadata | 7/10 → 9/10 |
| **Ease of Use** | 3 simple parameters | 7 advanced parameters | 8/10 → 10/10 |
| **Search Power** | Simple text search | Boolean + Regex support | Basic → Enterprise |
| **Filtering** | Type and count only | Date ranges, sorting, metadata | Limited → Comprehensive |
| **Referencing** | No unique identifiers | Message IDs and timestamps | Impossible → Precise |

**Key Enhancements:**
- ✅ **Timestamps**: Every message now has precise timing
- ✅ **Message IDs**: Unique identifiers for exact referencing  
- ✅ **Date Filtering**: Query specific time periods
- ✅ **Flexible Sorting**: Newest-first or oldest-first ordering
- ✅ **Boolean Logic**: AND, OR, NOT operators for complex queries
- ✅ **Regex Support**: Pattern matching for sophisticated searches
- ✅ **Tool Integration**: Search through tool inputs/outputs
- ✅ **Metadata Control**: Choose detail level in responses

This enhanced tool now provides enterprise-grade conversation history management with precise tracking, flexible filtering, and comprehensive metadata support.

### 📋 Sample Output

```
📋 Conversation History (3 messages)
📅 Date Range: 2024-01-15 to 2024-01-15
🔄 Sort Order: Newest First

1. [msg_0003] Assistant (2024-01-15 14:30:22):
   The weather in San Francisco is currently 72°F and sunny.
   🔧 Tools used:
      • weather_api (executed at 2024-01-15 14:30:20)
        Input: {"location": "San Francisco", "units": "fahrenheit"}
        Output: {"temperature": 72, "condition": "sunny", "humidity": 65}

2. [msg_0002] User (2024-01-15 14:30:15):
   What's the weather like in San Francisco?

3. [msg_0001] User (2024-01-15 14:29:45):
   Hello! Can you help me with some questions?
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/guinacio/langchain-mcp-client.git
cd langchain-mcp-client
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app with:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Setting Up an MCP Server

To use this application, you'll need an MCP server running or a valid URL to an MCP server. 
Use the simple MCP server available on weather_server.py for a quick test:

2. Install the MCP library:
```bash
pip install mcp
```

3. Run the server:
```bash
python weather_server.py
```

The server will start on port 8000 by default. In the Streamlit app, you can connect to it using the URL `http://localhost:8000/sse`.

## Troubleshooting

- **Connection Issues**: Ensure your MCP server is running and accessible
- **API Key Errors**: Verify that you've entered the correct API key for your chosen LLM provider
- **Tool Errors**: Check the server logs for details on any errors that occur when using tools
- **Reasoning Model Issues**: 
  - If you see "o1 Series Models Not Supported", use o3-mini or o4-mini instead
  - Reasoning models don't support custom temperature settings
  - Some reasoning models may not support streaming (check the model-specific warnings)
- **Custom Model Names**: When using "Other" option, ensure the model name is exactly as expected by the provider's API, and you have access to it.

## Resources

- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol](https://modelcontextprotocol.io/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
