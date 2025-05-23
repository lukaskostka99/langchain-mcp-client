# LangChain MCP Client Streamlit App

This Streamlit application provides a user interface for connecting to MCP (Model Context Protocol) servers and interacting with them using different LLM providers (OpenAI, Anthropic, Google...).

## Features

- Connect to MCP servers via SSE (Server-Sent Events)
- Support for both single server and multiple server configurations 
- Select between different LLM providers (OpenAI/Claude)
- View, test, and use available MCP tools directly from the UI
- **Individual Tool Testing**: Test each tool individually with custom parameters and view detailed results
- **Agent Memory**: Persistent conversation memory with thread management and export/import capabilities
- Chat interface for interacting with the LLM agent
- Tool execution results display

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

### Persistent Storage Features

- **Automatic Conversation Saving**: Conversations are automatically saved to SQLite during chat
- **Conversation Browser**: View and manage all stored conversations with metadata
- **Thread Management**: Switch between different conversation threads
- **Export/Import**: Export conversations as JSON files or import previous conversations
- **Database Statistics**: Monitor database size, conversation count, and message totals
- **Conversation Search**: Browse conversations by title, date, or message count

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

## Configuration Options

### LLM Providers
- **OpenAI**: Requires an OpenAI API key and supports models like gpt-4o, gpt-4, and gpt-3.5-turbo
- **Anthropic**: Requires an Anthropic API key and supports Claude models
- **Google**: Requires a Google Generative Language / Vertex AI API
- **Local LLMs**: Supports a local LLM using Ollama

### MCP Server Connection
- Currently supports SSE (Server-Sent Events) connections
- Enter the URL of your MCP server's SSE endpoint (e.g., `http://localhost:8000/sse`)

### Server Modes
- **Single Server**: Connect to a single MCP server
- **Multiple Servers**: Connect to multiple MCP servers simultaneously
  - Add servers with unique names
  - Manage (add/remove) servers through the UI
  - Connect to all configured servers at once

### Available Tools
- View all available tools from connected MCP servers in the sidebar
- Each tool displays:
  - Name and description
  - Required and optional parameters with their types
  - Parameter descriptions and constraints
- Tools are automatically available to the LLM agent in the chat interface
- Tool executions and their results are tracked in the chat history

### Tool Testing
- **Individual Tool Testing**: Access via the "🔧 Test Tools" tab
- Test each tool individually with custom parameters
- Dynamic form generation based on tool schema
- Real-time parameter validation
- Execution timing and success/failure tracking
- Test result history and statistics
- Export test results to JSON format
- Performance metrics (success rate, average execution time)

### Agent Memory
- **Conversation Persistence**: Enable/disable memory for chat history
- **Thread Management**: Separate conversations with unique thread IDs
- **Memory Controls**: Clear memory, reset threads, set message limits
- **Export/Import**: Save and restore conversation history
- **Memory Analytics**: Track message counts, memory usage, and statistics
- **Real-time Status**: Visual indicators for memory state and thread information
- **Flexible Configuration**: Choose memory types and limits based on needs
- **History Tool**: When memory is enabled, the agent gains access to a `get_conversation_history` tool that allows it to:
  - Search through previous conversations
  - Reference earlier discussions
  - Summarize conversation topics
  - Filter messages by type (user/assistant/tool)
  - Access specific parts of the conversation history
  - Clean, formatted output that's easy to read in the chat interface

## Future Improvements

- **STDIO MCP Servers**: Support for connecting to MCP servers using standard input/output (STDIO) for more flexible server configurations.
- ✅ **Test Tools Individually**: Implement functionality to test each tool individually from the UI to ensure they work as expected.
- ✅ **Using Local LLMs**: Support for connecting local LLMs (Llama, DeepSeek, Qwen...)
- ✅ **Agent Memory**: Introduce memory capabilities for the agent to retain context across interactions.
- **RAG (File Upload)**: Enable Retrieval-Augmented Generation (RAG) by allowing users to upload files that the agent can use to enhance its responses.

## Troubleshooting

- **Connection Issues**: Ensure your MCP server is running and accessible
- **API Key Errors**: Verify that you've entered the correct API key for your chosen LLM provider
- **Tool Errors**: Check the server logs for details on any errors that occur when using tools

## Resources

- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol](https://modelcontextprotocol.io/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
