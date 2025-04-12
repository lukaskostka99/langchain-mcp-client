# LangChain MCP Client Streamlit App

This Streamlit application provides a user interface for connecting to LangChain MCP (Model Context Protocol) servers and interacting with them using different LLM providers (OpenAI or Anthropic).

## Features

- Connect to MCP servers via SSE (Server-Sent Events)
- Support for both single server and multiple server configurations 
- Select between different LLM providers (OpenAI/Claude)
- View, test, and use available MCP tools directly from the UI
- Chat interface for interacting with the LLM agent
- Tool execution results display

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

## Future Improvements

- **STDIO MCP Servers**: Support for connecting to MCP servers using standard input/output (STDIO) for more flexible server configurations.
- **Test Tools Individually**: Implement functionality to test each tool individually from the UI to ensure they work as expected.
- **Agent Memory**: Introduce memory capabilities for the agent to retain context across interactions.
- **RAG (File Upload)**: Enable Retrieval-Augmented Generation (RAG) by allowing users to upload files that the agent can use to enhance its responses.

## Troubleshooting

- **Connection Issues**: Ensure your MCP server is running and accessible
- **API Key Errors**: Verify that you've entered the correct API key for your chosen LLM provider
- **Tool Errors**: Check the server logs for details on any errors that occur when using tools

## Resources

- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol](https://modelcontextprotocol.io/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)