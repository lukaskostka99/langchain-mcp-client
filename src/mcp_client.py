"""
MCP (Model Context Protocol) client management and tool retrieval.

This module handles the setup and management of MCP clients,
server configurations, and tool retrieval.
"""

from typing import Dict, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool


async def setup_mcp_client(server_config: Dict[str, Dict]) -> MultiServerMCPClient:
    """Initialize a MultiServerMCPClient with the provided server configuration."""
    client = MultiServerMCPClient(server_config)
    return client


async def get_tools_from_client(client: MultiServerMCPClient) -> List[BaseTool]:
    """Get tools from the MCP client."""
    return await client.get_tools()


def create_single_server_config(server_url: str, timeout: int = 600, sse_read_timeout: int = 900) -> Dict[str, Dict]:
    """
    Create a configuration for a single MCP server.
    
    Args:
        server_url: The URL of the MCP server
        timeout: Connection timeout in seconds
        sse_read_timeout: SSE read timeout in seconds
    
    Returns:
        Server configuration dictionary
    """
    # Ensure URL points to SSE endpoint
    if not server_url.endswith("/sse"):
        server_url = server_url.rstrip("/") + "/sse"
    return {
        "default_server": {
            "transport": "sse",
            "url": server_url,
            "headers": None,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout
        }
    }


def create_multi_server_config(servers: Dict[str, str], timeout: int = 600, sse_read_timeout: int = 900) -> Dict[str, Dict]:
    """
    Create a configuration for multiple MCP servers.
    
    Args:
        servers: Dictionary mapping server names to URLs
        timeout: Connection timeout in seconds
        sse_read_timeout: SSE read timeout in seconds
    
    Returns:
        Server configuration dictionary
    """
    config = {}
    for name, url in servers.items():
        # Ensure URL points to SSE endpoint
        if not url.endswith("/sse"):
            url = url.rstrip("/") + "/sse"
        config[name] = {
            "transport": "sse",
            "url": url,
            "headers": None,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout
        }
    return config


def validate_server_url(url: str) -> bool:
    """
    Validate if a server URL is properly formatted.
    
    Args:
        url: The URL to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False
    
    # Basic URL validation
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    
    return True


def get_default_server_config() -> Dict[str, str]:
    """Get default server configuration values."""
    return {
        "default_url": "http://localhost:8000/sse",
        "default_timeout": "600",
        "default_sse_timeout": "900"
    } 